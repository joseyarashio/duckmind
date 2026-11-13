/**
 * The front on the EC2 host: the shared front door (core/front.ts) over one in-memory gate
 * (core/gate.ts) and one SessionCore (core/session.ts) per visitor, each with its own Docker duck
 * (host/docker.ts). What the Durable Objects give the Worker, this gives the host: one gate, a timer per
 * session in place of its alarm, and one call at a time per session (start, status, leave, alarm).
 *
 * State lives in memory only. A restarted front forgets its line and removes every duck it had
 * started (removeOwned), so a slot is never held by a container nobody owns.
 */

import type { EdgeConfig } from "../core/config.ts";
import { type FrontDeps, type FrontGate, type FrontSession, handleFront, viewResponse } from "../core/front.ts";
import { emptyGate, GateCore, type GateLimits } from "../core/gate.ts";
import { type GatePort, SessionCore, type SessionView } from "../core/session.ts";
import { SITEVERIFY_URL, WIDGET_SCRIPT_URL } from "../core/turnstile.ts";
import { type HostSettings, hostBudget } from "./config.ts";
import { DockerBackend, type DockerPort, type PortFetch, removeOwned, SlotPool } from "./docker.ts";

export interface Timer {
	cancel(): void;
}

/** The clock and timers (tests drive their own). */
export interface Scheduler {
	now(): number;
	at(when: number, fn: () => void): Timer;
}

export const realScheduler: Scheduler = {
	now: () => Date.now(),
	at(when, fn) {
		const t = setTimeout(fn, Math.max(0, when - Date.now()));
		t.unref();
		return { cancel: () => clearTimeout(t) };
	},
};

export interface HostSecrets {
	turnstileSecret: string;
	addressKey: string;
}

export interface HostDeps {
	edge: EdgeConfig;
	host: HostSettings;
	secrets: HostSecrets;
	docker: DockerPort;
	fetchPort: PortFetch;
	scheduler: Scheduler;
	/** GET a file of the front's own (FrontDeps.asset). */
	asset(path: string): Promise<Response>;
	log(event: string, fields?: Record<string, unknown>): void;
	randomSeed?: () => number;
	/** Siteverify's fetch (tests answer it themselves). */
	fetchFn?: typeof fetch;
}

/** How a request reached the front. */
export interface Arrival {
	/** True only on the tunnel's socket, which only cloudflared can open. */
	tunnel: boolean;
	/** The socket's peer address (127.0.0.1 or nothing for a unix socket). */
	peer: string | null;
	headers: Headers;
}

const IP = /^(?:\d{1,3}(?:\.\d{1,3}){3}|[0-9A-Fa-f:.]{2,45})$/;

/** The visitor's address: Cloudflare's CF-Connecting-IP when the request came through the tunnel;
 * anywhere else the header is ignored (anyone could send it) and the socket's peer counts. */
export function clientAddress(a: Arrival): string | null {
	if (!a.tunnel) return a.peer;
	const h = a.headers.get("cf-connecting-ip")?.trim() ?? "";
	return IP.test(h) ? h : null;
}

function gateLimits(c: EdgeConfig): GateLimits {
	return {
		maxLive: c.maxLive,
		maxQueue: c.maxQueue,
		maxPerAddress: c.maxPerAddress,
		startRateMax: c.startRateMax,
		startWindowS: c.startWindowS,
		queueStaleS: c.queueStaleS,
		admitGraceS: c.admitGraceS,
	};
}

/** One visitor: a SessionCore, its duck, its timer, one call at a time. */
class HostSession implements FrontSession {
	readonly core: SessionCore;
	private chain: Promise<unknown> = Promise.resolve();
	private timer: Timer | null = null;
	private readonly app: HostFront;
	private readonly id: string;

	constructor(app: HostFront, id: string) {
		this.app = app;
		this.id = id;
		const backend = new DockerBackend({
			docker: app.deps.docker,
			settings: () => app.deps.host,
			slots: app.slots,
			fetchPort: app.deps.fetchPort,
			sessionId: id,
			onStop: (clean) => void this.serial(() => this.core.containerStopped(clean)),
			log: app.deps.log,
		});
		this.core = new SessionCore(undefined, {
			backend,
			gate: app.gatePort,
			store: {
				put: async () => {},
				setAlarm: async (at) => this.schedule(at),
			},
			config: app.deps.edge,
			now: () => app.deps.scheduler.now(),
			randomSeed: app.deps.randomSeed ?? (() => (crypto.getRandomValues(new Uint32Array(1))[0]! % 2_147_483_646) + 1),
			log: app.deps.log,
		});
	}

	/** One call at a time, as a Durable Object's input gate gives the Worker's session. */
	serial<T>(fn: () => Promise<T>): Promise<T> {
		const next = this.chain.then(fn);
		this.chain = next.catch(() => undefined);
		return next;
	}

	private schedule(at: number): void {
		this.timer?.cancel();
		this.timer = this.app.deps.scheduler.at(at, () => {
			this.timer = null;
			void this.serial(async () => {
				if ((await this.core.alarm()) === "purge") this.app.forget(this.id);
			}).catch((err: unknown) => this.app.deps.log("alarm_error", { error: String(err) }));
		});
	}

	cancel(): void {
		this.timer?.cancel();
		this.timer = null;
	}

	create(id: string, addr: string, admitted: boolean): Promise<SessionView> {
		return this.serial(async () => {
			await this.core.create(id, addr, admitted);
			return this.core.view();
		});
	}

	status(): Promise<SessionView> {
		return this.serial(() => this.core.status());
	}

	leave(): Promise<void> {
		return this.serial(() => this.core.leave());
	}

	/** Not serialised: page requests and long streams run side by side, as on the Worker. */
	async fetch(request: Request): Promise<Response> {
		return (await this.core.proxy(request)) ?? viewResponse(this.core.view());
	}
}

const GONE: FrontSession = {
	create: async () => ({ phase: "gone" }),
	status: async () => ({ phase: "gone" }),
	leave: async () => {},
	fetch: async () => viewResponse({ phase: "gone" }),
};

export class HostFront {
	readonly deps: HostDeps;
	readonly gate: GateCore;
	readonly slots: SlotPool;
	readonly gatePort: GatePort;
	private readonly sessions = new Map<string, HostSession>();
	private gateTimer: Timer | null = null;
	private readonly front: FrontDeps;

	constructor(deps: HostDeps) {
		this.deps = deps;
		this.gate = new GateCore(emptyGate(), gateLimits(deps.edge), hostBudget(deps.edge, deps.host));
		this.slots = new SlotPool(
			() => this.deps.host.portBase,
			() => this.deps.edge.maxLive,
		);
		const now = () => deps.scheduler.now();
		this.gatePort = {
			poll: async (id) => this.afterGate(this.gate.poll(id, now())),
			begin: async (id, until) => this.afterGate(this.gate.begin(id, until, now())),
			extend: async (id, until) => this.afterGate(this.gate.extend(id, until)),
			release: async (id, reason) => {
				const usd = this.afterGate(this.gate.release(id, now()));
				deps.log("gate_release", { session: id.slice(0, 8), reason, ...this.gate.stats() });
				return usd;
			},
		};
		const gate: FrontGate = {
			join: async (id, addr) => {
				const r = this.afterGate(this.gate.join(id, addr, now()));
				deps.log("gate_join", { session: id.slice(0, 8), ...r, ...this.gate.stats() });
				if (r.ok) this.sessions.set(id, new HostSession(this, id));
				return r;
			},
			status: async () => ({ ...this.gate.stats(), availability: this.gate.availability(now()) }),
		};
		this.front = {
			config: deps.edge,
			gate,
			session: (id) => this.sessions.get(id) ?? GONE,
			asset: deps.asset,
			addressKey: deps.secrets.addressKey,
			turnstileSecret: deps.secrets.turnstileSecret,
			turnstileScriptUrl: WIDGET_SCRIPT_URL,
			turnstileVerifyUrl: SITEVERIFY_URL,
			fetchFn: deps.fetchFn,
			log: deps.log,
		};
	}

	/** Remove every duck a previous run left behind. */
	async boot(): Promise<void> {
		const n = await removeOwned(this.deps.docker);
		this.deps.log("boot", { removed: n, open: this.deps.edge.open, max_live: this.deps.edge.maxLive, image: this.deps.host.image });
	}

	handle(request: Request, arrival: Arrival): Promise<Response> {
		return handleFront(request, clientAddress(arrival), this.front);
	}

	/** New settings (systemctl reload): the kill switch, limits, the image for the next duck. Running
	 * ducks keep their container; with the switch closed they are stopped at their next check. */
	reload(edge: EdgeConfig, host: HostSettings): void {
		Object.assign(this.deps.edge, edge);
		Object.assign(this.deps.host, host);
		Object.assign(this.gate.limits, gateLimits(edge));
		Object.assign(this.gate.budget, hostBudget(edge, host));
		this.afterGate(undefined);
		this.deps.log("reload", { open: edge.open, max_live: edge.maxLive, image: host.image });
	}

	/** Stop every duck (the front is shutting down). */
	async stop(): Promise<void> {
		this.gateTimer?.cancel();
		await Promise.all([...this.sessions.values()].map((s) => s.serial(() => s.core.finish("closed")).catch(() => undefined)));
		for (const s of this.sessions.values()) s.cancel();
		this.sessions.clear();
		await removeOwned(this.deps.docker);
	}

	forget(id: string): void {
		this.sessions.get(id)?.cancel();
		this.sessions.delete(id);
	}

	/** The gate's own timer: places nobody asks for and leases that ran out. */
	private afterGate<T>(result: T): T {
		this.gateTimer?.cancel();
		const wake = this.gate.nextWake();
		this.gateTimer =
			wake === null
				? null
				: this.deps.scheduler.at(Math.max(wake, this.deps.scheduler.now() + 1000), () => {
						this.gateTimer = null;
						for (const d of this.gate.sweep(this.deps.scheduler.now())) {
							this.deps.log("gate_drop", { session: d.id.slice(0, 8), why: d.why });
						}
						this.afterGate(undefined);
					});
		return result;
	}

	stats(): { live: number; queued: number; sessions: number; slots: number; month_gb: number; cap_gb: number | null } {
		const e = this.gate.egressMonth(this.deps.scheduler.now());
		return {
			...this.gate.stats(),
			sessions: this.sessions.size,
			slots: this.slots.inUse(),
			month_gb: Math.round(e.sentBytes / 1e6) / 1000,
			cap_gb: e.capBytes === null ? null : e.capBytes / 1e9,
		};
	}

	/** Sent bytes were booked: the line may have to stop or move (the gate's timer looks again). */
	egressBooked(): void {
		this.afterGate(undefined);
	}
}
