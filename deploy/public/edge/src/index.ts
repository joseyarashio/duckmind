/**
 * Public play edge on Cloudflare (CONTRACT.md, "The edge"): one Worker, a Gate Durable Object and one
 * DuckSession Durable Object per visitor, which owns that visitor's Container. The routes and pages are
 * the shared front door (src/core/front.ts); the EC2 host (src/host/) serves the same front.
 */

import { DurableObject } from "cloudflare:workers";
import { isLoopbackUrl, readConfig, sessionWorstSeconds, worstUsdPerSecond, type EdgeConfig } from "./core/config.ts";
import { handleFront, viewResponse } from "./core/front.ts";
import {
	type Availability,
	type BeginResult,
	emptyGate,
	GateCore,
	type GateState,
	type GateStats,
	type JoinResult,
	type PollResult,
} from "./core/gate.ts";
import {
	type Backend,
	SessionCore,
	type SessionRecord,
	type SessionView,
} from "./core/session.ts";
import { SITEVERIFY_URL, WIDGET_SCRIPT_URL } from "./core/turnstile.ts";

const GATE_NAME = "gate";

function log(event: string, fields: Record<string, unknown> = {}): void {
	console.log(JSON.stringify({ event, ...fields }));
}

// The generated binding types do not know which class a self-referencing namespace holds.
function gate(env: Env): DurableObjectStub<Gate> {
	return (env.GATE as DurableObjectNamespace<Gate>).getByName(GATE_NAME);
}

function session(env: Env, id: string): DurableObjectStub<DuckSession> {
	return (env.SESSIONS as DurableObjectNamespace<DuckSession>).getByName(id);
}

function vars(env: Env): Record<string, unknown> {
	return env as object as Record<string, unknown>;
}

// ---------------------------------------------------------------- Gate (one, named "gate")

export class Gate extends DurableObject<Env> {
	private core!: GateCore;

	constructor(ctx: DurableObjectState, env: Env) {
		super(ctx, env);
		const cfg = readConfig(env);
		void ctx.blockConcurrencyWhile(async () => {
			const state = (await ctx.storage.get<GateState>("gate")) ?? emptyGate();
			this.core = new GateCore(
				state,
				{
					maxLive: cfg.maxLive,
					maxQueue: cfg.maxQueue,
					maxPerAddress: cfg.maxPerAddress,
					startRateMax: cfg.startRateMax,
					startWindowS: cfg.startWindowS,
					queueStaleS: cfg.queueStaleS,
					admitGraceS: cfg.admitGraceS,
				},
				{
					open: cfg.open,
					monthUsd: cfg.budgetMonthUsd,
					billingDay: cfg.billingDay,
					usdPerSecond: worstUsdPerSecond(cfg.instance),
					sessionWorstS: sessionWorstSeconds(cfg),
				},
			);
		});
	}

	private async commit(): Promise<void> {
		await this.ctx.storage.put("gate", this.core.state);
		const wake = this.core.nextWake();
		if (wake === null) await this.ctx.storage.deleteAlarm();
		else await this.ctx.storage.setAlarm(Math.max(wake, Date.now() + 1000));
	}

	async join(id: string, addr: string): Promise<JoinResult> {
		const r = this.core.join(id, addr, Date.now());
		await this.commit();
		log("gate_join", { session: id.slice(0, 8), ...r, ...this.core.stats() });
		return r;
	}

	async poll(id: string): Promise<PollResult> {
		const r = this.core.poll(id, Date.now());
		await this.commit();
		return r;
	}

	async begin(id: string, until: number): Promise<BeginResult> {
		const r = this.core.begin(id, until, Date.now());
		await this.commit();
		log("gate_begin", { session: id.slice(0, 8), result: r });
		return r;
	}

	async extend(id: string, until: number): Promise<boolean> {
		const ok = this.core.extend(id, until);
		await this.commit();
		return ok;
	}

	async release(id: string, reason: string): Promise<boolean> {
		const now = Date.now();
		const usd = this.core.release(id, now);
		await this.commit();
		log("gate_release", { session: id.slice(0, 8), reason, usd, ...this.core.stats(), ...this.core.spending(now) });
		return usd !== null;
	}

	/** For the edge's pages: slots, line and whether a newcomer can come in (no money figures). */
	async status(): Promise<GateStats & { availability: Availability }> {
		const availability = this.core.availability(Date.now());
		return { ...this.core.stats(), availability };
	}

	async alarm(): Promise<void> {
		const now = Date.now();
		const dropped = this.core.sweep(now);
		await this.commit();
		for (const d of dropped) log("gate_drop", { session: d.id.slice(0, 8), why: d.why, usd: d.usd });
		if (dropped.some((d) => d.usd !== undefined)) log("gate_spending", this.core.spending(now));
	}
}

// ---------------------------------------------------------------- one visitor's session

/** The visitor's Cloudflare Container (Durable Object scheduling policy). */
class ContainerBackend implements Backend {
	constructor(
		private readonly ctx: DurableObjectState,
		private readonly container: Container,
		private readonly cfg: EdgeConfig,
		private readonly onStop: (clean: boolean) => Promise<void>,
	) {}

	/** Keep the container while its Durable Object idles between polls, until the hard stop. */
	private get keepAliveMs(): number {
		return (this.cfg.sessionMinutes + this.cfg.exportMinutes + 5) * 60_000;
	}

	watch(): void {
		this.ctx.waitUntil(
			this.container.monitor().then(
				() => this.onStop(true),
				(err: unknown) => {
					log("container_error", { error: String(err) });
					return this.onStop(false);
				},
			),
		);
	}

	async start(env: Record<string, string>): Promise<void> {
		this.container.start({
			image: this.container.images.duck,
			instance: this.cfg.instance,
			enableInternet: false,
			env,
		});
		await this.container.setInactivityTimeout(this.keepAliveMs);
		this.watch();
	}

	running(): boolean {
		return this.container.running;
	}

	fetch(request: Request): Promise<Response> {
		return this.container.getTcpPort(8080).fetch(request);
	}

	async destroy(reason: string): Promise<void> {
		if (this.container.running) await this.container.destroy(reason);
	}
}

/** `cf dev` only: the local stub container (dev/stub-container.mjs) at a loopback address. */
class StubBackend implements Backend {
	private stopped = false;
	constructor(private readonly base: string) {}

	async start(env: Record<string, string>): Promise<void> {
		this.stopped = false;
		const r = await fetch(`${this.base}/__stub/start`, { method: "POST", body: JSON.stringify(env) });
		if (!r.ok) throw new Error(`stub start ${r.status}`);
	}

	running(): boolean {
		return !this.stopped;
	}

	fetch(request: Request): Promise<Response> {
		const u = new URL(request.url);
		return fetch(new Request(this.base + u.pathname + u.search, request));
	}

	async destroy(): Promise<void> {
		this.stopped = true;
		await fetch(`${this.base}/__stub/stop`, { method: "POST" }).catch(() => undefined);
	}
}

export class DuckSession extends DurableObject<Env> {
	private core!: SessionCore;
	private readonly cfg: EdgeConfig;

	constructor(ctx: DurableObjectState, env: Env) {
		super(ctx, env);
		this.cfg = readConfig(env);
		const backend = this.makeBackend();
		void ctx.blockConcurrencyWhile(async () => {
			const record = await ctx.storage.get<SessionRecord>("session");
			this.core = new SessionCore(record, {
				backend,
				gate: gate(env),
				store: {
					put: (r) => ctx.storage.put("session", r),
					setAlarm: (at) => ctx.storage.setAlarm(at),
				},
				config: this.cfg,
				now: () => Date.now(),
				randomSeed: () => crypto.getRandomValues(new Uint32Array(1))[0]! % 2_147_483_646 + 1,
				log,
			});
			if (backend instanceof ContainerBackend && ctx.container?.running) {
				await ctx.container.setInactivityTimeout((this.cfg.sessionMinutes + this.cfg.exportMinutes + 5) * 60_000);
				backend.watch();
			}
		});
	}

	private makeBackend(): Backend {
		const container = this.ctx.container;
		if (container) {
			return new ContainerBackend(this.ctx, container, this.cfg, (clean) => this.core.containerStopped(clean));
		}
		const dev = vars(this.env).DEV_UPSTREAM_URL;
		if (isLoopbackUrl(dev)) return new StubBackend(dev.replace(/\/$/, ""));
		throw new Error("DuckSession has no container and no local stub");
	}

	async create(id: string, addr: string, admitted: boolean): Promise<SessionView> {
		await this.core.create(id, addr, admitted);
		return this.core.view();
	}

	async status(): Promise<SessionView> {
		return this.core.status();
	}

	async leave(): Promise<void> {
		await this.core.leave();
	}

	/** A visitor request: the container's answer, or 409 with the session's view as JSON. */
	async fetch(request: Request): Promise<Response> {
		const res = await this.core.proxy(request);
		if (res) return res;
		return viewResponse(this.core.view());
	}

	async alarm(): Promise<void> {
		if ((await this.core.alarm()) === "purge") {
			await this.ctx.storage.deleteAlarm();
			await this.ctx.storage.deleteAll();
		}
	}
}

// ---------------------------------------------------------------- the Worker

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const v = vars(env);
		return handleFront(request, request.headers.get("cf-connecting-ip"), {
			config: readConfig(env),
			gate: gate(env),
			session: (id) => session(env, id),
			// The Worker's static assets: fonts/ (the simulator's fonts) and img/ (vite.config.ts).
			asset: (path) => env.ASSETS.fetch(new Request(new URL(path, "https://assets.invalid"))),
			addressKey: String(v.IP_HASH_KEY ?? ""),
			turnstileSecret: String(v.TURNSTILE_SECRET ?? ""),
			// Development only: the local stub's widget and check (loopback URLs; never in production).
			turnstileScriptUrl: isLoopbackUrl(v.TURNSTILE_SCRIPT_URL) ? v.TURNSTILE_SCRIPT_URL : WIDGET_SCRIPT_URL,
			turnstileVerifyUrl: isLoopbackUrl(v.TURNSTILE_VERIFY_URL) ? v.TURNSTILE_VERIFY_URL : SITEVERIFY_URL,
			log,
		});
	},
} satisfies ExportedHandler<Env>;
