/**
 * One visitor's session: waits in line, starts its own container with a random seed, waits for
 * /api/ping, proxies the visitor's requests, and stops the container at ends_at + exportMinutes.
 *
 * Pure logic over injected ports (container, gate, storage, clock) so tests run it with fakes; the
 * Durable Object in src/session-do.ts wires the real ones. Every way out (time up, crash, failed start,
 * idle visitor, visitor left, lost place) goes through finish(), which stops the container and gives
 * the gate slot back.
 */

import type { EdgeConfig } from "./config.ts";
import type { BeginResult, PollResult } from "./gate.ts";
import { prepareForward, visitorResponse } from "./proxy.ts";

export const CONTAINER_ORIGIN = "http://container";

export interface Backend {
	/** Boot the container with these environment variables (returns before it is ready). */
	start(env: Record<string, string>): Promise<void>;
	running(): boolean;
	/** Send a request to the container's port 8080 (URL origin CONTAINER_ORIGIN). */
	fetch(request: Request): Promise<Response>;
	destroy(reason: string): Promise<void>;
}

export interface GatePort {
	poll(id: string): Promise<PollResult>;
	/** Metering starts: called before the container is started; anything but "ok" means do not start. */
	begin(id: string, until: number): Promise<BeginResult>;
	extend(id: string, until: number): Promise<boolean>;
	release(id: string, reason: string): Promise<unknown>;
}

export interface SessionStore {
	put(record: SessionRecord): Promise<void>;
	setAlarm(at: number): Promise<void>;
}

export type Phase = "queued" | "starting" | "running" | "ended" | "gone";

export type EndReason =
	| "time"
	| "exited"
	| "crash"
	| "start_failed"
	| "start_timeout"
	| "idle"
	| "left"
	| "lost_place"
	| "queue_idle"
	/** The day's or month's budget ran out while waiting in line. */
	| "full"
	/** The owner closed the edge (kill switch). */
	| "closed";

export interface SessionRecord {
	id: string;
	addr: string;
	phase: Phase;
	createdAt: number;
	lastSeen: number;
	seed?: number;
	startedAt?: number;
	startDeadline?: number;
	readyAt?: number;
	/** When play ends (ms); writes answer 410 after this, the download still works. */
	endsAt?: number;
	/** When the container is stopped (ms): endsAt + exportMinutes. */
	stopAt?: number;
	endReason?: EndReason;
	endedAt?: number;
}

export interface SessionView {
	phase: Phase;
	place?: number;
	seed?: number;
	startedAt?: number;
	endsAt?: number;
	stopAt?: number;
	leftS?: number;
	reason?: EndReason;
}

export interface SessionDeps {
	backend: Backend;
	gate: GatePort;
	store: SessionStore;
	config: EdgeConfig;
	now: () => number;
	/** A fresh duck seed: positive 31-bit integer. */
	randomSeed: () => number;
	log?: (event: string, fields: Record<string, unknown>) => void;
}

/** A record lingers this long after the end so a returning visitor sees "time is up", then is purged. */
export const KEEP_GONE_MS = 6 * 3600 * 1000;

export class SessionCore {
	record: SessionRecord | undefined;
	private readonly d: SessionDeps;

	constructor(record: SessionRecord | undefined, deps: SessionDeps) {
		this.record = record;
		this.d = deps;
	}

	private get r(): SessionRecord {
		if (!this.record) throw new Error("session not created");
		return this.record;
	}

	private log(event: string, fields: Record<string, unknown> = {}): void {
		this.d.log?.(event, { session: this.record?.id.slice(0, 8), ...fields });
	}

	private async save(): Promise<void> {
		await this.d.store.put(this.r);
	}

	/** New session after the gate let it in (`admitted`) or put it in line. */
	async create(id: string, addr: string, admitted: boolean): Promise<void> {
		const now = this.d.now();
		this.record = { id, addr, phase: "queued", createdAt: now, lastSeen: now };
		if (admitted) {
			await this.begin();
			return;
		}
		await this.save();
		await this.d.store.setAlarm(now + this.d.config.queueStaleS * 1000);
	}

	/** The visitor's page asks how things stand (also how a queued visitor keeps the place). */
	async status(): Promise<SessionView> {
		if (!this.record) return { phase: "gone" };
		const now = this.d.now();
		if (this.r.phase !== "gone") this.r.lastSeen = now;
		let place: number | undefined;
		if (this.r.phase === "queued") {
			const p = await this.d.gate.poll(this.r.id);
			if (p.state === "live") await this.begin();
			else if (p.state === "full") await this.finish("full");
			else if (p.state === "unknown") await this.finish("lost_place");
			else place = p.place;
		}
		if (this.r.phase !== "gone" && this.r.phase !== "starting") await this.save();
		return this.view(place);
	}

	view(place?: number): SessionView {
		if (!this.record) return { phase: "gone" };
		const r = this.r;
		const v: SessionView = { phase: r.phase };
		if (place !== undefined) v.place = place;
		if (r.seed !== undefined && r.phase !== "queued") v.seed = r.seed;
		if (r.startedAt !== undefined) v.startedAt = r.startedAt;
		if (r.endsAt !== undefined) {
			v.endsAt = r.endsAt;
			v.leftS = Math.max(0, (r.endsAt - this.d.now()) / 1000);
		}
		if (r.stopAt !== undefined) v.stopAt = r.stopAt;
		if (r.endReason) v.reason = r.endReason;
		return v;
	}

	private async begin(): Promise<void> {
		const now = this.d.now();
		const r = this.r;
		const startDeadline = now + this.d.config.startTimeoutS * 1000;
		// The gate meters the container from here on; without its yes no container is started.
		const allowed = await this.d.gate.begin(r.id, startDeadline + 60_000);
		if (allowed !== "ok") {
			await this.finish(allowed === "closed" ? "closed" : "lost_place");
			return;
		}
		r.seed = this.d.randomSeed();
		r.phase = "starting";
		r.startedAt = now;
		r.startDeadline = startDeadline;
		await this.save();
		try {
			await this.d.backend.start({
				DUCKMIND_PUBLIC_SEED: String(r.seed),
				DUCKMIND_PUBLIC_MINUTES: String(this.d.config.sessionMinutes),
			});
		} catch (err) {
			this.log("start_error", { error: String(err) });
			await this.finish("start_failed");
			return;
		}
		this.log("starting", { seed: r.seed });
		await this.d.store.setAlarm(now + this.d.config.pingIntervalS * 1000);
	}

	/** Scheduled work: readiness pings, health checks, the hard stop, purging. Returns "purge" when
	 * the record should be deleted. */
	async alarm(): Promise<"purge" | undefined> {
		if (!this.record) return "purge";
		const now = this.d.now();
		const r = this.r;
		const cfg = this.d.config;
		switch (r.phase) {
			case "queued":
				if (now - r.lastSeen > cfg.queueStaleS * 1000) await this.finish("queue_idle");
				else await this.d.store.setAlarm(now + cfg.queueStaleS * 1000);
				return;
			case "starting":
				if (!cfg.open) return void (await this.finish("closed"));
				if (!this.d.backend.running()) return void (await this.finish("crash"));
				if (now >= (r.startDeadline ?? 0)) return void (await this.finish("start_timeout"));
				if (await this.ping()) return void (await this.ready());
				await this.d.store.setAlarm(now + cfg.pingIntervalS * 1000);
				return;
			case "running":
			case "ended":
				await this.check();
				return;
			case "gone":
				if (now >= (r.endedAt ?? 0) + KEEP_GONE_MS) return "purge";
				await this.d.store.setAlarm((r.endedAt ?? now) + KEEP_GONE_MS);
				return;
		}
	}

	private async ping(): Promise<boolean> {
		try {
			const res = await this.d.backend.fetch(new Request(`${CONTAINER_ORIGIN}/api/ping`));
			await res.body?.cancel();
			return res.status === 200;
		} catch {
			return false;
		}
	}

	private async ready(): Promise<void> {
		const now = this.d.now();
		const r = this.r;
		const cfg = this.d.config;
		// The container's own clock decides the end; never later than our own budget allows.
		const budgetEnd = now + cfg.sessionMinutes * 60_000;
		let endsAt = budgetEnd;
		try {
			const res = await this.d.backend.fetch(new Request(`${CONTAINER_ORIGIN}/api/public/session`));
			if (res.ok) {
				const body = (await res.json()) as { ends_at?: unknown };
				const t = typeof body.ends_at === "string" ? Date.parse(body.ends_at) : NaN;
				if (Number.isFinite(t)) endsAt = Math.min(t, budgetEnd + 60_000);
			} else {
				await res.body?.cancel();
			}
		} catch {
			// keep our own budget
		}
		r.phase = "running";
		r.readyAt = now;
		r.endsAt = endsAt;
		r.stopAt = endsAt + cfg.exportMinutes * 60_000;
		await this.save();
		await this.d.gate.extend(r.id, r.stopAt + 60_000);
		this.log("ready", { boot_s: (now - (r.startedAt ?? now)) / 1000 });
		await this.d.store.setAlarm(Math.min(r.stopAt, now + cfg.checkIntervalS * 1000));
	}

	private async check(): Promise<void> {
		const now = this.d.now();
		const r = this.r;
		if (now >= (r.stopAt ?? 0)) return this.finish("time");
		if (!this.d.config.open) return this.finish("closed");
		if (!this.d.backend.running()) return this.finish(r.phase === "ended" ? "exited" : "crash");
		if (now - r.lastSeen > this.d.config.idleEndS * 1000) return this.finish("idle");
		if (r.phase === "running" && now >= (r.endsAt ?? 0)) {
			r.phase = "ended";
			await this.save();
		}
		await this.d.store.setAlarm(Math.min(r.stopAt ?? now, now + this.d.config.checkIntervalS * 1000));
	}

	/** The container stopped by itself (monitor() settled). */
	async containerStopped(exitedCleanly: boolean): Promise<void> {
		if (!this.record) return;
		const p = this.r.phase;
		if (p === "starting" || p === "running" || p === "ended") {
			await this.finish(exitedCleanly && p === "ended" ? "exited" : "crash");
		}
	}

	/** The visitor leaves the line or ends the duck early. */
	async leave(): Promise<void> {
		if (this.record && this.r.phase !== "gone") await this.finish("left");
	}

	async finish(reason: EndReason): Promise<void> {
		const r = this.r;
		if (r.phase === "gone") return;
		const now = this.d.now();
		const hadContainer = r.phase !== "queued";
		r.phase = "gone";
		r.endReason = reason;
		r.endedAt = now;
		await this.save();
		if (hadContainer) {
			try {
				await this.d.backend.destroy(reason);
			} catch (err) {
				this.log("destroy_error", { error: String(err) });
			}
		}
		await this.d.gate.release(r.id, reason);
		this.log("end", { reason, played_s: r.readyAt ? (now - r.readyAt) / 1000 : 0 });
		await this.d.store.setAlarm(now + KEEP_GONE_MS);
	}

	/** Pass a visitor request to the container, or null when there is nothing to pass it to yet. */
	async proxy(request: Request): Promise<Response | null> {
		if (!this.record) return null;
		const r = this.r;
		if (r.phase !== "running" && r.phase !== "ended") return null;
		const now = this.d.now();
		// Seen now; written down at most every 30 s (well inside the 2-minute idle stop) so a restarted
		// object does not think the visitor has gone idle.
		const unsaved = now - r.lastSeen > 30_000;
		r.lastSeen = now;
		if (unsaved) await this.save();
		const fwd = await prepareForward(request, CONTAINER_ORIGIN, this.d.config.maxBodyBytes);
		if (!fwd.ok) {
			return new Response(fwd.status === 413 ? "too large" : "not found", {
				status: fwd.status,
				headers: { "content-type": "text/plain; charset=utf-8" },
			});
		}
		try {
			return visitorResponse(await this.d.backend.fetch(fwd.request));
		} catch (err) {
			this.log("proxy_error", { error: String(err) });
			return new Response("the duck is not answering", {
				status: 502,
				headers: { "content-type": "text/plain; charset=utf-8" },
			});
		}
	}
}
