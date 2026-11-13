import assert from "node:assert/strict";
import { test } from "node:test";
import { readConfig, sessionWorstSeconds } from "../src/core/config.ts";
import { emptyGate, GateCore } from "../src/core/gate.ts";
import { type Backend, type GatePort, KEEP_GONE_MS, SessionCore, type SessionRecord } from "../src/core/session.ts";

/** A container that boots after `bootMs`, answers the contract, and can crash. */
class FakeContainer implements Backend {
	started: Record<string, string> | null = null;
	alive = false;
	destroyed: string[] = [];
	seen: Request[] = [];
	bootAt = Infinity;
	endsAtIso: string | null = null;
	private readonly clock: { t: number };
	private readonly bootMs: number;
	constructor(clock: { t: number }, bootMs = 4000) {
		this.clock = clock;
		this.bootMs = bootMs;
	}
	async start(env: Record<string, string>): Promise<void> {
		this.started = env;
		this.alive = true;
		this.bootAt = this.clock.t + this.bootMs;
		this.endsAtIso = new Date(this.bootAt + Number(env.DUCKMIND_PUBLIC_MINUTES) * 60_000).toISOString();
	}
	running(): boolean {
		return this.alive;
	}
	async fetch(request: Request): Promise<Response> {
		this.seen.push(request);
		if (!this.alive) throw new Error("no container");
		if (this.clock.t < this.bootAt) return new Response("booting", { status: 503 });
		const path = new URL(request.url).pathname;
		if (path === "/api/ping") return new Response("ok");
		if (path === "/api/public/session") return Response.json({ ok: true, ends_at: this.endsAtIso });
		return new Response(`duck ${path}`, { headers: { "set-cookie": "x=1", "content-type": "text/plain" } });
	}
	async destroy(reason: string): Promise<void> {
		this.alive = false;
		this.destroyed.push(reason);
	}
}

/** The real gate core behind the session's gate port, as the Durable Object would hold it. */
function gatePort(core: GateCore, clock: { t: number }): GatePort & { released: string[] } {
	const released: string[] = [];
	return {
		released,
		poll: async (id) => core.poll(id, clock.t),
		begin: async (id, until) => core.begin(id, until, clock.t),
		extend: async (id, until) => core.extend(id, until),
		release: async (id, reason) => {
			released.push(`${id}:${reason}`);
			return core.release(id, clock.t);
		},
	};
}

/** US$ per container second in these tests: one session's worst case (1 + 15 + 3 + 2 min) = US$1.26. */
const RATE = 0.001;

function setup(opts: { maxLive?: number; bootMs?: number; monthUsd?: number } = {}) {
	const clock = { t: Date.parse("2026-11-01T08:00:00Z") };
	// (15-minute visits: these tests' timings and money are written for that length, not for the default)
	const config = readConfig({ MAX_LIVE: String(opts.maxLive ?? 10), START_TIMEOUT_S: "60", OPEN: "true", SESSION_MINUTES: "15" });
	const core = new GateCore(
		emptyGate(),
		{
			maxLive: config.maxLive,
			maxQueue: config.maxQueue,
			maxPerAddress: config.maxPerAddress,
			startRateMax: 100,
			startWindowS: config.startWindowS,
			queueStaleS: config.queueStaleS,
			admitGraceS: config.admitGraceS,
		},
		{
			open: true,
			monthUsd: opts.monthUsd ?? 10_000,
			billingDay: 1,
			usdPerSecond: RATE,
			sessionWorstS: sessionWorstSeconds(config),
		},
	);
	const gate = gatePort(core, clock);
	const make = (record?: SessionRecord) => {
		const backend = new FakeContainer(clock, opts.bootMs);
		const alarms: number[] = [];
		const saved: SessionRecord[] = [];
		const s = new SessionCore(record, {
			backend,
			gate,
			store: {
				put: async (r) => void saved.push(structuredClone(r)),
				setAlarm: async (at) => void alarms.push(at),
			},
			config,
			now: () => clock.t,
			randomSeed: () => 4242,
		});
		/** Run alarms in time order until `until` (what the runtime does); `polling`: the visitor's
		 * page keeps asking meanwhile. */
		const runUntil = async (until: number, polling = false) => {
			for (;;) {
				const next = alarms.length ? alarms[alarms.length - 1]! : Infinity;
				if (next > until) {
					clock.t = until;
					return;
				}
				clock.t = Math.max(clock.t, next);
				alarms.length = 0;
				if (polling && s.record && s.record.phase !== "gone") s.record.lastSeen = clock.t;
				if ((await s.alarm()) === "purge") return;
			}
		};
		return { s, backend, alarms, saved, runUntil };
	};
	return { clock, config, core, gate, make };
}

async function admitted(env: ReturnType<typeof setup>, id: string) {
	const j = env.core.join(id, `ip-${id}`, env.clock.t);
	assert.equal(j.ok && j.state, "live");
	const m = env.make();
	await m.s.create(id, `ip-${id}`, true);
	return m;
}

test("an admitted visitor's container starts with a random seed and the minutes, then is proxied once /api/ping answers", async () => {
	const env = setup();
	const { s, backend, runUntil } = await admitted(env, "a");
	assert.deepEqual(backend.started, { DUCKMIND_PUBLIC_SEED: "4242", DUCKMIND_PUBLIC_MINUTES: "15" });
	assert.equal(s.view().phase, "starting");
	assert.equal(await s.proxy(new Request("https://edge.test/")), null, "nothing to proxy while starting");
	await runUntil(env.clock.t + 5000);
	assert.equal(s.view().phase, "running");
	const res = await s.proxy(new Request("https://edge.test/api/life"));
	assert.equal(res?.status, 200);
	assert.equal(await res?.text(), "duck /api/life");
	assert.equal(res?.headers.get("set-cookie"), null, "container cookies never reach the visitor");
});

test("play ends at ends_at (download still proxied) and the container stops at ends_at + 3 min, freeing the slot", async () => {
	const env = setup();
	const { s, backend, runUntil } = await admitted(env, "a");
	await runUntil(env.clock.t + 5000);
	const endsAt = s.view().endsAt!;
	assert.equal(endsAt, Date.parse(backend.endsAtIso!), "the container's ends_at is used");
	await runUntil(endsAt + 1000, true);
	assert.equal(s.view().phase, "ended");
	const dl = await s.proxy(new Request("https://edge.test/api/life/duckfile/export"));
	assert.equal(dl?.status, 200);
	await runUntil(endsAt + 3 * 60_000 - 1, true);
	assert.equal(s.view().phase, "ended", "still downloadable just before the stop");
	await runUntil(endsAt + 3 * 60_000 + 1, true);
	assert.equal(s.view().phase, "gone");
	assert.equal(s.view().reason, "time");
	assert.deepEqual(backend.destroyed, ["time"]);
	assert.deepEqual(env.gate.released, ["a:time"]);
	assert.equal(env.core.stats().live, 0);
});

test("a container that dies while playing ends the session and the next in line gets the slot", async () => {
	const env = setup({ maxLive: 1 });
	const a = await admitted(env, "a");
	await a.runUntil(env.clock.t + 5000);
	const jb = env.core.join("b", "ip-b", env.clock.t);
	assert.deepEqual(jb, { ok: true, state: "queued", place: 1 });
	const b = env.make();
	await b.s.create("b", "ip-b", false);
	assert.deepEqual((await b.s.status()).place, 1);
	a.backend.alive = false; // crash
	await a.runUntil(env.clock.t + 31_000);
	assert.equal(a.s.view().reason, "crash");
	assert.deepEqual(env.gate.released, ["a:crash"]);
	const vb = await b.s.status();
	assert.equal(vb.phase, "starting", "b starts when its page next asks");
	assert.ok(b.backend.started);
});

test("monitor() reporting an error ends the session as a crash", async () => {
	const env = setup();
	const { s, runUntil } = await admitted(env, "a");
	await runUntil(env.clock.t + 5000);
	await s.containerStopped(false);
	assert.equal(s.view().reason, "crash");
	assert.deepEqual(env.gate.released, ["a:crash"]);
});

test("a container that never answers /api/ping is stopped at the start deadline and the slot freed", async () => {
	const env = setup({ bootMs: 10 * 60_000 });
	const { s, backend, runUntil } = await admitted(env, "a");
	await runUntil(env.clock.t + 59_000);
	assert.equal(s.view().phase, "starting");
	await runUntil(env.clock.t + 3000);
	assert.equal(s.view().reason, "start_timeout");
	assert.deepEqual(backend.destroyed, ["start_timeout"]);
	assert.deepEqual(env.gate.released, ["a:start_timeout"]);
});

test("a container start that throws ends the session at once", async () => {
	const env = setup();
	env.core.join("a", "ip-a", env.clock.t);
	const m = env.make();
	m.backend.start = async () => {
		throw new Error("no capacity");
	};
	await m.s.create("a", "ip-a", true);
	assert.equal(m.s.view().reason, "start_failed");
	assert.deepEqual(env.gate.released, ["a:start_failed"]);
});

test("a container claiming a later end than the budget is held to the budget", async () => {
	const env = setup();
	const { s, backend, runUntil } = await admitted(env, "a");
	backend.endsAtIso = new Date(env.clock.t + 5 * 3600_000).toISOString();
	await runUntil(env.clock.t + 5000);
	assert.ok(s.view().endsAt! <= env.clock.t + 16 * 60_000);
});

test("a visitor whose page sends nothing for 2 minutes while playing ends early (slot to the line)", async () => {
	const env = setup();
	const { s, runUntil } = await admitted(env, "a");
	await runUntil(env.clock.t + 5000);
	const lastSeen = env.clock.t;
	await s.proxy(new Request("https://edge.test/api/life"));
	await runUntil(lastSeen + 110_000);
	assert.equal(s.view().phase, "running", "1 min 50 s of silence is not idle yet");
	await runUntil(lastSeen + 120_000 + 31_000);
	assert.equal(s.view().reason, "idle");
	assert.deepEqual(env.gate.released, ["a:idle"]);
});

test("the defaults: 3 ducks at once, 5 minutes of play, a 3-minute download tail, a 2-minute idle stop", () => {
	const cfg = readConfig({});
	assert.equal(cfg.maxLive, 3);
	assert.equal(cfg.exportMinutes, 3);
	assert.equal(cfg.idleEndS, 120);
	assert.equal(cfg.sessionMinutes, 5);
	assert.equal(sessionWorstSeconds(cfg), 180 + 5 * 60 + 3 * 60 + 120);
});

test("a queued visitor whose page goes away loses the place", async () => {
	const env = setup({ maxLive: 1 });
	await admitted(env, "a");
	env.core.join("b", "ip-b", env.clock.t);
	const b = env.make();
	await b.s.create("b", "ip-b", false);
	await b.runUntil(env.clock.t + 2 * env.config.queueStaleS * 1000 + 1);
	assert.equal(b.s.view().phase, "gone");
	assert.equal(b.backend.started, null, "no container for someone who left the line");
});

test("leaving the line frees the place without starting anything", async () => {
	const env = setup({ maxLive: 1 });
	await admitted(env, "a");
	env.core.join("b", "ip-b", env.clock.t);
	const b = env.make();
	await b.s.create("b", "ip-b", false);
	await b.s.leave();
	assert.deepEqual(env.gate.released, ["b:left"]);
	assert.equal(env.core.stats().queued, 0);
	assert.equal(b.backend.destroyed.length, 0);
});

test("the proxy refuses paths outside the allowlist, big bodies and foreign headers before the container sees them", async () => {
	const env = setup();
	const { s, backend, runUntil } = await admitted(env, "a");
	await runUntil(env.clock.t + 5000);
	backend.seen.length = 0;
	assert.equal((await s.proxy(new Request("https://edge.test/api/layout", { method: "POST", body: "{}" })))?.status, 404);
	assert.equal((await s.proxy(new Request("https://edge.test/lab")))?.status, 404);
	const big = new Uint8Array(env.config.maxBodyBytes + 1);
	assert.equal((await s.proxy(new Request("https://edge.test/api/call", { method: "POST", body: big })))?.status, 413);
	assert.equal(backend.seen.length, 0, "nothing refused reached the container");
	await s.proxy(
		new Request("https://edge.test/api/neuro/command?x=1", {
			method: "POST",
			body: '{"text":"hi"}',
			headers: { cookie: "dm_session=secret", authorization: "x", "content-type": "application/json", "cf-connecting-ip": "203.0.113.9" },
		}),
	);
	const sent = backend.seen[0]!;
	assert.equal(new URL(sent.url).pathname + new URL(sent.url).search, "/api/neuro/command?x=1");
	assert.equal(sent.headers.get("cookie"), null);
	assert.equal(sent.headers.get("authorization"), null);
	assert.equal(sent.headers.get("cf-connecting-ip"), null);
	assert.equal(sent.headers.get("content-type"), "application/json");
	assert.equal(await sent.text(), '{"text":"hi"}');
});

test("a finished session's record is purged after a while", async () => {
	const env = setup();
	const { s } = await admitted(env, "a");
	await s.leave();
	assert.equal(s.view().phase, "gone");
	assert.equal(await s.alarm(), undefined, "kept for a returning visitor");
	env.clock.t += KEEP_GONE_MS + 1;
	assert.equal(await s.alarm(), "purge");
});

test("a playing visitor's activity is written down, so a restarted session does not end them as idle", async () => {
	const env = setup();
	const m = await admitted(env, "a");
	await m.runUntil(env.clock.t + 5000);
	env.clock.t += 4 * 60_000;
	await m.s.proxy(new Request("https://edge.test/api/life")); // written down
	env.clock.t += 50_000;
	await m.s.proxy(new Request("https://edge.test/api/life")); // the last request before the restart
	// the Durable Object restarts: a new core from the last stored record
	const again = env.make(structuredClone(m.saved.at(-1)!));
	again.backend.alive = true; // the container kept running
	env.clock.t += 80_000; // 80 s after the visitor was last seen
	await again.s.alarm();
	assert.equal(again.s.view().phase, "running");
});

// ---------------------------------------------------------------- money and the kill switch

const today = (env: ReturnType<typeof setup>) => env.core.state.meter.days["2026-11-01"] ?? 0;

test("a whole session is metered from the container's start to its stop", async () => {
	const env = setup();
	const t0 = env.clock.t;
	const { s, runUntil } = await admitted(env, "a");
	await runUntil(env.clock.t + 5000);
	await runUntil(s.view().stopAt! + 1, true);
	assert.equal(s.view().reason, "time");
	const ran = (s.record!.endedAt! - t0) / 1000;
	assert.ok(Math.abs(today(env) - ran * RATE) < 1e-9, `${today(env)} vs ${ran * RATE}`);
});

test("a session whose stop never reaches the gate is booked until its lease ends", async () => {
	const env = setup();
	const t0 = env.clock.t;
	const { s, runUntil } = await admitted(env, "a");
	await runUntil(env.clock.t + 5000);
	const stopAt = s.view().stopAt!;
	// the session object is lost here: nobody calls release
	env.clock.t = stopAt + 3600_000;
	const dropped = env.core.sweep(env.clock.t);
	assert.equal(dropped[0]?.why, "lease");
	const booked = (stopAt + 60_000 - t0) / 1000; // the lease: stop time + 60 s
	assert.ok(Math.abs(today(env) - booked * RATE) < 1e-9, `${today(env)} vs ${booked * RATE}`);
});

test("a queued visitor is told the day is full when the money runs out, and no container starts", async () => {
	const env = setup({ maxLive: 1, monthUsd: 45 }); // US$1.5 a day, one session reserves US$1.26
	const a = await admitted(env, "a");
	await a.runUntil(env.clock.t + 5000);
	assert.equal(env.core.join("b", "ip-b", env.clock.t).ok, false, "a second reservation does not fit today");
	env.core.state.meter.days["2026-11-01"] = 0.1; // (spent earlier today)
	env.core.state.meter.months["2026-11-01"] = 0.1;
	const b = env.make();
	// b got in line just before (as if a's reservation had been lighter)
	env.core.state.entries.push({ id: "b", addr: "ip-b", seq: 99, state: "queued", joinedAt: env.clock.t, seenAt: env.clock.t, leaseUntil: 0 });
	await b.s.create("b", "ip-b", false);
	await a.runUntil(a.s.view().stopAt! + 1, true); // a plays to the end: ~US$1.09 + 0.1 spent
	env.core.state.entries.find((e) => e.id === "b")!.seenAt = env.clock.t; // b's page kept asking
	const v = await b.s.status();
	assert.equal(v.phase, "gone");
	assert.equal(v.reason, "full");
	assert.equal(b.backend.started, null);
});

test("the kill switch stops a playing duck at its next check and frees the slot", async () => {
	const env = setup();
	const { s, backend, runUntil } = await admitted(env, "a");
	await runUntil(env.clock.t + 5000);
	(env.config as { open: boolean }).open = false; // the owner set DUCK_EDGE_OPEN=false
	await runUntil(env.clock.t + 31_000, true);
	assert.equal(s.view().reason, "closed");
	assert.deepEqual(backend.destroyed, ["closed"]);
	assert.deepEqual(env.gate.released, ["a:closed"]);
});

test("no container is started once the gate is closed", async () => {
	const env = setup();
	env.core.join("a", "ip-a", env.clock.t);
	const closed = new GateCore(env.core.state, env.core.limits, { ...env.core.budget, open: false });
	const m = env.make();
	const port = gatePort(closed, env.clock);
	const s = new SessionCore(undefined, {
		backend: m.backend,
		gate: port,
		store: { put: async () => {}, setAlarm: async () => {} },
		config: env.config,
		now: () => env.clock.t,
		randomSeed: () => 1,
	});
	await s.create("a", "ip-a", true);
	assert.equal(s.view().reason, "closed");
	assert.equal(m.backend.started, null);
	assert.deepEqual(m.backend.destroyed, [], "nothing to stop");
});
