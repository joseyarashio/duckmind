import assert from "node:assert/strict";
import { test } from "node:test";
import { QUICK_START_URL } from "../src/core/pages.ts";
import { type Arrival, clientAddress, HostFront, type Scheduler, type Timer } from "../src/host/app.ts";
import { parseEnvFile, readEdge, readHost } from "../src/host/config.ts";
import { type DockerPort, duckRunArgs, type ExecResult, OWNER_LABEL } from "../src/host/docker.ts";
import { type CounterReading, defaultInterface, EgressMeter, type EgressSaved } from "../src/host/egress.ts";
import { assetServer, gzipFor } from "../src/host/http.ts";
import { fileURLToPath } from "node:url";
import { constants as zlibConstants, gunzipSync, gzipSync } from "node:zlib";

const BASE = "http://front.invalid";
// The folders the host serves under /_edge/ (deploy/public/host/front.env.example points at their installed copies).
const ASSET_DIRS = {
	fonts: fileURLToPath(new URL("../../../../src/duckmind/simweb/static/fonts", import.meta.url).href),
	img: fileURLToPath(new URL("../static/img", import.meta.url).href),
};
const HOSTNAME = "duck.example.test";

/** Timers on a hand-driven clock. */
class FakeScheduler implements Scheduler {
	t = Date.parse("2026-11-13T08:00:00Z");
	private timers: Array<{ at: number; fn: () => void; live: boolean }> = [];
	now(): number {
		return this.t;
	}
	at(when: number, fn: () => void): Timer {
		const timer = { at: when, fn, live: true };
		this.timers.push(timer);
		return { cancel: () => void (timer.live = false) };
	}
	/** Fire every timer due up to `until`, in time order, letting each one's work finish. */
	async advance(ms: number): Promise<void> {
		const until = this.t + ms;
		for (;;) {
			this.timers = this.timers.filter((x) => x.live);
			const next = this.timers.reduce<(typeof this.timers)[number] | null>((a, b) => (a && a.at <= b.at ? a : b), null);
			if (!next || next.at > until) break;
			this.t = Math.max(this.t, next.at);
			next.live = false;
			next.fn();
			await settle();
		}
		this.t = until;
		await settle();
	}
}

async function settle(): Promise<void> {
	for (let i = 0; i < 30; i++) await new Promise((r) => setImmediate(r));
}

interface FakeContainer {
	id: string;
	args: string[];
	port: number;
	env: Record<string, string>;
	alive: boolean;
	bootAt: number;
	exit: (code: number) => void;
	exited: Promise<number>;
	seen: Request[];
}

/** The docker command line and the ducks' ports, faked: containers boot after `bootMs`. */
class FakeDocker implements DockerPort {
	containers: FakeContainer[] = [];
	calls: string[][] = [];
	leftovers: string[] = [];
	failRun = false;
	bootMs = 5000;
	private n = 0;
	private readonly clock: FakeScheduler;
	constructor(clock: FakeScheduler) {
		this.clock = clock;
	}

	alive(): FakeContainer[] {
		return this.containers.filter((c) => c.alive);
	}

	async run(args: string[]): Promise<ExecResult> {
		this.calls.push(args);
		if (args[0] === "run") {
			if (this.failRun) return { code: 125, stdout: "", stderr: "docker: image not found" };
			const id = (++this.n).toString(16).padStart(64, "0");
			const publish = args[args.indexOf("--publish") + 1]!;
			const env: Record<string, string> = {};
			args.forEach((a, i) => {
				if (a === "--env") {
					const [k, v] = args[i + 1]!.split("=");
					env[k!] = v!;
				}
			});
			const { promise, resolve } = Promise.withResolvers<number>();
			const c: FakeContainer = {
				id,
				args,
				port: Number(publish.split(":")[1]),
				env,
				alive: true,
				bootAt: this.clock.t + this.bootMs,
				exited: promise,
				exit: (code) => {
					if (!c.alive) return;
					c.alive = false;
					resolve(code);
				},
				seen: [],
			};
			this.containers.push(c);
			return { code: 0, stdout: `${id}\n`, stderr: "" };
		}
		if (args[0] === "rm") {
			for (const id of args.slice(2)) {
				this.containers.find((c) => c.id === id)?.exit(137);
				this.leftovers = this.leftovers.filter((x) => x !== id);
			}
			return { code: 0, stdout: "", stderr: "" };
		}
		if (args[0] === "ps") {
			const ids = [...this.leftovers, ...this.alive().map((c) => c.id)];
			return { code: 0, stdout: ids.join("\n"), stderr: "" };
		}
		return { code: 1, stdout: "", stderr: "unknown" };
	}

	wait(id: string): Promise<number> {
		return this.containers.find((c) => c.id === id)!.exited;
	}

	/** The duck at 127.0.0.1:port, answering the contract. */
	fetchPort = async (port: number, request: Request): Promise<Response> => {
		const c = this.alive().find((x) => x.port === port);
		if (!c) throw new Error("connection refused");
		c.seen.push(request);
		if (this.clock.t < c.bootAt) throw new Error("connection refused");
		const path = new URL(request.url).pathname;
		if (path === "/api/ping") return new Response("ok");
		if (path === "/api/public/session") {
			const ends = c.bootAt + Number(c.env.DUCKMIND_PUBLIC_MINUTES) * 60_000;
			return Response.json({ ok: true, ends_at: new Date(ends).toISOString() });
		}
		return new Response(`duck ${path}`, { headers: { "set-cookie": "inside=1", "content-type": "text/plain" } });
	};
}

/** Siteverify that passes the token "ok" for the right action and hostname. */
let verifyCalls = 0;
const siteverify = (async (_url: string, init: RequestInit) => {
	verifyCalls++;
	const form = new URLSearchParams(init.body as URLSearchParams);
	const ok = form.get("response") === "ok";
	return Response.json({ success: ok, action: "start", hostname: HOSTNAME });
}) as unknown as typeof fetch;

interface Env {
	clock: FakeScheduler;
	docker: FakeDocker;
	app: HostFront;
	logs: Array<Record<string, unknown>>;
}

function setup(vars: Record<string, string> = {}): Env {
	const clock = new FakeScheduler();
	const docker = new FakeDocker(clock);
	const all = {
		OPEN: "true",
		SESSION_MINUTES: "15", // (these tests' timings are written for 15-minute visits)
		TURNSTILE_HOSTNAMES: HOSTNAME,
		TURNSTILE_SITEKEY: "sitekey",
		DUCK_HOST_IMAGE: "duckmind-public:test",
		...vars,
	};
	const logs: Array<Record<string, unknown>> = [];
	const app = new HostFront({
		edge: readEdge(all),
		host: readHost(all),
		secrets: { turnstileSecret: "secret", addressKey: "key" },
		docker,
		fetchPort: docker.fetchPort,
		scheduler: clock,
		asset: assetServer(ASSET_DIRS),
		log: (event, fields) => void logs.push({ event, ...fields }),
		randomSeed: () => 4242,
		fetchFn: siteverify,
	});
	return { clock, docker, app, logs };
}

/** One browser: its cookie, its address, and how it reaches the front. */
class Visitor {
	cookie = "";
	private readonly env: Env;
	readonly ip: string;
	readonly tunnel: boolean;
	constructor(env: Env, ip: string, tunnel = true) {
		this.env = env;
		this.ip = ip;
		this.tunnel = tunnel;
	}

	async send(method: string, path: string, init: { body?: string; type?: string; headers?: Record<string, string> } = {}): Promise<Response> {
		const headers = new Headers(init.headers);
		if (this.cookie) headers.set("cookie", this.cookie);
		if (init.type) headers.set("content-type", init.type);
		headers.set("cf-connecting-ip", this.ip);
		const request = new Request(BASE + path, { method, headers, body: init.body });
		const arrival: Arrival = { tunnel: this.tunnel, peer: this.tunnel ? null : "127.0.0.1", headers };
		const res = await this.env.app.handle(request, arrival);
		for (const c of res.headers.getSetCookie()) {
			const m = /^dm_session=([^;]*)/.exec(c);
			if (m) this.cookie = m[1] ? `dm_session=${m[1]}` : "";
		}
		return res;
	}

	start(): Promise<Response> {
		return this.send("POST", "/_edge/start?lang=en", { body: "cf-turnstile-response=ok", type: "application/x-www-form-urlencoded" });
	}

	async status(): Promise<{ phase: string; place?: number; reason?: string }> {
		return (await (await this.send("GET", "/_edge/status")).json()) as { phase: string; place?: number; reason?: string };
	}
}

/** A visitor whose duck is up and playing. */
async function playing(env: Env, ip = "198.51.100.1"): Promise<Visitor> {
	const a = new Visitor(env, ip);
	assert.equal((await a.start()).status, 303);
	await env.clock.advance(6000);
	assert.equal((await a.status()).phase, "running");
	return a;
}

/** Nothing of the ended duck is left: no container, no port, no slot. */
function assertFree(env: Env): void {
	assert.equal(env.docker.alive().length, 0, "no container left");
	assert.equal(env.app.slots.inUse(), 0, "the port is free");
	assert.equal(env.app.gate.stats().live, 0, "the gate slot is free");
}

test("one duck at a time: the line is first come, first served, and the next in line gets the duck", async () => {
	const env = setup();
	const a = await playing(env, "198.51.100.1");
	const b = new Visitor(env, "198.51.100.2");
	const c = new Visitor(env, "198.51.100.3");
	assert.equal((await b.start()).status, 303);
	assert.equal((await c.start()).status, 303);
	assert.deepEqual(await b.status(), { phase: "queued", place: 1 });
	assert.deepEqual(await c.status(), { phase: "queued", place: 2 });
	assert.equal(env.docker.alive().length, 1, "max live 1: only A's duck runs");
	const page = await b.send("GET", "/?lang=en");
	assert.ok((await page.text()).includes("You are number 1"));

	await a.send("POST", "/_edge/leave");
	assert.equal(env.docker.alive().length, 0);
	assert.deepEqual(await c.status(), { phase: "queued", place: 1 }, "C stays in line: the slot is B's");
	assert.equal(env.docker.alive().length, 0, "C's poll starts no duck");
	assert.equal((await b.status()).phase, "starting");
	assert.equal(env.docker.alive().length, 1);
	assert.equal(env.docker.alive()[0]!.env.DUCKMIND_PUBLIC_SEED, "4242");
});

/** Every way a duck ends frees its container, its port and the slot, and the line moves. */
const endings: Array<[string, (env: Env, a: Visitor, b: Visitor) => Promise<void>]> = [
	[
		"time",
		async (env, _a, b) => {
			// Play 15 min + the 3-minute download, with both pages asking.
			for (let i = 0; i < 18 * 6 + 1; i++) {
				await env.clock.advance(10_000);
				await b.status();
				await _a.send("GET", "/api/body");
			}
		},
	],
	[
		"exited",
		async (env, _a, b) => {
			for (let i = 0; i < 15 * 6 + 1; i++) {
				await env.clock.advance(10_000);
				await b.status();
				await _a.send("GET", "/api/body");
			}
			env.docker.alive()[0]!.exit(0); // the duck's own exit after the download tail
			await settle();
		},
	],
	["crash", async (env) => (env.docker.alive()[0]!.exit(137), settle())],
	[
		"idle",
		async (env, _a, b) => {
			for (let i = 0; i < 16; i++) {
				await env.clock.advance(10_000);
				await b.status();
			}
		},
	],
	["left", async (_env, a) => void (await a.send("POST", "/_edge/leave"))],
];

for (const [reason, end] of endings) {
	test(`a duck that ends by '${reason}' frees its slot and the next in line gets one`, async () => {
		const env = setup();
		const a = await playing(env);
		const b = new Visitor(env, "198.51.100.2");
		await b.start();
		const duckA = env.docker.alive()[0]!;
		await end(env, a, b);
		const ended = env.logs.find((l) => l.event === "end");
		assert.equal(ended?.reason, reason);
		assert.equal(duckA.alive, false, "A's container is gone");
		assert.notEqual((await b.status()).phase, "queued", "B got the duck");
		// One slot, one port: B's duck could start only on the port A's gave back.
		assert.equal(env.docker.alive().length, 1);
		assert.notEqual(env.docker.alive()[0]!.id, duckA.id);
		assert.equal(env.app.slots.inUse(), 1);
		assert.equal(env.app.gate.stats().live, 1);
	});
}

test("a duck that fails to start, or never answers /api/ping, frees its slot", async () => {
	const env = setup();
	env.docker.failRun = true;
	const a = new Visitor(env, "198.51.100.1");
	await a.start();
	assert.equal(env.logs.find((l) => l.event === "end")?.reason, "start_failed");
	assertFree(env);

	const env2 = setup({ START_TIMEOUT_S: "60" });
	env2.docker.bootMs = 10 * 60_000;
	const b = new Visitor(env2, "198.51.100.2");
	await b.start();
	for (let i = 0; i < 7; i++) {
		await env2.clock.advance(10_000);
		await b.status();
	}
	assert.equal(env2.logs.find((l) => l.event === "end")?.reason, "start_timeout");
	assertFree(env2);
});

test("the visitor's address comes from CF-Connecting-IP only through the tunnel", async () => {
	const h = (ip: string) => new Headers({ "cf-connecting-ip": ip });
	assert.equal(clientAddress({ tunnel: true, peer: null, headers: h("203.0.113.9") }), "203.0.113.9");
	assert.equal(clientAddress({ tunnel: false, peer: "127.0.0.1", headers: h("203.0.113.9") }), "127.0.0.1");
	assert.equal(clientAddress({ tunnel: true, peer: null, headers: h("not an address") }), null);

	// Locally the header is ignored: two "different" visitors are one address, which may hold one duck.
	const local = setup();
	await new Visitor(local, "203.0.113.1", false).start();
	assert.equal((await new Visitor(local, "203.0.113.2", false).start()).status, 409);
	// Through the tunnel they are two visitors; the same address twice is still refused.
	const tunnel = setup();
	await new Visitor(tunnel, "203.0.113.1").start();
	assert.equal((await new Visitor(tunnel, "203.0.113.2").start()).status, 303);
	assert.equal((await new Visitor(tunnel, "203.0.113.2").start()).status, 409);
});

test("only the allowlisted paths reach the duck, without cookies, and no container cookie comes back", async () => {
	const env = setup();
	const a = await playing(env);
	const duck = env.docker.alive()[0]!;
	duck.seen.length = 0;
	for (const [m, p] of [["GET", "/lab"], ["GET", "/api/layout"], ["POST", "/api/life/duckfile/import"], ["GET", "/fonts/%2e%2e/x"]]) {
		assert.equal((await a.send(m!, p!)).status, 404, `${m} ${p}`);
	}
	assert.equal((await a.send("PUT", "/api/call")).status, 405);
	const big = await a.send("POST", "/api/neuro/command", { body: "x".repeat(300 * 1024), type: "application/json" });
	assert.equal(big.status, 413);
	assert.equal(duck.seen.length, 0, "nothing refused reached the duck");

	const ok = await a.send("GET", "/api/life", { headers: { "x-forwarded-for": "192.0.2.4", authorization: "Basic eA==" } });
	assert.equal(await ok.text(), "duck /api/life");
	assert.equal(ok.headers.get("set-cookie"), null);
	const seen = duck.seen.at(-1)!;
	assert.equal(seen.headers.get("cookie"), null);
	assert.equal(seen.headers.get("cf-connecting-ip"), null);
	assert.equal(seen.headers.get("x-forwarded-for"), null);
	assert.equal(seen.headers.get("authorization"), null);
});

test("closed (kill switch): the guiding page, no Siteverify, no container; reopening lets visitors in", async () => {
	const env = setup({ OPEN: "false" });
	const v = new Visitor(env, "198.51.100.1");
	const page = await v.send("GET", "/?lang=es");
	const html = await page.text();
	assert.ok(html.includes(QUICK_START_URL), "links the README quick start");
	assert.ok(html.includes("Hoy ya adoptaron a todos los patitos"));
	assert.ok(!html.includes('id="startForm"'), "no start button");
	const before = verifyCalls;
	const start = await v.start();
	assert.equal(start.status, 503);
	assert.equal(verifyCalls, before, "Siteverify is not asked");
	assert.equal(env.docker.calls.filter((c) => c[0] === "run").length, 0, "no container started");

	env.app.reload(readEdge({ OPEN: "true", TURNSTILE_HOSTNAMES: HOSTNAME }), readHost({ DUCK_HOST_IMAGE: "duckmind-public:test" }));
	assert.equal((await v.start()).status, 303);
	assert.equal(env.docker.alive().length, 1);
});

test("closing while a duck plays stops it within a check and empties the line", async () => {
	const env = setup();
	const a = await playing(env);
	const b = new Visitor(env, "198.51.100.2");
	await b.start();
	env.app.reload(readEdge({ OPEN: "false", TURNSTILE_HOSTNAMES: HOSTNAME }), readHost({ DUCK_HOST_IMAGE: "duckmind-public:test" }));
	assert.equal((await b.status()).phase, "gone", "the line is emptied");
	await env.clock.advance(16_000);
	assert.equal(env.logs.find((l) => l.event === "end" && l.reason === "closed") !== undefined, true);
	assertFree(env);
	assert.ok((await (await a.send("GET", "/")).text()).includes(QUICK_START_URL));
});

test("a full line shows the guiding page to newcomers; the start button comes back when there is room", async () => {
	const env = setup({ MAX_QUEUE: "1" });
	await playing(env, "198.51.100.1");
	const b = new Visitor(env, "198.51.100.2");
	await b.start();
	const c = new Visitor(env, "198.51.100.3");
	const html = await (await c.send("GET", "/?lang=zh-Hant")).text();
	assert.ok(html.includes("今天的小鴨都被領養了"));
	assert.ok(html.includes(QUICK_START_URL));
	assert.equal((await c.start()).status, 503);
	await b.send("POST", "/_edge/leave");
	assert.ok((await (await c.send("GET", "/?lang=zh-Hant")).text()).includes('id="startForm"'));
});

test("a restarted front removes the ducks it left behind", async () => {
	const env = setup();
	env.docker.leftovers = ["a".repeat(64), "b".repeat(64)];
	await env.app.boot();
	const ps = env.docker.calls.find((c) => c[0] === "ps")!;
	assert.ok(ps.includes(`label=${OWNER_LABEL}`));
	assert.deepEqual(env.docker.calls.at(-1), ["rm", "--force", "a".repeat(64), "b".repeat(64)]);
	assert.deepEqual(env.docker.leftovers, []);
});

test("each duck runs fresh, read-only, capped, on the ducks' network, published on loopback only", () => {
	const host = readHost({ DUCK_HOST_IMAGE: "duckmind-public:abc-arm64" });
	const args = duckRunArgs(host, { sessionId: "Ab_c-1234567890xyzXYZ00", port: 18080, env: { DUCKMIND_PUBLIC_SEED: "7", DUCKMIND_PUBLIC_MINUTES: "15" } });
	const after = (flag: string) => args[args.indexOf(flag) + 1];
	assert.equal(args[0], "run");
	for (const f of ["--rm", "--read-only", "--detach"]) assert.ok(args.includes(f), f);
	assert.equal(after("--cpus"), "2");
	assert.equal(after("--memory"), "3g");
	assert.equal(after("--publish"), "127.0.0.1:18080:8080");
	assert.equal(after("--network"), "duckmind-public");
	assert.equal(after("--label"), `${OWNER_LABEL}=1`);
	assert.equal(after("--cap-drop"), "ALL");
	assert.ok(args.includes("DUCKMIND_PUBLIC_SEED=7") && args.includes("DUCKMIND_PUBLIC_MINUTES=15"));
	assert.equal(args.at(-1), "duckmind-public:abc-arm64");
});

test("settings files: comments, quotes, host defaults, loopback-only listener", () => {
	const v = parseEnvFile("# c\nMAX_LIVE=2\nDUCK_HOST_IMAGE='img:1'\n\nOPEN=\"true\"\nbad line\n");
	assert.deepEqual(v, { MAX_LIVE: "2", DUCK_HOST_IMAGE: "img:1", OPEN: "true" });
	assert.equal(readEdge({}).maxLive, 1);
	assert.equal(readEdge({}).open, false, "no OPEN=true: closed");
	assert.equal(readEdge({}).sessionMinutes, 5);
	assert.equal(readHost({}).egressMonthBytes, 90e9, "90 GB a month unless told otherwise");
	assert.equal(readHost({ MONTHLY_EGRESS_GB: "9O" }).egressMonthBytes, 0, "a typo closes");
	assert.deepEqual(readHost({}).listen, { host: "127.0.0.1", port: 8780 });
	assert.throws(() => readHost({ DUCK_HOST_LISTEN: "0.0.0.0:8780" }));
});

/** An egress meter on `env`'s gate over a hand-moved counter and an in-memory file. */
function egressMeter(env: Env, counter: { now: CounterReading | null }, disk: { saved: EgressSaved | null }): EgressMeter {
	return new EgressMeter({
		gate: env.app.gate,
		counter: () => counter.now,
		now: () => env.clock.t,
		store: { load: () => structuredClone(disk.saved), save: (x) => void (disk.saved = structuredClone(x)) },
	});
}

test("the month's bytes: booked as the counter grows, kept across a restart of the front and of the machine", () => {
	const counter: { now: CounterReading | null } = { now: { boot: "boot-1", bytes: 5_000_000_000 } };
	const disk: { saved: EgressSaved | null } = { saved: null };
	const env = setup();
	const m = egressMeter(env, counter, disk);
	assert.equal(m.start(), 0, "first run ever: counting starts now, not at the machine's start");
	counter.now = { boot: "boot-1", bytes: 5_300_000_000 };
	assert.equal(m.tick(), 300_000_000);
	assert.equal(env.app.gate.egressMonth(env.clock.t).sentBytes, 300_000_000);
	assert.equal(disk.saved?.last, 5_300_000_000);

	// The front restarts (same machine): what was sent while it was down is booked too.
	const env2 = setup();
	counter.now = { boot: "boot-1", bytes: 5_500_000_000 };
	const m2 = egressMeter(env2, counter, disk);
	assert.equal(m2.start(), 200_000_000);
	assert.equal(env2.app.gate.egressMonth(env2.clock.t).sentBytes, 500_000_000);

	// The machine restarted: its counter starts again from zero.
	counter.now = { boot: "boot-2", bytes: 40_000_000 };
	assert.equal(m2.tick(), 40_000_000);
	assert.equal(env2.app.gate.egressMonth(env2.clock.t).sentBytes, 540_000_000);

	// A new calendar month starts from nothing; the old one stays in the file.
	env2.clock.t = Date.parse("2026-12-01T00:00:30Z");
	counter.now = { boot: "boot-2", bytes: 50_000_000 };
	assert.equal(m2.tick(), 10_000_000);
	assert.equal(env2.app.gate.egressMonth(env2.clock.t).sentBytes, 10_000_000);
	assert.equal(disk.saved?.egress.months["2026-11-01"], 540_000_000);
});

test("past the month's traffic cap: the guiding page, no duck, until the next month", async () => {
	const env = setup({ MONTHLY_EGRESS_GB: "1", SESSION_EGRESS_MB: "400" });
	env.app.gate.bookEgress(700_000_000, env.clock.t); // 300 MB left: one 400 MB session does not fit
	const v = new Visitor(env, "198.51.100.1");
	const html = await (await v.send("GET", "/?lang=en")).text();
	assert.ok(html.includes(QUICK_START_URL) && !html.includes('id="startForm"'));
	assert.equal((await v.start()).status, 503);
	assert.equal(env.docker.calls.filter((c) => c[0] === "run").length, 0);

	env.clock.t = Date.parse("2026-12-01T00:00:01Z");
	assert.ok((await (await v.send("GET", "/?lang=en")).text()).includes('id="startForm"'));
	assert.equal((await v.start()).status, 303);
	assert.equal(env.docker.alive().length, 1);
});

test("the default route's interface is the one metered", () => {
	const table = "Iface\tDestination\tGateway\nens34\t00000000\t01002CAC\ndocker0\t000011AC\t00000000\n";
	assert.equal(defaultInterface(table), "ens34");
	assert.equal(defaultInterface("Iface\tDestination\n"), null);
});

test("the duck's event stream leaves gzipped, each event flushed through as it comes", async () => {
	const { readable, writable } = new TransformStream<Uint8Array, Uint8Array>();
	const writer = writable.getWriter();
	const sse = new Response(readable, { headers: { "content-type": "text/event-stream; charset=utf-8", "content-encoding": "identity" } });
	const out = gzipFor(new Request("http://front.invalid/api/neuro/stream", { headers: { "accept-encoding": "br, gzip" } }), sse);
	assert.equal(out.headers.get("content-encoding"), "gzip");
	const reader = out.body!.getReader();
	const got: Buffer[] = [];
	const event = (n: number) => `id: ${n}\ndata: ${JSON.stringify({ brain: { mood: "calm", n }, words: "x".repeat(4000) })}\n\n`;
	for (let n = 1; n <= 3; n++) {
		await writer.write(new TextEncoder().encode(event(n)));
		// this event can be read before the next one is written
		let text = "";
		while (!text.endsWith(event(n))) {
			const { value } = await reader.read();
			got.push(Buffer.from(value!));
			text = gunzipSync(Buffer.concat(got), { finishFlush: zlibConstants.Z_SYNC_FLUSH }).toString();
		}
	}
	await writer.close();
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		got.push(Buffer.from(value));
	}
	const all = Buffer.concat(got);
	assert.equal(gunzipSync(all).toString(), event(1) + event(2) + event(3));
	assert.ok(all.length < 1000, `three 4 KB events, nearly alike: ${all.length} bytes`);
});

test("text answers leave gzipped when the client accepts it; pictures, the sound and encoded ones as they are", async () => {
	const json = JSON.stringify({ frames: Array.from({ length: 200 }, (_, i) => ({ t: i / 50, q: [0.1, 0.2, 0.3] })) });
	const ask = (enc: string | null) => new Request("http://front.invalid/api/body", { headers: enc ? { "accept-encoding": enc } : {} });
	const answer = (type: string, body: BodyInit = json, extra: Record<string, string> = {}) =>
		new Response(body, { headers: { "content-type": type, ...extra } });

	const out = gzipFor(ask("gzip, br"), answer("application/json; charset=utf-8"));
	assert.equal(out.headers.get("content-encoding"), "gzip");
	assert.equal(out.headers.get("vary"), "accept-encoding");
	const bytes = Buffer.from(await out.arrayBuffer());
	assert.equal(gunzipSync(bytes).toString(), json, "the same answer");
	assert.ok(bytes.length < json.length / 3, `${bytes.length} of ${json.length}`);

	for (const [why, res, enc] of [
		["no gzip asked", answer("application/json"), null],
		["the sound", answer("application/octet-stream"), "gzip"],
		["a picture", answer("image/jpeg"), "gzip"],
		["already gzipped (a mesh)", answer("model/stl", gzipSync(json), { "content-encoding": "gzip" }), "gzip"],
	] as const) {
		const kept = gzipFor(ask(enc), res);
		assert.equal(kept, res, why);
	}
});

test("every picture and stylesheet a page asks for comes from the front itself; nothing else does", async () => {
	const env = setup();
	const v = new Visitor(env, "198.51.100.1");
	const pages = [await (await v.send("GET", "/?lang=ja")).text()];
	assert.equal((await v.start()).status, 303);
	pages.push(await (await v.send("GET", "/?lang=ko")).text()); // waking up
	const urls = new Set(pages.flatMap((html) => [...html.matchAll(/(?:href|src)="(\/_edge\/[^"]+)"|(\/_edge\/img\/[^\s",]+)/g)].map((m) => m[1] ?? m[2]!)));
	const files = [...urls].filter((u) => /\.(css|webp)$/.test(u));
	assert.ok(files.some((u) => u.startsWith("/_edge/img/duck-")) && files.some((u) => u.startsWith("/_edge/img/screenshot-")), [...urls].join(" "));
	assert.ok(files.includes("/_edge/fonts/fonts.css") && files.includes("/_edge/fonts/noto-sans-jp.css"), [...urls].join(" "));
	for (const url of files) {
		const res = await v.send("GET", url);
		assert.equal(res.status, 200, url);
		assert.equal(res.headers.get("content-type"), url.endsWith(".css") ? "text/css; charset=utf-8" : "image/webp", url);
	}
	for (const url of ["/_edge/img/..%2f..%2fsrc%2fcore%2ffront.ts", "/_edge/img/duck.png", "/_edge/img/nothing.webp", "/_edge/static/img/duck-420.webp", "/_edge/img/"]) {
		assert.equal((await v.send("GET", url)).status, 404, url);
	}
});
