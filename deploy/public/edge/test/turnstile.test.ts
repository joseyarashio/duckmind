import assert from "node:assert/strict";
import { test } from "node:test";
import { verifyTurnstile } from "../src/core/turnstile.ts";

const base = { secret: "s3cret", remoteIp: "198.51.100.7", hostnames: ["duck.example.test"] };

function answering(body: unknown, status = 200) {
	const calls: Array<{ url: string; form: URLSearchParams }> = [];
	const fetchFn = (async (url: string, init: RequestInit) => {
		calls.push({ url, form: new URLSearchParams(init.body as URLSearchParams) });
		return new Response(JSON.stringify(body), { status });
	}) as unknown as typeof fetch;
	return { calls, fetchFn };
}

test("a good token for this action and hostname passes, sending secret, token and address", async () => {
	const f = answering({ success: true, action: "start", hostname: "duck.example.test" });
	assert.deepEqual(await verifyTurnstile({ ...base, token: "tok", fetchFn: f.fetchFn }), { ok: true });
	assert.equal(f.calls[0]!.form.get("secret"), "s3cret");
	assert.equal(f.calls[0]!.form.get("response"), "tok");
	assert.equal(f.calls[0]!.form.get("remoteip"), "198.51.100.7");
});

test("every Turnstile failure means no duck", async () => {
	const cases: Array<[string, Parameters<typeof verifyTurnstile>[0]]> = [
		["rejected", { ...base, token: "tok", fetchFn: answering({ success: false, "error-codes": ["invalid-input-response"] }).fetchFn }],
		["action", { ...base, token: "tok", fetchFn: answering({ success: true, action: "login", hostname: "duck.example.test" }).fetchFn }],
		["hostname", { ...base, token: "tok", fetchFn: answering({ success: true, action: "start", hostname: "evil.test" }).fetchFn }],
		["http_500", { ...base, token: "tok", fetchFn: answering({}, 500).fetchFn }],
		["network", { ...base, token: "tok", fetchFn: (async () => { throw new Error("down"); }) as unknown as typeof fetch }],
		["network", { ...base, token: "tok", fetchFn: (async () => new Response("not json")) as unknown as typeof fetch }],
	];
	for (const [why, check] of cases) assert.deepEqual(await verifyTurnstile(check), { ok: false, why }, why);
});

test("no token, an oversized token or a missing secret fail without asking Siteverify", async () => {
	const f = answering({ success: true, action: "start", hostname: "duck.example.test" });
	assert.deepEqual(await verifyTurnstile({ ...base, token: null, fetchFn: f.fetchFn }), { ok: false, why: "token" });
	assert.deepEqual(await verifyTurnstile({ ...base, token: "x".repeat(2049), fetchFn: f.fetchFn }), { ok: false, why: "token" });
	assert.deepEqual(await verifyTurnstile({ ...base, secret: "", token: "tok", fetchFn: f.fetchFn }), { ok: false, why: "config" });
	assert.deepEqual(await verifyTurnstile({ ...base, hostnames: [], token: "tok", fetchFn: f.fetchFn }), { ok: false, why: "config" });
	assert.equal(f.calls.length, 0);
});
