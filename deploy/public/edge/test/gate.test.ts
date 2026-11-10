import assert from "node:assert/strict";
import { test } from "node:test";
import { emptyGate, GateCore, type GateLimits } from "../src/core/gate.ts";
import type { BudgetSettings } from "../src/core/meter.ts";

const LIMITS: GateLimits = {
	maxLive: 2,
	maxQueue: 3,
	maxPerAddress: 1,
	startRateMax: 3,
	startWindowS: 3600,
	queueStaleS: 60,
	admitGraceS: 90,
};

/** Plenty of money: these tests are about slots and the line. */
const RICH: BudgetSettings = { open: true, monthUsd: 10_000, billingDay: 1, usdPerSecond: 0.000001, sessionWorstS: 1380 };

function gate(limits: Partial<GateLimits> = {}): GateCore {
	return new GateCore(emptyGate(), { ...LIMITS, ...limits }, RICH);
}

test("the first maxLive visitors play, the rest wait in the order they came", () => {
	const g = gate();
	assert.deepEqual(g.join("a", "ip-a", 0), { ok: true, state: "live" });
	assert.deepEqual(g.join("b", "ip-b", 1), { ok: true, state: "live" });
	assert.deepEqual(g.join("c", "ip-c", 2), { ok: true, state: "queued", place: 1 });
	assert.deepEqual(g.join("d", "ip-d", 3), { ok: true, state: "queued", place: 2 });
	assert.deepEqual(g.join("e", "ip-e", 4), { ok: true, state: "queued", place: 3 });
	assert.deepEqual(g.stats(), { live: 2, queued: 3, max: 2 });
});

test("a freed slot goes to the first in line, and everyone behind moves up one", () => {
	const g = gate();
	for (const [i, id] of ["a", "b", "c", "d", "e"].entries()) g.join(id, `ip-${id}`, i);
	g.release("b", 10); // b's time is up
	assert.deepEqual(g.poll("c", 11), { state: "live" });
	assert.deepEqual(g.poll("d", 11), { state: "queued", place: 1 });
	assert.deepEqual(g.poll("e", 11), { state: "queued", place: 2 });
	g.release("a", 12); // a crashed
	assert.deepEqual(g.poll("d", 13), { state: "live" });
	assert.deepEqual(g.poll("e", 13), { state: "queued", place: 1 });
});

test("a newcomer never jumps the line even when a slot is free for a moment", () => {
	const g = gate({ maxLive: 1 });
	g.join("a", "ip-a", 0);
	g.join("b", "ip-b", 1);
	g.release("a", 2); // b is admitted at once, before the newcomer arrives
	assert.deepEqual(g.join("c", "ip-c", 3), { ok: true, state: "queued", place: 1 });
	assert.deepEqual(g.poll("b", 4), { state: "live" });
});

test("a slot whose lease runs out is freed (session lost or never reported its end)", () => {
	const g = gate({ maxLive: 1, queueStaleS: 10_000 });
	g.join("a", "ip-a", 0);
	g.begin("a", 1_000_000, 0); // running until t = 1000 s
	g.join("b", "ip-b", 1);
	assert.deepEqual(g.poll("b", 999_000), { state: "queued", place: 1 });
	assert.deepEqual(g.sweep(1_000_000), [{ id: "a", why: "lease", usd: 1000 * RICH.usdPerSecond }]);
	assert.deepEqual(g.poll("a", 1_000_001), { state: "unknown" });
	assert.equal(g.stats().live, 1);
});

test("an admitted visitor who never comes back loses the slot after the grace time", () => {
	const g = gate({ maxLive: 1 });
	g.join("a", "ip-a", 0);
	g.join("b", "ip-b", 0);
	g.release("a", 1000); // b admitted at t = 1 s, lease 90 s
	g.join("c", "ip-c", 2000);
	assert.equal(g.nextWake(), 2000 + 60_000); // c's place goes stale first unless c keeps asking
	g.poll("c", 50_000); // c keeps asking
	g.poll("c", 90_000);
	const dropped = g.sweep(91_001);
	assert.deepEqual(dropped, [{ id: "b", why: "lease" }]);
	assert.deepEqual(g.poll("c", 91_002), { state: "live" });
});

test("a place nobody asks for goes stale; the others keep theirs and move up", () => {
	const g = gate({ maxLive: 1 });
	g.join("a", "ip-a", 0);
	g.join("b", "ip-b", 0);
	g.join("c", "ip-c", 0);
	g.poll("c", 50_000);
	assert.deepEqual(g.sweep(61_000), [{ id: "b", why: "stale" }]);
	assert.deepEqual(g.poll("c", 61_001), { state: "queued", place: 1 });
});

test("one session per address at a time; the next one after it ends", () => {
	const g = gate();
	g.join("a", "same", 0);
	assert.deepEqual(g.join("a2", "same", 1), { ok: false, reason: "already" });
	g.release("a", 2);
	assert.deepEqual(g.join("a3", "same", 3), { ok: true, state: "live" });
});

test("the same session joining twice keeps its place (no double count)", () => {
	const g = gate({ maxLive: 1 });
	g.join("a", "ip-a", 0);
	g.join("b", "ip-b", 1);
	assert.deepEqual(g.join("b", "ip-b", 2), { ok: true, state: "queued", place: 1 });
	assert.equal(g.stats().queued, 1);
});

test("starts per address are rate limited within the window, then allowed again", () => {
	const g = gate({ maxPerAddress: 5, startRateMax: 2, startWindowS: 600 });
	assert.equal(g.join("a", "ip", 0).ok, true);
	g.release("a", 1);
	assert.equal(g.join("b", "ip", 2).ok, true);
	g.release("b", 3);
	assert.deepEqual(g.join("c", "ip", 4), { ok: false, reason: "rate" });
	assert.equal(g.join("d", "ip", 600_001).ok, true);
});

test("a full line turns newcomers away instead of growing", () => {
	const g = gate({ maxLive: 1, maxQueue: 1 });
	g.join("a", "ip-a", 0);
	g.join("b", "ip-b", 0);
	assert.deepEqual(g.join("c", "ip-c", 0), { ok: false, reason: "busy" });
});

test("state survives a JSON round trip (what the Durable Object stores)", () => {
	const g = gate();
	for (const id of ["a", "b", "c"]) g.join(id, `ip-${id}`, 0);
	const again = new GateCore(JSON.parse(JSON.stringify(g.state)), LIMITS, RICH);
	again.release("a", 1);
	assert.deepEqual(again.poll("c", 2), { state: "live" });
});

// ---------------------------------------------------------------- money

/** One session's worst case = 1000 s × US$0.001 = US$1. US$75 from 1 November (30 days) = US$2.5 a day. */
const B: BudgetSettings = { open: true, monthUsd: 75, billingDay: 1, usdPerSecond: 0.001, sessionWorstS: 1000 };
const T = Date.parse("2026-11-01T08:00:00Z");
const S = 1000;

function paid(budget: Partial<BudgetSettings> = {}, limits: Partial<GateLimits> = {}): GateCore {
	return new GateCore(emptyGate(), { ...LIMITS, maxLive: 3, maxQueue: 5, queueStaleS: 100_000, admitGraceS: 100_000, ...limits }, { ...B, ...budget });
}

const spentOn = (g: GateCore, day: string) => g.state.meter.days[day] ?? 0;
const close = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

test("a container is metered from begin to release at the instance's worst-case rate", () => {
	const g = paid();
	g.join("a", "ip-a", T);
	assert.equal(g.begin("a", T + 900 * S, T + 5 * S), "ok");
	g.extend("a", T + 950 * S);
	close(g.release("a", T + 605 * S)!, 0.6); // ran 600 s
	close(spentOn(g, "2026-11-01"), 0.6);
	close(g.state.meter.months["2026-11-01"]!, 0.6);
});

test("an admitted visitor who never starts a container costs nothing", () => {
	const g = paid();
	g.join("a", "ip-a", T);
	assert.equal(g.release("a", T + 60 * S), 0);
	assert.deepEqual(g.state.meter.days, {});
});

test("a stop that never comes (crash, lost session) is booked until the lease ends, not longer", () => {
	const g = paid();
	g.join("a", "ip-a", T);
	g.begin("a", T + 240 * S, T);
	g.extend("a", T + 700 * S); // the session last said: hold until 700 s
	const dropped = g.sweep(T + 5000 * S);
	assert.equal(dropped.length, 1);
	close(dropped[0]!.usd!, 0.7);
	close(spentOn(g, "2026-11-01"), 0.7);
	assert.equal(g.release("a", T + 6000 * S), null, "a late stop is not booked twice");
	close(spentOn(g, "2026-11-01"), 0.7);
});

test("a stop heard after the lease ended is booked only up to the lease end", () => {
	const g = paid();
	g.join("a", "ip-a", T);
	g.begin("a", T + 300 * S, T);
	close(g.release("a", T + 400 * S)!, 0.3);
});

test("no lease reaches past one session's worst case, so nothing runs beyond what was reserved", () => {
	const g = paid();
	g.join("a", "ip-a", T);
	g.begin("a", T + 5000 * S, T);
	assert.equal(g.extend("a", T + 9000 * S), true);
	g.sweep(T + 10_000 * S);
	close(spentOn(g, "2026-11-01"), 1); // 1000 s worst case
	const g2 = paid();
	g2.join("b", "ip-b", T);
	assert.equal(g2.extend("b", T + 900 * S), false, "no lease extension before the container is begun");
});

test("a newcomer is let in only if today's money left, minus every running session's reservation, pays one whole session", () => {
	const g = paid(); // US$2.5 today, US$1 per session
	assert.deepEqual(g.join("a", "ip-a", T), { ok: true, state: "live" });
	assert.deepEqual(g.join("b", "ip-b", T), { ok: true, state: "live" });
	const third = g.join("c", "ip-c", T);
	assert.equal(third.ok, false, "a third session would go past today's US$2.5");
	assert.equal(!third.ok && third.reason, "day");
	assert.equal(g.stats().live, 2, "the third slot stays empty");
	// a's container ran only 300 s: US$0.7 of its reservation comes back
	g.begin("a", T + 2000 * S, T);
	g.release("a", T + 300 * S);
	assert.deepEqual(g.join("c", "ip-c", T + 301 * S), { ok: true, state: "live" });
	// now 0.3 spent + 2 reserved: 0.2 left, not enough
	assert.equal(g.join("d", "ip-d", T + 302 * S).ok, false);
});

test("the line is only as long as the day's money can pay for", () => {
	const g = paid({ monthUsd: 300 }, { maxLive: 1 }); // US$10 today
	g.join("a", "ip-a", T);
	for (const id of ["b", "c", "d", "e", "f"]) assert.equal(g.join(id, `ip-${id}`, T).ok, true, id);
	assert.equal(g.join("g", "ip-g", T).ok, false, "maxQueue 5");
	const tight = paid({ monthUsd: 90 }, { maxLive: 1 }); // US$3 today: 1 playing + 2 waiting
	tight.join("a", "ip-a", T);
	assert.equal(tight.join("b", "ip-b", T).ok, true);
	assert.equal(tight.join("c", "ip-c", T).ok, true);
	const d = tight.join("d", "ip-d", T);
	assert.equal(!d.ok && d.reason, "day");
});

test("once the day's money is spent the line waits for none: queued visitors are told it is full", () => {
	const g = paid({ monthUsd: 60 }, { maxLive: 1 }); // US$2 today
	g.join("a", "ip-a", T);
	g.join("b", "ip-b", T);
	g.begin("a", T + 2000 * S, T);
	// a runs its whole worst case and crashes without a word: US$1 booked, 1 left for b
	g.sweep(T + 1000 * S);
	assert.deepEqual(g.poll("b", T + 1001 * S), { state: "live" });
	g.begin("b", T + 5000 * S, T + 1001 * S); // its lease is held to its worst case: T + 2001 s
	g.join("c", "ip-c", T + 1002 * S); // refused (US$2 used or reserved)
	g.sweep(T + 2001 * S);
	close(spentOn(g, "2026-11-01"), 2);
	assert.deepEqual(g.availability(T + 2002 * S), { state: "day", reopensAt: Date.parse("2026-11-02T00:00:00Z") });
	const q = paid({ monthUsd: 60 }, { maxLive: 1 });
	q.join("a", "ip-a", T);
	q.join("b", "ip-b", T); // waiting while a plays
	q.begin("a", T + 2000 * S, T);
	q.state.meter.days["2026-11-01"] = 1.5; // (other spending today)
	q.state.meter.months["2026-11-01"] = 1.5;
	q.release("a", T + 600 * S); // 0.6 more: 2.1 of 2 used
	assert.deepEqual(q.poll("b", T + 601 * S), { state: "full" });
	assert.deepEqual(q.poll("b", T + 602 * S), { state: "unknown" });
});

test("a new UTC day opens again with what the month has left (carry-over)", () => {
	const g = paid({ monthUsd: 30 }); // US$1 a day
	g.join("a", "ip-a", T);
	g.begin("a", T + 2000 * S, T);
	g.release("a", T + 1000 * S); // US$1: today is used up
	const refused = g.join("b", "ip-b", T + 1001 * S);
	assert.equal(!refused.ok && refused.reason, "day");
	assert.deepEqual(g.join("b", "ip-b", Date.parse("2026-11-02T00:00:01Z")), { ok: true, state: "live" }); // 29 / 29
	// nothing spent from the 2nd to the 11th: the 12th may spend (30 - 1) / 19
	const quiet = paid({ monthUsd: 30 });
	quiet.state.meter.months["2026-11-01"] = 1;
	quiet.state.meter.days["2026-11-01"] = 1;
	const t12 = Date.parse("2026-11-12T10:00:00Z");
	assert.equal(quiet.join("x", "ip-x", t12).ok, true);
	assert.equal(quiet.join("y", "ip-y", t12).ok, false, "29 / 19 = 1.53 pays one session, not two");
});

test("a month's budget used up closes until the next month, which starts afresh (UTC)", () => {
	const g = paid({ monthUsd: 30 });
	g.state.meter.months["2026-11-01"] = 29.5;
	g.state.meter.days["2026-11-30"] = 29.5;
	const late = Date.parse("2026-11-30T23:00:00Z");
	const r = g.join("a", "ip-a", late);
	assert.equal(!r.ok && r.reason, "month");
	assert.deepEqual(g.availability(late), { state: "month", reopensAt: Date.parse("2026-12-02T00:00:00Z") });
	assert.deepEqual(g.join("a", "ip-a", Date.parse("2026-12-02T00:00:01Z")), { ok: true, state: "live" });
});

test("the kill switch closes everything: no newcomers, the line is emptied, no container may start", () => {
	const open = paid({}, { maxLive: 1 });
	open.join("a", "ip-a", T);
	open.join("b", "ip-b", T);
	const closed = new GateCore(JSON.parse(JSON.stringify(open.state)), open.limits, { ...B, open: false });
	assert.deepEqual(closed.join("c", "ip-c", T), { ok: false, reason: "closed" });
	assert.deepEqual(closed.poll("b", T + S), { state: "full" });
	assert.equal(closed.begin("a", T + 900 * S, T + S), "closed");
	assert.deepEqual(closed.availability(T), { state: "closed" });
});

test("a full line with every slot busy is 'busy' while money is left", () => {
	const g = paid({ monthUsd: 3000 }, { maxLive: 1, maxQueue: 1 });
	g.join("a", "ip-a", T);
	g.join("b", "ip-b", T);
	assert.deepEqual(g.availability(T), { state: "busy" });
	const c = g.join("c", "ip-c", T);
	assert.equal(!c.ok && c.reason, "busy");
});

test("the meter survives a JSON round trip with the gate (what the Durable Object stores)", () => {
	const g = paid();
	g.join("a", "ip-a", T);
	g.begin("a", T + 900 * S, T);
	const again = new GateCore(JSON.parse(JSON.stringify(g.state)), g.limits, g.budget);
	close(again.release("a", T + 500 * S)!, 0.5);
	close(spentOn(again, "2026-11-01"), 0.5);
});
