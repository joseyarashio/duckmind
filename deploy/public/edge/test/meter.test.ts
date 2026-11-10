import assert from "node:assert/strict";
import { test } from "node:test";
import { PRICING, readConfig, sessionWorstSeconds, worstUsdPerSecond } from "../src/core/config.ts";
import {
	billingMonth,
	type BudgetSettings,
	budgetLeft,
	charge,
	emptyMeter,
	nextOpening,
	prune,
} from "../src/core/meter.ts";

// Days and months are UTC whatever the machine's zone is: run with a zone far from UTC.
process.env.TZ = "Pacific/Kiritimati";

/** 1 s of container = US$0.001, one session = 1000 s = US$1; US$30 a month = 1 session a day in November. */
const B: BudgetSettings = { open: true, monthUsd: 30, billingDay: 1, usdPerSecond: 0.001, sessionWorstS: 1000 };
const at = (iso: string) => Date.parse(iso);
const close = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

test("the worst-case price is memory + vCPU + disk of the instance type at the pricing page's rates", () => {
	assert.equal(PRICING.pageUpdated, "2026-10-05");
	close(worstUsdPerSecond("standard-2"), 6 * 0.0000025 + 1 * 0.00002 + 12 * 0.00000007);
	close(worstUsdPerSecond("standard-3"), 8 * 0.0000025 + 2 * 0.00002 + 16 * 0.00000007);
	close(worstUsdPerSecond({ vcpu: 2, memoryMib: 6144, diskMb: 12000 }), 6 * 0.0000025 + 2 * 0.00002 + 12 * 0.00000007);
	// one session's whole lease: start timeout, 15 min play (+60 s clock allowance), 3 min download, 2 lease margins
	assert.equal(sessionWorstSeconds(readConfig({ EXPORT_MINUTES: "3" })), 180 + 900 + 180 + 120);
});

test("budget settings: default US$25, kill switch only opens on an explicit true, a typo closes", () => {
	assert.equal(readConfig({}).budgetMonthUsd, 25);
	assert.equal(readConfig({ BUDGET_MONTH_USD: "12.5" }).budgetMonthUsd, 12.5);
	assert.equal(readConfig({ BUDGET_MONTH_USD: "25usd" }).budgetMonthUsd, 0);
	assert.equal(readConfig({ BUDGET_MONTH_USD: "-3" }).budgetMonthUsd, 0);
	assert.equal(readConfig({ OPEN: "true" }).open, true);
	assert.equal(readConfig({ OPEN: "false" }).open, false);
	assert.equal(readConfig({ OPEN: "ture" }).open, false);
	assert.equal(readConfig({}).open, false);
	assert.equal(readConfig({ BILLING_DAY: "31" }).billingDay, 28);
});

test("running time is booked to the UTC day it ran on, split at midnight", () => {
	const m = emptyMeter();
	const usd = charge(m, at("2026-11-10T23:50:00Z"), at("2026-11-11T00:10:00Z"), B);
	close(usd, 1.2);
	close(m.days["2026-11-10"]!, 0.6);
	close(m.days["2026-11-11"]!, 0.6);
	close(m.months["2026-11-01"]!, 1.2);
});

test("a session across the month's end is booked half to each month (UTC)", () => {
	const m = emptyMeter();
	charge(m, at("2026-10-31T23:55:00Z"), at("2026-11-01T00:05:00Z"), B);
	close(m.months["2026-10-01"]!, 0.3);
	close(m.months["2026-11-01"]!, 0.3);
	close(m.days["2026-10-31"]!, 0.3);
});

test("a billing month can start on another day than the 1st", () => {
	const bm = billingMonth(at("2026-11-10T12:00:00Z"), 15);
	assert.equal(bm.key, "2026-10-15");
	assert.equal(bm.end, at("2026-11-15T00:00:00Z"));
	assert.equal(billingMonth(at("2026-11-15T00:00:00Z"), 15).key, "2026-11-15");
	assert.equal(billingMonth(at("2026-01-03T00:00:00Z"), 5).key, "2025-12-05");
});

test("the daily cap is what is left this month over the days left; an unused day carries over", () => {
	const m = emptyMeter();
	close(budgetLeft(m, at("2026-11-01T08:00:00Z"), B).day, 1); // 30 / 30 days
	// nothing spent for ten days: 30 / 20 days left
	close(budgetLeft(m, at("2026-11-11T08:00:00Z"), B).day, 1.5);
	charge(m, at("2026-11-11T09:00:00Z"), at("2026-11-11T09:16:40Z"), B); // US$1 today
	close(budgetLeft(m, at("2026-11-11T20:00:00Z"), B).day, 0.5); // today's cap stays 1.5
	close(budgetLeft(m, at("2026-11-11T20:00:00Z"), B).month, 29);
	close(budgetLeft(m, at("2026-11-12T00:00:00Z"), B).day, 29 / 19); // the new day
	close(budgetLeft(m, at("2026-11-30T23:59:00Z"), B).day, 29); // the last day may use all of it
	close(budgetLeft(m, at("2026-12-01T00:00:00Z"), B).day, 30 / 31); // a new month starts over
});

test("the next opening is the first later day one whole session fits again", () => {
	const m = emptyMeter();
	charge(m, at("2026-11-10T00:00:00Z"), at("2026-11-10T08:00:00Z"), { ...B, usdPerSecond: 0.000875 }); // US$25.2
	// 4.8 left, one session needs 1: 4.8 / 5 days left on the 26th is too little, 4.8 / 4 on the 27th fits
	assert.equal(nextOpening(m, at("2026-11-10T09:00:00Z"), B), at("2026-11-27T00:00:00Z"));
	charge(m, at("2026-11-10T09:00:00Z"), at("2026-11-10T10:20:00Z"), B); // the rest of November
	// December: 30 / 31 days is just under one session on the 1st, 30 / 30 fits on the 2nd
	assert.equal(nextOpening(m, at("2026-11-10T11:00:00Z"), B), at("2026-12-02T00:00:00Z"));
	assert.equal(nextOpening(m, at("2026-11-10T11:00:00Z"), { ...B, monthUsd: 0.5 }), null);
});

test("old days and months are forgotten", () => {
	const m = emptyMeter();
	charge(m, at("2026-01-02T00:00:00Z"), at("2026-01-02T00:00:10Z"), B);
	charge(m, at("2026-11-02T00:00:00Z"), at("2026-11-02T00:00:10Z"), B);
	prune(m, at("2026-11-10T00:00:00Z"));
	assert.deepEqual(Object.keys(m.days), ["2026-11-02"]);
	assert.deepEqual(Object.keys(m.months).sort(), ["2026-01-01", "2026-11-01"]);
});
