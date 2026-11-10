/**
 * The spending meter: what the containers have cost, per billing month and per day (UTC), and how
 * much is left today.
 *
 * Pure functions over a JSON-serialisable state the gate stores with its own. Container time is
 * priced at the instance type's worst case (all of its memory, disk and vCPU, every second; see
 * PRICING in config.ts), so the real bill is never higher than what is metered here.
 *
 * Budget: `monthUsd` per billing month (from `billingDay` 00:00 UTC to the same day next month;
 * day 1 is the calendar month). Each day may spend (monthUsd - spent before today) / days left in the
 * month, today included: an unused day carries over to the rest of the month.
 */

export interface MeterState {
	/** Billing-month start date (YYYY-MM-DD) -> US$ spent. */
	months: Record<string, number>;
	/** Day (YYYY-MM-DD, UTC) -> US$ spent. */
	days: Record<string, number>;
}

export interface BudgetSettings {
	/** False closes the edge: no new sessions, the line is emptied, running ducks are stopped. */
	open: boolean;
	/** Container spending allowed per billing month (US$). */
	monthUsd: number;
	/** Day of the month (1-28) the billing month starts, 00:00 UTC. */
	billingDay: number;
	/** Worst-case price of one running container (US$ per second). */
	usdPerSecond: number;
	/** The longest one session's container can be metered for (s): its whole lease. */
	sessionWorstS: number;
}

export const DAY_MS = 86_400_000;

export function emptyMeter(): MeterState {
	return { months: {}, days: {} };
}

function isoDate(t: number): string {
	return new Date(t).toISOString().slice(0, 10);
}

export function dayStart(t: number): number {
	return Math.floor(t / DAY_MS) * DAY_MS;
}

/** The billing month holding `t`: [start, end) in ms, keyed by its start date. */
export function billingMonth(t: number, billingDay: number): { start: number; end: number; key: string } {
	const d = new Date(t);
	const month = d.getUTCMonth() - (d.getUTCDate() < billingDay ? 1 : 0);
	const start = Date.UTC(d.getUTCFullYear(), month, billingDay);
	const end = Date.UTC(d.getUTCFullYear(), month + 1, billingDay);
	return { start, end, key: isoDate(start) };
}

/** One session's worst-case cost (US$). */
export function sessionWorstUsd(b: BudgetSettings): number {
	return b.sessionWorstS * b.usdPerSecond;
}

/** Book a container's running time [from, to) (ms), split at UTC midnights. Returns the US$ booked. */
export function charge(m: MeterState, from: number, to: number, b: BudgetSettings): number {
	let total = 0;
	let t = from;
	while (t < to) {
		const next = Math.min(to, dayStart(t) + DAY_MS);
		const usd = ((next - t) / 1000) * b.usdPerSecond;
		const day = isoDate(t);
		const month = billingMonth(t, b.billingDay).key;
		m.days[day] = (m.days[day] ?? 0) + usd;
		m.months[month] = (m.months[month] ?? 0) + usd;
		total += usd;
		t = next;
	}
	return total;
}

export interface BudgetLeft {
	/** US$ the billing month still allows. */
	month: number;
	/** US$ today still allows (today's cap minus today's spending). */
	day: number;
}

/** What is left this month and today, before any running session's reservation. */
export function budgetLeft(m: MeterState, now: number, b: BudgetSettings): BudgetLeft {
	const bm = billingMonth(now, b.billingDay);
	const monthSpent = m.months[bm.key] ?? 0;
	const today = dayStart(now);
	const daySpent = m.days[isoDate(today)] ?? 0;
	const daysLeft = Math.round((bm.end - today) / DAY_MS);
	const dayCap = (b.monthUsd - (monthSpent - daySpent)) / daysLeft;
	return { month: b.monthUsd - monthSpent, day: dayCap - daySpent };
}

/**
 * The first day after today on which one whole session fits again, assuming nothing more is spent
 * (ms, 00:00 UTC), or null when the budget can never pay for one session.
 */
export function nextOpening(m: MeterState, now: number, b: BudgetSettings): number | null {
	const worst = sessionWorstUsd(b);
	if (b.monthUsd < worst) return null;
	const current = billingMonth(now, b.billingDay).key;
	for (let d = dayStart(now) + DAY_MS, i = 0; i < 64; d += DAY_MS, i++) {
		const bm = billingMonth(d, b.billingDay);
		const spent = bm.key === current ? (m.months[current] ?? 0) : 0;
		const daysLeft = Math.round((bm.end - d) / DAY_MS);
		if (b.monthUsd - spent >= worst && (b.monthUsd - spent) / daysLeft >= worst) return d;
	}
	return null;
}

/** Forget days older than ~2 months and billing months older than a year. */
export function prune(m: MeterState, now: number): void {
	const oldestDay = isoDate(now - 70 * DAY_MS);
	const oldestMonth = isoDate(now - 400 * DAY_MS);
	for (const k of Object.keys(m.days)) if (k < oldestDay) delete m.days[k];
	for (const k of Object.keys(m.months)) if (k < oldestMonth) delete m.months[k];
}
