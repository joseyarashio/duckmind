/**
 * The gate: at most `maxLive` sessions at once, everyone else waits in a first-come line, and no
 * session starts unless the spending budget can pay for all of it.
 *
 * Pure state machine over a JSON-serialisable state; the Gate Durable Object (src/index.ts) loads
 * it, calls one method and writes it back. Time is passed in, so tests drive the clock.
 *
 * A live slot is a lease. The session extends it while its container starts and runs, and gives it
 * back when the session ends. If a session never comes back (visitor gone, session object lost, crash
 * nobody reported), the lease runs out and the slot goes to the next in line: no slot is held forever.
 *
 * Money (src/core/meter.ts): a container's time is metered from the moment its session says it starts
 * one (begin) until the gate hears it stopped (release) or its lease ends, whichever comes first; a
 * lease never reaches past one session's worst case, so a lost stop costs at most what was reserved.
 * Every live slot reserves one whole session at the worst case; a newcomer gets a slot (or a place in
 * line) only if what is left today and this month, minus those reservations, pays for it as well.
 *
 * Bytes (the EC2 host, budget.egress): what the host sends to the internet is booked as it goes
 * (bookEgress) per calendar month (UTC). The same rule holds: every live slot reserves one session's worst
 * case, and nobody new comes in unless this month's cap still holds them all; it reopens next month.
 */

import {
	billingMonth,
	book,
	budgetLeft,
	type BudgetSettings,
	charge,
	emptyMeter,
	type MeterState,
	monthSpent,
	nextOpening,
	prune,
	sessionWorstUsd,
} from "./meter.ts";

export interface GateLimits {
	maxLive: number;
	maxQueue: number;
	maxPerAddress: number;
	startRateMax: number;
	startWindowS: number;
	queueStaleS: number;
	admitGraceS: number;
}

export interface GateEntry {
	id: string;
	/** Keyed hash of the visitor's address (never the address itself). */
	addr: string;
	seq: number;
	state: "queued" | "live";
	joinedAt: number;
	/** Last time a queued visitor's page asked for its place. */
	seenAt: number;
	/** A live slot is freed when this passes (ms). */
	leaseUntil: number;
	/** When its container was started (ms); metered from here. */
	runFrom?: number;
}

export interface GateState {
	seq: number;
	entries: GateEntry[];
	/** addr -> start times (ms) inside the rate window. */
	starts: Record<string, number[]>;
	meter: MeterState;
	/** Bytes sent to the internet per day and calendar month (only with budget.egress). */
	egress?: MeterState;
}

/** Why nobody new can come in: kill switch, today's or this month's budget, or slots and line full. */
export type Closed = "closed" | "day" | "month" | "busy";

export type Availability =
	| { state: "open" }
	| { state: "closed" | "busy" }
	| { state: "day" | "month"; /** 00:00 UTC of the day one session fits again, if ever. */ reopensAt: number | null };

export type JoinResult =
	| { ok: true; state: "live" }
	| { ok: true; state: "queued"; place: number }
	| { ok: false; reason: "already" | "rate" | Closed };

export type PollResult =
	| { state: "live" }
	| { state: "queued"; place: number }
	/** The budget ran out (or the edge closed) while waiting: the place is gone. */
	| { state: "full" }
	| { state: "unknown" };

export type BeginResult = "ok" | "closed" | "unknown";

export interface GateStats {
	live: number;
	queued: number;
	max: number;
}

export interface Dropped {
	id: string;
	why: "stale" | "lease";
	/** US$ booked for its container (lease ended without a stop). */
	usd?: number;
}

export function emptyGate(): GateState {
	return { seq: 0, entries: [], starts: {}, meter: emptyMeter() };
}

export class GateCore {
	readonly state: GateState;
	readonly limits: GateLimits;
	readonly budget: BudgetSettings;

	constructor(state: GateState, limits: GateLimits, budget: BudgetSettings) {
		this.state = state;
		this.state.meter ??= emptyMeter();
		this.limits = limits;
		this.budget = budget;
	}

	private find(id: string): GateEntry | undefined {
		return this.state.entries.find((e) => e.id === id);
	}

	private count(st: GateEntry["state"]): number {
		let n = 0;
		for (const e of this.state.entries) if (e.state === st) n++;
		return n;
	}

	place(entry: GateEntry): number {
		let ahead = 0;
		for (const e of this.state.entries) if (e.state === "queued" && e.seq < entry.seq) ahead++;
		return ahead + 1;
	}

	private recentStarts(addr: string, now: number): number[] {
		const since = now - this.limits.startWindowS * 1000;
		const kept = (this.state.starts[addr] ?? []).filter((t) => t > since);
		if (kept.length) this.state.starts[addr] = kept;
		else delete this.state.starts[addr];
		return kept;
	}

	private get worstUsd(): number {
		return sessionWorstUsd(this.budget);
	}

	/** Bytes this calendar month may still send, before reservations (Infinity without a cap). */
	private egressLeft(now: number): number {
		const cap = this.budget.egress;
		if (!cap) return Infinity;
		return cap.monthBytes - monthSpent(this.state.egress ?? emptyMeter(), now, 1);
	}

	/** Bytes left this month after every live session's reservation (Infinity without a cap). */
	private egressRoom(now: number): number {
		const cap = this.budget.egress;
		if (!cap) return Infinity;
		return this.egressLeft(now) - this.count("live") * cap.sessionBytes;
	}

	private egressShut(now: number): Availability {
		return { state: "month", reopensAt: billingMonth(now, 1).end };
	}

	/** Closed whatever the live sessions do: kill switch, today's / this month's money spent, or this
	 * month's bytes. */
	private shut(now: number): Availability | null {
		if (!this.budget.open) return { state: "closed" };
		const left = budgetLeft(this.state.meter, now, this.budget);
		if (left.month < this.worstUsd) return { state: "month", reopensAt: nextOpening(this.state.meter, now, this.budget) };
		if (left.day < this.worstUsd) return { state: "day", reopensAt: nextOpening(this.state.meter, now, this.budget) };
		if (this.budget.egress && this.egressLeft(now) < this.budget.egress.sessionBytes) return this.egressShut(now);
		return null;
	}

	/** The host sent `bytes` to the internet (booked now, to its UTC day and calendar month). */
	bookEgress(bytes: number, now: number): void {
		if (bytes <= 0) return;
		this.state.egress ??= emptyMeter();
		book(this.state.egress, now, bytes, 1);
	}

	/** US$ still free for new sessions now: today's and this month's money minus every live reservation. */
	private room(now: number): { day: number; month: number } {
		const left = budgetLeft(this.state.meter, now, this.budget);
		const reserved = this.count("live") * this.worstUsd;
		return { day: left.day - reserved, month: left.month - reserved };
	}

	/** Whether a newcomer could get a slot or a place in line now (budget left is never shown). */
	availability(now: number): Availability {
		const shut = this.shut(now);
		if (shut) return shut;
		const room = this.room(now);
		// Everyone already in line is owed one session too.
		const owed = (this.count("queued") + 1) * this.worstUsd;
		if (room.month < owed || room.day < owed) {
			const why = room.month < owed && room.month <= room.day ? "month" : "day";
			return { state: why, reopensAt: nextOpening(this.state.meter, now, this.budget) };
		}
		const cap = this.budget.egress;
		if (cap && this.egressRoom(now) < (this.count("queued") + 1) * cap.sessionBytes) return this.egressShut(now);
		if (this.count("live") >= this.limits.maxLive && this.count("queued") >= this.limits.maxQueue) return { state: "busy" };
		return { state: "open" };
	}

	private canAdmit(now: number): boolean {
		if (this.shut(now)) return false;
		const room = this.room(now);
		if (this.budget.egress && this.egressRoom(now) < this.budget.egress.sessionBytes) return false;
		return Math.min(room.day, room.month) >= this.worstUsd;
	}

	join(id: string, addr: string, now: number): JoinResult {
		this.sweep(now);
		const existing = this.find(id);
		if (existing) {
			return existing.state === "live"
				? { ok: true, state: "live" }
				: { ok: true, state: "queued", place: this.place(existing) };
		}
		let mine = 0;
		for (const e of this.state.entries) if (e.addr === addr) mine++;
		if (mine >= this.limits.maxPerAddress) return { ok: false, reason: "already" };
		const avail = this.availability(now);
		if (avail.state !== "open") return { ok: false, reason: avail.state };
		const starts = this.recentStarts(addr, now);
		if (starts.length >= this.limits.startRateMax) return { ok: false, reason: "rate" };
		const queued = this.count("queued");
		const free = this.count("live") < this.limits.maxLive && queued === 0;
		this.state.seq += 1;
		const entry: GateEntry = {
			id,
			addr,
			seq: this.state.seq,
			state: free ? "live" : "queued",
			joinedAt: now,
			seenAt: now,
			leaseUntil: free ? now + this.limits.admitGraceS * 1000 : 0,
		};
		this.state.entries.push(entry);
		this.state.starts[addr] = [...starts, now];
		return free ? { ok: true, state: "live" } : { ok: true, state: "queued", place: this.place(entry) };
	}

	/** A queued visitor's page asks for its place (keeps the place); a live one learns it may start. */
	poll(id: string, now: number): PollResult {
		this.sweep(now);
		const e = this.find(id);
		if (!e) return { state: "unknown" };
		if (e.state === "live") return { state: "live" };
		// No money will come back (nobody is playing) or none can: the wait is over.
		if (this.shut(now) || (this.count("live") === 0 && !this.canAdmit(now))) {
			this.state.entries = this.state.entries.filter((x) => x !== e);
			return { state: "full" };
		}
		e.seenAt = now;
		return { state: "queued", place: this.place(e) };
	}

	/** The session is about to start its container: metering starts now; the lease holds until `until`. */
	begin(id: string, until: number, now: number): BeginResult {
		const e = this.find(id);
		if (!e || e.state !== "live") return "unknown";
		if (!this.budget.open) return "closed";
		e.runFrom ??= now;
		e.leaseUntil = Math.min(until, e.runFrom + this.budget.sessionWorstS * 1000);
		return "ok";
	}

	/** The session holds its slot until `until` (ms), never past one session's worst case; only a live
	 * entry whose container was begun can be extended. */
	extend(id: string, until: number): boolean {
		const e = this.find(id);
		if (!e || e.state !== "live" || e.runFrom === undefined) return false;
		e.leaseUntil = Math.min(until, e.runFrom + this.budget.sessionWorstS * 1000);
		return true;
	}

	private settle(e: GateEntry, end: number): number {
		if (e.runFrom === undefined) return 0;
		return charge(this.state.meter, e.runFrom, Math.max(e.runFrom, end), this.budget);
	}

	/** The session ended (time up, crash, failed start, visitor left): book its container's time, free
	 * its slot or its place. Returns the US$ booked, or null when the gate did not know it. */
	release(id: string, now: number): number | null {
		const e = this.find(id);
		if (!e) {
			this.promote(now);
			return null;
		}
		const usd = this.settle(e, Math.min(now, e.leaseUntil));
		this.state.entries = this.state.entries.filter((x) => x !== e);
		this.promote(now);
		return usd;
	}

	/** Drop places nobody asks for and slots whose lease ran out (booked up to the lease end), then
	 * let the line move. */
	sweep(now: number): Dropped[] {
		const dropped: Dropped[] = [];
		const staleBefore = now - this.limits.queueStaleS * 1000;
		this.state.entries = this.state.entries.filter((e) => {
			if (e.state === "queued" && e.seenAt < staleBefore) {
				dropped.push({ id: e.id, why: "stale" });
				return false;
			}
			if (e.state === "live" && e.leaseUntil <= now) {
				const d: Dropped = { id: e.id, why: "lease" };
				if (e.runFrom !== undefined) d.usd = this.settle(e, e.leaseUntil);
				dropped.push(d);
				return false;
			}
			return true;
		});
		prune(this.state.meter, now);
		if (this.state.egress) prune(this.state.egress, now);
		this.promote(now);
		return dropped;
	}

	private promote(now: number): void {
		const waiting = this.state.entries
			.filter((e) => e.state === "queued")
			.sort((a, b) => a.seq - b.seq);
		for (const e of waiting) {
			if (this.count("live") >= this.limits.maxLive || !this.canAdmit(now)) break;
			e.state = "live";
			e.leaseUntil = now + this.limits.admitGraceS * 1000;
		}
	}

	/** When the gate must look again (ms), or null when nothing can expire. */
	nextWake(): number | null {
		let next: number | null = null;
		for (const e of this.state.entries) {
			const t = e.state === "live" ? e.leaseUntil : e.seenAt + this.limits.queueStaleS * 1000;
			if (next === null || t < next) next = t;
		}
		return next;
	}

	stats(): GateStats {
		return { live: this.count("live"), queued: this.count("queued"), max: this.limits.maxLive };
	}

	/** For the owner's log only (never shown to visitors): US$ spent this month and today, and what
	 * is left today and this month before reservations. */
	/** For the owner's log: bytes sent this calendar month, and the cap (null without one). */
	egressMonth(now: number): { sentBytes: number; capBytes: number | null } {
		return {
			sentBytes: monthSpent(this.state.egress ?? emptyMeter(), now, 1),
			capBytes: this.budget.egress?.monthBytes ?? null,
		};
	}

	spending(now: number): { monthUsd: number; dayUsd: number; dayLeftUsd: number; monthLeftUsd: number } {
		const left = budgetLeft(this.state.meter, now, this.budget);
		const day = new Date(now).toISOString().slice(0, 10);
		return {
			monthUsd: this.budget.monthUsd - left.month,
			dayUsd: this.state.meter.days[day] ?? 0,
			dayLeftUsd: left.day,
			monthLeftUsd: left.month,
		};
	}
}
