/**
 * What the host sends to the internet, booked into the gate's calendar-month byte meter (core/gate.ts,
 * budget.egress) and kept in a file across restarts of the front.
 *
 * The source is a counter that only grows until the machine restarts: the network interface's sent
 * bytes (what AWS bills: the tunnel's own overhead and anything else the host sends count too), or,
 * where there is none, the bytes the front itself wrote to visitors. Each tick books what the counter
 * grew by. With the same boot the counter is continuous, so bytes sent while the front was down are
 * booked at its next tick; after a restart of the machine the counter starts again from zero.
 */

import { readFileSync } from "node:fs";
import type { GateCore } from "../core/gate.ts";
import type { MeterState } from "../core/meter.ts";

export interface CounterReading {
	/** Changes when the counter starts again from zero (machine restart). */
	boot: string;
	bytes: number;
}

export type ByteCounter = () => CounterReading | null;

/** What is kept across restarts. */
export interface EgressSaved {
	egress: MeterState;
	boot: string;
	last: number;
}

export interface EgressStore {
	load(): EgressSaved | null;
	save(saved: EgressSaved): void;
}

/** The default route's interface (from /proc/net/route), or null. */
export function defaultInterface(routeTable: string): string | null {
	for (const line of routeTable.split("\n").slice(1)) {
		const f = line.trim().split(/\s+/);
		if (f.length > 2 && f[1] === "00000000") return f[0]!;
	}
	return null;
}

/** Sent bytes of `iface` (or the default route's), with the kernel's boot id; null where Linux's files are missing. */
export function interfaceCounter(iface: string): ByteCounter | null {
	let name = iface;
	try {
		if (!name) name = defaultInterface(readFileSync("/proc/net/route", "utf8")) ?? "";
		if (!/^[A-Za-z0-9_.-]+$/.test(name)) return null;
		const boot = readFileSync("/proc/sys/kernel/random/boot_id", "utf8").trim();
		const path = `/sys/class/net/${name}/statistics/tx_bytes`;
		readFileSync(path, "utf8");
		return () => {
			try {
				return { boot, bytes: Number(readFileSync(path, "utf8").trim()) };
			} catch {
				return null;
			}
		};
	} catch {
		return null;
	}
}

export class EgressMeter {
	private readonly gate: GateCore;
	private readonly counter: ByteCounter;
	private readonly store: EgressStore;
	private readonly now: () => number;
	private prev: CounterReading | null = null;

	constructor(o: { gate: GateCore; counter: ByteCounter; store: EgressStore; now: () => number }) {
		this.gate = o.gate;
		this.counter = o.counter;
		this.store = o.store;
		this.now = o.now;
	}

	/** Take the kept months back (and the counter where it was), then book what was sent since. */
	start(): number {
		const saved = this.store.load();
		if (saved) {
			this.gate.state.egress = saved.egress;
			this.prev = { boot: saved.boot, bytes: saved.last };
		}
		return this.tick();
	}

	/** Book what the counter grew by since the last reading; returns those bytes. */
	tick(): number {
		const cur = this.counter();
		if (!cur || !Number.isFinite(cur.bytes)) return 0;
		let sent = 0;
		if (this.prev) sent = this.prev.boot === cur.boot && cur.bytes >= this.prev.bytes ? cur.bytes - this.prev.bytes : cur.bytes;
		this.prev = cur;
		this.gate.bookEgress(sent, this.now());
		this.store.save({ egress: this.gate.state.egress ?? { months: {}, days: {} }, boot: cur.boot, last: cur.bytes });
		return sent;
	}
}
