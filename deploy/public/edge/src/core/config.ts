/**
 * Edge settings, read from the Worker's text bindings (see cloudflare.config.ts).
 * Every value has a safe default so a missing binding never opens the gate wider.
 */

export type NamedInstance = "lite" | "basic" | "standard-1" | "standard-2" | "standard-3" | "standard-4";

export type InstanceSpec = NamedInstance | { vcpu: number; memoryMib: number; diskMb: number };

/**
 * Cloudflare Containers prices (Workers Paid), from https://developers.cloudflare.com/containers/platform/pricing/
 * ("Last updated Oct 5, 2026", read 2026-11-10). The included monthly usage is left out on purpose:
 * the meter counts every second as paid, which leaves room for the Worker and Durable Object requests.
 */
export const PRICING = {
	pageUpdated: "2026-10-05",
	readOn: "2026-11-10",
	memoryGiBSecondUsd: 0.0000025,
	vcpuSecondUsd: 0.00002,
	diskGBSecondUsd: 0.00000007,
} as const;

/** Instance types on the same page: vCPU, memory (GiB), disk (GB). */
export const INSTANCE_SIZES: Record<NamedInstance, { vcpu: number; memoryGiB: number; diskGB: number }> = {
	lite: { vcpu: 1 / 16, memoryGiB: 0.25, diskGB: 2 },
	basic: { vcpu: 0.25, memoryGiB: 1, diskGB: 4 },
	"standard-1": { vcpu: 0.5, memoryGiB: 4, diskGB: 8 },
	"standard-2": { vcpu: 1, memoryGiB: 6, diskGB: 12 },
	"standard-3": { vcpu: 2, memoryGiB: 8, diskGB: 16 },
	"standard-4": { vcpu: 4, memoryGiB: 12, diskGB: 20 },
};

/** Worst-case price of one running container (US$/s): all its memory and disk, every vCPU busy. */
export function worstUsdPerSecond(spec: InstanceSpec): number {
	const s =
		typeof spec === "string"
			? INSTANCE_SIZES[spec]
			: { vcpu: spec.vcpu, memoryGiB: spec.memoryMib / 1024, diskGB: spec.diskMb / 1000 };
	return s.memoryGiB * PRICING.memoryGiBSecondUsd + s.vcpu * PRICING.vcpuSecondUsd + s.diskGB * PRICING.diskGBSecondUsd;
}

export interface EdgeConfig {
	/** Live sessions at once (gate slots). */
	maxLive: number;
	/** Visitors allowed to wait in line; beyond that the start page says "too busy". */
	maxQueue: number;
	/** Minutes each visitor plays (DUCKMIND_PUBLIC_MINUTES for the container). */
	sessionMinutes: number;
	/** Minutes after the end the .duck download still works; the container is stopped after that. */
	exportMinutes: number;
	/** Seconds the container gets to answer /api/ping before the start counts as failed. */
	startTimeoutS: number;
	/** Seconds between readiness pings while the container starts. */
	pingIntervalS: number;
	/** Seconds between health checks while it runs. */
	checkIntervalS: number;
	/** A queued visitor whose page stops asking for this long loses the place. */
	queueStaleS: number;
	/** A playing visitor whose page sends nothing for this long ends early; the slot goes to the line. */
	idleEndS: number;
	/** An admitted visitor has this long to come back before the slot goes to the next one. */
	admitGraceS: number;
	/** Starts allowed per address in startWindowS. */
	startRateMax: number;
	startWindowS: number;
	/** Live or queued sessions allowed per address at the same time. */
	maxPerAddress: number;
	/** Largest request body passed to the container. */
	maxBodyBytes: number;
	/** Container instance type. */
	instance: InstanceSpec;
	/** Hostnames a Turnstile token may come from. */
	turnstileHostnames: string[];
	turnstileSitekey: string;
	/** DUCK_EDGE_OPEN: only "true" opens; anything else closes the edge (kill switch). */
	open: boolean;
	/** DUCK_EDGE_BUDGET_MONTH_USD: container spending allowed per billing month (default 25). */
	budgetMonthUsd: number;
	/** DUCK_EDGE_BILLING_DAY: the day of the month the account's billing month starts (default 1). */
	billingDay: number;
}

/** The longest one session's container can be metered for (s): the start timeout, play with the
 * container's 60 s clock allowance, the download time, and the 60 s lease margin after each phase. */
export function sessionWorstSeconds(cfg: EdgeConfig): number {
	return cfg.startTimeoutS + cfg.sessionMinutes * 60 + cfg.exportMinutes * 60 + 120;
}

const NAMED: Record<NamedInstance, true> = {
	lite: true,
	basic: true,
	"standard-1": true,
	"standard-2": true,
	"standard-3": true,
	"standard-4": true,
};

function int(value: unknown, fallback: number, min: number, max: number): number {
	const n = typeof value === "string" && value.trim() !== "" ? Number(value) : NaN;
	if (!Number.isFinite(n)) return fallback;
	return Math.min(max, Math.max(min, Math.floor(n)));
}

/** Missing → `fallback`; present but not a number ≥ 0 → 0 (a typo closes rather than opens). */
function money(value: unknown, fallback: number): number {
	if (value === undefined || (typeof value === "string" && value.trim() === "")) return fallback;
	const n = typeof value === "string" ? Number(value) : NaN;
	return Number.isFinite(n) && n >= 0 ? Math.min(n, 10_000) : 0;
}

/** "standard-2", or a JSON object {"vcpu":1,"memoryMib":4096,"diskMb":8000}. */
export function parseInstance(value: unknown): InstanceSpec {
	if (typeof value !== "string" || value.trim() === "") return "standard-2";
	const v = value.trim();
	if (Object.hasOwn(NAMED, v)) return v as InstanceSpec;
	try {
		const o = JSON.parse(v) as Record<string, unknown>;
		const vcpu = Number(o.vcpu);
		const memoryMib = Number(o.memoryMib);
		const diskMb = Number(o.diskMb);
		if ([vcpu, memoryMib, diskMb].every((n) => Number.isInteger(n) && n > 0)) {
			return { vcpu, memoryMib, diskMb };
		}
	} catch {
		// fall through to the default
	}
	return "standard-2";
}

/** `bindings`: the Worker's env (or any object holding the same names). */
export function readConfig(bindings: object): EdgeConfig {
	const env = bindings as Record<string, unknown>;
	return {
		maxLive: int(env.MAX_LIVE, 3, 1, 200),
		maxQueue: int(env.MAX_QUEUE, 100, 0, 10_000),
		sessionMinutes: int(env.SESSION_MINUTES, 15, 1, 120),
		exportMinutes: int(env.EXPORT_MINUTES, 3, 0, 60),
		startTimeoutS: int(env.START_TIMEOUT_S, 180, 10, 900),
		pingIntervalS: int(env.PING_INTERVAL_S, 2, 1, 30),
		checkIntervalS: int(env.CHECK_INTERVAL_S, 30, 5, 600),
		queueStaleS: int(env.QUEUE_STALE_S, 60, 10, 3600),
		idleEndS: int(env.IDLE_END_S, 120, 60, 3600),
		admitGraceS: int(env.ADMIT_GRACE_S, 90, 10, 3600),
		startRateMax: int(env.START_RATE_MAX, 3, 1, 1000),
		startWindowS: int(env.START_WINDOW_S, 3600, 60, 86_400),
		maxPerAddress: int(env.MAX_PER_ADDRESS, 1, 1, 100),
		maxBodyBytes: int(env.MAX_BODY_BYTES, 256 * 1024, 1024, 10 * 1024 * 1024),
		instance: parseInstance(env.CONTAINER_INSTANCE),
		turnstileHostnames: String(env.TURNSTILE_HOSTNAMES ?? "")
			.split(",")
			.map((h) => h.trim().toLowerCase())
			.filter(Boolean),
		turnstileSitekey: String(env.TURNSTILE_SITEKEY ?? ""),
		open: String(env.OPEN ?? "").trim().toLowerCase() === "true",
		budgetMonthUsd: money(env.BUDGET_MONTH_USD, 25),
		billingDay: int(env.BILLING_DAY, 1, 1, 28),
	};
}

/** True for http(s)://127.0.0.1 or localhost URLs: the only place a dev override may point. */
export function isLoopbackUrl(value: unknown): value is string {
	if (typeof value !== "string" || value === "") return false;
	try {
		const u = new URL(value);
		return (
			(u.protocol === "http:" || u.protocol === "https:") &&
			(u.hostname === "127.0.0.1" || u.hostname === "localhost" || u.hostname === "[::1]")
		);
	} catch {
		return false;
	}
}
