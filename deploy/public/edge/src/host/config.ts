/**
 * The EC2 host's settings: the shared front's settings (core/config.ts, same names as the Worker's
 * bindings) plus the host's own DUCK_HOST_* ones, read from KEY=VALUE files (systemd EnvironmentFile
 * syntax) so `systemctl reload` can re-read them without a restart. Nothing machine- or
 * domain-specific has a default here: the install writes the files (deploy/public/host/).
 *
 * Host defaults that differ from the Worker's: one duck at a time (MAX_LIVE=1) and a short line
 * (MAX_QUEUE=6, about an hour and a half); no money meter (the instance is paid for whether a duck
 * runs or not). What can cost money is the traffic: AWS bills what the host sends to the internet past
 * 100 GB a month (shared with the owner's other machines), so the gate caps it per calendar month
 * (MONTHLY_EGRESS_GB, default 90; decimal GB, which is below AWS's binary GB).
 */

import { type EdgeConfig, readConfig, sessionWorstSeconds } from "../core/config.ts";
import type { BudgetSettings } from "../core/meter.ts";

export const HOST_DEFAULTS: Record<string, string> = {
	MAX_LIVE: "1",
	MAX_QUEUE: "6",
	SESSION_MINUTES: "5",
	EXPORT_MINUTES: "3",
	IDLE_END_S: "120",
	START_TIMEOUT_S: "180",
	PING_INTERVAL_S: "1",
	CHECK_INTERVAL_S: "15",
};

export interface HostSettings {
	/** The duck image (DUCK_HOST_IMAGE, e.g. duckmind-public:<ref>-arm64); empty → no duck starts. */
	image: string;
	/** Docker network the ducks join (made by the firewall unit; no route out). */
	network: string;
	/** First published port; slot n listens on 127.0.0.1:(portBase + n). */
	portBase: number;
	cpus: string;
	memory: string;
	/** Local listener "127.0.0.1:8780": the owner's own door (SSH forward); the CF-Connecting-IP header
	 * is ignored there and every visitor counts as the socket's address. */
	listen: { host: string; port: number } | null;
	/** Unix socket only cloudflared can open (group-only); the CF-Connecting-IP header is trusted there. */
	tunnelSocket: string | null;
	/** The simulator's fonts folder (served under /_edge/fonts/). */
	fontsDir: string;
	/** The pages' pictures, deploy/public/edge/static/img (served under /_edge/img/). */
	imagesDir: string;
	docker: string;
	/** Bytes the host may send to the internet per calendar month (UTC): MONTHLY_EGRESS_GB × 10^9. */
	egressMonthBytes: number;
	/** One session's worst case, reserved while it runs: SESSION_EGRESS_MB × 10^6. */
	egressSessionBytes: number;
	/** Where the month's bytes are kept across restarts. */
	egressFile: string;
	/** The network interface whose sent bytes are metered; "" = the default route's. */
	egressInterface: string;
}

/** KEY=VALUE lines; blank lines and # comments skipped; one pair of surrounding quotes removed. */
export function parseEnvFile(text: string): Record<string, string> {
	const out: Record<string, string> = {};
	for (const raw of text.split(/\r?\n/)) {
		const line = raw.trim();
		if (!line || line.startsWith("#")) continue;
		const i = line.indexOf("=");
		if (i <= 0) continue;
		const key = line.slice(0, i).trim();
		let value = line.slice(i + 1).trim();
		if (value.length >= 2 && (value[0] === '"' || value[0] === "'") && value.endsWith(value[0]!)) {
			value = value.slice(1, -1);
		}
		if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) out[key] = value;
	}
	return out;
}

function listenAddress(value: string | undefined): HostSettings["listen"] {
	const v = (value ?? "127.0.0.1:8780").trim();
	if (v === "" || v === "off") return null;
	const m = /^(127\.0\.0\.1|localhost|\[::1\]):(\d{1,5})$/.exec(v);
	if (!m) throw new Error(`DUCK_HOST_LISTEN must be a loopback address with a port, not ${JSON.stringify(v)}`);
	return { host: m[1] === "[::1]" ? "::1" : m[1]!, port: Number(m[2]) };
}

export function readHost(vars: Record<string, string>): HostSettings {
	const port = Number(vars.DUCK_HOST_PORT_BASE ?? "18080");
	if (!Number.isInteger(port) || port < 1024 || port > 65000) throw new Error("DUCK_HOST_PORT_BASE must be 1024-65000");
	const socket = (vars.DUCK_HOST_TUNNEL_SOCKET ?? "/run/duckmind-front/tunnel.sock").trim();
	return {
		image: (vars.DUCK_HOST_IMAGE ?? "").trim(),
		network: (vars.DUCK_HOST_NETWORK ?? "duckmind-public").trim(),
		portBase: port,
		cpus: (vars.DUCK_HOST_CPUS ?? "2").trim(),
		memory: (vars.DUCK_HOST_MEMORY ?? "3g").trim(),
		listen: listenAddress(vars.DUCK_HOST_LISTEN),
		tunnelSocket: socket === "" || socket === "off" ? null : socket,
		fontsDir: (vars.DUCK_HOST_FONTS ?? "").trim(),
		imagesDir: (vars.DUCK_HOST_IMAGES ?? "").trim(),
		docker: (vars.DUCK_HOST_DOCKER ?? "docker").trim(),
		egressMonthBytes: amount(vars.MONTHLY_EGRESS_GB, 90, 1e9),
		egressSessionBytes: amount(vars.SESSION_EGRESS_MB, SESSION_EGRESS_MB, 1e6),
		egressFile: (vars.DUCK_HOST_EGRESS_FILE ?? "/var/lib/duckmind-front/egress.json").trim(),
		egressInterface: (vars.DUCK_HOST_EGRESS_INTERFACE ?? "").trim(),
	};
}

/** One session's worst case (MB). Measured 2026-11-14 (gzip on): ~17 MB to open the duck's page, ~1.1 MB
 * a minute of play; the room's music, when a visitor keeps it on, is raw sound (~5.8 MB a minute). 18
 * minutes (play + download) of music and polls in two tabs: ~300 MB. */
export const SESSION_EGRESS_MB = 300;

/** Missing → `fallback`; present but not a number ≥ 0 → 0 (a typo closes rather than opens). */
function amount(value: string | undefined, fallback: number, unit: number): number {
	if (value === undefined || value.trim() === "") return fallback * unit;
	const n = Number(value);
	return Number.isFinite(n) && n >= 0 ? n * unit : 0;
}

/** The front's settings for the host: HOST_DEFAULTS, then the files' values. */
export function readEdge(vars: Record<string, string>): EdgeConfig {
	return readConfig({ ...HOST_DEFAULTS, ...vars });
}

/** The gate's budget on the host: open or closed (the kill switch), never limited by money, capped in
 * bytes sent per calendar month; a slot's lease still never outlasts one session's worst case. */
export function hostBudget(edge: EdgeConfig, host: HostSettings): BudgetSettings {
	return {
		open: edge.open,
		monthUsd: 0,
		billingDay: 1,
		usdPerSecond: 0,
		sessionWorstS: sessionWorstSeconds(edge),
		egress: { monthBytes: host.egressMonthBytes, sessionBytes: host.egressSessionBytes },
	};
}
