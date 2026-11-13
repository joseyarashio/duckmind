/**
 * The public play front on the EC2 host (deploy/public/host/duckmind-front.service runs it):
 *   node src/host/main.ts <settings file>...
 * Later files override earlier ones. The secrets (TURNSTILE_SECRET, IP_HASH_KEY) come from
 * $CREDENTIALS_DIRECTORY/secrets.env (systemd LoadCredential: the file stays root-only) or from
 * DUCK_HOST_SECRETS. SIGHUP re-reads every file (kill switch, limits, the next duck's image);
 * SIGTERM stops every duck and exits.
 */

import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import type { Server } from "node:http";
import { randomUUID } from "node:crypto";
import { dirname, join } from "node:path";
import { realScheduler, HostFront, type HostSecrets } from "./app.ts";
import { type HostSettings, parseEnvFile, readEdge, readHost } from "./config.ts";
import { dockerCli, loopbackFetch } from "./docker.ts";
import { type ByteCounter, EgressMeter, type EgressSaved, interfaceCounter } from "./egress.ts";
import { assetServer, frontServer, TrafficMeter } from "./http.ts";
import { TEST_SECRET_PASS } from "../core/turnstile.ts";

function log(event: string, fields: Record<string, unknown> = {}): void {
	console.log(JSON.stringify({ event, ...fields }));
}

const files = process.argv.slice(2);
if (files.length === 0) {
	console.error("usage: node src/host/main.ts <settings file>...");
	process.exit(2);
}

function readVars(): Record<string, string> {
	const vars: Record<string, string> = {};
	for (const f of files) Object.assign(vars, parseEnvFile(readFileSync(f, "utf8")));
	return vars;
}

function readSecrets(): HostSecrets {
	const dir = process.env.CREDENTIALS_DIRECTORY;
	const path = process.env.DUCK_HOST_SECRETS || (dir ? join(dir, "secrets.env") : "");
	const vars = path && existsSync(path) ? parseEnvFile(readFileSync(path, "utf8")) : {};
	return { turnstileSecret: vars.TURNSTILE_SECRET ?? "", addressKey: vars.IP_HASH_KEY ?? "" };
}

const vars = readVars();
const edge = readEdge(vars);
const host: HostSettings = readHost(vars);
const secrets = readSecrets();
if (!secrets.turnstileSecret) log("warning", { message: "no TURNSTILE_SECRET: every start is refused" });
if (secrets.turnstileSecret === TEST_SECRET_PASS) log("warning", { message: "Turnstile TEST secret: everyone passes" });
if (!secrets.addressKey) log("warning", { message: "no IP_HASH_KEY: addresses are hashed with a fixed key" });

const docker = dockerCli(host.docker);
const app = new HostFront({
	edge,
	host,
	secrets,
	docker,
	fetchPort: loopbackFetch,
	scheduler: realScheduler,
	asset: assetServer({ fonts: host.fontsDir, img: host.imagesDir }),
	log,
});
await app.boot();

const meter = new TrafficMeter();
// The month's bytes: the interface's sent bytes where Linux has them, else what the front wrote to visitors.
let visitorBytes = 0;
const run = randomUUID();
const fromVisitors: ByteCounter = () => ({ boot: run, bytes: visitorBytes });
const nic = interfaceCounter(host.egressInterface);
const egress = new EgressMeter({
	gate: app.gate,
	counter: nic ?? fromVisitors,
	now: () => Date.now(),
	store: {
		load(): EgressSaved | null {
			try {
				return JSON.parse(readFileSync(host.egressFile, "utf8")) as EgressSaved;
			} catch {
				return null;
			}
		},
		save(saved) {
			try {
				mkdirSync(dirname(host.egressFile), { recursive: true });
				writeFileSync(`${host.egressFile}.new`, JSON.stringify(saved));
				renameSync(`${host.egressFile}.new`, host.egressFile);
			} catch (err) {
				log("egress_save_error", { error: String(err) });
			}
		},
	},
});
const booted = egress.start();
log("egress", { source: nic ? "interface" : "visitors", booked_since_last_run: booted, ...app.stats() });
const handle = app.handle.bind(app);
const servers: Server[] = [];
if (host.listen) {
	const s = frontServer(handle, false, meter, log);
	s.listen(host.listen.port, host.listen.host);
	servers.push(s);
	log("listen", { at: `${host.listen.host}:${host.listen.port}`, tunnel: false });
}
if (host.tunnelSocket) {
	if (existsSync(host.tunnelSocket)) unlinkSync(host.tunnelSocket);
	const s = frontServer(handle, true, meter, log);
	const socket = host.tunnelSocket;
	s.listen(socket, () => chmodSync(socket, 0o660));
	servers.push(s);
	log("listen", { at: socket, tunnel: true });
}

// Once a minute: the month's bytes booked (and kept); a line while anything is sent to visitors.
setInterval(() => {
	const bytes = meter.take();
	visitorBytes += bytes;
	const sent = egress.tick();
	app.egressBooked();
	if (bytes > 0) log("traffic", { bytes_per_min: bytes, host_bytes_per_min: sent, ...app.stats() });
}, 60_000).unref();

process.on("SIGHUP", () => {
	try {
		const v = readVars();
		app.reload(readEdge(v), readHost(v));
	} catch (err) {
		log("reload_error", { error: String(err) });
	}
});

let stopping = false;
async function shutdown(): Promise<void> {
	if (stopping) return;
	stopping = true;
	log("stopping", app.stats());
	for (const s of servers) s.close();
	await app.stop();
	visitorBytes += meter.take();
	egress.tick();
	process.exit(0);
}
process.on("SIGTERM", () => void shutdown());
process.on("SIGINT", () => void shutdown());
