/**
 * One visitor's duck as a local Docker container: the host's implementation of the session's
 * Backend (core/session.ts), as the Cloudflare Container is the Worker's (src/index.ts).
 *
 * Each duck: a fresh container from the duck image (`--rm`, read-only root with /tmp in memory, 2 CPUs,
 * 3 GiB, no capabilities), on the ducks' own Docker network (no route out: deploy/public/host/
 * firewall.sh), its port 8080 published on 127.0.0.1 only, at the slot's port. Every container this
 * front starts carries the label OWNER_LABEL, so a restarted front can remove what it left behind.
 */

import { spawn } from "node:child_process";
import { request as httpRequest } from "node:http";
import { Readable } from "node:stream";
import type { Backend } from "../core/session.ts";
import type { HostSettings } from "./config.ts";

export const OWNER_LABEL = "duckmind.front";

export interface ExecResult {
	code: number;
	stdout: string;
	stderr: string;
}

/** The docker command line, as a port so tests can fake it. */
export interface DockerPort {
	/** `docker <args>`, collected. */
	run(args: string[]): Promise<ExecResult>;
	/** Resolves with the container's exit code when it stops (`docker wait`); -1 if that cannot be told. */
	wait(id: string): Promise<number>;
}

export function dockerCli(binary: string): DockerPort {
	const exec = (args: string[]): Promise<ExecResult> => {
		const { promise, resolve } = Promise.withResolvers<ExecResult>();
		const child = spawn(binary, args, { stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		child.stdout.on("data", (b: Buffer) => (stdout += b.toString()));
		child.stderr.on("data", (b: Buffer) => (stderr += b.toString()));
		child.on("error", (err) => resolve({ code: -1, stdout, stderr: String(err) }));
		child.on("close", (code) => resolve({ code: code ?? -1, stdout, stderr }));
		return promise;
	};
	return {
		run: exec,
		async wait(id) {
			const r = await exec(["wait", id]);
			const code = Number(r.stdout.trim());
			return r.code === 0 && Number.isInteger(code) ? code : -1;
		},
	};
}

/** Published ports for the live slots: portBase … portBase + size - 1 (size follows MAX_LIVE). */
export class SlotPool {
	private readonly used = new Set<number>();
	private readonly base: () => number;
	private readonly size: () => number;

	constructor(base: () => number, size: () => number) {
		this.base = base;
		this.size = size;
	}

	take(): number | null {
		for (let i = 0; i < this.size(); i++) {
			const port = this.base() + i;
			if (!this.used.has(port)) {
				this.used.add(port);
				return port;
			}
		}
		return null;
	}

	give(port: number): void {
		this.used.delete(port);
	}

	inUse(): number {
		return this.used.size;
	}
}

export interface DuckRun {
	sessionId: string;
	port: number;
	env: Record<string, string>;
}

/** The `docker run` arguments for one duck. */
export function duckRunArgs(s: HostSettings, run: DuckRun): string[] {
	const name = `duckmind-duck-${run.sessionId.replace(/[^A-Za-z0-9]/g, "").slice(0, 12).toLowerCase()}`;
	const args = [
		"run",
		"--detach",
		"--rm",
		"--name",
		name,
		"--label",
		`${OWNER_LABEL}=1`,
		"--network",
		s.network,
		// No name lookups either: the ducks' only resolver is a loopback address with nothing on it.
		"--dns",
		"127.0.0.1",
		"--publish",
		`127.0.0.1:${run.port}:8080`,
		"--cpus",
		s.cpus,
		"--memory",
		s.memory,
		"--memory-swap",
		s.memory,
		"--pids-limit",
		"1024",
		"--read-only",
		"--tmpfs",
		"/tmp:rw,nosuid,nodev,size=1g",
		"--cap-drop",
		"ALL",
		"--security-opt",
		"no-new-privileges",
	];
	for (const [k, v] of Object.entries(run.env)) args.push("--env", `${k}=${v}`);
	args.push(s.image);
	return args;
}

/** Ask a duck on 127.0.0.1:`port`; the answer streams (SSE, sound) and is never decoded or buffered. */
export type PortFetch = (port: number, request: Request) => Promise<Response>;

export const loopbackFetch: PortFetch = async (port, request) => {
	const url = new URL(request.url);
	const body = request.body ? Buffer.from(await request.arrayBuffer()) : null;
	const headers: Record<string, string> = {};
	request.headers.forEach((v, k) => (headers[k] = v));
	if (body) headers["content-length"] = String(body.byteLength);
	const { promise, resolve, reject } = Promise.withResolvers<Response>();
	const req = httpRequest(
		{ host: "127.0.0.1", port, method: request.method, path: url.pathname + url.search, headers },
		(res) => {
			const h = new Headers();
			for (const [k, v] of Object.entries(res.headers)) {
				if (v === undefined) continue;
				if (Array.isArray(v)) for (const x of v) h.append(k, x);
				else h.set(k, v);
			}
			const status = res.statusCode ?? 502;
			if (status === 204 || status === 205 || status === 304 || status < 200) {
				res.resume();
				resolve(new Response(null, { status: status < 200 ? 502 : status, headers: h }));
				return;
			}
			resolve(new Response(Readable.toWeb(res) as unknown as ReadableStream<Uint8Array>, { status, statusText: res.statusMessage, headers: h }));
		},
	);
	req.on("error", reject);
	req.end(body ?? undefined);
	return promise;
};

export interface DockerBackendDeps {
	docker: DockerPort;
	settings: () => HostSettings;
	slots: SlotPool;
	fetchPort: PortFetch;
	sessionId: string;
	/** The container stopped by itself (exit code 0 = clean). */
	onStop: (clean: boolean) => void;
	log: (event: string, fields?: Record<string, unknown>) => void;
}

export class DockerBackend implements Backend {
	private readonly d: DockerBackendDeps;
	private id: string | null = null;
	private port: number | null = null;
	private alive = false;
	private destroying = false;

	constructor(deps: DockerBackendDeps) {
		this.d = deps;
	}

	async start(env: Record<string, string>): Promise<void> {
		const s = this.d.settings();
		if (!s.image) throw new Error("no duck image configured (DUCK_HOST_IMAGE)");
		const port = this.d.slots.take();
		if (port === null) throw new Error("no free slot port");
		const r = await this.d.docker.run(duckRunArgs(s, { sessionId: this.d.sessionId, port, env }));
		const id = r.stdout.trim().split("\n").pop() ?? "";
		if (r.code !== 0 || !/^[0-9a-f]{12,64}$/.test(id)) {
			this.d.slots.give(port);
			throw new Error(`docker run failed (${r.code}): ${r.stderr.trim().slice(0, 300)}`);
		}
		this.id = id;
		this.port = port;
		this.alive = true;
		void this.d.docker.wait(id).then((code) => {
			this.alive = false;
			this.d.slots.give(port);
			this.d.log("container_exit", { session: this.d.sessionId.slice(0, 8), code });
			if (!this.destroying) this.d.onStop(code === 0);
		});
	}

	running(): boolean {
		return this.alive;
	}

	fetch(request: Request): Promise<Response> {
		if (!this.alive || this.port === null) return Promise.reject(new Error("no container"));
		return this.d.fetchPort(this.port, request);
	}

	async destroy(reason: string): Promise<void> {
		if (!this.id || !this.alive) return;
		this.destroying = true;
		const r = await this.d.docker.run(["rm", "--force", this.id]);
		if (r.code !== 0) this.d.log("destroy_error", { reason, error: r.stderr.trim().slice(0, 300) });
		this.alive = false;
		if (this.port !== null) this.d.slots.give(this.port);
	}
}

/** Remove every container this front started (any state); returns how many. */
export async function removeOwned(docker: DockerPort): Promise<number> {
	const ps = await docker.run(["ps", "--all", "--quiet", "--filter", `label=${OWNER_LABEL}`]);
	const ids = ps.stdout.split(/\s+/).filter((x) => /^[0-9a-f]{12,64}$/.test(x));
	if (ids.length) await docker.run(["rm", "--force", ...ids]);
	return ids.length;
}
