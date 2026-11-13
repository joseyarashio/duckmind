/**
 * Node's HTTP server around the front: requests in as web Requests, answers streamed back out as
 * they come (the duck's SSE and sound streams are never buffered), plus the front's files and a count
 * of the bytes sent to visitors (what AWS bills as egress, before the tunnel's own overhead).
 *
 * Text answers leave gzipped when the request accepts it (Cloudflare asks the origin for "br, gzip"
 * and hands each visitor an encoding it takes): the duck's event stream above all (a whole brain
 * snapshot five times a second, nearly the same each time: most of what a playing page costs), its
 * JSON polls, its page and scripts. Each piece the duck sends is flushed through at once (a sync
 * flush), so a stream's events arrive when they did before; the page gets the same bytes and AWS
 * bills a fraction. Pictures, fonts, the sound and anything already encoded go as they are.
 */

import { readFile } from "node:fs/promises";
import { constants as zlibConstants, createGzip } from "node:zlib";
import type { IncomingMessage, ServerResponse } from "node:http";
import { createServer, type Server } from "node:http";
import { extname, join } from "node:path";
import { Readable } from "node:stream";
import type { Socket } from "node:net";
import type { Arrival } from "./app.ts";

/** Every request URL is parsed against this base; the Host header is never trusted. */
const BASE = "http://front.invalid";

/** Hop-by-hop headers, never passed on. */
const HOP: Record<string, true> = {
	connection: true,
	"keep-alive": true,
	"transfer-encoding": true,
	upgrade: true,
	"proxy-connection": true,
	te: true,
	trailer: true,
};

/** Types worth compressing (already-compressed ones — JPEG, woff2, gzipped meshes — are not). */
const TEXT_TYPE = /^(?:application\/(?:json|javascript)|text\/(?:html|css|javascript|plain|event-stream)|image\/svg\+xml)\b/;

/** `source` gzipped, every piece flushed through as soon as it arrives (nothing is held back). */
export function flushingGzip(source: ReadableStream<Uint8Array>): ReadableStream<Uint8Array> {
	const gzip = createGzip({ level: 6 });
	const reader = source.getReader();
	return new ReadableStream<Uint8Array>({
		start(controller) {
			gzip.on("data", (chunk: Buffer) => controller.enqueue(new Uint8Array(chunk)));
			gzip.on("end", () => controller.close());
			gzip.on("error", (err) => controller.error(err));
			void (async () => {
				try {
					for (;;) {
						const { done, value } = await reader.read();
						if (done) break;
						gzip.write(value);
						const { promise, resolve } = Promise.withResolvers<void>();
						gzip.flush(zlibConstants.Z_SYNC_FLUSH, () => resolve());
						await promise;
					}
					gzip.end();
				} catch (err) {
					gzip.destroy(err as Error);
				}
			})();
		},
		cancel(reason) {
			gzip.destroy();
			return reader.cancel(reason);
		},
	});
}

/** `response` gzipped for `request` when that pays: a whole text answer the client accepts gzip for. */
export function gzipFor(request: Request, response: Response): Response {
	const type = response.headers.get("content-type") ?? "";
	const encoding = response.headers.get("content-encoding");
	if (
		!response.body ||
		response.status !== 200 ||
		(encoding !== null && encoding !== "identity") ||
		!TEXT_TYPE.test(type) ||
		!/\bgzip\b/.test(request.headers.get("accept-encoding") ?? "")
	) {
		return response;
	}
	const headers = new Headers(response.headers);
	headers.delete("content-length");
	headers.set("content-encoding", "gzip");
	headers.append("vary", "accept-encoding");
	return new Response(flushingGzip(response.body), { status: response.status, statusText: response.statusText, headers });
}

export function toRequest(req: IncomingMessage): Request {
	const headers = new Headers();
	for (let i = 0; i < req.rawHeaders.length; i += 2) {
		const k = req.rawHeaders[i]!.toLowerCase();
		if (!Object.hasOwn(HOP, k) && k !== "host") headers.append(k, req.rawHeaders[i + 1]!);
	}
	const method = req.method ?? "GET";
	const hasBody = method !== "GET" && method !== "HEAD";
	return new Request(new URL(req.url ?? "/", BASE), {
		method,
		headers,
		body: hasBody ? (Readable.toWeb(req) as unknown as ReadableStream<Uint8Array>) : undefined,
		duplex: "half",
	} as RequestInit);
}

/** Write `response` to the visitor, chunk by chunk; stops reading when the visitor goes away. */
export async function sendResponse(res: ServerResponse, response: Response): Promise<void> {
	res.statusCode = response.status;
	for (const [k, v] of response.headers) {
		if (!Object.hasOwn(HOP, k) && k !== "set-cookie") res.setHeader(k, v);
	}
	const cookies = response.headers.getSetCookie();
	if (cookies.length) res.setHeader("set-cookie", cookies);
	if (!response.body) {
		res.end();
		return;
	}
	res.flushHeaders();
	const reader = response.body.getReader();
	let gone = false;
	res.once("close", () => {
		gone = true;
		void reader.cancel().catch(() => undefined);
	});
	try {
		for (;;) {
			const { done, value } = await reader.read();
			if (done || gone) break;
			if (!res.write(value)) {
				const { promise, resolve } = Promise.withResolvers<void>();
				res.once("drain", resolve);
				res.once("close", resolve);
				await promise;
			}
		}
	} catch {
		// the duck's stream broke off: end what the visitor has
	} finally {
		if (!gone) res.end();
	}
}

/** Bytes written to visitors' sockets since the last take(). */
export class TrafficMeter {
	private readonly last = new Map<Socket, number>();
	private carried = 0;

	watch(socket: Socket): void {
		this.last.set(socket, 0);
		socket.once("close", () => {
			this.carried += socket.bytesWritten - (this.last.get(socket) ?? 0);
			this.last.delete(socket);
		});
	}

	take(): number {
		let n = this.carried;
		this.carried = 0;
		for (const [s, before] of this.last) {
			n += s.bytesWritten - before;
			this.last.set(s, s.bytesWritten);
		}
		return n;
	}
}

export type Handler = (request: Request, arrival: Arrival) => Promise<Response>;

/** An HTTP server for one listener (`tunnel`: the cloudflared socket). */
export function frontServer(handle: Handler, tunnel: boolean, meter: TrafficMeter, log: (e: string, f?: Record<string, unknown>) => void): Server {
	const server = createServer(async (req, res) => {
		res.socket?.setNoDelay(true);
		let response: Response;
		try {
			const request = toRequest(req);
			response = gzipFor(request, await handle(request, { tunnel, peer: req.socket.remoteAddress ?? null, headers: request.headers }));
		} catch (err) {
			log("request_error", { error: String(err) });
			response = new Response("the front had a problem", { status: 500, headers: { "content-type": "text/plain; charset=utf-8" } });
		}
		await sendResponse(res, response);
	});
	server.on("connection", (s: Socket) => meter.watch(s));
	// Streams last up to 5 minutes; cloudflared keeps its connections open between requests.
	server.requestTimeout = 120_000;
	server.headersTimeout = 30_000;
	server.keepAliveTimeout = 75_000;
	server.timeout = 0;
	return server;
}

const ASSET_TYPES: Record<string, string> = {
	".css": "text/css; charset=utf-8",
	".woff2": "font/woff2",
	".txt": "text/plain; charset=utf-8",
	".webp": "image/webp",
};
const SEGMENT = /^[A-Za-z0-9_-][A-Za-z0-9_.-]*$/;

/** A file of the front's own: the first path segment names the folder ("fonts", "img"), then plain
 * path segments only, known types only. */
export function assetServer(folders: Record<string, string>): (path: string) => Promise<Response> {
	return async (path) => {
		const [folder = "", ...segments] = path.split("/").slice(1);
		const dir = Object.hasOwn(folders, folder) ? folders[folder] : "";
		const type = ASSET_TYPES[extname(path)];
		if (!dir || !type || segments.length === 0 || !segments.every((s) => SEGMENT.test(s))) {
			return new Response("not found", { status: 404 });
		}
		try {
			const body = await readFile(join(dir, ...segments));
			return new Response(body, { headers: { "content-type": type, "cache-control": "public, max-age=86400" } });
		} catch {
			return new Response("not found", { status: 404 });
		}
	};
}
