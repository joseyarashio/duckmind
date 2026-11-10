/**
 * What the edge passes to a visitor's container (defence in depth: the container's public mode has
 * its own allowlist; this one keeps everything else from ever reaching it).
 *
 * The list mirrors CONTRACT.md ("The edge" → "Paths the edge passes"). The container never sees the
 * visitor's cookies or any header outside REQUEST_HEADERS, and a body larger than the limit is refused
 * before it is read past the limit.
 */

const GET_EXACT: Record<string, true> = {
	"/": true,
	"/index.html": true,
	"/api/ping": true,
	"/api/public/session": true,
	"/api/robot": true,
	"/api/body": true,
	"/api/home": true,
	"/api/eye.jpg": true,
	"/api/clips": true,
	"/api/life": true,
	"/api/life/growth": true,
	"/api/life/fastforward": true,
	"/api/life/duckfile": true,
	"/api/life/duckfile/export": true,
	"/api/neuro": true,
	"/api/neuro/status": true,
	"/api/neuro/telemetry": true,
	"/api/neuro/knows": true,
	"/api/neuro/stream": true,
	"/api/neuro/sound/status": true,
	"/api/neuro/sound/pcm": true,
	"/api/live/status": true,
	"/api/neuro/voice/status": true,
};

const POST_EXACT: Record<string, true> = {
	"/api/call": true,
	"/api/neuro/stimulus": true,
	"/api/neuro/help": true,
	"/api/neuro/command": true,
	"/api/neuro/feedback": true,
	"/api/neuro/gothere": true,
	"/api/neuro/motion": true,
	"/api/neuro/sound/world": true,
};

/** Folders whose files the page loads (any depth below them). */
const GET_FOLDERS = ["/fonts/", "/vendor/", "/api/voice/"];

/** One plain segment: letters, digits, dot, dash, underscore; never starting with a dot. */
const SEGMENT = /^[A-Za-z0-9_-][A-Za-z0-9_.-]*$/;
const TOP_LEVEL_FILE = /^\/[A-Za-z0-9_-][A-Za-z0-9_.-]*\.(?:html|css|js|svg|woff2)$/;

export const MAX_URL_LENGTH = 2048;

/** True when `method pathname` may reach the container. `pathname` is the raw URL path. */
export function allowedPath(method: string, pathname: string): boolean {
	if (pathname.length > MAX_URL_LENGTH || !pathname.startsWith("/")) return false;
	// Plain segments only: encoded bytes, backslashes, empty or dot segments never pass.
	if (pathname !== "/") {
		const segments = pathname.slice(1).split("/");
		if (!segments.every((s) => SEGMENT.test(s))) return false;
	}
	if (method === "POST") return Object.hasOwn(POST_EXACT, pathname);
	if (method !== "GET") return false;
	if (Object.hasOwn(GET_EXACT, pathname) || TOP_LEVEL_FILE.test(pathname)) return true;
	if (pathname.startsWith("/mesh/")) return pathname.split("/").length === 3;
	return GET_FOLDERS.some((f) => pathname.startsWith(f) && pathname.length > f.length);
}

/** Request headers the container may see; everything else (cookies, auth, client addresses) stays. */
const REQUEST_HEADERS = [
	"accept",
	"accept-encoding",
	"accept-language",
	"cache-control",
	"content-type",
	"if-modified-since",
	"if-none-match",
	"last-event-id",
	"range",
];

export type Forward = { ok: true; request: Request } | { ok: false; status: 403 | 404 | 405 | 411 | 413; reason: string };

/**
 * Build the request for the container at `origin` (e.g. "http://container"), or say why not.
 * A body is read at most `maxBody + 1` bytes deep.
 */
export async function prepareForward(request: Request, origin: string, maxBody: number): Promise<Forward> {
	const url = new URL(request.url);
	if (request.method !== "GET" && request.method !== "POST") {
		return { ok: false, status: 405, reason: "method" };
	}
	if (!allowedPath(request.method, url.pathname) || url.search.length > MAX_URL_LENGTH) {
		return { ok: false, status: 404, reason: "path" };
	}
	const headers = new Headers();
	for (const name of REQUEST_HEADERS) {
		const v = request.headers.get(name);
		if (v !== null) headers.set(name, v);
	}
	let body: ArrayBuffer | undefined;
	if (request.method === "POST") {
		const declared = request.headers.get("content-length");
		if (declared !== null) {
			const n = Number(declared);
			if (!Number.isInteger(n) || n < 0) return { ok: false, status: 411, reason: "length" };
			if (n > maxBody) return { ok: false, status: 413, reason: "too_large" };
		}
		const read = await readCapped(request.body, maxBody);
		if (read === null) return { ok: false, status: 413, reason: "too_large" };
		body = read;
	}
	const target = new URL(url.pathname + url.search, origin);
	return { ok: true, request: new Request(target.toString(), { method: request.method, headers, body }) };
}

/** The whole body if it is at most `max` bytes, else null (stops reading at max + 1). */
export async function readCapped(stream: ReadableStream<Uint8Array> | null, max: number): Promise<ArrayBuffer | null> {
	if (stream === null) return new ArrayBuffer(0);
	const reader = stream.getReader();
	const chunks: Uint8Array[] = [];
	let total = 0;
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		total += value.byteLength;
		if (total > max) {
			await reader.cancel();
			return null;
		}
		chunks.push(value);
	}
	const out = new Uint8Array(total);
	let at = 0;
	for (const c of chunks) {
		out.set(c, at);
		at += c.byteLength;
	}
	return out.buffer;
}

/** The container's answer for the visitor: streamed as is (the SSE stream too), minus cookies. */
export function visitorResponse(upstream: Response): Response {
	const headers = new Headers(upstream.headers);
	headers.delete("set-cookie");
	headers.set("x-content-type-options", "nosniff");
	headers.set("referrer-policy", "same-origin");
	return new Response(upstream.body, { status: upstream.status, statusText: upstream.statusText, headers });
}
