/**
 * The public front door (CONTRACT.md, "The edge"), shared by both hosts: the Cloudflare Worker
 * (src/index.ts) and the EC2 host (src/host/). Routes:
 *   /_edge/…            the front's own pages and API (never reach a container)
 *   everything else     with a live session cookie (dm_session): that visitor's container, through
 *                       the session's path filter; without one: the landing page at "/", else 404.
 * When no newcomer can get a duck (line full, or the owner closed public play) the landing page is the
 * guiding page instead: today's ducks are all adopted, raise your own at home.
 *
 * The host supplies the gate, the sessions, the front's files (fonts, pictures), the visitor's address
 * (only as far as it can trust it) and the Turnstile secret.
 */

import type { EdgeConfig } from "./config.ts";
import type { Availability, GateStats, JoinResult } from "./gate.ts";
import { type Lang, pickLang } from "./i18n.ts";
import { guidingPage, type LandingError, landingPage, waitingPage } from "./pages.ts";
import { readCapped } from "./proxy.ts";
import type { SessionView } from "./session.ts";
import { verifyTurnstile } from "./turnstile.ts";

export const SESSION_COOKIE = "dm_session";
export const LANG_COOKIE = "dm_lang";
const SESSION_ID = /^[A-Za-z0-9_-]{22}$/;
/** Set on a session's own answer (its view as JSON), as opposed to the container's. */
export const VIEW_HEADER = "x-dm-edge";

export interface FrontGate {
	join(id: string, addr: string): Promise<JoinResult>;
	status(): Promise<GateStats & { availability: Availability }>;
}

export interface FrontSession {
	create(id: string, addr: string, admitted: boolean): Promise<SessionView>;
	status(): Promise<SessionView>;
	leave(): Promise<void>;
	/** A visitor request: the container's answer, or viewResponse() while there is none. */
	fetch(request: Request): Promise<Response>;
}

export interface FrontDeps {
	config: EdgeConfig;
	gate: FrontGate;
	session(id: string): FrontSession;
	/** A file of the front's own: `/fonts/…` (the simulator's fonts folder) or `/img/…` (the pages'
	 * pictures, deploy/public/edge/static/img); anything but 200 is a 404. */
	asset(path: string): Promise<Response>;
	/** Key for the visitor-address hash (addresses are never kept as they are). */
	addressKey: string;
	turnstileSecret: string;
	turnstileScriptUrl: string;
	turnstileVerifyUrl: string;
	fetchFn?: typeof fetch;
	log(event: string, fields?: Record<string, unknown>): void;
}

/** The session's answer while it has no container to pass a request to. */
export function viewResponse(view: SessionView): Response {
	return Response.json(view, { status: 409, headers: { [VIEW_HEADER]: "view" } });
}

function cookies(request: Request): Record<string, string> {
	const out: Record<string, string> = {};
	for (const part of (request.headers.get("cookie") ?? "").split(";")) {
		const i = part.indexOf("=");
		if (i > 0) out[part.slice(0, i).trim()] = part.slice(i + 1).trim();
	}
	return out;
}

function sessionCookie(id: string, maxAgeS: number): string {
	return `${SESSION_COOKIE}=${id}; Path=/; Max-Age=${maxAgeS}; HttpOnly; Secure; SameSite=Lax`;
}

export function newSessionId(): string {
	const b = crypto.getRandomValues(new Uint8Array(16));
	return btoa(String.fromCharCode(...b)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** Keyed hash of the visitor's address: enough to count starts, never the address itself. */
export async function addressHash(address: string | null, secret: string): Promise<string> {
	const key = await crypto.subtle.importKey(
		"raw",
		new TextEncoder().encode(secret || "duckmind-edge"),
		{ name: "HMAC", hash: "SHA-256" },
		false,
		["sign"],
	);
	const mac = new Uint8Array(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(address ?? "unknown")));
	return Array.from(mac.slice(0, 16), (x) => x.toString(16).padStart(2, "0")).join("");
}

interface PageContext {
	lang: Lang;
	nonce: string;
	setLang: boolean;
}

function pageHeaders(ctx: PageContext, d: FrontDeps): Headers {
	const scriptOrigin = new URL(d.turnstileScriptUrl).origin;
	const h = new Headers({
		"content-type": "text/html; charset=utf-8",
		"cache-control": "no-store",
		"x-content-type-options": "nosniff",
		"referrer-policy": "same-origin",
		"content-security-policy": [
			"default-src 'self'",
			`script-src 'nonce-${ctx.nonce}' ${scriptOrigin}`,
			`style-src 'self' 'nonce-${ctx.nonce}'`,
			`frame-src ${scriptOrigin}`,
			"img-src 'self' data:",
			"font-src 'self'",
			`connect-src 'self' ${scriptOrigin}`,
			"form-action 'self'",
			"base-uri 'none'",
			"frame-ancestors 'none'",
		].join("; "),
	});
	if (ctx.setLang) h.append("set-cookie", `${LANG_COOKIE}=${ctx.lang}; Path=/; Max-Age=31536000; Secure; SameSite=Lax`);
	return h;
}

async function landing(
	d: FrontDeps,
	ctx: PageContext,
	opts: { status?: number; error?: LandingError; ended?: SessionView; clearCookie?: boolean } = {},
): Promise<Response> {
	const { availability, ...stats } = await d.gate.status();
	const html =
		availability.state === "open"
			? landingPage({
					lang: ctx.lang,
					nonce: ctx.nonce,
					minutes: d.config.sessionMinutes,
					stats,
					sitekey: d.config.turnstileSitekey,
					scriptUrl: d.turnstileScriptUrl,
					error: opts.error,
					ended: opts.ended,
				})
			: guidingPage({ lang: ctx.lang, nonce: ctx.nonce, minutes: d.config.sessionMinutes });
	const headers = pageHeaders(ctx, d);
	if (opts.clearCookie) headers.append("set-cookie", sessionCookie("", 0));
	return new Response(html, { status: opts.status ?? 200, headers });
}

function redirect(location: string, setCookie?: string): Response {
	const h = new Headers({ location, "cache-control": "no-store" });
	if (setCookie) h.append("set-cookie", setCookie);
	return new Response(null, { status: 303, headers: h });
}

async function readForm(request: Request): Promise<URLSearchParams | null> {
	const type = request.headers.get("content-type") ?? "";
	if (!type.startsWith("application/x-www-form-urlencoded")) return null;
	const body = await readCapped(request.body, 8192);
	return body === null ? null : new URLSearchParams(new TextDecoder().decode(body));
}

async function start(
	request: Request,
	client: string | null,
	d: FrontDeps,
	ctx: PageContext,
	current: string | null,
): Promise<Response> {
	const form = await readForm(request);
	if (!form) return landing(d, ctx, { status: 400, error: "turnstile" });
	// One session per visitor: a live one is reused, not doubled.
	if (current) {
		const view = await d.session(current).status();
		if (view.phase !== "gone") return redirect(`/?lang=${ctx.lang}`);
	}
	// Nobody new can come in: the guiding page, before Siteverify is asked.
	const { availability } = await d.gate.status();
	if (availability.state !== "open") return landing(d, ctx, { status: 503 });
	const verdict = await verifyTurnstile({
		token: form.get("cf-turnstile-response"),
		secret: d.turnstileSecret,
		remoteIp: client,
		hostnames: d.config.turnstileHostnames,
		verifyUrl: d.turnstileVerifyUrl,
		fetchFn: d.fetchFn,
	});
	if (!verdict.ok) {
		d.log("turnstile_fail", { why: verdict.why });
		return landing(d, ctx, { status: 403, error: "turnstile" });
	}
	const id = newSessionId();
	const addr = await addressHash(client, d.addressKey);
	const joined = await d.gate.join(id, addr);
	if (!joined.ok) {
		if (joined.reason === "already" || joined.reason === "rate") {
			return landing(d, ctx, { status: joined.reason === "already" ? 409 : 429, error: joined.reason });
		}
		return landing(d, ctx, { status: 503 });
	}
	const view = await d.session(id).create(id, addr, joined.state === "live");
	if (view.phase === "gone") return landing(d, ctx, { status: 503, error: "start" });
	const maxAge = 3600 + (d.config.sessionMinutes + d.config.exportMinutes) * 60;
	return redirect(`/?lang=${ctx.lang}`, sessionCookie(id, maxAge));
}

const PAGE_PATHS: Record<string, true> = { "/": true, "/index.html": true };

/** One visitor request. `client`: the visitor's address as far as the host can trust it, else null. */
export async function handleFront(request: Request, client: string | null, d: FrontDeps): Promise<Response> {
	const url = new URL(request.url);
	const jar = cookies(request);
	const queryLang = url.searchParams.get("lang");
	const ctx: PageContext = {
		lang: pickLang(queryLang, jar[LANG_COOKIE] ?? null, request.headers.get("accept-language")),
		nonce: newSessionId(),
		setLang: queryLang !== null,
	};
	const raw = jar[SESSION_COOKIE];
	const current = raw && SESSION_ID.test(raw) ? raw : null;
	const path = url.pathname;

	if (path === "/_edge" || path.startsWith("/_edge/")) {
		if ((path.startsWith("/_edge/fonts/") || path.startsWith("/_edge/img/")) && request.method === "GET") {
			const res = await d.asset(path.slice("/_edge".length));
			return res.status === 200 ? res : new Response("not found", { status: 404 });
		}
		if ((path === "/_edge" || path === "/_edge/") && request.method === "GET") return landing(d, ctx);
		if (path === "/_edge/start" && request.method === "POST") return start(request, client, d, ctx, current);
		if (path === "/_edge/status" && request.method === "GET") {
			const view = current ? await d.session(current).status() : { phase: "gone" };
			return Response.json(view, { headers: { "cache-control": "no-store" } });
		}
		if (path === "/_edge/leave" && request.method === "POST") {
			if (current) await d.session(current).leave();
			return redirect(`/_edge/?lang=${ctx.lang}`, sessionCookie("", 0));
		}
		return new Response("not found", { status: 404 });
	}

	const isPage = request.method === "GET" && Object.hasOwn(PAGE_PATHS, path);
	if (!current) {
		return isPage ? landing(d, ctx) : new Response("not found", { status: 404 });
	}
	if (request.method === "POST") {
		const declared = Number(request.headers.get("content-length") ?? "0");
		if (declared > d.config.maxBodyBytes) return new Response("too large", { status: 413 });
	}
	const res = await d.session(current).fetch(request);
	if (res.headers.get(VIEW_HEADER) !== "view") return res;
	const view = (await res.json()) as SessionView;
	if (view.phase === "gone") {
		return isPage
			? landing(d, ctx, { ended: view.reason ? view : undefined, clearCookie: true })
			: new Response("not found", { status: 404, headers: { "set-cookie": sessionCookie("", 0) } });
	}
	if (!isPage) {
		return Response.json({ ok: false, phase: view.phase }, { status: 503, headers: { "retry-after": "2" } });
	}
	// Waiting in line or waking up: a fresh view (also keeps a queued place).
	const fresh = await d.session(current).status();
	if (fresh.phase === "running" || fresh.phase === "ended") return redirect(`${path}${url.search}`);
	if (fresh.phase === "gone") return landing(d, ctx, { ended: fresh, clearCookie: true });
	const html = waitingPage({ lang: ctx.lang, nonce: ctx.nonce, minutes: d.config.sessionMinutes, view: fresh });
	return new Response(html, { headers: pageHeaders(ctx, d) });
}
