/**
 * Public play edge (CONTRACT.md, "The edge"). Routes:
 *   /_edge/…            the edge's own pages and API (never reach a container)
 *   everything else     with a live session cookie (dm_session): that visitor's container, through
 *                       the session's path filter; without one: the landing page at "/", else 404.
 */

import { DurableObject } from "cloudflare:workers";
import { isLoopbackUrl, readConfig, sessionWorstSeconds, worstUsdPerSecond, type EdgeConfig } from "./core/config.ts";
import {
	type Availability,
	type BeginResult,
	emptyGate,
	GateCore,
	type GateState,
	type GateStats,
	type JoinResult,
	type PollResult,
} from "./core/gate.ts";
import { type Lang, pickLang } from "./core/i18n.ts";
import { type LandingError, landingPage, waitingPage } from "./core/pages.ts";
import { readCapped } from "./core/proxy.ts";
import {
	type Backend,
	SessionCore,
	type SessionRecord,
	type SessionView,
} from "./core/session.ts";
import { SITEVERIFY_URL, verifyTurnstile, WIDGET_SCRIPT_URL } from "./core/turnstile.ts";

const SESSION_COOKIE = "dm_session";
const LANG_COOKIE = "dm_lang";
const SESSION_ID = /^[A-Za-z0-9_-]{22}$/;
const GATE_NAME = "gate";

function log(event: string, fields: Record<string, unknown> = {}): void {
	console.log(JSON.stringify({ event, ...fields }));
}

// The generated binding types do not know which class a self-referencing namespace holds.
function gate(env: Env): DurableObjectStub<Gate> {
	return (env.GATE as DurableObjectNamespace<Gate>).getByName(GATE_NAME);
}

function session(env: Env, id: string): DurableObjectStub<DuckSession> {
	return (env.SESSIONS as DurableObjectNamespace<DuckSession>).getByName(id);
}

function vars(env: Env): Record<string, unknown> {
	return env as object as Record<string, unknown>;
}

// ---------------------------------------------------------------- Gate (one, named "gate")

export class Gate extends DurableObject<Env> {
	private core!: GateCore;

	constructor(ctx: DurableObjectState, env: Env) {
		super(ctx, env);
		const cfg = readConfig(env);
		void ctx.blockConcurrencyWhile(async () => {
			const state = (await ctx.storage.get<GateState>("gate")) ?? emptyGate();
			this.core = new GateCore(
				state,
				{
					maxLive: cfg.maxLive,
					maxQueue: cfg.maxQueue,
					maxPerAddress: cfg.maxPerAddress,
					startRateMax: cfg.startRateMax,
					startWindowS: cfg.startWindowS,
					queueStaleS: cfg.queueStaleS,
					admitGraceS: cfg.admitGraceS,
				},
				{
					open: cfg.open,
					monthUsd: cfg.budgetMonthUsd,
					billingDay: cfg.billingDay,
					usdPerSecond: worstUsdPerSecond(cfg.instance),
					sessionWorstS: sessionWorstSeconds(cfg),
				},
			);
		});
	}

	private async commit(): Promise<void> {
		await this.ctx.storage.put("gate", this.core.state);
		const wake = this.core.nextWake();
		if (wake === null) await this.ctx.storage.deleteAlarm();
		else await this.ctx.storage.setAlarm(Math.max(wake, Date.now() + 1000));
	}

	async join(id: string, addr: string): Promise<JoinResult> {
		const r = this.core.join(id, addr, Date.now());
		await this.commit();
		log("gate_join", { session: id.slice(0, 8), ...r, ...this.core.stats() });
		return r;
	}

	async poll(id: string): Promise<PollResult> {
		const r = this.core.poll(id, Date.now());
		await this.commit();
		return r;
	}

	async begin(id: string, until: number): Promise<BeginResult> {
		const r = this.core.begin(id, until, Date.now());
		await this.commit();
		log("gate_begin", { session: id.slice(0, 8), result: r });
		return r;
	}

	async extend(id: string, until: number): Promise<boolean> {
		const ok = this.core.extend(id, until);
		await this.commit();
		return ok;
	}

	async release(id: string, reason: string): Promise<boolean> {
		const now = Date.now();
		const usd = this.core.release(id, now);
		await this.commit();
		log("gate_release", { session: id.slice(0, 8), reason, usd, ...this.core.stats(), ...this.core.spending(now) });
		return usd !== null;
	}

	/** For the edge's pages: slots, line and whether a newcomer can come in (no money figures). */
	async status(): Promise<GateStats & { availability: Availability }> {
		const availability = this.core.availability(Date.now());
		return { ...this.core.stats(), availability };
	}

	async alarm(): Promise<void> {
		const now = Date.now();
		const dropped = this.core.sweep(now);
		await this.commit();
		for (const d of dropped) log("gate_drop", { session: d.id.slice(0, 8), why: d.why, usd: d.usd });
		if (dropped.some((d) => d.usd !== undefined)) log("gate_spending", this.core.spending(now));
	}
}

// ---------------------------------------------------------------- one visitor's session

/** The visitor's Cloudflare Container (Durable Object scheduling policy). */
class ContainerBackend implements Backend {
	constructor(
		private readonly ctx: DurableObjectState,
		private readonly container: Container,
		private readonly cfg: EdgeConfig,
		private readonly onStop: (clean: boolean) => Promise<void>,
	) {}

	/** Keep the container while its Durable Object idles between polls, until the hard stop. */
	private get keepAliveMs(): number {
		return (this.cfg.sessionMinutes + this.cfg.exportMinutes + 5) * 60_000;
	}

	watch(): void {
		this.ctx.waitUntil(
			this.container.monitor().then(
				() => this.onStop(true),
				(err: unknown) => {
					log("container_error", { error: String(err) });
					return this.onStop(false);
				},
			),
		);
	}

	async start(env: Record<string, string>): Promise<void> {
		this.container.start({
			image: this.container.images.duck,
			instance: this.cfg.instance,
			enableInternet: false,
			env,
		});
		await this.container.setInactivityTimeout(this.keepAliveMs);
		this.watch();
	}

	running(): boolean {
		return this.container.running;
	}

	fetch(request: Request): Promise<Response> {
		return this.container.getTcpPort(8080).fetch(request);
	}

	async destroy(reason: string): Promise<void> {
		if (this.container.running) await this.container.destroy(reason);
	}
}

/** `cf dev` only: the local stub container (dev/stub-container.mjs) at a loopback address. */
class StubBackend implements Backend {
	private stopped = false;
	constructor(private readonly base: string) {}

	async start(env: Record<string, string>): Promise<void> {
		this.stopped = false;
		const r = await fetch(`${this.base}/__stub/start`, { method: "POST", body: JSON.stringify(env) });
		if (!r.ok) throw new Error(`stub start ${r.status}`);
	}

	running(): boolean {
		return !this.stopped;
	}

	fetch(request: Request): Promise<Response> {
		const u = new URL(request.url);
		return fetch(new Request(this.base + u.pathname + u.search, request));
	}

	async destroy(): Promise<void> {
		this.stopped = true;
		await fetch(`${this.base}/__stub/stop`, { method: "POST" }).catch(() => undefined);
	}
}

export class DuckSession extends DurableObject<Env> {
	private core!: SessionCore;
	private readonly cfg: EdgeConfig;

	constructor(ctx: DurableObjectState, env: Env) {
		super(ctx, env);
		this.cfg = readConfig(env);
		const backend = this.makeBackend();
		void ctx.blockConcurrencyWhile(async () => {
			const record = await ctx.storage.get<SessionRecord>("session");
			this.core = new SessionCore(record, {
				backend,
				gate: gate(env),
				store: {
					put: (r) => ctx.storage.put("session", r),
					setAlarm: (at) => ctx.storage.setAlarm(at),
				},
				config: this.cfg,
				now: () => Date.now(),
				randomSeed: () => crypto.getRandomValues(new Uint32Array(1))[0]! % 2_147_483_646 + 1,
				log,
			});
			if (backend instanceof ContainerBackend && ctx.container?.running) {
				await ctx.container.setInactivityTimeout((this.cfg.sessionMinutes + this.cfg.exportMinutes + 5) * 60_000);
				backend.watch();
			}
		});
	}

	private makeBackend(): Backend {
		const container = this.ctx.container;
		if (container) {
			return new ContainerBackend(this.ctx, container, this.cfg, (clean) => this.core.containerStopped(clean));
		}
		const dev = vars(this.env).DEV_UPSTREAM_URL;
		if (isLoopbackUrl(dev)) return new StubBackend(dev.replace(/\/$/, ""));
		throw new Error("DuckSession has no container and no local stub");
	}

	async create(id: string, addr: string, admitted: boolean): Promise<SessionView> {
		await this.core.create(id, addr, admitted);
		return this.core.view();
	}

	async status(): Promise<SessionView> {
		return this.core.status();
	}

	async leave(): Promise<void> {
		await this.core.leave();
	}

	/** A visitor request: the container's answer, or 409 with the session's view as JSON. */
	async fetch(request: Request): Promise<Response> {
		const res = await this.core.proxy(request);
		if (res) return res;
		return Response.json(this.core.view(), { status: 409, headers: { "x-dm-edge": "view" } });
	}

	async alarm(): Promise<void> {
		if ((await this.core.alarm()) === "purge") {
			await this.ctx.storage.deleteAlarm();
			await this.ctx.storage.deleteAll();
		}
	}
}

// ---------------------------------------------------------------- the Worker

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

function newSessionId(): string {
	const b = crypto.getRandomValues(new Uint8Array(16));
	return btoa(String.fromCharCode(...b)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** Keyed hash of the visitor's address: enough to count starts, never the address itself. */
async function addressKey(request: Request, env: Env): Promise<string> {
	const ip = request.headers.get("cf-connecting-ip") ?? "unknown";
	const secret = String(vars(env).IP_HASH_KEY ?? "");
	const key = await crypto.subtle.importKey(
		"raw",
		new TextEncoder().encode(secret || "duckmind-edge"),
		{ name: "HMAC", hash: "SHA-256" },
		false,
		["sign"],
	);
	const mac = new Uint8Array(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(ip)));
	return Array.from(mac.slice(0, 16), (x) => x.toString(16).padStart(2, "0")).join("");
}

interface PageContext {
	lang: Lang;
	nonce: string;
	setLang: boolean;
}

function pageHeaders(ctx: PageContext, env: Env, extra: Record<string, string> = {}): Headers {
	const script = turnstileScript(env);
	const scriptOrigin = new URL(script).origin;
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
		...extra,
	});
	if (ctx.setLang) h.append("set-cookie", `${LANG_COOKIE}=${ctx.lang}; Path=/; Max-Age=31536000; Secure; SameSite=Lax`);
	return h;
}

function turnstileScript(env: Env): string {
	const v = vars(env).TURNSTILE_SCRIPT_URL;
	return isLoopbackUrl(v) ? v : WIDGET_SCRIPT_URL;
}

async function landing(
	env: Env,
	cfg: EdgeConfig,
	ctx: PageContext,
	opts: { status?: number; error?: LandingError; ended?: SessionView; clearCookie?: boolean } = {},
): Promise<Response> {
	const { availability: _availability, ...stats } = await gate(env).status();
	const html = landingPage({
		lang: ctx.lang,
		nonce: ctx.nonce,
		minutes: cfg.sessionMinutes,
		stats,
		sitekey: cfg.turnstileSitekey,
		scriptUrl: turnstileScript(env),
		error: opts.error,
		ended: opts.ended,
	});
	const headers = pageHeaders(ctx, env);
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

async function start(request: Request, env: Env, cfg: EdgeConfig, ctx: PageContext, current: string | null): Promise<Response> {
	const form = await readForm(request);
	if (!form) return landing(env, cfg, ctx, { status: 400, error: "turnstile" });
	// One session per visitor: a live one is reused, not doubled.
	if (current) {
		const view = await session(env, current).status();
		if (view.phase !== "gone") return redirect(`/?lang=${ctx.lang}`);
	}
	const verdict = await verifyTurnstile({
		token: form.get("cf-turnstile-response"),
		secret: String(vars(env).TURNSTILE_SECRET ?? ""),
		remoteIp: request.headers.get("cf-connecting-ip"),
		hostnames: cfg.turnstileHostnames,
		verifyUrl: isLoopbackUrl(vars(env).TURNSTILE_VERIFY_URL) ? String(vars(env).TURNSTILE_VERIFY_URL) : SITEVERIFY_URL,
	});
	if (!verdict.ok) {
		log("turnstile_fail", { why: verdict.why });
		return landing(env, cfg, ctx, { status: 403, error: "turnstile" });
	}
	const id = newSessionId();
	const addr = await addressKey(request, env);
	const joined = await gate(env).join(id, addr);
	if (!joined.ok) {
		const error = joined.reason === "already" || joined.reason === "rate" ? joined.reason : "busy";
		return landing(env, cfg, ctx, { status: joined.reason === "already" ? 409 : 429, error });
	}
	const view = await session(env, id).create(id, addr, joined.state === "live");
	if (view.phase === "gone") return landing(env, cfg, ctx, { status: 503, error: "start" });
	const maxAge = 3600 + (cfg.sessionMinutes + cfg.exportMinutes) * 60;
	return redirect(`/?lang=${ctx.lang}`, sessionCookie(id, maxAge));
}

const PAGE_PATHS: Record<string, true> = { "/": true, "/index.html": true };

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const cfg = readConfig(env);
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
			if (path.startsWith("/_edge/fonts/") && request.method === "GET") {
				const rest = path.slice("/_edge/fonts".length);
				const res = await env.ASSETS.fetch(new Request(new URL(rest, url.origin)));
				return res.status === 200 ? res : new Response("not found", { status: 404 });
			}
			if ((path === "/_edge" || path === "/_edge/") && request.method === "GET") return landing(env, cfg, ctx);
			if (path === "/_edge/start" && request.method === "POST") return start(request, env, cfg, ctx, current);
			if (path === "/_edge/status" && request.method === "GET") {
				const view = current ? await session(env, current).status() : { phase: "gone" };
				return Response.json(view, { headers: { "cache-control": "no-store" } });
			}
			if (path === "/_edge/leave" && request.method === "POST") {
				if (current) await session(env, current).leave();
				return redirect(`/_edge/?lang=${ctx.lang}`, sessionCookie("", 0));
			}
			return new Response("not found", { status: 404 });
		}

		const isPage = request.method === "GET" && Object.hasOwn(PAGE_PATHS, path);
		if (!current) {
			return isPage ? landing(env, cfg, ctx) : new Response("not found", { status: 404 });
		}
		if (request.method === "POST") {
			const declared = Number(request.headers.get("content-length") ?? "0");
			if (declared > cfg.maxBodyBytes) return new Response("too large", { status: 413 });
		}
		const res = await session(env, current).fetch(request);
		if (res.headers.get("x-dm-edge") !== "view") return res;
		const view = (await res.json()) as SessionView;
		if (view.phase === "gone") {
			return isPage
				? landing(env, cfg, ctx, { ended: view.reason ? view : undefined, clearCookie: true })
				: new Response("not found", { status: 404, headers: { "set-cookie": sessionCookie("", 0) } });
		}
		if (!isPage) {
			return Response.json({ ok: false, phase: view.phase }, { status: 503, headers: { "retry-after": "2" } });
		}
		// Waiting in line or waking up: a fresh view (also keeps a queued place).
		const fresh = await session(env, current).status();
		if (fresh.phase === "running" || fresh.phase === "ended") return redirect(`${path}${url.search}`);
		if (fresh.phase === "gone") return landing(env, cfg, ctx, { ended: fresh, clearCookie: true });
		const html = waitingPage({ lang: ctx.lang, nonce: ctx.nonce, minutes: cfg.sessionMinutes, view: fresh });
		return new Response(html, { headers: pageHeaders(ctx, env) });
	},
} satisfies ExportedHandler<Env>;

