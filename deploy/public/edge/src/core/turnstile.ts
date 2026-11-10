/**
 * Server-side Turnstile check before a duck is started (browser → this Worker → Siteverify).
 * Fails closed: a missing secret, a network error, a non-2xx answer, a wrong action or a hostname
 * outside the list all mean "no".
 */

export const SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify";
export const WIDGET_SCRIPT_URL = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
export const TURNSTILE_ACTION = "start";

export interface TurnstileCheck {
	token: unknown;
	secret: string | undefined;
	remoteIp: string | null;
	hostnames: string[];
	verifyUrl?: string;
	fetchFn?: typeof fetch;
	timeoutMs?: number;
}

export type TurnstileResult = { ok: true } | { ok: false; why: string };

export async function verifyTurnstile(c: TurnstileCheck): Promise<TurnstileResult> {
	if (typeof c.token !== "string" || c.token.length === 0 || c.token.length > 2048) {
		return { ok: false, why: "token" };
	}
	if (!c.secret || c.hostnames.length === 0) return { ok: false, why: "config" };
	const form = new URLSearchParams({ secret: c.secret, response: c.token });
	if (c.remoteIp) form.set("remoteip", c.remoteIp);
	let result: { success?: unknown; action?: unknown; hostname?: unknown };
	try {
		const r = await (c.fetchFn ?? fetch)(c.verifyUrl ?? SITEVERIFY_URL, {
			method: "POST",
			headers: { "content-type": "application/x-www-form-urlencoded" },
			body: form,
			signal: AbortSignal.timeout(c.timeoutMs ?? 10_000),
		});
		if (!r.ok) return { ok: false, why: `http_${r.status}` };
		result = (await r.json()) as typeof result;
	} catch {
		return { ok: false, why: "network" };
	}
	if (result.success !== true) return { ok: false, why: "rejected" };
	if (result.action !== TURNSTILE_ACTION) return { ok: false, why: "action" };
	if (typeof result.hostname !== "string" || !c.hostnames.includes(result.hostname.toLowerCase())) {
		return { ok: false, why: "hostname" };
	}
	return { ok: true };
}
