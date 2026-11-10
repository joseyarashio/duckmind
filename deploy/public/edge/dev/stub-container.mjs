#!/usr/bin/env node
/**
 * A tiny stand-in for one visitor's container, for `cf dev` (no Docker, no network).
 * Answers the contract's container side (CONTRACT.md): GET /api/ping (200 once "booted"),
 * GET /api/public/session, a page at "/", 410 on writes after the end. It also plays Turnstile
 * locally: /__stub/turnstile.js (a checkbox widget) and /__stub/siteverify.
 *
 *   node dev/stub-container.mjs [--port 8790] [--boot-s 3]
 * Binds 127.0.0.1 only.
 */
import { createServer } from "node:http";

const args = process.argv.slice(2);
const opt = (name, fallback) => {
	const i = args.indexOf(`--${name}`);
	return i >= 0 ? Number(args[i + 1]) : fallback;
};
const PORT = opt("port", 8790);
const BOOT_S = opt("boot-s", 3);

let duck = null; // { seed, minutes, startedAt, endsAt, readyAt }
const stamp = (t) => new Date(t).toISOString();

function send(res, status, body, type = "application/json") {
	const data = typeof body === "string" ? body : JSON.stringify(body);
	res.writeHead(status, { "content-type": type, "cache-control": "no-store" });
	res.end(data);
}

function readBody(req) {
	return new Promise((resolve) => {
		const parts = [];
		req.on("data", (c) => parts.push(c));
		req.on("end", () => resolve(Buffer.concat(parts).toString("utf8")));
	});
}

const WIDGET = `(function(){
var q=new URL(document.currentScript.src).searchParams;
window.turnstile={render:function(el,o){
var label=document.createElement("label");var box=document.createElement("input");box.type="checkbox";box.id="stubHuman";
var input=document.createElement("input");input.type="hidden";input.name="cf-turnstile-response";
label.appendChild(box);label.appendChild(document.createTextNode(" local check (stub)"));
el.appendChild(label);el.appendChild(input);
box.addEventListener("change",function(){if(box.checked){input.value="stub-ok:"+(o.action||"")+":"+location.hostname;o.callback&&o.callback(input.value)}
else{input.value="";o["expired-callback"]&&o["expired-callback"]()}});return "stub"}};
var cb=q.get("onload");if(cb&&window[cb])window[cb]();})();`;

const server = createServer(async (req, res) => {
	const url = new URL(req.url, `http://127.0.0.1:${PORT}`);
	const now = Date.now();
	const path = url.pathname;
	if (path === "/__stub/start" && req.method === "POST") {
		const env = JSON.parse((await readBody(req)) || "{}");
		const minutes = Number(env.DUCKMIND_PUBLIC_MINUTES ?? 15);
		duck = {
			seed: Number(env.DUCKMIND_PUBLIC_SEED),
			minutes,
			startedAt: now,
			readyAt: now + BOOT_S * 1000,
			endsAt: now + BOOT_S * 1000 + minutes * 60_000,
		};
		console.log(`[stub] start seed=${duck.seed} minutes=${minutes}`);
		return send(res, 200, { ok: true });
	}
	if (path === "/__stub/stop" && req.method === "POST") {
		console.log("[stub] stop");
		duck = null;
		return send(res, 200, { ok: true });
	}
	if (path === "/__stub/turnstile.js") return send(res, 200, WIDGET, "text/javascript");
	if (path === "/__stub/siteverify" && req.method === "POST") {
		const form = new URLSearchParams(await readBody(req));
		const [ok, action, hostname] = (form.get("response") ?? "").split(":");
		const good = ok === "stub-ok" && form.get("secret") === "local-stub-secret";
		return send(res, 200, good ? { success: true, action, hostname } : { success: false, "error-codes": ["invalid-input-response"] });
	}
	if (!duck || now < duck.readyAt) return send(res, 503, { ok: false, booting: Boolean(duck) });
	const ended = now >= duck.endsAt;
	if (path === "/api/ping") return send(res, 200, { ok: true });
	if (path === "/api/public/session") {
		return send(res, 200, {
			ok: true,
			seed: duck.seed,
			name: "小鴨",
			started_at: stamp(duck.readyAt),
			ends_at: stamp(duck.endsAt),
			left_s: Math.max(0, (duck.endsAt - now) / 1000),
			ended,
		});
	}
	if (req.method === "POST") {
		await readBody(req);
		return ended ? send(res, 410, { ok: false, reason: "time is up" }) : send(res, 200, { ok: true });
	}
	if (path === "/" || path === "/index.html") {
		return send(
			res,
			200,
			`<!doctype html><html lang="en"><head><meta charset="utf-8"><link rel="icon" href="data:,"><title>stub duck</title></head>` +
				`<body><h1>Stub duck</h1><p id="seed">seed ${duck.seed}</p><p id="left"></p>` +
				`<script src="/stub.js"></script></body></html>`,
			"text/html; charset=utf-8",
		);
	}
	if (path === "/stub.js") {
		return send(
			res,
			200,
			`setInterval(function(){fetch("/api/public/session").then(function(r){return r.json()}).then(function(s){document.getElementById("left").textContent=Math.round(s.left_s)+" s left"})},2000);`,
			"text/javascript",
		);
	}
	return send(res, 404, "not found", "text/plain");
});

server.listen(PORT, "127.0.0.1", () => console.log(`[stub] container stub on http://127.0.0.1:${PORT}`));
