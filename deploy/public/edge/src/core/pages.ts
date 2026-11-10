/**
 * The edge's own pages: landing (start a duck), waiting (line / waking up) and "time is up".
 * Look follows docs/simulator-style.md: ink, cream dot-grid paper, square corners, thick ink borders,
 * hard offset shadows, Anton / Noto Sans TC 900 display type, orange primary button with a magenta
 * shadow, a yellow ticker. Fonts are the simulator's own (served under /_edge/fonts/).
 */

import { LANG_NAMES, LANGS, type Lang, t } from "./i18n.ts";
import type { EndReason, SessionView } from "./session.ts";

/** Turnstile widget language codes. */
const TURNSTILE_LANG: Record<Lang, string> = { "zh-Hant": "zh-tw", en: "en", ja: "ja", es: "es", "zh-Hans": "zh-cn", ko: "ko" };

export const LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/";

export function esc(value: unknown): string {
	return String(value)
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&#39;");
}

const CSS = `
:root{--ink:#101018;--ink-deep:#08080c;--ink-2:#1c1c28;--paper:#faf8f2;--cream:#f2ecdd;--white:#fff;
--yellow:#ffd23f;--orange:#ff7a2f;--orange-ink:#c2500a;--magenta:#ff2fa8;--cyan:#2ff0e6;--muted:#5e5e68;
--font-display:"Anton","Noto Sans TC",sans-serif;--font-body:"DM Sans","Noto Sans TC",system-ui,sans-serif}
*{box-sizing:border-box}
html{background:var(--ink)}
body{margin:0;color:var(--ink);font:16px/1.55 var(--font-body);background:var(--cream);
background-image:radial-gradient(rgba(16,16,24,.16) 1px,transparent 1.2px);background-size:18px 18px;overflow-x:clip;
min-height:100vh;display:flex;flex-direction:column}
main{overflow-x:clip;flex:1 0 auto}
a{color:inherit}
:focus-visible{outline:3px solid var(--orange);outline-offset:2px}
.topbar{background:var(--ink);color:var(--white);display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;
justify-content:space-between;padding:10px clamp(14px,4vw,40px);border-bottom:3px solid var(--ink-deep)}
.brand{font:400 22px/1 var(--font-display);font-synthesis:none;text-transform:uppercase;letter-spacing:.02em;color:var(--yellow);text-decoration:none}
.langs{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0;padding:0;list-style:none}
.langs a{display:inline-flex;align-items:center;min-height:36px;padding:4px 9px;border:2px solid var(--white);color:var(--white);
text-decoration:none;font-size:14px;font-weight:700}
.langs a[aria-current="true"]{background:var(--yellow);color:var(--ink);border-color:var(--yellow)}
.wrap{max-width:1080px;margin:0 auto;padding:clamp(24px,5vw,56px) clamp(14px,4vw,40px) 40px}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--orange-ink);margin:0 0 10px}
.display{font:400 clamp(40px,7vw,84px)/1.02 var(--font-display);font-synthesis:none;text-transform:uppercase;margin:0 0 18px;
color:var(--ink);text-shadow:5px 5px 0 var(--yellow);overflow-wrap:anywhere}
.display .outline{display:block;color:var(--cream);-webkit-text-stroke:2px var(--ink);paint-order:stroke fill;text-shadow:none}
:lang(zh-Hant) .display,:lang(zh-Hans) .display,:lang(ja) .display,:lang(ko) .display{font-weight:900;font-family:"Noto Sans TC","Anton",sans-serif}
.lead{font-size:clamp(17px,2vw,20px);max-width:44em;margin:0 0 24px}
.grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:28px;align-items:start}
@media (max-width:860px){.grid{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--white);border:2.5px solid var(--ink);box-shadow:8px 8px 0 var(--accent,var(--yellow));padding:20px 22px;margin:0 0 24px}
.card.accent-orange{--accent:var(--orange)}.card.accent-magenta{--accent:var(--magenta)}.card.accent-cyan{--accent:var(--cyan)}
.card-title{font:400 22px/1.1 var(--font-display);font-synthesis:none;text-transform:uppercase;margin:0 0 12px}
:lang(zh-Hant) .card-title,:lang(zh-Hans) .card-title,:lang(ja) .card-title,:lang(ko) .card-title{font-weight:900;font-family:"Noto Sans TC","Anton",sans-serif}
ol.how{margin:0;padding-left:1.4em}ol.how li{margin:0 0 6px}
.stats{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;font-size:14px;color:var(--muted);margin:14px 0 0}
.btn{display:inline-flex;align-items:center;justify-content:center;min-height:48px;padding:10px 20px;border:2px solid var(--ink);
background:var(--white);color:var(--ink);font:700 17px/1.2 var(--font-body);text-decoration:none;cursor:pointer;
box-shadow:3px 3px 0 var(--ink);border-radius:0}
.btn:hover{background:var(--yellow)}
.btn:active{transform:translate(3px,3px);box-shadow:none}
.btn.primary{background:var(--orange);box-shadow:4px 4px 0 var(--magenta);font-size:19px;min-height:56px;padding:12px 26px}
.btn.primary:active{transform:translate(4px,4px);box-shadow:none}
.btn[disabled]{opacity:.55;cursor:not-allowed}
.btn.small{min-height:44px;font-size:15px;padding:8px 14px}
.ts-box{min-height:68px;margin:0 0 14px}
.hint{font-size:14px;color:var(--muted);margin:10px 0 0}
.alert{border:2.5px solid var(--ink);background:var(--white);border-left:8px solid var(--magenta);box-shadow:4px 4px 0 var(--magenta);
padding:12px 16px;margin:0 0 24px;font-weight:700}
.place{font:400 clamp(56px,12vw,120px)/1 var(--font-display);font-synthesis:none;margin:6px 0 14px;color:var(--ink);text-shadow:6px 6px 0 var(--yellow)}
:lang(zh-Hant) .place,:lang(zh-Hans) .place,:lang(ja) .place,:lang(ko) .place{font-weight:900;font-family:"Noto Sans TC","Anton",sans-serif;font-size:clamp(40px,9vw,88px)}
.spinner{width:56px;height:56px;border:6px solid var(--ink);border-right-color:var(--orange);margin:8px 0 18px;animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.small-print{font-size:14px;color:var(--ink);margin:8px 0 0}
.ticker{background:var(--yellow);border-top:3px solid var(--ink);border-bottom:3px solid var(--ink);transform:rotate(-1.2deg);
margin:20px -20px 34px;padding:10px 24px;font:400 20px/1.2 var(--font-display);font-synthesis:none;text-transform:uppercase;text-align:center}
:lang(zh-Hant) .ticker,:lang(zh-Hans) .ticker,:lang(ja) .ticker,:lang(ko) .ticker{font-weight:900;font-family:"Noto Sans TC","Anton",sans-serif}
footer{background:var(--ink);color:#c9c9d1;padding:18px clamp(14px,4vw,40px);font-size:13px}
.offline{color:var(--magenta);font-weight:700;min-height:1.5em;margin:10px 0 0}
@media (prefers-reduced-motion:reduce){.spinner{animation:none}.ticker{transform:none}}
`;

function langBar(lang: Lang, path: string): string {
	const items = LANGS.map(
		(l) =>
			`<li><a href="${esc(path)}?lang=${l}" lang="${l}" hreflang="${l}"${l === lang ? ' aria-current="true"' : ""}>${esc(LANG_NAMES[l])}</a></li>`,
	).join("");
	return `<nav aria-label="${esc(t(lang, "page.lang"))}"><ul class="langs">${items}</ul></nav>`;
}

interface Shell {
	lang: Lang;
	nonce: string;
	body: string;
	path: string;
	head?: string;
	script?: string;
}

function shell(s: Shell): string {
	return `<!doctype html>
<html lang="${s.lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>${esc(t(s.lang, "page.title"))}</title>
<link rel="stylesheet" href="/_edge/fonts/fonts.css">
<style nonce="${s.nonce}">${CSS}</style>
${s.head ?? ""}
</head>
<body>
<header class="topbar"><a class="brand" href="/_edge/?lang=${s.lang}">${esc(t(s.lang, "page.brand"))}</a>${langBar(s.lang, s.path)}</header>
<main>${s.body}</main>
<footer>${esc(t(s.lang, "land.attr"))} <a href="${LICENSE_URL}" rel="license noopener">${esc(t(s.lang, "land.license"))}</a></footer>
${s.script ? `<script nonce="${s.nonce}">${s.script}</script>` : ""}
</body>
</html>`;
}

export type LandingError = "turnstile" | "already" | "rate" | "busy" | "start";

export interface LandingInput {
	lang: Lang;
	nonce: string;
	minutes: number;
	stats: { live: number; queued: number; max: number };
	sitekey: string;
	scriptUrl: string;
	error?: LandingError;
	/** Shown above the form after a session ended (the visitor came back to "/"). */
	ended?: SessionView;
}

export function landingPage(p: LandingInput): string {
	const L = p.lang;
	const alert = p.error ? `<p class="alert" role="alert">${esc(t(L, `err.${p.error}`))}</p>` : "";
	const ended = p.ended ? endedCard(L, p.ended.reason) : "";
	const body = `
<div class="wrap">
${alert}${ended}
<p class="eyebrow">${esc(t(L, "land.eyebrow"))}</p>
<h1 class="display">${esc(t(L, "land.h1a"))}<span class="outline">${esc(t(L, "land.h1b", { minutes: p.minutes }))}</span></h1>
<p class="lead">${esc(t(L, "land.lead", { minutes: p.minutes }))}</p>
<div class="grid">
<section class="card accent-orange" aria-labelledby="startTitle">
<h2 class="card-title" id="startTitle">${esc(t(L, "land.start"))}</h2>
<form method="post" action="/_edge/start?lang=${L}" id="startForm">
<div class="ts-box" id="turnstile" data-sitekey="${esc(p.sitekey)}"></div>
<button class="btn primary" type="submit" id="startBtn" disabled>${esc(t(L, "land.start"))}</button>
<p class="hint" id="verifyHint">${esc(t(L, "land.verify"))}</p>
</form>
<p class="stats" id="stats">${esc(t(L, "land.stats", p.stats))}</p>
</section>
<section class="card accent-cyan">
<ol class="how"><li>${esc(t(L, "land.how1"))}</li><li>${esc(t(L, "land.how2"))}</li><li>${esc(t(L, "land.how3"))}</li></ol>
</section>
</div>
<div class="ticker" aria-hidden="true">${esc(t(L, "land.nc.title"))} · CC BY-NC-SA 4.0 · Pollen Robotics Microduck</div>
<section class="card accent-magenta" aria-labelledby="ncTitle">
<h2 class="card-title" id="ncTitle">${esc(t(L, "land.nc.title"))}</h2>
<p>${esc(t(L, "land.nc.text"))}</p>
<p>${esc(t(L, "land.attr"))} <a href="${LICENSE_URL}" rel="license noopener">${esc(t(L, "land.license"))}</a></p>
<p class="small-print">${esc(t(L, "land.privacy"))}</p>
</section>
</div>`;
	// Explicit render: the button unlocks only with a token; the token goes with the form. The onload
	// callback is defined before the widget script loads and waits for the page if it comes early.
	const ready = `
window.dmTurnstileReady=function(){function go(){var box=document.getElementById("turnstile"),btn=document.getElementById("startBtn"),hint=document.getElementById("verifyHint");
window.turnstile.render(box,{sitekey:box.dataset.sitekey,action:"start",language:${JSON.stringify(TURNSTILE_LANG[L])},
callback:function(){btn.disabled=false;hint.hidden=true},"expired-callback":function(){btn.disabled=true;hint.hidden=false},
"error-callback":function(){btn.disabled=true;hint.hidden=false}});}
if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",go);else go();};`;
	const script = `document.getElementById("startForm").addEventListener("submit",function(){document.getElementById("startBtn").disabled=true;});`;
	const head = `<script nonce="${p.nonce}">${ready}</script>
<script nonce="${p.nonce}" src="${esc(withOnload(p.scriptUrl))}" async defer></script>`;
	return shell({ lang: L, nonce: p.nonce, body, path: "/", head, script });
}

function withOnload(url: string): string {
	return url + (url.includes("?") ? "&" : "?") + "onload=dmTurnstileReady";
}

function endedCard(lang: Lang, reason: EndReason | undefined): string {
	const extra: Partial<Record<EndReason, string>> = {
		crash: "gone.crash",
		start_failed: "gone.start",
		start_timeout: "gone.start",
		idle: "gone.idle",
		lost_place: "gone.place",
		queue_idle: "gone.place",
		left: "gone.left",
	};
	const key = reason ? extra[reason] : undefined;
	return `<section class="card accent-yellow" aria-labelledby="goneTitle">
<h2 class="card-title" id="goneTitle">${esc(t(lang, "gone.title"))}</h2>
<p>${esc(t(lang, key ?? "gone.note"))}</p></section>`;
}

export interface WaitingInput {
	lang: Lang;
	nonce: string;
	minutes: number;
	view: SessionView;
}

/** Line or waking up; the page asks /_edge/status every 2 s and opens the duck when it is ready. */
export function waitingPage(p: WaitingInput): string {
	const L = p.lang;
	const queued = p.view.phase === "queued";
	const place = p.view.place ?? 1;
	const words = {
		place: t(L, "queue.place", { place: "{place}" }),
		offline: t(L, "wait.offline"),
	};
	const body = `
<div class="wrap">
<section class="card ${queued ? "accent-orange" : "accent-cyan"}" aria-live="polite">
<p class="eyebrow">${esc(t(L, "land.eyebrow"))}</p>
<h1 class="card-title" id="waitTitle">${esc(t(L, queued ? "queue.title" : "start.title"))}</h1>
${queued ? `<p class="place" id="place">${esc(t(L, "queue.place", { place }))}</p>` : `<div class="spinner" role="img" aria-label="${esc(t(L, "start.title"))}"></div>`}
<p>${esc(t(L, queued ? "queue.note" : "start.note", { minutes: p.minutes }))}</p>
<p class="offline" id="offline"></p>
${queued ? `<form method="post" action="/_edge/leave?lang=${L}"><button class="btn small" type="submit">${esc(t(L, "queue.leave"))}</button></form>` : ""}
</section>
</div>`;
	const script = `
(function(){var W=${JSON.stringify(words)},phase=${JSON.stringify(p.view.phase)};
function tick(){fetch("/_edge/status",{cache:"no-store",credentials:"same-origin"}).then(function(r){return r.json()}).then(function(v){
document.getElementById("offline").textContent="";
if(v.phase!==phase){location.reload();return}
var el=document.getElementById("place");if(el&&v.place){el.textContent=W.place.replace("{place}",v.place)}
}).catch(function(){document.getElementById("offline").textContent=W.offline}).finally(function(){setTimeout(tick,2000)})}
setTimeout(tick,2000);})();`;
	return shell({ lang: L, nonce: p.nonce, body, path: "/", script });
}
