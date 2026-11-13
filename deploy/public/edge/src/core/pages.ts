/**
 * The edge's own pages: landing (start a duck), waiting (line / waking up), the guiding page (no duck
 * for a newcomer now) and "time is up" (the landing page after a visit). Every page is the README hero
 * banner as a page (docs/media/hero/src/template.html, the same markup and CSS approach): the top bar,
 * the DUCKMIND mark, the left column (the page's own words and buttons), on the right the simulator in
 * a window frame with the rendered duck in front of it, the tilted yellow ticker and the small print.
 * On a desktop the banner scales with the window (`--s` is one banner pixel); below 900 px it stacks.
 * Look follows docs/simulator-style.md (tokens, square corners, thick ink borders, hard offset
 * shadows, Anton / Noto 900 display type). Fonts are the simulator's own (/_edge/fonts/), the duck and
 * the screenshot are the banner's, made smaller (/_edge/img/, deploy/public/edge/static/img).
 */

import { LANG_NAMES, LANGS, type Lang, t } from "./i18n.ts";
import type { EndReason, SessionView } from "./session.ts";

/** Turnstile widget language codes. */
const TURNSTILE_LANG: Record<Lang, string> = { "zh-Hant": "zh-tw", en: "en", ja: "ja", es: "es", "zh-Hans": "zh-cn", ko: "ko" };

/** The other CJK languages' Noto subsets (the simulator's, in /_edge/fonts/); Noto Sans TC is in fonts.css. */
const CJK_FONT: Partial<Record<Lang, { family: string; css: string }>> = {
	ja: { family: "Noto Sans JP", css: "noto-sans-jp.css" },
	"zh-Hans": { family: "Noto Sans SC", css: "noto-sans-sc.css" },
	ko: { family: "Noto Sans KR", css: "noto-sans-kr.css" },
};

export const LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/";
export const REPO_URL = "https://github.com/joseyarashio/duckmind";
/** The public repository's README quick start (where to raise a duck at home). */
export const QUICK_START_URL = `${REPO_URL}#quick-start`;

/** The duck in the banner's screenshot: a name the family gave, never translated. */
const SCREENSHOT_DUCK = "蛋黃";

export function esc(value: unknown): string {
	return String(value)
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&#39;");
}

// `u(n)` is n banner pixels: calc(n * var(--s)).
const CSS = `
:root{--ink:#101018;--ink-deep:#08080c;--ink-2:#1c1c28;--paper:#faf8f2;--cream:#f2ecdd;--white:#fff;
--yellow:#ffd23f;--orange:#ff7a2f;--orange-ink:#c2500a;--magenta:#ff2fa8;--cyan:#2ff0e6;--green:#1f9d57;--muted:#5e5e68;
--cjk:"Noto Sans TC";--font-display:"Anton",var(--cjk),sans-serif;--font-body:"DM Sans",var(--cjk),system-ui,sans-serif;
--font-mono:"JetBrains Mono",var(--cjk),ui-monospace,monospace;
--s:min(calc(100vw / 1200),calc(100svh / 740),1.4px)}
:root:lang(ja){--cjk:"Noto Sans JP","Noto Sans TC"}
:root:lang(zh-Hans){--cjk:"Noto Sans SC","Noto Sans TC"}
:root:lang(ko){--cjk:"Noto Sans KR","Noto Sans TC"}
*{box-sizing:border-box;margin:0;padding:0}
html{background:var(--ink-deep)}
body{min-height:100svh;color:var(--white);font:400 u(14)/1.6 var(--font-body);font-synthesis:none;overflow-x:clip;
background:linear-gradient(180deg,var(--ink) 0%,#07070a 100%)}
a{color:inherit}
:focus-visible{outline:3px solid var(--orange);outline-offset:2px}
.display{font-family:var(--font-display)}
.mono{font-family:var(--font-mono)}
.stage{position:relative;min-height:100svh;display:flex;flex-direction:column;overflow:clip}
.backdrop{position:absolute;inset:0;pointer-events:none;
background-image:radial-gradient(circle at 80% 40%,rgba(255,122,47,.32) 0%,rgba(255,47,168,.16) 35%,transparent 65%),
radial-gradient(circle at 18% 18%,rgba(47,240,230,.15) 0%,transparent 48%)}
.halftone{position:absolute;inset:0;pointer-events:none;background-image:radial-gradient(rgba(255,255,255,.085) 1.5px,transparent 1.6px);
background-size:16px 16px;mask-image:radial-gradient(ellipse at center,#000 45%,transparent 82%);-webkit-mask-image:radial-gradient(ellipse at center,#000 45%,transparent 82%)}
.in{position:relative;width:u(1200);max-width:100%;margin:0 auto;padding:0 u(44)}
.topbar{position:relative;z-index:10;background:rgba(8,8,12,.85);border-bottom:1px solid rgba(255,255,255,.12)}
.topbar .in{min-height:u(36);display:flex;align-items:center;justify-content:space-between;gap:u(16);
font-size:u(11.5);letter-spacing:.04em;color:rgba(255,255,255,.75);white-space:nowrap}
.partner-left{display:flex;align-items:center;gap:u(10);font-weight:700;color:var(--white);min-width:0}
.partner-left span:last-child{overflow:hidden;text-overflow:ellipsis}
.partner-badge{background:var(--orange);color:var(--ink-deep);font-weight:700;font-size:u(10);padding:u(2) u(7);letter-spacing:.06em}
.partner-right{font-size:u(11);color:rgba(255,255,255,.55)}
.nav{position:relative;z-index:10}
.nav .in{min-height:u(56);display:flex;align-items:center;justify-content:space-between;gap:u(16)}
.logo-group{display:flex;align-items:center;gap:u(12);text-decoration:none}
.logo-mark{width:u(48);height:u(38);display:block}
.brand-name{font-size:u(28);letter-spacing:.03em;text-transform:uppercase;color:var(--white);line-height:1}
.brand-chip{font-size:u(11);font-weight:700;color:var(--yellow);background:rgba(255,210,63,.15);border:1px solid rgba(255,210,63,.4);
padding:u(2) u(8);letter-spacing:.08em}
.langs{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:u(6);list-style:none}
.langs a{display:inline-flex;align-items:center;min-height:u(30);padding:u(3) u(9);border:2px solid rgba(255,255,255,.85);color:var(--white);
text-decoration:none;font-size:u(13);font-weight:700;line-height:1.2}
.langs a:not(:lang(en)):not(:lang(es)),:root:not(:lang(zh-Hant)) .duck-name{font-family:system-ui,sans-serif}
.langs a:hover{background:var(--white);color:var(--ink)}
.langs a[aria-current="true"]{background:var(--yellow);color:var(--ink);border-color:var(--yellow)}
.hero{position:relative;z-index:10;display:grid;grid-template-columns:u(500) minmax(0,1fr);gap:u(28);padding-top:u(8);padding-bottom:u(8);
align-items:center;min-height:u(440)}
.left-col{display:flex;flex-direction:column;justify-content:center;min-width:0}
.kicker-tag{font-size:u(12);font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--orange);margin-bottom:u(4)}
.hero-headline{font-size:u(84);font-weight:400;line-height:.9;letter-spacing:.015em;text-transform:uppercase;color:var(--yellow);
text-shadow:.05em .05em 0 var(--ink-deep),-.025em 0 0 var(--cyan),.025em 0 0 var(--magenta);margin:u(4) 0 u(8)}
.page-title{font-size:u(58);font-weight:900;line-height:1.02;text-transform:uppercase;color:var(--yellow);overflow-wrap:anywhere;
text-shadow:.05em .05em 0 var(--ink-deep),-.025em 0 0 var(--cyan),.025em 0 0 var(--magenta);margin:u(4) 0 u(14)}
.hero-subheadline,.place{font-size:u(28);font-weight:900;line-height:1.1;letter-spacing:.04em;text-transform:uppercase;color:var(--white);
-webkit-text-stroke:1.2px var(--magenta);text-shadow:2px 2px 0 var(--ink-deep);margin-bottom:u(14)}
.place{font-size:u(44);font-variant-numeric:tabular-nums}
.hero-desc{font-size:u(14);line-height:1.6;color:rgba(255,255,255,.82);max-width:u(480);margin-bottom:u(18);text-wrap:pretty}
.hero-desc strong{color:var(--white);font-weight:700}
.cta-group{display:flex;flex-wrap:wrap;align-items:center;gap:u(16)}
.btn-primary,.btn-secondary{display:inline-flex;align-items:center;justify-content:center;border:3px solid var(--ink-deep);border-radius:0;
color:var(--ink-deep);font-family:var(--font-display);font-weight:900;letter-spacing:.04em;text-transform:uppercase;line-height:1.2;
white-space:nowrap;text-decoration:none;cursor:pointer;gap:.4em}
.ico{width:.9em;height:.9em;fill:currentColor;flex:none;vertical-align:-.1em}
.btn-primary{background:var(--orange);box-shadow:6px 6px 0 var(--magenta);font-size:u(18);padding:u(12) u(22);transform:rotate(-.5deg)}
.btn-secondary{background:var(--paper);box-shadow:5px 5px 0 var(--yellow);font-size:u(16);padding:u(12) u(18);transform:rotate(.8deg)}
.btn-primary:not([disabled]):hover,.btn-secondary:hover{background:var(--yellow)}
.btn-primary:not([disabled]):active{transform:translate(6px,6px);box-shadow:none}
.btn-secondary:active{transform:translate(5px,5px);box-shadow:none}
.btn-primary[disabled]{opacity:.55;cursor:not-allowed}
.btn-secondary.small{font-size:u(14);padding:u(8) u(14)}
.hint,.stats,.offline{font-size:u(12.5);color:rgba(255,255,255,.7);margin-top:u(14)}
.hint.warn{color:var(--yellow);font-weight:700}
.ts-box.shown{margin:0 0 u(14)}
.stats{font-family:var(--font-mono);font-variant-numeric:tabular-nums;font-size:u(12);color:rgba(255,255,255,.6);margin-top:u(8)}
.offline{color:var(--magenta);font-weight:700;min-height:1.5em;margin-top:0}
.alert,.notice{background:var(--paper);color:var(--ink);border:3px solid var(--ink-deep);box-shadow:6px 6px 0 var(--magenta);
padding:u(10) u(14);margin:0 0 u(18);max-width:u(480);font-size:u(14);font-weight:700}
.notice{font-weight:400}
.notice.cyan{box-shadow:6px 6px 0 var(--cyan)}
.notice-title{font-family:var(--font-display);font-weight:900;font-size:u(20);line-height:1.15;text-transform:uppercase;margin-bottom:u(4)}
.spinner{width:u(52);height:u(52);border:u(6) solid var(--white);border-right-color:var(--orange);margin:u(6) 0 u(18);animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.right-col{position:relative;height:u(440);display:flex;align-items:center;justify-content:flex-start}
.screen-frame{position:relative;width:u(495);height:u(338);background:var(--cream);border:4px solid var(--ink-deep);
box-shadow:10px 10px 0 var(--cyan),10px 10px 0 3px var(--ink-deep);transform:rotate(.6deg);overflow:hidden;display:flex;flex-direction:column;z-index:5}
.frame-header{height:u(32);flex:none;background:var(--yellow);border-bottom:3px solid var(--ink-deep);display:flex;align-items:center;
justify-content:space-between;gap:u(8);padding:0 u(12);font-size:u(12.5);font-weight:900;letter-spacing:.05em;color:var(--ink-deep);
text-transform:uppercase;white-space:nowrap}
.frame-header .dots{display:flex;gap:u(6)}
.frame-header .dots i{width:u(9);height:u(9);border:1.5px solid var(--ink-deep);background:var(--magenta)}
.frame-header .dots i+i{background:var(--paper)}.frame-header .dots i+i+i{background:var(--green)}
.frame-body{flex:1;position:relative;background:#000;overflow:hidden}
.frame-img{width:100%;height:100%;object-fit:cover;object-position:center 30%;display:block}
.comic-bubble{position:absolute;bottom:u(10);left:u(10);max-width:u(330);background:var(--paper);border:3px solid var(--ink-deep);
box-shadow:4px 4px 0 var(--magenta);padding:u(7) u(11);display:flex;flex-direction:column;gap:u(3);z-index:10}
.comic-badge-row{display:flex;align-items:center;gap:u(6)}
.comic-badge{background:var(--magenta);color:var(--white);font-size:u(11);font-weight:900;letter-spacing:.05em;padding:u(1) u(6);
border:1.5px solid var(--ink-deep);text-transform:uppercase}
.comic-tag-sub{font-size:u(10);font-weight:600;color:var(--muted)}
.comic-text{font-size:u(11.5);color:var(--ink-deep);font-weight:700;line-height:1.35}
.comic-text span{color:var(--orange-ink);white-space:nowrap}
.duck-render{position:absolute;right:u(6);bottom:u(10);height:u(350);z-index:15;transform:rotate(-2deg);
filter:drop-shadow(0 15px 25px rgba(0,0,0,.75)) drop-shadow(4px 4px 0 rgba(8,8,12,.5))}
.duck-render img{height:100%;width:auto;display:block}
.robot-badge{position:absolute;right:u(28);top:u(14);background:var(--cyan);border:3px solid var(--ink-deep);box-shadow:5px 5px 0 var(--ink-deep);
padding:u(5) u(12);font-size:u(13);font-weight:900;letter-spacing:.06em;color:var(--ink-deep);text-transform:uppercase;transform:rotate(4deg);
z-index:16;white-space:nowrap}
.ticker{position:relative;z-index:20;margin:u(12) u(-40) 0;height:u(38);background:var(--yellow);border-top:3px solid var(--ink-deep);
border-bottom:3px solid var(--ink-deep);box-shadow:0 4px 18px rgba(0,0,0,.45);transform:rotate(-1.3deg);overflow:hidden;
display:flex;align-items:center;font-family:var(--font-display);font-size:u(13);font-weight:900;letter-spacing:.06em;text-transform:uppercase;
color:var(--ink-deep);white-space:nowrap}
.ticker-track{display:flex;width:100%;padding:0 u(60)}
.tk-set{flex:1;min-width:0;display:flex;align-items:center;justify-content:space-around;gap:u(18)}
.tk-copy{display:none}
.marquee-item{display:inline-flex;align-items:center;gap:u(8)}
.star-bullet{display:inline-flex;color:var(--magenta);font-size:u(16)}
.robot-badge .ico{margin:0 .3em}
.small-print{position:relative;z-index:21;display:flex;flex-direction:column;align-items:flex-end;text-align:right;gap:u(2);
padding-top:u(14);padding-bottom:u(14);font-size:max(12px,u(10.5));line-height:1.45;color:rgba(255,255,255,.6);letter-spacing:.02em}
.small-print p{max-width:u(1000)}
.small-print .nc{color:var(--white);font-weight:700}
@media (max-width:899px){
:root{--s:1px}
.in{padding:0 16px}
.topbar .in{flex-wrap:wrap;gap:4px 12px;padding-top:7px;padding-bottom:7px;white-space:normal}
.partner-left{align-items:flex-start}
.nav .in{flex-wrap:wrap;padding-top:12px;padding-bottom:4px}
.langs{justify-content:flex-start}
.langs a{min-height:44px;font-size:14px;padding:4px 10px}
.hero{grid-template-columns:minmax(0,1fr);gap:34px;min-height:0;padding-top:14px;padding-bottom:6px}
.hero-headline{font-size:min(84px,21vw)}
.page-title{font-size:min(48px,12vw)}
.hero-subheadline{font-size:24px}
.place{font-size:36px}
.hero-desc{font-size:16px}
.cta-group{flex-direction:column;align-items:stretch;gap:18px}
.btn-primary,.btn-secondary{white-space:normal;text-align:center;min-height:56px;font-size:18px}
.hint,.stats{font-size:14px}
.alert,.notice{font-size:15px}
.right-col{height:auto;padding:0 0 18px;justify-content:flex-start}
.screen-frame{width:86%;height:auto;aspect-ratio:495/338}
.frame-header{font-size:11px;height:28px}
.frame-header span:nth-child(2){overflow:hidden;text-overflow:ellipsis}
.comic-bubble{display:none}
.duck-render{height:min(250px,64vw);right:-6px;bottom:0}
.robot-badge{top:-14px;right:0;font-size:12px}
.ticker{margin:18px -24px 0;height:40px;font-size:14px}
.ticker-track{width:max-content;padding:0;animation:tick 30s linear infinite}
.ticker:hover .ticker-track{animation-play-state:paused}
.tk-set{flex:none;justify-content:flex-start;gap:24px;padding-right:24px}
.tk-copy{display:flex}
.small-print{align-items:flex-start;text-align:left;font-size:13px;padding-top:22px;padding-bottom:22px}
}
@keyframes tick{to{transform:translateX(-50%)}}
@media (prefers-reduced-motion:reduce){.spinner,.ticker-track{animation:none}
.ticker,.screen-frame,.duck-render,.robot-badge,.btn-primary,.btn-secondary{transform:none}}
`.replace(/u\((-?[\d.]+)\)/g, "calc($1 * var(--s))");

/** The tape is one line on a desktop: in long languages its words get a little smaller until they fit (as the banner's). */
const FIT_TICKER = `(function(){var set=document.querySelector(".tk-set"),wide=matchMedia("(min-width:900px)");
function fit(){set.parentNode.parentNode.style.fontSize="";if(!wide.matches)return;
var size=parseFloat(getComputedStyle(set).fontSize),min=size*.65;
while(set.scrollWidth>set.clientWidth&&size>min){size-=.25;set.parentNode.parentNode.style.fontSize=size+"px"}}
document.fonts.ready.then(fit);addEventListener("resize",fit);})();`;

const LOGO = `<svg class="logo-mark" viewBox="0 0 40 32" aria-hidden="true"><rect x="1" y="3" width="30" height="26" rx="13" fill="#ffd23f" stroke="#101018" stroke-width="2.5"/><circle cx="20" cy="14" r="5.5" fill="#ff7a2f" stroke="#101018" stroke-width="2"/><circle cx="20" cy="14" r="2" fill="#101018"/><path d="M29 17h9l-9 6z" fill="#ff7a2f" stroke="#101018" stroke-width="2" stroke-linejoin="round"/></svg>`;
// ★ and → drawn, not typed: no font has to be fetched for them (Anton and DM Sans have neither; Noto
// Sans TC would cost a Latin page two 75 KB slices).
const STAR = `<svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 1l3.4 7.2 7.6.9-5.6 5.3 1.5 7.6L12 18.2 5.1 22l1.5-7.6L1 9.1l7.6-.9z"/></svg>`;
const ARROW = `<svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M2 9.5h13.2L10 4.3 13.2 1 24 12 13.2 23 10 19.7l5.2-5.2H2z"/></svg>`;

function langBar(lang: Lang, path: string): string {
	const items = LANGS.map(
		(l) =>
			`<li><a href="${esc(path)}?lang=${l}" lang="${l}" hreflang="${l}"${l === lang ? ' aria-current="true"' : ""}>${esc(LANG_NAMES[l])}</a></li>`,
	).join("");
	return `<nav aria-label="${esc(t(lang, "page.lang"))}"><ul class="langs">${items}</ul></nav>`;
}

/** The simulator in its window with the duck in front of it (decoration: the banner's picture). */
function showcase(lang: Lang, minutes: number): string {
	return `<div class="right-col" aria-hidden="true">
<div class="screen-frame">
<div class="frame-header display"><span class="dots"><i></i><i></i><i></i></span><span>${esc(t(lang, "land.frame"))}</span><span class="mono">${minutes}:00</span></div>
<div class="frame-body"><img class="frame-img" src="/_edge/img/screenshot-600.webp" srcset="/_edge/img/screenshot-600.webp 600w, /_edge/img/screenshot-1200.webp 1200w"
sizes="(max-width: 899px) 86vw, calc(495 * min(100vw / 1200, 100svh / 740, 1.4px))" width="600" height="390" alt="" decoding="async">
<div class="comic-bubble"><div class="comic-badge-row"><span class="comic-badge display">${esc(t(lang, "land.badge"))}</span><span class="comic-tag-sub mono">SYSTEM 1 &amp; 2</span></div>
<span class="comic-text">${esc(t(lang, "land.comic"))} <span>(Zzz • <span class="duck-name" lang="zh-Hant">${SCREENSHOT_DUCK}</span>)</span></span></div></div>
</div>
<div class="duck-render"><img src="/_edge/img/duck-420.webp" srcset="/_edge/img/duck-420.webp 1x, /_edge/img/duck-840.webp 2x" width="228" height="420" alt="" decoding="async"></div>
<div class="robot-badge display">${STAR}${esc(t(lang, "land.sticker"))}${STAR}</div>
</div>`;
}

function ticker(lang: Lang, minutes: number): string {
	const items = [t(lang, "tick.reflex"), t(lang, "tick.seed"), t(lang, "tick.growth"), t(lang, "tick.play", { minutes }), t(lang, "land.nc.title"), "GITHUB: JOSEYARASHIO/DUCKMIND"]
		.map((s) => `<span class="marquee-item"><span class="star-bullet">${STAR}</span><span>${esc(s)}</span></span>`)
		.join("");
	return `<div class="ticker" aria-hidden="true"><div class="ticker-track"><div class="tk-set">${items}</div><div class="tk-set tk-copy">${items}</div></div></div>`;
}

interface Shell {
	lang: Lang;
	nonce: string;
	minutes: number;
	/** The left column: the page's own words and buttons. */
	left: string;
	path: string;
	head?: string;
	script?: string;
}

function shell(s: Shell): string {
	const L = s.lang;
	const cjk = CJK_FONT[L];
	return `<!doctype html>
<html lang="${L}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>${esc(t(L, "page.title"))}</title>
<link rel="stylesheet" href="/_edge/fonts/fonts.css">
${cjk ? `<link rel="stylesheet" href="/_edge/fonts/${cjk.css}">\n` : ""}<style nonce="${s.nonce}">${CSS}</style>
${s.head ?? ""}
</head>
<body>
<div class="stage">
<div class="backdrop"></div><div class="halftone"></div>
<div class="topbar"><div class="in"><div class="partner-left"><span class="partner-badge">PET ROBOTICS</span><span>${esc(t(L, "land.unofficial"))}</span></div>
<div class="partner-right mono">${esc(t(L, "land.stack"))}</div></div></div>
<header class="nav"><div class="in"><a class="logo-group" href="/_edge/?lang=${L}">${LOGO}<span class="brand-name display">duckmind</span><span class="brand-chip">PET BRAIN</span></a>
${langBar(L, s.path)}</div></header>
<main class="in hero">
<div class="left-col">${s.left}</div>
${showcase(L, s.minutes)}
</main>
${ticker(L, s.minutes)}
<footer class="in small-print">
<p>${esc(t(L, "land.credit"))} <a href="${LICENSE_URL}" rel="license noopener">${esc(t(L, "land.license"))}</a></p>
<p>${esc(t(L, "land.footnote"))}</p>
<p><span class="nc">${esc(t(L, "land.nc.title"))}</span> · ${esc(t(L, "land.nc.text"))}</p>
<p>${esc(t(L, "land.privacy"))}</p>
</footer>
</div>
<script nonce="${s.nonce}">${FIT_TICKER}${s.script ?? ""}</script>
</body>
</html>`;
}

export type LandingError = "turnstile" | "already" | "rate" | "start";

export interface LandingInput {
	lang: Lang;
	nonce: string;
	minutes: number;
	stats: { live: number; queued: number; max: number };
	sitekey: string;
	scriptUrl: string;
	error?: LandingError;
	/** The visitor came back to "/" after a session ended: the page says time is up (and why). */
	ended?: SessionView;
}

export function landingPage(p: LandingInput): string {
	const L = p.lang;
	const alert = p.error ? `<p class="alert" role="alert">${esc(t(L, `err.${p.error}`))}</p>` : "";
	const title = p.ended
		? `<h1 class="page-title display" id="goneTitle">${esc(t(L, "gone.title"))}</h1>
<p class="notice">${esc(t(L, endedKey(p.ended.reason)))}</p>`
		: `<h1 class="hero-headline display">DUCKMIND</h1>
<p class="hero-subheadline display">${esc(t(L, "land.tagline"))}</p>`;
	const turn = p.stats.max === 1 ? t(L, "land.turn.one") : t(L, "land.turn.many", { max: p.stats.max });
	const left = `${alert}
<p class="kicker-tag">${esc(t(L, "land.eyebrow"))} • ${esc(t(L, "land.kicker"))}</p>
${title}
<p class="hero-desc"><strong>${esc(t(L, "land.lead1", { minutes: p.minutes }))}</strong> ${esc(turn)} ${esc(t(L, "land.lead2"))}</p>
<form method="post" action="/_edge/start?lang=${L}" id="startForm">
<div class="ts-box" id="turnstile" data-sitekey="${esc(p.sitekey)}"></div>
<div class="cta-group">
<button class="btn-primary" type="submit" id="startBtn" disabled>${esc(t(L, p.ended ? "gone.again" : "land.start"))}${ARROW}</button>
<a class="btn-secondary" href="${REPO_URL}" rel="noopener">${esc(t(L, "land.repo"))}${STAR}</a>
</div>
<p class="hint" id="verifyHint" role="status">${esc(t(L, "land.checking"))}</p>
</form>
<p class="stats" id="stats">${esc(t(L, "land.stats", p.stats))}</p>`;
	// Explicit render: the button unlocks only with a token; the token goes with the form. The onload
	// callback is defined before the widget script loads and waits for the page if it comes early. The
	// widget first shows itself only when the visitor has to do something (appearance
	// "interaction-only"). If the check fails, expires, times out or brings no token within 8 s, the
	// widget is drawn again visibly (Cloudflare's own box and message, its retries) with one plain line
	// under the buttons, so nobody waits on a disabled button with nothing to do.
	const words = { retry: t(L, "land.check.retry"), browser: t(L, "land.check.browser") };
	const ready = `
window.dmTurnstileReady=function(){function go(){var box=document.getElementById("turnstile"),btn=document.getElementById("startBtn"),hint=document.getElementById("verifyHint"),
W=${JSON.stringify(words)},id=null,shown=false,timer=null;
function opts(appearance,retry){return{sitekey:box.dataset.sitekey,action:"start",language:${JSON.stringify(TURNSTILE_LANG[L])},appearance:appearance,retry:retry,
callback:function(){clearTimeout(timer);btn.disabled=false;hint.hidden=true},
"expired-callback":function(){say(W.retry)},"timeout-callback":function(){say(W.retry)},
"error-callback":function(code){say(/^1105/.test(String(code))?W.browser:W.retry)}}}
function say(text){clearTimeout(timer);btn.disabled=true;hint.textContent=text;hint.hidden=false;hint.classList.add("warn");
if(shown)return;shown=true;try{if(id!==null)window.turnstile.remove(id)}catch(e){}
box.classList.add("shown");id=window.turnstile.render(box,opts("always","auto"))}
id=window.turnstile.render(box,opts("interaction-only","never"));timer=setTimeout(function(){say(W.retry)},8000);}
if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",go);else go();};`;
	const script = `document.getElementById("startForm").addEventListener("submit",function(){document.getElementById("startBtn").disabled=true;});`;
	const head = `<script nonce="${p.nonce}">${ready}</script>
<script nonce="${p.nonce}" src="${esc(withOnload(p.scriptUrl))}" async defer></script>`;
	return shell({ lang: L, nonce: p.nonce, minutes: p.minutes, left, path: "/", head, script });
}

function withOnload(url: string): string {
	return url + (url.includes("?") ? "&" : "?") + "onload=dmTurnstileReady";
}

function endedKey(reason: EndReason | undefined): string {
	const extra: Partial<Record<EndReason, string>> = {
		crash: "gone.crash",
		start_failed: "gone.start",
		start_timeout: "gone.start",
		idle: "gone.idle",
		lost_place: "gone.place",
		queue_idle: "gone.place",
		left: "gone.left",
	};
	return (reason && extra[reason]) || "gone.note";
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
	const left = `<section aria-live="polite" aria-labelledby="waitTitle">
<p class="kicker-tag">${esc(t(L, "land.eyebrow"))}</p>
<h1 class="page-title display" id="waitTitle">${esc(t(L, queued ? "queue.title" : "start.title"))}</h1>
${queued ? `<p class="place display" id="place">${esc(t(L, "queue.place", { place }))}</p>` : `<div class="spinner" role="img" aria-label="${esc(t(L, "start.title"))}"></div>`}
<p class="hero-desc">${esc(t(L, queued ? "queue.note" : "start.note", { minutes: p.minutes }))}</p>
<p class="offline" id="offline"></p>
${queued ? `<form method="post" action="/_edge/leave?lang=${L}"><button class="btn-secondary small" type="submit">${esc(t(L, "queue.leave"))}</button></form>` : ""}
</section>`;
	const script = `
(function(){var W=${JSON.stringify(words)},phase=${JSON.stringify(p.view.phase)};
function tick(){fetch("/_edge/status",{cache:"no-store",credentials:"same-origin"}).then(function(r){return r.json()}).then(function(v){
document.getElementById("offline").textContent="";
if(v.phase!==phase){location.reload();return}
var el=document.getElementById("place");if(el&&v.place){el.textContent=W.place.replace("{place}",v.place)}
}).catch(function(){document.getElementById("offline").textContent=W.offline}).finally(function(){setTimeout(tick,2000)})}
setTimeout(tick,2000);})();`;
	return shell({ lang: L, nonce: p.nonce, minutes: p.minutes, left, path: "/", script });
}

/** No duck for a newcomer now (the line is full, or the owner closed public play): no error, a way on. */
export function guidingPage(p: { lang: Lang; nonce: string; minutes: number }): string {
	const L = p.lang;
	const left = `<p class="kicker-tag">${esc(t(L, "land.eyebrow"))}</p>
<h1 class="page-title display">${esc(t(L, "guide.title"))}</h1>
<p class="hero-desc">${esc(t(L, "guide.later"))}</p>
<section class="notice cyan" aria-labelledby="homeTitle">
<h2 class="notice-title" id="homeTitle">${esc(t(L, "guide.home.title"))}</h2>
<p>${esc(t(L, "guide.home.text"))}</p>
</section>
<div class="cta-group">
<a class="btn-primary" id="quickStart" href="${QUICK_START_URL}" rel="noopener">${esc(t(L, "guide.home.link"))}${ARROW}</a>
<a class="btn-secondary" id="lookAgain" href="/_edge/?lang=${L}">${esc(t(L, "guide.again"))}</a>
</div>`;
	return shell({ lang: L, nonce: p.nonce, minutes: p.minutes, left, path: "/" });
}
