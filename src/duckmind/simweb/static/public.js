/* 公開試玩 on the simulator page (python -m duckmind.simweb --public; duckmind/simweb/public.py): one visitor,
 * one fresh duck, a set time. It reads
 *   GET /api/ping             public: true only on a public server (anything else: this module does nothing)
 *   GET /api/public/session   {seed, name, started_at, ends_at, left_s, ended, export_until, export_left_s, export_minutes,
 *                             refuses: [codes], repo}   (again every SYNC_MS; the countdown runs between)
 * and then:
 *   - body.public-play; #publicBar (the time left, the non-commercial notice, what is not open, Pollen's 3D
 *     model and its licence, the repository) and the ticker's #chipPublic;
 *   - every control of a refused feature greyed out and inert (no script can switch it back on), the one plain
 *     reason (page.public.refused) as its title and, for a card, as a visible line inside it (touch has no hover);
 *   - when the time is up: body.public-ended, the play controls greyed out with page.public.ended_grace, and the end
 *     card with the duck file's download until export_until; after that, the thanks.
 *   - when the visit is over (the download time is about to end, or the duck answers 404/502 for its session): the
 *     page goes quiet (no poll, no stream, nothing new asked: see quieter) and, once the front says the session is
 *     gone (GET /_edge/status → phase "gone"), opens "/", the front's own page. A page left open logs no errors.
 *     Without the front (no /_edge/status) it stays quiet on the thanks.
 * Words: page.public.* (catalog keys; the server says the same in web.public.*). */
"use strict";

(function publicModule() {
  const REFUSED = "page.public.refused";
  const ENDED = "page.public.ended_grace";
  const SYNC_MS = 15000;
  // A refused feature: the server's code, the controls (a card gets its note inside it).
  const GATES = [
    { code: "duck_picker", selectors: ["#neuroPicker", "#neuroPickJump"] },
    { code: "clock", selectors: ["#neuroDay"] },
    { code: "duckfile_import", selectors: ["#duckFileImport", "#duckFileSwitch"] },
    { code: "lab", selectors: ['a[data-view-link="data"]'] },
    { code: "remote", selectors: ["#cmdRemoteSlot"] },
    { code: "voice_commands", selectors: ["#cmdMic"] },
    { code: "body_link", selectors: ["#bodyLink"] },
    { code: "live_voice", selectors: ["#liveVoice"] },
    { code: "layout", selectors: ["#homeEditor"] },
  ];
  // What a visitor plays with: held off once the time is up.
  const PLAY = ["#cmdStrip", "#neuroStim", "#emotionFold", "#trickBook", "#neuroSoundMount"];
  const CARDS = new Set(["neuroPicker", "neuroDay", "bodyLink", "neuroStim", "trickBook"]);
  // A held control whose reason is said on a line right after it (a folded <details> would hide a line inside).
  const AFTER = new Set(["duckFileSwitch"]);
  // Go quiet this long before the download time ends: no poll may reach a duck that is already gone.
  const LEAVE_EARLY_S = 2;
  // The duck's answer for a session that is over (the front: 404 once the session is gone, 502 a dead duck).
  const GONE = new Set([404, 502]);
  const STATUS_EVERY_MS = 1000;
  const STATUS_TRIES = 60;

  /* m:ss of a number of seconds (never below 0). */
  function clock(seconds) {
    const s = Math.max(0, Math.ceil(Number(seconds) || 0));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  /* Hold `el` off: inert (also against scripts that enable their own buttons), greyed (a card: its contents, not
     the line that says why), the reason as its title. */
  function hold(el, key, t, card, params) {
    if (!el.dataset.publicGate) el.dataset.publicTitle = el.getAttribute("title") || "";
    el.dataset.publicGate = key;
    el.inert = true;
    el.classList.add(card ? "public-gated-card" : "public-gated");
    el.setAttribute("aria-disabled", "true");
    el.setAttribute("title", t(key, params));
  }

  /* Wrap win.fetch and win.EventSource (before any other script takes them) so the page can go quiet at the end.
     Until armed it passes everything through unchanged (the normal page never arms it). Armed, an answer in GONE
     calls onGone. Closed, a new fetch never settles (its caller just waits; nothing is sent) and every stream is
     closed and stays closed. */
  function quieter(win) {
    const q = { armed: false, closed: false, onGone: null, fetch: null };
    if (typeof win.fetch !== "function") return q;
    const nativeFetch = win.fetch.bind(win);
    q.fetch = nativeFetch;
    win.fetch = function fetch(input, init) {
      if (q.closed) return new Promise(() => {});
      const answer = nativeFetch(input, init);
      if (q.armed) {
        answer.then(
          (response) => {
            if (GONE.has(response.status) && q.onGone) q.onGone();
          },
          () => {},
        );
      }
      return answer;
    };
    const NativeSource = win.EventSource;
    const sources = new Set();
    if (typeof NativeSource === "function") {
      win.EventSource = class EventSource extends NativeSource {
        constructor(url, config) {
          // Closed: a stream that never connects (a derived constructor may hand back its own object).
          if (q.closed) return { url: String(url), readyState: 2, close() {}, addEventListener() {}, removeEventListener() {} };
          super(url, config);
          sources.add(this);
        }
      };
    }
    q.close = () => {
      q.closed = true;
      for (const source of sources) source.close();
      sources.clear();
    };
    return q;
  }

  const api = { REFUSED, ENDED, GATES, PLAY, LEAVE_EARLY_S, clock, quieter };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
    return;
  }

  /* ---------------- in the page ---------------- */
  const I18N = () => window.I18N;
  const t = (key, params) => (I18N() ? I18N().t(key, params) : key);
  const $ = (selector) => document.querySelector(selector);
  const st = { session: null, at: 0, ended: false, timer: 0, sync: 0, leaving: false, checking: false };
  // Before app.js and the other scripts take fetch and EventSource.
  const quiet = quieter(window);

  // One visible line with the reason inside a held card, or right after a held control of AFTER (once).
  function note(el, key, after, params) {
    let line = after ? el.nextElementSibling : el.querySelector(":scope > [data-public-note]");
    if (!line || line.dataset.publicNote === undefined) {
      line = document.createElement("p");
      line.className = "hint real-note";
      line.dataset.publicNote = "";
      if (after) el.after(line);
      else el.appendChild(line);
    }
    if (line.dataset.key !== key && I18N()) {
      line.dataset.key = key;
      I18N().text(line, key, params);
    }
  }

  function gate(selectors, key, params) {
    for (const el of document.querySelectorAll(selectors.join(", "))) {
      const card = CARDS.has(el.id);
      hold(el, key, t, card, params);
      if (card || AFTER.has(el.id)) note(el, key, !card, params);
    }
  }

  function refuse(codes) {
    for (const g of GATES) if (codes.has(g.code)) gate(g.selectors, REFUSED);
  }

  function left() {
    const s = st.session;
    if (!s) return { play: 0, grace: 0 };
    const gone = (performance.now() - st.at) / 1000;
    return { play: Math.max(0, s.left_s - gone), grace: Math.max(0, s.export_left_s - gone) };
  }

  function paint() {
    const s = st.session;
    const i18n = I18N();
    if (!s || !i18n) return;
    const now = left();
    const ended = s.ended || now.play <= 0;
    const chip = $("#chipPublic");
    const timer = $("#publicLeft");
    for (const el of [chip, timer]) {
      if (!el) continue;
      if (ended) i18n.text(el, "page.public.chip_ended");
      else i18n.text(el, "page.public.chip_left", { time: clock(now.play) });
    }
    if (ended && !st.ended) {
      st.ended = true;
      document.body.classList.add("public-ended");
      gate(PLAY, ENDED, { minutes: s.export_minutes || 3 });
    }
    const end = $("#publicEnd");
    if (end) {
      if (end.hidden === ended) end.hidden = !ended;
      const text = $("#publicEndText");
      const download = $("#publicDownload");
      if (ended && text) {
        if (now.grace > 0) i18n.text(text, "page.public.end_text", { name: s.name, time: clock(now.grace) });
        else i18n.text(text, "page.public.end_gone", { name: s.name });
      }
      if (download && download.hidden !== (now.grace <= 0)) download.hidden = now.grace <= 0;
    }
    if (ended && now.grace <= LEAVE_EARLY_S) leave();
  }

  /* The front's word on this session: "gone", "on", or null without a front. Its status answers 200 either way. */
  async function frontSays() {
    try {
      const response = await quiet.fetch("/_edge/status", { cache: "no-store", credentials: "same-origin" });
      const type = response.headers.get("content-type") || "";
      if (!response.ok || !type.includes("json")) return null;
      const view = await response.json();
      return view && view.phase === "gone" ? "gone" : "on";
    } catch {
      return null;
    }
  }

  /* An answer said the session may be over: ask the front once, leave if it is. */
  async function check() {
    if (st.leaving || st.checking) return;
    st.checking = true;
    try {
      if ((await frontSays()) === "gone") leave();
    } finally {
      st.checking = false;
    }
  }

  /* The visit is over: quiet now, then the front's page once the front has let the duck go. */
  async function leave() {
    if (st.leaving) return;
    st.leaving = true;
    quiet.close();
    clearInterval(st.timer);
    clearInterval(st.sync);
    for (let i = 0; i < STATUS_TRIES; i++) {
      const said = await frontSays();
      if (said === null) return; // no front: stay on the thanks, quiet
      if (said === "gone") break;
      await new Promise((resolve) => setTimeout(resolve, STATUS_EVERY_MS));
    }
    window.location.replace("/");
  }

  async function sync() {
    try {
      const response = await fetch("/api/public/session", { cache: "no-store" });
      if (!response.ok) {
        if (GONE.has(response.status)) check();
        return;
      }
      const session = await response.json();
      if (!session || !session.ok) return;
      st.session = session;
      st.at = performance.now();
      refuse(new Set(Array.isArray(session.refuses) ? session.refuses.map(String) : []));
      paint();
    } catch {
      /* (the server went away at the end: the countdown carries on from the last answer) */
    }
  }

  async function start() {
    let ping = null;
    try {
      const response = await fetch("/api/ping", { cache: "no-store" });
      ping = response.ok ? await response.json() : null;
    } catch {
      ping = null;
    }
    if (!ping || ping.public !== true) return;
    quiet.armed = true;
    quiet.onGone = check;
    document.body.classList.add("public-play");
    for (const el of document.querySelectorAll("[data-public]")) el.hidden = false;
    await sync();
    st.timer = setInterval(paint, 1000);
    st.sync = setInterval(sync, SYNC_MS);
    if (I18N()) I18N().onChange(() => paint());
  }

  window.duckPublic = {
    ...api,
    get on() {
      return !!st.session;
    },
    get ended() {
      return st.ended;
    },
  };
  (I18N() && I18N().ready ? I18N().ready : Promise.resolve()).then(start, start);
})();
