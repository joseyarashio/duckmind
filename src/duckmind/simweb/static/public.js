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

  const api = { REFUSED, ENDED, GATES, PLAY, clock };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
    return;
  }

  /* ---------------- in the page ---------------- */
  const I18N = () => window.I18N;
  const t = (key, params) => (I18N() ? I18N().t(key, params) : key);
  const $ = (selector) => document.querySelector(selector);
  const st = { session: null, at: 0, ended: false, timer: 0 };

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
  }

  async function sync() {
    try {
      const response = await fetch("/api/public/session", { cache: "no-store" });
      if (!response.ok) return;
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
    document.body.classList.add("public-play");
    for (const el of document.querySelectorAll("[data-public]")) el.hidden = false;
    await sync();
    st.timer = setInterval(paint, 1000);
    setInterval(sync, SYNC_MS);
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
