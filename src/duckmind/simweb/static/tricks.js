/* 學會聽話 on the simulator page (phase 2; docs/design/neural-duck/LEARN-TRICKS-2026-10-25.md):
 *
 * - the three feedback buttons in the command strip (稱讚 / 點心 / 不對, `[data-feedback]`):
 *   `POST /api/neuro/feedback {kind}` — the duck's own brain credits the trick it just did (told, or by itself
 *   within three seconds) and answers where that trick now stands; the line under the buttons says so, or why
 *   nothing was credited (no trick lately, this performance already had its word, today's cap);
 * - the 訓練 tab (`#ctl-trick`): the trick book from `GET /api/life/growth` (`growth.learned.tricks`): per trick its
 *   proficiency bar, today's tries, its state, today's cap and the one to practise next, and a button that sends
 *   the trick's command through the command strip's own path (`window.duckCommand`, neuro.js); the card has the same
 *   three feedback buttons (`#trickFeedback`), because a phone puts the command strip away while the panel is open.
 *
 * The book is fetched when the tab opens, every 15 s while it is open (like the growth record), and again after a
 * feedback. The server keeps its growth picture for up to 8 s, so the row the feedback named is updated at once
 * from the feedback's own answer and the book is fetched again once that picture has expired; a read that is still
 * older than the answer (fewer counted tries) does not take the answer back (`mergeRows`).
 *
 * Words come from the catalogs (`page.trick.*`, the verbs `page.cmd.verb.*`). `createTricks(env)` takes everything
 * it touches, so tests run it in node over fakes; the last lines start it on the real page. */

export const POLL_MS = 15000; // the book is read this often while the tab is open (the growth record's pace)
export const STALE_MS = 9000; // the server's growth picture lives up to 8 s: read again after a feedback this late
export const TRICK_IDS = ["lie_down", "spin", "bow", "shake", "flip"];
const JSON_HEADERS = { "Content-Type": "application/json" };

const KIND_KEYS = { praise: "page.trick.fb.praise", treat: "page.trick.fb.treat", scold: "page.trick.fb.scold" };
const STATE_KEYS = {
  none: "page.trick.state.none",
  half: "page.trick.state.learning",
  learned: "page.trick.state.learned",
  locked: "page.trick.state.locked",
};
const verbKey = (trick) => `page.cmd.verb.${trick}`;

/* What one press of a feedback button came to, as a catalog key and its values (the kind and the trick are
   catalog keys too, so a language switch says it again in the new words). `status` is the HTTP status, `data`
   the answer (null when nothing came back). */
export function feedbackAnswer(kind, status, data) {
  const base = { kindKey: KIND_KEYS[kind] || KIND_KEYS.praise, trickKey: null, pct: null };
  if (!data || typeof data !== "object") return { ...base, key: "page.trick.fb.failed", tone: "error" };
  if (status !== 200 || data.ok !== true) {
    const key = { no_trick: "page.trick.fb.no_trick", not_running: "page.trick.fb.not_running" }[data.reason] || "page.trick.fb.failed";
    return { ...base, key, tone: "error" };
  }
  const trickKey = TRICK_IDS.includes(data.trick) ? verbKey(data.trick) : null;
  const pct = Number.isFinite(data.proficiency) ? Math.round(100 * data.proficiency) : null;
  const out = { ...base, trickKey, pct };
  if (data.reason === "already_taken") return { ...out, key: "page.trick.fb.already_taken", tone: "none" };
  if (data.reason === "not_just_done") return { ...out, key: "page.trick.fb.not_just_done", tone: "none" };
  if (data.reason) return { ...out, key: "page.trick.fb.failed", tone: "error" };
  if (kind === "scold") return { ...out, key: "page.trick.fb.scolded", tone: "down" };
  if (data.capped) return { ...out, key: "page.trick.fb.capped", tone: "none" };
  if (data.shaped) return { ...out, key: "page.trick.fb.shaped", tone: "up" };
  return { ...out, key: "page.trick.fb.praised", tone: "up" };
}

/* The trick book's rows as the page draws them, from the growth view (`growth.learned.tricks`; ids and numbers
   only, never words). null: there is no book to show. */
export function bookRows(tricks) {
  if (!Array.isArray(tricks)) return null;
  return tricks
    .filter((row) => row && TRICK_IDS.includes(row.id))
    .map((row) => {
      const pct = Math.max(0, Math.min(100, Math.round(Number(row.proficiency_pct) || 0)));
      const state = STATE_KEYS[row.state] ? row.state : "none";
      return {
        id: row.id,
        pct,
        state,
        today: Math.max(0, Number(row.practised_today) || 0),
        total: Math.max(0, Number(row.total) || 0),
        capped: !!row.capped_today,
        suggest: !!row.suggest,
        open: row.unlocked !== false && state !== "locked",
      };
    });
}

/* A feedback's answer on the row it named, until the book is read again: the new proficiency, a counted try,
   today's cap. (A counted try means the trick has been practised: 沒學過 becomes 學了一半 — one day can add at
   most 12 points, never enough to reach 學會了 from nothing.) */
export function patchRow(row, data) {
  if (!row || !data || data.ok !== true || data.trick !== row.id) return row;
  const next = { ...row };
  if (Number.isFinite(data.proficiency)) next.pct = Math.max(0, Math.min(100, Math.round(100 * data.proficiency)));
  if (data.counted) {
    next.today += 1;
    next.total += 1;
    if (next.state === "none") next.state = "half";
  }
  if (data.capped) next.capped = true;
  return next;
}

/* A book read just after a feedback can still be an older picture than the answer the page already shows (the
   server keeps its picture a few seconds). A row's `total` only grows, so a read row with fewer counted tries than
   the answer's row is older: the shown row stays, for HOLD_MS at most (a switched duck starts a new book).
   `held`: trick → {total, at} for the rows a feedback's answer moved. */
export const HOLD_MS = 30000;
export function mergeRows(read, shown, held, now) {
  const keep = {};
  if (!read) return { rows: read, held: keep };
  const rows = read.map((row) => {
    const h = held[row.id];
    if (!h || now - h.at > HOLD_MS || row.total >= h.total) return row;
    const mine = shown ? shown.find((r) => r.id === row.id) : null;
    if (!mine) return row;
    keep[row.id] = h;
    return { ...mine, suggest: row.suggest, open: row.open };
  });
  return { rows, held: keep };
}

export function createTricks(env) {
  const { document, fetchFn, i18n, timers } = env;
  const now = env.now || (() => Date.now());
  const $ = (id) => document.getElementById(id);
  const T = (key, params) => (i18n && typeof i18n.t === "function" ? i18n.t(key, params) : key);
  const els = {
    strip: $("cmdFeedback"),
    out: $("cmdFeedbackOut"),
    panel: $("ctl-trick"),
    book: $("trickBook"),
    list: $("trickList"),
    status: $("trickStatus"),
    note: $("trickNote"),
    cardFeedback: $("trickFeedback"), // (the same three buttons in the card: a phone puts the strip away while the panel is open)
  };
  const groups = [els.strip, els.cardFeedback].filter(Boolean);
  const buttons = groups.flatMap((group) => Array.from(group.querySelectorAll("[data-feedback]")));
  const s = { rows: null, held: {}, phase: "loading", sending: false, answer: null, timer: 0, poll: 0 };

  // ── the feedback buttons ──────────────────────────────────────────────────────────────────────

  function answerText(a) {
    return T(a.key, { kind: T(a.kindKey), trick: a.trickKey ? T(a.trickKey) : "", pct: a.pct == null ? 0 : a.pct });
  }

  function paintAnswer() {
    const a = s.answer;
    if (!a) return;
    for (const el of [els.out, els.note]) {
      if (!el) continue;
      el.hidden = false;
      el.dataset.state = a.tone;
      delete el.dataset.i18n; // (the first paint's hint is not said again on a language switch)
      el.textContent = answerText(a);
    }
  }

  async function sendFeedback(kind) {
    if (s.sending || !KIND_KEYS[kind]) return;
    s.sending = true;
    for (const b of buttons) b.disabled = true;
    if (els.out) {
      els.out.hidden = false;
      els.out.dataset.state = "sending";
      delete els.out.dataset.i18n;
      els.out.textContent = T("page.trick.fb.sending");
    }
    let status = 0;
    let data = null;
    try {
      const response = await fetchFn("/api/neuro/feedback", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify({ kind }) });
      status = response.status;
      data = await response.json();
    } catch (error) {
      data = null;
    }
    s.answer = feedbackAnswer(kind, status, data);
    paintAnswer();
    if (status === 200 && data && data.ok === true && s.rows) {
      s.rows = s.rows.map((row) => patchRow(row, data));
      const row = data.counted ? s.rows.find((r) => r.id === data.trick) : null;
      if (row) s.held[row.id] = { total: row.total, at: now() };
      paintBook();
    }
    // the server's picture may still be the one from before the word: read the book again once it has expired
    if (status === 200 && data && data.ok === true) later(STALE_MS);
    s.sending = false;
    for (const b of buttons) b.disabled = false;
  }

  for (const group of groups) {
    group.addEventListener("click", (event) => {
      const button = event.target && event.target.closest ? event.target.closest("[data-feedback]") : null;
      if (!button || button.disabled) return;
      event.preventDefault();
      button.classList.add("sent");
      timers.setTimeout(() => button.classList.remove("sent"), 700);
      sendFeedback(button.dataset.feedback);
    });
  }

  // ── the trick book ────────────────────────────────────────────────────────────────────────────

  function h(tag, attrs = {}, children = []) {
    const el = document.createElement(tag);
    for (const [name, value] of Object.entries(attrs)) {
      if (value == null || value === false) continue;
      if (name === "class") el.className = value;
      else if (name === "text") el.textContent = value;
      else if (name.startsWith("data-")) el.dataset[name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = String(value);
      else el.setAttribute(name, value === true ? "" : String(value));
    }
    for (const child of children) if (child) el.append(child);
    return el;
  }

  function rowEl(row) {
    const name = T(verbKey(row.id));
    const fill = h("i");
    fill.style.width = `${row.pct}%`;
    const bar = h(
      "div",
      {
        class: "bar trick-bar",
        role: "progressbar",
        "aria-valuemin": 0,
        "aria-valuemax": 100,
        "aria-valuenow": row.pct,
        "aria-valuetext": T("page.trick.pct", { pct: row.pct }),
        "aria-label": T("page.trick.bar.aria", { trick: name }),
      },
      [fill],
    );
    const today = [h("span", { text: T("page.trick.today", { today: row.today, total: row.total }) })];
    if (row.capped) today.push(h("span", { class: "trick-capped", text: T("page.trick.capped") }));
    if (!row.open) today.push(h("span", { class: "trick-locked", text: T("page.trick.locked_note") }));
    const button = h("button", {
      type: "button",
      class: "btn small trick-do",
      "data-trick-cmd": row.id,
      "aria-label": T("page.trick.do.aria", { trick: name }),
      disabled: !row.open,
      text: T("page.trick.do"),
    });
    return h("li", { class: "trick-row", "data-trick": row.id, "data-state": row.state, "data-suggest": row.suggest ? "true" : null }, [
      h("div", { class: "trick-head" }, [
        h("strong", { class: "trick-name", text: name }),
        h("span", { class: "chip trick-state", text: T(STATE_KEYS[row.state]) }),
        row.suggest ? h("span", { class: "trick-suggest", text: T("page.trick.suggest") }) : null,
      ]),
      h("div", { class: "trick-meter" }, [bar, h("span", { class: "trick-pct", text: T("page.trick.pct", { pct: row.pct }) })]),
      h("p", { class: "trick-today" }, today),
      button,
    ]);
  }

  function paintBook() {
    if (!els.list || !els.status) return;
    const status = { loading: "page.trick.loading", off: "page.trick.off", no_book: "page.trick.no_book", offline: "page.trick.offline" }[s.phase];
    // (offline with a book already drawn: the book stays, the line says it could not be read just now)
    els.status.hidden = !status;
    if (status) {
      delete els.status.dataset.i18n;
      els.status.textContent = T(status);
    }
    if (s.rows && s.phase !== "off" && s.phase !== "no_book") els.list.replaceChildren(...s.rows.map(rowEl));
    else els.list.replaceChildren();
  }

  function open() {
    return !!els.panel && !els.panel.hidden && !document.hidden;
  }

  async function refresh() {
    if (!els.list) return;
    let view;
    try {
      const response = await fetchFn("/api/life/growth", { cache: "no-store" });
      const data = await response.json();
      view = data && data.growth ? data.growth : null;
      if (!view) {
        s.rows = null;
        s.phase = "off";
      } else {
        const merged = mergeRows(bookRows(view.learned && view.learned.tricks), s.rows, s.held, now());
        s.rows = merged.rows;
        s.held = merged.held;
        s.phase = s.rows ? "ok" : "no_book";
      }
    } catch (error) {
      s.phase = "offline";
    }
    paintBook();
  }

  function later(ms) {
    if (s.timer) timers.clearTimeout(s.timer);
    s.timer = timers.setTimeout(() => {
      s.timer = 0;
      refresh();
    }, ms);
  }

  if (els.list) {
    els.list.addEventListener("click", (event) => {
      const button = event.target && event.target.closest ? event.target.closest("[data-trick-cmd]") : null;
      if (!button || button.disabled) return;
      button.classList.add("sent");
      timers.setTimeout(() => button.classList.remove("sent"), 700);
      if (typeof env.sendCommand === "function") env.sendCommand(button.dataset.trickCmd);
    });
  }

  // open the tab: read the book at once; while it is open, every POLL_MS
  let wasOpen = false;
  function onVisibility() {
    const now = open();
    if (now && !wasOpen) refresh();
    wasOpen = now;
  }
  if (els.panel && env.MutationObserverCtor) {
    new env.MutationObserverCtor(onVisibility).observe(els.panel, { attributes: true, attributeFilter: ["hidden"] });
  }
  document.addEventListener("visibilitychange", onVisibility);
  s.poll = timers.setInterval(() => {
    if (open()) refresh();
  }, POLL_MS);
  if (i18n && typeof i18n.onChange === "function")
    i18n.onChange(() => {
      paintBook();
      paintAnswer();
    });
  onVisibility();

  return {
    refresh,
    sendFeedback,
    get rows() {
      return s.rows;
    },
    get phase() {
      return s.phase;
    },
  };
}

if (typeof document !== "undefined" && typeof window !== "undefined" && document.getElementById("cmdFeedback")) {
  // (the words first: the catalogs load before this file says anything)
  const start = () => {
    window.duckTricks = createTricks({
      document,
      fetchFn: (...args) => window.fetch(...args),
      i18n: window.I18N,
      MutationObserverCtor: window.MutationObserver,
      sendCommand: (verb) => (window.duckCommand ? window.duckCommand(verb) : undefined),
      timers: {
        setTimeout: (fn, ms) => window.setTimeout(fn, ms),
        clearTimeout: (id) => window.clearTimeout(id),
        setInterval: (fn, ms) => window.setInterval(fn, ms),
        clearInterval: (id) => window.clearInterval(id),
      },
    });
  };
  (window.I18N ? window.I18N.ready : Promise.resolve()).then(start, start);
}
