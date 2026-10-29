// Runs static/tricks.js in Node over a fake page: a fake clock, a fake fetch, a tiny fake DOM.
//   node tricks_page_node_harness.mjs <static dir> <scenario>
// Prints one JSON value (tests/test_tricks_page.py asserts on it).
import { pathToFileURL } from "node:url";
import path from "node:path";

const [staticDir, scenario, input] = process.argv.slice(2);
const mod = await import(pathToFileURL(path.join(staticDir, "tricks.js")).href);

// ── a fake page ──────────────────────────────────────────────────────────────────────────────────────────

const camel = (name) => name.replace(/-([a-z])/g, (_, c) => c.toUpperCase());

class El {
  constructor(tag = "div", id = "") {
    this.tagName = tag.toUpperCase();
    this.id = id;
    this.dataset = {};
    this.attrs = {};
    this.style = {};
    this.children = [];
    this.parent = null;
    this.hidden = false;
    this.disabled = false;
    this.className = "";
    this._text = "";
    this.listeners = {};
    const self = this;
    this.classList = {
      add: (c) => (self.className = [...new Set([...self.className.split(" ").filter(Boolean), c])].join(" ")),
      remove: (c) => (self.className = self.className.split(" ").filter((x) => x && x !== c).join(" ")),
      contains: (c) => self.className.split(" ").includes(c),
    };
  }
  get textContent() {
    return this._text + this.children.map((c) => c.textContent).join("");
  }
  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }
  setAttribute(name, value) {
    this.attrs[name] = String(value);
    if (name === "disabled") this.disabled = true;
  }
  getAttribute(name) {
    return name in this.attrs ? this.attrs[name] : null;
  }
  append(...kids) {
    for (const kid of kids) {
      kid.parent = this;
      this.children.push(kid);
    }
  }
  replaceChildren(...kids) {
    this.children = [];
    this._text = "";
    this.append(...kids);
  }
  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }
  matches(selector) {
    const m = /^\[data-([\w-]+)\]$/.exec(selector);
    return !!m && camel(m[1]) in this.dataset;
  }
  closest(selector) {
    for (let el = this; el; el = el.parent) if (el.matches(selector)) return el;
    return null;
  }
  all() {
    return this.children.flatMap((c) => [c, ...c.all()]);
  }
  querySelectorAll(selector) {
    return this.all().filter((el) => el.matches(selector));
  }
  // a click bubbles from this element up to every listener on the way
  click() {
    const event = { type: "click", target: this, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; } };
    for (let el = this; el; el = el.parent) for (const fn of el.listeners.click || []) fn(event);
    return event;
  }
}

function makePage({ growth, feedback = [] } = {}) {
  let time = 1_000_000;
  const timers = [];
  let nextId = 1;
  const clock = {
    async advance(ms) {
      const end = time + ms;
      for (;;) {
        const due = timers.filter((t) => t.at <= end).sort((a, b) => a.at - b.at)[0];
        if (!due) break;
        time = due.at;
        if (due.every) due.at += due.every;
        else timers.splice(timers.indexOf(due), 1);
        due.fn();
        await settle();
      }
      time = end;
    },
  };
  const fakeTimers = {
    setTimeout: (fn, ms) => (timers.push({ id: nextId, at: time + ms, fn }), nextId++),
    setInterval: (fn, ms) => (timers.push({ id: nextId, at: time + ms, every: ms, fn }), nextId++),
    clearTimeout: (id) => {
      const i = timers.findIndex((t) => t.id === id);
      if (i >= 0) timers.splice(i, 1);
    },
  };
  fakeTimers.clearInterval = fakeTimers.clearTimeout;

  const strip = new El("div", "cmdFeedback");
  const buttons = ["praise", "treat", "scold"].map((kind) => {
    const b = new El("button");
    b.dataset.feedback = kind;
    strip.append(b);
    return b;
  });
  const card = new El("div", "trickFeedback");
  const cardButtons = ["praise", "treat", "scold"].map((kind) => {
    const b = new El("button");
    b.dataset.feedback = kind;
    card.append(b);
    return b;
  });
  buttons.push(...cardButtons);
  const out = new El("p", "cmdFeedbackOut");
  out.dataset.i18n = "page.trick.fb.hint";
  strip.append(out);
  const els = {
    cmdFeedback: strip,
    cmdFeedbackOut: out,
    "ctl-trick": Object.assign(new El("div", "ctl-trick"), { hidden: true }),
    trickBook: new El("section", "trickBook"),
    trickList: new El("ol", "trickList"),
    trickStatus: new El("p", "trickStatus"),
    trickNote: Object.assign(new El("p", "trickNote"), { hidden: true }),
    trickFeedback: card,
  };
  const docListeners = {};
  const doc = {
    hidden: false,
    getElementById: (id) => els[id] || null,
    createElement: (tag) => new El(tag),
    addEventListener: (type, fn) => (docListeners[type] = docListeners[type] || []).push(fn),
  };
  const observers = [];
  class FakeObserver {
    constructor(fn) {
      this.fn = fn;
      observers.push(this);
    }
    observe(target) {
      this.target = target;
    }
  }
  const log = { gets: 0, posts: [], commands: [] };
  const answers = [...feedback];
  const fetchFn = async (url, options = {}) => {
    if (url === "/api/life/growth") {
      log.gets += 1;
      const g = typeof growth === "function" ? growth(log.gets) : growth;
      if (g instanceof Error) throw g;
      return { status: 200, json: async () => ({ ok: true, growth: g }) };
    }
    if (url === "/api/neuro/feedback") {
      log.posts.push({ ...JSON.parse(options.body), disabledWhileSent: buttons.map((b) => b.disabled) });
      const [status, data] = answers.shift();
      return { status, json: async () => data };
    }
    throw new Error(`unexpected ${url}`);
  };
  const i18n = {
    t: (key, params) => (params ? `${key}${JSON.stringify(params)}` : key),
    onChange: () => {},
  };
  const api = mod.createTricks({
    document: doc,
    fetchFn,
    i18n,
    timers: fakeTimers,
    MutationObserverCtor: FakeObserver,
    now: () => time,
    sendCommand: (verb) => log.commands.push(verb),
  });
  const openTab = async (open = true) => {
    els["ctl-trick"].hidden = !open;
    for (const o of observers) o.fn();
    await settle();
  };
  // a read of the book waiting longer than a button's flash (700 ms): the one after a feedback
  const pendingReads = () => timers.filter((t) => !t.every && t.at - time > 1000).length;
  return { api, els, buttons, log, clock, openTab, doc, pendingReads };
}

const settle = async () => {
  for (let i = 0; i < 10; i++) await new Promise((r) => setImmediate(r));
};

// what one drawn row says, read back from the fake DOM
function rowOf(li) {
  const all = li.all();
  const bar = all.find((el) => el.attrs.role === "progressbar");
  const button = all.find((el) => "trickCmd" in el.dataset);
  const has = (cls) => all.find((el) => el.className.split(" ").includes(cls)) || null;
  return {
    id: li.dataset.trick,
    state: li.dataset.state,
    suggest: li.dataset.suggest === "true",
    name: has("trick-name").textContent,
    chip: has("trick-state").textContent,
    bar: { now: bar.attrs["aria-valuenow"], min: bar.attrs["aria-valuemin"], max: bar.attrs["aria-valuemax"], text: bar.attrs["aria-valuetext"], label: bar.attrs["aria-label"], width: bar.children[0].style.width },
    today: has("trick-today").textContent,
    capped: !!has("trick-capped"),
    locked: !!has("trick-locked"),
    marker: has("trick-suggest") ? has("trick-suggest").textContent : null,
    button: { cmd: button.dataset.trickCmd, disabled: button.disabled, label: button.attrs["aria-label"] },
  };
}

const BOOK = [
  { id: "lie_down", label: "趴下", proficiency_pct: 0, state: "none", practised_today: 0, capped_today: false, total: 0, shaped: 0, unlocked: true, suggest: true },
  { id: "spin", label: "轉圈", proficiency_pct: 42, state: "half", practised_today: 2, capped_today: false, total: 9, shaped: 1, unlocked: true, suggest: false },
  { id: "bow", label: "鞠躬", proficiency_pct: 6, state: "half", practised_today: 1, capped_today: false, total: 1, shaped: 0, unlocked: true, suggest: false },
  { id: "shake", label: "甩一甩", proficiency_pct: 85, state: "learned", practised_today: 4, capped_today: true, total: 30, shaped: 0, unlocked: true, suggest: false },
  { id: "flip", label: "翻跟斗", proficiency_pct: 0, state: "locked", practised_today: 0, capped_today: false, total: 0, shaped: 0, unlocked: false, suggest: false },
];
const view = (tricks = BOOK) => ({ learned: { tricks } });
const rows = (els) => els.trickList.children.map(rowOf);

const scenarios = {
  answers() {
    const ok = (extra) => ({ ok: true, kind: "praise", trick: "bow", proficiency: 0.123, step: 0.06, capped: false, shaped: false, counted: true, reason: null, ...extra });
    const a = (kind, status, data) => {
      const r = mod.feedbackAnswer(kind, status, data);
      return [r.key, r.kindKey, r.trickKey, r.pct, r.tone];
    };
    return {
      praised: a("praise", 200, ok()),
      treat: a("treat", 200, ok({ kind: "treat" })),
      shaped: a("praise", 200, ok({ shaped: true })),
      capped: a("praise", 200, ok({ capped: true })),
      scolded: a("scold", 200, ok({ kind: "scold", counted: false, proficiency: 0.09 })),
      already: a("praise", 200, ok({ counted: false, reason: "already_taken" })),
      notJustDone: a("praise", 200, ok({ counted: false, reason: "not_just_done" })),
      noTrick: a("praise", 400, { ok: false, reason: "no_trick", error: "…" }),
      notRunning: a("scold", 400, { ok: false, reason: "not_running", error: "…" }),
      badKind: a("praise", 400, { ok: false, reason: "bad_kind", error: "…" }),
      nothing: a("treat", 0, null),
      server503: a("praise", 503, { ok: false, error: "…" }),
    };
  },

  rows() {
    return {
      book: mod.bookRows(BOOK),
      none: mod.bookRows(null),
      odd: mod.bookRows([{ id: "dance", proficiency_pct: 50 }, { id: "bow", proficiency_pct: 140, state: "??", practised_today: -1 }]),
      patched: mod.patchRow(mod.bookRows(BOOK)[0], { ok: true, trick: "lie_down", proficiency: 0.06, counted: true, capped: false }),
      patchedScold: mod.patchRow(mod.bookRows(BOOK)[1], { ok: true, trick: "spin", proficiency: 0.39, counted: false }),
      patchedOther: mod.patchRow(mod.bookRows(BOOK)[1], { ok: true, trick: "bow", proficiency: 0.5, counted: true }),
    };
  },

  // a book the server really drew (tests/test_tricks_page.py passes growth_view's rows as JSON)
  real() {
    return mod.bookRows(JSON.parse(input));
  },

  async flow() {
    let book = BOOK;
    const p = makePage({
      growth: () => view(book),
      feedback: [
        [200, { ok: true, kind: "praise", trick: "bow", proficiency: 0.12, step: 0.06, capped: false, shaped: false, counted: true, reason: null }],
        [400, { ok: false, reason: "no_trick", error: "牠剛剛沒有做把戲" }],
      ],
    });
    await settle();
    const r = {};
    r.closedGets = p.log.gets; // the tab is shut: nothing is read
    await p.clock.advance(30000);
    r.closedAfterPoll = p.log.gets;
    await p.openTab(true);
    r.openGets = p.log.gets;
    r.drawn = rows(p.els);
    r.statusHidden = p.els.trickStatus.hidden;
    // a row's button sends that trick's command through the strip's path
    p.els.trickList.children[2].all().find((el) => "trickCmd" in el.dataset).click();
    const locked = p.els.trickList.children[4].all().find((el) => "trickCmd" in el.dataset);
    locked.click();
    r.commands = [...p.log.commands];
    // praise: the answer line and the card's note say it, the named row moves at once
    p.buttons[0].click();
    await settle();
    r.post = p.log.posts[0];
    r.out = { text: p.els.cmdFeedbackOut.textContent, state: p.els.cmdFeedbackOut.dataset.state, keyGone: !("i18n" in p.els.cmdFeedbackOut.dataset) };
    r.note = { text: p.els.trickNote.textContent, hidden: p.els.trickNote.hidden };
    r.bowNow = rows(p.els)[2];
    r.buttonsAfter = p.buttons.map((b) => b.disabled);
    r.getsRightAfter = p.log.gets;
    r.readsWaiting = p.pendingReads();
    // the server's picture expires: the book is read again (and what it says replaces the patch)
    book = BOOK.map((row) => (row.id === "bow" ? { ...row, proficiency_pct: 12, practised_today: 3, total: 7 } : row));
    await p.clock.advance(mod.STALE_MS);
    r.getsAfterStale = p.log.gets;
    r.bowRead = rows(p.els)[2];
    // a word with no trick behind it (the card's 不對 this time): said, nothing moves, the book is not read for it
    p.buttons[5].click();
    await settle();
    r.noTrick = { text: p.els.cmdFeedbackOut.textContent, state: p.els.cmdFeedbackOut.dataset.state, post: p.log.posts[1].kind };
    r.readsWaitingAfterRefusal = p.pendingReads();
    // while open, every POLL_MS; shut again, never
    const open0 = p.log.gets;
    await p.clock.advance(mod.POLL_MS * 2);
    r.pollsOpen = p.log.gets - open0;
    await p.openTab(false);
    const shut0 = p.log.gets;
    await p.clock.advance(mod.POLL_MS * 3);
    r.pollsShut = p.log.gets - shut0;
    return r;
  },

  // the read after a feedback is still the server's older picture, then a newer one; and a duck switched meanwhile
  async staleRead() {
    let book = BOOK;
    const p = makePage({
      growth: () => view(book),
      feedback: [
        [200, { ok: true, kind: "praise", trick: "bow", proficiency: 0.12, step: 0.06, capped: false, shaped: false, counted: true, reason: null }],
        [200, { ok: true, kind: "praise", trick: "spin", proficiency: 0.46, step: 0.04, capped: false, shaped: false, counted: true, reason: null }],
      ],
    });
    await p.openTab(true);
    const bow = () => rows(p.els)[2];
    const spin = () => rows(p.els)[1];
    p.buttons[0].click();
    await settle();
    const r = { patched: [bow().bar.now, bow().today] };
    await p.clock.advance(mod.STALE_MS); // the server still says 6 %, one try: older than the answer
    r.olderRead = [bow().bar.now, bow().today];
    book = BOOK.map((row) => (row.id === "bow" ? { ...row, proficiency_pct: 12, practised_today: 2, total: 2, suggest: true } : { ...row, suggest: false }));
    await p.clock.advance(mod.POLL_MS);
    r.newerRead = [bow().bar.now, bow().today, bow().suggest];
    // a praise for spin, then the duck is switched: a new book with fewer tries; held only for HOLD_MS
    book = BOOK.map((row) => ({ ...row, proficiency_pct: 0, practised_today: 0, total: 0, state: row.id === "flip" ? "locked" : "none" }));
    p.buttons[1].click();
    await settle();
    r.spinPatched = [spin().bar.now, spin().today];
    await p.clock.advance(mod.STALE_MS);
    r.spinHeld = [spin().bar.now, spin().today];
    await p.clock.advance(mod.HOLD_MS);
    r.spinAfterHold = [spin().bar.now, spin().today, spin().state];
    return r;
  },

  async states() {
    const p = makePage({ growth: (n) => [null, { learned: {} }, new Error("down")][n - 1] ?? null });
    await p.openTab(true);
    const off = { text: p.els.trickStatus.textContent, hidden: p.els.trickStatus.hidden, rows: p.els.trickList.children.length };
    await p.api.refresh();
    const noBook = { text: p.els.trickStatus.textContent, rows: p.els.trickList.children.length };
    await p.api.refresh();
    const offline = { text: p.els.trickStatus.textContent, phase: p.api.phase };
    return { off, noBook, offline };
  },
};

const result = await scenarios[scenario]();
process.stdout.write(JSON.stringify(result));
