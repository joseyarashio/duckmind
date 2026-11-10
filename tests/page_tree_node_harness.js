// Runs the simulator page's own scripts in Node over a small fake page built from the page's real markup
// (tests/page_tree_node_harness.py parses static/index.html and writes the tree), with a fake clock, a fake
// fetch and a fake history. One scenario per run; prints one JSON value with what the page did.
//   node page_tree_node_harness.js <static dir> <input json>
// input: {tree: [{tag, attrs, parent} | {text, parent}], scripts: [{name, module}, in the order the page runs them], scenario, data}
"use strict";
const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");

const [staticDir, inputPath] = process.argv.slice(2);
const input = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const realImmediate = setImmediate;
// everything the scripts started has had its turn (promises, observers)
const settled = async () => {
  for (let i = 0; i < 4; i++) await new Promise((resolve) => realImmediate(resolve));
};

const page = { layout: false, time: 1_000_000, answers: {}, frames: [] };
const log = { fetches: [], intervals: [], scrolled: [], created: [], markup: 0, history: [], views: [] };

/* ---------------- a small DOM ---------------- */

const observers = [];
function mutated(el, kind, name) {
  for (const ob of observers) {
    const hit = ob.targets.some(([target, options]) => {
      if (target !== el && !(options.subtree && target.contains(el))) return false;
      if (kind === "attr") return (options.attributes || options.attributeFilter) && (!options.attributeFilter || options.attributeFilter.includes(name));
      return !!(options.childList || options.characterData);
    });
    if (!hit || ob.queued) continue;
    ob.queued = true;
    queueMicrotask(() => {
      ob.queued = false;
      ob.callback([], ob);
    });
  }
}
class MutationObserver {
  constructor(callback) {
    this.callback = callback;
    this.targets = [];
    observers.push(this);
  }
  observe(target, options) {
    this.targets.push([target, options || {}]);
  }
  disconnect() {
    this.targets = [];
  }
}

class Target {
  constructor() {
    this.listeners = {};
  }
  addEventListener(type, fn, options) {
    (this.listeners[type] = this.listeners[type] || []).push({ fn, once: !!(options && options.once) });
  }
  removeEventListener(type, fn) {
    this.listeners[type] = (this.listeners[type] || []).filter((item) => item.fn !== fn);
  }
  hear(event) {
    for (const item of [...(this.listeners[event.type] || [])]) {
      if (item.once) this.removeEventListener(event.type, item.fn);
      item.fn.call(this, event);
    }
  }
}

class Text {
  constructor(text) {
    this.nodeType = 3;
    this.text = String(text);
    this.parentNode = null;
  }
  get textContent() {
    return this.text;
  }
}

const kebab = (name) => name.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);

function compound(text) {
  const out = { tag: "", id: "", classes: [], attrs: [] };
  const part = /([a-zA-Z][\w-]*)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:(\^?=)"([^"]*)")?\]/y;
  let at = 0;
  while (at < text.length) {
    part.lastIndex = at;
    const m = part.exec(text);
    if (!m) throw new Error(`selector not understood: ${text}`);
    if (m[1]) out.tag = m[1].toLowerCase();
    else if (m[2]) out.id = m[2];
    else if (m[3]) out.classes.push(m[3]);
    else out.attrs.push({ name: m[4], op: m[5] || "", value: m[6] });
    at = part.lastIndex;
  }
  return out;
}
function fits(el, c) {
  if (c.tag && el.tagName.toLowerCase() !== c.tag) return false;
  if (c.id && el.id !== c.id) return false;
  if (!c.classes.every((name) => el.classList.contains(name))) return false;
  return c.attrs.every(({ name, op, value }) => {
    const got = el.getAttribute(name);
    if (got === null) return false;
    return op === "=" ? got === value : op === "^=" ? got.startsWith(value) : true;
  });
}

class El extends Target {
  constructor(tag) {
    super();
    this.nodeType = 1;
    this.tagName = String(tag).toUpperCase();
    this.attrs = {};
    this.childNodes = [];
    this.parentNode = null;
    this.props = {};
    const el = this;
    this.classList = {
      list: () => (el.attrs.class || "").split(/\s+/).filter(Boolean),
      contains: (name) => this.classList.list().includes(name),
      add: (name) => this.classList.toggle(name, true),
      remove: (name) => this.classList.toggle(name, false),
      toggle(name, force) {
        const have = this.contains(name);
        const want = force === undefined ? !have : !!force;
        if (want !== have) el.setAttribute("class", want ? [...this.list(), name].join(" ") : this.list().filter((n) => n !== name).join(" "));
        return want;
      },
    };
    this.dataset = new Proxy(
      {},
      {
        get: (_, key) => (typeof key === "string" ? (el.attrs[`data-${kebab(key)}`] ?? undefined) : undefined),
        set: (_, key, value) => (el.setAttribute(`data-${kebab(key)}`, value), true),
        has: (_, key) => `data-${kebab(key)}` in el.attrs,
        deleteProperty: (_, key) => (el.removeAttribute(`data-${kebab(key)}`), true),
      },
    );
    const sheet = {};
    this.style = new Proxy(sheet, {
      get: (_, key) => (key === "setProperty" ? (name, value) => (sheet[name] = value) : sheet[key]),
      set: (_, key, value) => ((sheet[key] = value), true),
    });
  }
  /* attributes */
  setAttribute(name, value) {
    this.attrs[name] = String(value);
    mutated(this, "attr", name);
  }
  getAttribute(name) {
    return name in this.attrs ? this.attrs[name] : null;
  }
  hasAttribute(name) {
    return name in this.attrs;
  }
  removeAttribute(name) {
    if (!(name in this.attrs)) return;
    delete this.attrs[name];
    mutated(this, "attr", name);
  }
  flag(name, on) {
    if (on) this.setAttribute(name, "");
    else this.removeAttribute(name);
  }
  get id() {
    return this.attrs.id || "";
  }
  set id(value) {
    this.setAttribute("id", value);
  }
  get className() {
    return this.attrs.class || "";
  }
  set className(value) {
    this.setAttribute("class", value);
  }
  get hidden() {
    return "hidden" in this.attrs;
  }
  set hidden(on) {
    if (!!on !== this.hidden) this.flag("hidden", on);
  }
  get disabled() {
    return "disabled" in this.attrs;
  }
  set disabled(on) {
    this.flag("disabled", on);
  }
  get open() {
    return "open" in this.attrs;
  }
  set open(on) {
    this.flag("open", on);
  }
  get title() {
    return this.attrs.title || "";
  }
  set title(value) {
    this.setAttribute("title", value);
  }
  get href() {
    return this.attrs.href || "";
  }
  get tabIndex() {
    return Number(this.attrs.tabindex ?? -1);
  }
  set tabIndex(value) {
    this.setAttribute("tabindex", value);
  }
  // a field's value: what was typed or chosen, else what the markup gives (a select: its selected option, else its first)
  get value() {
    if ("value" in this.props) return this.props.value;
    if (this.tagName === "SELECT") {
      const options = this.querySelectorAll("option");
      const chosen = options.find((option) => option.hasAttribute("selected")) || options[0];
      return chosen ? chosen.attrs.value ?? chosen.textContent : "";
    }
    return this.attrs.value ?? "";
  }
  set value(value) {
    this.props.value = String(value);
  }
  /* the tree */
  get children() {
    return this.childNodes.filter((node) => node.nodeType === 1);
  }
  get parentElement() {
    return this.parentNode instanceof El ? this.parentNode : null;
  }
  get isConnected() {
    for (let at = this; at; at = at.parentNode) if (at === document.documentElement) return true;
    return false;
  }
  append(...kids) {
    for (const kid of kids) {
      const node = typeof kid === "string" ? new Text(kid) : kid;
      if (node.parentNode) node.remove ? node.remove() : null;
      node.parentNode = this;
      this.childNodes.push(node);
    }
    mutated(this, "kids");
  }
  appendChild(kid) {
    this.append(kid);
    return kid;
  }
  replaceChildren(...kids) {
    for (const node of this.childNodes) node.parentNode = null;
    this.childNodes = [];
    this.append(...kids);
  }
  remove() {
    const parent = this.parentNode;
    if (!parent) return;
    parent.childNodes = parent.childNodes.filter((node) => node !== this);
    this.parentNode = null;
    mutated(parent, "kids");
  }
  contains(node) {
    for (let at = node; at; at = at.parentNode) if (at === this) return true;
    return false;
  }
  get textContent() {
    return this.childNodes.map((node) => node.textContent).join("");
  }
  set textContent(value) {
    this.replaceChildren(...(value === "" || value == null ? [] : [new Text(value)]));
  }
  // (markup written from a string is never parsed here: it is counted, so a test can say there was none)
  set innerHTML(value) {
    log.markup += 1;
    this.replaceChildren();
  }
  insertAdjacentHTML() {
    log.markup += 1;
  }
  /* finding */
  all() {
    const out = [];
    const walk = (el) => {
      for (const kid of el.children) {
        out.push(kid);
        walk(kid);
      }
    };
    walk(this);
    return out;
  }
  matches(selector) {
    return selector.split(",").some((one) => {
      const chain = one.trim().split(/\s+/).map(compound);
      if (!fits(this, chain[chain.length - 1])) return false;
      let at = this.parentElement;
      for (let i = chain.length - 2; i >= 0; i--) {
        while (at && !fits(at, chain[i])) at = at.parentElement;
        if (!at) return false;
        at = at.parentElement;
      }
      return true;
    });
  }
  querySelectorAll(selector) {
    return this.all().filter((el) => el.matches(selector));
  }
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
  closest(selector) {
    for (let at = this; at; at = at.parentElement) if (at.matches(selector)) return at;
    return null;
  }
  /* layout: none here; with `page.layout` an element is "shown" when neither it nor a parent is hidden */
  getClientRects() {
    if (!page.layout || !this.isConnected) return [];
    for (let at = this; at; at = at.parentElement) if (at.hidden) return [];
    return [{}];
  }
  getBoundingClientRect() {
    return { left: 0, right: 0, top: 0, bottom: 0, width: 0, height: 0 };
  }
  focus() {
    document.activeElement = this;
  }
  scrollIntoView() {
    log.scrolled.push(this.id);
  }
  click() {
    return fire(this, "click");
  }
}

// an event from `target` up through its parents to the document and the window
function fire(target, type, init = {}) {
  const event = {
    type,
    target,
    button: 0,
    defaultPrevented: false,
    stopped: false,
    preventDefault() {
      this.defaultPrevented = true;
    },
    stopPropagation() {
      this.stopped = true;
    },
    ...init,
  };
  const route = [];
  for (let at = target; at instanceof El; at = at.parentNode) route.push(at);
  route.push(docTarget, winTarget);
  for (const stop of route) {
    if (event.stopped) break;
    stop.hear(event);
  }
  return event;
}

const docTarget = new Target();
const winTarget = new Target();
const root = new El("html");
const built = [];
for (const node of input.tree) {
  const made = "text" in node ? new Text(node.text) : new El(node.tag);
  if (made instanceof El) Object.assign(made.attrs, node.attrs);
  built.push(made);
  const parent = node.parent === null || node.parent === undefined ? null : built[node.parent];
  if (made instanceof El && made.tagName === "HTML") continue;
  made.parentNode = parent || root;
  (parent || root).childNodes.push(made);
}
const top = built.find((node) => node instanceof El && node.tagName === "HTML");
if (top) Object.assign(root.attrs, top.attrs), (root.childNodes = top.childNodes), root.childNodes.forEach((node) => (node.parentNode = root));

const document = {
  documentElement: root,
  body: root.querySelector("body"),
  activeElement: null,
  hidden: false,
  visibilityState: "visible",
  readyState: "complete",
  getElementById: (id) => root.all().find((el) => el.id === id) || null,
  querySelector: (selector) => root.querySelector(selector),
  querySelectorAll: (selector) => root.querySelectorAll(selector),
  createElement(tag) {
    log.created.push(String(tag).toLowerCase());
    return new El(tag);
  },
  createElementNS(_, tag) {
    return new El(tag);
  },
  createTextNode: (text) => new Text(text),
  addEventListener: (...args) => docTarget.addEventListener(...args),
  removeEventListener: (...args) => docTarget.removeEventListener(...args),
};
document.activeElement = document.body;

/* ---------------- the window: a clock, the network, the address ---------------- */

const timers = [];
let nextTimer = 1;
const clock = {
  timeout: (fn, ms = 0) => (timers.push({ id: nextTimer, at: page.time + ms, fn }), nextTimer++),
  interval(fn, ms) {
    log.intervals.push(ms);
    timers.push({ id: nextTimer, at: page.time + ms, every: ms, fn });
    return nextTimer++;
  },
  clear(id) {
    const i = timers.findIndex((timer) => timer.id === id);
    if (i >= 0) timers.splice(i, 1);
  },
  // let `ms` of the page's time pass: every timer due in it runs at its moment, and what it started settles
  async pass(ms) {
    const end = page.time + ms;
    for (;;) {
      const due = timers.filter((timer) => timer.at <= end).sort((a, b) => a.at - b.at)[0];
      if (!due) break;
      page.time = Math.max(page.time, due.at);
      if (due.every) due.at += due.every;
      else timers.splice(timers.indexOf(due), 1);
      due.fn();
      await settled();
    }
    page.time = end;
    await settled();
  },
};

const address = new URL("http://duck.test/");
const location = {
  get pathname() {
    return address.pathname;
  },
  get search() {
    return address.search;
  },
  get hash() {
    return address.hash;
  },
  get origin() {
    return address.origin;
  },
  get href() {
    return address.href;
  },
};
const go = (url) => {
  address.href = new URL(url, address.href).href;
  log.history.push(`${address.pathname}${address.hash}`);
};
const store = () => {
  const kept = new Map();
  return { getItem: (key) => (kept.has(key) ? kept.get(key) : null), setItem: (key, value) => kept.set(key, String(value)), kept };
};

class CustomEvent {
  constructor(type, init) {
    this.type = type;
    this.detail = init ? init.detail : undefined;
  }
}

Object.assign(globalThis, {
  document,
  location,
  MutationObserver,
  CustomEvent,
  ResizeObserver: class {
    observe() {}
  },
  history: { pushState: (_, __, url) => go(url), replaceState: (_, __, url) => go(url) },
  sessionStorage: store(),
  localStorage: store(),
  innerWidth: 1280,
  matchMedia: () => ({ matches: false, addEventListener() {}, addListener() {} }),
  getComputedStyle: () => ({ getPropertyValue: () => "" }),
  requestAnimationFrame: (fn) => page.frames.push(fn),
  setTimeout: clock.timeout,
  clearTimeout: clock.clear,
  setInterval: clock.interval,
  clearInterval: clock.clear,
  scrollTo() {},
  addEventListener: (...args) => winTarget.addEventListener(...args),
  removeEventListener: (...args) => winTarget.removeEventListener(...args),
  dispatchEvent(event) {
    if (event.type === "duck:view") log.views.push(event.detail.view);
    winTarget.hear(event);
    return true;
  },
  // every request is written down; `page.answers` (by "METHOD path", or by path) says what the server answers
  fetch(url, options = {}) {
    const asked = { url: String(url), method: options.method || "GET", body: options.body ? JSON.parse(options.body) : null, at: page.time };
    log.fetches.push(asked);
    const where = asked.url.split("?")[0];
    const answer = page.answers[`${asked.method} ${where}`] ?? page.answers[where];
    if (answer === undefined) return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) });
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(typeof answer === "function" ? answer(asked) : answer) });
  },
  // the catalogs are not loaded: a word is its key (and its values), a field of the server's is itself
  I18N: {
    ready: Promise.resolve(),
    t: (key, values) => (values && Object.keys(values).length ? `${key}|${JSON.stringify(values)}` : key),
    text(el, key, values) {
      el.setAttribute("data-i18n", key);
      el.textContent = this.t(key, values);
    },
    attr: (el, name, key) => el.setAttribute(name, key),
    field: (obj, name) => obj[name],
    items: (obj, name) => obj[name],
    show: (el, obj, name) => (el.textContent = obj[name] ?? ""),
    lang: "zh-Hant",
    number: (x) => String(x),
    date: (d, options, zh) => zh(d),
    list: (items) => items.join("+"),
    onChange() {},
  },
});
globalThis.window = globalThis;

// one turn of the screen: what asked for the next frame gets it
const frame = async () => {
  for (const fn of page.frames.splice(0)) fn(page.time);
  await settled();
};
const byId = (id) => document.getElementById(id);
const loadScripts = async () => {
  for (const script of input.scripts) {
    const file = path.join(staticDir, script.name);
    if (script.module) await import(pathToFileURL(file).href);
    // (a classic script, as the page runs it: in the global scope, where there is no `module`)
    else (0, eval)(fs.readFileSync(file, "utf8"));
  }
  await settled();
  await frame();
};
const count = (where, method = "GET") => log.fetches.filter((asked) => asked.url.split("?")[0] === where && asked.method === method).length;

/* ---------------- the scenarios ---------------- */

const scenarios = {
  // shell.js: the panel's tabs, the two views, a link into a part of the other view
  async shell(data) {
    page.layout = true;
    if (data && data.start) address.href = new URL(data.start, address.href).href;
    await loadScripts();
    const tabs = document.querySelectorAll("[data-ctl-tab]");
    const panels = document.querySelectorAll("[data-ctl-panel]");
    const state = () => ({
      path: `${location.pathname}${location.hash}`,
      lab: document.body.classList.contains("view-data"),
      sim_hidden: byId("viewSim").hidden,
      lab_hidden: byId("viewData").hidden,
      selected: tabs.filter((tab) => tab.getAttribute("aria-selected") === "true").map((tab) => tab.dataset.ctlTab),
      shown: panels.filter((panel) => !panel.hidden).map((panel) => panel.dataset.ctlPanel),
      current: document.querySelectorAll(".jumpnav [data-view-link]").filter((link) => link.hasAttribute("aria-current")).map((link) => link.dataset.viewLink),
    });
    const out = { first: state(), tabs: {}, links: [] };
    // every tab, pressed (the mode that offers it has shown it)
    for (const tab of tabs) tab.hidden = false;
    await settled();
    await frame();
    for (const tab of tabs) {
      tab.click();
      await settled();
      out.tabs[tab.dataset.ctlTab] = { ...state(), controls: tab.getAttribute("aria-controls"), panel_ids: panels.filter((panel) => !panel.hidden).map((panel) => panel.id) };
    }
    out.remembered = globalThis.sessionStorage.getItem("duck.ctl.tab");
    // the lab, by its link in the header; back with the browser's own button
    document.querySelector('.jumpnav [data-view-link="data"]').click();
    await settled();
    out.lab = state();
    address.href = new URL("/", address.href).href;
    winTarget.hear({ type: "popstate" });
    await settled();
    out.back = state();
    // every link into a part of the other view opens that view and shows the part
    for (const link of document.querySelectorAll("[data-view-link]")) {
      const to = new URL(link.href, address.href);
      if (!to.hash) continue;
      address.href = new URL(to.pathname === "/lab" ? "/" : "/lab", address.href).href;
      winTarget.hear({ type: "popstate" });
      log.scrolled.length = 0;
      link.click();
      await settled();
      const target = byId(to.hash.slice(1));
      out.links.push({ from: link.id || link.getAttribute("href"), to: to.hash.slice(1), found: !!target, scrolled: [...log.scrolled], ...state() });
    }
    // a link in the lab to a part that sits in the panel: back to the simulator, its tab chosen, its fold-out open
    out.parts = [];
    for (const link of byId("viewData").querySelectorAll('a[href^="#"]')) {
      const target = byId(link.getAttribute("href").slice(1));
      const panel = target && target.closest("[data-ctl-panel]");
      if (!panel) {
        out.parts.push({ to: link.getAttribute("href").slice(1), found: !!target, in_lab: !!target && byId("viewData").contains(target), panel: null });
        continue;
      }
      address.href = new URL("/lab", address.href).href;
      winTarget.hear({ type: "popstate" });
      for (const fold of document.querySelectorAll("details")) fold.open = false;
      log.scrolled.length = 0;
      const event = link.click();
      await settled();
      const folds = [];
      for (let fold = target.closest("details"); fold; fold = fold.parentElement && fold.parentElement.closest("details")) folds.push(fold.open);
      out.parts.push({ to: target.id, found: true, panel: panel.dataset.ctlPanel, handled: event.defaultPrevented, folds, scrolled: [...log.scrolled], ...state() });
    }
    // the enlarged view keeps its panel: it opens as a floating one, and shuts again when the view is small
    const view = byId("duckView");
    view.classList.add("is-big");
    await settled();
    out.big = view.dataset.ctl;
    view.classList.remove("is-big");
    await settled();
    out.small = view.dataset.ctl;
    out.handle = byId("ctlHandle").getAttribute("aria-expanded");
    byId("ctlHandle").click();
    await settled();
    out.opened = [view.dataset.ctl, byId("ctlHandle").getAttribute("aria-expanded")];
    // the label that says which duck is drawn rides on the HUD's tooltip
    const lines = ["page.truth.TRUTH_TITLE", "its line"].map((text) => {
      const part = new El("span");
      part.textContent = text;
      return part;
    });
    byId("stageMode").replaceChildren(...lines);
    await settled();
    out.hud_title = byId("hud").title;
    out.views = log.views;
    return out;
  },

  // plan2d.js + minimap.js: the home from above, the duck where the body's truth has it
  async minimap(data) {
    page.answers["/api/home"] = data.home;
    page.answers["/api/neuro/knows"] = { on: false };
    const root = byId("miniMap");
    const out = { before: { hidden: root.hidden } };
    await loadScripts();
    const drawn = () => {
      const svg = byId("miniMapSvg");
      const duck = svg.querySelector(".mm-duck");
      return {
        hidden: root.hidden,
        view_box: svg.getAttribute("viewBox"),
        fixed: [...new Set(svg.querySelectorAll("[data-fixed]").map((node) => node.getAttribute("data-fixed")))],
        rooms: svg.querySelectorAll(".mm-room").length,
        labels: svg.querySelectorAll(".mm-label").map((node) => node.textContent),
        duck: duck ? { visibility: duck.getAttribute("visibility"), transform: duck.getAttribute("transform") } : null,
        people: svg.querySelectorAll(".mm-person").map((node) => [Number(node.getAttribute("cx")), Number(node.getAttribute("cy"))]),
        toys: Object.fromEntries(svg.querySelectorAll("[data-thing]").map((node) => [node.getAttribute("data-thing"), node.getAttribute("transform") || ""])),
      };
    };
    const body = async (info) => {
      globalThis.dispatchEvent(new CustomEvent("duck:body", { detail: info }));
      await settled();
      await frame();
    };
    out.loaded = drawn(); // the home is known, no physics body yet
    await body(data.body);
    out.live = drawn();
    await body({ ...data.body, available: false, frame: null });
    out.starting = drawn(); // the body is there, its truth is not (yet)
    await body({ ...data.body, real: { rehearsal: false } });
    out.real = drawn();
    await body({ enabled: false });
    out.off = drawn();
    await body(data.body);
    // a minute of the page's time: what the map itself asked the server for
    const before = log.fetches.length;
    await clock.pass(60000);
    out.asked = log.fetches.map((asked) => [asked.method, asked.url.split("?")[0]]);
    out.asked_in_a_minute = log.fetches.slice(before).map((asked) => [asked.method, asked.url.split("?")[0]]);
    out.intervals = log.intervals;
    // folded by its button, remembered by this viewer
    byId("miniMapFold").click();
    await settled();
    out.folded = { state: root.dataset.state, expanded: byId("miniMapFold").getAttribute("aria-expanded"), kept: globalThis.localStorage.getItem("duck.minimap.fold") };
    return out;
  },

  // neuro.js: the neural duck's cards from the server's own answers (one telemetry answer after another, a
  // second each); then every stimulus button of the page pressed once
  async neuro(data) {
    page.layout = true;
    let step = 0;
    page.answers["/api/neuro/status"] = data.status;
    page.answers["/api/neuro/telemetry"] = () => data.telemetry[Math.min(step, data.telemetry.length - 1)];
    page.answers["POST /api/neuro/stimulus"] = (asked) => ({ ok: true, kind: asked.body.kind });
    await loadScripts();
    const out = { skills: [], mode: document.body.classList.contains("neuro-mode") };
    const line = (card, name) => {
      const el = card.querySelector(`.neuro-skill-${name}`);
      return el && !el.hidden ? el.textContent : null;
    };
    for (step = 0; step < data.telemetry.length; step++) {
      await clock.pass(1000);
      out.skills.push(
        Object.fromEntries(
          byId("neuroSkillCards").children.map((card) => [
            card.dataset.skill,
            { state: card.dataset.state, says: line(card, "state"), intent: line(card, "intent") !== null, phase: line(card, "phase"), result: line(card, "result"), reason: line(card, "reason") },
          ]),
        ),
      );
    }
    const card = byId("neuroStim");
    const buttons = document.querySelectorAll("[data-stim]");
    out.panel = card.closest("[data-ctl-panel]").dataset.ctlPanel;
    out.elsewhere = buttons.filter((button) => !card.contains(button)).map((button) => button.dataset.stim);
    const before = log.fetches.length;
    out.buttons = [];
    for (const button of buttons) {
      out.buttons.push(button.dataset.stim);
      button.click();
      await clock.pass(1000);
    }
    out.sent = log.fetches.slice(before).filter((asked) => asked.method === "POST").map((asked) => ({ url: asked.url, body: asked.body }));
    return out;
  },

  // neuro_sound.js: the sound scenes' strip in 玩; every button of it pressed once
  async sound(data) {
    page.answers["/api/neuro/sound/status"] = data.status;
    page.answers["POST /api/neuro/sound/world"] = { ok: true };
    const told = [];
    globalThis.addEventListener("duck:sound-status", (event) => told.push(event.detail));
    byId("neuroStim").hidden = false; // (the prototype's mode is on: neuro.js has shown the 玩 card)
    await loadScripts();
    await clock.pass(2000);
    const strip = byId("neuroSound");
    const out = { built: !!strip, in_play_tab: !!strip && byId("ctl-stim").contains(strip), buttons: [] };
    const before = log.fetches.length;
    for (const button of strip ? strip.querySelectorAll("button[data-sound]") : []) {
      out.buttons.push(button.dataset.sound);
      button.click();
      await settled();
    }
    out.sent = log.fetches.slice(before).filter((asked) => asked.method !== "GET").map((asked) => ({ url: asked.url, body: asked.body }));
    await clock.pass(2000);
    out.asked = [...new Set(log.fetches.map((asked) => `${asked.method} ${asked.url.split("?")[0]}`))];
    out.told = told[told.length - 1];
    return out;
  },

  // shell.js + bodylink.js: the card in 我的小鴨 says which body the brain drives, while its tab is open
  async bodylink(data) {
    page.layout = true;
    page.answers["/api/body"] = data.body;
    for (const tab of document.querySelectorAll("[data-ctl-tab]")) tab.hidden = false;
    await loadScripts();
    const tab = (id) => document.querySelector(`[data-ctl-tab="${id}"]`);
    const out = { asked: {} };
    tab("stim").click();
    await clock.pass(30000);
    out.asked.elsewhere = count("/api/body");
    tab("duck").click();
    await settled();
    out.asked.opened = count("/api/body");
    const line = byId("bodyLinkState");
    const stats = byId("bodyLinkStats");
    out.line = { state: line.dataset.state, text: line.textContent, in_tab: byId("ctl-duck").contains(line) };
    out.stats = { hidden: stats.hidden, text: stats.textContent };
    await clock.pass(30000);
    out.asked.half_minute_open = count("/api/body");
    tab("stim").click();
    await clock.pass(30000);
    out.asked.left = count("/api/body");
    out.sent = log.fetches.filter((asked) => asked.method !== "GET").length;
    out.markup = log.markup;
    return out;
  },

  // shell.js + growth.js: the growth record reads only while it can be seen; the fast-forward form sends what it offers
  async growth(data) {
    page.layout = true;
    page.answers["/api/life/growth"] = { ok: true, growth: data.view };
    page.answers["GET /api/life/fastforward"] = { ok: true, fastforward: { state: "idle" } };
    page.answers["POST /api/life/fastforward"] = { ok: true, fastforward: { state: "idle" } };
    for (const tab of document.querySelectorAll("[data-ctl-tab]")) tab.hidden = false;
    await loadScripts();
    const reads = () => count("/api/life/growth");
    const texts = (el) => el.all().flatMap((node) => node.childNodes.filter((kid) => kid.nodeType === 3).map((kid) => kid.text));
    const out = { reads: {} };
    const tab = (id) => document.querySelector(`[data-ctl-tab="${id}"]`);
    // another tab is open: nothing is read, however long
    tab("stim").click();
    await clock.pass(60000);
    out.reads.elsewhere = reads();
    // the 成長 tab opens: read at once, then now and then
    tab("grow").click();
    await settled();
    out.reads.opened = reads();
    out.short = texts(byId("growthTabBody"));
    await clock.pass(60000);
    out.reads.minute_open = reads();
    // the browser tab goes to the back: nothing is read; to the front again: at once
    document.hidden = true;
    await clock.pass(60000);
    out.reads.in_the_back = reads();
    document.hidden = false;
    docTarget.hear({ type: "visibilitychange" });
    await settled();
    out.reads.to_the_front = reads();
    // away to another tab of the panel: nothing more
    tab("duck").click();
    await clock.pass(60000);
    out.reads.left = reads();
    // the lab opens: its whole record is read at once and drawn
    document.querySelector('.jumpnav [data-view-link="data"]').click();
    await settled();
    out.reads.lab = reads();
    out.long = texts(byId("gvLive"));
    out.made = [...new Set(log.created)];
    out.markup = log.markup;
    // the fast-forward: every care style either select offers, sent with the form's own days and seed
    out.offered = {};
    out.sent = [];
    for (const id of ["gvFfStyleA", "gvFfStyleB"]) {
      out.offered[id] = byId(id).querySelectorAll("option").map((option) => option.getAttribute("value"));
      for (const value of out.offered[id]) {
        byId(id).value = value;
        const before = log.fetches.length;
        byId("gvFfStart").click();
        await settled();
        out.sent.push(...log.fetches.slice(before).filter((asked) => asked.method === "POST").map((asked) => ({ url: asked.url, body: asked.body })));
      }
    }
    out.intervals = log.intervals;
    return out;
  },
};

scenarios[input.scenario](input.data).then(
  (out) => {
    console.log(JSON.stringify(out));
    process.exit(0);
  },
  (error) => {
    console.error(error);
    process.exit(1);
  },
);
