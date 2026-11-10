// Runs a page script in Node over the page's own markup: static/index.html is read here into stand-in elements, so
// a script that asks for an element, an attribute or a selector the page does not have is seen (the other page
// harnesses build their own small page; this one checks a script against the page it is shipped with). Timers
// are real, and the voice scenario really fetches from the test's own server on 127.0.0.1.
//   node real_page_node_harness.mjs <static dir> <scenario> [input json]
// Prints one JSON value (tests/test_real_duck_page.py, test_live_voice_page.py, test_simweb.py,
// test_simweb_voice.py and test_neuro_voice.py assert on it).
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const [staticDir, scenario, inputJson] = process.argv.slice(2);
const input = inputJson ? JSON.parse(inputJson) : {};

// ── the page: stand-in elements with the few DOM methods the page scripts use ────────────────────────────

const registry = new Map(); // id → element
const missing = new Set(); // ids a script asked for that the page does not have
const camel = (name) => name.replace(/-([a-z])/g, (_, c) => c.toUpperCase());

class El {
  constructor(tag, attrs = {}) {
    this.tagName = tag.toUpperCase();
    this.parent = null;
    this.children = []; // elements and strings
    this.attrs = {};
    this.dataset = {};
    this.style = {};
    this.listeners = {};
    this.classes = new Set();
    this.scrollTop = 0;
    for (const [name, value] of Object.entries(attrs)) this.setAttribute(name, value);
    this.hidden = "hidden" in attrs;
    this.disabled = "disabled" in attrs;
    this.checked = "checked" in attrs;
    const classes = this.classes;
    this.classList = {
      add: (...names) => names.forEach((name) => classes.add(name)),
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      contains: (name) => classes.has(name),
      toggle(name, on = !classes.has(name)) {
        if (on) classes.add(name);
        else classes.delete(name);
        return on;
      },
    };
  }
  set id(value) {
    this._id = String(value);
    if (value) registry.set(this._id, this);
  }
  get id() {
    return this._id || "";
  }
  set className(value) {
    this.classes.clear();
    for (const name of String(value).split(/\s+/).filter(Boolean)) this.classes.add(name);
  }
  get className() {
    return [...this.classes].join(" ");
  }
  set title(value) {
    this.attrs.title = String(value);
  }
  get title() {
    return this.attrs.title || "";
  }
  set textContent(value) {
    this.children = value === "" || value === null || value === undefined ? [] : [String(value)];
  }
  get textContent() {
    return this.children.map((child) => (child instanceof El ? child.textContent : child)).join("");
  }
  set innerHTML(html) {
    this.html = String(html);
    this.children = [];
    readInto(this, this.html, "markup a script wrote");
  }
  get innerHTML() {
    return this.html || "";
  }
  // a <select> has its first option's value until one is chosen (as in a browser)
  set value(value) {
    this._value = String(value);
  }
  get value() {
    if (this._value !== undefined) return this._value;
    if (this.tagName === "SELECT") return this.options.length ? this.options[0].value : "";
    return this.attrs.value || "";
  }
  get options() {
    return this.all().filter((el) => el.tagName === "OPTION");
  }
  get selectedOptions() {
    return this.options.filter((option) => option.value === this.value).slice(0, 1);
  }
  get parentElement() {
    return this.parent;
  }
  append(...items) {
    for (const item of items) {
      if (item instanceof El) item.parent = this;
      this.children.push(item);
    }
  }
  appendChild(item) {
    this.append(item);
    return item;
  }
  replaceChildren(...items) {
    this.children = [];
    this.append(...items);
  }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this);
  }
  cloneNode() {
    return new El(this.tagName);
  }
  setAttribute(name, value) {
    if (name === "id") this.id = value;
    else if (name === "class") this.className = value;
    else if (name.startsWith("data-")) this.dataset[camel(name.slice(5))] = String(value);
    else this.attrs[name] = String(value);
  }
  getAttribute(name) {
    if (name === "id") return this.id || null;
    if (name.startsWith("data-")) return this.dataset[camel(name.slice(5))] ?? null;
    return name in this.attrs ? this.attrs[name] : null;
  }
  removeAttribute(name) {
    if (name.startsWith("data-")) delete this.dataset[camel(name.slice(5))];
    else delete this.attrs[name];
  }
  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }
  removeEventListener() {}
  fire(type, event = {}) {
    for (const fn of this.listeners[type] || []) fn({ type, target: this, preventDefault() {}, ...event });
  }
  setPointerCapture() {}
  all() {
    const out = [];
    const walk = (el) => {
      for (const child of el.children) {
        if (!(child instanceof El)) continue;
        out.push(child);
        walk(child);
      }
    };
    walk(this);
    return out;
  }
  querySelectorAll(selector) {
    return this.all().filter((el) => el.matches(selector));
  }
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
  closest(selector) {
    for (let at = this; at; at = at.parent) if (at.matches(selector)) return at;
    return null;
  }
  // The selector shapes the page scripts use: tag, #id, .class, [attr], [attr="v"] joined into compounds,
  // descendant chains and lists. Anything else is an error here, never a silent "no match".
  matches(selector) {
    const one = (el, compound) => {
      const m = /^([a-z][\w-]*)?((?:#[\w-]+|\.[\w-]+|\[[\w-]+(?:="[^"]*")?\])*)$/.exec(compound);
      if (!m || !compound) throw new Error(`selector not understood: ${selector}`);
      if (m[1] && el.tagName !== m[1].toUpperCase()) return false;
      for (const [, id, cls, attr, value] of m[2].matchAll(/#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:="([^"]*)")?\]/g)) {
        if (id && el.id !== id) return false;
        if (cls && !el.classes.has(cls)) return false;
        if (attr) {
          const got = el.getAttribute(attr);
          if (got === null || (value !== undefined && got !== value)) return false;
        }
      }
      return true;
    };
    return selector.split(",").some((part) => {
      const chain = part.trim().split(/\s+/);
      if (!one(this, chain[chain.length - 1])) return false;
      let at = this.parent;
      for (let i = chain.length - 2; i >= 0; i--) {
        while (at && !one(at, chain[i])) at = at.parent;
        if (!at) return false;
        at = at.parent;
      }
      return true;
    });
  }
}

// Markup into elements under `root`. The page (and what its scripts write) has every tag closed; a tag this
// reader does not understand stops the run.
const VOID = new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"]);
function readInto(root, html, what) {
  const open = [root];
  const token =
    /<!--[\s\S]*?-->|<!doctype[^>]*>|<(script|style)\b[^>]*>[\s\S]*?<\/\1\s*>|<\/([a-zA-Z][\w-]*)\s*>|<([a-zA-Z][\w-]*)((?:\s+[^\s=>\/]+(?:\s*=\s*(?:"[^"]*"|'[^']*'))?)*)\s*(\/?)>|([^<]+)/gi;
  let at = 0;
  for (const m of html.matchAll(token)) {
    if (m.index !== at) throw new Error(`${what}: not understood at ${at}: ${html.slice(at, at + 60)}`);
    at = m.index + m[0].length;
    const [, , closed, tag, attrText, selfClosed, text] = m;
    if (closed) {
      if (open.length < 2 || open[open.length - 1].tagName !== closed.toUpperCase()) throw new Error(`${what}: </${closed}> closes nothing at ${m.index}`);
      open.pop();
    } else if (tag) {
      const attrs = {};
      for (const [, name, , double, single] of attrText.matchAll(/([^\s=>\/]+)(\s*=\s*(?:"([^"]*)"|'([^']*)'))?/g)) attrs[name.toLowerCase()] = double ?? single ?? "";
      const el = new El(tag, attrs);
      open[open.length - 1].append(el);
      if (!selfClosed && !VOID.has(tag.toLowerCase())) open.push(el);
    } else if (text && text.trim()) {
      open[open.length - 1].append(text);
    }
  }
  if (at !== html.length || open.length !== 1) throw new Error(`${what}: a tag is left open`);
  return root;
}

const page = readInto(new El("#document"), fs.readFileSync(path.join(staticDir, "index.html"), "utf-8"), "index.html");
const body = page.querySelector("body");
const byId = (id) => registry.get(id) || null;

const say = (key, params) => key + (params && Object.keys(params).length ? JSON.stringify(params) : "");
globalThis.window = globalThis;
globalThis.document = {
  body,
  hidden: false,
  visibilityState: "visible",
  readyState: "complete",
  activeElement: null,
  getElementById(id) {
    if (!registry.has(id)) missing.add(id);
    return byId(id);
  },
  querySelector(selector) {
    const found = page.querySelector(selector);
    if (!found && /^#[\w-]+$/.test(selector)) missing.add(selector.slice(1));
    return found;
  },
  querySelectorAll: (selector) => page.querySelectorAll(selector),
  createElement: (tag) => new El(tag),
  addEventListener() {},
  removeEventListener() {},
};
const windowListeners = {};
globalThis.addEventListener = (type, fn) => (windowListeners[type] = windowListeners[type] || []).push(fn);
const fireWindow = (type) => (windowListeners[type] || []).forEach((fn) => fn({ type }));
globalThis.removeEventListener = () => {};
globalThis.dispatchEvent = () => true;
globalThis.CustomEvent = class {
  constructor(type, init = {}) {
    this.type = type;
    this.detail = init.detail;
  }
};
globalThis.MutationObserver = class {
  observe() {}
  disconnect() {}
};
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
globalThis.requestAnimationFrame = () => 0; // (a frame never comes here)
// the catalog: a key's words are the key itself with its values ('page.x{"n":1}'), a server field its 繁體中文
globalThis.I18N = {
  ready: Promise.resolve(),
  lang: "zh-Hant",
  SOURCE: "zh-Hant",
  t: say,
  tHtml: say,
  field: (obj, name) => obj[name],
  items: (obj, name) => obj[name] || [],
  text(el, key, params) {
    el.dataset.i18n = key;
    el.textContent = say(key, params);
  },
  attr(el, name, key, params) {
    el.setAttribute(name, say(key, params));
  },
  onChange() {},
};

// A classic page script, run as the page runs it (its top-level names are the page's globals).
const runScript = (name) => vm.runInThisContext(fs.readFileSync(path.join(staticDir, name), "utf-8"), { filename: name });
const settle = (ms = 30) => new Promise((resolve) => setTimeout(resolve, ms));
const until = async (what, done, ms = 15000) => {
  for (const end = Date.now() + ms; !done(); await settle(5)) if (Date.now() > end) throw new Error(`never happened: ${what}`);
};

// Web Audio, as far as voice.js uses it: nothing sounds; every buffer source that starts is kept in `started`.
// There is no oscillator here: a call the page synthesized instead of playing the bank's file would not start.
function fakeAudio() {
  const clock = { currentTime: 0 };
  const started = [];
  const level = (value) => ({ value, cancelScheduledValues() {}, setValueAtTime(v) { this.value = v; }, linearRampToValueAtTime(v) { this.value = v; }, setTargetAtTime(v) { this.value = v; } });
  const buffer = (length, rate) => {
    const data = new Float32Array(length);
    return { length, sampleRate: rate, duration: length / rate, getChannelData: () => data };
  };
  window.AudioContext = class {
    constructor() {
      this.state = "suspended";
      this.sampleRate = 48000;
      this.destination = {};
    }
    get currentTime() {
      return clock.currentTime;
    }
    createGain() {
      return { gain: level(1), connect() {}, disconnect() {} };
    }
    createBuffer(channels, length, rate) {
      return buffer(length, rate);
    }
    createBufferSource() {
      return { buffer: null, connect() {}, stop() {}, start() { started.push(this.buffer.length); } };
    }
    resume() {
      this.state = "running";
      return Promise.resolve();
    }
  };
  window.OfflineAudioContext = class {
    createBuffer(channels, length, rate) {
      return buffer(length, rate);
    }
    // (the bank's files: 16-bit mono PCM after a 44-byte header)
    decodeAudioData(bytes, resolve) {
      const pcm = new Int16Array(bytes.slice(44));
      const out = buffer(pcm.length, 48000);
      out.getChannelData(0).set(Float32Array.from(pcm, (v) => v / 32768));
      resolve(out);
    }
  };
  return { clock, started };
}
const shown = (id) => {
  const el = byId(id);
  return el ? { hidden: el.hidden, text: el.textContent } : null;
};

// ── the scenarios ────────────────────────────────────────────────────────────────────────────────────────

const scenarios = {
  // realduck.js: /api/body says a real duck (input.real, input.remote), then a simulated body again.
  realDuck() {
    runScript("realduck.js");
    const api = window.duckReal;
    const notes = () => Object.fromEntries(page.querySelectorAll("[data-real-note]").map((el) => [el.id, el.hidden]));
    const chips = () => Object.fromEntries(["chipReal", "chipLink", "chipBattery"].map((id) => [id, shown(id)]));
    const start = { notes: notes(), chips: chips() };
    api.takeBody({ enabled: true, real: input.real, remote: input.remote });
    const reason = say(api.REASON);
    const held = (el, gate) =>
      el.disabled && el.dataset.realGate !== undefined && el.getAttribute("title") === reason && (el.getAttribute("aria-describedby") || "").split(/\s+/).includes(gate.note);
    const gates = [];
    for (const gate of api.GATES) {
      for (const selector of gate.selectors) {
        const found = page.querySelectorAll(selector);
        gates.push({ code: gate.code, selector, found: found.length, held: found.filter((el) => held(el, gate)).length, note: gate.note, noteOnPage: registry.has(gate.note) });
      }
    }
    const wasHeld = page.querySelectorAll("[data-real-gate]");
    const real = { mode: body.classList.contains("real-duck"), notes: notes(), chips: chips(), battery: shown("worldBatteryVal") };
    api.takeBody({ enabled: true, real: null, remote: input.remote });
    const simulated = {
      mode: body.classList.contains("real-duck"),
      notes: notes(),
      chips: chips(),
      stillHeld: wasHeld.filter((el) => el.disabled || el.dataset.realGate !== undefined).length,
    };
    return { start, gates, real, simulated, held: wasHeld.length, missing: [...missing] };
  },

  // live_voice.js: the page starts, GET /api/live/status answers input.status, the panel is drawn.
  async liveVoice() {
    const asked = [];
    globalThis.fetch = async (url) => {
      asked.push(String(url));
      return { ok: true, status: 200, json: async () => input.status };
    };
    runScript("live_voice.js");
    await settle();
    const el = (id) => byId(id) || new El("missing"); // (an element the page lacks is in `missing`)
    return {
      asked,
      on: window.__liveVoice.state.on,
      panelHidden: el("liveVoice").hidden,
      chipHidden: el("liveChipCloud").hidden,
      chip: [el("liveChipCloud").dataset.sending, el("liveChipCloud").textContent],
      cloudLabel: el("liveCloudLabel").textContent,
      sending: [el("liveSendAudio").dataset.sending, el("liveSendVideo").dataset.sending],
      state: el("liveState").textContent,
      heard: el("liveHeard").children.map((li) => li.textContent),
      calls: el("liveCalls").children.map((li) => li.textContent),
      eye: { hidden: el("liveEye").hidden, src: el("liveEye").src || "", note: el("liveEyeNote").textContent },
      usage: el("liveUsage").textContent,
      lines: el("liveLine").options.map((option) => option.value),
      sayDisabled: el("liveSay").disabled,
      mic: el("liveMic").textContent,
      micNote: el("liveMicNote").textContent,
      missing: [...missing],
    };
  },

  // voice.js against the server at input.base (its manifest and its files are really fetched): a call before any
  // click, a click, more calls (input.draws: what Math.random gives each call's pick, the first call's first),
  // two calls of one moment, the page's own mute button and volume slider; with input.duck the neural duck on
  // screen, and with input.switch ({preset, seed}) the page asks the server for another duck as the picker does.
  async voice() {
    const asked = [];
    const get = globalThis.fetch;
    globalThis.fetch = (url, init) => {
      asked.push(String(url));
      return get(input.base + url, init);
    };
    let draw = 0;
    Math.random = () => draw;
    const sound = fakeAudio();
    runScript("voice.js");
    const voice = window.duckVoice;
    const state = voice.state;
    const settled = () => !state.loadingManifest && !state.reloadManifest;
    const ready = (seed) => until(`the manifest of duck ${seed}`, () => settled() && state.manifest && state.manifest.ready && (seed == null || state.manifest.duck_seed === seed));
    // The calls of one moment (each a list of tags), picked with `pick` as the random draw, to their end in the
    // page's own log; the audio clock then moves on (what sounded is over before the next). A call whose manifest
    // is read again while its sound is fetched is let go by the page without a word: it is made again here.
    const call = async (pick, ...calls) => {
      for (;;) {
        await until("the manifest", settled);
        const before = state.log.length;
        const manifest = state.manifest;
        draw = pick;
        const voiced = calls.map((tags) => voice.call(tags)).every(Boolean);
        await until(`the call ${calls}`, () => state.log.length > before || state.manifest !== manifest);
        if (state.log.length === before) continue;
        sound.clock.currentTime += 10;
        const { files, outcome, audible } = state.log[state.log.length - 1];
        return { voiced, files, outcome, audible };
      }
    };
    const controls = () => ({ pressed: byId("callMute").getAttribute("aria-pressed"), slider: byId("callVolume").value, note: shown("callNote"), level: state.master ? state.master.gain.value : null });
    const out = {};
    if (input.duck != null) voice.duck(input.duck);
    await ready(input.duck);
    out.manifest = { duck: state.manifest.duck_seed, voice: state.manifest.voice_seed };
    out.beforeClick = { call: await call(input.draws[0], ["chirp"]), started: sound.started.length, ...controls() };
    fireWindow("click");
    await settle();
    out.afterClick = controls();
    out.calls = [];
    for (const pick of input.draws.slice(1)) out.calls.push(await call(pick, ["chirp"]));
    out.together = await call(0, ["coo"], ["coo"]);
    out.started = sound.started.filter((frames) => frames > 1); // (not the one-sample blank that opens the speaker)
    byId("callMute").fire("click");
    out.muted = { ...controls(), call: await call(0, ["chirp"]) };
    byId("callVolume").value = "50";
    byId("callVolume").fire("input");
    out.half = controls();
    if (input.switch) {
      voice.switching();
      out.switching = { call: await call(0, ["chirp"]), stale: state.stale };
      await get(`${input.base}/api/neuro/duck`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ preset: input.switch.preset }) });
      voice.duck(input.switch.seed);
      await ready(input.switch.seed);
      out.switched = { manifest: { duck: state.manifest.duck_seed, voice: state.manifest.voice_seed }, call: await call(0, ["chirp"]) };
    }
    out.asked = asked;
    out.missing = [...missing];
    return out;
  },

  // app.js: its check of each server answer in input.servers (the line that says the server is another version).
  serverWarn() {
    const check = loadApp()("checkServer");
    return input.servers.map((server) => {
      check(server);
      return shown("serverWarn");
    });
  },

  // app.js: the server answered (input.server), then its ping (input.ping) is asked five times: answered, twice
  // not at all, answered again, answered by another process. Only the ping is answered: drawing a duck's state
  // needs a browser.
  async serverGone() {
    const app = loadApp();
    const asked = [];
    const events = [];
    let ping = input.ping;
    globalThis.dispatchEvent = (event) => {
      events.push(event.type === "duck:net" ? `duck:net ${event.detail.down ? "down" : "up"}` : event.type);
      return true;
    };
    globalThis.fetch = async (url, init = {}) => {
      asked.push(init.body ? `${url} ${JSON.parse(init.body).method}` : String(url));
      if (!ping || url !== "/api/ping") throw new TypeError("Failed to fetch");
      return { ok: true, status: 200, json: async () => ping };
    };
    const buttons = () => byId("bannerActions").querySelectorAll("button");
    const beat = async () => {
      await app("heartbeat")(true);
      await settle();
      return {
        down: body.classList.contains("server-down"),
        warn: shown("serverWarn"),
        banner: byId("bannerTitle").textContent,
        actions: buttons().map((button) => button.textContent),
        asked: asked.splice(0),
        events: events.splice(0),
      };
    };
    // (as the page takes the server part of any answer)
    app("checkServer")(input.server);
    app("syncServer")(input.server);
    events.splice(0);
    const out = { first: await beat() };
    ping = null;
    out.once = await beat();
    out.twice = await beat();
    const [button] = buttons();
    out.still = await beat();
    out.sameButton = buttons()[0] === button;
    ping = input.ping;
    out.back = await beat();
    ping = { ...input.ping, boot: "another-process" };
    out.restarted = await beat();
    return out;
  },
};

// app.js as the page loads it, without its start-up (the first request, the timers): the catalog never says
// it is ready. Returns a reader of the page's own top-level names.
function loadApp() {
  window.I18N.ready = new Promise(() => {});
  runScript("app.js");
  return (name) => vm.runInThisContext(name);
}

const out = await scenarios[scenario]();
console.log(JSON.stringify(out));
process.exit(0);
