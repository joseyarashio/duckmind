// Runs static/realduck.js (its page part) and static/neuro_sound.js in Node over a small fake page, to see
// whether the sound scenarios' strip — built by neuro_sound.js whenever it likes — is held off on a real duck.
//   node realduck_sound_node_harness.mjs <static dir> <order> <real json>
// order "real-first": the page knows it is a real duck before the strip is built; "strip-first": after.
// Prints {control: [disabled, hidden, gate]} for the strip's buttons and the hand slider (JSON).
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [staticDir, order, realJson] = process.argv.slice(2);
const registry = new Map();
const camel = (name) => name.replace(/-([a-z])/g, (_, c) => c.toUpperCase());

class El {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.parent = null;
    this.dataset = {};
    this.attrs = {};
    this.hidden = false;
    this.disabled = false;
    this.style = {};
    this.classList = { toggle() {}, add() {}, remove() {}, contains: () => false };
  }
  set id(v) {
    this._id = v;
    if (v) registry.set(v, this);
  }
  get id() {
    return this._id || "";
  }
  set title(v) {
    this.attrs.title = String(v);
  }
  get title() {
    return this.attrs.title || "";
  }
  append(...items) {
    for (const item of items) {
      if (item instanceof El) item.parent = this;
      this.children.push(item);
    }
  }
  replaceChildren(...items) {
    this.children = [];
    this.append(...items);
  }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this);
  }
  setAttribute(name, value) {
    if (name.startsWith("data-")) this.dataset[camel(name.slice(5))] = String(value);
    else this.attrs[name] = String(value);
  }
  getAttribute(name) {
    if (name.startsWith("data-")) return this.dataset[camel(name.slice(5))] ?? null;
    return name in this.attrs ? this.attrs[name] : null;
  }
  removeAttribute(name) {
    delete this.attrs[name];
  }
  addEventListener() {}
  closest() {
    return null;
  }
  getClientRects() {
    return [];
  }
  all() {
    const out = [];
    const walk = (el) => {
      for (const c of el.children) {
        if (!(c instanceof El)) continue;
        out.push(c);
        walk(c);
      }
    };
    walk(this);
    return out;
  }
  querySelectorAll(selector) {
    return this.all().filter((el) => el.matches(selector));
  }
  // The few selector shapes realduck.js uses: [tag][#id][[attr]|[attr="v"]], descendant chains, lists.
  matches(selector) {
    const one = (el, compound) => {
      const m = /^([a-z]+)?(?:#([\w-]+))?(?:\[([\w-]+)(?:="([^"]*)")?\])?$/.exec(compound);
      if (!m) throw new Error(`selector not understood: ${compound}`);
      const [, tag, id, attr, value] = m;
      if (tag && el.tagName !== tag.toUpperCase()) return false;
      if (id && el.id !== id) return false;
      if (attr) {
        const got = el.getAttribute(attr);
        if (got === null || (value !== undefined && got !== value)) return false;
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

const body = new El("body");
const mount = new El("div");
mount.id = "neuroSoundMount";
const stim = new El("div");
stim.id = "neuroStim";
const note = new El("p");
note.id = "realNoteSound";
note.setAttribute("data-real-note", "");
note.hidden = true;
body.append(stim, mount, note);

globalThis.window = globalThis;
globalThis.document = {
  body,
  activeElement: null,
  getElementById: (id) => registry.get(id) || null,
  createElement: (tag) => new El(tag),
  querySelectorAll: (selector) => body.querySelectorAll(selector),
  addEventListener() {},
};
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.dispatchEvent = () => true;
globalThis.localStorage = { getItem: () => null, setItem() {} };
globalThis.requestAnimationFrame = () => 0; // (the playout clock never ticks here)
globalThis.I18N = {
  ready: Promise.resolve(),
  lang: "zh-Hant",
  t: (key) => `T:${key}`,
  text(el, key) {
    el.textContent = `T:${key}`;
  },
  attr(el, name, key) {
    el.setAttribute(name, `T:${key}`);
  },
  onChange() {},
};
globalThis.fetch = async (url) =>
  String(url).includes("/api/neuro/sound/status")
    ? { ok: true, status: 200, json: async () => ({}) }
    : { ok: false, status: 404, body: null, json: async () => ({}) };

// realduck.js is a classic script: its page part runs where there is no `module` (as in the page).
(0, eval)(fs.readFileSync(path.join(staticDir, "realduck.js"), "utf-8"));
const real = JSON.parse(realJson);
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const built = async () => {
  for (let i = 0; i < 100 && !registry.get("neuroSound"); i++) await wait(20);
};

if (order === "real-first") window.duckReal.take(real);
await import(pathToFileURL(path.join(staticDir, "neuro_sound.js")).href);
await built();
if (order === "strip-first") window.duckReal.take(real);
await wait(50);

const out = {};
for (const el of registry.get("neuroSound").querySelectorAll("button, input")) {
  out[el.id || `${el.dataset.sound}${el.dataset.tempo ? `:${el.dataset.tempo}` : ""}`] = [el.disabled, el.hidden, el.dataset.realGate || null];
}
out._note = [note.hidden];
console.log(JSON.stringify(out));
process.exit(0);
