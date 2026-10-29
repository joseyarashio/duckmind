// Runs static/i18n.js in Node with a tiny fake page (tests/test_i18n_runtime_js.py).
//   node i18n_node_harness.js <static dir> <cases.json>
// cases.json: {"search": "?lang=…", "stored": "…", "languages": [...], "cases": [{op, lang, …}]}
// Prints one JSON object: {"ready": lang, "results": [...]}.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const [staticDir, casesFile] = process.argv.slice(2);
const spec = JSON.parse(fs.readFileSync(casesFile, "utf8"));

const store = new Map(spec.stored ? [["duckmind.lang", spec.stored]] : []);
const elements = [];
function makeElement(tag) {
  return {
    tagName: String(tag).toUpperCase(),
    dataset: {},
    attrs: {},
    textContent: "",
    innerHTML: "",
    children: [],
    setAttribute(name, value) {
      this.attrs[name] = String(value);
    },
    getAttribute(name) {
      return this.attrs[name];
    },
    matches() {
      return false;
    },
    append(...kids) {
      this.children.push(...kids);
    },
    appendChild(kid) {
      this.children.push(kid);
      return kid;
    },
    addEventListener() {},
    querySelector() {
      return null;
    },
  };
}
const events = [];
const context = {
  console,
  Intl,
  JSON,
  Math,
  Number,
  String,
  Object,
  Array,
  Map,
  Set,
  Promise,
  Error,
  RegExp,
  Date,
  URLSearchParams,
  setTimeout,
  CustomEvent: class {
    constructor(type, init) {
      this.type = type;
      this.detail = init && init.detail;
    }
  },
  navigator: { languages: spec.languages || ["zh-TW"], language: (spec.languages || ["zh-TW"])[0] },
  localStorage: {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
  },
  location: { search: spec.search || "" },
};
context.window = context;
context.document = {
  currentScript: { src: "http://page/i18n.js", dataset: spec.domains ? { domains: spec.domains } : {} },
  documentElement: { lang: "zh-Hant", dataset: {} },
  head: {
    appendChild(el) {
      if (el.tagName !== "SCRIPT") return el;
      setImmediate(() => {
        const name = el.src.replace(/^.*\//, "").replace(/\?.*$/, "");
        const file = path.join(staticDir, name);
        if (fs.existsSync(file)) {
          vm.runInContext(fs.readFileSync(file, "utf8"), context, { filename: name });
          if (el.onload) el.onload();
        } else if (el.onerror) {
          el.onerror();
        }
      });
      return el;
    },
  },
  createElement: makeElement,
  querySelectorAll(selector) {
    if (selector.includes("data-i18n-picker")) return [];
    return elements.filter((el) => el.dataset.i18n || el.dataset.i18nHtml || el.dataset.i18nAttr || el.dataset.i18nField);
  },
  querySelector() {
    return null;
  },
  getElementById() {
    return null;
  },
  dispatchEvent(event) {
    events.push({ type: event.type, detail: event.detail });
  },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(staticDir, "i18n.js"), "utf8"), context, { filename: "i18n.js" });

(async () => {
  const I18N = context.I18N;
  const ready = await I18N.ready;
  const results = [];
  for (const c of spec.cases) {
    try {
      if (c.lang && c.lang !== I18N.lang) await I18N.setLang(c.lang, { remember: !!c.remember });
      let out;
      if (c.op === "t") out = I18N.t(c.key, c.params);
      else if (c.op === "tHtml") out = I18N.tHtml(c.key, c.params);
      else if (c.op === "wire") out = I18N.renderStrict(c.wire, c.code || I18N.lang);
      else if (c.op === "field") out = I18N.field(c.obj, c.name);
      else if (c.op === "items") out = I18N.items(c.obj, c.name);
      else if (c.op === "normalize") out = I18N.normalize(c.tag);
      else if (c.op === "negotiate") out = I18N.negotiate(c.tags);
      else if (c.op === "weekday") out = [0, 1, 2, 3, 4, 5, 6].map((i) => I18N.weekday(i));
      else if (c.op === "number") out = I18N.number(c.value, c.digits);
      else if (c.op === "lang") out = { lang: I18N.lang, htmlLang: context.document.documentElement.lang, stored: store.get("duckmind.lang") || null };
      else if (c.op === "apply") {
        const el = makeElement("p");
        I18N.text(el, c.key, c.params);
        const before = el.textContent;
        elements.push(el);
        await I18N.setLang(c.to);
        out = { before, after: el.textContent };
      } else if (c.op === "show") {
        const el = makeElement("li");
        I18N.show(el, c.obj, c.name);
        const before = el.textContent;
        elements.push(el);
        await I18N.setLang(c.to);
        out = { before, after: el.textContent };
      } else if (c.op === "textAndTitle") {
        const el = makeElement("button");
        I18N.text(el, c.key, c.params);
        I18N.attr(el, "title", c.titleKey);
        elements.push(el);
        await I18N.setLang(c.to);
        out = { text: el.textContent, title: el.attrs.title };
      } else if (c.op === "events") out = events.map((e) => e.type);
      else throw new Error("unknown op " + c.op);
      results.push({ ok: true, out });
    } catch (error) {
      results.push({ ok: false, error: String(error && error.message) });
    }
  }
  process.stdout.write(JSON.stringify({ ready, results, missing: Array.from(I18N.missing).slice(0, 50) }));
})();
