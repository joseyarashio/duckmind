/* duckmind i18n runtime — the simulator pages in six languages (owner, 2026-09-28).
 * 繁體中文 (zh-Hant) is the default and the source; English, 日本語, Español, 简体中文 and
 * 한국어 are translations. How to add a string in all six: docs/i18n.md.
 *
 * Catalogs are plain static files, i18n.<lang>.<domain>.js, each one call
 * DuckI18n.load(lang, domain, {key: ICU message}); i18n.manifest.js says which exist, so the
 * page never asks for a missing file. Messages use ICU MessageFormat (the same subset as
 * src/duckmind/i18n/icu.py): {name}, {n, number, ::.0}, {n, plural, one {…} other {…}},
 * {n, selectordinal, …}, {x, select, a {…} other {…}}.
 *
 * window.I18N:
 *   t(key, params)            the words of a key in the viewer's language
 *   tHtml(key, params)        the same with its markup kept and the values escaped
 *   field(obj, name)          a server text field: obj[name] in 繁體中文, else its message
 *                             from obj._i18n[name] (falls back to the 繁體中文 words)
 *   items(obj, name)          the same for a list field
 *   text(el, key, params) / html(el, key, params) / attr(el, name, key, params) / show(el, obj, name)
 *                             set an element's words and remember how, so a language switch
 *                             re-renders them by itself
 *   apply(root)               re-render [data-i18n] [data-i18n-html] [data-i18n-attr] elements
 *   number(x, digits) time(date, zhFormat) date(date, opts, zhFormat) weekday(n) relTime(v, unit)
 *   list(items, sepKey)       formats that follow the language (繁體中文 keeps today's exact text)
 *   lang, isSource(), setLang(code), onChange(fn), ready (a Promise), mountPicker(el)
 * The language: ?lang= > the viewer's own choice (localStorage) > navigator.languages > zh-Hant.
 * ?lang=pseudo (tests) wraps every catalog text in ⟦ ⟧: words outside them are hard-coded. */
(function () {
  "use strict";

  const SOURCE = "zh-Hant";
  const PSEUDO = "pseudo";
  const OPEN = "⟦";
  const CLOSE = "⟧";
  // The six languages: code, own name, script, Noto family for CJK text, decimal mark.
  // (src/duckmind/i18n/locales.py holds the same table; a test compares them.)
  const LANGS = [
    { code: "zh-Hant", native: "繁體中文", script: "Hant", font: "Noto Sans TC", decimal: "." },
    { code: "en", native: "English", script: "Latn", font: null, decimal: "." },
    { code: "ja", native: "日本語", script: "Jpan", font: "Noto Sans JP", decimal: "." },
    { code: "es", native: "Español", script: "Latn", font: null, decimal: "," },
    { code: "zh-Hans", native: "简体中文", script: "Hans", font: "Noto Sans SC", decimal: "." },
    { code: "ko", native: "한국어", script: "Kore", font: "Noto Sans KR", decimal: "." },
  ];
  const CODES = LANGS.map((l) => l.code);
  const DOMAINS = ["page", "web", "core", "neuro"];
  const STORE = "duckmind.lang";
  // Each other CJK language's Noto font is a local subset of the glyphs its catalogs use, loaded when
  // that language is chosen (fonts/noto-sans-jp.css …; python -m duckmind.i18n.fonts subset). No CDN.
  const FONT_URL = (family) => base + "fonts/" + family.toLowerCase().replace(/ /g, "-") + ".css";

  const script = document.currentScript;
  const base = script && script.src ? script.src.replace(/[^/]*$/, "") : "/";
  const wantDomains = (script && script.dataset.domains ? script.dataset.domains.split(",") : DOMAINS)
    .map((d) => d.trim())
    .filter((d) => DOMAINS.includes(d));

  const catalogs = {}; // lang → {key → message}
  const pending = {}; // "lang/domain" → Promise
  const waiting = {}; // "lang/domain" → resolve
  const missing = new Set();
  const listeners = [];
  let manifest = null;
  let lang = SOURCE;

  // ------------------------------------------------------------------ which language
  function normalize(tag) {
    if (!tag) return null;
    const parts = String(tag).trim().toLowerCase().split(/[-_]/).filter(Boolean);
    if (!parts.length) return null;
    if (parts[0] === PSEUDO) return PSEUDO;
    const rest = parts.slice(1);
    if (parts[0] === "zh") {
      if (rest.includes("hant") || rest.some((p) => ["tw", "hk", "mo"].includes(p))) return "zh-Hant";
      if (rest.includes("hans") || rest.some((p) => ["cn", "sg", "my"].includes(p))) return "zh-Hans";
      return "zh-Hant";
    }
    return CODES.includes(parts[0]) ? parts[0] : null;
  }
  function negotiate(list) {
    for (const tag of list || []) {
      const code = normalize(tag);
      if (code && code !== PSEUDO) return code;
    }
    return SOURCE;
  }
  function stored() {
    try {
      return normalize(window.localStorage.getItem(STORE));
    } catch (_) {
      return null;
    }
  }
  function remember(code) {
    try {
      window.localStorage.setItem(STORE, code);
    } catch (_) {
      /* private mode: the choice lasts this visit only */
    }
  }
  function fromUrl() {
    try {
      return normalize(new URLSearchParams(window.location.search).get("lang"));
    } catch (_) {
      return null;
    }
  }
  function initial() {
    return fromUrl() || stored() || negotiate(navigator.languages || [navigator.language]);
  }
  const info = (code) => LANGS.find((l) => l.code === (code === PSEUDO ? SOURCE : code)) || LANGS[0];
  const intlLocale = (code) => (code === PSEUDO ? SOURCE : code);
  const catalogLang = (code) => (code === PSEUDO ? SOURCE : code);

  // ------------------------------------------------------------------ catalogs
  function inject(src, onDone) {
    const el = document.createElement("script");
    el.src = src;
    el.async = true;
    el.onload = () => onDone(true);
    el.onerror = () => onDone(false);
    document.head.appendChild(el);
  }
  window.DuckI18n = {
    load(code, domain, messages) {
      catalogs[code] = Object.assign(catalogs[code] || {}, messages || {});
      const key = code + "/" + domain;
      if (waiting[key]) waiting[key](true);
    },
    manifest(data) {
      manifest = data || {};
      if (waiting.manifest) waiting.manifest(true);
    },
  };
  function loadManifest() {
    if (!pending.manifest) {
      pending.manifest = new Promise((resolve) => {
        waiting.manifest = resolve;
        inject(base + "i18n.manifest.js", (ok) => {
          if (!ok || manifest === null) manifest = { [SOURCE]: { page: "" } };
          resolve(true);
        });
      });
    }
    return pending.manifest;
  }
  const available = (code, domain) => !!(manifest && manifest[code] && domain in manifest[code]);
  function loadDomain(code, domain) {
    const key = code + "/" + domain;
    if (!pending[key]) {
      pending[key] = new Promise((resolve) => {
        if (!available(code, domain)) return resolve(false);
        waiting[key] = resolve;
        const version = manifest[code][domain] ? "?v=" + manifest[code][domain] : "";
        const src = base + "i18n." + code + "." + domain + ".js" + version;
        // one quiet retry: a busy proxy can drop one of the parallel loads
        inject(src, (ok) => (ok ? resolve(true) : setTimeout(() => inject(src, resolve), 400)));
      });
    }
    return pending[key];
  }
  async function loadLang(code) {
    await loadManifest();
    const cat = catalogLang(code);
    // the source language needs only the page's own words (server words arrive in it)
    const domains = cat === SOURCE && code !== PSEUDO ? ["page"] : wantDomains;
    const jobs = domains.map((d) => loadDomain(cat, d));
    if (cat !== SOURCE) jobs.push(loadDomain(SOURCE, "page")); // fallback for missing keys
    const results = await Promise.all(jobs);
    return results[0] !== false;
  }

  // ------------------------------------------------------------------ ICU messages
  const parsed = new Map();
  function MessageError(message) {
    const error = new Error(message);
    error.name = "MessageError";
    return error;
  }
  function parse(text) {
    let nodes = parsed.get(text);
    if (!nodes) {
      nodes = new Parser(text).message(0, false);
      if (parsed.size < 50000) parsed.set(text, nodes);
    }
    return nodes;
  }
  const NAME_START = /[A-Za-z_]/;
  const NAME_CHAR = /[A-Za-z0-9_]/;
  class Parser {
    constructor(text) {
      this.text = text;
      this.pos = 0;
    }
    error(what) {
      return MessageError(what + " at " + this.pos + " in " + JSON.stringify(this.text));
    }
    ws() {
      while (this.pos < this.text.length && " \t\r\n".includes(this.text[this.pos])) this.pos++;
    }
    message(depth, inPlural) {
      const nodes = [];
      let buf = "";
      const flush = () => {
        if (buf) nodes.push({ t: "text", v: buf });
        buf = "";
      };
      const text = this.text;
      while (this.pos < text.length) {
        const ch = text[this.pos];
        if (ch === "'") {
          const nxt = text[this.pos + 1] || "";
          if (nxt === "'") {
            buf += "'";
            this.pos += 2;
            continue;
          }
          if (nxt === "{" || nxt === "}" || (inPlural && nxt === "#") || nxt === "|") {
            this.pos++;
            while (this.pos < text.length) {
              const c = text[this.pos];
              if (c === "'") {
                if (text[this.pos + 1] === "'") {
                  buf += "'";
                  this.pos += 2;
                  continue;
                }
                this.pos++;
                break;
              }
              buf += c;
              this.pos++;
            }
            continue;
          }
          buf += "'";
          this.pos++;
          continue;
        }
        if (ch === "{") {
          flush();
          nodes.push(this.argument(depth, inPlural));
          continue;
        }
        if (ch === "}") {
          if (depth === 0) throw this.error("unmatched }");
          flush();
          return nodes;
        }
        if (ch === "#" && inPlural) {
          flush();
          nodes.push({ t: "pound" });
          this.pos++;
          continue;
        }
        buf += ch;
        this.pos++;
      }
      if (depth > 0) throw this.error("unclosed {");
      flush();
      return nodes;
    }
    name() {
      const start = this.pos;
      if (!NAME_START.test(this.text[this.pos] || "")) throw this.error("expected an argument name");
      while (this.pos < this.text.length && NAME_CHAR.test(this.text[this.pos])) this.pos++;
      return this.text.slice(start, this.pos);
    }
    word() {
      const start = this.pos;
      while (this.pos < this.text.length && /[A-Za-z0-9_\-=.]/.test(this.text[this.pos])) this.pos++;
      return this.text.slice(start, this.pos);
    }
    expect(ch) {
      if (this.text[this.pos] !== ch) throw this.error("expected " + ch);
      this.pos++;
    }
    argument(depth, inPlural) {
      this.expect("{");
      this.ws();
      const name = this.name();
      this.ws();
      if (this.text[this.pos] === "}") {
        this.pos++;
        return { t: "arg", name };
      }
      this.expect(",");
      this.ws();
      const kind = this.word();
      if (!["number", "plural", "selectordinal", "select"].includes(kind)) throw this.error("unknown type " + kind);
      this.ws();
      if (kind === "number") {
        if (this.text[this.pos] === "}") {
          this.pos++;
          return { t: "number", name, style: numberStyle(null) };
        }
        this.expect(",");
        const end = this.text.indexOf("}", this.pos);
        if (end < 0) throw this.error("unclosed number style");
        const style = numberStyle(this.text.slice(this.pos, end).trim());
        this.pos = end + 1;
        return { t: "number", name, style };
      }
      this.expect(",");
      let offset = 0;
      const options = {};
      for (;;) {
        this.ws();
        if (this.pos >= this.text.length) throw this.error("unclosed choice");
        if (this.text[this.pos] === "}") {
          this.pos++;
          break;
        }
        if (kind === "plural" && this.text.startsWith("offset:", this.pos)) {
          this.pos += 7;
          this.ws();
          const digits = this.word();
          if (!/^\d+$/.test(digits)) throw this.error("expected a whole number after offset:");
          offset = Number(digits);
          continue;
        }
        const selector = this.word();
        if (!selector) throw this.error("expected a selector");
        this.ws();
        this.expect("{");
        if (selector in options) throw this.error("selector twice: " + selector);
        options[selector] = this.message(depth + 1, inPlural || kind !== "select");
        this.expect("}");
      }
      if (!("other" in options)) throw this.error(kind + " " + name + " needs an other case");
      return { t: kind, name, options, offset };
    }
  }
  function numberStyle(style) {
    const out = { minF: 0, maxF: 3, minI: 1, group: false };
    if (!style) return out;
    if (style === "integer") return { minF: 0, maxF: 0, minI: 1, group: false };
    if (!style.startsWith("::")) throw MessageError("unsupported number style " + style);
    for (const token of style.slice(2).trim().split(/\s+/)) {
      let m;
      if ((m = /^\.(0*)(#*)$/.exec(token)) && (m[1] || m[2])) {
        out.minF = m[1].length;
        out.maxF = m[1].length + m[2].length;
      } else if (token === "precision-integer") {
        out.minF = out.maxF = 0;
      } else if ((m = /^integer-width\/[*+](0+)$/.exec(token))) {
        out.minI = m[1].length;
      } else if (["group-auto", "group-on-aligned", "group-min2"].includes(token)) {
        out.group = true;
      } else if (token === "group-off" || token === "") {
        out.group = false;
      } else {
        throw MessageError("unsupported skeleton token " + token);
      }
    }
    return out;
  }
  const formatters = new Map();
  function formatNumber(value, code, style) {
    const loc = intlLocale(code);
    const key = loc + "|" + style.minF + "|" + style.maxF + "|" + style.minI + "|" + style.group;
    let nf = formatters.get(key);
    if (!nf) {
      nf = new Intl.NumberFormat(loc, {
        minimumFractionDigits: style.minF,
        maximumFractionDigits: style.maxF,
        minimumIntegerDigits: style.minI,
        useGrouping: style.group ? "auto" : false, // "auto": the language's own rule (es groups from 10 000)
      });
      formatters.set(key, nf);
    }
    return nf.format(value);
  }
  function plain(value, code) {
    if (code === SOURCE || code === PSEUDO || Number.isInteger(value)) return String(value);
    const text = String(value);
    return text.includes("e") ? text : text.replace(".", info(code).decimal);
  }
  const pluralRules = new Map();
  function category(code, value, ordinal) {
    const key = intlLocale(code) + (ordinal ? "|o" : "|c");
    let rules = pluralRules.get(key);
    if (!rules) {
      rules = new Intl.PluralRules(intlLocale(code), { type: ordinal ? "ordinal" : "cardinal" });
      pluralRules.set(key, rules);
    }
    return rules.select(Math.abs(value));
  }
  function asNumber(value) {
    if (typeof value === "number") return value;
    if (typeof value === "string" && /^\s*[+-]?\d+(\.\d+)?\s*$/.test(value)) return Number(value);
    return null;
  }
  function formatNodes(nodes, params, code, sub, strict, pound) {
    let out = "";
    for (const node of nodes) {
      if (node.t === "text") {
        out += code === PSEUDO && node.v.trim() ? OPEN + node.v + CLOSE : node.v;
        continue;
      }
      if (node.t === "pound") {
        out += pound === undefined ? "#" : pound;
        continue;
      }
      if (!params || !(node.name in params)) {
        if (strict) throw MessageError("missing value for {" + node.name + "}");
        out += "{" + node.name + "}";
        continue;
      }
      const value = params[node.name];
      if (node.t === "arg") {
        out += typeof value === "number" ? plain(value, code) : sub(value);
      } else if (node.t === "number") {
        const number = asNumber(value);
        out += number === null ? sub(value) : formatNumber(number, code, node.style);
      } else if (node.t === "select") {
        const branch = node.options[String(value)] || node.options.other;
        out += formatNodes(branch, params, code, sub, strict, pound);
      } else {
        let number = asNumber(value);
        if (number === null) {
          if (strict) throw MessageError("{" + node.name + "} needs a number");
          number = 0;
        }
        const shown = number - node.offset;
        const branch =
          node.options["=" + number] || node.options[category(code, shown, node.t === "selectordinal")] || node.options.other;
        out += formatNodes(branch, params, code, sub, strict, formatNumber(shown, code, numberStyle(null)));
      }
    }
    return out;
  }

  // ------------------------------------------------------------------ rendering
  class MissingKey extends Error {}
  function lookup(key, code, strict) {
    const cat = catalogLang(code);
    let message = catalogs[cat] && catalogs[cat][key];
    if (message === undefined && cat !== SOURCE) {
      missing.add(cat + ":" + key);
      if (strict) throw new MissingKey(key);
      message = catalogs[SOURCE] && catalogs[SOURCE][key];
    }
    if (message === undefined) {
      missing.add(SOURCE + ":" + key);
      if (strict) throw new MissingKey(key);
      return null;
    }
    return message;
  }
  // A wire value (see src/duckmind/i18n/message.py): "data", 1.5, {k, p}, {join}, {list, sep}, {cut, text}
  function renderWire(value, code, strict) {
    if (value === null || value === undefined) return "";
    if (typeof value === "string") return value;
    if (typeof value === "number") return plain(value, code);
    if (typeof value === "boolean") return value ? "true" : "false";
    if (Array.isArray(value)) value = { list: value, sep: "fmt.sep.enum" };
    const sub = (inner) => renderWire(inner, code, strict);
    if ("k" in value) {
      const message = lookup(value.k, code, strict);
      if (message === null) return value.k;
      return formatNodes(parse(message), value.p || {}, code, sub, strict);
    }
    if ("join" in value) return value.join.map(sub).join("");
    if ("list" in value) return value.list.map(sub).join(renderWire({ k: value.sep || "fmt.sep.enum" }, code, strict));
    if ("cut" in value) return code === SOURCE && "text" in value ? String(value.text) : sub(value.cut);
    throw MessageError("not a wire value");
  }
  function t(key, params) {
    try {
      return renderWire({ k: key, p: params || {} }, lang, false);
    } catch (error) {
      missing.add("error:" + key);
      return key;
    }
  }
  const escapeHtml = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  function tHtml(key, params) {
    const safe = {};
    for (const [name, value] of Object.entries(params || {})) {
      safe[name] = typeof value === "number" ? value : escapeHtml(typeof value === "string" ? value : renderWire(value, lang, false));
    }
    return t(key, safe);
  }
  function renderIn(wire, fallback) {
    if (!wire || (lang === SOURCE && fallback !== undefined)) return fallback;
    try {
      return renderWire(wire, lang, true);
    } catch (error) {
      return fallback === undefined ? "" : fallback;
    }
  }
  function field(obj, name) {
    if (!obj) return "";
    const words = obj[name];
    const wire = obj._i18n && obj._i18n[name];
    if (lang === SOURCE || !wire || Array.isArray(wire)) return words;
    return renderIn(wire, words);
  }
  function items(obj, name) {
    const list = (obj && obj[name]) || [];
    const wires = obj && obj._i18n && obj._i18n[name];
    if (lang === SOURCE || !Array.isArray(wires)) return list.slice();
    return list.map((words, index) => (wires[index] ? renderIn(wires[index], words) : words));
  }

  // ------------------------------------------------------------------ the DOM
  const params = (el) => {
    try {
      return el.dataset.i18nParams ? JSON.parse(el.dataset.i18nParams) : {};
    } catch (_) {
      return {};
    }
  };
  function applyOne(el) {
    const p = params(el);
    if (el.dataset.i18n) el.textContent = t(el.dataset.i18n, p);
    if (el.dataset.i18nHtml) el.innerHTML = tHtml(el.dataset.i18nHtml, p);
    if (el.dataset.i18nAttr) {
      for (const pair of el.dataset.i18nAttr.split(";")) {
        const at = pair.indexOf(":");
        if (at > 0) el.setAttribute(pair.slice(0, at).trim(), t(pair.slice(at + 1).trim(), p));
      }
    }
    if (el._i18nField) el.textContent = field(el._i18nField.obj, el._i18nField.name);
  }
  function apply(root) {
    const scope = root || document;
    const all = scope.querySelectorAll("[data-i18n],[data-i18n-html],[data-i18n-attr],[data-i18n-field]");
    for (const el of all) applyOne(el);
    if (scope.nodeType === 1 && scope.matches("[data-i18n],[data-i18n-html],[data-i18n-attr],[data-i18n-field]")) applyOne(scope);
  }
  // one set of values per element (merged: a text and a tooltip on the same element share it)
  const setParams = (el, p) => {
    if (p && Object.keys(p).length) el.dataset.i18nParams = JSON.stringify(Object.assign(params(el), p));
  };
  const forgetField = (el) => {
    delete el._i18nField;
    delete el.dataset.i18nField;
  };
  function text(el, key, p) {
    if (!el) return;
    forgetField(el);
    delete el.dataset.i18nHtml;
    el.dataset.i18n = key;
    setParams(el, p);
    el.textContent = t(key, p);
  }
  function html(el, key, p) {
    if (!el) return;
    forgetField(el);
    delete el.dataset.i18n;
    el.dataset.i18nHtml = key;
    setParams(el, p);
    el.innerHTML = tHtml(key, p);
  }
  function attr(el, name, key, p) {
    if (!el) return;
    const pairs = (el.dataset.i18nAttr || "").split(";").filter((x) => x && !x.startsWith(name + ":"));
    pairs.push(name + ":" + key);
    el.dataset.i18nAttr = pairs.join(";");
    setParams(el, p);
    el.setAttribute(name, t(key, p));
  }
  function show(el, obj, name) {
    if (!el) return;
    delete el.dataset.i18n;
    delete el.dataset.i18nHtml;
    el._i18nField = { obj, name };
    el.dataset.i18nField = name;
    el.textContent = field(obj, name) ?? "";
  }
  function loadFont(code) {
    const family = info(code).font;
    if (!family || family === "Noto Sans TC" || document.querySelector('link[data-i18n-font="' + code + '"]')) return;
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = FONT_URL(family);
    link.dataset.i18nFont = code;
    document.head.appendChild(link);
  }

  // ------------------------------------------------------------------ switching
  let switching = Promise.resolve();
  function setLang(code, options) {
    const next = normalize(code) || SOURCE;
    const opts = options || {};
    // a failed switch never blocks the next one (nor the page's start: ready never rejects)
    switching = switching.catch(() => lang).then(async () => {
      const ok = await loadLang(next).catch(() => false);
      const was = lang;
      lang = ok || next === SOURCE ? next : SOURCE;
      const root = document.documentElement;
      root.lang = lang === PSEUDO ? SOURCE : lang;
      if (lang === PSEUDO) root.dataset.i18nPseudo = "";
      else delete root.dataset.i18nPseudo;
      loadFont(lang);
      if (opts.remember && lang !== PSEUDO) remember(lang);
      if (lang !== was || opts.force) apply(document);
      syncPickers();
      if (!ok && next !== SOURCE && opts.remember) note(t("page.i18n.load_failed"));
      if (lang !== was) {
        for (const fn of listeners.slice()) {
          try {
            fn(lang, was);
          } catch (error) {
            setTimeout(() => {
              throw error;
            });
          }
        }
        document.dispatchEvent(new CustomEvent("i18n:change", { detail: { lang, was } }));
      }
      return lang;
    });
    return switching;
  }
  function note(words) {
    // the page shows it (app.js toasts "i18n:note"); nothing is written to the console
    document.dispatchEvent(new CustomEvent("i18n:note", { detail: { text: words } }));
  }
  function onChange(fn) {
    listeners.push(fn);
    return () => {
      const at = listeners.indexOf(fn);
      if (at >= 0) listeners.splice(at, 1);
    };
  }

  // ------------------------------------------------------------------ the language picker
  const pickers = [];
  function syncPickers() {
    for (const select of pickers) {
      select.value = lang === PSEUDO ? SOURCE : lang;
      for (const option of select.options) {
        const ready = available(option.value, "page");
        option.disabled = !ready && option.value !== SOURCE;
      }
      // the word in this language, plus "Language" so a lost viewer can still find it
      const label = t("page.i18n.picker.label");
      select.setAttribute("aria-label", lang === "en" ? label : label + " · Language");
      select.title = t("page.i18n.picker.title");
    }
  }
  function mountPicker(host) {
    if (!host || host.querySelector("select.lang-select")) return;
    const label = document.createElement("label");
    label.className = "lang-picker";
    const icon = document.createElement("span");
    icon.className = "lang-picker-icon";
    icon.setAttribute("aria-hidden", "true");
    const select = document.createElement("select");
    select.className = "lang-select";
    for (const l of LANGS) {
      const option = document.createElement("option");
      option.value = l.code;
      option.lang = l.code;
      option.textContent = l.native;
      select.appendChild(option);
    }
    select.addEventListener("change", () => setLang(select.value, { remember: true }));
    label.append(icon, select);
    host.appendChild(label);
    pickers.push(select);
    syncPickers();
  }

  // ------------------------------------------------------------------ formats
  function number(value, digits) {
    const style = digits === undefined ? numberStyle(null) : { minF: digits, maxF: digits, minI: 1, group: false };
    return lang === SOURCE && digits !== undefined ? Number(value).toFixed(digits) : formatNumber(Number(value), lang, style);
  }
  const pad = (n) => String(n).padStart(2, "0");
  function toDate(value) {
    if (value instanceof Date) return value;
    if (typeof value === "string" && /^\d{1,2}:\d{2}$/.test(value)) {
      const [h, m] = value.split(":").map(Number);
      const d = new Date(2000, 0, 3, h, m);
      return d;
    }
    return new Date(value);
  }
  function time(value, zhFormat) {
    const d = toDate(value);
    if (lang === SOURCE || lang === PSEUDO) return zhFormat ? zhFormat(d) : pad(d.getHours()) + ":" + pad(d.getMinutes());
    return new Intl.DateTimeFormat(lang, { hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(d);
  }
  function date(value, options, zhFormat) {
    const d = toDate(value);
    if ((lang === SOURCE || lang === PSEUDO) && zhFormat) return zhFormat(d);
    return new Intl.DateTimeFormat(intlLocale(lang), options || { month: "2-digit", day: "2-digit" }).format(d);
  }
  function weekday(index, style) {
    // 0 = Monday … 6 = Sunday (the engine's order); 2000-01-03 was a Monday
    const d = new Date(2000, 0, 3 + (((Number(index) % 7) + 7) % 7));
    // 繁體中文: the one character the page writes after 星期 today (CLDR's narrow weekday)
    if (lang === SOURCE || lang === PSEUDO) return new Intl.DateTimeFormat(SOURCE, { weekday: "narrow" }).format(d);
    return new Intl.DateTimeFormat(lang, { weekday: style || "short" }).format(d);
  }
  function relTime(value, unit) {
    return new Intl.RelativeTimeFormat(intlLocale(lang), { numeric: "auto" }).format(value, unit);
  }
  function list(values, sepKey) {
    return values.join(t(sepKey || "fmt.sep.enum"));
  }

  // ------------------------------------------------------------------ start
  const I18N = {
    SOURCE,
    PSEUDO,
    LANGS,
    get lang() {
      return lang;
    },
    isSource: () => lang === SOURCE,
    normalize,
    negotiate,
    t,
    tHtml,
    field,
    items,
    render: (wire) => renderWire(wire, lang, false),
    renderStrict: (wire, code) => renderWire(wire, code || lang, true),
    text,
    html,
    attr,
    show,
    apply,
    setLang,
    onChange,
    mountPicker,
    number,
    time,
    date,
    weekday,
    relTime,
    list,
    missing,
    catalogs,
    parse,
  };
  window.I18N = I18N;
  const first = initial();
  I18N.ready = loadManifest()
    .then(() => setLang(first, { force: first !== SOURCE }))
    .catch(() => lang)
    .then(() => {
      document.querySelectorAll("[data-i18n-picker]").forEach(mountPicker);
      return lang;
    })
    .catch(() => lang);
})();
