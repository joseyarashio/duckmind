// Runs static/growth.js in Node over a tiny fake page, on views the server really makes
// (tests/test_growth_page.py writes them). Prints one JSON line with what the component drew.
"use strict";
const fs = require("fs");
const path = require("path");

class Text {
  constructor(text) {
    this.nodeType = 3;
    this.text = String(text);
  }
}
class El {
  constructor(tag) {
    this.nodeType = 1;
    this.tag = tag;
    this.attrs = {};
    this.children = [];
    this.listeners = {};
    this.className = "";
  }
  setAttribute(name, value) {
    this.attrs[name] = String(value);
  }
  append(...kids) {
    for (const kid of kids) this.children.push(typeof kid === "string" ? new Text(kid) : kid);
  }
  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }
  set textContent(value) {
    this.children = value === "" ? [] : [new Text(value)];
  }
  get textContent() {
    return this.children.map((c) => (c.nodeType === 3 ? c.text : c.textContent)).join("");
  }
}

const words = (key, values) => (values ? `${key}|${JSON.stringify(values)}` : key);
const calls = [];
global.document = {
  createElement: (tag) => new El(tag),
  createTextNode: (text) => new Text(text),
  getElementById: () => null,
  readyState: "complete",
  addEventListener() {},
  body: { classList: { contains: () => false } },
  hidden: false,
};
global.window = {
  I18N: {
    t: words,
    field: (obj, name) => obj[name],
    number: (x) => String(x),
    date: (d, options, zh) => zh(d),
    list: (items) => items.join("+"),
    onChange() {},
  },
  duckVoice: {
    call(tags) {
      calls.push(tags);
      return window.duckVoice.ready;
    },
    refresh() {
      calls.push("refresh");
    },
    ready: true,
  },
  addEventListener() {},
};

const api = require(path.join(process.argv[2], "growth.js"));
const input = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));

function walk(node, visit) {
  visit(node);
  for (const child of node.children || []) walk(child, visit);
}
function describe(tree) {
  const texts = [];
  const buttons = [];
  const classes = new Set();
  walk(tree, (node) => {
    if (node.nodeType === 3) texts.push(node.text);
    else {
      for (const c of String(node.className || "").split(/\s+/)) if (c) classes.add(c);
      if (node.tag === "button") buttons.push(node);
    }
  });
  return { text: texts.join("\n"), buttons, classes: [...classes] };
}

const out = {};
const live = describe(api.renderView(input.live, ""));
out.live = { text: live.text, classes: live.classes, buttons: live.buttons.map((b) => b.textContent) };
// press the play buttons: the page's voice player is asked for the clip; when it cannot, it says why
for (const button of live.buttons) for (const fn of button.listeners.click || []) fn();
out.played = calls.slice();
window.duckVoice.ready = false;
calls.length = 0;
const tree = api.renderView(input.live, "");
for (const button of describe(tree).buttons) for (const fn of button.listeners.click || []) fn();
out.refused = calls.slice();
out.note = describe(tree).text.includes("page.growth.voice.play_wait");

const test = describe(api.renderView(input.test, "rough"));
out.test = { text: test.text, buttons: test.buttons.length, classes: test.classes };
out.rows = api.compareRows([input.live, input.test]);
out.rows_same = api.compareRows([input.live, input.live]).map((r) => r.differs);
out.eta = [api.etaSeconds(7, 5, 30), api.etaSeconds(0, 0, 30), api.etaSeconds(100, 30, 30), api.etaSeconds(50, 15, 30)];
out.regrow_line = api.regrowLine(input.live, input.live.regrow.last, "WHEN");
out.keys = {
  kinds: Object.keys(api.KIND_KEYS),
  waits: Object.keys(api.WAIT_KEYS),
  styles: Object.keys(api.STYLE_KEYS),
  how: Object.keys(api.HOW_KEYS),
  falls: Object.keys(api.FALL_KEYS),
};
console.log(JSON.stringify(out));
