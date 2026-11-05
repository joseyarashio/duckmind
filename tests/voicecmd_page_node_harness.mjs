// Runs static/voicecmd.js in Node over a fake page: a fake clock, a fake fetch, a fake microphone, a tiny fake DOM.
//   node voicecmd_page_node_harness.mjs <static dir> <scenario> [input json]
// Prints one JSON value (tests/test_voice_commands_page.py asserts on it).
import { pathToFileURL } from "node:url";
import path from "node:path";

const [staticDir, scenario, input] = process.argv.slice(2);
const mod = await import(pathToFileURL(path.join(staticDir, "voicecmd.js")).href);
const given = input ? JSON.parse(input) : {};

class El {
  constructor(id = "") {
    this.id = id;
    this.dataset = {};
    this.attrs = {};
    this.hidden = false;
    this.disabled = false;
    this.title = "";
    this.textContent = "";
    this.listeners = {};
    this.label = null;
  }
  setAttribute(name, value) {
    this.attrs[name] = String(value);
  }
  getAttribute(name) {
    return name in this.attrs ? this.attrs[name] : null;
  }
  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }
  querySelector(selector) {
    return selector === "[data-voice-label]" ? this.label : null;
  }
  fire(type, extra = {}) {
    const event = { type, button: 0, pointerId: 1, repeat: false, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...extra };
    for (const fn of this.listeners[type] || []) fn(event);
    return event;
  }
}

function makePage({ status, answers = [], mic = {} }) {
  let time = 1000;
  let timers = [];
  let nextId = 1;
  const settle = () => new Promise((resolve) => setImmediate(resolve));
  const clock = {
    async advance(ms) {
      const until = time + ms;
      for (;;) {
        await settle();
        timers.sort((a, b) => a.at - b.at);
        const due = timers.find((t) => t.at <= until);
        if (!due) break;
        timers = timers.filter((t) => t !== due);
        time = due.at;
        due.fn();
      }
      time = until;
      await settle();
    },
  };
  const t = {
    setTimeout(fn, ms) {
      const id = nextId++;
      timers.push({ id, at: time + Math.max(0, ms || 0), fn });
      return id;
    },
    clearTimeout(id) {
      timers = timers.filter((x) => x.id !== id);
    },
  };
  const button = new El("cmdMic");
  button.label = new El();
  const out = new El("cmdVoiceOut");
  out.hidden = true;
  const els = { cmdMic: button, cmdVoiceOut: out };
  const document = { getElementById: (id) => els[id] || null };
  const posts = [];
  const queue = [...answers];
  const fetchFn = async (url, opts = {}) => {
    if (url === "/api/neuro/voice/status") return { status: 200, json: async () => status };
    const body = opts.body;
    posts.push({ url, method: opts.method, type: (opts.headers || {})["Content-Type"], bytes: body ? body.byteLength : 0, at: time });
    if (mic.answerDelayMs) time += mic.answerDelayMs;
    const next = queue.shift() || { status: 500, data: null };
    return { status: next.status, json: async () => next.data };
  };
  const micLog = { opened: 0, closed: 0 };
  let feed = null;
  const openMic = async () => {
    micLog.opened += 1;
    if (mic.fail) throw Object.assign(new Error(mic.fail), { kind: mic.fail });
    if (mic.openDelayMs) await new Promise((resolve) => t.setTimeout(resolve, mic.openDelayMs));
    return {
      rate: mic.rate || 48000,
      onBlock(fn) {
        feed = fn;
      },
      close() {
        micLog.closed += 1;
        feed = null;
      },
    };
  };
  const speak = (seconds, rate = mic.rate || 48000) => {
    const block = 2400;
    const n = Math.round(seconds * rate);
    for (let at = 0; at < n; at += block) {
      const b = new Float32Array(Math.min(block, n - at));
      for (let i = 0; i < b.length; i += 1) b[i] = 0.3 * Math.sin((2 * Math.PI * 220 * (at + i)) / rate);
      if (feed) feed(b);
    }
  };
  const shown = { commands: [], feedback: [] };
  const i18n = { t: (key, values) => (values && Object.keys(values).length ? `${key}${JSON.stringify(values)}` : key), onChange() {} };
  const voice = mod.createVoiceCommands({
    document,
    fetchFn,
    i18n,
    timers: t,
    openMic,
    now: () => time,
    lang: () => given.lang || "zh-Hant",
    showCommand: (answer, verb) => shown.commands.push({ answer, verb }),
    showFeedback: (kind, st, answer) => shown.feedback.push({ kind, status: st, answer }),
  });
  const view = () => ({
    phase: voice.phase,
    disabled: button.disabled,
    state: button.dataset.state,
    pressed: button.attrs["aria-pressed"],
    title: button.title,
    titleHook: button.attrs["data-i18n-attr"],
    label: button.label.textContent,
    line: out.hidden ? null : out.textContent,
    tone: out.dataset.state,
  });
  return { voice, button, clock, posts, micLog, speak, shown, view };
}

const ON = { ok: true, enabled: true, running: true, ready: true, engine: "mlx", model: "m", problem: null };
const HEARD = (text, understood, answer) => ({ status: 200, data: { ok: true, heard: { text, lang: "zh-Hant", audio_s: 1.2, took_s: 0.4, empty: text ? null : "no_speech" }, understood, answer, took_s: 0.42 } });

const out = {};
if (scenario === "pcm") {
  const rate = 48000;
  const sine = new Float32Array(rate);
  for (let i = 0; i < rate; i += 1) sine[i] = 0.5 * Math.sin((2 * Math.PI * 440 * i) / rate);
  const pcm = mod.toPcm16([sine.subarray(0, 20000), sine.subarray(20000)], rate);
  let peak = 0;
  for (const v of pcm) peak = Math.max(peak, Math.abs(v));
  const same = mod.toPcm16([new Float32Array([0, 1, -1, 2])], 16000);
  Object.assign(out, { length: pcm.length, peak, same: Array.from(same) });
} else if (scenario === "globals") {
  // the page's own start-up: voice.js has already put the duck's calls (and its beak, `mouth`) on window.duckVoice;
  // voicecmd.js starts on the same page and must leave that alone
  const button = new El("cmdMic");
  const els = { cmdMic: button, cmdVoiceOut: new El("cmdVoiceOut") };
  const calls = { mouth: () => 0, call: () => true };
  globalThis.window = {
    duckVoice: calls,
    fetch: async () => ({ status: 503, json: async () => ({ ok: false }) }),
    setTimeout: () => 0,
    clearTimeout: () => {},
    performance: { now: () => 0 },
    navigator: {},
  };
  globalThis.document = { getElementById: (id) => els[id] || null };
  await import(`${pathToFileURL(path.join(staticDir, "voicecmd.js")).href}?page`);
  await new Promise((resolve) => setImmediate(resolve));
  Object.assign(out, {
    calls_kept: window.duckVoice === calls && typeof window.duckVoice.mouth === "function",
    push_to_talk: !!(window.duckVoiceCmd && typeof window.duckVoiceCmd.press === "function"),
  });
} else if (scenario === "lines") {
  const cases = given.cases || [];
  out.lines = cases.map(([status, data, ms]) => mod.heardLine(status, data, ms));
} else {
  const status = given.status || ON;
  const page = makePage({ status, answers: given.answers || [], mic: given.mic || {} });
  const { voice, button, clock, posts, micLog, speak, shown, view } = page;
  await voice.readStatus();
  out.start = view();
  if (scenario === "flow" || scenario === "feedback" || scenario === "keys") {
    if (scenario === "keys") button.fire("keydown", { key: " " });
    else button.fire("pointerdown");
    await clock.advance(10);
    out.held = view();
    speak(given.seconds || 1.0);
    if (scenario === "keys") button.fire("keyup", { key: " " });
    else button.fire("pointerup");
    button.fire("lostpointercapture"); // (the browser says this too: one release only)
    out.thinking = view();
    await clock.advance(mod.TAIL_MS + 5);
    await clock.advance(10);
  } else if (scenario === "short") {
    page.voice.press();
    page.voice.release(); // (let go before the microphone opened)
    await clock.advance(50);
  } else if (scenario === "tap") {
    button.fire("pointerdown");
    await clock.advance(10);
    speak(0.1);
    button.fire("pointerup");
    await clock.advance(mod.TAIL_MS + 20);
  } else if (scenario === "denied") {
    button.fire("pointerdown");
    await clock.advance(10);
    button.fire("pointerup");
    await clock.advance(10);
  } else if (scenario === "cap") {
    button.fire("pointerdown");
    await clock.advance(10);
    speak(given.seconds || 12);
    await clock.advance(mod.MAX_MS + 500);
  }
  Object.assign(out, { end: view(), posts, mic: micLog, shown });
}
process.stdout.write(JSON.stringify(out));
