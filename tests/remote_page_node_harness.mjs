// Runs static/remote.js (and drive.js / pointing.js) in Node over a fake page: a fake clock, a fake WebSocket, a fake fetch.
//   node remote_page_node_harness.mjs <static dir> <scenario>
// Prints one JSON value (tests/test_remote_page.py asserts on it).
import { pathToFileURL } from "node:url";
import path from "node:path";

const [staticDir, scenario] = process.argv.slice(2);
const load = (name) => import(pathToFileURL(path.join(staticDir, name)).href);

// ── a fake page ──────────────────────────────────────────────────────────────────────────────────────────

class Target {
  constructor() {
    this.listeners = {};
  }
  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }
  fire(type, event = {}) {
    const e = { type, defaultPrevented: false, stopped: false, preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.stopped = true; }, ...event };
    for (const fn of this.listeners[type] || []) fn(e);
    return e;
  }
}
class El extends Target {
  constructor(id = "", dataset = {}) {
    super();
    this.id = id;
    this.dataset = { ...dataset };
    this.attrs = {};
    this.hidden = false;
    this.textContent = "";
    this.children = [];
  }
  setAttribute(name, value) {
    this.attrs[name] = String(value);
  }
  setPointerCapture() {}
  querySelectorAll() {
    return this.children;
  }
}

function makePage({ sockets = "ok", answers = {} } = {}) {
  let time = 1_000_000;
  const timers = [];
  let nextId = 1;
  const clock = {
    now: () => time,
    advance(ms) {
      const end = time + ms;
      for (;;) {
        const due = timers.filter((t) => t.at <= end).sort((a, b) => a.at - b.at)[0];
        if (!due) break;
        time = due.at;
        if (due.every) due.at += due.every;
        else timers.splice(timers.indexOf(due), 1);
        due.fn();
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

  const buttons = ["left", "forward", "right", "stop"].map((k) => new El("", { remoteKey: k }));
  const els = {
    remoteToggle: new El("remoteToggle"),
    remoteState: new El("remoteState"),
    remotePad: Object.assign(new El("remotePad"), { hidden: true, children: buttons }),
  };
  const doc = Object.assign(new Target(), { hidden: false, getElementById: (id) => els[id] || null });
  const log = { sockets: [], posts: [], sent: [] };

  class FakeSocket {
    constructor(url) {
      this.url = url;
      this.readyState = 0;
      this.sent = [];
      log.sockets.push(this);
      if (sockets === "throw") throw new Error("no sockets here");
      if (sockets === "ok") {
        // opened on the next turn of the fake clock
        fakeTimers.setTimeout(() => {
          if (this.readyState !== 0) return;
          this.readyState = 1;
          this.onopen && this.onopen();
        }, 5);
      } else if (sockets === "refuse") {
        fakeTimers.setTimeout(() => this.onerror && this.onerror(), 5);
      }
      // "silent": never opens, never fails (the connect timeout decides)
    }
    send(text) {
      const message = JSON.parse(text);
      this.sent.push(message);
      log.sent.push(message);
      if ("on" in message && message.on && answers.ws !== false) {
        const answer = answers.ws || { re: "on", ok: true, on: true };
        fakeTimers.setTimeout(() => this.onmessage && this.onmessage({ data: JSON.stringify(answer) }), 3);
      }
    }
    close() {
      this.readyState = 3;
      this.closedByPage = true;
    }
    reply(data) {
      this.onmessage && this.onmessage({ data: JSON.stringify(data) });
    }
    drop() {
      this.readyState = 3;
      this.onclose && this.onclose();
    }
  }
  const fetchFn = (url, init) => {
    const body = init && init.body ? JSON.parse(init.body) : null;
    log.posts.push({ url, body });
    const answer = (answers.post && answers.post[url]) || (url === "/api/neuro/remote" ? { ok: true, on: !!(body && body.on) } : { ok: true, on: true });
    return Promise.resolve({ ok: answer.ok !== false, json: () => Promise.resolve(answer) });
  };
  const win = Object.assign(new Target(), {
    location: { protocol: "http:", host: "duck.local:8790" },
    performance: { timeOrigin: 0 },
    navigator: { getGamepads: () => page.pads },
  });
  const i18n = {
    text(el, key, params) {
      el.dataset.i18n = key;
      el.textContent = params ? `${key}|${JSON.stringify(params)}` : key;
    },
    show(el, obj, name) {
      el.textContent = `field:${obj[name]}`;
    },
  };
  const page = { els, doc, win, log, clock, buttons, pads: [], FakeSocket, fakeTimers, fetchFn, i18n };
  return page;
}

async function boot(options) {
  const { createRemote } = await load("remote.js");
  const page = makePage(options);
  const remote = createRemote({
    document: page.doc,
    window: page.win,
    WebSocketCtor: page.FakeSocket,
    fetchFn: page.fetchFn,
    i18n: page.i18n,
    timers: page.fakeTimers,
    now: page.clock.now,
  });
  return { page, remote, createRemote };
}

const key = (page, type, code, extra = {}) => page.doc.fire(type, { code, key: code, timeStamp: page.clock.now(), target: new El(), ...extra });
const click = (page) => page.els.remoteToggle.fire("click");
const settle = () => new Promise((resolve) => setImmediate(resolve));
const twists = (page) => page.log.sent.filter((m) => "vx" in m);

// ── the scenarios ────────────────────────────────────────────────────────────────────────────────────────

const scenarios = {
  async websocketFlow() {
    const { page, remote } = await boot();
    const out = { start: { mode: remote.mode, pressed: page.els.remoteToggle.attrs["aria-pressed"], padHidden: page.els.remotePad.hidden, state: page.els.remoteState.dataset.i18n } };
    click(page);
    out.connecting = { mode: remote.mode, state: page.els.remoteState.dataset.i18n, url: page.log.sockets[0].url };
    page.clock.advance(5);
    out.afterOpen = page.log.sent.slice();
    page.clock.advance(5);
    out.on = { mode: remote.mode, pressed: page.els.remoteToggle.attrs["aria-pressed"], padHidden: page.els.remotePad.hidden, state: page.els.remoteState.dataset.i18n };
    // hold the up arrow: one twist at once (carrying the key time), more every 50 ms, a zero when let go
    const down = key(page, "keydown", "ArrowUp");
    out.keyPrevented = down.defaultPrevented;
    page.clock.advance(200);
    key(page, "keydown", "ArrowUp"); // auto-repeat: not a new press
    key(page, "keyup", "ArrowUp");
    out.held = twists(page).slice();
    page.clock.advance(1500);
    out.afterRelease = twists(page).slice(out.held.length);
    // Escape: off, the socket told and closed, the pad away
    const esc = key(page, "keydown", "Escape", { code: "Escape" });
    out.escape = { stopped: esc.stopped, mode: remote.mode, padHidden: page.els.remotePad.hidden, sentOff: page.log.sent.at(-1), closed: page.log.sockets[0].closedByPage, pressed: page.els.remoteToggle.attrs["aria-pressed"], state: page.els.remoteState.dataset.i18n };
    return out;
  },

  async keysAreNotTypingAndTurnsAreArcs() {
    const { page, remote } = await boot();
    click(page);
    page.clock.advance(10);
    // typing in a box: not a drive key
    const input = new El();
    input.tagName = "INPUT";
    const typed = key(page, "keydown", "KeyW", { target: input });
    const before = twists(page).length;
    key(page, "keydown", "ArrowLeft"); // left alone: the slowest arc
    key(page, "keydown", "KeyD"); // + right: cancels
    key(page, "keyup", "KeyD");
    key(page, "keyup", "ArrowLeft");
    key(page, "keydown", "ArrowUp");
    key(page, "keydown", "ArrowRight"); // forward + right: an arc, 0.4 m/s
    key(page, "keydown", "ArrowDown"); // the brake wins
    key(page, "keyup", "ArrowUp");
    key(page, "keyup", "ArrowRight");
    key(page, "keyup", "ArrowDown");
    return { typedPrevented: typed.defaultPrevented, typedSent: before, twists: twists(page).map((m) => [m.vx, m.vyaw]), mode: remote.mode };
  },

  async padButtonsAndGamepad() {
    const { page } = await boot();
    click(page);
    page.clock.advance(10);
    const [left, forward] = page.buttons;
    forward.fire("pointerdown", { pointerId: 1, timeStamp: page.clock.now() });
    const held = forward.dataset.held;
    page.clock.advance(100);
    forward.fire("pointerup", { pointerId: 1 });
    const out = { held, released: forward.dataset.held ?? null, twists: twists(page).map((m) => [m.vx, m.vyaw]) };
    // a stick pushed forward walks on the beat; back to the middle is a zero
    const mark = twists(page).length;
    page.pads = [{ connected: true, axes: [0, -1, 0, 0] }];
    page.clock.advance(120);
    page.pads = [{ connected: true, axes: [0.05, 0.1, 0, 0] }];
    page.clock.advance(120);
    out.stick = twists(page).slice(mark).map((m) => [m.vx, m.vyaw]);
    out.leftHeld = left.dataset.held ?? null;
    return out;
  },

  async everyWayOutTellsTheServer() {
    const out = {};
    for (const how of ["hidden", "pagehide", "toggle", "blur"]) {
      const { page, remote } = await boot();
      click(page);
      page.clock.advance(10);
      key(page, "keydown", "ArrowUp");
      if (how === "hidden") {
        page.doc.hidden = true;
        page.doc.fire("visibilitychange");
      } else if (how === "pagehide") page.win.fire("pagehide");
      else if (how === "toggle") click(page);
      else page.win.fire("blur");
      out[how] = { mode: remote.mode, last: page.log.sent.at(-1), closed: !!page.log.sockets[0].closedByPage };
    }
    return out;
  },

  async serverEndsItAndLeaseNotice() {
    const { page, remote } = await boot();
    click(page);
    page.clock.advance(10);
    // a snapshot from just before our own on must not end it
    page.win.fire("duck:telemetry", { detail: { remote: { on: false, ended: null } } });
    const early = remote.mode;
    page.clock.advance(2100);
    page.win.fire("duck:telemetry", { detail: { remote: { on: true, latency_ms: { n: 3, key_to_applied: { p50: 120.4, p95: 150, max: 160 } } } } });
    const shown = page.els.remoteState.textContent;
    page.win.fire("duck:telemetry", { detail: { remote: { on: false, ended: "lease" } } });
    const out = { early, shown, mode: remote.mode, notice: page.els.remoteState.dataset.i18n, closed: page.log.sockets[0].closedByPage, toldServer: page.log.sent.filter((m) => m.on === false).length };
    page.clock.advance(6100);
    out.afterNotice = page.els.remoteState.dataset.i18n;
    return out;
  },

  async socketDropEndsIt() {
    const { page, remote } = await boot();
    click(page);
    page.clock.advance(10);
    page.log.sockets[0].drop();
    return { mode: remote.mode, notice: page.els.remoteState.dataset.i18n };
  },

  async noSocketFallsBackToPost() {
    const out = {};
    for (const mode of ["refuse", "throw", "silent"]) {
      const { page, remote } = await boot({ sockets: mode });
      click(page);
      page.clock.advance(mode === "silent" ? 3100 : 20);
      await settle();
      const first = { mode: remote.mode, posts: page.log.posts.map((p) => [p.url, p.body]) };
      key(page, "keydown", "ArrowUp");
      await settle();
      key(page, "keyup", "ArrowUp");
      await settle();
      await settle();
      const twistPosts = page.log.posts.filter((p) => p.url === "/api/neuro/remote/twist").map((p) => [p.body.vx, p.body.vyaw]);
      click(page);
      await settle();
      out[mode] = { ...first, twistPosts, offPost: page.log.posts.at(-1), after: remote.mode };
    }
    return out;
  },

  async refusedIsAnAnswer() {
    const { page, remote } = await boot({ answers: { ws: { re: "on", ok: false, error: "身體還沒準備好，現在不能遙控" } } });
    click(page);
    page.clock.advance(20);
    return { mode: remote.mode, state: page.els.remoteState.textContent, closed: !!page.log.sockets[0].closedByPage, pressed: page.els.remoteToggle.attrs["aria-pressed"] };
  },

  async latency() {
    const { latencyOf } = await load("remote.js");
    return [
      latencyOf({ latency_ms: { key_to_applied: { p50: 120.4 } } }),
      latencyOf({ latency_ms: { applied: { p50: 80.6 } } }),
      latencyOf({ latency_ms: { n: 0 } }),
      latencyOf(null),
    ];
  },

  async drive() {
    const D = await load("drive.js");
    const set = (...c) => new Set(c);
    return {
      fwd: D.twistOf(D.keyInput(set("ArrowUp"))),
      wasd: D.twistOf(D.keyInput(set("KeyW", "KeyA"))),
      turnAlone: D.twistOf(D.keyInput(set("ArrowLeft"))),
      cancel: D.twistOf(D.keyInput(set("ArrowLeft", "ArrowRight"))),
      brake: D.twistOf(D.keyInput(set("ArrowUp", "KeyS"))),
      none: D.twistOf(D.keyInput(set())),
      stickFull: D.twistOf(D.stickInput(0, -1)),
      stickDead: D.twistOf(D.stickInput(0.15, -0.15)),
      stickRight: D.twistOf(D.stickInput(1, -1)),
      stickBack: D.twistOf(D.stickInput(0, 1)),
      stickNaN: D.twistOf(D.stickInput(NaN, "x")),
      pad: D.twistOf(D.padInput({ forward: true, left: true })),
      merged: D.twistOf(D.mergeInputs(D.keyInput(set("ArrowUp")), D.stickInput(0, -1))),
      typing: [D.typing({ tagName: "input" }), D.typing({ tagName: "BUTTON" }), D.typing({ tagName: "DIV", isContentEditable: true }), D.typing(null)],
      epoch: [D.epochOf(500, 1000, 1600), D.epochOf(5e9, 1000, 1600)],
      range: { vx: [D.WALK_VX_MIN, D.WALK_VX_MAX, D.ARC_VX_MAX], vyaw: D.VYAW_MAX },
    };
  },

  async pointing() {
    const P = await load("pointing.js");
    const out = {};
    // the ray from the scene point (0.5, 1, 1) down toward (0.5, 0, -1): world y = +1, since scene z = −world y
    out.floor = P.floorPoint([0.5, 1, 1], [0, -0.4472136, -0.8944272]);
    out.sky = [P.floorPoint([0, 1, 0], [0, 0.2, -1]), P.floorPoint([0, 1, 0], [0, -1e-6, -1]), P.floorPoint([0, -1, 0], [0, -1, 0]), P.floorPoint([0, 1, 0], [0, -0.01, -1]), P.floorPoint([NaN, 1, 0], [0, -1, 0])];
    out.sceneOfWorld = [P.sceneOfWorld(1.5, 2), P.sceneOfWorld(0, -3)];
    const ray = [[0, 1, 1], [0, -0.6, -0.8]];
    const furniture = { distance: 1, x: 0, y: 0.5, z: 0, kind: "furniture" };
    out.pick = {
      open: P.resolvePick(...ray, { distance: 1.6667, x: 0, y: 0, z: -0.3333, kind: "floor" }, null, true),
      noHomeYet: P.resolvePick(...ray, null, null, false),
      outsideTheHome: P.resolvePick(...ray, null, null, true),
      blocked: P.resolvePick(...ray, furniture, null, true),
      duck: P.resolvePick(...ray, { distance: 1.6667, x: 0, y: 0, z: -0.3333, kind: "floor" }, { distance: 1.2 }, true),
      duckBehind: P.resolvePick(...ray, furniture, { distance: 1.4 }, true),
      rug: P.resolvePick(...ray, { distance: 1.6, x: 0.2, y: 0.01, z: -0.3, kind: "rug" }, null, true),
      step: P.resolvePick(...ray, { distance: 1.6, x: 0.2, y: 0.15, z: -0.3, kind: "stairs" }, null, true),
    };
    out.news = [
      P.errandNews(null, 3),
      P.errandNews({ id: 2, state: "arrived" }, 3),
      P.errandNews({ id: 3, state: "walking" }, 3),
      P.errandNews({ id: 3, state: "gave_up", reason: "stuck" }, 3),
      P.errandNews({ id: 4, state: "walking" }, 3),
      P.errandNews({ id: 3, state: "idle" }, 3),
    ];
    // the errand as the page follows it
    const t = P.makeErrandTrack();
    const seq = [];
    const snap = (at) => seq.push([at, t.marker(at), (t.line(at) || {}).kind ?? null]);
    snap(0);
    t.start(7, 1000);
    snap(1001);
    t.update({ id: 6, state: "arrived" }, 1500); // the telemetry still shows the older errand
    snap(1501);
    t.update({ id: 7, state: "walking" }, 2000);
    snap(2001);
    t.fail("refused", 3000, { reason: "too_far" }); // a refused second click leaves the first errand alone
    snap(3001);
    snap(3000 + P.OUTCOME_MS + 1);
    t.update({ id: 7, state: "gave_up", reason: "stuck" }, 12000);
    snap(12001);
    snap(12000 + P.OUTCOME_MS + 1);
    out.track = seq;
    const quiet = P.makeErrandTrack();
    quiet.start(1, 0);
    out.marker30s = [quiet.marker(P.MARKER_MS - 1), quiet.marker(P.MARKER_MS + 1), quiet.line(P.MARKER_MS + 1)];
    const other = P.makeErrandTrack();
    other.start(1, 0);
    other.update({ id: 2, state: "walking" }, 100); // another page sent the duck elsewhere
    out.replaced = [other.marker(101), other.line(101)];
    // click or drag
    const c = P.makeClickTracker();
    const clicks = [];
    c.down(1, 100, 100, 0, {});
    clicks.push(c.up(1, 102, 101, 120));
    c.down(1, 100, 100, 1000, {});
    c.move(1, 140, 100);
    clicks.push(c.up(1, 100, 100, 1100)); // dragged away and back: still a drag
    c.down(1, 100, 100, 2000, {});
    clicks.push(c.up(1, 100, 100, 2000 + P.CLICK_MS + 1)); // a long press
    c.down(1, 5, 5, 3000, { touch: true });
    c.down(2, 50, 50, 3010, { touch: true }); // a second finger: a pinch
    c.up(2, 50, 50, 3100);
    clicks.push(c.up(1, 5, 5, 3110));
    c.down(1, 7, 7, 4000, { button: 2 });
    clicks.push(c.up(1, 7, 7, 4050)); // the right button
    c.down(1, 8, 8, 5000, { modifier: true });
    clicks.push(c.up(1, 8, 8, 5050));
    c.down(1, 20, 20, 6000, { touch: true });
    clicks.push(c.up(1, 28, 20, 6080)); // a finger moves a little more than a mouse
    out.clicks = clicks;
    return out;
  },
};

const result = await scenarios[scenario]();
process.stdout.write(JSON.stringify(result));
