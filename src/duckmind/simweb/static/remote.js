/* The remote hand, page side (docs/design/neural-duck/REMOTE-GOTHERE-2026-10-23.md §1).
 *
 * The toggle in the command strip hands the duck's legs to the person: while it is on, the arrow keys / WASD,
 * a gamepad's left stick and the four on-screen buttons become a twist (drive.js) that goes to the server
 * about 20 times a second while held, and a zero the moment the hand lets go. The duck's brain keeps its head,
 * voice and senses; only the legs are the person's. The server clamps every twist again and has the last word:
 * a 300 ms deadman (a twist not renewed is zero) and a 5 s lease (no message at all ends remote mode).
 *
 * Way in: a WebSocket (`/api/neuro/remote/ws`: `{"on": bool}`, `{"vx","vyaw","t"}`; one connection carries the
 * twists, and closing it ends remote mode at once). If the socket cannot be made, `POST /api/neuro/remote` and
 * `POST /api/neuro/remote/twist` do the same, one request at a time. `t` is the key event's epoch ms, sent only
 * with the twist that starts a walk (the server times key → legs from it).
 *
 * Remote mode ends here on: the toggle, Esc, the page being hidden or closed, and the telemetry saying the
 * server ended it (lease, a closed socket, another page). The status line says on or off and the typical
 * delay the telemetry measured (`remote.latency_ms`). Words come from the catalogs (`page.remote.*`).
 *
 * `createRemote(env)` takes everything it touches, so tests run it in node over fakes; the last lines start it
 * on the real page. */
import { DRIVE_KEYS, NO_INPUT, epochOf, isZero, keyInput, mergeInputs, padInput, stickInput, twistOf, typing } from "./drive.js";

export const SEND_MS = 50; // a twist is renewed this often while one is asked (20 Hz; the deadman is 300 ms)
export const IDLE_MS = 1000; // nothing asked: a zero twist this often keeps the lease (5 s) alive
export const GRACE_MS = 2000; // after switching on, a telemetry snapshot older than this may say it is off
export const CONNECT_MS = 3000; // a socket not open by now: the POST way instead
export const NOTICE_MS = 6000; // how long the status line says why remote mode ended by itself
export const POST_FAILS = 15; // twists in a row that got no answer (POST way): the server is gone
export const NUDGE_MS = 400; // a button pressed with the keyboard walks this long
export const OTHER_MS = 1500; // after our own off, a snapshot saying "on" is still the old one for this long
const WS_OPEN = 1; // WebSocket.OPEN
const WS_PATH = "/api/neuro/remote/ws";
const JSON_HEADERS = { "Content-Type": "application/json" };
const PAD_KEYS = new Set(["forward", "left", "right", "stop"]);

// why remote mode ended by itself (telemetry `remote.ended`) → the notice; the words are catalog keys
const NOTICE = {
  lease: "page.remote.notice.lease",
  lost: "page.remote.notice.lost",
  stopped: "page.remote.notice.stopped",
  failed: "page.remote.notice.failed",
};
const ENDED_AS = { lease: "lease", disconnect: "lost" };
const STATE_KEYS = {
  off: "page.remote.state.off",
  connecting: "page.remote.state.connecting",
  on: "page.remote.state.on",
};

/** The delay to show: the median key → legs-moving delay the telemetry measured, else server → legs (the
 * page's own hop to the server is not in it). Whole milliseconds, or null when nothing was measured yet. */
export function latencyOf(remote) {
  const l = remote && remote.latency_ms;
  const p50 = l && ((l.key_to_applied && l.key_to_applied.p50) ?? (l.applied && l.applied.p50));
  return Number.isFinite(p50) ? Math.round(p50) : null;
}

export function createRemote(env) {
  const { document: doc, window: win, WebSocketCtor, fetchFn, i18n, timers } = env;
  const now = env.now || (() => Date.now());
  const toggle = doc.getElementById("remoteToggle");
  const pad = doc.getElementById("remotePad");
  const stateEl = doc.getElementById("remoteState");
  if (!toggle || !stateEl) return null;
  const padButtons = pad ? Array.from(pad.querySelectorAll("[data-remote-key]")) : [];
  const origin = win.performance && win.performance.timeOrigin;

  const s = {
    mode: "idle", // idle | connecting | on
    via: null, // ws | post
    ws: null,
    attempt: 0, // one per switch-on (a late answer of an old one is dropped)
    onAt: 0,
    offAt: -1e9,
    keys: new Set(), // KeyboardEvent.code held now
    held: { forward: false, left: false, right: false, stop: false }, // the on-screen buttons
    last: { vx: 0, vyaw: 0 }, // the twist asked at the last look
    sentAt: 0,
    edgeAt: null, // epoch ms of the input that started the walk, until a twist carries it
    tick: 0,
    connectTimer: 0,
    notice: null, // {key} or {obj, name}: why remote mode ended by itself, for NOTICE_MS
    noticeTimer: 0,
    tel: null, // telemetry's `remote` block
    post: { busy: false, next: null, fails: 0 },
    shown: "",
  };

  const say = (el, key, params) => {
    if (i18n) i18n.text(el, key, params);
    else el.textContent = key;
  };
  const sayField = (el, obj, name) => {
    if (i18n) i18n.show(el, obj, name);
    else el.textContent = obj[name];
  };

  /* ---------------- what the page shows ---------------- */

  function render() {
    const engaged = s.mode !== "idle";
    toggle.setAttribute("aria-pressed", String(engaged));
    if (pad) pad.hidden = s.mode !== "on";
    // (somebody else has the legs: another page, the API; not the snapshot that is still on its way back from our own off)
    const other = s.mode === "idle" && !!(s.tel && s.tel.on) && now() - s.offAt > OTHER_MS;
    const ms = s.mode === "on" || other ? latencyOf(s.tel) : null;
    const kind = s.notice ? "notice" : s.mode === "idle" && !other ? "off" : s.mode === "connecting" ? "connecting" : "on";
    const key = `${kind}|${ms}|${s.notice ? s.notice.key || s.notice.name : ""}`;
    stateEl.dataset.state = kind;
    if (key === s.shown) return;
    s.shown = key;
    if (s.notice && s.notice.obj) sayField(stateEl, s.notice.obj, s.notice.name);
    else if (s.notice) say(stateEl, s.notice.key);
    else if (kind === "on" && ms != null) say(stateEl, "page.remote.state.on_latency", { ms });
    else say(stateEl, STATE_KEYS[kind]);
  }

  function setNotice(notice) {
    timers.clearTimeout(s.noticeTimer);
    s.notice = notice;
    s.shown = "";
    if (notice) {
      s.noticeTimer = timers.setTimeout(() => {
        s.notice = null;
        render();
      }, NOTICE_MS);
    }
  }

  function setHeld(name, on) {
    s.held[name] = on;
    for (const button of padButtons) {
      if (button.dataset.remoteKey !== name) continue;
      if (on) button.dataset.held = "true";
      else delete button.dataset.held;
    }
  }

  /* ---------------- the twist ---------------- */

  function readSticks() {
    const nav = win.navigator;
    if (!nav || typeof nav.getGamepads !== "function") return NO_INPUT;
    let pads;
    try {
      pads = nav.getGamepads();
    } catch (error) {
      return NO_INPUT;
    }
    const sticks = Array.from(pads || [], (p) => (p && p.connected && p.axes && p.axes.length >= 2 ? stickInput(p.axes[0], p.axes[1]) : NO_INPUT));
    return mergeInputs(...sticks);
  }

  /** Look at every hand now; send the twist when it changed (or, on the 20 Hz beat, while one is asked, and a
   * zero now and then to keep the lease). `eventAt`: epoch ms of the key / press that caused this look. */
  function apply(eventAt, beat = false) {
    if (s.mode !== "on") return;
    const twist = twistOf(mergeInputs(keyInput(s.keys), padInput(s.held), readSticks()));
    const t = now();
    if (isZero(s.last) && !isZero(twist)) s.edgeAt = eventAt ?? t;
    const changed = twist.vx !== s.last.vx || twist.vyaw !== s.last.vyaw;
    s.last = twist;
    if (changed || (beat && (!isZero(twist) || t - s.sentAt >= IDLE_MS))) send(twist, t);
  }

  function send(twist, t) {
    s.sentAt = t;
    const edge = s.edgeAt;
    s.edgeAt = null;
    if (s.via === "ws") {
      if (!s.ws || s.ws.readyState !== WS_OPEN) return;
      const message = { vx: twist.vx, vyaw: twist.vyaw };
      if (edge != null) message.t = Math.round(edge);
      s.ws.send(JSON.stringify(message));
    } else {
      sendPost({ vx: twist.vx, vyaw: twist.vyaw });
    }
  }

  function postJson(path, body, keepalive = false) {
    // (never rejects: {ok, data}; data is null when no answer came)
    return fetchFn(path, { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body), cache: "no-store", keepalive }).then(
      (response) => response.json().then((data) => ({ ok: response.ok, data }), () => ({ ok: response.ok, data: null })),
      () => ({ ok: false, data: null }),
    );
  }

  function sendPost(payload) {
    // one request at a time; while one is out the newest twist waits (a zero is never dropped)
    const post = s.post;
    if (post.busy) {
      post.next = payload;
      return;
    }
    post.busy = true;
    postJson("/api/neuro/remote/twist", payload).then(({ ok, data }) => {
      post.busy = false;
      post.fails = ok ? 0 : post.fails + 1;
      const next = post.next;
      post.next = null;
      if (s.mode !== "on" || s.via !== "post") return;
      if (post.fails >= POST_FAILS) finish("lost", false);
      else if (ok && data && data.on === false && now() - s.onAt > GRACE_MS) finish("stopped", false);
      else if (next) sendPost(next);
    });
  }

  /* ---------------- on and off ---------------- */

  function start() {
    s.attempt += 1;
    s.mode = "connecting";
    s.via = null;
    setNotice(null);
    render();
    open();
  }

  function wsUrl() {
    return `${win.location.protocol === "https:" ? "wss:" : "ws:"}//${win.location.host}${WS_PATH}`;
  }

  function detach(ws) {
    ws.onopen = ws.onmessage = ws.onclose = ws.onerror = null;
  }

  function open() {
    let ws;
    try {
      ws = new WebSocketCtor(wsUrl());
    } catch (error) {
      fallback();
      return;
    }
    s.ws = ws;
    s.via = "ws";
    let opened = false;
    timers.clearTimeout(s.connectTimer);
    s.connectTimer = timers.setTimeout(() => {
      if (s.ws !== ws || s.mode !== "connecting") return;
      if (opened) {
        refused(null); // the socket is up but the server never answered {on:true}
        return;
      }
      s.ws = null;
      detach(ws);
      try {
        ws.close();
      } catch (error) {
        /* gone already */
      }
      fallback();
    }, CONNECT_MS);
    ws.onopen = () => {
      opened = true;
      ws.send(JSON.stringify({ on: true }));
    };
    ws.onmessage = (event) => {
      if (s.ws !== ws || s.mode !== "connecting") return;
      let data = null;
      try {
        data = JSON.parse(event.data);
      } catch (error) {
        return;
      }
      if (!data || data.re !== "on") return;
      if (data.ok && data.on) engage("ws"); // (engage clears the connect timer)
      else refused(data);
    };
    const gone = () => {
      if (s.ws !== ws) return;
      s.ws = null;
      timers.clearTimeout(s.connectTimer);
      if (s.mode === "connecting" && !opened) fallback(); // no socket here (a proxy, an old server): the POST way
      else finish("lost", false);
    };
    ws.onclose = gone;
    ws.onerror = gone;
  }

  function fallback() {
    if (s.mode !== "connecting") return;
    const attempt = s.attempt;
    s.via = "post";
    postJson("/api/neuro/remote", { on: true }).then(({ ok, data }) => {
      const good = ok && data && data.on;
      if (attempt !== s.attempt || s.mode !== "connecting") {
        // switched off while it was being switched on (and not on again since: that one's own on is the same state)
        if (good && s.mode === "idle") postJson("/api/neuro/remote", { on: false }, true);
        return;
      }
      if (good) engage("post");
      else refused(data);
    });
  }

  function refused(data) {
    // the server's own words when it gave any (a refusal is a normal answer), else the plain failure
    const name = data && data.error ? "error" : data && data.reason_zh ? "reason_zh" : null;
    const ws = s.ws;
    s.ws = null;
    if (ws) {
      detach(ws);
      try {
        ws.close();
      } catch (error) {
        /* gone already */
      }
    }
    timers.clearTimeout(s.connectTimer);
    s.mode = "idle";
    setNotice(name ? { obj: data, name } : { key: NOTICE.failed });
    render();
  }

  function engage(via) {
    timers.clearTimeout(s.connectTimer);
    s.mode = "on";
    s.via = via;
    s.onAt = now();
    s.sentAt = s.onAt;
    s.last = { vx: 0, vyaw: 0 };
    s.edgeAt = null;
    s.post = { busy: false, next: null, fails: 0 };
    s.tick = timers.setInterval(() => apply(null, true), SEND_MS);
    render();
    apply(); // (a key or a button already held)
  }

  /** Remote mode is over here. `notice`: the key of NOTICE when it ended by itself (null: the person ended it).
   * `tellServer`: say so (`{on:false}`, then close); false when the server is the one that ended it. */
  function finish(notice, tellServer = true) {
    if (s.mode === "idle") return;
    const was = s.mode;
    const { ws, via } = s;
    timers.clearTimeout(s.connectTimer);
    timers.clearInterval(s.tick);
    s.tick = 0;
    s.mode = "idle";
    s.offAt = now();
    s.attempt += 1;
    s.keys.clear();
    for (const name of PAD_KEYS) setHeld(name, false);
    s.last = { vx: 0, vyaw: 0 };
    s.edgeAt = null;
    s.ws = null;
    if (ws) {
      detach(ws);
      if (tellServer && ws.readyState === WS_OPEN) ws.send(JSON.stringify({ on: false }));
      try {
        ws.close(1000);
      } catch (error) {
        /* gone already */
      }
    } else if (tellServer && via === "post" && was === "on") {
      postJson("/api/neuro/remote", { on: false }, true);
    }
    setNotice(notice ? { key: NOTICE[notice] } : null);
    render();
  }

  function onTelemetry(t) {
    s.tel = (t && t.remote) || null;
    if (s.mode === "on" && s.tel && s.tel.on === false && now() - s.onAt > GRACE_MS) {
      finish(ENDED_AS[s.tel.ended] || "stopped", false);
      return;
    }
    render();
  }

  /* ---------------- the hands ---------------- */

  // Esc hands the duck back first (capture phase: it does not also close the enlarged view or the panel)
  doc.addEventListener(
    "keydown",
    (event) => {
      if (event.key !== "Escape" || s.mode === "idle") return;
      event.stopPropagation();
      finish(null);
    },
    true,
  );
  doc.addEventListener("keydown", (event) => {
    if (s.mode !== "on" || !DRIVE_KEYS.has(event.code)) return;
    // (a key somebody else took, a shortcut, or typing in a box is not a drive key)
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || typing(event.target)) return;
    event.preventDefault();
    if (s.keys.has(event.code)) return; // (auto-repeat)
    s.keys.add(event.code);
    apply(epochOf(event.timeStamp, origin, now()));
  });
  doc.addEventListener("keyup", (event) => {
    if (s.keys.delete(event.code)) apply();
  });
  win.addEventListener("blur", () => {
    // (a key let go while the page was not looking never reports it)
    s.keys.clear();
    for (const name of PAD_KEYS) setHeld(name, false);
    apply();
  });
  win.addEventListener("pagehide", () => finish(null));
  doc.addEventListener("visibilitychange", () => {
    if (doc.hidden) finish(null);
  });
  win.addEventListener("duck:telemetry", (event) => onTelemetry(event.detail));
  toggle.addEventListener("click", () => (s.mode === "idle" ? start() : finish(null)));

  for (const button of padButtons) {
    const name = button.dataset.remoteKey;
    if (!PAD_KEYS.has(name)) continue;
    const release = () => {
      if (!s.held[name]) return;
      setHeld(name, false);
      apply();
    };
    button.addEventListener("pointerdown", (event) => {
      if (s.mode !== "on") return;
      event.preventDefault();
      try {
        button.setPointerCapture(event.pointerId);
      } catch (error) {
        /* no capture: pointerup / pointercancel still end it */
      }
      setHeld(name, true);
      apply(epochOf(event.timeStamp, origin, now()));
    });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) button.addEventListener(type, release);
    button.addEventListener("click", (event) => {
      // a pointer already did its press; a click with no pointer (Enter / Space on the button) walks a moment
      if (event.detail !== 0 || s.mode !== "on") return;
      setHeld(name, true);
      apply();
      timers.setTimeout(release, NUDGE_MS);
    });
    button.addEventListener("contextmenu", (event) => event.preventDefault());
  }

  render();
  return {
    get mode() {
      return s.mode;
    },
    start,
    stop: () => finish(null),
  };
}

if (typeof document !== "undefined" && typeof window !== "undefined" && document.getElementById("remoteToggle")) {
  // (the words first: the catalogs load before this line says anything)
  const start = () => {
    window.duckRemote = createRemote({
      document,
      window,
      WebSocketCtor: window.WebSocket,
      fetchFn: (...args) => window.fetch(...args),
      i18n: window.I18N,
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
