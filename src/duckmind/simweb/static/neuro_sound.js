/* 聲音遊戲（神經鴨原型, a:sound 2026-09-28): 手的特雷門琴、「咻～」兜風、聽音樂跳舞.
 *
 * The page's buttons make WORLD events (POST /api/neuro/sound/world: music with a tempo, a hand
 * hovering in front of the duck's head, a ride through the air) — the duck notices them only
 * through its own senses and decides by itself what to do. What the page plays is the server's
 * sound stream (GET /api/neuro/sound/pcm, 20-ms frames on the physics clock): the duck's own
 * streamed voice (robotd's theremin and wheee ride, upstream's sound program) and the room's
 * music, each frame played at the moment the 3D view draws the body sample of the same instant
 * (the view's playout clock, mirrored here from the same /api/body answers), so the head's nod,
 * the beak and the sound come together. The duck's streamed voice follows the call volume and
 * mute (voice.js); the music has its own volume. Browsers make sound only after a click.
 *
 * window.__duckSound: the listening check's counters (frames, late ones, the scheduling lead,
 * beats and calls on one clock). Design: docs/simulator-style.md (tokens and component
 * classes only). */
import { makePlayout, playoutArrive, playoutTime } from "./truth.js";

const SR = 48000;
const HEADER = 28;
const STATUS_MS = 500;
const RETRY_MS = 1500;
const LATE_DROP_S = 0.25; // a frame this late is dropped (the stream catches up)
// The frames are played back to back (the stream is continuous by construction); where the view
// puts each one (its stamp on the view's clock) only steers them: the smoothed difference slows
// or quickens the playing a touch (at most 1 %, none within 5 ms), and only when the view and
// the sound are persistently apart — the median of the last 11 frames over 80 ms (a stall of the
// server or of the physics) — is the sound put back where the view is, once.
const DRIFT_FRAMES = 25; // ~0.5 s
const RESYNC_S = 0.08;
const HARD_RESYNC_S = 0.25;
const SLEW_DEAD_S = 0.005;
const SLEW_GAIN = 0.5; // rate change per second of smoothed difference
const SLEW_MAX = 0.01;
const MUSIC_KEY = "duckmind.sound";

const $ = (id) => document.getElementById(id);
const tr = (key, params = {}) => (window.I18N ? window.I18N.t(key, params) : key);
const emitStatus = (extra = {}) => window.dispatchEvent(new CustomEvent("duck:sound-status", { detail: { ...(st.status || {}), ...extra, pcm: { frames: probe.frames, unlocked: probe.unlocked } } }));
const make = (tag, props = {}, children = []) => {
  const el = Object.assign(document.createElement(tag), props);
  for (const child of children) el.append(child);
  return el;
};

const st = {
  on: false,
  ctx: null,
  duckGain: null,
  roomGain: null,
  musicVolume: 0.5,
  play: makePlayout(), // mirror of the 3D view's playout clock
  drawnW: null,
  drawnAt: 0,
  lastSeq: 0,
  nextAt: 0,
  drift: 0, // smoothed (where the view puts a frame − where the sound plays it), s
  recent: [], // the last 11 such differences (for the median)
  streaming: false,
  status: null,
  tof: null,
  hand: 30,
  handTimer: 0,
  handEdited: -1e9, // (performance.now() of the owner's last move of the slider)
};

// the listening check's counters
const probe = {
  frames: 0, bytes: 0, duckFrames: 0, roomFrames: 0, late: 0, dropped: 0, joined: 0, resyncs: 0,
  slewed: 0, leads: [], beats: [], calls: [], errors: [], unlocked: false, mirror: null, restarts: 0,
  map: [], sched: [],
};
window.__duckSound = probe;

function loadPrefs() {
  try {
    const saved = JSON.parse(localStorage.getItem(MUSIC_KEY) || "null");
    if (saved && Number.isFinite(saved.music)) st.musicVolume = Math.max(0, Math.min(1, saved.music));
  } catch (error) {
    /* (no storage: the default) */
  }
}

function savePrefs() {
  try {
    localStorage.setItem(MUSIC_KEY, JSON.stringify({ music: st.musicVolume }));
  } catch (error) {
    /* (a private window: not remembered) */
  }
}

/* ---------------- the view's clock, mirrored ---------------- */

// duck3d.js plays every body sample at its server wall time minus a display delay (truth.js's
// playout); the same /api/body answers reach this mirror, so its clock is the view's.
const baseFetch = window.fetch.bind(window);
window.fetch = async (...args) => {
  const response = await baseFetch(...args);
  try {
    const url = typeof args[0] === "string" ? args[0] : args[0] && args[0].url;
    if (url && url.includes("/api/body") && url.includes("since_t")) {
      response
        .clone()
        .json()
        .then((info) => {
          const frames = info && info.frames;
          if (frames && frames.length) playoutArrive(st.play, frames[frames.length - 1].w, performance.now() / 1000);
        })
        .catch(() => {});
    }
  } catch (error) {
    /* (never break the page's own fetch) */
  }
  return response;
};

let lastTick = performance.now();
function tickClock(now) {
  const dt = (now - lastTick) / 1000;
  lastTick = now;
  const at = playoutTime(st.play, now / 1000, dt);
  if (at != null) {
    st.drawnW = at;
    st.drawnAt = now / 1000;
    probe.mirror = { delay: +st.play.delay.toFixed(3) };
  }
  requestAnimationFrame(tickClock);
}
requestAnimationFrame(tickClock);

// The server wall time the view draws right now (extrapolated from the last frame).
function drawnNow() {
  if (st.drawnW == null) return null;
  return st.drawnW + (performance.now() / 1000 - st.drawnAt);
}

/* ---------------- audio ---------------- */

function audio() {
  if (st.ctx) return st.ctx;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return null;
  const ctx = new Ctx({ sampleRate: SR, latencyHint: "interactive" });
  st.duckGain = ctx.createGain();
  st.roomGain = ctx.createGain();
  st.duckGain.connect(ctx.destination);
  st.roomGain.connect(ctx.destination);
  st.ctx = ctx;
  probe.graph = { ctx, duck: st.duckGain, room: st.roomGain }; // (the listening check's tap)
  applyVolumes();
  return ctx;
}

function unlock() {
  const ctx = audio();
  if (!ctx) return;
  const note = $("soundNote");
  if (note) note.hidden = true;
  const update = () => {
    probe.unlocked = ctx.state === "running";
    emitStatus();
  };
  if (ctx.state !== "running") ctx.resume().then(update).catch(update);
  else update();
  ctx.addEventListener("statechange", update);
}
document.addEventListener("pointerdown", unlock, { capture: true });
document.addEventListener("keydown", unlock, { capture: true });

function callGain() {
  // the duck's own voice: the same volume and mute as its calls (voice.js's own gain)
  const dv = window.duckVoice;
  if (dv && typeof dv.gain === "function") {
    const g = Number(dv.gain());
    if (Number.isFinite(g)) return g;
  }
  const voice = dv && dv.state;
  if (voice && Number.isFinite(voice.volume)) return voice.muted ? 0 : voice.volume * voice.volume;
  const slider = $("callVolume");
  const mute = $("callMute");
  const volume = slider ? Number(slider.value) / 100 : 0.6;
  const muted = mute && mute.getAttribute("aria-pressed") === "true";
  return muted ? 0 : volume * volume;
}

function applyVolumes() {
  if (!st.ctx) return;
  const t = st.ctx.currentTime;
  st.duckGain.gain.setTargetAtTime(callGain(), t, 0.05);
  st.roomGain.gain.setTargetAtTime(0.8 * st.musicVolume * st.musicVolume, t, 0.05);
}

// the calls (voice.js) on the same clock as the beats, for the listening check
function watchCalls() {
  const voice = window.duckVoice;
  if (!voice || voice.__soundWatched) return;
  const call = voice.call;
  voice.call = (tag, level, who) => {
    probe.calls.push({ tag, perf: performance.now() });
    if (probe.calls.length > 400) probe.calls.shift();
    return call(tag, level, who);
  };
  voice.__soundWatched = true;
}

/* ---------------- the stream ---------------- */

function schedule(frame, bytes) {
  const ctx = st.ctx;
  probe.frames += 1;
  probe.bytes += bytes;
  if (!ctx || ctx.state !== "running") return;
  const drawn = drawnNow();
  if (drawn == null) return; // (the view has not drawn a body sample yet)
  const out = (ctx.outputLatency || ctx.baseLatency || 0);
  const target = ctx.currentTime + (frame.w - drawn) - out; // where the view puts it
  let at = target;
  let rate = 1;
  const resync = () => {
    at = target;
    st.drift = 0;
    st.recent = [];
  };
  if (frame.seq === st.lastSeq + 1 && st.nextAt > 0) {
    const err = target - st.nextAt; // + : the view has it later than the sound would play it
    st.drift += (err - st.drift) / DRIFT_FRAMES;
    st.recent.push(err);
    if (st.recent.length > 11) st.recent.shift();
    const med = [...st.recent].sort((a, b) => a - b)[st.recent.length >> 1];
    if (Math.abs(err) > HARD_RESYNC_S || (st.recent.length >= 11 && Math.abs(med) > RESYNC_S)) {
      resync();
      probe.resyncs += 1;
    } else {
      at = st.nextAt;
      probe.joined += 1;
      if (Math.abs(st.drift) > SLEW_DEAD_S) {
        rate = 1 - Math.max(-SLEW_MAX, Math.min(SLEW_MAX, SLEW_GAIN * st.drift));
        probe.slewed += 1;
      }
    }
  } else {
    if (st.lastSeq) probe.resyncs += 1;
    resync();
  }
  const lead = at - ctx.currentTime;
  probe.leads.push(+lead.toFixed(4));
  if (probe.leads.length > 500) probe.leads.shift();
  st.lastSeq = frame.seq;
  if (lead < -LATE_DROP_S) {
    probe.dropped += 1;
    st.nextAt = 0;
    return;
  }
  if (lead < 0.002) {
    probe.late += 1;
    at = ctx.currentTime + 0.002;
  }
  // on the sample grid: a start between two samples is interpolated by the browser (the song's
  // hats and the voice's edges lose their top); joined frames stay exactly back to back
  at = Math.round(at * ctx.sampleRate) / ctx.sampleRate;
  st.nextAt = at + frame.n / (SR * rate);
  if (frame.seq % 25 === 0) {
    // (the listening check: where the server's clock lands on the page's audio clock)
    probe.map.push([frame.w, at, frame.seq]);
    if (probe.map.length > 3000) probe.map.shift();
  }
  probe.sched.push([frame.seq, +at.toFixed(6), rate === 1 ? 1 : +rate.toFixed(5), frame.room ? 1 : 0]);
  if (probe.sched.length > 12000) probe.sched.shift();
  for (const [plane, gain, key] of [
    [frame.duck, st.duckGain, "duckFrames"],
    [frame.room, st.roomGain, "roomFrames"],
  ]) {
    if (!plane) continue;
    probe[key] += 1;
    const buffer = ctx.createBuffer(1, frame.n, SR);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < frame.n; i++) data[i] = plane[i] / 32768;
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    if (rate !== 1) source.playbackRate.value = rate;
    source.connect(gain);
    source.start(at);
  }
  for (const [offset, kind] of frame.marks) {
    // a beat of the world's music: where it will sound on the page's clock
    const perf = performance.now() + (at + offset / (SR * rate) - ctx.currentTime + out) * 1000;
    probe.beats.push({ perf, kind });
    if (probe.beats.length > 400) probe.beats.shift();
  }
}

function parse(buffer, used) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  let pos = used;
  const frames = [];
  while (pos + HEADER <= buffer.byteLength) {
    const magic = String.fromCharCode(buffer[pos], buffer[pos + 1], buffer[pos + 2], buffer[pos + 3]);
    if (magic !== "DKSD") throw new Error("not a sound frame");
    const seq = view.getUint32(pos + 4, true);
    const w = view.getFloat64(pos + 8, true);
    const sim = view.getFloat64(pos + 16, true);
    const n = view.getUint16(pos + 24, true);
    const flags = view.getUint8(pos + 26);
    const nmarks = view.getUint8(pos + 27);
    const size = HEADER + 4 * nmarks + 2 * n * ((flags & 1) + ((flags >> 1) & 1));
    if (pos + size > buffer.byteLength) break;
    let at = pos + HEADER;
    const marks = [];
    for (let k = 0; k < nmarks; k++) {
      marks.push([view.getUint16(at, true), view.getUint8(at + 2)]);
      at += 4;
    }
    const plane = () => {
      const out = new Int16Array(n);
      for (let i = 0; i < n; i++) out[i] = view.getInt16(at + 2 * i, true);
      at += 2 * n;
      return out;
    };
    const duck = flags & 1 ? plane() : null;
    const room = flags & 2 ? plane() : null;
    frames.push({ seq, w, sim, n, duck, room, marks, size });
    pos += size;
  }
  return { frames, used: pos };
}

async function stream() {
  if (st.streaming) return;
  // (no sound lane yet — the body is still starting, or the page check's made-up numbers: the
  // stream is asked for only once the status says it runs, so no request fails in the meantime)
  if (st.status && st.status.running === false) {
    if (st.on) setTimeout(stream, RETRY_MS);
    return;
  }
  st.streaming = true;
  try {
    const response = await baseFetch(`/api/neuro/sound/pcm?since=${st.lastSeq || 0}`, { cache: "no-store" });
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
    const reader = response.body.getReader();
    let pending = new Uint8Array(0);
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const joined = new Uint8Array(pending.length + value.length);
      joined.set(pending, 0);
      joined.set(value, pending.length);
      const { frames, used } = parse(joined, 0);
      for (const frame of frames) schedule(frame, frame.size);
      pending = joined.slice(used);
    }
  } catch (error) {
    probe.errors.push(String(error && error.message ? error.message : error));
    if (probe.errors.length > 20) probe.errors.shift();
  } finally {
    st.streaming = false;
    probe.restarts += 1;
    if (st.on) setTimeout(stream, RETRY_MS);
  }
}

/* ---------------- the buttons (world events) ---------------- */

async function world(body, button) {
  unlock();
  if (button) button.classList.add("sent");
  try {
    const response = await baseFetch("/api/neuro/sound/world", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const answer = await response.json();
    if (!answer.ok) throw new Error("world update failed");
    say(tr("page.neuro.sound.world.sent"));
  } catch (error) {
    say(tr("page.neuro.sound.world.failed"));
  } finally {
    if (button) setTimeout(() => button.classList.remove("sent"), 400);
  }
}

function say(text) {
  if (typeof window.toast === "function") window.toast(text);
  window.dispatchEvent(new CustomEvent("duck:caption", { detail: { text, strong: false, reply: true } }));
}

function build() {
  if ($("neuroSound")) return;
  const anchor = $("neuroQuick");
  if (!anchor) return;
  const hand = make("input", { type: "range", id: "soundHand", min: "10", max: "70", step: "1", value: String(st.hand) });
  hand.setAttribute("aria-label", tr("page.neuro.sound.hand.aria"));
  const handOut = make("output", { id: "soundHandOut", textContent: tr("page.neuro.sound.cm", { cm: st.hand }) });
  const musicVol = make("input", { type: "range", id: "soundMusicVol", min: "0", max: "100", step: "5", value: String(Math.round(st.musicVolume * 100)) });
  musicVol.setAttribute("aria-label", tr("page.neuro.sound.music.volume"));
  const btn = (key, data) => {
    const b = make("button", { type: "button", className: "btn small", textContent: tr(key), title: tr(`${key}.title`) });
    for (const [k, v] of Object.entries(data)) b.dataset[k] = v;
    return b;
  };
  const strip = make("div", { className: "stage-stim sound-stim", id: "neuroSound" }, [
    make("span", { className: "stage-stim-label", textContent: tr("page.neuro.sound.header") }),
    make("span", { className: "sound-group" }, [
      btn("page.neuro.sound.music.slow", { sound: "music", tempo: "slow" }),
      btn("page.neuro.sound.music.mid", { sound: "music", tempo: "mid" }),
      btn("page.neuro.sound.music.fast", { sound: "music", tempo: "fast" }),
      btn("page.neuro.sound.music.stop", { sound: "music_off" }),
    ]),
    make("span", { className: "sound-group" }, [
      btn("page.neuro.sound.hand.play", { sound: "hover_play" }),
      make("label", { className: "sound-range", title: tr("page.neuro.sound.hand.label") }, [tr("page.neuro.sound.hand.label"), hand, handOut]),
      btn("page.neuro.sound.hand.remove", { sound: "hover_end" }),
    ]),
    btn("page.neuro.sound.ride.start", { sound: "swing" }),
    make("label", { className: "sound-range", title: tr("page.neuro.sound.music.volume.title") }, [tr("page.neuro.sound.music.volume"), musicVol]),
    make("span", { className: "sound-read", id: "soundRead" }),
    make("small", { className: "call-note", id: "soundNote", textContent: tr("page.neuro.sound.note.click") }),
  ]);
  strip.dataset.neuro = "";
  strip.dataset.scenarios = "";
  strip.setAttribute("aria-label", tr("page.neuro.sound.header"));
  // (the 逗逗牠 tab has its own place for it; without one, right under the quick strip as before)
  const mount = $("neuroSoundMount");
  if (mount) mount.append(strip);
  else anchor.after(strip);
  strip.addEventListener("click", (event) => {
    const b = event.target.closest("button[data-sound]");
    if (!b) return;
    const what = b.dataset.sound;
    if (what === "music") world({ do: "music", tempo: b.dataset.tempo }, b);
    else if (what === "hover_play") world({ do: "hover", play: true }, b);
    else world({ do: what }, b);
  });
  hand.addEventListener("input", () => {
    st.handEdited = performance.now();
    st.hand = Number(hand.value);
    handOut.textContent = tr("page.neuro.sound.cm", { cm: st.hand });
    clearTimeout(st.handTimer);
    st.handTimer = setTimeout(() => world({ do: "hover", distance_mm: st.hand * 10 }), 250);
  });
  musicVol.addEventListener("input", () => {
    st.musicVolume = Number(musicVol.value) / 100;
    savePrefs();
    applyVolumes();
  });
  for (const id of ["callVolume", "callMute"]) {
    const el = $(id);
    if (el) el.addEventListener(id === "callMute" ? "click" : "input", () => setTimeout(applyVolumes, 0));
  }
}

/* ---------------- what the duck does (its own report) ---------------- */

function chip(text, tone = "") {
  return make("span", { className: `chip sound-chip ${tone}`.trim(), textContent: text });
}

// The slider shows where the world's hand is (a page opened later, or another viewer's move),
// unless the owner is moving it right now.
function syncHand(status) {
  const hover = (status.world || {}).hover;
  const slider = $("soundHand");
  if (!slider || !hover || !Number.isFinite(hover.distance_mm)) return;
  if (performance.now() - st.handEdited < 2000 || document.activeElement === slider) return;
  const cm = Math.max(10, Math.min(70, Math.round(hover.distance_mm / 10)));
  if (cm === st.hand) return;
  st.hand = cm;
  slider.value = String(cm);
  const out = $("soundHandOut");
  if (out) out.textContent = tr("page.neuro.sound.cm", { cm });
}

function render(status) {
  syncHand(status);
  const read = $("soundRead");
  if (!read) return;
  const parts = [];
  const skills = status.skills || {};
  const voice = status.voice || {};
  const dance = skills.dance || {};
  if (dance.beat_found) {
    const sure = dance.lock >= 0.8 ? "steady" : dance.lock >= 0.5 ? "aligning" : "unsure";
    const bpm = tr("page.neuro.sound.status.dance.beat", { bpm: Math.round(dance.bpm) });
    const certainty = tr(`page.neuro.sound.status.dance.certainty.${sure}`);
    const words = dance.on
      ? tr("page.neuro.sound.status.dance.chip_nod", { beat: bpm, certainty, nodding: tr("page.neuro.sound.status.dance.nodding") })
      : tr("page.neuro.sound.status.dance.chip", { beat: bpm, certainty });
    parts.push(chip(words, dance.on ? "on" : ""));
  } else if ((status.world || {}).music && status.world.music.on) {
    parts.push(chip(tr("page.neuro.sound.status.dance.looking")));
  }
  const th = skills.theremin || {};
  if (status.tof === false) {
    parts.push(chip(tr("page.neuro.skills.reason.no_sensor"), "muted"));
  } else if (th.up) {
    const hz = voice.note_hz ? Math.round(voice.note_hz) : 0;
    const hand = voice.hand_mm ? Math.round(voice.hand_mm / 10) : 0;
    parts.push(chip(tr("page.neuro.sound.status.theremin.playing", { hand, hz }), "on"));
  } else if (th.hand) {
    parts.push(chip(tr("page.neuro.sound.status.theremin.hand_seen")));
  } else if (voice.note_hz) {
    parts.push(chip(tr("page.neuro.sound.status.theremin.silent")));
  }
  const ride = skills.ride || {};
  if (ride.riding) parts.push(chip(tr(ride.thrilled ? "page.neuro.sound.status.ride.fun" : "page.neuro.sound.status.ride.scared"), ride.thrilled ? "on" : "warn"));
  else if (voice.ride === "landing") parts.push(chip(tr("page.neuro.sound.status.ride.landing"), "on"));
  else if (ride.after > 0.3) parts.push(chip(tr("page.neuro.sound.status.ride.after")));
  read.replaceChildren(...parts);
}

async function poll() {
  if (!st.on) return;
  try {
    const response = await baseFetch("/api/neuro/sound/status", { cache: "no-store" });
    const status = await response.json();
    st.status = status;
    st.tof = status.tof;
    render(status);
    emitStatus({ unavailable: false });
  } catch (error) {
    emitStatus({ unavailable: true });
  }
  setTimeout(poll, STATUS_MS);
}

async function boot() {
  // The source locale can already be "zh-Hant", so i18n.js may finish loading its page
  // catalog without firing a language-change event. Wait before building this panel or its
  // first paint would permanently expose message keys.
  if (window.I18N?.ready) await window.I18N.ready;
  try {
    const response = await baseFetch("/api/neuro/sound/status", { cache: "no-store" });
    if (!response.ok) throw new Error("sound route unavailable");
  } catch (error) {
    emitStatus({ unavailable: true, running: false });
    return; // server without the sound route
  }
  loadPrefs();
  st.on = true;
  if (window.I18N && typeof window.I18N.onChange === "function") window.I18N.onChange(() => {
    if ($("neuroSound")) {
      $("neuroSound").remove();
      build();
      if (st.status) render(st.status);
    }
  });
  const ready = () => {
    if (!$("neuroQuick") || $("neuroQuick").hidden) {
      setTimeout(ready, 500);
      return;
    }
    build();
    watchCalls();
    poll().then(stream, stream); // (the status first: it says whether the sound stream runs)
  };
  ready();
}

boot();
