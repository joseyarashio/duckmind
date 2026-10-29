/* 小鴨的叫聲: the duck's own official voice on the page (server side: simweb/voice.py).
 *
 * This is the main tree's voice.js (voice:main, 蛋黃's page) with the neural duck's parts added
 * (voice:neuro 2026-09-28; each one is marked 神經鴨原型 below). The owner: 「模擬器鴨子叫時要真的
 * 播放出聲音，希望能跟官方的叫聲一致」, and for the calls 「維持 但可以音量調整或靜音」.
 *
 * Every call the duck makes is played from this duck's own voice bank: the official Microduck
 * `sounds` synth, rendered by the server from the duck's seed (/api/voice/manifest). On 蛋黃's
 * page those are the sound tags Duck Core records on its events (啾 chirp, 嗯？ inquire, 咕～ coo,
 * 嘎！ greet, 咻～ wheee); in the neural-duck prototype they are the calls its motor voices
 * (neuro.js / duck3d.js: each body sample carries the calls made at it, so the sound starts with
 * the beak the duck's own robot.mouth command opened for it). Events that are only words make no
 * sound.
 *
 * Like robotd: each tag is a random variant of it, picked when it plays; one voice at a time — a
 * new call cuts the one still sounding (a fast-forward drops it instead of chopping every call),
 * except that a wheee ride keeps the speaker while it lasts and a call meanwhile is dropped. A
 * wheee streams start → loop → end: the loop repeats while the move that carries the call lasts
 * (`holdMs`), decided at each pass the way robotd's writer does, ~250 ms ahead of what is heard;
 * a wheee with no move to carry it is robotd's bare `tag: wheee` (a hold that decays after
 * 300 ms: start, then end). The tags of one call (「晚安咕～咕～」) follow each other with a breath,
 * and so do the calls of one moment (a greeting and the move that goes with it: robotd merges
 * what arrives in one tick instead of letting the second chop the first). robotd plays every
 * call at the speaker's one level (a sound request carries no loudness), so a call's `level`
 * changes nothing here either: in the neural duck it only opens the beak wider (its motor).
 *
 * The beak opens with the sound itself: `duckVoice.mouth()` (0..1) is the loudness envelope of
 * what is sounding now, read on the audio clock minus the output latency, so it moves when you
 * hear it (duck3d.js; the 2D duck here). The neural duck's 3D beak is its own robot.mouth command
 * instead, where the body samples carry it (duck3d.js); this envelope is its beak only without
 * them. Muted, or before the browser lets the page make sound, the call still happens: the beak
 * moves, silently.
 *
 * 神經鴨原型: `duckVoice.duck(seed)` (neuro.js) says which neural duck is on screen. Each duck has
 * its own voice from its seed, so a switch asks for the new duck's manifest and, until it is
 * here, no call is played in the old duck's voice; Duck Core's own duck (duck:state) is not the
 * one voiced then, and there is no fast-forward to drop calls for. The whole bank is fetched
 * soon after the manifest (the neural duck calls often), so a call starts with its beak; a call
 * whose sound is not ready by `maxWaitMs` is dropped, never played after its beak.
 *
 * The volume and the mute are this viewer's own (localStorage when the browser allows it; a
 * private window just forgets them). Browsers let a page make sound only after a click or a key:
 * the note under the stage says so until the first one anywhere unlocks it. `state.log` keeps the
 * last calls (asked, heard, files, outcome) for anyone checking that the sound really plays.
 */
"use strict";

(function duckVoiceModule() {
  // Page words (static/i18n.js): 繁體中文 is the source, five translations follow the viewer.
  const I18N = window.I18N;
  const T = (key, values) => (I18N ? I18N.t(key, values) : key);
  const F = (obj, name) => (obj ? (I18N ? I18N.field(obj, name) : obj[name]) ?? "" : "");
  const KEY = "duckmind.voice";
  // the official bank's seven tags, then the 14 emotions of microduck_emotions in this duck's voice
  // (emotion_voice.py: ``emo_<name>``, several variants for no and angry; played by the emotions on the page)
  const EMOTIONS = ["sad", "devastated", "curious", "mmh", "yes", "yes_fast", "no", "defiant", "angry", "mock", "excited", "impatient", "laugh", "play_dead"];
  const TAGS = ["alarm", "greet", "inquire", "peck", "chirp", "coo", "wheee", ...EMOTIONS.map((name) => `emo_${name}`)];
  const COMMON = ["chirp", "greet", "inquire", "coo", "wheee"]; // what Duck Core says: one of each is fetched early
  // the calls the duck learned (server: lineage_voice.py): a phrase `ph_<code>` and its `song`, clips of its own
  const isLineageTag = (tag) => tag === "song" || tag.startsWith("ph_");
  const LINEAGE_ASK_MS = 3000; // the manifest is asked again for a new library at most this often
  const RATE = 48000; // the bank's own rate (sounds::SR); the audio graph resamples if it must
  const GAP_S = 0.06; // a breath between the tags of one call
  const RIDE_LEAD_S = 0.25; // robotd's wheee writer runs this far ahead of what is heard
  const BARE_HOLD_S = 0.3; // robotd's WHEEE_HOLD_FRESH: a bare wheee's hold decays after this
  const MAX_PASSES = 8; // loop passes at most (no move lasts that long)
  const MAX_WAIT_MS = 1500; // a call whose sounds are not here by then is dropped, never played late
  const MAX_TAGS = 8; // sounds in one utterance at most (the calls of one moment)
  const FADE_S = 0.012; // a cut is a very short fade, never a click
  const ENV_HZ = 100; // the beak's envelope: one value every 10 ms
  const GATE = 0.1; // quieter than this share of the call's loudest moment: the beak is shut
  const PREFETCH_MS = 4000; // after the page settles, a little of the bank is fetched ahead
  const NEURO_PREFETCH_MS = 1500; // 神經鴨原型: its whole bank, a little after the manifest
  const SWITCH_RETRY_MS = 400; // 神經鴨原型: a manifest still of the duck before a switch: asked again
  const SWITCH_TRIES = 25;
  const SWITCHING = -1; // 神經鴨原型: state.duck while the page is asking the server for another duck
  const LOG_KEEP = 60;
  const DEFAULT_VOLUME = 0.6;

  const state = {
    volume: DEFAULT_VOLUME,
    muted: false,
    ctx: null, // the AudioContext (made on the first click or key: the browsers' rule)
    master: null,
    manifest: null, // /api/voice/manifest (the current duck's voice)
    manifestError: "",
    loadingManifest: false,
    reloadManifest: false, // (asked again while a load was on its way)
    switchTries: 0,
    retryTimer: 0,
    prefetchTimer: 0,
    lineageAsked: -1e9, // when the manifest was last asked again because a call came from a newer library
    buffers: new Map(), // url → Promise<AudioBuffer> (decoded once)
    current: null, // what is sounding now: {id, env, duration, ride, perf0, ctx0, source, gain}
    gather: null, // the calls asked for at this moment, said together as one utterance
    duck: null, // 神經鴨原型: the neural duck's seed (neuro.js); null: the page voices Duck Core's duck
    seq: 0,
    started: 0, // newest call that began
    played: 0, // calls heard (or muted by this viewer)
    silent: 0, // calls before sound was allowed: the beak only
    dropped: 0,
    late: 0, // (of the dropped: their sound was not ready in time)
    unvoiced: 0, // calls asked for while there was no voice here (no bank yet, no binary, an old server)
    stale: 0, // 神經鴨原型: calls not voiced because the voice here was still the duck before a switch
    cut: 0,
    failed: 0,
    lastCall: null, // what the last call was, for anyone checking: {tags, files, duration, peak}
    log: [], // the last calls: {id, tags, asked, heard, files, duration, audible, outcome}
    stage2d: false, // (the 2D duck's beak is open right now)
    raf: 0,
  };
  const $ = (id) => document.getElementById(id);
  let decoder = null;

  /* ---------------- this viewer's volume and mute ---------------- */

  function load() {
    try {
      const saved = JSON.parse(localStorage.getItem(KEY) || "null");
      if (saved && typeof saved === "object") {
        if (Number.isFinite(saved.volume)) state.volume = Math.max(0, Math.min(1, saved.volume));
        state.muted = !!saved.muted;
      }
    } catch (error) {
      /* (no storage here: the defaults) */
    }
  }

  function save() {
    try {
      localStorage.setItem(KEY, JSON.stringify({ volume: state.volume, muted: state.muted }));
    } catch (error) {
      /* (a private window: not remembered, still works) */
    }
  }

  // Loudness feels logarithmic: the slider's middle is about half as loud. (`duckVoice.gain()`: the
  // same level for the duck's streamed voice, neuro_sound.js — set it with a ramp that ends, as
  // render() does, or a mute between sounds is heard at the start of the next one.)
  const gainOf = () => (state.muted ? 0 : state.volume * state.volume);

  function setVolume(value) {
    state.volume = Math.max(0, Math.min(1, Number(value) || 0));
    save();
    render();
  }

  function setMuted(muted) {
    state.muted = !!muted;
    save();
    render();
  }

  /* ---------------- sound needs a click or a key first (the browsers' rule) ---------------- */

  function unlocked() {
    return !!(state.ctx && state.ctx.state === "running");
  }

  function unlock() {
    if (unlocked()) return; // (every later click and key: nothing to do)
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    try {
      if (!state.ctx) {
        try {
          state.ctx = new Ctx({ latencyHint: "interactive" });
        } catch (error) {
          state.ctx = new Ctx(); // (an older Safari takes no options)
        }
        state.master = state.ctx.createGain();
        state.master.gain.value = gainOf();
        state.master.connect(state.ctx.destination);
      }
      if (state.ctx.state !== "running") {
        // (iOS: a silent sample started inside the click is what really opens the speaker)
        const blank = state.ctx.createBufferSource();
        blank.buffer = state.ctx.createBuffer(1, 1, state.ctx.sampleRate);
        blank.connect(state.ctx.destination);
        blank.start(0);
        const resumed = state.ctx.resume();
        if (resumed && resumed.then) resumed.then(render, () => {});
      }
    } catch (error) {
      state.ctx = null;
      state.master = null;
    }
    render();
  }
  for (const type of ["pointerdown", "touchend", "click", "keydown"]) window.addEventListener(type, unlock, { capture: true, passive: true });

  /* ---------------- the duck's voice bank ---------------- */

  function loadManifest() {
    if (state.loadingManifest) {
      state.reloadManifest = true; // (a switch while the last one is on its way: once more after it)
      return;
    }
    state.loadingManifest = true;
    state.reloadManifest = false;
    clearTimeout(state.retryTimer);
    fetch("/api/voice/manifest", { cache: "no-store" })
      .then((response) => {
        if (response.status === 404) throw new Error("old-server");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((manifest) => {
        const before = state.manifest;
        if (!before || before.voice_seed !== manifest.voice_seed || before.bank_version !== manifest.bank_version) {
          stop();
          state.buffers.clear();
        }
        // a new lineage library (the duck learned something, or grew a day): the old library's clips are
        // let go; what is sounding is already one buffer of its own and is not touched
        const libKey = manifest.lineage && manifest.lineage.key ? manifest.lineage.key : "";
        for (const url of [...state.buffers.keys()]) {
          if (url.includes("/lineage/") && (!libKey || !url.includes(`/lineage/${libKey}/`))) state.buffers.delete(url);
        }
        state.manifest = manifest;
        state.manifestError = "";
        // 神經鴨原型: still the duck before a switch (the server takes the new one as the switch
        // is answered): asked again in a moment, a few times; its voice is not played (call())
        if (state.duck != null && manifest.duck_seed !== state.duck) {
          if (state.duck !== SWITCHING && state.switchTries < SWITCH_TRIES) {
            state.switchTries += 1;
            state.retryTimer = setTimeout(loadManifest, SWITCH_RETRY_MS);
          }
          return;
        }
        state.switchTries = 0;
        if (!manifest.ready && manifest.retry_ms) state.retryTimer = setTimeout(loadManifest, manifest.retry_ms);
        // (the emotions' sounds are rendered a moment after the bank: the manifest is asked once more)
        if (manifest.ready && manifest.emotion_retry_ms) state.retryTimer = setTimeout(loadManifest, manifest.emotion_retry_ms);
        if (manifest.ready) {
          clearTimeout(state.prefetchTimer);
          state.prefetchTimer = setTimeout(prefetch, state.duck != null ? NEURO_PREFETCH_MS : PREFETCH_MS);
        }
      })
      .catch((error) => {
        // An older server process (started before the voice existed) has no such route: asked
        // again when it restarts (duck:reset). Unreachable: asked again in a while.
        const old = !!error && error.message === "old-server";
        state.manifestError = old ? "old-server" : "offline";
        if (!old) state.retryTimer = setTimeout(loadManifest, 15000);
      })
      .finally(() => {
        state.loadingManifest = false;
        render();
        if (state.reloadManifest) loadManifest();
      });
  }

  // Decoded at the bank's own rate, without an AudioContext (none is allowed before a click).
  function decode(bytes) {
    if (!decoderReady()) return Promise.reject(new Error("no Web Audio"));
    return new Promise((resolve, reject) => {
      const promise = decoder.decodeAudioData(bytes, resolve, reject);
      if (promise && promise.then) promise.then(resolve, reject);
    });
  }

  function fetchBuffer(url) {
    let promise = state.buffers.get(url);
    if (!promise) {
      promise = fetch(url)
        .then((response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return response.arrayBuffer();
        })
        .then(decode);
      // (a failed file is asked for again next time instead of staying failed)
      promise.catch(() => state.buffers.delete(url));
      state.buffers.set(url, promise);
    }
    return promise;
  }

  function urlsOf(entry, letter) {
    const variant = entry.variants[letter];
    return entry.segmented ? [variant.start, variant.loop, variant.end] : [variant];
  }

  // A little of the bank ahead of time (one variant of each tag Duck Core uses); every other
  // variant is fetched the first time its tag is heard, so a muted viewer fetches little.
  // 神經鴨原型: the whole bank (it calls often, and a call must start with its beak).
  function prefetch() {
    const manifest = state.manifest;
    if (!manifest || !manifest.ready || state.muted) return;
    const urls = [];
    if (state.duck != null) {
      for (const tag of [...TAGS, ...Object.keys(manifest.tags).filter(isLineageTag)]) {
        const entry = manifest.tags[tag];
        if (!entry) continue;
        entry.fetchedAll = true;
        for (const letter of Object.keys(entry.variants).sort()) urls.push(...urlsOf(entry, letter));
      }
    } else {
      for (const tag of COMMON) {
        const entry = manifest.tags[tag];
        const first = entry && Object.keys(entry.variants).sort()[0];
        if (first) urls.push(...urlsOf(entry, first));
      }
    }
    sequential(urls, manifest);
  }

  function prefetchTag(tag) {
    const manifest = state.manifest;
    const entry = manifest && manifest.tags && manifest.tags[tag];
    if (!entry || entry.fetchedAll) return;
    entry.fetchedAll = true;
    const urls = [];
    for (const letter of Object.keys(entry.variants).sort()) urls.push(...urlsOf(entry, letter));
    sequential(urls, manifest);
  }

  function sequential(urls, manifest) {
    const next = () => {
      const url = urls.shift();
      if (!url || state.manifest !== manifest) return;
      // (神經鴨原型: another duck by now — its voice is on its way; this one is not fetched on)
      if (state.duck != null && manifest.duck_seed !== state.duck) return;
      fetchBuffer(url).then(next, (error) => {
        // Not served any more (another duck on the server, a restart): ask which voice now.
        if (error && /HTTP 404/.test(error.message)) loadManifest();
        else next();
      });
    };
    next();
    next(); // (two at a time)
  }

  /* ---------------- a call: its sounds, one after another, as one buffer ---------------- */

  // Like robotd: a random variant of the tag, picked each time it plays.
  function pick(manifest, tag) {
    const entry = manifest.tags[tag];
    const letters = Object.keys(entry.variants).sort();
    const letter = letters[Math.floor(Math.random() * letters.length)];
    return { tag, letter, segmented: !!entry.segmented, urls: urlsOf(entry, letter) };
  }

  // The utterance's sounds laid end to end (at the bank's rate), a wheee ridden like robotd's
  // writer; each pick carries the hold of the call it came from (`holdS`, 0: none).
  function compose(picks, buffers) {
    const parts = [];
    const files = [];
    let at = 0; // samples
    let ride = null;
    let index = 0;
    picks.forEach((choice, n) => {
      if (choice.segmented) {
        const [start, loop, end] = buffers.slice(index, index + 3);
        index += 3;
        const from = at;
        parts.push([start, at]);
        at += start.length;
        // Its hold: the move that carries the call, or a bare wheee's 300 ms from when it began.
        const holdEnd = choice.holdS > 0 ? choice.holdS : from / RATE + BARE_HOLD_S;
        let passes = 0;
        while (at / RATE - RIDE_LEAD_S < holdEnd && passes < MAX_PASSES) {
          parts.push([loop, at]);
          at += loop.length;
          passes += 1;
        }
        parts.push([end, at]);
        at += end.length;
        ride = [from / RATE, at / RATE];
        files.push(`${choice.tag}_${choice.letter}（start＋loop×${passes}＋end）`);
      } else {
        const buffer = buffers[index];
        index += 1;
        parts.push([buffer, at]);
        at += buffer.length;
        files.push(`${choice.tag}_${choice.letter}`);
      }
      if (n < picks.length - 1) at += Math.round(GAP_S * RATE);
    });
    const out = decoder.createBuffer(1, Math.max(1, at), RATE);
    const data = out.getChannelData(0);
    for (const [buffer, offset] of parts) data.set(buffer.getChannelData(0), offset);
    return { buffer: out, env: envelope(data), duration: at / RATE, ride, files, peak: peakOf(data) };
  }

  function peakOf(data) {
    let peak = 0;
    for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i]));
    return peak;
  }

  // The beak's opening, every 10 ms: the sound's loudness (RMS), 1 at the call's loudest moment,
  // shut below a small share of it.
  function envelope(data) {
    const hop = Math.round(RATE / ENV_HZ);
    const env = new Float32Array(Math.ceil(data.length / hop));
    let top = 0;
    for (let k = 0; k < env.length; k++) {
      let sum = 0;
      const end = Math.min(data.length, (k + 1) * hop);
      for (let i = k * hop; i < end; i++) sum += data[i] * data[i];
      env[k] = Math.sqrt(sum / Math.max(1, end - k * hop));
      top = Math.max(top, env[k]);
    }
    for (let k = 0; k < env.length; k++) {
      const share = top > 0 ? env[k] / top : 0;
      env[k] = share < GATE ? 0 : Math.pow((share - GATE) / (1 - GATE), 0.7);
    }
    return env;
  }

  // Seconds into what is sounding now, as it is heard (the audio clock minus the output latency).
  function heardAt(current) {
    if (current.ctx0 != null && state.ctx) {
      const latency = state.ctx.outputLatency || state.ctx.baseLatency || 0;
      return state.ctx.currentTime - latency - current.ctx0;
    }
    return (performance.now() - current.perf0) / 1000;
  }

  function sounding(current) {
    return !!current && heardAt(current) < current.duration;
  }

  function stop(fade = FADE_S) {
    const current = state.current;
    state.current = null;
    if (!current || !current.source || !state.ctx) return;
    try {
      const now = state.ctx.currentTime;
      current.gain.gain.setTargetAtTime(0, now, fade / 3);
      current.source.stop(now + fade);
    } catch (error) {
      /* (already over) */
    }
  }

  // What happened to a call, for anyone checking (the page check counts the sources it started).
  function logCall(entry) {
    state.log.push(entry);
    if (state.log.length > LOG_KEEP) state.log.shift();
  }

  function begin(id, tags, phrase, asked) {
    if (id < state.started) {
      state.dropped += 1; // a newer call is already sounding
      logCall({ id, tags, asked, outcome: "dropped: a newer call" });
      return;
    }
    const current = state.current;
    if (sounding(current)) {
      const t = heardAt(current);
      const riding = current.ride && t >= current.ride[0] && t < current.ride[1];
      // (神經鴨原型: its clock is always real time — no fast-forward to drop calls for)
      const time = state.duck != null ? {} : window.duckTime || {};
      // robotd: the ride owns the speaker while it lasts. A fast-forward drops a call rather
      // than chopping every call off at once.
      if (riding || (time.playing && !time.realtime)) {
        state.dropped += 1;
        logCall({ id, tags, asked, outcome: riding ? "dropped: a wheee ride has the speaker" : "dropped: fast-forward" });
        return;
      }
      state.cut += 1;
      stop();
    }
    state.started = id;
    const next = { id, env: phrase.env, duration: phrase.duration, ride: phrase.ride, perf0: performance.now() + 20, ctx0: null, source: null, gain: null };
    let heard = next.perf0;
    if (unlocked() && state.master) {
      try {
        const source = state.ctx.createBufferSource();
        source.buffer = phrase.buffer;
        const gain = state.ctx.createGain();
        source.connect(gain);
        gain.connect(state.master);
        next.ctx0 = state.ctx.currentTime + 0.02;
        source.start(next.ctx0);
        source.onended = () => {
          try {
            gain.disconnect();
          } catch (error) {
            /* (already) */
          }
        };
        next.source = source;
        next.gain = gain;
        state.played += 1;
        // (when it is heard, on the page's clock: the start plus the output latency)
        const latency = state.ctx.outputLatency || state.ctx.baseLatency || 0;
        heard = performance.now() + (next.ctx0 - state.ctx.currentTime + latency) * 1000;
      } catch (error) {
        next.ctx0 = null;
        state.silent += 1;
      }
    } else {
      state.silent += 1; // (no sound allowed yet: the beak still moves)
    }
    state.current = next;
    state.lastCall = { tags, files: phrase.files, duration: Math.round(phrase.duration * 1000) / 1000, peak: Math.round(phrase.peak * 1000) / 1000, audible: next.source != null && !state.muted, at: Date.now() };
    logCall({ id, tags, asked, heard, files: phrase.files, duration: state.lastCall.duration, audible: state.lastCall.audible, started: next.source != null, outcome: next.source != null ? "played" : "silent" });
    if (!document.body.classList.contains("has3d")) run2d();
  }

  // A call now: the duck's tags for one event; `holdMs` how long the move carrying it lasts;
  // `maxWaitMs` how late its sound may still start (神經鴨原型: its beak is already moving).
  // Returns whether the call will be voiced (false: no voice bank, nothing to play). The calls
  // of one moment (a greeting and the move that goes with it; waking up and saying hello) are
  // one utterance, said in order — robotd would merge what arrives in one tick; a later call
  // cuts it.
  function call(tags, opts = {}) {
    const manifest = state.manifest;
    const options = opts && typeof opts === "object" ? opts : {};
    // a call from a library this manifest does not have (the server swapped it between two calls):
    // the manifest is asked again; this call is voiced from the one in hand (a phrase it does not list is dropped)
    if (options.lib && manifest && manifest.ready && state.duck != null && (!manifest.lineage || manifest.lineage.key8 !== options.lib)) {
      if (performance.now() - state.lineageAsked > LINEAGE_ASK_MS && !state.loadingManifest) {
        state.lineageAsked = performance.now();
        loadManifest();
      }
    }
    // (a tag the manifest does not list yet — the emotions' sounds still rendering — is not voiced)
    const list = (Array.isArray(tags) ? tags : [tags]).map(String).filter((tag) => (TAGS.includes(tag) || isLineageTag(tag)) && (!manifest || !manifest.ready || (manifest.tags && manifest.tags[tag])));
    if (!list.length) return false;
    if (!manifest || !manifest.ready || !manifest.tags || !decoderReady()) {
      state.unvoiced += 1;
      logCall({ tags: list, asked: performance.now(), outcome: "not voiced: no voice here yet" });
      // (an older server has no voice at all: asked again only when it restarts)
      if (!manifest && !state.loadingManifest && state.manifestError !== "old-server") loadManifest();
      return false;
    }
    // 神經鴨原型: never in another duck's voice (a switch whose voice is still on its way)
    if (state.duck != null && manifest.duck_seed !== state.duck) {
      state.stale += 1;
      logCall({ tags: list, asked: performance.now(), outcome: "not voiced: the voice here is still the duck's before the switch" });
      if (state.duck !== SWITCHING && !state.loadingManifest && state.switchTries < SWITCH_TRIES) loadManifest();
      return false;
    }
    const holdS = Math.max(0, Number(options.holdMs) || 0) / 1000;
    const wait = Number(options.maxWaitMs);
    const waitMs = Number.isFinite(wait) && wait > 0 ? Math.min(MAX_WAIT_MS, wait) : MAX_WAIT_MS;
    if (!state.gather) {
      state.gather = { manifest, asked: performance.now(), items: [], waitMs };
      setTimeout(say, 0); // (after everything this same moment asks for)
    }
    state.gather.waitMs = Math.min(state.gather.waitMs, waitMs);
    for (const tag of list) if (state.gather.items.length < MAX_TAGS) state.gather.items.push({ tag, holdS });
    return true;
  }

  function say() {
    const gather = state.gather;
    state.gather = null;
    if (!gather || !gather.items.length || state.manifest !== gather.manifest) return;
    const { manifest, asked, items, waitMs } = gather;
    const id = ++state.seq;
    const picks = items.map((item) => ({ ...pick(manifest, item.tag), holdS: item.holdS }));
    const tags = items.map((item) => item.tag);
    for (const tag of new Set(tags)) prefetchTag(tag);
    Promise.all(picks.flatMap((choice) => choice.urls).map(fetchBuffer))
      .then((buffers) => {
        if (state.manifest !== manifest) return; // another duck by now
        if (performance.now() - asked > waitMs) {
          // late is worse than never: robotd never plays a call late
          state.dropped += 1;
          state.late += 1;
          logCall({ id, tags, asked, outcome: `dropped: its sound was not ready in ${Math.round(waitMs)} ms` });
          return;
        }
        begin(id, tags, compose(picks, buffers), asked);
      })
      .catch(() => {
        state.failed += 1;
        logCall({ id, tags, asked, outcome: "failed: its sound could not be fetched" });
        if (state.failed % 5 === 1) loadManifest(); // (another duck, or the server restarted)
      });
  }

  function decoderReady() {
    if (decoder) return true;
    const Offline = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!Offline) return false;
    try {
      decoder = new Offline(1, 1, RATE);
      return true;
    } catch (error) {
      return false;
    }
  }

  // The beak: how far open (0..1) the sound being heard now has it.
  function mouth() {
    const current = state.current;
    if (!current) return 0;
    const t = heardAt(current);
    if (t < 0) return 0;
    if (t >= current.duration) {
      if (state.current === current) state.current = null;
      return 0;
    }
    return current.env[Math.min(current.env.length - 1, Math.floor(t * ENV_HZ))] || 0;
  }

  // The 2D duck (no 3D model): its beak opens with the sound as well.
  function run2d() {
    if (state.raf) return;
    const stage = $("duckStage");
    const frame = () => {
      state.raf = 0;
      const open = document.body.classList.contains("has3d") ? 0 : mouth();
      const wide = open > 0.25;
      if (stage && wide !== state.stage2d) stage.classList.toggle("mouth", wide);
      state.stage2d = wide;
      if (state.current) state.raf = requestAnimationFrame(frame);
    };
    state.raf = requestAnimationFrame(frame);
  }

  // 神經鴨原型: the neural duck on screen (neuro.js, from its card), whose own voice this page
  // plays; its manifest is asked for when it is another duck's.
  function setDuck(seed) {
    const value = Number(seed);
    if (!Number.isFinite(value) || value < 0 || state.duck === value) return;
    state.duck = value;
    state.switchTries = 0;
    const manifest = state.manifest;
    // (a manifest on its way is checked against this duck when it comes)
    if ((!manifest || manifest.duck_seed !== value) && !state.loadingManifest) loadManifest();
  }

  // 神經鴨原型: the page is asking the server for another duck (neuro.js's picker): until the new
  // duck is known (duck()), no call in this voice and no more of this bank.
  function switching() {
    state.duck = SWITCHING;
    clearTimeout(state.prefetchTimer);
  }

  /* ---------------- the controls under the stage ---------------- */

  function noteText() {
    const manifest = state.manifest;
    if (state.manifestError === "old-server") return T("page.voice.noteText.8947e4");
    if (state.manifestError) return T("page.voice.noteText.ad56a1");
    if (!manifest) return "";
    if (!manifest.ready) return F(manifest, "note");
    if (state.muted || state.volume <= 0) return "";
    if (!unlocked()) return T("page.voice.noteText.1234af");
    return "";
  }

  function render() {
    const mute = $("callMute");
    const slider = $("callVolume");
    const note = $("callNote");
    const manifest = state.manifest;
    const voiceWords = manifest && manifest.ready ? T("page.voice.render.27d9b9", { voice_seed: manifest.voice_seed }) : "";
    if (mute) {
      mute.setAttribute("aria-pressed", String(state.muted));
      // (written here, not by index.html's data-i18n: a language switch calls render() again)
      delete mute.dataset.i18n;
      delete mute.dataset.i18nAttr;
      mute.textContent = T(state.muted ? "page.voice.render.229583" : "page.voice.render.727b60");
      mute.title = state.muted ? T("page.voice.pending.d96cd0", { voice_words: voiceWords }) : T("page.voice.pending.70827a", { voice_words: voiceWords });
    }
    if (slider) {
      const value = String(Math.round(state.volume * 100));
      if (slider.value !== value) slider.value = value;
      slider.setAttribute("aria-valuetext", state.muted ? T("page.voice.render.195cbe") : `${value}%`);
    }
    if (note) {
      const text = noteText();
      if (note.textContent !== text) note.textContent = text;
      note.hidden = !text;
    }
    if (state.master && state.ctx) {
      // A 30-ms ramp that ENDS: a gain with nothing flowing through it is not processed, so an
      // open-ended approach (setTargetAtTime) would only start moving with the next call — a
      // viewer who muted between calls heard the first ~0.1 s of it (voice:neuro's listening
      // check). After a ramp's end the level is the new one whether or not anything played.
      const level = state.master.gain;
      const now = state.ctx.currentTime;
      try {
        level.cancelScheduledValues(now);
        level.setValueAtTime(level.value, now);
        level.linearRampToValueAtTime(gainOf(), now + 0.03);
      } catch (error) {
        level.value = gainOf();
      }
    }
  }

  function wire() {
    const mute = $("callMute");
    const slider = $("callVolume");
    if (mute) mute.addEventListener("click", () => setMuted(!state.muted));
    if (slider) {
      slider.addEventListener("input", () => {
        state.muted = false; // (moving the slider means: I want to hear it)
        setVolume(Number(slider.value) / 100);
      });
    }
    render();
    loadManifest();
  }

  // Another duck (hatched, or the server restarted with another one): its own voice.
  window.addEventListener("duck:reset", () => {
    stop();
    state.switchTries = 0;
    loadManifest();
  });
  window.addEventListener("duck:state", (event) => {
    if (state.duck != null) return; // (神經鴨原型: the neural duck is voiced, not Duck Core's)
    const identity = event.detail && event.detail.identity;
    const manifest = state.manifest;
    if (identity && manifest && Number(identity.seed) !== manifest.duck_seed) loadManifest();
  });
  // Away from the tab, what was sounding is over (its moves are applied, not played, on return).
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "visible") stop();
  });

  load();
  // (the words first: the controls are written once the catalogs are in; a switch rewrites them)
  const start = () => (I18N ? I18N.ready : Promise.resolve()).then(wire, wire);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
  if (I18N) I18N.onChange(() => render());
  window.duckVoice = { call, mouth, stop, setVolume, setMuted, refresh: loadManifest, duck: setDuck, switching, gain: gainOf, state };
})();
