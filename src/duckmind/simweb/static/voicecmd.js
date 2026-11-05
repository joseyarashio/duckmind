/* 按住說話 on the simulator page (voice commands; docs/design/neural-duck/VOICE-COMMANDS-2026-11-04.md):
 *
 * - the command strip's microphone button (`#cmdMic`) is push-to-talk: hold it (mouse, finger, or Space / Enter
 *   while it has the focus), speak, let go. While it is held the page records the microphone; 0.15 s after it is let
 *   go the clip, as 16 kHz mono 16-bit PCM, is posted to THIS simulator server only
 *   (`POST /api/neuro/voice/command?lang=<page language>`), which recognises it on this computer and hands what it
 *   understood to the duck: a command, as the text box does, or the owner's 「乖」/「不對」, as the feedback buttons do;
 * - the line under the strip (`#cmdVoiceOut`) says what was heard and what it was understood as, and how long it took
 *   from letting go; the command's own answer goes to the command line (`#cmdOut`, neuro.js) and a praise or scold's
 *   to the feedback line (tricks.js), so every way of telling the duck reads the same;
 * - the button is on only when the server has a recogniser and the duck's brain runs
 *   (`GET /api/neuro/voice/status`); otherwise its tooltip says why. The microphone is opened for each press and
 *   closed right after, so the browser's recording mark is on only while the button is held.
 *
 * Words come from the catalogs (`page.voice.*`, the verbs `page.cmd.verb.*`, the feedback words `page.trick.fb.*`).
 * `createVoiceCommands(env)` takes everything it touches (document, fetch, the microphone, timers, clock), so tests run
 * it in node over fakes; the last lines start it on the real page. */

export const RATE = 16000; // what the server's recogniser takes
export const MAX_MS = 10000; // the longest clip (the server refuses longer)
export const MIN_MS = 250; // a shorter press is a slip: nothing is sent
export const TAIL_MS = 150; // the microphone keeps listening this long after the button is let go (the last syllable)
export const STATUS_MS = 10000; // the status is read again this often while the button is off

const VERB_KEYS = {
  sit: "page.cmd.verb.sit",
  stand: "page.cmd.verb.stand",
  come: "page.cmd.verb.come",
  look_at_me: "page.cmd.verb.look_at_me",
  stop: "page.cmd.verb.stop",
  follow: "page.cmd.verb.follow",
  go_home: "page.cmd.verb.go_home",
  lie_down: "page.cmd.verb.lie_down",
  spin: "page.cmd.verb.spin",
  bow: "page.cmd.verb.bow",
  shake: "page.cmd.verb.shake",
  flip: "page.cmd.verb.flip",
};
const FEEDBACK_KEYS = { praise: "page.trick.fb.praise", scold: "page.trick.fb.scold" };
const MIC_ERRORS = {
  denied: "page.voice.err.denied",
  insecure: "page.voice.err.insecure",
  nodevice: "page.voice.err.nodevice",
  failed: "page.voice.err.mic_failed",
};
const SERVER_ERRORS = {
  not_running: "page.voice.err.not_running",
  no_recognizer: "page.voice.err.no_recognizer",
  recognizer_failed: "page.voice.err.no_recognizer",
  busy: "page.voice.err.busy",
  too_long: "page.voice.err.too_long",
};

/* Float32 blocks recorded at `srcRate` → one Int16Array at 16 kHz (each output sample the mean of the input samples
   it covers, so a 48 kHz microphone does not alias; a 16 kHz one is copied). */
export function toPcm16(blocks, srcRate) {
  let total = 0;
  for (const b of blocks) total += b.length;
  const all = new Float32Array(total);
  let at = 0;
  for (const b of blocks) {
    all.set(b, at);
    at += b.length;
  }
  const ratio = srcRate / RATE;
  const n = ratio === 1 ? total : Math.floor(total / ratio);
  const out = new Int16Array(n);
  for (let i = 0; i < n; i += 1) {
    let v;
    if (ratio === 1) v = all[i];
    else {
      const from = Math.floor(i * ratio);
      const to = Math.max(from + 1, Math.min(total, Math.floor((i + 1) * ratio)));
      let sum = 0;
      for (let j = from; j < to; j += 1) sum += all[j];
      v = sum / (to - from);
    }
    out[i] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
  }
  return out;
}

/* What one answer of `POST /api/neuro/voice/command` is said as: a catalog key, its values (some of them catalog keys
   themselves, so a language switch says it again in the new words) and the line's tone. `status`: the HTTP status;
   `data`: the answer (null when nothing came back); `ms`: from letting go of the button to the answer. */
export function heardLine(status, data, ms) {
  const secs = Number.isFinite(ms) ? Math.max(0, ms) / 1000 : null;
  if (!data || typeof data !== "object") return { key: "page.voice.err.failed", values: {}, tone: "error", secs: null };
  if (status !== 200 || data.ok !== true) {
    return { key: SERVER_ERRORS[data.reason] || "page.voice.err.failed", values: {}, tone: "error", secs: null };
  }
  const heard = data.heard || {};
  const text = typeof heard.text === "string" ? heard.text.trim() : "";
  if (!text) {
    return { key: heard.empty === "too_short" ? "page.voice.too_short" : "page.voice.nothing", values: {}, tone: "none", secs };
  }
  const u = data.understood || {};
  let meaningKey = "page.voice.meaning.none";
  let tone = "none";
  if (u.kind === "command" && VERB_KEYS[u.verb]) {
    meaningKey = VERB_KEYS[u.verb];
    tone = data.answer && data.answer.ok === false ? "refused" : "heard";
  } else if (u.kind === "feedback" && FEEDBACK_KEYS[u.feedback]) {
    meaningKey = FEEDBACK_KEYS[u.feedback];
    tone = u.feedback === "scold" ? "down" : "up";
  }
  return { key: "page.voice.heard", values: { text }, meaningKey, tone, secs };
}

export function createVoiceCommands(env) {
  const { document, fetchFn, i18n, timers, openMic } = env;
  const now = env.now || (() => Date.now());
  const lang = env.lang || (() => "zh-Hant");
  const $ = (id) => document.getElementById(id);
  const T = (key, params) => (i18n && typeof i18n.t === "function" ? i18n.t(key, params) : key);
  const button = $("cmdMic");
  const out = $("cmdVoiceOut");
  const label = button ? button.querySelector("[data-voice-label]") : null;
  const s = { phase: "off", status: null, pressed: false, mic: null, blocks: [], samples: 0, line: null, poll: 0, cap: 0, releasedAt: 0 };

  function paintLine() {
    if (!out || !s.line) return;
    const a = s.line;
    const values = { ...a.values };
    if (a.meaningKey) values.meaning = T(a.meaningKey);
    let text = T(a.key, values);
    if (a.secs != null) text = T("page.voice.latency", { line: text, secs: a.secs.toFixed(1) });
    out.hidden = false;
    out.dataset.state = a.tone;
    delete out.dataset.i18n;
    out.textContent = text;
  }

  function say(key, tone, extra = {}) {
    s.line = { key, values: {}, tone, secs: null, ...extra };
    paintLine();
  }

  function paintButton() {
    if (!button) return;
    const on = s.phase !== "off";
    button.disabled = !on;
    button.dataset.state = s.phase;
    button.setAttribute("aria-pressed", s.phase === "listening" ? "true" : "false");
    let titleKey = "page.voice.title";
    if (!on) {
      const st = s.status;
      titleKey = !st ? "page.voice.off.title" : st.engine == null ? "page.voice.missing.title" : !st.running ? "page.voice.off.title" : "page.voice.title";
    }
    if (button.dataset.titleKey !== titleKey) {
      button.dataset.titleKey = titleKey;
      button.setAttribute("data-i18n-attr", `title:${titleKey}`); // (the page's own language switch then says the same)
      button.title = T(titleKey);
    }
    if (label) {
      const key = s.phase === "listening" ? "page.voice.release" : s.phase === "thinking" ? "page.voice.thinking" : "page.cmd.mic";
      if (label.dataset.i18n !== key || label.textContent !== T(key)) {
        label.dataset.i18n = key;
        label.textContent = T(key);
      }
    }
  }

  async function readStatus() {
    let st = null;
    try {
      const response = await fetchFn("/api/neuro/voice/status", { cache: "no-store" });
      st = response.status === 200 ? await response.json() : null;
    } catch (error) {
      st = null;
    }
    s.status = st;
    const enabled = !!(st && st.enabled);
    if (enabled && s.phase === "off") s.phase = "idle";
    if (!enabled && s.phase === "idle") s.phase = "off";
    paintButton();
    timers.clearTimeout(s.poll);
    if (!enabled) s.poll = timers.setTimeout(readStatus, STATUS_MS);
    return st;
  }

  function closeMic() {
    timers.clearTimeout(s.cap);
    const mic = s.mic;
    s.mic = null;
    if (mic) {
      try {
        mic.close();
      } catch (error) {
        /* (already closed) */
      }
    }
  }

  async function press() {
    if (s.phase !== "idle") return;
    s.pressed = true;
    s.phase = "opening";
    s.blocks = [];
    s.samples = 0;
    paintButton();
    let mic;
    try {
      mic = await openMic();
    } catch (error) {
      s.phase = "idle";
      s.pressed = false;
      say(MIC_ERRORS[error && error.kind] || MIC_ERRORS.failed, "error");
      paintButton();
      return;
    }
    if (!s.pressed) {
      // (let go before the microphone was open: nothing to send)
      try {
        mic.close();
      } catch (error) {
        /* (already closed) */
      }
      s.phase = "idle";
      say("page.voice.too_short", "none");
      paintButton();
      return;
    }
    s.mic = mic;
    mic.onBlock((block) => {
      const room = Math.floor((mic.rate * MAX_MS) / 1000) - s.samples; // (never more than the server takes)
      if (s.mic !== mic || room <= 0) return;
      const piece = block.length > room ? block.subarray(0, room) : block;
      s.blocks.push(piece);
      s.samples += piece.length;
    });
    s.phase = "listening";
    s.cap = timers.setTimeout(release, MAX_MS - TAIL_MS);
    say("page.voice.listening", "listening");
    paintButton();
  }

  async function release() {
    s.pressed = false;
    if (s.phase !== "listening") return;
    s.phase = "thinking";
    s.releasedAt = now();
    timers.clearTimeout(s.cap);
    paintButton();
    say("page.voice.thinking_line", "listening");
    await new Promise((resolve) => timers.setTimeout(resolve, TAIL_MS));
    const mic = s.mic;
    const rate = mic ? mic.rate : RATE;
    closeMic();
    const pcm = toPcm16(s.blocks, rate);
    s.blocks = [];
    if (pcm.length < (RATE * MIN_MS) / 1000) {
      s.phase = "idle";
      say("page.voice.too_short", "none");
      paintButton();
      return;
    }
    let status = 0;
    let data = null;
    try {
      const response = await fetchFn(`/api/neuro/voice/command?lang=${encodeURIComponent(lang())}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: pcm.buffer,
      });
      status = response.status;
      data = await response.json();
    } catch (error) {
      data = null;
    }
    const ms = now() - s.releasedAt;
    s.line = heardLine(status, data, ms);
    s.lastMs = ms;
    paintLine();
    if (status === 200 && data && data.ok === true && data.answer) {
      const u = data.understood || {};
      if (u.kind === "command" && env.showCommand) env.showCommand(data.answer, u.verb);
      else if (u.kind === "feedback" && env.showFeedback) env.showFeedback(u.feedback, data.answer.ok === true ? 200 : 400, data.answer);
    }
    s.phase = "idle";
    if (status === 503 && data && (data.reason === "not_running" || data.reason === "no_recognizer")) readStatus();
    paintButton();
  }

  if (button) {
    button.addEventListener("pointerdown", (event) => {
      if (event.button !== undefined && event.button !== 0) return;
      event.preventDefault();
      if (button.setPointerCapture && event.pointerId !== undefined) {
        try {
          button.setPointerCapture(event.pointerId);
        } catch (error) {
          /* (a synthetic pointer) */
        }
      }
      press();
    });
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) button.addEventListener(name, () => release());
    button.addEventListener("keydown", (event) => {
      if ((event.key === " " || event.key === "Enter") && !event.repeat) {
        event.preventDefault();
        press();
      }
    });
    button.addEventListener("keyup", (event) => {
      if (event.key === " " || event.key === "Enter") {
        event.preventDefault();
        release();
      }
    });
    button.addEventListener("click", (event) => event.preventDefault()); // (the press is the gesture: no form submit)
    button.addEventListener("contextmenu", (event) => event.preventDefault()); // (a long press on a phone)
  }
  if (i18n && typeof i18n.onChange === "function")
    i18n.onChange(() => {
      if (button) delete button.dataset.titleKey;
      paintButton();
      paintLine();
    });
  paintButton();

  return {
    readStatus,
    press,
    release,
    get phase() {
      return s.phase;
    },
    get lastMs() {
      return s.lastMs;
    },
  };
}

/* The real microphone: opened for one press (getUserMedia, echo cancellation and noise suppression on), recorded by the
   live-voice capture node (live_voice_worklet.js: ~50 ms Float32 blocks), closed when the press ends. */
function browserMic(window) {
  return async function openMic() {
    if (!window.isSecureContext || !window.navigator.mediaDevices || !window.navigator.mediaDevices.getUserMedia) {
      throw Object.assign(new Error("insecure"), { kind: "insecure" });
    }
    let stream;
    try {
      stream = await window.navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    } catch (error) {
      const name = error && error.name;
      const kind = name === "NotAllowedError" || name === "SecurityError" ? "denied" : name === "NotFoundError" || name === "OverconstrainedError" ? "nodevice" : "failed";
      throw Object.assign(error || new Error(kind), { kind });
    }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    let ctx;
    let source;
    let node;
    let take = () => {};
    try {
      try {
        ctx = new Ctx({ sampleRate: RATE });
        source = ctx.createMediaStreamSource(stream);
      } catch (error) {
        if (ctx) ctx.close().catch(() => {});
        ctx = new Ctx();
        source = ctx.createMediaStreamSource(stream);
      }
      const mute = ctx.createGain();
      mute.gain.value = 0;
      mute.connect(ctx.destination);
      if (ctx.audioWorklet && window.AudioWorkletNode) {
        await ctx.audioWorklet.addModule("/live_voice_worklet.js");
        node = new window.AudioWorkletNode(ctx, "duck-mic", { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
        node.port.onmessage = (event) => take(event.data);
      } else {
        node = ctx.createScriptProcessor(2048, 1, 1);
        node.onaudioprocess = (event) => take(new Float32Array(event.inputBuffer.getChannelData(0)));
      }
      source.connect(node);
      node.connect(mute);
      if (ctx.state === "suspended") await ctx.resume();
    } catch (error) {
      for (const track of stream.getTracks()) track.stop();
      if (ctx) ctx.close().catch(() => {});
      throw Object.assign(error, { kind: "failed" });
    }
    return {
      rate: ctx.sampleRate,
      onBlock(fn) {
        take = fn;
      },
      close() {
        take = () => {};
        try {
          node.disconnect();
          source.disconnect();
          if (node.port) node.port.onmessage = null;
        } catch (error) {
          /* (already gone) */
        }
        for (const track of stream.getTracks()) track.stop();
        ctx.close().catch(() => {});
      },
    };
  };
}

if (typeof document !== "undefined" && typeof window !== "undefined" && document.getElementById("cmdMic")) {
  const start = () => {
    // (its own name: window.duckVoice is the duck's calls, voice.js, whose beak duck3d.js reads every frame)
    window.duckVoiceCmd = createVoiceCommands({
      document,
      fetchFn: (...args) => window.fetch(...args),
      i18n: window.I18N,
      openMic: browserMic(window),
      lang: () => (window.I18N && window.I18N.lang && window.I18N.lang !== "pseudo" ? window.I18N.lang : "zh-Hant"),
      showCommand: (answer, verb) => (window.duckShowCommand ? window.duckShowCommand(answer, verb) : undefined),
      showFeedback: (kind, status, answer) => (window.duckTricks && window.duckTricks.showFeedback ? window.duckTricks.showFeedback(kind, status, answer) : undefined),
      timers: {
        setTimeout: (fn, ms) => window.setTimeout(fn, ms),
        clearTimeout: (id) => window.clearTimeout(id),
      },
      now: () => window.performance.now(),
    });
    window.duckVoiceCmd.readStatus();
  };
  (window.I18N ? window.I18N.ready : Promise.resolve()).then(start, start);
}
