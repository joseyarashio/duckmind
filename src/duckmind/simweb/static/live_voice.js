/* 說話 (live voice, LV5): the talk panel, card 12 (docs/simulator-style.md).
 *
 * When the server runs --live-voice, GET /api/live/status says so and this script shows the panel:
 *   - the cloud switch (POST /api/live/cloud; off at start) and two chips that say whether the home's
 *     sound and the duck's eye picture are leaving the machine right now (also a chip in the status strip);
 *   - the microphone: the browser's microphone as a family member at a place in the home (16 kHz mono
 *     16-bit pieces, every 200 ms, to POST /api/neuro/mic/chunk — the same endpoint a script can use);
 *     toggle, or push-to-talk (hold the button or the space bar); or one of the scripted lines
 *     (POST /api/live/say) for a machine with no microphone;
 *   - what the cloud heard (data: never translated), what it proposed and what the duck did about each
 *     proposal (call names with their plain meaning, refusals with their reasons), the last picture sent
 *     and what the duck's own senses pick out, the people it knows, and the usage line.
 * The page only shows what the server reports and carries sound up; it decides nothing for the duck.
 * Words: catalog keys (page.live.*) and the server's own messages (_i18n), in the viewer's language. */
"use strict";

(function liveVoice() {
  const $ = (id) => document.getElementById(id);
  const tr = (key, params = {}) => (window.I18N ? window.I18N.t(key, params) : key);
  const F = (obj, name) => (obj ? (window.I18N ? window.I18N.field(obj, name) : obj[name]) ?? "" : "");
  // An element's words (remembered by i18n.js, so a language switch writes them again by itself). The same
  // words again leave the element alone: the status strip's chip is mirrored by app.js on every change.
  const words = (el, key, params) => {
    const values = params && Object.keys(params).length ? JSON.stringify(params) : "";
    if (el.dataset.i18n === key && el.dataset.liveValues === values) return el;
    el.dataset.liveValues = values;
    if (window.I18N) window.I18N.text(el, key, params);
    else el.textContent = tr(key, params);
    return el;
  };
  // A list drawn again from the last answer keeps where the viewer had scrolled it to.
  const fill = (list, draw) => {
    const top = list.scrollTop;
    list.replaceChildren();
    draw(list);
    list.scrollTop = top;
  };
  const make = (tag, props = {}, children = []) => {
    const el = Object.assign(document.createElement(tag), props);
    for (const child of children) el.append(child);
    return el;
  };
  const note = (text) => {
    if (typeof window.toast === "function" && text) window.toast(text);
  };

  const STATUS_MS = 1000;
  const FAST_MS = 500;
  const RATE = 16000; // what the simulator takes
  const CHUNK_MS = 200;
  const MAX_QUEUE = 5; // pieces waiting to go (1 s): older ones are dropped, never sent faster than real time
  const IDLE_CLOSE_MS = 6000; // push-to-talk: the browser's microphone is let go this long after a release
  const PREFS = "duckmind.livevoice";

  // every key the script names, written out (the page-key checks read them)
  const K_SENDING = { on: "page.live.sending.on", wait: "page.live.sending.wait", off: "page.live.sending.off" };
  const K_CHIP = { on: "page.live.chip.on", wait: "page.live.chip.wait", bad: "page.live.chip.bad", off: "page.live.cloud.label_off" };
  const K_MIC_ERR = {
    denied: "page.live.mic.denied",
    insecure: "page.live.mic.insecure",
    nodevice: "page.live.mic.nodevice",
    failed: "page.live.mic.failed",
    lost: "page.live.mic.lost",
  };

  const st = {
    on: false, // the panel is shown (the server runs live voice)
    status: null,
    timer: 0,
    asking: false,
    fastUntil: 0,
    linesKey: "",
    eyeSeq: -1,
    micError: "",
    micRefusal: null, // the simulator's own refusal of a piece (too fast, too many speakers…)
    pressing: false, // push-to-talk: the button or the key is held down
    speakerAt: null, // [x_mm, y_mm] where the person stands while the microphone is on
    mic: null,
  };

  /* ---------------- preferences (this viewer's own) ---------------- */

  function loadPrefs() {
    try {
      const saved = JSON.parse(localStorage.getItem(PREFS) || "null");
      if (saved && typeof saved === "object") return saved;
    } catch (error) {
      /* (no storage: the defaults) */
    }
    return {};
  }
  function savePrefs() {
    try {
      localStorage.setItem(PREFS, JSON.stringify({ who: $("liveWho").value, place: $("livePlace").value, push: $("livePush").checked, line: $("liveLine").value }));
    } catch (error) {
      /* (a private window: not remembered) */
    }
  }

  /* ---------------- talking to the server ---------------- */

  async function getStatus() {
    const response = await fetch("/api/live/status", { cache: "no-store" });
    return response.ok ? response.json() : null;
  }
  async function post(path, body) {
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    return response.json().catch(() => ({ ok: false }));
  }

  function schedule() {
    clearTimeout(st.timer);
    const fast = performance.now() < st.fastUntil || (st.mic && st.mic.gate);
    st.timer = setTimeout(poll, fast ? FAST_MS : STATUS_MS);
  }

  async function poll() {
    if (st.asking) return;
    if (document.hidden) {
      schedule();
      return;
    }
    st.asking = true;
    try {
      const status = await getStatus();
      if (!status || !status.enabled) {
        if (!st.on) return; // (no live voice here: the panel stays away and nothing is asked again)
        st.status = null;
      } else {
        if (!st.on) show();
        st.status = status;
        render(status);
      }
    } catch (error) {
      /* (the server is not there for a moment: the last answer stays; app.js says it) */
    } finally {
      st.asking = false;
    }
    schedule();
  }

  function show() {
    st.on = true;
    $("liveVoice").hidden = false;
    $("liveChipCloud").hidden = false;
    $("liveChipStar").hidden = false;
  }

  /* ---------------- drawing ---------------- */

  function agoWords(el, seconds) {
    const s = Math.max(0, Math.round(seconds));
    if (s >= 90) return words(el, "page.live.ago_m", { n: Math.round(s / 60) });
    return words(el, "page.live.ago_s", { n: s });
  }

  function sendingMode(cloud, channel) {
    if (cloud.sending && cloud.sending[channel]) return "on";
    if (cloud.on && (cloud.state === "connecting" || cloud.state === "reconnecting" || cloud.state === "online")) return "wait";
    return "off";
  }

  function renderCloud(status) {
    const cloud = status.cloud || {};
    const sw = $("liveCloudSwitch");
    if (!sw.disabled) sw.checked = !!cloud.on;
    words($("liveCloudLabel"), cloud.on ? "page.live.cloud.label_on" : "page.live.cloud.label_off");
    for (const [channel, chip, label] of [["audio", "liveSendAudio", "liveSendAudioState"], ["video", "liveSendVideo", "liveSendVideoState"]]) {
      const mode = sendingMode(cloud, channel);
      $(chip).dataset.sending = mode;
      words($(label), K_SENDING[mode]);
    }
    const state = $("liveState");
    state.dataset.state = cloud.state || "off";
    const said = F(cloud, "words");
    if (state.textContent !== said) state.textContent = said;
    // why the cloud is not working (plain words from the server: which failure, how many times, the next try)
    const err = $("liveError");
    const why = ["reconnecting", "down"].includes(cloud.state) ? F(cloud, "error_words") : "";
    if (err.textContent !== why) err.textContent = why;
    // the status strip: always says whether the cloud is working, wherever the page is scrolled
    const bad = ["down", "quota", "capped", "no_key"].includes(cloud.state);
    const sending = cloud.sending && (cloud.sending.audio || cloud.sending.video);
    const mode = sending ? "on" : bad ? "bad" : cloud.on ? "wait" : "off";
    $("liveChipCloud").dataset.sending = mode;
    words($("liveChipCloud"), K_CHIP[mode]);
  }

  function renderHeard(status) {
    const rows = status.heard || [];
    fill($("liveHeard"), (list) => {
      if (!rows.length) {
        list.append(words(make("li", { className: "empty" }), "page.live.heard.empty"));
        return;
      }
      for (const row of rows) {
        const li = make("li");
        li.dataset.echo = row.echo ? "true" : "false";
        li.append(agoWords(make("span", { className: "live-t" }), row.age_s), make("q", { textContent: row.text }));
        if (row.echo) li.append(words(make("span", { className: "live-echo" }), "page.live.heard.echo"));
        list.append(li);
      }
    });
  }

  function tagNodes(call) {
    const out = [];
    for (const tag of call.tags || []) {
      out.push(make("code", { className: "live-tagname", textContent: tag.tag }));
      if (tag.meaning) out.push(make("span", { className: "live-meaning", textContent: F(tag, "meaning") }));
      if (tag.level !== null && tag.level !== undefined) out.push(words(make("span", { className: "live-meaning" }), "page.live.call.level", { level: tag.level }));
    }
    return out;
  }

  function callDetail(call) {
    const out = [];
    if (call.tool === "vocalize") out.push(...tagNodes(call));
    else if (call.tool === "attend") {
      if (call.kind_words) out.push(make("span", { className: "live-meaning", textContent: F(call, "kind_words") }));
      if (call.hint) out.push(make("q", { textContent: String(call.hint) }));
    } else if (call.tool === "recognize") {
      if (call.name) out.push(make("q", { textContent: String(call.name) }));
    } else if (call.tool === "remember") {
      if (call.note) out.push(make("q", { textContent: String(call.note) }));
    } else if (call.tool === "feel") {
      out.push(words(make("span", { className: "live-meaning" }), "page.live.call.feel", { valence: Number(call.valence) || 0, talk: Number(call.talk) || 0 }));
    }
    return out;
  }

  function outcomeNodes(call) {
    const out = make("span", { className: "chip live-out", textContent: F(call, "outcome_words") });
    out.dataset.outcome = call.outcome;
    const nodes = [out];
    if (call.reason_words) nodes.push(make("span", { className: "live-reason", textContent: F(call, "reason_words") }));
    return nodes;
  }

  function renderCalls(status) {
    const rows = status.calls || [];
    fill($("liveCalls"), (list) => {
      if (!rows.length) {
        list.append(words(make("li", { className: "empty" }), "page.live.calls.empty"));
        return;
      }
      for (const call of rows) {
        const li = make("li");
        li.dataset.outcome = call.outcome;
        li.append(
          make("div", { className: "live-call-head" }, [
            agoWords(make("span", { className: "live-t" }), call.age_s),
            make("span", { className: "live-tool", textContent: F(call, "tool_words") }),
            ...callDetail(call),
          ]),
          make("div", { className: "live-call-out" }, outcomeNodes(call)),
        );
        list.append(li);
      }
    });
  }

  function renderLook(status) {
    const look = status.look || {};
    const cloud = status.cloud || {};
    const img = $("liveEye");
    const caption = $("liveEyeNote");
    const frame = look.frame;
    if (frame) {
      if (st.eyeSeq !== frame.seq) {
        st.eyeSeq = frame.seq;
        img.src = frame.url;
      }
      img.hidden = false;
      words(caption, "page.live.eye.caption", { n: Math.round(frame.age_s) });
    } else {
      img.hidden = true;
      st.eyeSeq = -1;
      words(caption, cloud.on ? "page.live.eye.empty_on" : "page.live.eye.empty_off");
    }
    const view = look.in_view || { people: 0, things: 0, sounds: 0 };
    words($("liveSeen"), "page.live.eye.in_view", { people: view.people, things: view.things, sounds: view.sounds });
    const attended = $("liveAttended");
    attended.replaceChildren();
    if (!look.attended) {
      words(attended, "page.live.attended.none");
    } else {
      const call = look.attended;
      attended.append(
        words(make("span"), "page.live.attended.title"),
        " ",
        make("span", { className: "live-tool", textContent: F(call, "tool_words") }),
        " ",
        ...callDetail(call),
        make("div", { className: "live-call-out" }, outcomeNodes(call)),
      );
    }
  }

  function renderPeople(status) {
    const people = status.people || { named: [], nameless: 0 };
    fill($("livePeople"), (list) => {
      if (!people.named.length && !people.nameless) {
        list.append(words(make("li", { className: "empty" }), "page.live.people.empty"));
        return;
      }
      for (const person of people.named) {
        const li = make("li", {}, [make("b", { className: "live-name", textContent: person.name }), " "]);
        li.append(words(make("span"), "page.live.people.row", { n: person.greeted }));
        if (person.pending.length) {
          li.append(" ", words(make("span", { className: "live-reason" }), "page.live.people.pending", { names: person.pending.join(" / ") }));
        }
        list.append(li);
      }
      if (people.nameless) list.append(words(make("li"), "page.live.people.nameless", { n: people.nameless }));
    });
  }

  function renderUsage(status) {
    const usage = status.usage;
    const el = $("liveUsage");
    el.replaceChildren();
    if (!usage) return;
    const last = usage.last_minute;
    const totals = usage.totals;
    el.append(
      words(make("span"), "page.live.usage", {
        audio: last.audio_s,
        frames: last.frames,
        tokens: last.tokens,
        t_audio: totals.audio_s,
        t_frames: totals.frames,
        turns: totals.turns,
        calls: totals.tool_calls,
      }),
    );
    if (usage.caps && usage.caps.audio_s) {
      el.append(" ", words(make("span"), "page.live.usage.cap", { minutes: usage.caps.audio_s / 60, used: totals.audio_s / 60 }));
    }
  }

  function renderLines(status) {
    const select = $("liveLine");
    const lines = status.lines || [];
    const key = lines.map((l) => l.id).join("|");
    if (key !== st.linesKey) {
      const keep = select.value || loadPrefs().line || "";
      st.linesKey = key;
      select.replaceChildren(...lines.map((l) => make("option", { value: l.id, textContent: l.text })));
      if (lines.some((l) => l.id === keep)) select.value = keep;
    }
    $("liveSay").disabled = !lines.length;
    select.disabled = !lines.length;
  }

  function renderPlaces(status) {
    const places = status.places || [];
    const known = new Set(places.filter((p) => p.x_mm !== undefined).map((p) => p.id));
    const select = $("livePlace");
    for (const option of select.options) option.disabled = option.value !== "auto" && !known.has(option.value);
    if (select.selectedOptions[0] && select.selectedOptions[0].disabled) select.value = "auto";
  }

  function render(status) {
    renderCloud(status);
    renderHeard(status);
    renderCalls(status);
    renderLook(status);
    renderPeople(status);
    renderUsage(status);
    renderLines(status);
    renderPlaces(status);
    renderMicNote();
  }

  /* ---------------- the microphone ---------------- */

  const placeOf = () => {
    const status = st.status;
    const id = $("livePlace").value;
    if (!status || id === "auto") return null;
    const place = (status.places || []).find((p) => p.id === id);
    return place && place.x_mm !== undefined ? [place.x_mm, place.y_mm] : null;
  };

  function makeResampler(srcRate) {
    // box-filter average of the source samples under each 16 kHz sample (fractional edges weighted)
    const ratio = srcRate / RATE;
    let carry = new Float32Array(0);
    let offset = 0;
    return {
      push(block) {
        const src = new Float32Array(carry.length + block.length);
        src.set(carry, 0);
        src.set(block, carry.length);
        const out = [];
        let a = offset;
        while (a + ratio <= src.length + 1e-9) {
          const b = a + ratio;
          let sum = 0;
          let weight = 0;
          for (let i = Math.floor(a); i < Math.ceil(b - 1e-9) && i < src.length; i++) {
            const w = Math.min(b, i + 1) - Math.max(a, i);
            if (w > 0) {
              sum += src[i] * w;
              weight += w;
            }
          }
          out.push(weight ? sum / weight : 0);
          a = b;
        }
        const keep = Math.min(src.length, Math.floor(a));
        carry = src.slice(keep);
        offset = a - keep;
        return out;
      },
    };
  }

  function newMic() {
    return {
      open: false,
      opening: null,
      gate: false, // sound is being sent to the simulator
      ctx: null,
      stream: null,
      node: null,
      source: null,
      resample: null,
      pending: [], // Int16 samples waiting to make a piece
      queue: [], // pieces waiting to go
      sending: false,
      closeTimer: 0,
      level: 0,
    };
  }

  const chunkSamples = () => Math.round((RATE * CHUNK_MS) / 1000);

  function takeBlock(mic, block) {
    // the level bar follows what the microphone hears (log scale, -60 to -10 dBFS)
    let sum = 0;
    for (let i = 0; i < block.length; i++) sum += block[i] * block[i];
    const rms = Math.sqrt(sum / Math.max(1, block.length));
    const db = rms > 1e-6 ? 20 * Math.log10(rms) : -120;
    mic.level = Math.max(0, Math.min(1, (db + 60) / 50));
    $("liveLevelBar").style.transform = `scaleX(${mic.gate ? mic.level.toFixed(3) : 0})`;
    if (!mic.gate) return;
    for (const sample of mic.resample.push(block)) mic.pending.push(Math.max(-1, Math.min(1, sample)) * 32767);
    while (mic.pending.length >= chunkSamples()) {
      const piece = Int16Array.from(mic.pending.splice(0, chunkSamples()));
      enqueue(mic, piece);
    }
  }

  function flushMic(mic) {
    if (mic.pending.length >= RATE / 50) enqueue(mic, Int16Array.from(mic.pending));
    mic.pending = [];
  }

  function enqueue(mic, piece) {
    mic.queue.push(piece);
    while (mic.queue.length > MAX_QUEUE) mic.queue.shift();
    pumpMic(mic);
  }

  async function pumpMic(mic) {
    if (mic.sending) return;
    mic.sending = true;
    try {
      while (mic.queue.length) {
        const piece = mic.queue.shift();
        const params = new URLSearchParams({ who: $("liveWho").value });
        if (st.speakerAt) {
          params.set("x_mm", String(st.speakerAt[0]));
          params.set("y_mm", String(st.speakerAt[1]));
        }
        let answer = null;
        try {
          const response = await fetch(`/api/neuro/mic/chunk?${params}`, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: piece.buffer });
          answer = await response.json().catch(() => null);
        } catch (error) {
          answer = null; // (the server is away for a moment: this piece is lost, the next goes on)
        }
        if (answer && answer.ok === false) {
          st.micError = "";
          st.micRefusal = answer;
          renderMicNote();
        } else if (st.micRefusal) {
          st.micRefusal = null;
          renderMicNote();
        }
      }
    } finally {
      mic.sending = false;
    }
  }

  async function micOpen() {
    if (!st.mic) st.mic = newMic();
    const mic = st.mic;
    if (mic.open) return;
    if (mic.opening) return mic.opening;
    mic.opening = (async () => {
      if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw Object.assign(new Error("insecure"), { kind: "insecure" });
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false } });
      } catch (error) {
        const name = error && error.name;
        const kind = name === "NotAllowedError" || name === "SecurityError" ? "denied" : name === "NotFoundError" || name === "OverconstrainedError" ? "nodevice" : "failed";
        throw Object.assign(error, { kind });
      }
      try {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        let ctx;
        let source;
        try {
          ctx = new Ctx({ sampleRate: RATE }); // (the browser resamples the microphone for us)
          source = ctx.createMediaStreamSource(stream);
        } catch (error) {
          if (ctx) ctx.close().catch(() => {});
          ctx = new Ctx(); // (a browser that cannot: its own rate, resampled here)
          source = ctx.createMediaStreamSource(stream);
        }
        const mute = ctx.createGain();
        mute.gain.value = 0;
        mute.connect(ctx.destination);
        let node;
        if (ctx.audioWorklet && window.AudioWorkletNode) {
          await ctx.audioWorklet.addModule("/live_voice_worklet.js");
          node = new AudioWorkletNode(ctx, "duck-mic", { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
          node.port.onmessage = (event) => takeBlock(mic, event.data);
        } else {
          node = ctx.createScriptProcessor(2048, 1, 1);
          node.onaudioprocess = (event) => takeBlock(mic, event.inputBuffer.getChannelData(0));
        }
        source.connect(node);
        node.connect(mute);
        if (ctx.state === "suspended") await ctx.resume();
        mic.ctx = ctx;
        mic.stream = stream;
        mic.node = node;
        mic.source = source;
        mic.resample = makeResampler(ctx.sampleRate);
        for (const track of stream.getTracks()) {
          track.addEventListener("ended", () => {
            if (st.mic === mic && mic.open) {
              micClose();
              st.micError = "lost";
              renderMicNote();
            }
          });
        }
        mic.open = true;
      } catch (error) {
        for (const track of stream.getTracks()) track.stop();
        throw Object.assign(error, { kind: error.kind || "failed" });
      }
    })();
    try {
      await mic.opening;
    } finally {
      mic.opening = null;
    }
  }

  function micClose() {
    const mic = st.mic;
    if (!mic) return;
    clearTimeout(mic.closeTimer);
    mic.gate = false;
    flushMic(mic);
    mic.open = false;
    try {
      if (mic.node) {
        mic.node.disconnect();
        if (mic.node.port) mic.node.port.onmessage = null;
        mic.node.onaudioprocess = null;
      }
      if (mic.source) mic.source.disconnect();
    } catch (error) {
      /* (already gone) */
    }
    if (mic.stream) for (const track of mic.stream.getTracks()) track.stop();
    if (mic.ctx) mic.ctx.close().catch(() => {});
    mic.ctx = mic.stream = mic.node = mic.source = null;
    $("liveLevelBar").style.transform = "scaleX(0)";
    renderMicButton();
  }

  async function micStart() {
    clearTimeout(st.mic && st.mic.closeTimer);
    st.micError = "";
    st.micRefusal = null;
    renderMicButton(true);
    try {
      await micOpen();
    } catch (error) {
      st.micError = error && error.kind ? error.kind : "failed";
      micClose();
      renderMicNote();
      return;
    }
    const mic = st.mic;
    if ($("livePush").checked && !st.pressing) {
      // (let go before the microphone was open: nothing is sent; it is let go again shortly)
      mic.closeTimer = setTimeout(micClose, IDLE_CLOSE_MS);
      renderMicButton();
      return;
    }
    st.speakerAt = placeOf(); // (where the person stands while they talk: fixed until they stop)
    mic.gate = true;
    st.fastUntil = performance.now() + 3000;
    renderMicButton();
    renderMicNote();
    schedule();
  }

  function micStop() {
    const mic = st.mic;
    if (!mic) return;
    mic.gate = false;
    flushMic(mic);
    $("liveLevelBar").style.transform = "scaleX(0)";
    if ($("livePush").checked) {
      mic.closeTimer = setTimeout(micClose, IDLE_CLOSE_MS); // (the next press comes quickly: keep it open a moment)
    } else {
      micClose();
    }
    renderMicButton();
    renderMicNote();
  }

  function renderMicButton(starting = false) {
    const button = $("liveMic");
    const mic = st.mic;
    const push = $("livePush").checked;
    const talking = !!(mic && mic.gate);
    button.setAttribute("aria-pressed", talking ? "true" : "false");
    $("liveLevel").hidden = !(mic && mic.open);
    if (starting && !talking) words(button, "page.live.mic.starting");
    else if (push) words(button, talking ? "page.live.mic.holding" : "page.live.mic.hold");
    else words(button, talking ? "page.live.mic.stop" : "page.live.mic.start");
    button.setAttribute("aria-keyshortcuts", push ? "Space Enter" : "");
  }

  function renderMicNote() {
    const el = $("liveMicNote");
    const cloudOn = !!(st.status && st.status.cloud && st.status.cloud.on);
    el.replaceChildren();
    if (st.micError) {
      el.append(words(make("span"), K_MIC_ERR[st.micError] || K_MIC_ERR.failed));
    } else if (st.micRefusal) {
      el.append(make("span", { textContent: F(st.micRefusal, "error") }));
    } else if (st.mic && st.mic.gate) {
      el.append(words(make("span"), "page.live.mic.live"));
    }
    el.append(words(make("span"), cloudOn ? "page.live.mic.note_on" : "page.live.mic.note_off"));
  }

  /* ---------------- the controls ---------------- */

  function wire() {
    const prefs = loadPrefs();
    if (prefs.who && Array.from($("liveWho").options).some((o) => o.value === prefs.who)) $("liveWho").value = prefs.who;
    if (prefs.place && Array.from($("livePlace").options).some((o) => o.value === prefs.place)) $("livePlace").value = prefs.place;
    $("livePush").checked = !!prefs.push;

    $("liveCloudSwitch").addEventListener("change", async (event) => {
      const sw = event.target;
      const want = sw.checked;
      sw.disabled = true;
      try {
        const answer = await post("/api/live/cloud", { on: want });
        if (answer.ok) note(tr(want ? "page.live.toast.on" : "page.live.toast.off"));
        else {
          sw.checked = !want;
          note(F(answer, "error") || tr("page.neuro.why"));
        }
      } catch (error) {
        sw.checked = !want;
        note(tr("page.neuro.why"));
      } finally {
        sw.disabled = false;
        st.fastUntil = performance.now() + 8000;
        poll();
      }
    });

    const button = $("liveMic");
    button.addEventListener("click", () => {
      if ($("livePush").checked) return; // (push-to-talk is held, not clicked)
      if (st.mic && st.mic.gate) micStop();
      else micStart();
    });
    const press = () => {
      st.pressing = true;
      if (st.mic && st.mic.gate) return;
      micStart();
    };
    const release = () => {
      st.pressing = false;
      if ($("livePush").checked && st.mic && st.mic.gate) micStop();
    };
    button.addEventListener("pointerdown", (event) => {
      if (!$("livePush").checked || event.button !== 0) return;
      event.preventDefault();
      try {
        button.setPointerCapture(event.pointerId);
      } catch (error) {
        /* (a synthetic pointer: no capture) */
      }
      press();
    });
    for (const type of ["pointerup", "pointercancel"]) button.addEventListener(type, release);
    button.addEventListener("keydown", (event) => {
      if ($("livePush").checked && (event.key === " " || event.key === "Enter") && !event.repeat) {
        event.preventDefault();
        press();
      }
    });
    button.addEventListener("keyup", (event) => {
      if ($("livePush").checked && (event.key === " " || event.key === "Enter")) {
        event.preventDefault();
        release();
      }
    });
    button.addEventListener("blur", release);
    button.addEventListener("contextmenu", (event) => {
      if ($("livePush").checked) event.preventDefault();
    });
    $("livePush").addEventListener("change", () => {
      if (st.mic && st.mic.gate) micStop();
      renderMicButton();
      savePrefs();
    });
    $("liveWho").addEventListener("change", savePrefs);
    $("livePlace").addEventListener("change", () => {
      if (st.mic && st.mic.gate) st.speakerAt = placeOf();
      savePrefs();
    });
    $("liveLine").addEventListener("change", savePrefs);
    $("liveSay").addEventListener("click", async () => {
      const button2 = $("liveSay");
      button2.disabled = true;
      try {
        const answer = await post("/api/live/say", { utterance: $("liveLine").value, position: $("livePlace").value });
        note(answer.ok ? F(answer, "note_zh") : F(answer, "error") || tr("page.neuro.why"));
        st.fastUntil = performance.now() + 12000;
      } catch (error) {
        note(tr("page.neuro.why"));
      } finally {
        button2.disabled = false;
        poll();
      }
    });
    window.addEventListener("pagehide", micClose);
    if (window.I18N) {
      window.I18N.onChange(() => {
        renderMicButton();
        if (st.status) render(st.status);
      });
    }
    renderMicButton();
    renderMicNote();
  }

  window.__liveVoice = { state: st };
  const start = () => {
    wire();
    poll();
  };
  if (window.I18N && window.I18N.ready) window.I18N.ready.then(start);
  else start();
})();
