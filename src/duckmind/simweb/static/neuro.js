/* 神經鴨原型（示範）: the page's demo mode for the neural-duck prototype.
 *
 * When the server runs the prototype (neuro:live's --brain neuro; --neuro-stub checks the page
 * with made-up numbers), /api/neuro/status says so and this script:
 *   - shows the prototype's label, its duck (name, one plain line, temperament) and puts Duck
 *     Core's own cards away (body.neuro-mode: [data-core] hides, [data-neuro] shows);
 *   - fills 「小鴨的腦」 from what the brain reports, a few times a second
 *     (/api/neuro/stream, else /api/neuro/telemetry): what it does now and why, its reflexes as
 *     they happen, the behaviours pulling at it, its drives and arousal, how used it is to each
 *     stimulus, the places it learned;
 *   - sends 「逗逗牠」's stimuli to the physical world (POST /api/neuro/stimulus: the duck notices
 *     them only through its senses) and the picker's duck (POST /api/neuro/duck);
 *   - tells the 3D view (duck:neuro): the behaviour for its caption, new reflexes, calls, sleep.
 * Without the prototype it does nothing at all. The page only shows what the brain reports; it
 * never decides anything for the duck. Design: docs/simulator-style.md. */
"use strict";

(function neuroMode() {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  // The words: catalog keys (docs: i18n.js) in the viewer's language; a server text field comes with
  // its message in the same object (`_i18n`), F / FS take the one in the viewer's language.
  const tr = (key, params = {}) => (window.I18N ? window.I18N.t(key, params) : key);
  const F = (obj, name) => (obj ? (window.I18N ? window.I18N.field(obj, name) : obj[name]) ?? "" : "");
  const FS = (obj, name) => (obj ? (window.I18N ? window.I18N.items(obj, name) : obj[name]) ?? [] : []);
  // (an element's words, remembered so a language switch writes them again by itself)
  const words = (el, key, params) => {
    if (window.I18N) window.I18N.text(el, key, params);
    else el.textContent = tr(key, params);
    return el;
  };
  const dot = () => tr("fmt.sep.dot");
  const K_SIZE = { big: "page.neuro.sizeZh.0889c3", mid: "page.neuro.sizeZh.086907", small: "page.neuro.sizeZh.922517" };
  const POLL_MS = 250;
  const STREAM_WAIT_MS = 4000;
  const REFLEX_ROWS = 40;
  const REPEAT_MS = 3000;
  const SKILLS = [
    { id: "follow", category: "social", behaviours: ["follow_mom"], accent: "magenta" },
    { id: "hide", category: "social", behaviours: ["hide"], accent: "purple" },
    { id: "circle", category: "social", behaviours: ["circle_settle"], accent: "cyan" },
    { id: "ball", category: "play", behaviours: ["play_ball"], accent: "orange" },
    { id: "carry", category: "play", behaviours: ["carry_toy"], accent: "blue" },
    { id: "sun", category: "play", behaviours: ["sunbathe"], accent: "yellow" },
    { id: "theremin", category: "sound", behaviours: ["theremin"], accent: "magenta" },
    { id: "ride", category: "sound", behaviours: ["wheee_ride"], accent: "cyan" },
    { id: "dance", category: "sound", behaviours: ["dance"], accent: "purple" },
  ];
  const SKILL_STATES = new Set(["active", "waiting", "done", "interrupted", "refused", "missing", "unknown"]);
  const SKILL_RESULTS = new Set(["success", "partial", "failed", "positive", "neutral", "negative", "delivered", "dropped_home", "landed", "held", "calm", "gave_up", "complete", "rolled", "lost", "missed", "unseen", "fear", "joy"]);
  const SKILL_REASONS = new Set(["no_sensor", "no_voice", "no_voice_helper", "no_body_skill", "no_jaw", "jaw_unknown", "no_one_shot", "sensor_not_ready", "skill_state_unknown", "driver_unavailable", "module_missing", "dependency_unavailable", "missing_brain", "missing_senses", "missing_motor", "voice_unavailable", "policy_busy", "policy_unsettled", "reflex_active", "battery_unknown", "battery", "unloaded", "lifted", "fallen_or_righting", "body_gate", "body_not_still", "head_not_home", "hide_failed", "circle_stopped", "world_sensor_error", "active_robotd_skill_not_cancelable_by_stop", "timeout", "forbidden_intent", "wrong_act", "body_unsafe", "guard_refused", "head_not_settled", "intent_not_allowed", "selection_changed", "readiness_timeout", "policy_start_timeout"]);
  const SKILL_PHASES = new Set(["intent", "approach", "engage", "hold", "deliver", "search", "settle", "listen", "follow_beat", "ride", "bask", "kick", "carry", "circle", "hide", "idle", "waiting", "following", "go", "in", "out", "failed", "turning", "seeking", "done", "stopped", "playing", "listening", "riding", "lifted", "settling", "dancing", "chase", "line_up", "beak", "chest", "watch", "to_toy", "survey", "around", "grab", "drop", "backoff", "regrip", "into", "deeper", "check", "shift", "waiting_policy", "waiting_safety", "dispatched", "policy_started", "interrupted_waiting_for_exit", "refused", "readiness_timeout", "policy_start_timeout", "interrupted_policy_exited", "selection_ended", "policy_exited"]);
  const nu = {
    on: false,
    status: null,
    since: 0,
    stream: null,
    streamOk: false,
    streamTimer: 0,
    poll: 0,
    asking: false,
    down: false,
    last: null,
    duckKey: "",
    lights: true,
    sending: false,
    keys: {},
    lastReflex: null,
    retry: 0,
    noButtons: false,
    rowOf: new Map(), // a reflex's seq → its row (a folded repeat → the row it went into)
    voice: null, // whose voice the calls have ({seed, gd}: voice.js)
    skillCards: null,
    soundStatus: null,
    motions: null, // 動作試玩: id → the server's row
    motionBusy: false,
    motionSeen: null, // the newest motion result already said
    hudReflex: null, // the newest reflex the HUD says ({text, at})
    emotionSeen: null, // the newest emotion (the page's press or the duck's own) already said
  };

  const pct = (value) => Math.round(Math.max(0, Math.min(1, Number(value) || 0)) * 100);
  const two = (n) => String(n).padStart(2, "0");
  const clock = (at) => {
    const d = at != null ? new Date(at) : new Date();
    return `${two(d.getHours())}:${two(d.getMinutes())}:${two(d.getSeconds())}`;
  };
  // The duck's own time of day (its clock: 小鴨的時間), from a minute of the day.
  const duckClock = (minute) => {
    const m = ((Number(minute) % 1440) + 1440) % 1440;
    const secs = Math.floor(m * 60);
    return `${two(Math.floor(secs / 3600))}:${two(Math.floor(secs / 60) % 60)}:${two(secs % 60)}`;
  };
  // How big a reaction was, in words (the number is in the row's title).
  const sizeZh = (strength) => tr(strength >= 0.7 ? K_SIZE.big : strength >= 0.3 ? K_SIZE.mid : K_SIZE.small);
  // What went wrong, in plain words: the server's own message, or (a network failure, whose
  // browser text is English) that the simulator could not be reached.
  const why = (error, fallback) => {
    const text = (error && error.message) || "";
    if (!text || error instanceof TypeError || error instanceof SyntaxError || /fetch|network|load failed|json|unexpected/i.test(text)) {
      return text ? tr("page.neuro.why") : fallback;
    }
    return text;
  };
  const make = (tag, props = {}, children = []) => {
    const el = Object.assign(document.createElement(tag), props);
    for (const child of children) el.append(child);
    return el;
  };
  function say(text, done = false) {
    // (the 3D view's caption, as app.js's announce does: a click echoes at once)
    window.dispatchEvent(new CustomEvent("duck:caption", { detail: { text, strong: !done, reply: done } }));
  }
  function note(text) {
    if (typeof window.toast === "function") window.toast(text);
  }

  /* ---------------- on or off ---------------- */

  async function boot() {
    let status = null;
    try {
      const response = await fetch("/api/neuro/status", { cache: "no-store" });
      status = response.ok ? await response.json() : null;
    } catch (error) {
      status = null; // (the server is not there: app.js says so; a restart re-checks)
    }
    if (!status || !status.enabled) {
      if (nu.on) location.reload(); // (a restarted server without the prototype)
      return;
    }
    enable(status);
  }

  function enable(status) {
    const first = !nu.on;
    nu.on = true;
    nu.status = status;
    document.body.classList.add("neuro-mode");
    for (const el of $$("[data-neuro]")) el.hidden = false;
    if (first) {
      buildStimuli(status);
      buildMotions(status);
      buildDay(status.day_choices || []);
      document.title = tr("page.neuro.enable");
    }
    buildPicker(status);
    if (status.duck) renderDuck(status.duck);
    if (status.waiting_zh) renderNote({ waiting_zh: status.waiting_zh, _i18n: status._i18n });
    window.dispatchEvent(new CustomEvent("duck:neuro", { detail: { on: true, who: whoOf(status.duck) } }));
    connect();
  }

  /* ---------------- the telemetry: a stream, else a poll ---------------- */

  function connect() {
    stopStream();
    stopPoll();
    if (!nu.on || nu.down) return;
    if (!("EventSource" in window)) {
      startPoll();
      return;
    }
    const stream = new EventSource(`/api/neuro/stream?since=${nu.since}`);
    nu.stream = stream;
    nu.streamOk = false;
    nu.streamTimer = setTimeout(() => {
      if (!nu.streamOk) {
        stopStream();
        startPoll(); // (a proxy that holds the stream back: ask instead)
      }
    }, STREAM_WAIT_MS);
    stream.onmessage = (event) => {
      nu.streamOk = true;
      nu.retry = 0;
      clearTimeout(nu.streamTimer);
      let data = null;
      try {
        data = JSON.parse(event.data);
      } catch (error) {
        return;
      }
      take(data);
    };
    stream.onerror = () => {
      // Never a flood of retries while the server is away: close, and come back when app.js's
      // heartbeat says it is there again (duck:net), or after a short wait.
      const worked = nu.streamOk;
      stopStream();
      if (!worked) {
        startPoll();
        return;
      }
      nu.retry = Math.min(5, nu.retry + 1);
      setTimeout(() => {
        if (nu.on && !nu.stream && !nu.poll && !nu.down) connect();
      }, 1000 * nu.retry);
    };
  }

  function stopStream() {
    clearTimeout(nu.streamTimer);
    if (nu.stream) {
      nu.stream.onmessage = null;
      nu.stream.onerror = null;
      nu.stream.close();
    }
    nu.stream = null;
  }

  function startPoll() {
    stopPoll();
    nu.poll = setInterval(pollOnce, POLL_MS);
    pollOnce();
  }

  function stopPoll() {
    clearInterval(nu.poll);
    nu.poll = 0;
  }

  async function pollOnce() {
    if (nu.asking || nu.down || document.hidden) return;
    nu.asking = true;
    try {
      const response = await fetch(`/api/neuro/telemetry?since=${nu.since}`, { cache: "no-store" });
      if (response.ok) take(await response.json());
    } catch (error) {
      /* app.js's heartbeat says whether the server is there */
    } finally {
      nu.asking = false;
    }
  }

  window.addEventListener("duck:net", (event) => {
    const down = !!(event.detail && event.detail.down);
    if (down === nu.down) return;
    nu.down = down;
    if (!nu.on) return;
    if (down) {
      stopStream();
      stopPoll();
    } else {
      boot(); // (back: the same server, or a restarted one with or without the prototype)
    }
  });
  // A restarted server (or a new Duck Core duck): check the prototype again from the start.
  window.addEventListener("duck:reset", () => {
    nu.since = 0;
    boot();
  });

  /* ---------------- one snapshot ---------------- */

  function take(t) {
    if (!t || t.enabled === false) {
      if (t && t.enabled === false) location.reload();
      return;
    }
    // The source started over (a new duck, a restarted server): its numbers begin again.
    if (typeof t.seq === "number" && t.seq < nu.since) {
      nu.since = 0;
      clearReflexes();
    }
    nu.last = t;
    // (the first snapshot is the past: its calls are listed, not sounded again)
    const fresh = nu.since > 0;
    let newest = null;
    let spoken = null;
    for (const item of t.reflexes || []) {
      if (item.seq <= nu.since) continue;
      newest = item;
      // (when it happened, on the duck's own clock: a page opened later lists what came before
      // at its own time)
      const ago = Number(t.t_s) - Number(item.t_s);
      const minute = t.day && Number.isFinite(Number(t.day.minute)) ? Number(t.day.minute) : null;
      const rate = t.day && Number.isFinite(Number(t.day.rate)) ? Number(t.day.rate) : 1;
      const at = Number.isFinite(ago) ? Date.now() - Math.max(0, ago) * 1000 : null;
      addReflex(item, at, minute != null && Number.isFinite(ago) ? minute - (Math.max(0, ago) * rate) / 60 : null);
    }
    for (const item of t.vocal || []) {
      if (item.seq <= nu.since) continue;
      spoken = item;
    }
    if (!fresh) spoken = null;
    // (voice:neuro) without the 3D view (still loading, or a browser without one) its call is
    // voiced here, the 2D duck's beak opening with the sound (voice.js); the 3D view voices its
    // own calls with the beak its robot.mouth command opened (duck3d.js)
    if (spoken && window.duckVoice && !document.body.classList.contains("has3d")) window.duckVoice.call([spoken.tag]);
    // what the body measurably did after each reflex, onto the reflex's own row
    let bodyNote = null;
    for (const item of t.reflex_body || []) {
      if (item.seq <= nu.since) continue;
      if (applyBody(item)) bodyNote = item;
    }
    for (const item of [...(t.reflexes || []), ...(t.vocal || []), ...(t.reflex_body || [])]) nu.since = Math.max(nu.since, item.seq || 0);
    renderNote(t);
    if (t.duck) renderDuck(t.duck);
    renderNow(t);
    renderSkillPortfolio(t);
    renderBody(t.body_now || null);
    renderHud(t, newest);
    renderCands(t.candidates || []);
    renderDrives(t.drives || [], t.arousal, t.valence);
    renderHabits(t.habituation || []);
    renderPlaces(t.places || []);
    renderRhythm(t.rhythm || null);
    renderSenses(t.senses || [], t.driver);
    renderButtons(t);
    renderCommand(t);
    // the remote hand (remote.js) and the click on the floor (duck3d.js) read their own blocks of the snapshot
    window.dispatchEvent(new CustomEvent("duck:telemetry", { detail: { remote: t.remote || null, errand: t.errand || null } }));
    renderMotions(t);
    renderDay(t.day || null);
    renderHelp(t.body || null);
    const b = t.behaviour || {};
    window.dispatchEvent(
      new CustomEvent("duck:neuro", {
        detail: {
          on: true,
          lead: b.zh ? leadOf(b, t.body_now) : F(t, "waiting_zh") || "…",
          // (a brain reflex is said as the brain's; a world event or a body event as it is)
          reflex: newest ? (["world", "body", "emotion"].includes(newest.kind) ? F(newest, "zh") : tr("page.neuro.take", { newest_zh: F(newest, "zh") })) : null,
          body: bodyNote ? F(bodyNote, "zh") : null, // (what the body did after its last reflex)
          moment: !!(newest && ["startle", "freeze"].includes(newest.kind)),
          say: spoken ? F(spoken, "zh") : null,
          call: spoken ? { tag: spoken.tag, level: spoken.level, t_ms: spoken.t_ms } : null,
          voice: nu.voice,
          asleep: !!t.asleep,
          who: whoOf(t.duck || (nu.status && nu.status.duck)),
          mouth: t.mouth || null, // its own beak (robot.mouth, 10 Hz peaks): the 3D beak follows
          day: t.day && Number.isFinite(Number(t.day.minute)) ? Number(t.day.minute) : null, // its clock
        },
      }),
    );
  }

  // What it does now, for the 3D view's caption: a reflex holding the body says so once; and
  // when the body is not doing what the brain says, the caption says that too (fid:page).
  function leadOf(b, body) {
    const zh = F(b, "zh");
    const reflex = F(b, "reflex_zh");
    let lead = zh;
    if (b.family === "reflex" || !b.reflex_zh || reflex === zh) lead = b.family === "reflex" ? tr("page.neuro.leadOf.436078", { zh }) : zh;
    else lead = tr("page.neuro.leadOf.224738", { zh, reflex_zh: reflex });
    if (body && body.mismatch_zh) lead = tr("page.neuro.leadOf.9f5191", { lead, mismatch_zh: F(body, "mismatch_zh") });
    return lead;
  }

  function whoOf(duck) {
    if (!duck) return "";
    return tr("page.neuro.whoOf", { duck_name: duck.name, branch_zh: duck.branch_zh ? tr("page.neuro.renderDuck.branch_stage", { branch: F(duck, "branch_zh"), stage: F(duck, "stage_zh") }) : F(duck, "stage_zh") });
  }

  /* ---------------- the duck ---------------- */

  function renderDuck(duck) {
    const key = `${duck.seed}|${duck.gd}|${duck.name}|${duck.branch}|${window.I18N ? window.I18N.lang : ""}`;
    if (key === nu.duckKey) return;
    nu.duckKey = key;
    nu.voice = { seed: duck.seed, gd: duck.gd };
    // (voice:neuro) its calls in its own official voice, from its seed: voice.js asks for it
    if (window.duckVoice && window.duckVoice.duck) window.duckVoice.duck(duck.seed);
    $("#neuroName").textContent = duck.name || "小鴨";
    const stage = F(duck, "stage_zh");
    $("#neuroKind").textContent = duck.branch_zh
      ? tr("page.neuro.renderDuck.8b5c63", { branch_zh: F(duck, "branch_zh"), stage_zh: stage })
      : tr("page.neuro.renderDuck.6dc9c6", { stage_zh: stage, nature_zh: F(duck, "nature_zh") });
    $("#neuroLine").textContent = F(duck, "line_zh");
    $("#neuroChipDuck").textContent = tr("page.neuro.renderDuck.b1d8fa", { duck_name: duck.name, stage_zh: stage });
    const axes = duck.axes || [];
    $("#neuroAxes").replaceChildren(...axes.map((axis) => meter(F(axis, "zh"), axis.value, `${pct(axis.value)}`)));
    const facts = [
      [tr("page.neuro.renderDuck.d3ab73"), tr("page.neuro.renderDuck.6f1ead", { stage_zh: stage, gd: duck.gd })],
      [tr("page.neuro.renderDuck.9ea856"), duck.branch_zh ? F(duck, "branch_zh") : tr("page.neuro.renderDuck.ffee85", { nature_zh: F(duck, "nature_zh") })],
      [tr("page.neuro.renderDuck.330db5"), duck.side_zh ? tr("page.neuro.renderDuck.344b5d", { side_zh: F(duck, "side_zh") }) : "—"],
      [tr("page.neuro.renderDuck.7fe0de"), duck.habit_zh ? F(duck, "habit_zh") : "—"],
      [tr("page.neuro.renderDuck.f71e6b"), tr("page.neuro.renderDuck.585993", { motion_tempo: Number(duck.motion_tempo || 1) })],
      [tr("page.neuro.renderDuck.85e2ec"), tr("page.neuro.renderDuck.72ab2e", { seed: duck.seed })],
      [tr("page.neuro.renderDuck.voice_label"), duck.voice_seed != null ? tr("page.neuro.renderDuck.voice_fact", { voice_seed: duck.voice_seed }) : "—"],
    ];
    $("#neuroFacts").replaceChildren(...facts.flatMap(([k, v]) => [make("dt", { textContent: k }), make("dd", { textContent: v })]));
    const buttons = $$("#neuroDucks .duck-pick");
    const preset = buttons.some((b) => b.dataset.seed === String(duck.seed) && b.dataset.gd === String(duck.gd));
    for (const button of buttons) {
      const on = button.dataset.duck === "random" ? !preset : button.dataset.seed === String(duck.seed) && button.dataset.gd === String(duck.gd);
      button.classList.toggle("on", on);
      button.setAttribute("aria-checked", String(on));
    }
  }

  /* ---------------- notes: made-up numbers, stand-ins, waiting ---------------- */

  function renderNote(t) {
    const box = $("#neuroNote");
    const text = F(t, "note_zh") || F(t, "waiting_zh");
    box.hidden = !text;
    if (!text) return;
    if (box.textContent !== text) box.textContent = text;
    box.dataset.tone = t.note_zh ? t.note_tone || "warn" : "info";
  }

  /* ---------------- what it does now, and why ---------------- */

  function renderNow(t) {
    const b = t.behaviour || null;
    const now = $("#neuroNow");
    const text = b ? F(b, "zh") : t.waiting_zh ? tr("page.neuro.renderNow.d79743") : "—";
    if (now.textContent !== text) now.textContent = text;
    const why = b ? F(b, "why_zh") || tr("page.neuro.renderNow.38da0e") : F(t, "waiting_zh");
    if ($("#neuroWhy").textContent !== why) $("#neuroWhy").textContent = why;
    const bits = [];
    if (b && b.since_s != null) bits.push(tr("page.neuro.renderNow.da81fe", { since_s: Math.round(b.since_s) }));
    if (b && b.satiation != null) bits.push(tr("page.neuro.renderNow.6f653a", { satiation: pct(b.satiation) }));
    if (b && b.moment_zh) bits.push(tr("page.neuro.renderNow.63df76", { moment_zh: F(b, "moment_zh") }));
    if (b && b.reflex_zh && b.family !== "reflex" && F(b, "reflex_zh") !== F(b, "zh")) bits.push(tr("page.neuro.renderNow.98fdc7", { reflex_zh: F(b, "reflex_zh") }));
    if (b && b.family === "reflex") bits.push(tr("page.neuro.renderNow.668127"));
    if (t.locomotion && t.locomotion.zh) bits.push(F(t.locomotion, "zh"));
    const wall = t.locomotion ? Number(t.locomotion.wall_free_m) : NaN;
    if (Number.isFinite(wall) && wall < 1.0) bits.push(tr("page.neuro.renderNow.311967", { wall, wall_urgency: Number(t.locomotion.wall_urgency) > 0.3 ? tr("page.neuro.renderNow.578ceb") : "" }));
    if (t.asleep) bits.push(tr("page.neuro.renderNow.a52b59"));
    if (t.looks && Number.isFinite(Number(t.looks.per_s))) bits.push(tr("page.neuro.renderNow.e180fa", { per_s: Number(t.looks.per_s) }));
    const meta = bits.join(dot());
    if ($("#neuroMeta").textContent !== meta) $("#neuroMeta").textContent = meta;
    const chip = tr("page.neuro.renderNow.7c1eb5", { b: b ? F(b, "zh") : "…" });
    if ($("#neuroChipNow").textContent !== chip) $("#neuroChipNow").textContent = chip;
    const sub = $("#neuroBrainSub");
    const driver = t.driver || {};
    const where = t.source === "stub" ? tr("page.neuro.renderNow.fed543") : F(driver, "state_zh") || (t.source ? tr("page.neuro.renderNow.6524f1") : tr("page.neuro.renderNow.dafbba"));
    const rt = typeof driver.realtime === "number" ? tr("page.neuro.renderNow.645615", { realtime: driver.realtime }) : "";
    const said = tr("page.neuro.renderNow.7de765", { streamOk: nu.stream && nu.streamOk ? 5 : 4, where, rt });
    if (sub.textContent !== said) sub.textContent = said;
  }

  /* ---------------- the nine integrated skills ---------------- */

  const skillTextKey = (id, field) => `page.neuro.skills.${id}.${field}`;
  const skillNode = (tag, className = "") => make(tag, { className });

  function buildSkillCards() {
    const box = $("#neuroSkillCards");
    if (!box) return;
    const nodes = new Map();
    const cards = SKILLS.map((skill) => {
      const card = make("article", { className: `card accent-${skill.accent} neuro-skill-card` });
      card.dataset.skill = skill.id;
      card.dataset.state = "unknown";
      const heading = make("div", { className: "neuro-skill-head" });
      const name = make("h3");
      const category = make("span", { className: "chip neuro-skill-category" });
      const state = make("span", { className: "neuro-skill-state" });
      const intent = make("span", { className: "chip neuro-skill-intent", hidden: true });
      const trigger = make("p", { className: "neuro-skill-trigger" });
      const phase = make("p", { className: "neuro-skill-detail neuro-skill-phase", hidden: true });
      const result = make("p", { className: "neuro-skill-detail neuro-skill-result", hidden: true });
      const reason = make("p", { className: "neuro-skill-detail neuro-skill-reason", hidden: true });
      const stream = skill.category === "sound" ? make("p", { className: "neuro-skill-stream neuro-skill-stream-status", hidden: true }) : null;
      heading.append(name, category, state, intent);
      card.append(heading, trigger, phase, result, reason);
      if (stream) card.append(stream);
      nodes.set(skill.id, { card, name, category, state, intent, trigger, phase, result, reason, stream });
      return card;
    });
    box.replaceChildren(...cards);
    nu.skillCards = nodes;
  }

  function phaseKey(phase) {
    const code = String(phase || "");
    return SKILL_PHASES.has(code) ? `page.neuro.skills.phase.${code}` : null;
  }

  function streamWords() {
    const status = nu.soundStatus;
    if (!status || status.unavailable || status.running === false) return tr("page.neuro.skills.stream.unavailable");
    const probe = status.pcm || window.__duckSound || {};
    if (Number(probe.frames) > 0) {
      return tr(probe.unlocked ? "page.neuro.skills.stream.playing" : "page.neuro.skills.stream.received");
    }
    const stream = status.stream || {};
    if (Number(stream.sent) > 0) return tr("page.neuro.skills.stream.waiting");
    return tr("page.neuro.skills.stream.idle");
  }

  function renderSkillPortfolio(t) {
    const box = $("#neuroSkillCards");
    if (!box) return;
    if (!nu.skillCards) buildSkillCards();
    const i18n = window.I18N;
    if (i18n) {
      i18n.text($("#neuroSkillsTitle"), "page.neuro.skills.title");
      i18n.text($("#neuroSkillsNote"), "page.neuro.skills.intro");
      i18n.attr($("#neuroSkills"), "aria-label", "page.neuro.skills.title");
    }
    const current = t.behaviour || {};
    const reports = t.skills && typeof t.skills === "object" ? t.skills : {};
    for (const skill of SKILLS) {
      const item = nu.skillCards.get(skill.id);
      const report = reports[skill.id] && typeof reports[skill.id] === "object" ? reports[skill.id] : null;
      const selected = skill.behaviours.includes(String(current.key || ""));
      let state = report && SKILL_STATES.has(report.status) ? report.status : "unknown";
      if (report && report.available === false && ["active", "waiting", "unknown"].includes(state)) state = "missing";
      else if (state === "active" && (!selected || !report || report.available !== true)) state = report && report.available === true ? "waiting" : "unknown";
      else if (report && report.available === true && state === "unknown") state = "waiting";
      item.card.dataset.state = state;
      item.name.textContent = tr(skillTextKey(skill.id, "name"));
      item.category.textContent = tr(`page.neuro.skills.category.${skill.category}`);
      item.state.textContent = tr(`page.neuro.skills.status.${state}`);
      item.intent.hidden = !selected;
      if (selected) item.intent.textContent = tr("page.neuro.skills.selected");
      item.trigger.textContent = tr(skillTextKey(skill.id, "trigger"));
      const phase = report ? phaseKey(report.phase) : null;
      item.phase.hidden = !phase;
      if (phase) item.phase.textContent = `${tr("page.neuro.skills.label.phase")} · ${tr(phase)}`;
      const result = report && (state === "done" || ["fear", "joy"].includes(report.result)) && typeof report.result === "string" ? report.result : "";
      const resultKey = result && SKILL_RESULTS.has(result) ? `page.neuro.skills.result.${result}` : null;
      item.result.hidden = !resultKey;
      if (resultKey) item.result.textContent = `${tr("page.neuro.skills.label.result")} · ${tr(resultKey)}`;
      const reasonCode = report && (state === "missing" || state === "refused") ? report.reason : "";
      const reasonKey = typeof reasonCode === "string" && SKILL_REASONS.has(reasonCode) ? `page.neuro.skills.reason.${reasonCode}` : null;
      item.reason.hidden = !reasonKey;
      if (reasonKey) item.reason.textContent = tr(reasonKey);
      else if (state === "missing" || state === "refused") {
        item.reason.hidden = false;
        item.reason.textContent = tr("page.neuro.skills.reason.unknown");
      }
      if (item.stream) {
        item.stream.hidden = false;
        item.stream.textContent = tr("page.neuro.skills.stream.line", { label: tr("page.neuro.skills.label.stream"), status: streamWords() });
      }
    }
  }

  window.addEventListener("duck:sound-status", (event) => {
    nu.soundStatus = event.detail || null;
    if (nu.last) renderSkillPortfolio(nu.last);
  });
  if (window.I18N && typeof window.I18N.onChange === "function") {
    window.I18N.onChange(() => {
      if (nu.last) renderSkillPortfolio(nu.last);
    });
  }

  /* ---------------- reflexes, as they happen ---------------- */

  function clearReflexes() {
    nu.lastReflex = null;
    nu.rowOf.clear();
    $("#neuroReflexes").replaceChildren(words(make("li", { className: "empty" }), "page.neuro.clearReflexes"));
  }

  function addReflex(item, at = null, minute = null) {
    const list = $("#neuroReflexes");
    const empty = list.querySelector(".empty");
    if (empty) empty.remove();
    const now = at != null ? at : Date.now();
    const last = nu.lastReflex;
    const strength0 = Number(item.strength);
    // The same reflex again within a moment without a stimulus of its own (steadying the head
    // on every step…), or the same faint glance again (a habituated look at the things on the
    // floor as it walks by, under 20 %): counted, not repeated line after line. A clear reaction
    // to a stimulus always gets its own line with its size, so habituation shows (the tenth
    // clap, a smaller start).
    const faint = !!item.stimulus && Number.isFinite(strength0) && strength0 < 0.2;
    const span = faint ? 4 * REPEAT_MS : REPEAT_MS;
    if ((!item.stimulus || faint) && last && last.text === F(item, "zh") && now - last.at < span && last.row.isConnected) {
      last.count += 1;
      nu.rowOf.set(item.seq, last.row);
      last.at = now;
      last.times.textContent = faint ? tr("page.neuro.addReflex.73d280", { count: last.count, strength0: sizeZh(strength0) }) : `×${last.count}`;
      last.row.classList.remove("new");
      void last.row.offsetWidth;
      last.row.classList.add("new");
      return;
    }
    const times = make("span", { className: "times" });
    const strength = Number(item.strength);
    // (how big the response was: the tenth clap gets a smaller one — habituation)
    const size = Math.round(Math.max(0, strength) * 100);
    if (item.kind !== "world" && Number.isFinite(strength) && item.stimulus) times.textContent = sizeZh(strength);
    const stamp = minute != null ? duckClock(minute) : clock(at);
    const row = make("li", { className: item.kind === "world" ? "new world" : "new" }, [
      make("span", { className: "t", textContent: stamp, title: minute != null ? tr("page.neuro.addReflex.1bfad7") : "" }),
      make("span", {}, [make("span", { className: `chip rx-${item.kind || "body"}` }), F(item, "zh"), times]),
    ]);
    if (Number.isFinite(strength) && item.kind !== "world") {
      row.title = tr("page.neuro.addReflex.9efba3", {
        strength: sizeZh(strength),
        size,
        gain: item.gain != null ? tr("page.neuro.addReflex.1d6773", { gain: pct(item.gain) }) : "",
        stimulus_zh: item.stimulus_zh ? tr("page.neuro.addReflex.4abf58", { stimulus_zh: F(item, "stimulus_zh") }) : "",
      });
    }
    list.prepend(row);
    while (list.children.length > REFLEX_ROWS) list.lastChild.remove();
    nu.lastReflex = { text: F(item, "zh"), at: now, count: 1, row, times };
    row.dataset.seq = String(item.seq);
    nu.rowOf.set(item.seq, row);
    if (nu.rowOf.size > 4 * REFLEX_ROWS) {
      for (const [seq, li] of nu.rowOf) if (!li.isConnected) nu.rowOf.delete(seq);
    }
  }

  // fid:page: what the body measurably did after a reflex (the driver's witness), as a line
  // under the reflex's row: 「身體：頭往後縮 18°（0.3 秒內）、停下來」, or 「身體做不到：…」
  // (magenta) / 「身體只做到一部分：…」 (orange). A folded repeat shows its latest.
  const VERDICT_KEY = { done: "page.neuro.neuroMode.044fd8", partial: "page.neuro.neuroMode.fa66f8", not_done: "page.neuro.neuroMode.47220a" };
  const HEAD_DOF = [["pitch", "page.neuro.applyBody.746d9c"], ["yaw", "page.neuro.applyBody.2e9d27"], ["roll", "page.neuro.applyBody.8cee30"]];
  function applyBody(item) {
    const row = nu.rowOf.get(item.of);
    if (!row || !row.isConnected || !item.zh) return false;
    const cell = row.lastElementChild;
    let line = cell.querySelector(".rx-body");
    if (!line) {
      line = make("span", { className: "rx-body" });
      cell.append(line);
    }
    line.dataset.verdict = item.verdict || "none";
    line.textContent = F(item, "zh");
    const bits = [];
    const head = item.head || {};
    for (const [dof, key] of HEAD_DOF) {
      const h = head[dof];
      if (!h) continue;
      const cmd = Number(h.cmd_deg);
      bits.push(`${tr(key)} ${Number(h.deg || 0).toFixed(0)}°${Number.isFinite(cmd) && Math.abs(cmd) >= 1 ? tr("page.neuro.applyBody.a72b1a", { cmd: Math.round(cmd) }) : ""}`);
    }
    const legs = item.legs || {};
    if (legs.before_mps != null) {
      bits.push(tr("page.neuro.applyBody.e46e3d", { before_mps: Number(legs.before_mps), after_mps: legs.after_mps != null ? Number(legs.after_mps).toFixed(2) : "?" }));
    }
    line.title = tr("page.neuro.applyBody.440d9b", {
      bits: bits.join(dot()),
      verdict: item.verdict ? tr("page.neuro.applyBody.88ba58", { verdict: VERDICT_KEY[item.verdict] ? tr(VERDICT_KEY[item.verdict]) : "" }) : "",
    });
    row.classList.toggle("rx-failed", item.verdict === "not_done");
    return true;
  }

  function renderBody(now) {
    const box = $("#neuroBody");
    const hud = $("#hudBody");
    if (!box) return;
    box.hidden = !now || !now.zh;
    if (hud) hud.hidden = box.hidden;
    if (box.hidden) return;
    const text = tr("page.neuro.renderBody.05baea", {
      now_zh: F(now, "zh"),
      mismatch_zh: now.mismatch_zh ? tr("page.neuro.renderBody.dd9e09", { mismatch_zh: F(now, "mismatch_zh") }) : "",
    });
    if (box.textContent !== text) box.textContent = text;
    box.dataset.state = now.mismatch_zh ? "mismatch" : "ok";
    // (the view's HUD says the same line)
    if (hud) {
      if (hud.textContent !== text) hud.textContent = text;
      hud.dataset.state = box.dataset.state;
    }
  }

  // The HUD on the view: what it does now (the same words as the brain's), and its newest
  // reflex for a while; the body line is renderBody's.
  const HUD_REFLEX_MS = 20000;
  function renderHud(t, newest) {
    const now = $("#hudNow");
    if (!now) return;
    const b = t.behaviour || null;
    const text = b ? F(b, "zh") : t.waiting_zh ? tr("page.neuro.renderNow.d79743") : "—";
    if (now.textContent !== text) now.textContent = text;
    if (newest) nu.hudReflex = { text: F(newest, "zh"), at: Date.now() };
    const line = $("#hudReflex");
    const fresh = nu.hudReflex && Date.now() - nu.hudReflex.at < HUD_REFLEX_MS;
    line.hidden = !fresh;
    if (fresh && line.textContent !== nu.hudReflex.text) line.textContent = nu.hudReflex.text;
  }

  /* ---------------- bars ---------------- */

  // One meter row (the page's .meter: label, bar, value), made once and then only updated.
  function meter(label, value, valueText, extra = "") {
    const bar = make("i", { style: `width:${pct(value)}%` });
    return make("div", { className: `meter ${extra}`.trim() }, [
      make("span", { textContent: label }),
      make("div", { className: "bar" }, [bar]),
      make("span", { className: "val", textContent: valueText }),
    ]);
  }

  function setMeter(row, label, value, valueText, extra) {
    const [name, bar, val] = row.children;
    if (name.textContent !== label) name.textContent = label;
    const width = `${pct(value)}%`;
    if (bar.firstChild.style.width !== width) bar.firstChild.style.width = width;
    if (typeof valueText === "string") {
      if (val.textContent !== valueText) val.textContent = valueText;
    } else if (val.firstChild !== valueText) {
      val.replaceChildren(...[].concat(valueText));
    }
    const cls = `meter ${extra || ""}`.trim();
    if (row.className !== cls) row.className = cls;
  }

  // Rows keyed by `key`: rebuilt only when the set of keys changes.
  function rows(box, name, items, build, update) {
    const key = items.map((item) => item.key).join("|");
    if (nu.keys[name] !== key || box.children.length !== items.length) {
      nu.keys[name] = key;
      box.replaceChildren(...items.map(build));
    }
    items.forEach((item, index) => update(box.children[index], item));
  }

  function renderCands(items) {
    const box = $("#neuroCands");
    const list = items.slice(0, 6);
    rows(
      box,
      "cands",
      list,
      (item) => meter(F(item, "zh"), item.pull, `${pct(item.pull)}`),
      (row, item) => {
        setMeter(row, F(item, "zh"), item.pull, `${pct(item.pull)}`, item.current ? "on" : "");
        row.title = F(item, "why_zh");
      },
    );
  }

  function renderDrives(items, arousal, valence) {
    const box = $("#neuroDrives");
    const named = (item) => ({ ...item, zh: F(item, "zh") });
    const main = items.filter((item) => item.main !== false).map(named);
    const minor = items.filter((item) => item.main === false).map(named);
    const list = [...main];
    if (arousal) list.push({ key: "arousal", zh: tr("page.neuro.renderDrives.a14d80"), value: arousal.value, cls: "wide", title: F(arousal, "zh") });
    for (const item of minor) list.push({ ...item, cls: "minor" });
    if (typeof valence === "number") list.push({ key: "valence", zh: tr("page.neuro.renderDrives.300ee5"), value: (valence + 1) / 2, cls: "minor", text: `${valence > 0 ? "+" : ""}${Math.round(valence * 100)}` });
    rows(
      box,
      "drives",
      list,
      (item) => meter(item.zh, item.value, item.text || `${pct(item.value)}`, item.cls || ""),
      (row, item) => {
        setMeter(row, item.zh, item.value, item.text || `${pct(item.value)}`, item.cls || "");
        if (item.title) row.title = item.title;
      },
    );
  }

  function renderHabits(items) {
    const box = $("#neuroHabits");
    rows(
      box,
      "habits",
      items,
      (item) => meter(F(item, "zh"), item.level, ""),
      (row, item) => {
        const zh = F(item, "zh");
        const text = `${pct(item.level)}%${item.count ? tr("page.neuro.renderHabits.9f2621", { count: item.count }) : ""}`;
        const parts = item.sensitised ? [text, make("span", { className: "tag memory", textContent: tr("page.neuro.renderHabits.a88158") })] : text;
        setMeter(row, zh, item.level, parts, item.sensitised ? "scared" : "");
        row.title = item.sensitised
          ? tr("page.neuro.renderHabits.ddfa5f", { zh, level: pct(item.level) })
          : tr("page.neuro.renderHabits.c2943f", { zh, level: pct(item.level) });
      },
    );
  }

  function renderPlaces(items) {
    const box = $("#neuroPlaces");
    if (!items.length) {
      if (!box.querySelector(".empty")) box.replaceChildren(words(make("li", { className: "empty" }), "page.neuro.renderPlaces.18018c"));
      nu.keys.places = "";
      return;
    }
    const list = items.map((item) => ({ ...item, key: item.key }));
    rows(
      box,
      "places",
      list,
      (item) => make("li", {}, [meter(F(item, "zh"), (item.value + 1) / 2, "")]),
      (li, item) => {
        const verdict = item.avoid
          ? tr("page.neuro.renderPlaces.82a03d", { falls: item.falls ? tr("page.neuro.renderPlaces.8ef7ea", { falls: item.falls }) : "" })
          : tr(item.value > 0.05 ? "page.neuro.renderPlaces.07990d" : item.value < -0.05 ? "page.neuro.renderPlaces.a9d699" : "page.neuro.renderPlaces.7cda07");
        setMeter(li.firstChild, F(item, "zh"), (item.value + 1) / 2, verdict, item.avoid ? "avoid" : "");
        li.title = F(item, "note_zh");
      },
    );
  }

  /* ---------------- its walking rhythm: bursts and pauses, never one speed ---------------- */

  const SVG = "http://www.w3.org/2000/svg";
  function renderRhythm(r) {
    const svg = $("#neuroRhythm");
    const bins = (r && r.bins) || [];
    const turns = (r && r.turns) || [];
    const key = bins.join(",") + "|" + turns.join(",");
    if (nu.keys.rhythm === key) return;
    nu.keys.rhythm = key;
    const n = bins.length || 60;
    const step = 600 / n;
    const kids = [];
    const base = document.createElementNS(SVG, "line");
    Object.entries({ x1: 0, x2: 600, y1: 50.5, y2: 50.5, class: "base" }).forEach(([k, v]) => base.setAttribute(k, v));
    kids.push(base);
    let bursts = 0;
    let fastest = 0;
    let turning = 0;
    bins.forEach((vx, i) => {
      const age = n - 1 - i;
      const raw = Math.max(0, Number(vx) || 0);
      const speed = raw > 0.05 ? raw : 0; // (under 5 cm/s is standing sway, not a step)
      const turn = Number(turns[i]) || 0;
      if (speed > 0 && !(i > 0 && Number(bins[i - 1]) > 0)) bursts += 1;
      fastest = Math.max(fastest, speed);
      const g = document.createElementNS(SVG, "g");
      const title = document.createElementNS(SVG, "title");
      title.textContent =
        speed > 0
          ? tr("page.neuro.renderRhythm.b6db28", { age, speed, turn: Math.abs(turn) > 0.3 ? tr(turn > 0 ? "page.neuro.renderRhythm.0b0557" : "page.neuro.renderRhythm.18af6a") : "" })
          : tr("page.neuro.renderRhythm.6e376a", { age });
      g.append(title);
      if (speed > 0) {
        // (fid:fix: the body's real speed — ~0.1–0.3 m/s on this body — fills the strip)
        const h = Math.max(3, Math.min(48, (speed / 0.35) * 48));
        const bar = document.createElementNS(SVG, "rect");
        Object.entries({ x: i * step + 1, y: 50 - h, width: step - 2, height: h }).forEach(([k, v]) => bar.setAttribute(k, v));
        g.append(bar);
        if (Math.abs(turn) > 0.3) {
          turning += 1;
          const mark = document.createElementNS(SVG, "rect");
          Object.entries({ x: i * step + step / 2 - 2, y: 55, width: 4, height: 6, class: "turn-mark" }).forEach(([k, v]) => mark.setAttribute(k, v));
          g.append(mark);
        }
      }
      const hit = document.createElementNS(SVG, "rect");
      Object.entries({ x: i * step, y: 0, width: step, height: 64, class: "hit" }).forEach(([k, v]) => hit.setAttribute(k, v));
      g.append(hit);
      kids.push(g);
    });
    svg.replaceChildren(...kids);
    const share = r && typeof r.share === "number" ? Math.round(r.share * 100) : 0;
    $("#neuroRhythmNote").textContent = bins.length
      ? share > 0
        ? tr("page.neuro.renderRhythm.9de6d1", { share, bursts, fastest, turning })
        : tr("page.neuro.renderRhythm.3efb2b")
      : tr("page.neuro.renderRhythm.68870a");
  }

  /* ---------------- senses and lanes (the mind section) ---------------- */

  function renderSenses(items, driver) {
    const key = JSON.stringify(items.map((s) => [s.key, s.source])) + JSON.stringify(driver ? [driver.lanes_zh, driver.notes] : null);
    if (nu.keys.senses === key) return;
    nu.keys.senses = key;
    const src = {
      real: tr("page.neuro.renderSenses.8f81bb"),
      eye: tr("page.neuro.renderSenses.206c27"),
      model: tr("page.neuro.renderSenses.c9cc56"),
      none: tr("page.neuro.renderSenses.a9b748"),
    };
    $("#neuroSenses").replaceChildren(
      ...items.map((s) =>
        make("li", {}, [
          make("span", {}, [make("span", { className: `chip src-${s.source || "none"}`, textContent: F(s, "zh") })]),
          make("small", { textContent: F(s, "note_zh") || src[s.source] || s.source || "" }),
        ]),
      ),
    );
    const lanes = driver && driver.lanes_zh ? tr("page.neuro.renderSenses.906f2c", { lanes_zh: F(driver, "lanes_zh") }) : "";
    const notes = driver && driver.notes && driver.notes.length ? FS(driver, "notes").join(tr("fmt.sep.semicolon")) : "";
    $("#neuroLanes").textContent = `${lanes}${notes}` || tr("page.neuro.renderSenses.dc8fcb");
  }

  /* ---------------- 逗逗牠: the stimuli ---------------- */

  function buildStimuli(status) {
    const more = status.more || [];
    const box = $("#neuroMoreButtons");
    box.replaceChildren(
      ...more.map((item) => make("button", { type: "button", className: "btn small", textContent: F(item, "zh"), title: tr("page.neuro.buildStimuli.2181dd", { channels: FS(item, "channels").join(tr("fmt.sep.enum")) }) }, [])),
    );
    more.forEach((item, index) => (box.children[index].dataset.stim = item.key));
    $("#neuroMore").hidden = !more.length;
    for (const item of status.stimuli || []) {
      const button = $(`#neuroStim [data-stim="${item.key}"]`);
      if (button) button.title = tr("page.neuro.buildStimuli.7698a1", { world_zh: F(item, "world_zh"), sense_zh: F(item, "sense_zh") });
    }
  }

  /* ---------------- 指令列: tell the duck what to do (POST /api/neuro/command) ----------------
   * What is sent is what the duck HEARS (words typed, or a button): it decides, from its own state, whether and when to
   * do it, and says why not. The line under the buttons is the last command and what became of it; it comes from the
   * brain's telemetry (`t.command.last`), so a command from another page or the API shows here too. Locked tricks are
   * sent as well (the duck hears them and shows it does not know them yet). */

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
  const cmdUi = { sending: false, line: "", shown: "" };

  // the page's own 繁中 phrase is what a button means: `lang` is the viewer's language (the words a person types are read in it)
  const cmdLang = () => (window.I18N && window.I18N.lang && window.I18N.lang !== "pseudo" ? window.I18N.lang : "zh-Hant");

  function cmdLine(last) {
    // "<verb>: <state>[, <why>][ (<ms> ms)]" in the viewer's language, from the telemetry's last command
    const verb = VERB_KEYS[last.verb] ? tr(VERB_KEYS[last.verb]) : String(last.verb || "");
    const state = F(last, "state_zh");
    const why = last.reason ? F(last, "reason_zh") : "";
    let line = why ? tr("page.cmd.line.why", { verb, state, why }) : tr("page.cmd.line.base", { verb, state });
    const ms = last.state === "done" || last.state === "doing" ? last.motion_ms : null;
    if (Number.isFinite(ms) && ms >= 0) line = tr("page.cmd.line.latency", { line, ms });
    return line;
  }

  function renderCommand(t) {
    const out = $("#cmdOut");
    const strip = $("#cmdStrip");
    if (!out || !strip) return;
    if (strip.hidden) strip.hidden = false;
    const last = t.command && t.command.last;
    if (!last) return;
    // the same words again are not written again (the stream says them four times a second)
    const key = `${last.seq}|${last.state}|${last.reason || ""}|${last.motion_ms ?? ""}|${window.I18N ? window.I18N.lang : ""}`;
    if (key === cmdUi.shown) return;
    cmdUi.shown = key;
    out.dataset.state = last.state || "none";
    delete out.dataset.i18n;
    out.textContent = cmdLine(last);
  }

  function cmdAnswerLine(data, verbKey) {
    // the answer to one press: what the duck said at once (the telemetry then keeps the line up to date)
    const verb = verbKey ? tr(verbKey) : "";
    const state = F(data, "state_zh");
    const why = data.reason ? F(data, "reason_zh") : "";
    return why ? tr("page.cmd.line.why", { verb, state, why }) : tr("page.cmd.line.base", { verb, state });
  }

  async function sendCommand(params, verbKey) {
    if (cmdUi.sending || !nu.on) return;
    cmdUi.sending = true;
    const out = $("#cmdOut");
    const keys = $$("#cmdStrip button:not(#cmdMic):not([data-remote]):not([data-feedback])"); // (the microphone is a placeholder: always off; the remote hand's buttons are remote.js's, never locked while a button is held; the feedback buttons are tricks.js's)
    for (const key of keys) key.disabled = true;
    try {
      const response = await fetch("/api/neuro/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lang: cmdLang(), ...params }),
      });
      const data = await response.json();
      if (!response.ok || data.ok === undefined) throw Object.assign(new Error(F(data, "error") || tr("page.neuro.sendStim.a20b11")), { data });
      const key = verbKey || (VERB_KEYS[data.verb] ? VERB_KEYS[data.verb] : "");
      out.dataset.state = data.refused ? "refused" : data.state === "done" ? "done" : "heard";
      delete out.dataset.i18n;
      out.textContent = cmdAnswerLine(data, key);
      cmdUi.shown = "";
      const text = $("#cmdText");
      if (text && params.text !== undefined && !data.refused) text.value = "";
    } catch (error) {
      out.dataset.state = "error";
      delete out.dataset.i18n;
      out.textContent = tr("page.neuro.sendStim.076a5d", { label: tr("page.cmd.label"), error: why(error, tr("page.neuro.sendStim.a20b11")) });
    } finally {
      cmdUi.sending = false;
      for (const key of keys) key.disabled = false;
    }
  }

  function setupCommandStrip() {
    const strip = $("#cmdStrip");
    if (!strip) return;
    strip.addEventListener("submit", (event) => {
      event.preventDefault();
      const text = $("#cmdText").value.trim();
      if (text) sendCommand({ text, source: "text" }, "");
    });
    strip.addEventListener("click", (event) => {
      const button = event.target.closest("[data-cmd]");
      if (!button || button.disabled) return;
      button.classList.add("sent");
      setTimeout(() => button.classList.remove("sent"), 700);
      sendCommand({ verb: button.dataset.cmd, source: "button" }, VERB_KEYS[button.dataset.cmd]);
    });
    // a phone starts with the buttons folded (the text box and the last line stay); a desktop shows them
    const keys = $("#cmdKeys");
    if (keys && window.matchMedia && window.matchMedia("(max-width: 760px)").matches) keys.open = false;
  }
  setupCommandStrip();
  // the trick book's buttons (tricks.js, the 訓練 tab) send a trick's command through this same path
  window.duckCommand = (verb) => (VERB_KEYS[verb] ? sendCommand({ verb, source: "button" }, VERB_KEYS[verb]) : undefined);

  function renderButtons(t) {
    if (t.world && typeof t.world.lights_on === "boolean") nu.lights = t.world.lights_on;
    const label = tr(nu.lights ? "page.neuro.renderButtons.24ac92" : "page.neuro.renderButtons.f5c48e");
    for (const lights of $$("[data-lights]")) if (lights.textContent !== label) lights.textContent = label;
    const can = t.stimuli || null;
    const quick = $("#neuroQuick");
    if (quick && quick.hidden) quick.hidden = false;
    // (fid:fix) the stage's strip too: the same stimuli right under the 3D view
    for (const button of $$("#neuroStim [data-stim], #neuroQuick [data-stim]")) {
      const ok = !can || can[button.dataset.stim] !== false;
      const off = !ok || nu.sending;
      if (button.disabled !== off) button.disabled = off;
    }
    const none = !!can && Object.values(can).every((v) => v === false);
    if (none !== nu.noButtons) {
      nu.noButtons = none;
      $("#neuroStimNote").textContent = tr(none ? "page.neuro.renderButtons.13153b" : "page.neuro.renderButtons.589a4c");
    }
  }

  async function sendStim(event) {
    const button = event.target.closest("[data-stim]");
    if (!button || button.disabled || !nu.on) return;
    const kind = button.dataset.stim;
    const label = button.textContent.trim();
    say(`${label}…`);
    button.classList.add("sent");
    setTimeout(() => button.classList.remove("sent"), 700);
    try {
      const response = await fetch("/api/neuro/stimulus", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind }),
      });
      const data = await response.json();
      if (!data.ok) throw new Error(F(data, "error") || tr("page.neuro.sendStim.a20b11"));
      if (typeof data.lights_on === "boolean") nu.lights = data.lights_on;
      const noteZh = F(data, "note_zh");
      const said = noteZh ? (noteZh.startsWith(label) ? noteZh : tr("page.neuro.sendStim.339b1a", { label, note_zh: noteZh })) : label;
      $("#neuroStimNote").textContent = `${clock()} ${said}`;
      say(said, true);
    } catch (error) {
      note(tr("page.neuro.sendStim.076a5d", { label, error: why(error, tr("page.neuro.sendStim.a20b11")) }));
    }
  }
  $("#neuroStim").addEventListener("click", sendStim);
  const quickStrip = $("#neuroQuick");
  if (quickStrip) quickStrip.addEventListener("click", sendStim);

  /* ---------------- 動作試玩: every trained motion (POST /api/neuro/motion) ---------------- */

  // id → its words (catalog keys, spelled out: the page's key checks read these literals)
  const MOTION_WORDS = {
    startle_1: { name: "page.motion.name.startle_1", limit: "page.motion.limit.startle" },
    startle_2: { name: "page.motion.name.startle_hop_2", limit: "page.motion.limit.startle_hop" },
    startle_3: { name: "page.motion.name.startle_hop_3", limit: "page.motion.limit.startle_hop" },
    startle_4: { name: "page.motion.name.startle_hop_4", limit: "page.motion.limit.startle_hop" },
    freeze_2s: { name: "page.motion.name.freeze_2s", limit: "page.motion.limit.freeze2" },
    freeze_3s: { name: "page.motion.name.freeze_3s", limit: "page.motion.limit.freeze3" },
    hop_back: { name: "page.motion.name.hop_back", limit: "page.motion.limit.hop_back" },
    shake_1: { name: "page.motion.name.shake_1", limit: "page.motion.limit.shake" },
    shake_2: { name: "page.motion.name.shake_2", limit: "page.motion.limit.shake" },
    stretch_tall: { name: "page.motion.name.stretch_tall", limit: "page.motion.limit.stretch_tall" },
    stretch_sit: { name: "page.motion.name.stretch_sit", limit: "page.motion.limit.stretch_sit" },
    turn_left: { name: "page.motion.name.turn_left", limit: "page.motion.limit.turn" },
    turn_right: { name: "page.motion.name.turn_right", limit: "page.motion.limit.turn" },
    quick_turn_left: { name: "page.motion.name.quick_turn_left", limit: "page.motion.limit.quick_turn_gate" },
    quick_turn_right: { name: "page.motion.name.quick_turn_right", limit: "page.motion.limit.quick_turn_gate" },
    hop_over_2: { name: "page.motion.name.hop_over_2", limit: "page.motion.limit.hop_over" },
    hop_over_4: { name: "page.motion.name.hop_over_4", limit: "page.motion.limit.hop_over" },
    bow: { name: "page.motion.name.bow", limit: "page.motion.limit.bow" },
    lie_low: { name: "page.motion.name.lie_low", limit: "page.motion.limit.lie" },
    // 「情緒」: Pollen Robotics' microduck_emotions, the sounds in this duck's own voice (neuro/emotions.py)
    emo_sad: { name: "page.emotion.name.sad", limit: "page.emotion.desc.sad" },
    emo_devastated: { name: "page.emotion.name.devastated", limit: "page.emotion.desc.devastated" },
    emo_curious: { name: "page.emotion.name.curious", limit: "page.emotion.desc.curious" },
    emo_mmh: { name: "page.emotion.name.mmh", limit: "page.emotion.desc.mmh" },
    emo_yes: { name: "page.emotion.name.yes", limit: "page.emotion.desc.yes" },
    emo_yes_fast: { name: "page.emotion.name.yes_fast", limit: "page.emotion.desc.yes_fast" },
    emo_no: { name: "page.emotion.name.no", limit: "page.emotion.desc.no" },
    emo_defiant: { name: "page.emotion.name.defiant", limit: "page.emotion.desc.defiant" },
    emo_angry: { name: "page.emotion.name.angry", limit: "page.emotion.desc.angry" },
    emo_mock: { name: "page.emotion.name.mock", limit: "page.emotion.desc.mock" },
    emo_excited: { name: "page.emotion.name.excited", limit: "page.emotion.desc.excited" },
    emo_impatient: { name: "page.emotion.name.impatient", limit: "page.emotion.desc.impatient" },
    emo_laugh: { name: "page.emotion.name.laugh", limit: "page.emotion.desc.laugh" },
    emo_play_dead: { name: "page.emotion.name.play_dead", limit: "page.emotion.desc.play_dead" },
  };
  const MOTION_GROUPS = {
    fear: "page.motion.group.fear",
    comfort: "page.motion.group.comfort",
    move: "page.motion.group.move",
    play: "page.motion.group.play",
    emotion: "page.motion.group.emotion",
  };
  const MOTION_GROUP_ORDER = ["fear", "comfort", "move", "play", "emotion"];
  const MOTION_VERDICT = {
    pass: "page.motion.verdict.pass",
    limits: "page.motion.verdict.limits",
    fail: "page.motion.verdict.fail",
  };
  const MOTION_POSTURE = { stand: "page.motion.posture.stand", walk: "page.motion.posture.walk", sit: "page.motion.posture.sit" };
  // how the brain uses it (by the server's `brain` flag only says whether; the words are per id)
  const MOTION_BRAIN = {
    startle_1: "page.motion.brain.startle",
    startle_2: "page.motion.brain.startle_hop",
    startle_3: "page.motion.brain.startle_hop",
    startle_4: "page.motion.brain.startle_hop",
    freeze_2s: "page.motion.brain.freeze",
    freeze_3s: "page.motion.brain.freeze",
    hop_back: "page.motion.brain.hop_back_scare",
    shake_1: "page.motion.brain.shake",
    shake_2: "page.motion.brain.shake",
    stretch_tall: "page.motion.brain.stretch",
    stretch_sit: "page.motion.brain.stretch",
    turn_left: "page.motion.brain.corner",
    turn_right: "page.motion.brain.corner",
    quick_turn_left: "page.motion.brain.pivot",
    quick_turn_right: "page.motion.brain.pivot",
    bow: "page.motion.brain.greet",
    lie_low: "page.motion.brain.peekaboo",
    emo_curious: "page.emotion.brain.novelty",
    emo_mmh: "page.emotion.brain.speech",
    emo_excited: "page.emotion.brain.arrival",
    emo_laugh: "page.emotion.brain.arrival",
    emo_mock: "page.emotion.brain.tease",
    emo_impatient: "page.emotion.brain.blocked",
    emo_defiant: "page.emotion.brain.blocked",
    emo_angry: "page.emotion.brain.blocked",
    emo_sad: "page.emotion.brain.lonely",
    emo_devastated: "page.emotion.brain.lonely",
    emo_play_dead: "page.emotion.brain.scare",
  };
  const NO_BRAIN = "page.motion.brain.none";

  // the words of what a motion cannot do: play dead on a robotd without the lying-down pose only sits (status `lies` false)
  function limitKey(m, word) {
    return m && m.id === "emo_play_dead" && m.lies === false ? "page.emotion.desc.play_dead_sit" : word.limit;
  }

  function buildMotions(status) {
    const list = [...(status.motions || []), ...(status.emotions || [])];
    const box = $("#motionGroups");
    if (!box || !list.length) return;
    nu.motions = new Map(list.map((m) => [m.id, m]));
    const groups = [];
    for (const m of list) {
      let g = groups.find((x) => x.id === m.group);
      if (!g) groups.push((g = { id: m.group, items: [] }));
      g.items.push(m);
    }
    groups.sort((a, b) => MOTION_GROUP_ORDER.indexOf(a.id) - MOTION_GROUP_ORDER.indexOf(b.id));
    // (the emotions have their own tab: 情緒; the other motions stay in 動作)
    const emotionBox = $("#emotionGroups");
    const groupNode = (g) =>
      make("div", { className: "motion-group" }, [
        words(make("p", { className: "eyebrow" }), MOTION_GROUPS[g.id] || MOTION_GROUPS.fear),
        make(
          "div",
          { className: "motion-row" },
          g.items.map((m) => {
            const word = MOTION_WORDS[m.id];
            if (!word) return make("span");
            const button = make("button", { type: "button", className: "btn small motion-btn" });
            button.dataset.motion = m.id;
            button.dataset.verdict = m.verdict;
            button.append(words(make("span", { className: "mn" }), word.name), words(make("span", { className: "mv" }), MOTION_VERDICT[m.verdict] || MOTION_VERDICT.pass));
            if (window.I18N) window.I18N.attr(button, "title", limitKey(m, word));
            return button;
          }),
        ),
      ]);
    const toEmotions = (g) => !!emotionBox && g.id === "emotion";
    box.replaceChildren(...groups.filter((g) => !toEmotions(g)).map(groupNode));
    if (emotionBox) emotionBox.replaceChildren(...groups.filter(toEmotions).map(groupNode));
    // every motion's verdict, what it may be done from, how the brain uses it
    $("#motionList").replaceChildren(
      ...list.map((m) => {
        const word = MOTION_WORDS[m.id];
        if (!word) return make("li");
        const where = m.postures.map((p) => tr(MOTION_POSTURE[p] || MOTION_POSTURE.stand));
        const li = make("li", { className: "motion-item" });
        li.dataset.verdict = m.verdict;
        li.append(
          make("strong", {}, [words(make("span"), word.name), " ", words(make("span", { className: "mv" }), MOTION_VERDICT[m.verdict] || MOTION_VERDICT.pass)]),
          words(make("span", { className: "motion-limit" }), limitKey(m, word)),
          words(make("span", { className: "motion-where" }), "page.motion.d.where", { postures: where.join(dot()) }),
          words(make("span", { className: "motion-brain" }), m.brain && MOTION_BRAIN[m.id] ? MOTION_BRAIN[m.id] : NO_BRAIN),
        );
        return li;
      }),
    );
    $("#motionDetails").hidden = false;
  }

  function motionName(id) {
    const word = MOTION_WORDS[id];
    return word ? tr(word.name) : id;
  }

  function motionNote(text) {
    for (const el of [$("#motionNote"), $("#emotionNote")]) {
      if (!el) continue;
      el.hidden = !text;
      el.textContent = text ? `${clock()} ${text}` : "";
    }
  }

  function renderMotions(t) {
    const info = t.motion || null;
    const installed = info && Array.isArray(info.installed) ? new Set(info.installed) : null;
    const emo = info && info.emotion ? info.emotion : null;
    const emoting = !!(emo && emo.running);
    for (const button of $$("#neuroMotions [data-motion], #neuroEmotions [data-motion]")) {
      const m = nu.motions && nu.motions.get(button.dataset.motion);
      const missing = !!(installed && m && m.needs && !m.needs.every((n) => installed.has(n)));
      const off = missing || nu.motionBusy || emoting;
      if (button.disabled !== off) button.disabled = off;
    }
    if (!info) return;
    if (emo) {
      // an emotion that ended (the page's press or the duck's own): said once, in the viewer's language
      if (nu.emotionSeen == null) nu.emotionSeen = emo.last ? emo.last.seq : 0; // (what ran before this page opened is not announced)
      else if (emo.last && emo.last.seq > nu.emotionSeen) {
        nu.emotionSeen = emo.last.seq;
        const name = motionName(emo.last.id);
        const said =
          emo.last.source === "brain"
            ? tr("page.emotion.say.brain", { name, why: F(emo.last, "why_zh"), line: F(emo.last, "zh") })
            : tr("page.emotion.say.done", { name, line: F(emo.last, "zh") });
        motionNote(said);
        say(said, true);
      }
    }
    const last = info.last || null;
    if (!last) return;
    if (nu.motionSeen === null) {
      nu.motionSeen = last.seq; // (what ran before this page opened is not announced)
      return;
    }
    if (last.seq > nu.motionSeen) {
      nu.motionSeen = last.seq;
      const said = tr("page.motion.say.done", { name: motionName(last.id), line: F(last, "zh"), tail: F(last, "tail_zh") });
      motionNote(said);
      say(said, true);
    }
  }

  async function playMotion(event) {
    const button = event.target.closest("[data-motion]");
    if (!button || button.disabled || !nu.on) return;
    const id = button.dataset.motion;
    const name = motionName(id);
    say(tr("page.motion.say.pressing", { name }));
    button.classList.add("sent");
    setTimeout(() => button.classList.remove("sent"), 700);
    nu.motionBusy = true;
    renderMotions(nu.last || {});
    try {
      const response = await fetch("/api/neuro/motion", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id }),
      });
      const data = await response.json();
      if (data.ok === false && !data.refused) throw new Error(F(data, "error") || tr("page.neuro.sendStim.a20b11"));
      if (data.ok) {
        motionNote(tr("page.motion.say.started", { name }));
      } else {
        const refused = tr("page.motion.say.refused", { name, why: F(data, "reason_zh") });
        motionNote(refused);
        say(refused, true);
      }
    } catch (error) {
      const failed = tr("page.motion.say.failed", { name, error: why(error, tr("page.neuro.sendStim.a20b11")) });
      motionNote(failed);
      note(failed);
    } finally {
      nu.motionBusy = false;
      renderMotions(nu.last || {});
    }
  }
  for (const card of [$("#neuroMotions"), $("#neuroEmotions")]) if (card) card.addEventListener("click", playMotion);

  /* ---------------- the duck's own clock ---------------- */

  function buildDay(choices) {
    const box = $("#neuroDayButtons");
    if (!box) return;
    box.replaceChildren(
      ...choices.map((item) => {
        const button = make("button", { type: "button", className: "btn small", textContent: F(item, "zh") });
        button.dataset.day = item.zh; // (the server's own word for the choice: what a click sends back)
        button.dataset.minute = String(item.minute);
        button.setAttribute("role", "radio");
        button.setAttribute("aria-checked", "false");
        return button;
      }),
    );
    $("#neuroDay").hidden = !choices.length;
  }

  function renderDay(day) {
    if (!day) return;
    const text = tr("page.neuro.renderDay.69ce4f", { day_zh: F(day, "zh") });
    const now = $("#neuroDayNow");
    if (now && now.textContent !== text) now.textContent = text;
    // the topbar's clock is the neural duck's own in this mode (Duck Core's is put away)
    const top = tr("page.neuro.renderDay.9ee6ac", { day_zh: F(day, "zh") });
    const topNow = $("#clockNow");
    if (topNow && topNow.textContent !== top) topNow.textContent = top;
    const week = $("#clockWeek");
    if (week && week.textContent) week.textContent = "";
    const minute = Number(day.minute) || 0;
    let best = null;
    let gap = 1e9;
    for (const button of $$("#neuroDayButtons [data-day]")) {
      const m = Number(button.dataset.minute);
      const d = Math.min(Math.abs(m - minute), 1440 - Math.abs(m - minute));
      if (d < gap) {
        gap = d;
        best = button;
      }
    }
    for (const button of $$("#neuroDayButtons [data-day]")) {
      const on = button === best && gap < 90;
      button.classList.toggle("on", on);
      button.setAttribute("aria-checked", String(on));
    }
  }

  $("#neuroDayButtons").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-day]");
    if (!button || !nu.on) return;
    const choice = button.dataset.day;
    try {
      const response = await fetch("/api/neuro/clock", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ choice }),
      });
      const data = await response.json();
      if (!data.ok) throw new Error(F(data, "error") || tr("page.neuro.neuroMode.08ad6e"));
      if (data.day) renderDay(data.day);
      const said = F(data, "note_zh") || tr("page.neuro.neuroMode.2d552a", { choice: button.textContent });
      $("#neuroDayNote").textContent = `${clock()} ${said}`;
      say(said, true);
    } catch (error) {
      note(tr("page.neuro.neuroMode.749319", { error: why(error, tr("page.neuro.neuroMode.08ad6e")) }));
    }
  });

  /* ---------------- 扶牠起來: only while it is down and could not get up ---------------- */

  function renderHelp(body) {
    const row = $("#neuroHelpRow");
    if (!row) return;
    const down = !!(body && body.waiting_help);
    const stuck = !!(body && body.stuck) && !down;
    const show = down || stuck;
    const kind = down ? "down" : stuck ? "stuck" : "";
    if (show && row.dataset.kind !== kind) {
      // (a new reason to offer help: say it; after a click the row keeps what the world did)
      $("#neuroHelpNote").textContent = tr(stuck ? "page.neuro.renderHelp.944bf2" : "page.neuro.renderHelp.6103d3");
    }
    row.dataset.kind = kind;
    if (row.hidden === show) row.hidden = !show;
    const help = $("#neuroHelp");
    const carry = $("#neuroCarry");
    if (help.hidden === down) help.hidden = !down;
    if (carry && carry.hidden === stuck) carry.hidden = !stuck;
  }

  $("#neuroHelp").addEventListener("click", async () => {
    if (!nu.on) return;
    const button = $("#neuroHelp");
    button.disabled = true;
    say(tr("page.neuro.neuroMode.b02abe"));
    try {
      const response = await fetch("/api/neuro/help", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      const data = await response.json();
      if (!data.ok) throw new Error(F(data, "error") || tr("page.neuro.neuroMode.da8b2b"));
      const said = F(data, "note_zh") || tr("page.neuro.neuroMode.f23fa8");
      $("#neuroHelpNote").textContent = `${clock()} ${said}`;
      say(said, true);
    } catch (error) {
      note(tr("page.neuro.neuroMode.da0e07", { error: why(error, tr("page.neuro.neuroMode.da8b2b")) }));
    } finally {
      setTimeout(() => (button.disabled = false), 3000);
    }
  });

  $("#neuroCarry").addEventListener("click", async () => {
    if (!nu.on) return;
    const button = $("#neuroCarry");
    button.disabled = true;
    say(tr("page.neuro.neuroMode.7f1fe9"));
    try {
      const response = await fetch("/api/neuro/stimulus", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: "carry_to_mat" }),
      });
      const data = await response.json();
      if (!data.ok) throw new Error(F(data, "error") || tr("page.neuro.neuroMode.e7a22d"));
      const said = F(data, "note_zh") || tr("page.neuro.neuroMode.85099e");
      $("#neuroHelpNote").textContent = `${clock()} ${said}`;
      say(said, true);
    } catch (error) {
      note(tr("page.neuro.neuroMode.d73ac1", { error: why(error, tr("page.neuro.neuroMode.e7a22d")) }));
    } finally {
      setTimeout(() => (button.disabled = false), 3000);
    }
  });

  /* ---------------- 換一隻小鴨 ---------------- */

  function buildPicker(status) {
    const box = $("#neuroDucks");
    const ducks = status.ducks || [];
    const key = ducks.map((d) => `${d.preset}:${d.seed}:${d.gd}`).join("|");
    if (nu.keys.picker === key) return;
    nu.keys.picker = key;
    const pick = (duck) => {
      const button = make("button", { type: "button", className: "duck-pick" }, [
        make("b", { textContent: duck.name }),
        make("span", { textContent: F(duck, "title_zh") || tr("page.neuro.renderDuck.branch_stage", { branch: F(duck, "branch_zh") || F(duck, "nature_zh"), stage: F(duck, "stage_zh") }) }),
        make("small", { textContent: F(duck, "line_zh") }),
      ]);
      button.setAttribute("role", "radio");
      button.setAttribute("aria-checked", "false");
      button.dataset.duck = duck.preset;
      button.dataset.seed = String(duck.seed);
      button.dataset.gd = String(duck.gd);
      return button;
    };
    const random = make("button", { type: "button", className: "duck-pick" }, [
      make("b", { textContent: tr("page.neuro.buildPicker.77f3f5") }),
      make("span", { textContent: tr("page.neuro.buildPicker.58b0df") }),
      make("small", { textContent: tr("page.neuro.buildPicker.689128") }),
    ]);
    random.setAttribute("role", "radio");
    random.setAttribute("aria-checked", "false");
    random.dataset.duck = "random";
    box.replaceChildren(...ducks.map(pick), random);
    nu.duckKey = ""; // (mark the duck now once the buttons are there)
  }

  $("#neuroDucks").addEventListener("click", async (event) => {
    const button = event.target.closest(".duck-pick");
    if (!button || !nu.on || button.disabled) return;
    const which = button.dataset.duck;
    const body = which === "random" ? { random: true } : { preset: which };
    const all = $$("#neuroDucks .duck-pick");
    all.forEach((b) => (b.disabled = true));
    const name = button.querySelector("b").textContent;
    say(tr("page.neuro.neuroMode.ed7f2a", { name }));
    // (voice:neuro) no call in the old duck's voice while the new one is coming
    const voice = window.duckVoice;
    if (voice && voice.switching) voice.switching();
    try {
      const response = await fetch("/api/neuro/duck", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      if (!data.ok) throw new Error(F(data, "error") || tr("page.neuro.neuroMode.d27cde"));
      nu.duckKey = "";
      nu.since = 0;
      nu.keys = { picker: nu.keys.picker, senses: nu.keys.senses };
      clearReflexes();
      if (data.duck) renderDuck(data.duck);
      const said = data.duck
        ? tr("page.neuro.neuroMode.cf6e8b", { duck_name: data.duck.name, line_zh: F(data.duck, "line_zh"), seed: data.duck.seed, gd: data.duck.gd })
        : tr("page.neuro.neuroMode.85d34a");
      note(said);
      say(said, true);
      $("#neuroPickNote").textContent = said;
      connect(); // (its numbers start over)
    } catch (error) {
      note(tr("page.neuro.neuroMode.7a2f03", { error: why(error, tr("page.neuro.neuroMode.d27cde")) }));
      if (voice && voice.duck && nu.voice) voice.duck(nu.voice.seed); // (still the same duck)
    } finally {
      all.forEach((b) => (b.disabled = false));
    }
  });

  // A language switch: what was built once (the buttons, the picker, the title) is built again in
  // the new words; the rows that keep the words they were written in (the reflexes) are listed
  // again from the start of the stream.
  if (window.I18N && typeof window.I18N.onChange === "function") {
    window.I18N.onChange(() => {
      if (!nu.on) return;
      document.title = tr("page.neuro.enable");
      nu.keys = {};
      nu.duckKey = "";
      if (nu.status) {
        buildStimuli(nu.status);
        buildDay(nu.status.day_choices || []);
        buildPicker(nu.status);
      }
      const helpRow = $("#neuroHelpRow");
      if (helpRow) helpRow.dataset.kind = "";
      clearReflexes();
      nu.since = 0;
      connect();
    });
  }

  // (the words first: the catalogs load before this page says anything in the demo's name)
  (window.I18N ? window.I18N.ready : Promise.resolve()).then(boot, boot);
})();
