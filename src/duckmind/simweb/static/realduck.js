/* 真的小鴨 on the simulator page (docs/real-microduck.md): the brain drives a real Microduck (or a
 * rehearsal body that plays one) over the link, with the duck's own senses only. It reads
 *   GET /api/body           real: null for a simulated body, else {rehearsal, reason, refuses: [codes],
 *                           turn_on: words while the duck is not on and waits for a person to turn it
 *                           on (duck3d.js's line, bodylink.js's card) | null,
 *                           battery: {percent, volts, charging, source, age_s} | null,
 *                           has_dock: false when this duck has no charging dock}; remote: the
 *                           link's stats (up, rtt_ms_p50, drops, max_gap_ms, …)   (app.js takeBody)
 *   /api/neuro telemetry    real: the same, without the battery; left out while the brain has no
 *                           body (starting, a restart): the page keeps its mode   (neuro.js take)
 * and then, in real mode:
 *   - body.real-duck; every control of a refused feature disabled, the reason as its title and a
 *     visible plain note (touch has no hover): GATES below, selector → the server's refusal code;
 *   - the ticker's chips: real duck / rehearsal, the link's quality, the battery (or that it is not
 *     known yet); the lab's battery bar shows the duck's own battery;
 *   - the 3D view's two checkboxes say they draw the duck's own measured joints, not physics.
 * Scripts that set `disabled` themselves ask blocked(el) first (neuro.js, app.js), call hold(scope)
 * after (layout_editor.js; neuro_sound.js once its strip is built) or let a button go with
 * release(el) (its own re-enable timer: duck3d.js, neuro.js), so a gate is never undone by them.
 * A held control is greyed out, never hidden. Words: page.real.* (catalog keys). */
"use strict";

(function realDuckModule() {
  const REASON = "page.real.no_world";
  // A control the simulated world does, the server's code for refusing it, and the visible line that
  // says why (the id of a note: an [data-real-note] line this module shows, or a line its own script
  // words). The more specific codes come first: a control takes the first refused gate it matches.
  const GATES = [
    { code: "family", selectors: ['#neuroStim [data-stim="family_walk"]'], note: "neuroStimNote" },
    { code: "battery_swap", selectors: ['#neuroStim [data-stim="swap_battery"]'], note: "neuroStimNote" },
    { code: "stimuli", selectors: ["#neuroStim [data-stim]"], note: "neuroStimNote" },
    // (the sound scenarios are the simulated world's music and hand: neuro_sound.js builds them)
    { code: "stimuli", selectors: ["#neuroSound button[data-sound]", "#soundHand"], note: "realNoteSound" },
    { code: "help_up", selectors: ["#neuroHelp", "#neuroCarry"], note: "neuroHelpNote" },
    { code: "world_buttons", selectors: ["#worldTools [data-world]"], note: "realNoteWorld" },
    { code: "demo", selectors: ["#liveDemo"], note: "realNoteLive" },
    { code: "help_up", selectors: ["#liveUpright"], note: "realNoteLive" },
    { code: "reset", selectors: ["#liveReset"], note: "realNoteLive" },
    {
      code: "layout",
      selectors: ['#homeEditor [data-le="apply"]', '#homeEditor [data-le="apply-yes"]', '#homeEditor [data-le="open"]'],
      note: "realNoteLayout",
    },
    { code: "battery_swap", selectors: ['#scenarios [data-act="swap_battery"]', "#setBatteryHelp"], note: "realNoteCore" },
  ];
  // Words that say "physics" in a simulated body and the duck's own readings with a real one:
  // [element id, which part ("label": the label around it, "span": that label's or its own first
  // span, "self"), attribute (null: the text), simulated key, real key].
  const RELABEL = [
    ["liveToggle", "span", null, "page.html.duck3dLive.label.text.db43d0", "page.real.live.toggle"],
    ["liveToggle", "label", "title", "page.html.duck3dLive.label.title.80116f", "page.real.live.toggle_title"],
    ["physicsHead", "span", null, "page.html.duck3dLive.label.text.fdf3b0", "page.real.live.head"],
    ["physicsHead", "label", "title", "page.html.duck3dLive.label.title.4881b0", "page.real.live.head_title"],
    ["worldBattery", "span", null, "page.html.worldBattery.span.text", "page.html.battery.span.text"],
    ["worldBattery", "self", "title", "page.html.worldBattery.title", "page.real.battery.title"],
    ["btnEye", "self", "title", "page.html.btnEye.title", "page.real.eye.title"],
    ["eyeImg", "self", "alt", "page.html.eyeImg.alt", "page.real.eye.alt"],
  ];

  /* What one answer says: null (a simulated body), or {rehearsal, refuses: Set}. `undefined`: the
     answer does not say (an older server, a stub). */
  function realOf(real) {
    if (real === undefined) return undefined;
    if (!real || typeof real !== "object") return null;
    return { rehearsal: !!real.rehearsal, refuses: new Set(Array.isArray(real.refuses) ? real.refuses.map(String) : []) };
  }

  function keyOf(real) {
    return real ? `${real.rehearsal ? "r" : "R"}:${[...real.refuses].sort().join(",")}` : "";
  }

  // The first refused gate this element matches, or null.
  function gateOf(real, el) {
    if (!real || !el || typeof el.matches !== "function") return null;
    for (const gate of GATES) if (real.refuses.has(gate.code) && el.matches(gate.selectors.join(", "))) return gate;
    return null;
  }

  function addToken(list, token) {
    const tokens = String(list || "").split(/\s+/).filter(Boolean);
    return tokens.includes(token) ? tokens.join(" ") : [...tokens, token].join(" ");
  }
  function dropToken(list, token) {
    return String(list || "").split(/\s+/).filter((t) => t && t !== token).join(" ");
  }

  /* Hold every refused control in `root` off, its reason as title and described by its note; let go
     of the ones this module held when nothing refuses them any more; show the notes that have a
     refused control. Writes only what changes. `t`: the catalog (key → words). */
  function apply(root, real, t) {
    const held = new Set();
    const notes = new Set();
    for (const gate of GATES) {
      if (!real || !real.refuses.has(gate.code)) continue;
      notes.add(gate.note);
      for (const el of root.querySelectorAll(gate.selectors.join(", "))) {
        if (held.has(el) || gateOf(real, el) !== gate) continue; // (a more specific gate holds it)
        held.add(el);
        if (el.dataset.realGate === undefined) el.dataset.realTitle = el.getAttribute("title") || "";
        el.dataset.realGate = gate.code;
        if (!el.disabled) el.disabled = true;
        const why = t(REASON);
        if (el.getAttribute("title") !== why) el.setAttribute("title", why);
        // (which note describes it: data-real-described — never data-real-note, which marks the notes)
        const was = el.dataset.realDescribed;
        const described = addToken(was && was !== gate.note ? dropToken(el.getAttribute("aria-describedby"), was) : el.getAttribute("aria-describedby"), gate.note);
        el.dataset.realDescribed = gate.note;
        if (el.getAttribute("aria-describedby") !== described) el.setAttribute("aria-describedby", described);
      }
    }
    for (const el of root.querySelectorAll("[data-real-gate]")) {
      if (held.has(el)) continue;
      el.disabled = false;
      if (el.dataset.realTitle) el.setAttribute("title", el.dataset.realTitle);
      else el.removeAttribute("title");
      const described = dropToken(el.getAttribute("aria-describedby"), el.dataset.realDescribed);
      if (described) el.setAttribute("aria-describedby", described);
      else el.removeAttribute("aria-describedby");
      delete el.dataset.realGate;
      delete el.dataset.realTitle;
      delete el.dataset.realDescribed;
    }
    for (const note of root.querySelectorAll("[data-real-note]")) {
      const hide = !notes.has(note.id);
      if (note.hidden !== hide) note.hidden = hide;
    }
    return held;
  }

  /* A button's own timer lets `el` go again (after a click it held off itself): back on — unless a
     real duck refuses it, which wins (also when real mode was first detected meanwhile). */
  function release(real, el) {
    if (!el) return;
    const off = !!gateOf(real, el);
    if (el.disabled !== off) el.disabled = off;
  }

  /* The link's quality in a line: {key, params, state} ("up" / "down"), or null without link stats. */
  function linkLine(remote) {
    if (!remote || typeof remote !== "object") return null;
    const drops = remote.drops || 0;
    if (!remote.up) return { key: "page.real.chip.link_down", params: { drops }, state: "down" };
    const gap = Math.round(remote.max_gap_ms || 0);
    const ms = remote.rtt_ms_p50;
    if (typeof ms === "number") return { key: "page.real.chip.link_up", params: { ms: Math.round(ms), drops, gap }, state: "up" };
    return { key: "page.real.chip.link_new", params: { drops, gap }, state: "up" };
  }

  /* The duck's own battery: {key, params, pct} (pct null: not reported yet). A real robotd does not
     say whether it charges (charging null): only a plain true is said. hasDock false: this duck has
     no charging dock (the family swaps or charges its pack), said beside the percent. */
  function batteryLine(battery, hasDock) {
    const pct = battery && Number.isFinite(Number(battery.percent)) && battery.percent !== null ? Math.max(0, Math.min(100, Math.round(Number(battery.percent)))) : null;
    if (pct === null) return { key: "page.real.chip.battery_unknown", params: {}, pct: null };
    const key = battery.charging === true ? "page.real.chip.battery_charging" : hasDock === false ? "page.real.chip.battery_no_dock" : "page.real.chip.battery";
    return { key, params: { pct }, pct };
  }

  const api = { REASON, GATES, RELABEL, realOf, gateOf, apply, release, linkLine, batteryLine };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
    return;
  }

  /* ---------------- in the page ---------------- */
  const state = { real: null, key: null };
  const I18N = () => window.I18N;
  const t = (key, params) => (I18N() ? I18N().t(key, params) : key);

  function relabel(real) {
    const i18n = I18N();
    if (!i18n) return;
    for (const [id, part, attr, simKey, realKey] of RELABEL) {
      const own = document.getElementById(id);
      const label = own && own.closest("label");
      const el = !own ? null : part === "self" ? own : part === "label" ? label : (label || own).querySelector("span");
      if (!el) continue;
      const key = real ? realKey : simKey;
      if (attr) i18n.attr(el, attr, key);
      else i18n.text(el, key);
    }
  }

  function paint(force) {
    const real = state.real;
    const key = keyOf(real);
    if (!force && key === state.key) return;
    const changed = key !== state.key;
    state.key = key;
    document.body.classList.toggle("real-duck", !!real);
    document.body.classList.toggle("real-rehearsal", !!(real && real.rehearsal));
    for (const el of document.querySelectorAll("[data-real]")) if (el.hidden === !!real) el.hidden = !real;
    if (changed) relabel(real);
    const chip = document.getElementById("chipReal");
    if (chip && real) I18N() && I18N().text(chip, real.rehearsal ? "page.real.chip.rehearsal" : "page.real.chip.real");
    apply(document, real, t);
    if (changed) window.dispatchEvent(new CustomEvent("duck:real", { detail: real ? { rehearsal: real.rehearsal, refuses: [...real.refuses] } : null }));
  }

  // One `real` field (undefined: the answer does not say; keep what is known).
  function take(raw, force) {
    const real = realOf(raw);
    if (real !== undefined) state.real = real;
    paint(!!force);
  }

  // One /api/body answer: the mode, the link's line, the battery.
  const shown = { link: "", battery: "" };
  function takeBody(info) {
    if (!info || typeof info !== "object") return;
    take(info.enabled ? info.real : null);
    const real = state.real;
    const i18n = I18N();
    if (!real || !i18n) return;
    const link = linkLine(info.remote);
    const linkChip = document.getElementById("chipLink");
    if (linkChip) {
      if (linkChip.hidden === !!link) linkChip.hidden = !link;
      const said = link ? `${link.key}|${JSON.stringify(link.params)}|${i18n.lang}` : "";
      if (link && said !== shown.link) {
        i18n.text(linkChip, link.key, link.params);
        linkChip.dataset.state = link.state;
      }
      shown.link = said;
    }
    const noDock = !!(real && info.real && info.real.has_dock === false);
    const battery = batteryLine(real && info.real ? info.real.battery : null, !noDock);
    const said = `${battery.key}|${battery.pct}|${noDock}|${state.key}|${i18n.lang}`;
    if (said === shown.battery) return;
    shown.battery = said;
    const batteryChip = document.getElementById("chipBattery");
    if (batteryChip) {
      i18n.text(batteryChip, battery.key, battery.params);
      batteryChip.dataset.state = battery.pct === null ? "unknown" : "known";
    }
    const box = document.getElementById("worldBattery");
    const bar = document.getElementById("worldBatteryBar");
    const value = document.getElementById("worldBatteryVal");
    if (box && bar && value) {
      box.hidden = false;
      i18n.attr(box, "title", noDock ? "page.real.battery.title_no_dock" : "page.real.battery.title");
      bar.style.width = `${battery.pct === null ? 0 : battery.pct}%`;
      if (battery.pct === null) i18n.text(value, "page.real.battery.unknown");
      else {
        delete value.dataset.i18n; // (a number: the bar's label says what it is)
        value.textContent = `${battery.pct}%${battery.key === "page.real.chip.battery_charging" ? t("page.app.renderWorld.d33e82") : ""}`;
      }
    }
  }

  window.duckReal = {
    ...api,
    take,
    takeBody,
    get on() {
      return !!state.real;
    },
    refuses: (code) => !!(state.real && state.real.refuses.has(code)),
    blocked: (el) => !!gateOf(state.real, el),
    // (after a script set `disabled` in `scope` itself: the refused controls there go off again)
    hold: (scope) => apply(scope || document, state.real, t),
    // (a button's own re-enable timer: on again unless a real duck refuses it)
    release: (el) => release(state.real, el),
  };
  const start = () => {
    if (I18N()) I18N().onChange(() => {
      shown.link = "";
      shown.battery = "";
      paint(true);
    });
  };
  (I18N() && I18N().ready ? I18N().ready : Promise.resolve()).then(start, start);
})();
