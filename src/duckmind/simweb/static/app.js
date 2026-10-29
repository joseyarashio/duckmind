/* Duck Core simulator front end.
 * The page only renders what duckmind.duckcore decides on the Python side; it never
 * re-implements the duck's logic. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// The page's words come from the i18n catalogs (static/i18n.js, docs/i18n.md): 繁體中文 is the
// source, five translations follow the viewer's language. T(key, values) the words of a key;
// F(obj, field) a server field in the viewer's language (in 繁體中文 exactly obj[field]);
// ZH(key) the 繁體中文 words whatever the language (what Duck Core's parsers understand).
const T = (key, values) => I18N.t(key, values);
const F = (obj, name) => (obj ? I18N.field(obj, name) ?? "" : "");
const FS = (obj, name) => (obj ? I18N.items(obj, name) : []);
const ZH = (key, values) => I18N.renderStrict({ k: key, p: values || {} }, I18N.SOURCE);
// Words a script writes into an element that carries a static data-i18n key: the key is
// dropped so a language switch does not put the static words back (the page re-renders).
function put(el, text) {
  if (!el) return;
  delete el.dataset.i18n;
  delete el.dataset.i18nHtml;
  el.textContent = text;
}
// Markup from a key (its own <b>, <br>, <code> kept; the values escaped), with ``raw`` values
// that are markup the page built itself (event links, buttons).
function H(key, values, raw) {
  const params = { ...(values || {}) };
  const marks = [];
  for (const [name, html] of Object.entries(raw || {})) {
    const mark = "\u0001" + marks.length + "\u0001";
    marks.push([mark, html]);
    params[name] = mark;
  }
  let out = I18N.tHtml(key, params);
  for (const [mark, html] of marks) out = out.split(mark).join(html);
  return out;
}

// The server's side of the contract this page was written for (simweb.server.API_VERSION).
const API_VERSION = 8;

const ui = {
  cursor: { events: 0, journal: 0 },
  state: null,
  persona: "owner",
  playing: false,
  busy: false,
  timer: null,
  tab: "graph",
  lastExpr: null,
  graphKey: "",
  sim: null,
};

// Example commands and questions (Duck Core parses 繁體中文 only: the chips show the words in
// the viewer's language and fill in the 繁體中文 command the duck understands).
const SUGGESTIONS = [
  "page.app.SUGGESTIONS.bf1f78",
  "page.app.SUGGESTIONS.fb1e46",
  "page.app.SUGGESTIONS.9b5c22",
  "page.app.SUGGESTIONS.e9cba1",
  "page.app.SUGGESTIONS.b618d2",
  "page.app.SUGGESTIONS.506f8b",
  "page.app.SUGGESTIONS.f5e433",
  "page.app.SUGGESTIONS.b90c32",
  "page.app.SUGGESTIONS.123050",
  "page.app.SUGGESTIONS.7331ef",
  "page.app.SUGGESTIONS.c7903f",
  "page.app.SUGGESTIONS.bb169f",
  "page.app.SUGGESTIONS.a3c902",
  "page.app.SUGGESTIONS.1f8166",
];
const QUERY_EXAMPLES = [
  "page.app.QUERY_EXAMPLES.bb169f",
  "page.app.QUERY_EXAMPLES.a3c902",
  "page.app.QUERY_EXAMPLES.c0a40f",
  "page.app.QUERY_EXAMPLES.e438fa",
  "page.app.QUERY_EXAMPLES.aea023",
  "page.app.QUERY_EXAMPLES.a82906",
  "page.app.QUERY_EXAMPLES.39e408",
  "page.app.QUERY_EXAMPLES.8fe855",
  "page.app.QUERY_EXAMPLES.95af34",
  "page.app.QUERY_EXAMPLES.584341",
  "page.app.QUERY_EXAMPLES.20c4c7",
  "page.app.QUERY_EXAMPLES.e9170c",
];
const TYPE_COLOR = {
  self: "--duck",
  person: "--teal",
  fact: "--reed",
  milestone: "--lilac",
  phrase: "--orange",
  song: "--orange",
  episode: "--muted",
  truth: "--rose",
};

/* ---------------- transport ---------------- */

const OFFLINE_ERROR = "page.app.OFFLINE_ERROR";

// ``opts.ff``: one step of the page's own fast-forward (its moves are stale once time is
// paused or back to real time while the answer was on its way).
async function call(method, params = {}, opts = {}) {
  let response;
  try {
    response = await fetch("/api/call", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ method, params, cursor: ui.cursor }),
    });
  } catch (error) {
    netFail();
    throw new Error(T(OFFLINE_ERROR));
  }
  let data;
  try {
    data = await response.json();
  } catch (error) {
    // A proxy's error page while the server is down, or an answer cut off by a restart.
    netFail();
    throw new Error(T("page.app.call.e8a38f", { OFFLINE_ERROR: T(OFFLINE_ERROR) }));
  }
  // A different server process answered (it was restarted): this answer belongs to the old
  // cursor, so the page starts following the new one afresh instead of applying it.
  const restarted = netAlive(data.server ? data.server.boot : null);
  if (!data.ok) throw new Error(F(data, "error") || T("page.app.call.2e6017"));
  if (!restarted) apply(data, method === "state", opts);
  return data.result;
}

async function safe(method, params, onResult, opts) {
  try {
    const result = await call(method, params, opts);
    if (onResult) onResult(result);
    return result;
  } catch (error) {
    // While the server is down the banner says so; a timer's own call adds nothing.
    if (!(opts && opts.quiet && ui.net.down)) toast("⚠️ " + error.message);
    return null;
  }
}

function apply(data, poll = false, opts = {}) {
  checkServer(data.server);
  if (data.server) syncServer(data.server);
  // The 3D view gets every event with how to show it: at its wall-clock moment (real
  // time), at once, or applied without replaying it (fast-forwards). The first answer
  // carries recent history for the lists below; the 3D view only shows what is still to
  // come, never a replay of the last hour as if it were happening again. A fast-forward
  // step that answers after time was paused (or set to real time) is applied, not played.
  const first = !ui.shownFeed;
  ui.shownFeed = true;
  const stale = !!(opts.ff && (!ui.playing || ui.realtime || ui.rtPending));
  window.dispatchEvent(
    new CustomEvent("duck:feed", {
      detail: { events: data.feed.events, fidgets: data.fidgets || [], first, stale, realtime: !!(data.server && data.server.realtime) },
    }),
  );
  appendFeed(data.feed);
  ui.cursor = data.feed.cursor;
  // Real time shows every move a moment after it happened (show_delay_ms), the 3D view
  // and the live body together. The words (banner, clock, chips) wait just as long, so
  // they never say "fell" or "asleep" before the duck on screen does. What you just did
  // (any other answer) shows at once and replaces whatever was still waiting.
  const server = data.server || {};
  if (poll && !first && server.realtime) {
    queueState(data.state, (server.now_ms || Date.now()) + (server.show_delay_ms || 0));
  } else {
    ui.stateQueue = [];
    clearTimeout(ui.stateTimer);
    showState(data.state);
  }
  maybeRefreshTab();
}

function queueState(state, at) {
  ui.stateQueue = (ui.stateQueue || []).concat([{ at, state }]);
  releaseStates();
}

function releaseStates() {
  clearTimeout(ui.stateTimer);
  const queue = ui.stateQueue || [];
  let latest = null;
  while (queue.length && queue[0].at <= Date.now() + 30) latest = queue.shift().state;
  if (latest) showState(latest);
  if (queue.length) ui.stateTimer = setTimeout(releaseStates, Math.max(20, queue[0].at - Date.now()));
}

function showState(state) {
  ui.state = state;
  window.duckLast = state; // the 3D module may load after this answer: it replays it
  renderState();
  window.dispatchEvent(new CustomEvent("duck:state", { detail: state }));
}

// A server started before the code changed keeps running the old Duck Core while it
// serves today's page: say so plainly instead of showing a frozen duck.
function checkServer(server) {
  const el = $("#serverWarn");
  if (!el || ui.net.down) return; // (an unreachable server is said first: renderNet)
  let text = "";
  if (!server || server.api !== API_VERSION) {
    text = T("page.app.checkServer.75de7e");
  } else if (server.code_changed) {
    text = T("page.app.checkServer.131fd7");
  }
  el.hidden = !text;
  el.textContent = text;
}

/* ---------------- is the server there? (stopped, restarting, back) ---------------- */

// Every answer counts as a sign of life; with none for a moment a cheap ping asks. Two
// failed tries (or one after 4 s of silence) and the page says so everywhere — banner,
// 3D clock and readout, physics line, status chip — instead of showing the last status
// as if it were now. A different server process (restarted) is followed afresh.
ui.net = { boot: null, lastOk: Date.now(), down: false, since: 0, fails: 0, checking: false };

const two = (n) => String(n).padStart(2, "0");
const hhmmss = (ms) => {
  const d = new Date(ms);
  return `${two(d.getHours())}:${two(d.getMinutes())}:${two(d.getSeconds())}`;
};

// A sign of life (boot: which server process answered). Returns whether it was restarted.
function netAlive(boot) {
  const net = ui.net;
  net.lastOk = Date.now();
  net.fails = 0;
  const restarted = !!(boot && net.boot && boot !== net.boot);
  if (boot && !net.boot) net.boot = boot;
  if (net.down) {
    net.down = false;
    const gone = Math.max(1, Math.round((Date.now() - net.since) / 1000));
    renderNet();
    if (!restarted) {
      const said = T("page.app.netAlive", { gone });
      toast(said);
      announce(said, true);
      setTimeout(() => call("state", {}).catch(() => {}), 0);
    }
  }
  if (restarted) serverRestarted(boot);
  return restarted;
}

function netFail() {
  const net = ui.net;
  net.fails += 1;
  if (net.down || !(net.fails >= 2 || Date.now() - net.lastOk > 4000)) return;
  net.down = true;
  net.since = net.lastOk;
  // The page's own clocks stop (they would only fail); the ping notices when it is back
  // and the server's answer then says whether time runs.
  clearInterval(ui.timer);
  ui.realtime = false;
  ui.rtPending = false;
  if (ui.playing) showPlaying(false);
  ui.pauseWhy = "page.app.netFail";
  renderNet();
}

// Another server process answers: the duck is followed afresh from what it has now (its
// save, or a new duck), and the page says so instead of mixing the two.
function serverRestarted(boot) {
  const before = ui.state;
  ui.net.boot = boot;
  ui.cursor = { events: 0, journal: 0 };
  ui.shownFeed = false;
  ui.graphKey = "";
  ui.stateQueue = [];
  clearTimeout(ui.stateTimer);
  $("#laneTruth").innerHTML = "";
  $("#laneDuck").innerHTML = "";
  clearInterval(ui.timer);
  ui.realtime = false;
  ui.rtPending = false;
  if (ui.playing) showPlaying(false);
  window.dispatchEvent(new CustomEvent("duck:reset"));
  call("state", {})
    .then(() => {
      // The same duck from its save (its clock carried on), or a new life: a server run
      // without a save (or with another one) hatches afresh.
      const now = ui.state;
      const same = !!(before && now) && before.identity.seed === now.identity.seed && before.identity.hatched_at === now.identity.hatched_at && before.identity.name === now.identity.name;
      const carried = same && Date.parse(now.clock.iso) >= Date.parse(before.clock.iso) - 120000;
      const who = !now
        ? T("page.app.serverRestarted.09c8dd")
        : carried
          ? T("page.app.serverRestarted.c26680", { now_name: now.identity.name })
          : same
            ? T("page.app.serverRestarted.c33987", { now_name: now.identity.name })
            : T("page.app.serverRestarted.43ce21", { now_name: now.identity.name });
      const said = T("page.app.serverRestarted.728a5c", {
        time: hhmmss(Date.now()),
        who,
        realtime: T(ui.realtime ? "page.app.serverRestarted.ad8dc2" : "page.app.serverRestarted.cf09b1"),
      });
      toast(said);
      announce(said, true);
      refreshTab(true);
    })
    .catch(() => {});
}

async function heartbeat(force = false) {
  const net = ui.net;
  if (net.checking || (!force && !net.down && Date.now() - net.lastOk < 2500)) return;
  net.checking = true;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 2500);
  try {
    const response = await fetch("/api/ping", { cache: "no-store", signal: ctrl.signal });
    let info = null;
    try {
      info = await response.json();
    } catch (error) {
      info = null;
    }
    // (404: an older server without the ping, which is there all the same)
    if (!response.ok && response.status !== 404) throw new Error(String(response.status));
    netAlive(info && info.boot);
  } catch (error) {
    netFail();
  } finally {
    clearTimeout(timer);
    net.checking = false;
    if (net.down) renderBanner(); // the "for how long" counts on
  }
}
// (started in boot(), once the words are loaded)

function renderNet() {
  const net = ui.net;
  document.body.classList.toggle("server-down", net.down);
  const warn = $("#serverWarn");
  if (net.down) {
    warn.hidden = false;
    warn.textContent = T("page.app.renderNet.d744f8", { time: hhmmss(net.since) });
    const chip = $("#chipMode");
    put(chip, T("page.app.renderNet.ee7586"));
    chip.className = "chip mode-warn";
  } else {
    checkServer(ui.server);
    if (ui.state) renderState();
  }
  window.dispatchEvent(new CustomEvent("duck:net", { detail: { down: net.down, since: net.since } }));
  renderBanner();
}

/* ---------------- state rendering ---------------- */

// The weekday of the duck's clock: 繁體中文 the server's own character, other languages theirs.
function weekdayOf(clock) {
  if (I18N.isSource() || I18N.lang === I18N.PSEUDO) return clock.weekday;
  const day = new Date(String(clock.now || "").slice(0, 10) + "T12:00:00");
  return Number.isNaN(day.getTime()) ? clock.weekday : I18N.weekday((day.getDay() + 6) % 7);
}

const MODE_KEYS = {
  off: "page.app.renderState.177116",
  nap: "page.app.renderState.827011",
  awake: "page.app.renderState.60ddd9",
  stop: "page.app.renderState.1d374e",
  lockout: "page.app.renderState.6294ba",
  yield: "page.app.renderState.82d6f9",
  quiet: "page.app.renderState.ad373e",
  hot: "page.app.renderState.9e46ea",
  bedtime: "page.app.renderState.64bf17",
  fallen: "page.app.renderState.022d9e",
  stranded: "page.app.renderState.0ceda4",
  unplaced: "page.app.renderState.5469b8",
  homing: "page.app.renderState.4f8e20",
};

function renderState() {
  const st = ui.state;
  if (!st) return;
  if (!document.body.classList.contains("neuro-mode")) {
    // (neural duck prototype: the topbar shows the neural duck's own clock — neuro.js)
    $("#clockNow").textContent = st.clock.now;
    $("#clockWeek").textContent = T("page.app.renderState.c776fd", { weekday: weekdayOf(st.clock) });
  }
  put($("#chipGd"), T("page.app.renderState.6f6ca4", { gd: st.clock.gd, half: st.clock.half ? "½" : "" }));
  put($("#chipStage"), F(st.clock, "stage_zh"));
  const mode = $("#chipMode");
  let modeText = MODE_KEYS[st.body.mode] ? T(MODE_KEYS[st.body.mode]) : st.body.mode;
  // A duck with a nest: where its day is at (in the nest, out on a trip, the family asleep).
  const nestNow = st.body.nest;
  if (nestNow && (st.body.mode === "awake" || (st.body.mode === "bedtime" && nestNow.nested))) {
    modeText = T(
      nestNow.nested
        ? nestNow.charging
          ? "page.app.renderState.33a5d3"
          : "page.app.renderState.41aed7"
        : st.body.excursion
          ? "page.app.renderState.e2451a"
          : "page.app.renderState.60ddd9",
    );
  }
  if (st.body.day && st.body.day.phase === "solo" && st.body.awake) modeText += T("page.app.renderState.b936b0");
  put(mode, modeText);
  mode.className = "chip " + (st.body.mode === "off" ? "mode-off" : st.body.mode === "nap" ? "mode-nap" : st.body.mode === "awake" ? "mode-awake" : "mode-warn");
  if (ui.net.down) {
    // (a state that was still waiting for its moment: the chip keeps saying the server is gone)
    put(mode, T("page.app.renderState.ee7586"));
    mode.className = "chip mode-warn";
  }

  $("#duckName").textContent = st.identity.name;
  const branch = $("#duckBranch");
  branch.hidden = !st.growth.branch && !st.growth.adult;
  branch.textContent = st.growth.adult
    ? T("page.app.renderState.6ef68c", { adult: F(st.growth, "adult"), branch: F(st.growth, "branch") })
    : F(st.growth, "branch") || "";
  $("#reasonText").textContent = F(st.body, "reason");
  const off = st.body.mode === "off";
  // Fast-forwarding a switched-off duck would only skip its days: the page's own fast
  // speed stops and the banner says why. Real time never stops by itself.
  if (off && ui.wasAwake && ui.playing && !ui.realtime) {
    setPlaying(false);
    // (the server's own 繁體中文 words say whether it ran flat: 「沒電」)
    ui.pausedFlat = !!(st.body.banner && /\u6c92\u96fb/.test(st.body.banner.text || ""));
    ui.pauseWhy = ui.pausedFlat ? "page.app.renderState.cf0dc6" : "page.app.renderState.66fda6";
  }
  ui.wasAwake = !off;

  const stage = $("#duckStage");
  stage.classList.toggle("off", !st.body.awake);
  stage.classList.toggle("awake", st.body.awake && !st.body.napping);
  stage.classList.toggle("sleep", st.body.napping || st.body.mode === "off");
  stage.classList.toggle("rest", ["stop", "lockout", "hot", "yield"].includes(st.body.mode));
  for (const posture of ["stand", "walk", "fallen"]) stage.classList.toggle("posture-" + posture, st.body.posture === posture);
  $("#chipFloor").hidden = !st.body.floor;
  const badge = $("#stageBadge");
  const badgeText = st.body.yielding
    ? T("page.app.renderState.ec1d61")
    : st.body.guest_mode
      ? T("page.app.renderState.eee3c9")
      : st.body.conservative
        ? T("page.app.renderState.9e6092")
        : "";
  badge.hidden = !badgeText;
  badge.textContent = badgeText;

  const last = st.body.last;
  if (last && last.event_id !== ui.lastExpr) {
    ui.lastExpr = last.event_id;
    // Without 3D the 2D duck plays the feed's moves in order (see play2d); this is
    // only the fallback for a first answer.
    if (st.body.awake && !ui.moves2d.length && !document.body.classList.contains("has3d")) animateVerb(last.verb, F(last, "sound_text"));
  }

  const trying = $("#trying");
  trying.hidden = !st.body.trying;
  if (st.body.trying) {
    const t = st.body.trying;
    const words = { text: F(t, "text") };
    $("#tryingText").textContent =
      t.kind === "phrase"
        ? T("page.app.renderState.e7ce00", words)
        : t.kind === "moment"
          ? T("page.app.renderState.9c1b33", words)
          : T("page.app.renderState.f8a09a", words);
  }

  const battery = st.body.battery;
  $("#batteryBar").style.width = `${battery}%`;
  $("#batteryBar").style.background = battery <= 15 ? "var(--orange)" : "var(--reed)";
  $("#batteryVal").textContent = st.clock.docked ? T("page.app.renderState.d551e9") : `${battery}%`;

  renderMeters(st.mood);
  renderQuirks(st.identity);
  $("#setFloor").checked = st.body.floor;
  // Where the duck is, right under the 3D view (why it does or does not walk is the
  // banner's job, in Duck Core's own words).
  const spots = (st.body.avoid_spots || []).length;
  const nest = st.body.nest;
  const day = st.body.day;
  let dayText = day ? T("page.app.renderState.acf104", { phase_zh: F(day, "phase_zh"), family_zh: F(day, "family_zh") }) : "";
  if (day && nest && day.phase === "solo") {
    // The family still asleep: its outings stay close to the nest, quietly.
    dayText = T("page.app.renderState.e5101b", { phase_zh: F(day, "phase_zh"), leash_m: nest.leash_m ?? Math.min(1.5, nest.roam_m) });
  }
  const spotsText = spots ? T("page.app.renderState.96844d", { spots }) : "";
  if (nest) {
    // A duck with a nest: where it is relative to its nest, and what it is up to.
    const where = nest.nested
      ? T("page.app.renderState.d09556", { charging: nest.charging ? T("page.app.renderState.5564fa") : "" })
      : nest.unplaced
        ? T("page.app.renderState.cd19ef")
        : nest.stranded
          ? T("page.app.renderState.38cb80")
          : nest.homing
            ? T("page.app.renderState.ce3661", { dist_m: nest.dist_m })
            : st.body.excursion
              ? T("page.app.renderState.4d0c48", { dist_m: nest.dist_m })
              : T("page.app.renderState.3e507e", { dist_m: nest.dist_m });
    put($("#placeText"), T("page.app.renderState.ba2441", { where, docks: nest.docks, dayText, spots: spotsText }));
  } else {
    put(
      $("#placeText"),
      st.body.floor
        ? T("page.app.renderState.3dfb9d", {
            position: st.body.position ? T("page.app.renderState.09ed19", { position: Math.hypot(st.body.position[0], st.body.position[1]) }) : "",
            spots: spotsText,
            dayText,
          })
        : T("page.app.renderState.13f590", { dayText }),
    );
  }
  if (nest) {
    // A nest duck is never "on the table": back into its nest, or out onto the floor.
    put($("#btnPlace"), T(nest.nested ? "page.app.renderState.66ae0c" : "page.app.renderState.0412fe"));
    $("#btnPlace").dataset.nest = nest.nested ? "out" : "in";
  } else {
    put($("#btnPlace"), T(st.body.floor ? "page.app.renderState.b90041" : "page.app.renderState.58ec22"));
    $("#btnPlace").dataset.nest = "";
  }
  $("#btnPlace").dataset.floor = st.body.floor ? "1" : "";
  // Every routine works with a nest duck too (the family puts it back in its nest instead
  // of switching it off; it charges there and lives its own day between visits).
  $("#setDocked").checked = st.clock.docked;
  $("#setBatteryHelp").checked = !!st.body.battery_help;
  $("#setNtp").checked = st.clock.ntp_ok;
  $("#setHot").checked = st.body.hot;
  $("#setConservative").checked = st.body.conservative;
  put($("#btnTv"), T(st.body.tv ? "page.app.renderState.db69b5" : "page.app.renderState.aaa9bf"));

  renderBanner();
  if (ui.tab === "growth") renderGrowth();
  if (ui.tab === "voice") renderVoice();
}

/* ---------------- the banner over the duck: what it is doing, why, and what to do ---------------- */

const SPEED_KEYS = { rt: "page.app.SPEED_ZH.rt", 1: "page.app.SPEED_ZH.1", 5: "page.app.SPEED_ZH.5", 15: "page.app.SPEED_ZH.15" };
const speedWords = (speed) => (SPEED_KEYS[speed] ? T(SPEED_KEYS[speed]) : speed);

function offsetText(minutes) {
  const m = Math.abs(minutes);
  const words =
    m >= 60
      ? T("page.app.offsetText.c5bf9f", { m: Math.floor(m / 60), m2: m % 60 ? T("page.app.offsetText.3d5262", { m: m % 60 }) : "" })
      : T("page.app.offsetText.a6c433", { m });
  return T("page.app.offsetText.70fc14", { minutes: T(minutes > 0 ? "page.app.offsetText.e6e99d" : "page.app.offsetText.9aadfb"), words });
}

// The server does not answer: the banner says so (with what it last said, marked as the
// last word), and offers to try again now; the time line stops claiming time runs.
function renderOfflineBanner(el) {
  const net = ui.net;
  const st = ui.state;
  const last = st && st.body.banner ? F(st.body.banner, "title") : "";
  const seconds = Math.max(1, Math.round((Date.now() - net.since) / 1000));
  el.dataset.tone = "warn";
  $("#bannerTitle").textContent = T("page.app.renderOfflineBanner.78d1e7");
  $("#bannerText").textContent = T("page.app.renderOfflineBanner.6a2f4f", {
    time: hhmmss(net.since),
    seconds,
    last: last ? T("page.app.renderOfflineBanner.a9246a", { last }) : "",
  });
  if ($("#bannerTags").innerHTML) $("#bannerTags").innerHTML = "";
  renderBannerActions([{ label: T("page.app.renderOfflineBanner.183cef"), calls: [], local: "reconnect", primary: true }]);
  const time = $("#bannerTime");
  time.dataset.state = "offline";
  put($("#bannerTimeText"), T("page.app.renderOfflineBanner.b59b60"));
  $("#bannerPlay").hidden = true;
  $("#bannerAlign").hidden = true;
  $("#bannerMorning").hidden = true;
}

function renderBanner() {
  const st = ui.state;
  const el = $("#duckBanner");
  if (!el) return;
  if (ui.net.down) {
    renderOfflineBanner(el);
    return;
  }
  if (!st) return;
  // An older server sends no banner: still say what Duck Core says about the duck.
  const b = st.body.banner || { tone: "info", title: F(st.body, "reason") || "—", text: "", actions: [], tags: [] };
  el.dataset.tone = b.tone || "info";
  $("#bannerTitle").textContent = (b.night ? "🌙 " : "") + (F(b, "title") || "");
  let text = F(b, "text") || "";
  const offset = (ui.server && ui.server.offset_min) || 0;
  if (b.night && ui.playing && ui.realtime && Math.abs(offset) < 2) text += T("page.app.renderBanner.2e912c");
  $("#bannerText").textContent = text;
  const tags = FS(b, "tags").map((tag) => `<span>${esc(tag)}</span>`).join("");
  if ($("#bannerTags").innerHTML !== tags) $("#bannerTags").innerHTML = tags;
  renderBannerActions((b.actions || []).map((action) => ({ ...action, label: F(action, "label") })));
  // Time: paused (and why), real time (and whether it matches your clock), or fast.
  const speed = $("#speed").value;
  const time = $("#bannerTime");
  const morning = $("#bannerMorning");
  morning.hidden = true;
  $("#bannerPlay").classList.add("primary");
  if (!ui.playing) {
    time.dataset.state = "paused";
    let text = T("page.app.renderBanner.ef4033", {
      now: String(st.clock.now || "").slice(5),
      pauseWhy: ui.pauseWhy ? T("page.app.renderBanner.48e1c3", { pauseWhy: T(ui.pauseWhy) }) : "",
    });
    // Real time follows your clock: say beforehand when that means night (asleep, no
    // walking), and offer the morning instead.
    const lands = speed === "rt" ? landingTime(st, offset) : null;
    if (lands && lands.night && st.body.awake) {
      text += T("page.app.renderBanner.0e6176", { lands_text: lands.text });
      morning.hidden = false;
      $("#bannerPlay").classList.remove("primary");
    }
    put($("#bannerTimeText"), text);
    $("#bannerPlay").hidden = false;
    put($("#bannerPlay"), T("page.app.renderBanner.d0c957", { speed: speedWords(speed) }));
    $("#bannerAlign").hidden = true;
  } else if (ui.realtime || ui.rtPending) {
    // (Pending: ▶ was pressed and the server has not confirmed real time yet — never
    // shown as a fast-forward in between.)
    time.dataset.state = "realtime";
    put(
      $("#bannerTimeText"),
      !ui.realtime
        ? T("page.app.renderBanner.3d43e0")
        : Math.abs(offset) < 2
          ? T("page.app.renderBanner.21203b")
          : T("page.app.renderBanner.bd9ca3", { offset: offsetText(offset) }),
    );
    $("#bannerPlay").hidden = true;
    $("#bannerAlign").hidden = !ui.realtime || Math.abs(offset) < 2;
    if (ui.realtime && Math.abs(offset) >= 2) put($("#bannerAlign"), alignLabel(offset));
  } else {
    time.dataset.state = "fast";
    put(
      $("#bannerTimeText"),
      T("page.app.renderBanner.4bdc54", {
        speed: speedWords(speed),
        speed2: T(speed === "1" ? "page.app.renderBanner.da919b" : "page.app.renderBanner.544004"),
      }),
    );
    $("#bannerPlay").hidden = true;
    $("#bannerAlign").hidden = true;
  }
}

// The banner's buttons are rebuilt only when the list of actions changes; otherwise the
// same buttons stay and only their words are updated. Real time re-renders the banner
// every second, and a rebuilt button under a press (or with keyboard focus) swallows the
// click.
function renderBannerActions(actions) {
  ui.bannerActions = actions;
  const box = $("#bannerActions");
  const buttons = Array.from(box.querySelectorAll("button"));
  if (buttons.length !== actions.length) {
    box.innerHTML = actions
      .map((action, index) => `<button type="button" class="btn small${action.primary ? " primary" : ""}" data-banner="${index}">${esc(action.label)}</button>`)
      .join("");
  } else {
    actions.forEach((action, index) => {
      if (buttons[index].textContent !== action.label) buttons[index].textContent = action.label;
      buttons[index].classList.toggle("primary", !!action.primary);
    });
  }
  box.querySelectorAll("button").forEach((button) => (button.disabled = !!ui.busyBanner));
}

const hhmm = (date) => `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
const nightHour = (hour) => hour >= 22 || hour < 7;

// Where ▶ in real time puts the duck: at your time of day (it catches up to your clock),
// unless its own clock is ahead of yours by less than half a day (then it keeps its own).
function landingTime(st, offset) {
  if (offset > 0) {
    const text = String(st.clock.now || "").slice(-5);
    return { text, night: nightHour(Number(text.slice(0, 2))) };
  }
  const wall = new Date();
  return { text: hhmm(wall), night: nightHour(wall.getHours()) };
}

// "Align with my clock" lives the time up to your time of day: say how long, and whether
// that lands in the night.
function alignLabel(offset) {
  const minutes = (((-offset) % 1440) + 1440) % 1440;
  const words =
    minutes >= 60
      ? T("page.app.alignLabel.9c8bfe", { minutes: Math.floor(minutes / 60), minutes2: minutes % 60 ? T("page.app.alignLabel.d38ada", { minutes: minutes % 60 }) : "" })
      : T("page.app.alignLabel.fb8b9d", { minutes });
  const wall = new Date();
  return T("page.app.alignLabel.3f8450", { words, time: hhmm(wall), night: nightHour(wall.getHours()) ? T("page.app.alignLabel.aa1e68") : "" });
}

async function runBannerAction(action) {
  if (!action || ui.busyBanner) return;
  if (action.local === "reconnect") {
    announce(T("page.app.runBannerAction.8b6625"));
    await heartbeat(true);
    if (ui.net.down) toast(T("page.app.runBannerAction.a7a7c2"));
    return;
  }
  ui.busyBanner = true;
  announce(`${action.label}…`);
  $$("#bannerActions button").forEach((button) => (button.disabled = true));
  let result = null;
  try {
    for (const step of action.calls) {
      const params = step.method === "act" ? { persona: ui.persona, ...step.params } : step.params;
      result = await safe(step.method, params);
      if (result === null) break;
    }
  } finally {
    ui.busyBanner = false;
    renderBanner();
  }
  if (result === null) return;
  const note = result && result.note ? String(F(result, "note")) : "";
  const slept = result && result.until === "nap_end" ? T("page.app.runBannerAction.4650f5", { seconds: Math.max(1, Math.round((result.seconds || 0) / 60)) }) : "";
  const said = note
    ? withLabel(action.label, note)
    : result && result.routine
      ? T("page.app.runBannerAction.c38b35", { action_label: action.label, routine: F(result, "routine"), gd_from: result.gd_from, gd_to: result.gd_to })
      : slept
        ? T("page.app.runBannerAction.7a8f5b", { action_label: action.label, slept })
        : result && result.minutes
          ? T("page.app.runBannerAction.48b353", { action_label: action.label, minutes: result.minutes })
          : action.label;
  announce(said, true);
  toast(said);
  // The point of every banner button is to see the duck live: time flows again. After it
  // ran flat in a fast-forward with nobody swapping batteries, fast time would only run
  // it flat again within seconds: it carries on in real time instead.
  ui.pauseWhy = "";
  if (!ui.playing) {
    if (ui.pausedFlat && ui.state && !ui.state.body.battery_help && $("#speed").value !== "rt") {
      $("#speed").value = "rt";
      toast(T("page.app.runBannerAction.c463af", { said }));
    }
    ui.pausedFlat = false;
    // Show this action live: real time from the duck's own clock, no catching up hours
    // (「對齊我的時鐘」 does that when you want it).
    setPlaying(true, { catchUp: false });
  }
}

// "What was pressed: what came of it" (the answer already names it: said once).
function withLabel(label, note) {
  const text = String(note || "");
  return text.startsWith(label) ? text : T("page.app.withLabel", { label, text });
}

// Say it on the 3D view right away (before the server answers): a click that shows
// nothing for a second looks broken, on a slow device most of all. ``done``: the answer.
function announce(text, done = false) {
  window.dispatchEvent(new CustomEvent("duck:caption", { detail: { text, strong: !done, reply: done } }));
}

$("#bannerActions").addEventListener("click", (event) => {
  const button = event.target.closest("[data-banner]");
  if (button) runBannerAction(ui.bannerActions[Number(button.dataset.banner)]);
});
$("#bannerPlay").addEventListener("click", () => {
  ui.pauseWhy = "";
  setPlaying(true);
  announcePlay();
});

// ▶ / ⏸ say on the 3D view what they did (a switched-off duck does not move when time
// flows: say so, instead of a click that seems to do nothing).
function announcePlay() {
  const st = ui.state;
  const speed = $("#speed").value;
  // (Said like a click's own echo: always shown at once, and never holding back the
  // server's answer that may follow, such as how long real time caught up.)
  if (!ui.playing) {
    announce(T("page.app.announcePlay.7d49ad", { value: st ? String(st.clock.now).slice(5) : T("page.app.announcePlay.1f72b0") }));
    return;
  }
  const off = st && !st.body.awake;
  announce(T("page.app.announcePlay.da11ae", { speed: speedWords(speed), off: off ? T("page.app.announcePlay.9f97bd") : "" }));
}
$("#bannerAlign").addEventListener("click", async () => {
  announce(T("page.app.on_bannerAlign.5d8af4"));
  const result = await safe("realtime", { on: true, align: true });
  if (result) {
    const said = result.caught_up_minutes
      ? T("page.app.on_bannerAlign.b3a790", { caught_up_minutes: result.caught_up_minutes })
      : T("page.app.on_bannerAlign.b9040c");
    toast(said);
    announce(said, true);
  }
});
// Night on your clock: start the duck's day at 07:00 instead (it keeps its own clock from
// there, at real speed, until you align it).
$("#bannerMorning").addEventListener("click", async () => {
  announce(T("page.app.on_bannerMorning.a23c28"));
  const result = await safe("advance", { until: "07:00" });
  if (result === null) return;
  ui.pauseWhy = "";
  setPlaying(true);
  const said = T("page.app.on_bannerMorning.bb1d44");
  announce(said, true);
  toast(T("page.app.on_bannerMorning.230117", { said, minutes: result.minutes }));
});

// ``calls``: the sounds the move made (the feed's tags), played in the duck's own voice
// (voice.js), which then opens the 2D beak with the sound itself.
function animateVerb(verb, soundText, calls) {
  const stage = $("#duckStage");
  stage.className.split(" ").filter((c) => c.startsWith("v-")).forEach((c) => stage.classList.remove(c));
  void stage.offsetWidth;
  stage.classList.add("v-" + verb);
  const voiced = voice2d(calls, 2900);
  if (soundText) {
    if (!voiced) {
      stage.classList.add("mouth");
      setTimeout(() => stage.classList.remove("mouth"), 700);
    }
    bubble(soundText);
  }
  clearTimeout(ui.verbTimer);
  ui.verbTimer = setTimeout(() => stage.classList.remove("v-" + verb), 2900);
}

function bubble(text) {
  const el = $("#bubble");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(ui.bubbleTimer);
  ui.bubbleTimer = setTimeout(() => el.classList.remove("show"), 2200);
}

function renderMeters(mood) {
  const rows = [
    [T("page.app.renderMeters.300ee5"), mood.valence, true],
    [T("page.app.renderMeters.3bc405"), mood.energy],
    [T("page.app.renderMeters.bfd06f"), mood.arousal],
    [T("page.app.renderMeters.818022"), mood.social],
    [T("page.app.renderMeters.c832b5"), mood.overload],
    [T("page.app.renderMeters.b9e43c"), mood.bond_owner],
  ];
  $("#meters").innerHTML = rows
    .map(([label, value, centered]) => {
      if (centered) {
        const w = Math.min(50, Math.abs(value) / 2);
        const left = value < 0 ? 50 - w : 50;
        return `<div class="meter"><span>${esc(label)}</span><div class="bar center"><i style="left:${left}%;width:${w}%;background:${value < 0 ? "var(--rose)" : "var(--orange)"}"></i></div><span class="val">${value > 0 ? "+" : ""}${value}</span></div>`;
      }
      return `<div class="meter"><span>${esc(label)}</span><div class="bar"><i style="width:${Math.max(0, Math.min(100, value))}%"></i></div><span class="val">${value}%</span></div>`;
    })
    .join("");
}

function renderQuirks(identity) {
  const rows = identity.quirks.map((q) => `<div class="quirk">${esc(F(q, "text"))}</div>`);
  for (let i = 0; i < identity.hidden_quirks; i++) rows.push(`<div class="quirk hidden">${esc(T("page.app.renderQuirks.c2f789"))}</div>`);
  const hw = identity.hardware;
  const has = (on) => T(on ? "page.app.renderQuirks.fbd5b7" : "page.app.renderQuirks.a9b748");
  rows.push(
    `<p class="hint">${I18N.tHtml("page.app.renderQuirks.a1d293", {
      seed: identity.seed,
      table_hash: identity.table_hash,
      tof: has(hw.tof),
      pickup_detect: has(hw.pickup_detect),
      ear: T(hw.ear ? "page.app.renderQuirks.619163" : "page.app.renderQuirks.1c4613"),
      name_ear: T(hw.name_ear ? "page.app.renderQuirks.d4342d" : "page.app.renderQuirks.ba08ad"),
      motion_validated: T(hw.motion_validated ? "page.app.renderQuirks.90ff8a" : "page.app.renderQuirks.ef3626"),
    })}</p>`,
  );
  $("#quirks").innerHTML = rows.join("");
}

/* ---------------- feed ---------------- */

function appendFeed(feed) {
  const truth = $("#laneTruth");
  const duck = $("#laneDuck");
  for (const item of feed.journal) {
    const li = document.createElement("li");
    li.className = "new" + (item.perceived ? "" : " miss");
    li._feed = { item, lane: "truth" };
    li.innerHTML = feedRow(li._feed);
    truth.prepend(li);
  }
  let speak = null;
  for (const item of feed.events) {
    const li = document.createElement("li");
    li.className = "new";
    li._feed = { item, lane: "duck" };
    li.innerHTML = feedRow(li._feed);
    duck.prepend(li);
    if (item.bubble) speak = item;
  }
  for (const lane of [truth, duck]) while (lane.children.length > 250) lane.lastChild.remove();
  if (speak && ui.state && ui.state.body.awake) bubble(F(speak, "bubble"));
}

// One row of the feed lanes (kept on the row: a language switch writes it again).
function feedRow({ item, lane }) {
  if (lane === "truth") {
    return `<span class="t">${esc(item.at.slice(6))}</span><span><span class="tag">${esc(F(item, "persona_zh"))}</span>${esc(F(item, "detail"))}${item.event_ids
      .slice(0, 2)
      .map((id) => `<span class="eid" data-eid="${esc(id)}">${esc(id)}</span>`)
      .join("")}</span>`;
  }
  return `<span class="t">${esc(item.at.slice(6))}</span><span><span class="tag ${esc(item.category)}">${esc(F(item, "category_zh"))}</span>${esc(F(item, "text"))}<span class="eid" data-eid="${esc(item.event_id)}">${esc(item.event_id)}</span></span>`;
}

function rerenderFeed() {
  for (const li of document.querySelectorAll("#laneTruth > li, #laneDuck > li")) if (li._feed) li.innerHTML = feedRow(li._feed);
}

document.addEventListener("click", async (event) => {
  const target = event.target.closest(".eid");
  if (!target) return;
  const result = await safe("event", { event_id: target.dataset.eid });
  if (result && result.event) showEvent(result.event);
});

function showEvent(ev, again = false) {
  if (!again) setTab("graph");
  ui.detail = () => showEvent(ev, true);
  $("#gDetail").innerHTML = `<h3>${I18N.tHtml("page.app.showEvent.2bf68d", { event_id: ev.event_id })}</h3>
    <p>${esc(F(ev, "text"))}</p>
    <p class="meta">${esc(
      T("page.app.showEvent.90fa38", {
        kind: ev.kind,
        occurred_at: ev.occurred_at,
        kind2: ev.source.kind,
        source_id: ev.source.source_id,
        reliability: ev.source.reliability,
        trace_id: ev.trace_id,
      }),
    )}</p>
    ${ev.parent_event_ids.length ? `<p class="meta">${H("page.app.showEvent.108a28", {}, { span: ev.parent_event_ids.map((id) => `<span class="eid" data-eid="${esc(id)}">${esc(id)}</span>`).join(" ") })}</p>` : ""}
    <details><summary>${esc(T("page.app.showEvent.8a9c36"))}</summary><pre class="mono" style="white-space:pre-wrap">${esc(JSON.stringify(ev.payload, null, 2))}</pre></details>`;
}

/* ---------------- actions ---------------- */

$$(".persona").forEach((button) =>
  button.addEventListener("click", () => {
    ui.persona = button.dataset.persona;
    $$(".persona").forEach((b) => {
      b.classList.toggle("on", b === button);
      b.setAttribute("aria-checked", b === button ? "true" : "false");
    });
  }),
);

// Scenario buttons and settings live in more than one place on the page (the interaction
// cards and the simulation tools); every [data-scenarios] container gets the same handlers.
async function onScenarioClick(event) {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.dataset.act) {
    let params = button.dataset.params ? JSON.parse(button.dataset.params) : {};
    // What was pressed, read before the answer relabels the button (the TV button says
    // 開電視 / 關電視 by what the TV is now).
    let label = button.textContent.trim();
    if (button.dataset.act === "tv") {
      params = { on: !(ui.state && ui.state.body.tv) };
      label = T(params.on ? "page.app.onScenarioClick.aaa9bf" : "page.app.onScenarioClick.db69b5");
    }
    announce(`${label}…`);
    // (`visit`: as this family member — with live physics that person walks over first)
    const result = await safe("act", { action: button.dataset.act, persona: ui.persona, params, visit: true });
    // (Duck Core's own short 繁體中文 note 「沒有 ToF」: said in full)
    if (result && result.note === "\u6c92\u6709 ToF") {
      result.note = T("page.app.onScenarioClick.1b8d35");
      delete (result._i18n || {}).note;
    }
    const note = result ? F(result, "note") : "";
    // The toast keeps the explanation readable after a move's words replace the caption.
    if (note) toast(withLabel(label, note));
    // A family member walks over first (live physics, D3): the duck feels it a few seconds
    // later; follow along meanwhile even while time is paused.
    if (result && result.deferred) followVisit();
    // Say on the 3D view too when the duck could not notice it (no sensor, asleep, off):
    // otherwise a button that changes nothing looks broken.
    if (result && result.perceived === false) {
      announce(T("page.app.onScenarioClick.79d959", { note: note || T("page.app.onScenarioClick.70f169") }), true);
    } else if (note) {
      // The duck's own summary; a move it makes replaces it when it starts.
      announce(withLabel(label, note), true);
    }
  } else if (button.dataset.world) {
    // The physics world's own buttons (示範, roll a ball, tidy up, set the body upright):
    // never Duck Core's decisions, and it is not told.
    const params = { kind: button.dataset.kind || "", thing: $("#worldThing").value };
    const label = button.textContent.trim();
    announce(`${label}…`);
    const result = await safe(button.dataset.world, params);
    if (result && result.note) {
      toast(withLabel(label, F(result, "note")));
      announce(withLabel(label, F(result, "note")), true);
    }
  } else if (button.dataset.party) {
    toast(T("page.app.onScenarioClick.c80b2b"));
    announce(`${button.textContent.trim()}…`);
    const result = await safe("party", { guest_mode: button.dataset.party === "true" });
    if (result) {
      const counted = Object.values(result.care_after).reduce((a, b) => a + b, 0) - Object.values(result.care_before).reduce((a, b) => a + b, 0);
      const said = T("page.app.onScenarioClick.b47337", { touches: result.touches, counted, note: F(result, "note") });
      toast(said);
      announce(said, true);
    }
  } else if (button.id === "btnLive") {
    const days = Number($("#liveDays").value);
    const routine = $("#routine").value;
    button.disabled = true;
    toast(T("page.app.onScenarioClick.7c1d78", { days }));
    // "family:<kind>": a whole-day family (the duck lives its own day with them).
    const result = await safe(
      "live",
      routine.startsWith("family:") ? { days, family: routine.slice(7) } : { days, routine },
    );
    button.disabled = false;
    if (result) {
      const st = ui.state;
      const now = st ? T("page.app.onScenarioClick.a9c7b0", { now: st.clock.now.slice(5), banner_title: F(st.body.banner, "title") }) : "";
      const milestones = FS(result, "new_milestones");
      toast(
        T("page.app.onScenarioClick.ee4f1d", {
          routine: F(result, "routine"),
          gd_from: result.gd_from,
          gd_to: result.gd_to,
          events: result.events,
          new_milestones: milestones.length ? T("page.app.onScenarioClick.3be992", { new_milestones: I18N.list(milestones) }) : "",
          now,
        }),
      );
      window.dispatchEvent(new CustomEvent("duck:caption", { detail: T("page.app.onScenarioClick.1cbe02", { routine: F(result, "routine"), days: result.days }) }));
      refreshTab(true);
    }
  }
}

async function onScenarioChange(event) {
  const map = { setFloor: "floor", setDocked: "docked", setNtp: "ntp_ok", setHot: "hot", setConservative: "conservative", setBatteryHelp: "battery_help" };
  const key = map[event.target.id];
  if (!key) return;
  const label = (event.target.closest("label") || event.target).textContent.trim();
  announce(T("page.app.onScenarioChange.bf3557", { checked: T(event.target.checked ? "page.app.onScenarioChange.492596" : "page.app.onScenarioChange.13d930"), label }));
  const result = await safe("set_setting", { key, value: event.target.checked });
  if (result && result.note) {
    toast(F(result, "note"));
    announce(T("page.app.onScenarioChange.fc4eb4", { note: F(result, "note") }), true);
  }
}

$$("[data-scenarios]").forEach((box) => {
  box.addEventListener("click", onScenarioClick);
  box.addEventListener("change", onScenarioChange);
});

$("#btnTryingGood").addEventListener("click", async () => {
  const action = ui.persona === "owner" ? "owner_good" : "pad_good";
  const result = await safe("act", { action, persona: ui.persona, params: {} });
  if (result && result.note) toast(F(result, "note"));
});

$$("[data-until]").forEach((button) =>
  button.addEventListener("click", async () => {
    announce(`${button.textContent.trim()}…`);
    const result = await safe("advance", { until: button.dataset.until });
    if (result) toast(T("page.app.on_click.59c1a9", { until: button.dataset.until, minutes: result.minutes, events: result.events }));
  }),
);

$$("[data-advance]").forEach((button) =>
  button.addEventListener("click", async () => {
    const minutes = Number(button.dataset.advance);
    announce(`${button.textContent.trim()}…`);
    const result = await safe("advance", { minutes });
    if (result) {
      const span =
        minutes >= 1440
          ? T("page.app.on_click.aa2584", { minutes: minutes / 1440 })
          : minutes >= 60
            ? T("page.app.on_click.c8207b", { minutes: minutes / 60 })
            : T("page.app.on_click.fb8b9d", { minutes });
      toast(T("page.app.on_click.dcaf2d", { minutes: span, events: result.events }));
    }
  }),
);

// Real time runs on the server (the duck lives on with the page closed); the page only
// follows along. Faster speeds are driven from here, one step per second. Only the ▶/⏸
// button turns the server's real time on or off: nothing pauses it behind your back.
function syncServer(server) {
  ui.server = server;
  ui.live = !!server.live;
  if (server.lights) {
    // The light at the duck's time (world/lights.py): the 3D view draws its room with it.
    window.duckLights = server.lights;
    window.dispatchEvent(new CustomEvent("duck:lights", { detail: server.lights }));
  }
  const rtOption = $("#speed").querySelector('option[value="rt"]');
  if (rtOption) rtOption.hidden = false;
  if (server.realtime && !ui.realtime) {
    ui.realtime = true;
    ui.rtPending = false;
    $("#speed").value = "rt";
    showPlaying(true);
    clearInterval(ui.timer);
    // Every second: the server shows each move a couple of seconds after it happens, so
    // the page always has it in hand before its moment comes.
    ui.timer = setInterval(async () => {
      if (ui.busy || ui.polling || ui.net.down) return;
      ui.polling = true;
      try {
        await call("state", {});
        ui.pollErrors = 0;
      } catch (error) {
        // The next second tries again (two misses in a row: the page says the server is
        // gone); a server that answers with an error is said once, not hidden.
        pollFailed(error);
      } finally {
        ui.polling = false;
      }
    }, 1000);
  } else if (!server.realtime && ui.realtime) {
    ui.realtime = false;
    if ($("#speed").value === "rt") {
      clearInterval(ui.timer);
      showPlaying(false);
    }
  }
}

function showPlaying(playing) {
  ui.playing = playing;
  I18N.text($("#btnPlay"), playing ? "page.app.showPlaying.ec0c93" : "page.app.showPlaying.330e35");
  $("#btnPlay").setAttribute("aria-pressed", String(playing));
  if (playing) ui.pauseWhy = "";
  // (real time asked for and not yet confirmed counts as real time: the 3D view never
  // treats it as a fast-forward in between)
  window.duckTime = { playing, realtime: !!(ui.realtime || (playing && ui.rtPending)), speed: $("#speed").value };
  window.dispatchEvent(new CustomEvent("duck:time", { detail: window.duckTime }));
  renderBanner();
}

function setPlaying(playing, { catchUp = true } = {}) {
  const realtime = $("#speed").value === "rt";
  clearInterval(ui.timer);
  ui.rtPending = playing && realtime;
  const wasRealtime = ui.realtime;
  // (Leaving real time for a faster speed: the 3D view is told at once, never that a
  // fast-forward is real time — its clock and its 「物理真相」 label follow this.)
  if (!realtime) ui.realtime = false;
  showPlaying(playing);
  if (realtime || wasRealtime) {
    ui.realtime = false; // let the server's answer set it
    const on = playing && realtime;
    safe("realtime", { on, catch_up: catchUp }).then((result) => {
      ui.rtPending = false;
      if (on && result === null) {
        showPlaying(false); // it did not start (the error is on screen)
        return;
      }
      // Real time follows your clock: when the duck's clock was behind, the hours in
      // between were lived first (on the charger). Say so instead of jumping silently.
      const minutes = result && result.caught_up_minutes;
      if (on && minutes) {
        const h = Math.floor(minutes / 60);
        const m = minutes % 60;
        const span = h
          ? T("page.app.setPlaying.55902d", { h, m: m ? T("page.app.setPlaying.e35afc", { m }) : "" })
          : T("page.app.setPlaying.a6c433", { m });
        const said = T("page.app.setPlaying.805f51", { span });
        toast(said);
        announce(said, true);
      }
    });
    if (realtime) return;
  }
  if (playing) {
    ui.timer = setInterval(async () => {
      if (ui.busy || ui.net.down || !ui.playing) return;
      ui.busy = true;
      await safe("advance", { minutes: Number($("#speed").value) }, null, { ff: true, quiet: true });
      ui.busy = false;
    }, 1000);
  }
}

$("#btnPlay").addEventListener("click", () => {
  setPlaying(!ui.playing);
  announcePlay();
});
$("#btnPlace").addEventListener("click", async () => {
  const button = $("#btnPlace");
  announce(`${button.textContent}…`);
  let result;
  if (button.dataset.nest === "in") {
    result = await safe("act", { action: "place_on_nest", persona: ui.persona, params: {}, visit: true });
    if (result && result.deferred) followVisit();
  } else if (button.dataset.nest === "out") {
    // Out of the nest (floor off takes it off the charger), then set down on the floor
    // in front of it, free to walk (floor on).
    result = await safe("set_setting", { key: "floor", value: false });
    if (result !== null) result = await safe("set_setting", { key: "floor", value: true });
  } else {
    result = await safe("set_setting", { key: "floor", value: !button.dataset.floor });
  }
  if (result && result.note) {
    announce(F(result, "note"), true);
    toast(F(result, "note"));
  }
});
$("#speed").addEventListener("change", () => {
  if (ui.playing) setPlaying(true);
  else renderBanner();
});

$("#btnReplay").addEventListener("click", async () => {
  const result = await safe("replay_check", {});
  if (result) {
    toast(
      T("page.app.on_btnReplay", {
        identical: result.identical ? "✅" : "❌",
        note: F(result, "note"),
        events: result.events,
        original_head: result.original_head,
      }),
    );
  }
});

$("#btnExport").addEventListener("click", async () => {
  const result = await safe("export", {});
  if (!result) return;
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `duck-${result.config.name}-${result.config.seed}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 2000);
  toast(T("page.app.on_btnExport", { download: link.download }));
});

$("#btnNew").addEventListener("click", () => $("#dlgNew").showModal());
$("#newCancel").addEventListener("click", () => $("#dlgNew").close());
$("#newForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const params = {
    // (the default name 「小鴨」 is the input's own: a name, the same in every language)
    name: $("#newName").value.trim() || $("#newName").defaultValue,
    seed: Number($("#newSeed").value) || 0,
    tof: $("#newTof").checked,
    pickup_detect: $("#newPickup").checked,
    ear: $("#newEar").checked,
    name_ear: $("#newNameEar").checked,
    motion_validated: $("#newMotion").checked,
    fall_ppm_per_s: $("#newClumsy").checked ? 150000 : 50,
    conservative: $("#newConservative").checked,
    battery_help: $("#newBatteryHelp").checked,
    nest: $("#newNest").checked,
  };
  $("#dlgNew").close();
  ui.cursor = { events: 0, journal: 0 };
  $("#laneTruth").innerHTML = "";
  $("#laneDuck").innerHTML = "";
  $("#cmdOut").innerHTML = "";
  ui.cmdResults = [];
  ui.graphKey = "";
  await safe("new", params, () => toast(T("page.app.cb.2a9ad0", { params_name: params.name })));
  window.dispatchEvent(new CustomEvent("duck:reset"));
  refreshTab(true);
});

/* ---------------- bigger duck (fullscreen, or a full-window overlay where that is missing) ---------------- */

const duckView = $("#duckView");
const expandButton = $("#btnExpand");

function fullscreenElement() {
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}

function expanded() {
  return fullscreenElement() === duckView || duckView.classList.contains("expanded");
}

function syncExpand() {
  const on = expanded();
  I18N.text(expandButton, on ? "page.app.syncExpand.f50f54" : "page.app.syncExpand.9c7125");
  expandButton.setAttribute("aria-pressed", String(on));
  document.body.classList.toggle("duck-expanded", duckView.classList.contains("expanded"));
  // (the view and its control panel are one element: while it is big, the panel floats over it —
  // style.css `.is-big`, shell.js)
  duckView.classList.toggle("is-big", on);
}

async function setExpanded(on) {
  if (!on) {
    if (fullscreenElement()) {
      try {
        await (document.exitFullscreen ? document.exitFullscreen() : document.webkitExitFullscreen());
      } catch (error) {
        /* already left */
      }
    }
    duckView.classList.remove("expanded");
    syncExpand();
    return;
  }
  const request = duckView.requestFullscreen || duckView.webkitRequestFullscreen;
  if (request) {
    try {
      await request.call(duckView);
      syncExpand();
      return;
    } catch (error) {
      /* refused (e.g. iPhone Safari): use the overlay instead */
    }
  }
  duckView.classList.add("expanded");
  syncExpand();
}

expandButton.addEventListener("click", () => setExpanded(!expanded()));
document.addEventListener("fullscreenchange", syncExpand);
document.addEventListener("webkitfullscreenchange", syncExpand);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && duckView.classList.contains("expanded")) setExpanded(false);
});

/* ---------------- text commands ---------------- */

// The chips: the words in the viewer's language; a click fills in the 繁體中文 command (data-zh).
function renderChips() {
  const chip = (key) => `<button type="button" data-zh="${esc(ZH(key))}">${esc(T(key))}</button>`;
  $("#suggest").innerHTML = SUGGESTIONS.map(chip).join("");
  $("#querySuggest").innerHTML = QUERY_EXAMPLES.map(chip).join("");
  // (the duck understands 繁體中文 commands only: said under the boxes in the other languages)
  for (const id of ["#cmdZhNote", "#queryZhNote"]) {
    const note = $(id);
    if (!note) continue;
    note.hidden = I18N.isSource();
    I18N.text(note, "page.app.cmd.zh_only");
  }
}
$("#suggest").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  $("#cmdInput").value = button.dataset.zh || button.textContent;
  $("#cmdInput").focus();
});
$("#cmdForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = $("#cmdInput").value.trim();
  if (!text) {
    toast(T("page.app.on_cmdForm"));
    $("#cmdInput").focus();
    return;
  }
  const result = await safe("command", { text, persona: ui.persona });
  if (!result) return;
  ui.cmdResults = result.results.concat(ui.cmdResults || []).slice(0, 60);
  $("#cmdOut").innerHTML = ui.cmdResults.map(renderCommandResult).join("");
  $("#cmdInput").value = "";
});

function renderCommandResult(item) {
  const r = item.result || {};
  let body = "";
  if (item.kind === "query") {
    const top = r.facts && r.facts.length ? r.facts[0] : null;
    body = `${esc(F(r, "answer"))}${top ? `<br><small>${esc(T("page.app.renderCommandResult.389465", { facts_text: F(top, "text"), recall: F(top, "recall") }))}</small>` : ""}`;
  } else if (item.kind === "why" || item.kind === "status") body = esc(F(r, "text"));
  else if (item.kind === "diary") {
    body =
      (r.diary || [])
        .slice(0, 2)
        .map((d) => esc(T("page.app.renderCommandResult.f0a068", { date: d.date, lines_text: d.lines.map((l) => F(l, "text")).join(" ") })))
        .join("<br>") || esc(T("page.app.renderCommandResult.8645c5"));
  } else if (item.kind === "live") {
    const milestones = FS(r, "new_milestones");
    body = esc(
      T("page.app.renderCommandResult.3d16a3", {
        routine: F(r, "routine"),
        gd_from: r.gd_from,
        gd_to: r.gd_to,
        new_milestones: milestones.length ? T("page.app.renderCommandResult.3be992", { new_milestones: I18N.list(milestones) }) : "",
      }),
    );
  } else if (item.kind === "advance") body = esc(T("page.app.renderCommandResult.040663", { minutes: r.minutes, events: r.events }));
  else if (item.kind === "forget") body = esc(F(r, "note") || "");
  else body = esc(r.note ? F(r, "note") : JSON.stringify(r));
  const PERSONA_KEYS = {
    owner: "page.app.renderCommandResult.636e77",
    partner: "page.app.renderCommandResult.c7e1d2",
    child: "page.app.renderCommandResult.c5a869",
    elder: "page.app.renderCommandResult.d1c48d",
    guest: "page.app.renderCommandResult.0e710e",
  };
  const who = item.persona
    ? T("page.app.renderCommandResult.180616", { persona: PERSONA_KEYS[item.persona] ? T(PERSONA_KEYS[item.persona]) : item.persona })
    : "";
  return `<div class="cmd-item ${item.kind === "unknown" ? "unknown" : ""}"><span class="clause">${esc(item.clause)}</span> <small>${esc(item.kind)}${esc(who)}</small><br>${body}</div>`;
}

/* ---------------- tabs ---------------- */

$$(".tabs button").forEach((button) => button.addEventListener("click", () => setTab(button.dataset.tab)));

function setTab(name) {
  ui.tab = name;
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".tab").forEach((panel) => (panel.hidden = panel.id !== "tab-" + name));
  refreshTab(true);
}

function maybeRefreshTab() {
  const st = ui.state;
  if (!st) return;
  if (ui.tab === "graph") {
    const key = `${st.stats.facts}|${st.voice.phrases.length}|${st.growth.milestones.length}|${st.stats.touches > 0}`;
    if (key !== ui.graphKey && !ui.playing) refreshTab(true);
  }
}

function refreshTab(force) {
  if (ui.tab === "graph" && force) loadGraph();
  else if (ui.tab === "diary") loadDiary();
  else if (ui.tab === "growth") renderGrowth();
  else if (ui.tab === "voice") renderVoice();
  else if (ui.tab === "chain" && force) loadChain();
}

/* ---------------- memory graph ---------------- */

$("#gRefresh").addEventListener("click", () => loadGraph());
$("#gTruth").addEventListener("change", () => loadGraph());
$("#gEpisodes").addEventListener("change", () => loadGraph());

async function loadGraph() {
  const st = ui.state;
  if (st) ui.graphKey = `${st.stats.facts}|${st.voice.phrases.length}|${st.growth.milestones.length}|${st.stats.touches > 0}`;
  const data = await safe("graph", { truth: $("#gTruth").checked, episodes: $("#gEpisodes").checked ? 10 : 0 });
  if (data) drawGraph(data);
}

function drawGraph(data) {
  const svgEl = $("#graph");
  ui.graphData = data;
  $("#gCounts").textContent = T("page.app.drawGraph.3960e9", {
    facts: data.counts.facts,
    phrases: data.counts.phrases,
    events: data.counts.events,
    excluded: data.counts.excluded ? T("page.app.drawGraph.563de3", { excluded: data.counts.excluded }) : "",
  });
  if (typeof d3 === "undefined") {
    svgEl.outerHTML = `<div class="graph-empty">${H("page.app.drawGraph.40a5fd", { nodes_label: I18N.list(data.nodes.map((n) => F(n, "label"))) })}</div>`;
    return;
  }
  const width = svgEl.clientWidth || 600;
  const height = 460;
  const svg = d3.select(svgEl);
  svg.selectAll("*").remove();
  svg.attr("viewBox", `0 0 ${width} ${height}`);
  if (ui.sim) ui.sim.stop();
  const nodes = data.nodes.map((n) => ({ ...n }));
  const links = data.edges.map((e) => ({ ...e }));
  const root = svg.append("g");
  ui.zoom = d3.zoom().scaleExtent([0.2, 3]).on("zoom", (event) => root.attr("transform", event.transform));
  svg.call(ui.zoom);
  const radius = (d) => (d.type === "self" ? 24 : d.type === "person" ? 15 : 5 + (d.strength || 30) / 9);
  const link = root.append("g").selectAll("line").data(links).join("line").attr("class", (d) => "link " + d.type);
  const node = root
    .append("g")
    .selectAll("g")
    .data(nodes)
    .join("g")
    .attr("class", (d) => "node " + d.type + (d.recall === "\u4f11\u7720" ? " dim" : "")) // (「休眠」: the server's word)
    .call(
      d3
        .drag()
        .on("start", (event, d) => {
          if (!event.active) ui.sim.alphaTarget(0.25).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on("end", (event, d) => {
          if (!event.active) ui.sim.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }),
    )
    .on("click", (event, d) => selectNode(d, node));
  node.append("circle").attr("r", radius).attr("style", (d) => `fill: var(${TYPE_COLOR[d.type] || "--muted"})`);
  node.append("title").text((d) => F(d, "detail") || F(d, "label"));
  node
    .append("text")
    .attr("dx", (d) => radius(d) + 3)
    .attr("dy", 4)
    .text((d) => graphLabel(d));
  ui.sim = d3
    .forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance((l) => (l.type === "lineage" ? 46 : l.type === "supports" ? 70 : 85)).strength(0.5))
    .force("charge", d3.forceManyBody().strength(-210))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius((d) => radius(d) + (d.type === "episode" ? 6 : 14)))
    .force("x", d3.forceX(width / 2).strength(0.07))
    .force("y", d3.forceY(height / 2).strength(0.09))
    .on("end", () => fitGraph(svg, root, nodes, width, height))
    .on("tick", () => {
      link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });
}

// A node's label on the graph (short: the first 10 characters, a little more in Latin script).
function graphLabel(d) {
  const label = String(F(d, "label") || "");
  return I18N.isSource() ? label.slice(0, 10) : label.length > 18 ? label.slice(0, 17) + "…" : label;
}

// A language switch: the graph keeps its layout; only its words change.
function rewordGraph() {
  if (!ui.graphData || typeof d3 === "undefined") return;
  const data = ui.graphData;
  $("#gCounts").textContent = T("page.app.drawGraph.3960e9", {
    facts: data.counts.facts,
    phrases: data.counts.phrases,
    events: data.counts.events,
    excluded: data.counts.excluded ? T("page.app.drawGraph.563de3", { excluded: data.counts.excluded }) : "",
  });
  const svg = d3.select("#graph");
  svg.selectAll("g.node title").text((d) => F(d, "detail") || F(d, "label"));
  svg.selectAll("g.node text").text((d) => graphLabel(d));
}

function fitGraph(svg, root, nodes, width, height) {
  if (!nodes.length) return;
  const xs = nodes.map((n) => n.x);
  const ys = nodes.map((n) => n.y);
  const minX = Math.min(...xs) - 60;
  const maxX = Math.max(...xs) + 90;
  const minY = Math.min(...ys) - 40;
  const maxY = Math.max(...ys) + 40;
  const scale = Math.min(1.6, 0.95 / Math.max((maxX - minX) / width, (maxY - minY) / height));
  const tx = width / 2 - scale * (minX + maxX) / 2;
  const ty = height / 2 - scale * (minY + maxY) / 2;
  svg.transition().duration(500).call(ui.zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

async function selectNode(d, nodeSel) {
  nodeSel.classed("sel", (n) => n.id === d.id);
  if (d.type === "fact" || d.type === "milestone") {
    const result = await safe("fact", { fact_id: d.id });
    if (!result || !result.fact) return;
    showFact(result.fact, d, nodeSel);
  } else {
    showNode(d);
  }
}

const NODE_TYPE_KEYS = {
  self: "page.app.selectNode.c26690",
  person: "page.app.selectNode.491277",
  episode: "page.app.selectNode.2b9221",
  phrase: "page.app.selectNode.b69ce4",
  song: "page.app.selectNode.c0a40f",
  truth: "page.app.selectNode.da06aa",
};

function showNode(d) {
  ui.detail = () => showNode(d);
  $("#gDetail").innerHTML = `<h3>${esc(F(d, "label"))}</h3><p>${esc(F(d, "detail") || "")}</p><p class="meta">${esc(NODE_TYPE_KEYS[d.type] ? T(NODE_TYPE_KEYS[d.type]) : d.type)}</p>`;
}

function showFact(f, d, nodeSel) {
  const box = $("#gDetail");
  ui.detail = () => showFact(f, d, nodeSel);
  const canForget = !["trait", "milestone", "quirk"].includes(f.category);
  box.innerHTML = `<h3>${esc(f.predicate ? F(f, "predicate") : F(f, "text"))}</h3>
      <p>${esc(F(f, "text"))}</p>
      <p class="meta">${esc(T("page.app.selectNode.19d9a0", { category_zh: F(f, "category_zh"), recall: F(f, "recall") }))}<span class="strength"><i style="width:${f.strength}%"></i></span>${H("page.app.selectNode.ffb0c0", {
        strength: f.strength,
        half_life_days: f.half_life_days,
        evidence_count: f.evidence_count,
        first_at: f.first_at || "",
        last_at: f.last_at || "",
        generator: f.generator,
      })}</p>
      <ul class="evidence">${f.evidence.map((e) => `<li><span class="eid" data-eid="${esc(e.event_id)}">${esc(e.event_id)}</span><span class="muted">${esc(e.at)}</span><span>${esc(F(e, "text"))}</span></li>`).join("")}</ul>
      ${
        canForget
          ? `<div class="confirm"><button class="btn small" id="btnForget">${esc(T("page.app.selectNode.f656d0"))}</button><span class="muted">${esc(T("page.app.selectNode.053b7c"))}</span></div>`
          : `<p class="hint">${esc(T("page.app.selectNode.72a333"))}</p>`
      }`;
  if (!canForget) return;
  $("#btnForget").addEventListener("click", () => {
    box.querySelector(".confirm").innerHTML = `${esc(T("page.app.selectNode.54796d", { predicate: F(f, "predicate") }))} <button class="btn small accent" id="btnForgetYes">${esc(T("page.app.selectNode.76a8f8"))}</button> <button class="btn small" id="btnForgetNo">${esc(T("page.app.selectNode.4d0b46"))}</button>`;
    $("#btnForgetNo").addEventListener("click", () => selectNode(d, nodeSel));
    $("#btnForgetYes").addEventListener("click", async () => {
      const r = await safe("forget", { fact_id: f.fact_id });
      if (!r) return;
      showForgotten(r);
      loadGraph();
    });
  });
}

function showForgotten(r) {
  const box = $("#gDetail");
  ui.detail = () => showForgotten(r);
  box.innerHTML = `<h3>${esc(T("page.app.selectNode.68e7fd"))}</h3><p>${esc(F(r, "note"))}</p><p class="meta">${esc(T("page.app.selectNode.3532ef"))} <span class="eid" data-eid="${esc(r.receipt)}">${esc(r.receipt)}</span>${esc(
    T("page.app.selectNode.1d7fe2", {
      removed_events: r.removed_events,
      rolled_back: r.rolled_back,
      still_there: T(r.still_there ? "page.app.selectNode.3a26a3" : "page.app.selectNode.5f01d5"),
    }),
  )}</p><button class="btn small" id="btnUndoForget">${esc(T("page.app.selectNode.a4ff3f"))}</button>`;
  $("#btnUndoForget").addEventListener("click", async () => {
    const u = await safe("undo_forget", {});
    if (u) toast(F(u, "note"));
    loadGraph();
  });
}

/* ---------------- distilled memory query ---------------- */

$("#querySuggest").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  const q = button.dataset.zh || button.textContent;
  $("#queryInput").value = q;
  runQuery(q);
});
$("#queryForm").addEventListener("submit", (event) => {
  event.preventDefault();
  runQuery($("#queryInput").value.trim());
});

async function runQuery(q) {
  if (!q) {
    toast(T("page.app.runQuery.22d908"));
    $("#queryInput").focus();
    return;
  }
  const r = await safe("query", { q });
  if (!r) return;
  ui.queryAnswer = r;
  renderQueryAnswer(r);
}

function renderQueryAnswer(r) {
  // (the server's 繁體中文 answer says when it does not know: 「分不出」「不記得」)
  const honest = /\u5206\u4e0d\u51fa|\u4e0d\u8a18\u5f97/.test(r.answer);
  $("#queryOut").innerHTML = `<div class="answer ${honest ? "honest" : ""}">🦆 ${esc(F(r, "answer"))}</div>
    <div class="facts">${r.facts
      .map(
        (f) => `<div class="fact"><div class="top"><span class="tag learning">${esc(F(f, "category_zh"))}</span><span>${esc(F(f, "recall"))}</span><span class="strength"><i style="width:${f.strength}%"></i></span><span>${f.strength}%</span><span>${esc(T("page.app.runQuery.135614", { evidence_count: f.evidence_count }))}</span></div>
        <div class="txt">${esc(F(f, "text"))}</div>
        <details><summary>${esc(T("page.app.runQuery.f2ed3c"))}</summary><ul class="evidence">${f.evidence.map((e) => `<li><span class="eid" data-eid="${esc(e.event_id)}">${esc(e.event_id)}</span><span class="muted">${esc(e.at)}</span><span>${esc(F(e, "text"))}</span></li>`).join("")}</ul></details></div>`,
      )
      .join("")}</div>
    ${r.diary.length ? `<h3 style="margin-top:12px">${esc(T("page.app.runQuery.61f26e"))}</h3>${r.diary.map((d) => `<p>${esc(T("page.app.runQuery.1604cc", { date: d.date, gd: d.gd, text: F(d, "text") }))}</p>`).join("")}` : ""}
    ${
      r.phrases.length
        ? `<h3 style="margin-top:12px">${esc(T("page.app.runQuery.f53fcf"))}</h3><p>${esc(
            I18N.list(r.phrases.map((p) => T("page.app.runQuery.def027", { text: F(p, "text"), gen: p.gen, promoted: p.promoted ? T("page.app.runQuery.a7b2bc") : "" }))),
          )}</p>`
        : ""
    }
    <p class="hint">${esc(F(r, "honesty"))}</p>`;
}

/* ---------------- diary ---------------- */

async function loadDiary() {
  const r = await call("diary", { limit: 40 }).catch(() => null);
  if (!r) return;
  ui.diary = r;
  renderDiary(r);
}

function renderDiary(r) {
  $("#diaryOut").innerHTML = r.diary.length
    ? r.diary
        .map(
          (d) => `<div class="diary-day"><h3>${esc(d.date)} <span class="gauges">${esc(
            T("page.app.loadDiary.3a04ba", { gd: d.gd, positive: d.positive, negative: d.negative ? T("page.app.loadDiary.1df714", { negative: d.negative }) : "" }),
          )}</span></h3>${d.lines
            .map((l) => `<p class="${esc(l.tone)}">${esc(F(l, "text"))} ${l.cites.slice(0, 3).map((id) => `<span class="eid" data-eid="${esc(id)}">${esc(id)}</span>`).join("")}</p>`)
            .join("")}</div>`,
        )
        .join("")
    : `<p class="muted">${esc(T("page.app.loadDiary.2a9cae"))}</p>`;
}

/* ---------------- growth ---------------- */

function renderGrowth() {
  const st = ui.state;
  if (!st) return;
  const g = st.growth;
  const gd = st.clock.gd;
  // Each duck grows up at its own pace: its own stage days (P1), else the usual ones.
  const days = Object.assign({ duckling: 10, juvenile: 45, adult: 180 }, g.stage_days || {});
  const segs = [
    [T("page.app.renderGrowth.924ae1"), 0, days.duckling, "var(--duck)"],
    [T("page.app.renderGrowth.52e222"), days.duckling, days.juvenile, "var(--orange)"],
    [T("page.app.renderGrowth.4fd20d"), days.juvenile, days.adult, "var(--reed)"],
    [T("page.app.renderGrowth.5801b4"), days.adult, Math.max(365, days.adult + 1), "var(--teal)"],
  ];
  const widths = [14, 24, 36, 26];
  let left = 0;
  let marker = 0;
  const segHtml = segs
    .map(([name, from, to, color], i) => {
      const html = `<div class="seg" style="left:${left}%;width:${widths[i] - 1}%;background:${color}"><span>${esc(name)} ${from}</span></div>`;
      if (gd >= from && (gd < to || i === segs.length - 1)) marker = left + (widths[i] * Math.min(1, (gd - from) / (to - from)));
      left += widths[i];
      return html;
    })
    .join("");
  const compass = g.compass
    ? `<div class="compass">${Object.values(g.compass)
        .map((c) => `<div class="crow"><span>${esc(F(c, "zh"))}</span><div class="bar"><i style="width:${c.share}%"></i></div><span class="mono">${c.share}%</span></div>`)
        .join("")}</div><p class="hint">${esc(
        T("page.app.renderGrowth.231d24", { care_total: g.care_total, care_total2: g.care_total < 30 ? T("page.app.renderGrowth.d6ddcb") : "" }),
      )}</p>`
    : `<p class="muted">${esc(T("page.app.renderGrowth.87b3a2"))}</p>`;
  const word = (option, key) => `<button class="btn small" data-choose="first_word:${option}">${esc(T(key))}</button>`;
  const fw = !st.voice.first_word
    ? `<div class="sub"><h3>${esc(T("page.app.renderGrowth.453d1f"))}</h3><div class="row" style="display:flex;gap:6px;flex-wrap:wrap">${word("peep", "page.app.renderGrowth.2ec799")}${word("hm", "page.app.renderGrowth.f9fb73")}${word("mm", "page.app.renderGrowth.3fb5b3")}${word("hi", "page.app.renderGrowth.c78146")}${word("yay", "page.app.renderGrowth.d6d10e")}</div><p class="hint">${esc(T("page.app.renderGrowth.b57618"))}</p></div>`
    : "";
  const branch = F(g, "branch");
  const runnerUp = F(g, "runner_up");
  $("#growthOut").innerHTML = `
    <div class="track">${segHtml}<div class="marker" style="left:${marker}%" title="${esc(T("page.app.renderGrowth.now"))}"></div></div>
    <p class="muted" style="margin-top:-6px">${esc(T("page.app.renderGrowth.510b69", { hint: F(g, "hint") }))}</p>
    <div class="grid2">
      <div class="sub"><h3>${esc(T("page.app.renderGrowth.9b6eab"))}</h3><ul class="milestones">${g.milestones
        .map(
          (m) =>
            `<li class="${m.status}"><span class="ic">${m.status === "debuted" ? "✓" : "✨"}</span><span><b>${esc(F(m, "title"))}</b>${m.gd != null ? `<small>${esc(T("page.app.renderGrowth.71cf2c", { gd: m.gd }))}</small>` : ""}<br>${m.status === "debuted" ? esc(F(m, "text")) : esc(T("page.app.renderGrowth.47e908"))}</span></li>`,
        )
        .join("")}</ul></div>
      <div class="sub"><h3>${esc(T("page.app.renderGrowth.fb162f"))}</h3>${g.pace_note ? `<p class="hint">${esc(F(g, "pace_note"))}</p>` : ""}${g.branch ? `<p><b>${esc(branch)}</b>${g.adult ? esc(T("page.app.renderGrowth.fc6537", { title: g.title ? F(g, "title") : F(g, "adult") })) : ""}</p>${g.can_rewind ? `<p class="hint">${esc(T("page.app.renderGrowth.995ff2", { runner_up: runnerUp }))}</p><button class="btn small" data-choose="branch:${esc(g.runner_up)}">${esc(T("page.app.renderGrowth.17d35e", { runner_up: runnerUp }))}</button>` : ""}` : ""}${compass}</div>
      ${fw}
      <div class="sub"><h3>${esc(T("page.app.renderGrowth.bac0d1"))}</h3>${st.learning.prefs.length ? `<ul class="milestones">${st.learning.prefs.map((p) => `<li><span class="ic">♥</span><span>${esc(F(p, "zh"))} ${p.value}%</span></li>`).join("")}</ul>` : `<p class="muted">${esc(T("page.app.renderGrowth.fa3587"))}</p>`}</div>
      ${renderMoments(st.moments)}
      <div class="sub"><h3>${esc(T("page.app.renderGrowth.024405"))}</h3>${st.album.length ? `<div class="albums">${FS(st, "album").map((a) => `<span>${esc(a)}</span>`).join("")}</div>` : `<p class="muted">${esc(T("page.app.renderGrowth.0e9fa4"))}</p>`}</div>
      <div class="sub"><h3>${esc(T("page.app.renderGrowth.cc645e"))}</h3><dl class="kv"><dt>${esc(T("page.app.renderGrowth.275c2e"))}</dt><dd>${st.budgets.bids_today} / 10</dd><dt>${esc(T("page.app.renderGrowth.0cea52"))}</dt><dd>${st.budgets.bids_hour} / ${st.budgets.bids_cap_hour}</dd><dt>${esc(T("page.app.renderGrowth.927274"))}</dt><dd>${st.budgets.self_phrases_hour} / 10</dd><dt>${esc(T("page.app.renderGrowth.68d803"))}</dt><dd>${esc(T(st.budgets.bids_stopped ? "page.app.renderGrowth.17656c" : "page.app.renderGrowth.f78d03"))}</dd><dt>${esc(T("page.app.renderGrowth.9997be"))}</dt><dd>${esc(T("page.app.renderGrowth.cd0b3b", { lived_hours: st.clock.lived_hours, boot_seq: st.clock.boot_seq }))}</dd></dl><p class="hint">${esc(T("page.app.renderGrowth.d19fe9"))}</p></div>
    </div>`;
}

function renderMoments(m) {
  if (!m) return "";
  const today = m.today || {};
  const form = m.form_zh ? T("page.app.renderMoments.4e0485", { form_zh: F(m, "form_zh") }) : "";
  const wits = m.wits != null ? T("page.app.renderMoments.395a80", { wits: m.wits, form }) : "";
  const gags = (m.gags || []).length
    ? `<p>${esc(T("page.app.renderMoments.7026a6", { gags: I18N.list(FS(m, "gags")) }))}</p>`
    : `<p class="hint">${esc(T("page.app.renderMoments.6bb844"))}</p>`;
  return `<div class="sub"><h3>${esc(T("page.app.renderMoments.7f5553"))}</h3><p>${esc(
    T("page.app.renderMoments.22c168", { clever: today.clever || 0, goofy: today.goofy || 0, wits: wits ? T("page.app.renderMoments.c4c970", { wits }) : "" }),
  )}</p>${gags}<p class="hint">${esc(T("page.app.renderMoments.43d908", { album: m.album || 0 }))}</p></div>`;
}

/* ---------------- voice lineage ---------------- */

function renderVoice() {
  const st = ui.state;
  if (!st) return;
  const v = st.voice;
  const gens = {};
  v.phrases.forEach((p) => (gens[p.gen] = gens[p.gen] || []).push(p));
  const cols = Object.keys(gens)
    .sort((a, b) => a - b)
    .map(
      (gen) => `<div class="gen"><h3>${esc(T("page.app.renderVoice.37c962", { gen: Number(gen) }))}</h3>${gens[gen]
        .map(
          (p) =>
            `<div class="phrase ${p.promoted ? "promoted" : ""}">${esc(F(p, "text"))}<small>${esc(
              T("page.app.renderVoice.95175c", {
                promoted: p.promoted ? T("page.app.renderVoice.6aeb8c", { praise: p.praise }) : T("page.app.renderVoice.e04474"),
                parents: p.parents.length ? T("page.app.renderVoice.8d4c1b", { parents: p.parents.join(",") }) : "",
                born_gd: p.born_gd,
              }),
            )}</small></div>`,
        )
        .join("")}</div>`,
    )
    .join("");
  const promotions = FS(v, "promotions");
  $("#voiceOut").innerHTML = `
    <p>${esc(T("page.app.renderVoice.53ea46"))}<b>${esc(v.first_word ? F(v, "first_word") : T("page.app.renderVoice.938516"))}</b></p>
    ${v.song ? `<div class="song">${esc(T("page.app.renderVoice.2acd9f", { version: v.song.version, song_text: F(v.song, "text") }))}</div>` : `<p class="muted">${esc(T("page.app.renderVoice.693ea2"))}</p>`}
    ${v.molt ? `<div class="molt"><b>${esc(T("page.app.renderVoice.9e558a"))}</b><span>${esc(T("page.app.renderVoice.54650a", { molt_text: F(v.molt, "text") }))}</span><div style="display:flex;gap:6px"><button class="btn small" data-choose="molt:keep">${esc(T("page.app.renderVoice.ff9bf7"))}</button><button class="btn small accent" data-choose="molt:change">${esc(T("page.app.renderVoice.968a0d"))}</button></div></div>` : ""}
    <h3 style="margin:12px 0 6px">${esc(T("page.app.renderVoice.46215c"))}</h3>
    <p>${esc(promotions.length ? I18N.list(promotions) : T("page.app.renderVoice.6a4be5"))}</p>
    <h3 style="margin:12px 0 6px">${esc(T("page.app.renderVoice.fcc7ff"))}</h3>
    ${cols ? `<div class="lineage">${cols}</div>` : `<p class="muted">${esc(T("page.app.renderVoice.c87fb5"))}</p>`}
    <p class="hint">${esc(T("page.app.renderVoice.144e59"))}</p>`;
}

document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-choose]");
  if (!button) return;
  const [kind, option] = button.dataset.choose.split(":");
  // (the growth tab's 「改成…」 button carries the server's 繁體中文 branch name)
  const branchIds = { "\u611b\u5531\u6b4c\u578b": "songbird", "\u597d\u5947\u578b": "explorer", "\u4f9d\u504e\u578b": "snuggler", "\u60a0\u54c9\u578b": "dreamer" };
  const result = await safe("choose", { kind, option: branchIds[option] || option });
  if (result && result.note) toast(F(result, "note"));
});

/* ---------------- event chain ---------------- */

async function loadChain() {
  const r = await call("events", { tail: true, limit: 150 }).catch(() => null);
  if (!r) return;
  ui.chain = r;
  renderChain(r);
}

function renderChain(r) {
  $("#chainHead").innerHTML = `<span>${esc(T("page.app.loadChain.faa63a", { total: r.total }))}</span><span>${I18N.tHtml("page.app.loadChain.c4575f", { head: r.head.slice(0, 16) })}</span><span>${esc(
    T(r.chain_ok ? "page.app.loadChain.86a90c" : "page.app.loadChain.e1ff46"),
  )}</span><button class="btn small" id="btnChainReplay">${esc(T("page.app.loadChain.e9e57b"))}</button>`;
  $("#btnChainReplay").addEventListener("click", () => $("#btnReplay").click());
  $("#chainBody").innerHTML = r.events
    .slice()
    .reverse()
    .map((e) => `<tr><td class="k"><span class="eid" data-eid="${esc(e.event_id)}">${esc(e.event_id)}</span></td><td class="k">${esc(e.occurred_at.slice(5, 16).replace("T", " "))}</td><td class="k">${esc(e.kind)}</td><td>${esc(F(e, "text"))}</td></tr>`)
    .join("");
}

/* ---------------- toast and boot ---------------- */

// Long enough to read: 5 s plus a little per character (a long explanation stays up to 15 s).
function toast(text) {
  const el = $("#toast");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(ui.toastTimer);
  ui.toastTimer = setTimeout(() => el.classList.remove("show"), Math.min(15000, 5000 + String(text).length * 80));
}

/* ---------------- 2D fallback: every move in order, and say why there is no 3D ---------------- */

ui.moves2d = [];
function play2d() {
  if (ui.timer2d || !ui.moves2d.length) return;
  const move = ui.moves2d.shift();
  animateVerb(move.clip, move.sound, move.calls);
  ui.timer2d = setTimeout(() => {
    ui.timer2d = 0;
    play2d();
  }, 3000);
}
// The 2D duck's calls (the 3D view plays its own while it is shown): in the duck's own voice.
function voice2d(calls, holdMs = 0) {
  if (!calls || !calls.length || !window.duckVoice || document.body.classList.contains("has3d")) return false;
  return window.duckVoice.call(calls, { holdMs });
}
window.addEventListener("duck:feed", (event) => {
  if (document.body.classList.contains("has3d")) return;
  for (const item of event.detail.events || []) {
    if (item.settle || (event.detail.first && !(item.due_ms > Date.now()))) continue;
    const at = item.due_ms ? Math.max(0, item.due_ms - Date.now()) : 0;
    if (item.motion && item.motion.clip) {
      setTimeout(() => {
        ui.moves2d.push({ clip: item.motion.clip, sound: item.bubble, calls: item.calls });
        while (ui.moves2d.length > 6) ui.moves2d.shift();
        play2d();
      }, at);
    } else if (item.calls && item.calls.length && !(event.detail.stale && !(item.due_ms > Date.now()))) {
      // A greeting, a bid, a babbled phrase: no move carries it; its sound at its moment.
      setTimeout(() => voice2d(item.calls), at);
    }
  }
});
function watch3dLoad() {
  setTimeout(() => {
    if (document.body.classList.contains("has3d")) return;
    const note = $("#duck3dNote");
    // (still the page's own 「正在載入…」 note: the 3D view never replaced it)
    if (note && note.dataset.i18n === "page.html.duck3dNote.text") I18N.text(note, "page.app.three_failed");
  }, 15000);
}

// A paused page still follows the duck (another tab or phone may have changed it).
function pollPaused() {
  if (!ui.playing && !ui.busy && !ui.busyBanner && !document.hidden && !ui.net.down) {
    call("state", {})
      .then(() => (ui.pollErrors = 0))
      .catch(pollFailed);
  }
}

// A poll that keeps failing while the server is there (it answers with an error): say it
// once, instead of a page that silently stops following the duck.
function pollFailed(error) {
  if (ui.net.down) return;
  ui.pollErrors = (ui.pollErrors || 0) + 1;
  if (ui.pollErrors === 3) toast(T("page.app.pollFailed", { message: error.message }));
}

// (the first answer: in boot())

/* ---------------- live physics: the duck-eye window and 「物理不可信」 (3D and 2D alike) ---------------- */
// With live physics the 3D view asks /api/body many times a second and passes every answer
// on (duck:body); without the 3D view (the 2D duck) the page asks now and then itself. The
// duck's-eye picture is fetched every 2 s only while its window is open, on screen, the tab
// is visible and time is not fast-forwarded (the physics body only lives real time).

const EYE_EVERY_MS = 2000;
const EYE_NARROW = window.matchMedia ? window.matchMedia("(max-width: 760px)") : null;
const LIGHT_LEVEL_CLASS = { bright: "light-bright", dim: "light-dim", dark: "light-dark", black: "light-black" };
const eyeUi = { body: null, bodyAt: 0, asking: false, open: true, inView: true, timer: 0, fetching: false, tag: "", url: "", why: "", trustKey: "", noteKey: "" };

function eyeFast() {
  return !!(ui.playing && !ui.realtime && !ui.rtPending);
}

function eyeWanted() {
  const body = eyeUi.body;
  const eye = body && body.enabled ? body.eye : null;
  // The world's camera (phase 2) gives a frame only while the window asks (F4: asking is what
  // opens it); phase 0's own render only once it has a picture (its note says why not yet).
  const ready = !!(eye && !eye.error && (eye.window || eye.frames > 0));
  return ready && eyeUi.open && eyeUi.inView && !document.hidden && !eyeFast() && !ui.net.down;
}

function eyeSchedule() {
  const want = eyeWanted();
  if (want && !eyeUi.timer) {
    eyeFetch();
    eyeUi.timer = setInterval(eyeFetch, EYE_EVERY_MS);
  } else if (!want && eyeUi.timer) {
    clearInterval(eyeUi.timer);
    eyeUi.timer = 0;
  }
}

async function eyeFetch() {
  if (eyeUi.fetching || !eyeWanted()) return;
  eyeUi.fetching = true;
  try {
    const response = await fetch("/api/eye.jpg", { cache: "no-store", headers: eyeUi.tag ? { "If-None-Match": eyeUi.tag } : {} });
    if (response.status === 304) return; // (no newer picture yet)
    if (response.status === 202) {
      // (the window just opened: the world's camera is taking the first picture)
      eyeUi.why = eyeWhy(await response.text());
      renderEye(eyeUi.body);
      return;
    }
    if (!response.ok) {
      eyeUi.why = response.status === 404 ? eyeWhy(await response.text()) : T("page.app.eyeFetch");
      renderEye(eyeUi.body);
      return;
    }
    eyeUi.why = "";
    eyeUi.tag = response.headers.get("ETag") || "";
    const url = URL.createObjectURL(await response.blob());
    const img = $("#eyeImg");
    const old = eyeUi.url;
    eyeUi.url = url;
    img.onload = () => {
      if (old) URL.revokeObjectURL(old);
    };
    img.src = url;
  } catch (error) {
    /* the server is gone: the banner says so, and the last picture stays */
  } finally {
    eyeUi.fetching = false;
  }
}

// Why there is no duck's-eye picture: the server's words (a JSON {text, _i18n} answer from a
// server with the i18n layer, plain 繁體中文 words from an older one).
function eyeWhy(body) {
  try {
    const said = JSON.parse(body);
    if (said && typeof said.text === "string") return String(F(said, "text")).slice(0, 160);
  } catch (error) {
    /* plain words */
  }
  return String(body).slice(0, 160);
}

function renderEye(info) {
  const box = $("#eyeWindow");
  const eye = info && info.enabled ? info.eye : null;
  box.hidden = !eye;
  if (!eye) return;
  const chip = $("#eyeLight");
  // (a phone's narrow window: the same words, shorter)
  const narrow = !!(EYE_NARROW && EYE_NARROW.matches);
  // (right after the light changed the camera is still adjusting: the picture alone decided)
  const adapting = eye.adapting ? T(narrow ? "page.app.renderEye.79cd00" : "page.app.renderEye.9cc01b") : "";
  const measured = eye.level_zh
    ? T(narrow ? "page.app.renderEye.d4a2a2" : "page.app.renderEye.2a1830", { level_zh: F(eye, "level_zh"), adapting })
    : "";
  const light = eye.label ? `${F(eye, "label")}${measured}` : T("page.app.renderEye.e567ae");
  if (chip.textContent !== light) chip.textContent = light;
  const level = `chip ${LIGHT_LEVEL_CLASS[eye.level] || ""}`.trim();
  if (chip.className !== level) chip.className = level;
  let note;
  let routine = false; // (just how old the picture is: a phone leaves it out)
  if (eye.error) note = F(eye, "error");
  else if (ui.net.down) note = T("page.app.renderEye.3561dc");
  else if (eyeFast()) note = T("page.app.renderEye.410644");
  else if (eye.frozen && eye.age_s != null) note = T("page.app.renderEye.1e415b", { age_s: Math.max(0, Math.round(eye.age_s)) });
  else if (eye.age_s != null) {
    note = T("page.app.renderEye.555e95", { age_s: Math.max(0, Math.round(eye.age_s)) });
    routine = true;
  } else note = eyeUi.why || F(eye, "note") || T("page.app.renderEye.54ee10");
  if (note !== eyeUi.noteKey) {
    eyeUi.noteKey = note;
    $("#eyeNote").textContent = note;
    $("#eyeNote").classList.toggle("routine", routine);
  }
  $("#eyeNote").title = (eye.error && eye.error_detail) || ""; // (the program's own words)
}

function renderTrust(info) {
  const badge = $("#trustBadge");
  const live = !!(info && info.enabled && !ui.net.down);
  // A body that broke (robotd or the body server stopped) is the plainest reason of all; the
  // monitor's own reasons follow for a body that was running (it had frames).
  const broken = live && info.error ? F(info, "error_zh") || T("page.app.renderTrust.521db0") : "";
  const monitor = live && (info.available || (broken && info.frame)) ? info.monitor || (info.frame && info.frame.monitor) : null;
  const doubted = !!(monitor && monitor.trust === "suspect");
  badge.hidden = !(broken || doubted);
  if (badge.hidden) {
    eyeUi.trustKey = "";
    return;
  }
  const said = doubted && monitor.why_zh ? FS(monitor, "why_zh") : [];
  const reasons = broken ? [broken, ...said] : said.length ? said : [T("page.app.renderTrust.55e19c")];
  const key = reasons.join("\n");
  if (key === eyeUi.trustKey) return;
  eyeUi.trustKey = key;
  $("#trustWhy").replaceChildren(...reasons.map((text) => Object.assign(document.createElement("li"), { textContent: text })));
}

// Without the 3D view the 2D duck plays Duck Core's plan, never the physics body: with live
// physics the label says so, in the 3D view's words (truth.js PLAN_TITLE) with the reason.
const PLAN_TITLE = "page.truth.PLAN_TITLE"; // (truth.js's own words: the 3D view's label)
const FLAT_DETAIL = "page.app.FLAT_DETAIL";
function renderMode2d(info) {
  const el = $("#stageMode");
  if (!el || document.body.classList.contains("has3d")) return;
  const want = !!(info && info.enabled);
  if (!want) {
    if (el.dataset.flat) {
      el.hidden = true;
      delete el.dataset.flat;
    }
    return;
  }
  if (el.dataset.flat && !el.hidden) return;
  const title = Object.assign(document.createElement("b"), { className: "chip plan" });
  const detail = document.createElement("span");
  I18N.text(title, PLAN_TITLE);
  I18N.text(detail, FLAT_DETAIL);
  el.replaceChildren(title, detail);
  el.dataset.mode = "plan";
  el.dataset.flat = "1";
  el.hidden = false;
}

function takeBody(info) {
  eyeUi.body = info;
  eyeUi.bodyAt = Date.now();
  renderTrust(info);
  renderEye(info);
  renderMode2d(info);
  renderWorld(info);
  eyeSchedule();
}

/* ---------------- the physical home (phase 2): its line, the world battery, the toys ---------------- */

const HAND_DO_ZH = {
  pickup: "page.app.HAND_DO_ZH.pickup",
  place: "page.app.HAND_DO_ZH.place",
  upright: "page.app.HAND_DO_ZH.upright",
  touch: "page.app.HAND_DO_ZH.touch",
  hold_head: "page.app.HAND_DO_ZH.hold_head",
  near: "page.app.HAND_DO_ZH.near",
  cancel: "page.app.HAND_DO_ZH.cancel",
};
const PERSON_STATE_ZH = {
  walking: "page.app.PERSON_STATE_ZH.walking",
  waiting: "page.app.PERSON_STATE_ZH.waiting",
  standing: "page.app.PERSON_STATE_ZH.standing",
  stepping_back: "page.app.PERSON_STATE_ZH.stepping_back",
  leaving: "page.app.PERSON_STATE_ZH.leaving",
  away: "page.app.PERSON_STATE_ZH.away",
};
const worldUi = { key: "", things: "", done: 0, battery: "", fingerprint: "", dockApart: false };

// The home's own toys in card 09's chooser: from /api/body's home_json (2D) or from the home
// the 3D view drew (duck:home; the view fetches /api/home itself, so /api/body never sends it).
function takeHome(home) {
  const layout = home && home.layout;
  if (layout) worldUi.home = home; // (a language switch writes the toys' names again)
  if (home && home.fingerprint) worldUi.fingerprint = home.fingerprint;
  // A charging dock against a wall that is not where Duck Core's own origin (its believed nest)
  // is: the default home keeps Duck Core's frame, so the two differ until phase 5.
  const dock = layout && layout.nest && layout.nest.kind === "dock" ? layout.nest : null;
  const frame = (layout && layout.frame) || { x_mm: 0, y_mm: 0 };
  worldUi.dockApart = !!(dock && dock.seat && Math.hypot(dock.seat.x_mm - frame.x_mm, dock.seat.y_mm - frame.y_mm) > 300);
  if (!layout || !layout.things) return;
  const key = layout.things.map((t) => t.id).join(",");
  if (key + "|" + I18N.lang === worldUi.things) return;
  worldUi.things = key + "|" + I18N.lang;
  const select = $("#worldThing");
  const keep = select.value;
  select.replaceChildren(...layout.things.map((t) => Object.assign(document.createElement("option"), { value: t.id, textContent: F(t, "zh") || t.id })));
  if (layout.things.some((t) => t.id === keep)) select.value = keep;
}
window.addEventListener("duck:home", (event) => takeHome(event.detail));

function renderWorld(info) {
  const world = info && info.enabled ? info.world : null;
  const line = $("#worldLine");
  // Card 09 works only with live physics: without it its buttons stay greyed out (the card's
  // hint says why) instead of each click ending in a refusal.
  const physics = !!(info && info.enabled);
  for (const button of document.querySelectorAll("#worldTools [data-world]")) {
    if (button.disabled !== !physics) {
      button.disabled = !physics;
      button.title = physics ? "" : T("page.app.renderWorld.75c8b7");
    }
  }
  // The toys to choose from (the home's own things: /api/body carries the home when it changes).
  if (info && info.home_json) takeHome(info.home_json);
  const box = $("#worldBattery");
  const battery = world && world.battery;
  box.hidden = !battery;
  if (battery) {
    const pct = Math.max(0, Math.min(100, Number(battery.percent) || 0));
    const said = `${pct}%${battery.charging ? T("page.app.renderWorld.d33e82") : ""}`;
    if (said !== worldUi.battery) {
      worldUi.battery = said;
      $("#worldBatteryBar").style.width = `${pct}%`;
      $("#worldBatteryVal").textContent = said;
    }
  }
  if (!world) {
    line.hidden = true;
    return;
  }
  const parts = [];
  const home = world.home;
  // (the home's own name, before its description: 「預設的家」)
  const where = String((info.home && (info.home.zh ? F(info.home, "zh") : info.home.name)) || world.asked || "").split(/[：:]/)[0];
  parts.push(
    world.physical
      ? `<b>${esc(T("page.app.renderWorld.9d95a1", { where }))}</b>${esc(T("page.app.renderWorld.bb323b"))}`
      : `<b>${esc(T("page.app.renderWorld.15d57f", { asked: T(world.asked ? "page.app.renderWorld.a76e8c" : "page.app.renderWorld.a74967") }))}</b>${esc(T("page.app.renderWorld.5a7d5f"))}`,
  );
  const nest = world.nest;
  if (nest && nest.present) {
    // (the physical contact: the dock's, or phase 2's open nest's for its variants)
    const what = T(nest.kind === "dock" ? "page.app.renderWorld.e7f39c" : "page.app.renderWorld.53ad35");
    parts.push(
      nest.contact
        ? `<span class="contact-on">${esc(T("page.app.renderWorld.7d01d1", { what, charging: nest.charging ? T("page.app.renderWorld.e1776c") : "" }))}</span>`
        : esc(T("page.app.renderWorld.56ffd4", { what })),
    );
    // (Duck Core's belief about its nest; the neural-duck prototype's page does not show Duck Core's plan)
    if (nest.kind === "dock" && worldUi.dockApart && !document.body.classList.contains("neuro-mode")) parts.push(esc(T("page.app.renderWorld.dc65d2")));
  }
  const hand = world.hand;
  if (hand && hand.shown) {
    const doing = HAND_DO_ZH[hand.do] ? T(HAND_DO_ZH[hand.do]) : T(hand.holding ? "page.app.renderWorld.e3fa93" : "page.app.renderWorld.90661c");
    parts.push(esc(T("page.app.renderWorld.797a77", { do: doing, holding: hand.holding ? T("page.app.renderWorld.062116") : "" })));
  }
  for (const person of world.people || []) {
    const away = person.clearance_mm != null ? T("page.app.renderWorld.bd061c", { clearance_mm: person.clearance_mm / 1000 }) : "";
    const state = PERSON_STATE_ZH[person.state] ? T(PERSON_STATE_ZH[person.state]) : person.state || "";
    parts.push(esc(T("page.app.renderWorld.89725a", { person_zh: person.zh ? F(person, "zh") : person.who, person: state, away })));
  }
  const placed = world.placed;
  if (placed && placed.ok === false && placed.why_zh && placed.why_zh.length) parts.push(esc(FS(placed, "why_zh")[0]));
  const text = parts.join(esc(T("fmt.sep.dot")));
  if (text !== worldUi.key) {
    worldUi.key = text;
    line.innerHTML = text;
  }
  line.hidden = false;
  // A family visit's touch that reached the duck: said once.
  const done = (world.family && world.family.done) || [];
  for (const item of done) {
    if (item.id <= worldUi.done) continue;
    worldUi.done = item.id;
    if (item.note && Date.now() - item.at_ms < 15000) toast(T("page.app.renderWorld.70606d", { note: F(item, "note") }));
  }
}

// While a family member walks over (a few seconds), follow the duck even with time paused.
function followVisit() {
  let left = 12;
  const tick = () => {
    if (left-- <= 0 || ui.realtime) return;
    if (!ui.busy && !ui.net.down) call("state", {}).catch(() => {});
    setTimeout(tick, 1000);
  };
  setTimeout(tick, 1000);
}

window.addEventListener("duck:body", (event) => {
  if (ui.ready) takeBody(event.detail);
});

async function askBody() {
  const live = !!(eyeUi.body && eyeUi.body.enabled);
  if (eyeUi.asking || document.hidden || ui.net.down || Date.now() - eyeUi.bodyAt < (live ? 1500 : 5000)) return;
  eyeUi.asking = true;
  try {
    // (with the home the page has: the answer carries the home when it is another one)
    takeBody(await fetch(`/api/body?home=${encodeURIComponent(worldUi.fingerprint)}`, { cache: "no-store" }).then((r) => r.json()));
  } catch (error) {
    /* the heartbeat says whether the server is there */
  } finally {
    eyeUi.asking = false;
  }
}
function setupEye() {
  // Open unless someone closed it here before; on a phone it starts closed (it would cover
  // the duck), as does the list of reasons under 「物理不可信」.
  const narrow = !!(window.matchMedia && window.matchMedia("(max-width: 760px)").matches);
  let saved = null;
  try {
    saved = localStorage.getItem("duckEyeOpen");
  } catch (error) {
    saved = null;
  }
  eyeUi.open = saved != null ? saved === "1" : !narrow;
  $("#trustBadge").open = !narrow;
  const toggle = $("#btnEye");
  const sync = () => {
    $("#eyeBody").hidden = !eyeUi.open;
    $("#eyeWindow").classList.toggle("closed", !eyeUi.open);
    I18N.text($("#eyeTitle"), eyeUi.open ? "page.app.sync.2868fc" : "page.app.sync.f0050c");
    toggle.setAttribute("aria-expanded", String(eyeUi.open));
  };
  sync();
  toggle.addEventListener("click", () => {
    eyeUi.open = !eyeUi.open;
    try {
      localStorage.setItem("duckEyeOpen", eyeUi.open ? "1" : "0");
    } catch (error) {
      /* remembered for this visit only */
    }
    sync();
    renderEye(eyeUi.body);
    eyeSchedule();
  });
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((entries) => {
      eyeUi.inView = entries[entries.length - 1].isIntersecting;
      eyeSchedule();
    }).observe($("#eyeWindow"));
  }
  document.addEventListener("visibilitychange", eyeSchedule);
}

/* ---------------- ticker ---------------- */
// The status band scrolls like a marquee: a second, hidden copy of the live chips follows
// the first so the loop is seamless. The copy has no ids and mirrors every change.
(function setupTicker() {
  const track = document.getElementById("tickerTrack");
  if (!track || !track.parentElement) return;
  const copy = document.createElement("div");
  copy.className = "ticker-track";
  copy.setAttribute("aria-hidden", "true");
  track.parentElement.appendChild(copy);
  const sync = () => {
    const clone = track.cloneNode(true);
    clone.querySelectorAll("[id]").forEach((el) => el.removeAttribute("id"));
    copy.innerHTML = clone.innerHTML;
  };
  new MutationObserver(sync).observe(track, { subtree: true, childList: true, characterData: true, attributes: true });
  sync();
})();

/* ---------------- start (once the words are loaded) and language switches ---------------- */

// A language switch re-renders everything drawn from the last answers; elements written with
// I18N.text / data-i18n re-render by themselves (static/i18n.js).
function rerenderAll() {
  renderChips();
  if (ui.state) renderState();
  else renderBanner();
  if (ui.net.down) renderNet();
  checkServer(ui.server);
  rerenderFeed();
  if (ui.cmdResults) $("#cmdOut").innerHTML = ui.cmdResults.map(renderCommandResult).join("");
  if (ui.queryAnswer) renderQueryAnswer(ui.queryAnswer);
  if (ui.diary) renderDiary(ui.diary);
  if (ui.chain) renderChain(ui.chain);
  rewordGraph();
  if (ui.detail) ui.detail();
  if (ui.playing !== undefined) showPlaying(ui.playing);
  syncExpand();
  worldUi.things = "";
  if (worldUi.home) takeHome(worldUi.home);
  eyeUi.trustKey = "";
  eyeUi.noteKey = "";
  worldUi.key = "";
  worldUi.battery = "";
  if (eyeUi.body) takeBody(eyeUi.body);
}

function boot() {
  ui.ready = true;
  renderChips();
  setupEye();
  syncExpand();
  I18N.onChange(rerenderAll);
  document.addEventListener("i18n:note", (event) => toast(event.detail.text));
  setInterval(heartbeat, 2000);
  setInterval(pollPaused, 4000);
  setInterval(askBody, 2000);
  watch3dLoad();
  safe("state", {}).then(() => loadGraph());
}
I18N.ready.then(boot);
