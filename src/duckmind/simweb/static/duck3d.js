/* 3D view of the official Microduck, driven by Duck Core's intents.
 *
 * Geometry and kinematics come from the official MJCF + STL meshes (read from the
 * local microduck_rl checkout by the server). Head joints follow Duck Core's clip
 * frames exactly as duck-body's clip player would send them: HOME + delta, low-pass
 * filtered like robotd's head smoothing. The body blends the official SIT and STAND
 * keyframes (legs and trunk height) for posture changes; a walk shows an illustrative
 * stepping cycle around STAND while the body moves from Duck Core's `from` pose to its `to`
 * pose at the ground speed and yaw rate the official physics measured, so the duck is
 * always exactly where Duck Core says it is; a fall lays the trunk on its side, and each
 * self-righting try rocks it before it sits back up (or stays down).
 *
 * Every frame ends in applyPose(): {joints: {name: angle}, trunk: [x, y, z],
 * quat: [w, x, y, z], mouth: 0..1} in MuJoCo's frame.
 *
 * Live physics (phase 0 「看見真相」): the solid duck is drawn from the physics truth only —
 * each /api/body frame's `truth` (where the body is in the world) with its height, lean and
 * joints (truth.js) — and never jumps to where Duck Core thinks it is. Duck Core's plan (the
 * animation above) moves a translucent shadow duck with a ring of its own position error;
 * the last 10 minutes of the truth are a line on the floor; a label says which duck is drawn.
 * In a fast-forward or without live physics the solid duck is the plan, and the label says
 * so. The room's light comes from the server (world/lights.py).
 *
 * Phase 2 (「真的家與真的感測器」): the room is the home the physics runs (home3d.js, from
 * the same layout: walls, sofa, table… are solid in the physics too), drawn in world
 * coordinates; Duck Core's plan is moved into it by the home's frame (its nest or mat;
 * identity in default_home). The world's hand, the family's bodies, the toys, the nest's
 * charging light and backlight move as /api/body's `world` says; walls between the camera and
 * the duck fade (B4). Only while the physics runs another world (phase 0's flat floor) is the
 * room translucent: then it really is only on the page. */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { buildHome, disposeHome, fadeWalls, homeSpan, planToWorld, setGhost, setWorld, stepWorld, woodTexture, worldToPlan } from "./home3d.js";
import {
  callsBetween,
  beakCaption,
  compose,
  follow,
  framePose,
  makePlayout,
  mergeFrames,
  modeLabel,
  playoutArrive,
  playoutTime,
  poseGap,
  sampleFrames,
  trailPush,
  trailSegments,
  TRAIL_MAX,
  truthTarget,
  unpackFrame,
} from "./truth.js";
import { OVERHEAD_KINDS, makeClickTracker, makeErrandTrack, resolvePick, sceneOfWorld } from "./pointing.js";

const HEAD = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"];
// Fallbacks only: the server sends Duck Core's own sign table and mouth travel.
const DEFAULT_SIGN = { neck_pitch: -1, head_pitch: -1, head_yaw: 1, head_roll: 1 };
const DEFAULT_MOUTH_TRAVEL = 0.6109; // −5°..+30°
// The official sim / robotctl monitor open on this three-quarter view (radians).
const VIEW_AZIMUTH = -0.7;
const VIEW_ELEVATION = 0.32;
// Clips that hold their last frame until something else happens (a stir, waking up).
const HOLDS = new Set(["nap"]);
const POSE_MS = 1000; // sit ↔ stand takes the policies about a second
const FALL_MS = 650;
const RIGHTING_MS = 1300; // one try to roll back up after a fall
const GAIT_HZ = 1.6;
const QUEUE_MAX = 24; // body actions waiting to be shown; older ones are applied instantly
const CLIP_QUEUE_MAX = 12; // expressions waiting to be shown
const MAX_RATE = 10; // fast-forward plays a backlog up to this many times faster
const YAW_RATIO = 0.3; // Duck Core's model: a 1.0 rad/s turn command turns ~0.3 rad/s
// Page words (static/i18n.js, docs/i18n.md): 繁體中文 is the source, five translations follow
// the viewer's language. T(key, values) the words of a key; F(obj, field) a server field in the
// viewer's language (in 繁體中文 exactly obj[field]).
const I18N = window.I18N;
const T = (key, values) => (I18N ? I18N.t(key, values) : key);
const F = (obj, name) => (obj ? (I18N ? I18N.field(obj, name) : obj[name]) ?? "" : "");
const FS = (obj, name) => (obj ? (I18N ? I18N.items(obj, name) : obj[name]) || [] : []);
const SOURCE_WORDS = () => !I18N || I18N.isSource() || I18N.lang === I18N.PSEUDO;
const POSTURE_ZH = {
  sit: "page.duck3d.POSTURE_ZH.sit",
  stand: "page.duck3d.POSTURE_ZH.stand",
  walk: "page.duck3d.POSTURE_ZH.walk",
  fallen: "page.duck3d.POSTURE_ZH.fallen",
  rise: "page.duck3d.POSTURE_ZH.rise",
};
const BODY_KEYS = ["walk", "place", "righting", "posture", "fall"];
const DAY_SKY = new THREE.Color(0xe9eef0);
const NIGHT_SKY = new THREE.Color(0x2b3446);
const DAY_LIGHT = new THREE.Color(0xf4f7f2);
const NIGHT_LIGHT = new THREE.Color(0x9fb0d8); // moonlight
const DUSK_SKY = new THREE.Color(0xf0c7a0); // sunrise / sunset glow
const SUN_WARM = new THREE.Color(0xffb46e);
const SUN_DAY = new THREE.Color(0xffffff);
const LAMP_COLOR = 0xffd3a0;
const HOME_LAT = 25.03; // the duck's home: Taipei (sunrise and sunset follow its date)
const HOME_LON = 121.56;
const HOME_UTC = 8;
// 繁體中文's one-character weekday after 星期 (Sunday first, like Date.getDay()).
const WEEKDAY_ZH = [
  "page.duck3d.WEEKDAY_ZH.15917f",
  "page.duck3d.WEEKDAY_ZH.d274ee",
  "page.duck3d.WEEKDAY_ZH.1d5639",
  "page.duck3d.WEEKDAY_ZH.49ddb0",
  "page.duck3d.WEEKDAY_ZH.4f8874",
  "page.duck3d.WEEKDAY_ZH.8f07f5",
  "page.duck3d.WEEKDAY_ZH.3d72c7",
];
// Duck Core's own room (its plan's limits, in its own frame): 6 m × 6 m around its nest or mat.
const ROOM_HALF = 3.0; // huge at a 20 cm duck's scale
const WALL_MARGIN = 0.25; // the duck turns back before it reaches furniture at the walls
const LIVE_POLL_MS = 60;
// Without live physics on the server there is nothing to follow frame by frame: ask now
// and then (in case it is switched on) instead of ~17 times a second.
const IDLE_POLL_MS = 3000;
const TRAIL_Y = 0.0068; // the truth line lies on the floor, above the rug, mat and nest pad
const TRAIL_HALF_W = 0.006; // 1.2 cm wide
const RING_Y = 0.0072;
const SHADOW_OPACITY = 0.32;

const box = document.getElementById("duck3d");
const readout = document.getElementById("duck3dReadout");
const note = document.getElementById("duck3dNote");
const bubbleEl = document.getElementById("duck3dBubble");
const fxEl = document.getElementById("duck3dFx");
const captionEl = document.getElementById("duck3dCaption");
const zzzEl = document.getElementById("duck3dZzz");
const modeEl = document.getElementById("stageMode");

// The colours of the truth (its line) and of the plan (the shadow duck and its ring): the
// page's tokens (style.css --truth, --plan), so the words and the 3D view agree.
function token(name, fallback) {
  try {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  } catch (error) {
    return fallback;
  }
}
const TRUTH_COLOR = token("--truth", "#1f9d57");
const PLAN_COLOR = token("--plan", "#3b6fd8");

// What the duck is doing right now, in words, while it does it: head moves on a 20 cm
// duck are a few degrees, easy to miss from across the room.
let captionTimer = 0;
let captionAt = 0;
const captionMain = { text: "", until: 0 };
// A line above the words of the moves: what the duck is up to (a clever or goofy moment,
// setting off, the way home, a guess…), kept for a few seconds while its moves play.
const captionLead = { text: "", until: 0, moment: false };

function renderCaption() {
  if (!captionEl) return;
  const now = performance.now();
  const lead = captionLead.text && now < captionLead.until ? captionLead.text : "";
  const main = captionMain.text && now < captionMain.until && captionMain.text !== lead ? captionMain.text : "";
  const parts = [];
  if (lead) {
    const b = document.createElement("b");
    b.className = "lead";
    b.textContent = lead;
    parts.push(b);
  }
  if (main) {
    const span = document.createElement("span");
    span.textContent = main;
    parts.push(span);
  }
  captionEl.replaceChildren(...parts);
  captionEl.classList.toggle("show", parts.length > 0);
  captionEl.classList.toggle("moment", !!lead && captionLead.moment);
  clearTimeout(captionTimer);
  const ends = [lead ? captionLead.until : 0, main ? captionMain.until : 0].filter((t) => t > now);
  if (ends.length) captionTimer = setTimeout(renderCaption, Math.min(...ends) - now + 30);
}

// ``weak``: a reply or a note, which never covers the words of a move that just started.
// ``own`` (a move): later replies wait a second; a click's instant echo is not one.
function caption(text, weak = false, own = true) {
  if (!captionEl || !text) return;
  if (weak && performance.now() - captionAt < 1000) return;
  if (!weak && own) captionAt = performance.now();
  captionMain.text = text;
  captionMain.until = performance.now() + (weak ? 4200 : 3200);
  renderCaption();
}

// Duck Core's words for what it is up to (no move of its own): the lead line. A clever or
// goofy moment stays longest, and the notes that follow it go below it meanwhile.
// Each lead stays at least NOTE_MIN_MS before the next one takes its place (two in the same
// second — waking up, then a goofy moment — are both readable); a short queue keeps their
// order and drops the oldest when a fast-forward says more than can be read.
const NOTE_MIN_MS = 1600;
const noteQueue = [];
let noteTimer = 0;
let noteShownAt = -Infinity;

function drainNotes() {
  noteTimer = 0;
  const next = noteQueue.shift();
  if (!next) return;
  noteShownAt = -Infinity; // its turn now
  sayNote(next.text, next.kind, true);
}

function sayNote(text, kind, queued = false) {
  if (!captionEl || !text) return;
  const now = performance.now();
  const moment = kind === "duck.moment";
  if (!moment && captionLead.moment && now < captionLead.until - 1500) {
    caption(text, true); // a moment is showing: this goes below it
    return;
  }
  const reading = now - noteShownAt < NOTE_MIN_MS && captionLead.text && now < captionLead.until;
  if (reading || (noteQueue.length && !queued)) {
    noteQueue.push({ text, kind });
    while (noteQueue.length > 2) noteQueue.shift();
    if (!noteTimer) noteTimer = setTimeout(drainNotes, Math.max(50, NOTE_MIN_MS - (now - noteShownAt)));
    return;
  }
  noteShownAt = now;
  captionLead.text = moment ? `✨ ${text}` : text;
  captionLead.moment = moment;
  captionLead.until = now + (moment ? 6500 : 4200);
  captionMain.text = ""; // what was said before is over
  renderCaption();
  if (noteQueue.length && !noteTimer) noteTimer = setTimeout(drainNotes, NOTE_MIN_MS);
}

const view = {
  joints: {},
  rendered: {},
  clip: null,
  queue: [],
  awake: true,
  napping: false,
  rest: false,
  lift: 0,
  liftUntil: 0,
  walkUntil: 0, // gamepad replay (illustrative)
  holdUntil: 0,
  mouth: 0,
  drift: { yaw: 0, next: 0 },
  clips: null,
  model: null,
  jawPivot: null,
  headAnchor: null,
  bubbleUntil: 0,
  trunk: null,
  camera: null,
  controls: null,
  fitBounds: null,
  fitted: false,
  poses: null,
  body: {
    shown: "sit", // settled posture
    to: null, // posture being blended toward
    from: null, // pose the blend started from
    t0: 0,
    dur: 0,
    walk: null, // {vx, vyaw, speed, yawRate, start, end, verb, from, to}
    rock: null, // a self-righting try being shown: {start, end, ok}
    yaw: 0, // heading (Duck Core's, radians, counter-clockwise from +x)
    x: 0, // where Duck Core says it is on the floor (MuJoCo x/y, metres; mat = 0, 0)
    y: 0,
    queue: [],
    state: "sit", // what Duck Core's last snapshot said
    position: null, // [x, y, heading] from Duck Core's last snapshot
  },
  spots: null, // floor markings for the spots it learned to avoid
  spotsKey: "",
  external: null,
  playNow: 0, // the animation's own clock: runs faster than real time while catching up
  rate: 1,
  follow: null, // last duck position the camera followed
  // 「近看小鴨」: the close camera on the head (on, where it aims, the viewer's own drag until)
  closeup: { on: false, aim: null, anchor: null, userUntil: 0, button: null, centre: null },
  sun: null,
  // Live physics: the latest frame (`external`), the truth as drawn (`shown`), the last world
  // placement, hand request and demonstration seen (a new one is said in words), whether the
  // head is drawn from physics too. `frames`: every body sample the server sent (truth.js
  // mergeFrames), played at its own moment on the `play` clock (fid:page: never eased); `lastT`
  // the newest one's simulated time (the next ask: ?since_t=); `drawnRate` samples drawn a second.
  live: {
    enabled: false, on: true, info: null, shown: null, busy: false, demo: null, placedSeen: undefined, handSeen: undefined, placedNoteSeen: undefined, demoSeen: undefined, physicsHead: true,
    frames: [], lastT: null, play: makePlayout(), played: null, playState: "", sampleHz: null, drawnMouth: null, drawnMouthSource: "unavailable",
    playedW: null, frameCalls: false,
  },
  drawnMode: "plan", // "truth": the solid duck is the physics body; "plan": Duck Core's
  shadow: null, // the translucent duck at Duck Core's plan: {mount, joints}
  planRing: null, // Duck Core's own position error around the plan
  gapLine: null, // from the solid duck to the shadow, on the floor
  poseErr: 0, // metres (Duck Core's pose_err_mm)
  trail: [], // the truth's last 10 minutes: {x, y, a (the placement it follows), t}
  trailMesh: null,
  trailDirty: false,
  trailBuilt: 0,
  serverLight: null, // world/lights.py at the duck's time (server.lights); null: an older server
  // Things Duck Core did that are still to come on the wall clock (real time): each plays
  // at the second it happens, together with the live body. {at: performance ms, item}.
  pending: [],
  seenFidgets: new Set(),
  realtimeClock: false,
  time: { playing: false, realtime: false, speed: "rt" },
  sleepAmp: 0.6, // the nap pose is Duck Core's nap move at its own amplitude
  hemi: null,
  scene: null,
  // The home drawn (home3d.js, from the server's layout): `home` once loaded; `fallback` the
  // plain floor shown until then; `physical` while the physics runs this very home (its
  // fingerprint); `leash` Duck Core's morning leash around its nest (drawn here only).
  room: { home: null, fingerprint: "", loading: false, fallback: null, physical: false, leash: null, belief: null, frame: { x_mm: 0, y_mm: 0, heading_mrad: 0 } },
  world: null, // /api/body's world (the hand, people, things, battery, the nest's contact)
  hasNest: false,
  phaseZh: null, // the day (its phase_zh), for a duck that lives the family's day
  offline: false, // the server does not answer: nothing here is "now" any more
};

function fail(message) {
  if (note) {
    delete note.dataset.i18n; // (its own words now, not the page's 「正在載入…」)
    note.textContent = message;
  }
  document.body.classList.remove("has3d");
}

async function init() {
  if (!box) return;
  let model;
  try {
    const [m, c] = await Promise.all([fetch("/api/robot").then((r) => r.json()), fetch("/api/clips").then((r) => r.json())]);
    model = m;
    view.clips = c;
  } catch (error) {
    fail(T("page.duck3d.init.9d1ec6", { message: error.message }));
    return;
  }
  if (!model.available) {
    fail(T("page.duck3d.init.e1b65b", { reason: F(model, "reason") }));
    return;
  }
  view.model = model;
  view.poses = makePoses(model);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  box.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 5 / 4, 0.01, 80);
  camera.position.set(0.36, 0.2, 0.42);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.1, 0);
  controls.enableDamping = true;
  controls.minDistance = 0.18;
  controls.maxDistance = 7;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.update();
  view.camera = camera;
  view.controls = controls;

  const hemi = new THREE.HemisphereLight(0xf4f7f2, 0x6d7d74, 1.25);
  scene.add(hemi);
  view.hemi = hemi;
  view.scene = scene;
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(0.6, 1.2, 0.5);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.left = -0.6;
  sun.shadow.camera.right = 0.6;
  sun.shadow.camera.top = 0.6;
  sun.shadow.camera.bottom = -0.6;
  sun.shadow.camera.far = 6;
  scene.add(sun);
  scene.add(sun.target);
  view.sun = sun;
  const fill = new THREE.DirectionalLight(0xdfeee9, 0.5);
  fill.position.set(-0.5, 0.4, -0.6);
  scene.add(fill);
  view.fill = fill;
  // Room lamps: switched on in the evening, like a family's living room.
  const lamp = new THREE.PointLight(LAMP_COLOR, 0, 9, 1.6);
  lamp.position.set(0.9, 1.9, -1.6);
  scene.add(lamp);
  view.lamp = lamp;

  setupRoom(scene, renderer);
  loadHome(null);
  view.spots = new THREE.Group();
  scene.add(view.spots);

  // MuJoCo is Z-up; three.js is Y-up.
  const world = new THREE.Group();
  world.rotation.x = -Math.PI / 2;
  scene.add(world);
  const trunkMount = new THREE.Group();
  world.add(trunkMount);
  view.trunk = trunkMount;

  const loader = new STLLoader();
  const cache = new Map();
  const loadMesh = (name) => {
    if (!cache.has(name)) cache.set(name, loader.loadAsync(`/mesh/${encodeURIComponent(name)}`));
    return cache.get(name);
  };
  const materials = new Map();
  const materialFor = (rgba) => {
    const key = rgba.join(",");
    if (!materials.has(key)) {
      materials.set(
        key,
        new THREE.MeshStandardMaterial({
          color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
          roughness: 0.62,
          metalness: 0.04,
          transparent: rgba[3] < 1,
          opacity: rgba[3],
        }),
      );
    }
    return materials.get(key);
  };

  const pending = [];
  // One copy of the official model: the solid duck (`into.ghost` unset: its joints are
  // view.joints, with the jaw on its hinge) or the shadow duck (one translucent material,
  // its own joints, no shadows cast).
  const build = (body, parent, into) => {
    const group = new THREE.Group();
    group.name = body.name || "";
    group.position.set(...body.pos);
    group.quaternion.set(body.quat[1], body.quat[2], body.quat[3], body.quat[0]);
    parent.add(group);
    let inner = group;
    if (body.joint) {
      const joint = new THREE.Group();
      group.add(joint);
      into.joints[body.joint.name] = { group: joint, axis: new THREE.Vector3(...body.joint.axis).normalize(), range: body.joint.range };
      inner = joint;
    }
    let jawGroup = null;
    for (const geom of body.geoms) {
      const holder = new THREE.Group();
      holder.position.set(...geom.pos);
      holder.quaternion.set(geom.quat[1], geom.quat[2], geom.quat[3], geom.quat[0]);
      if (!into.ghost && (geom.mesh === "jaw" || geom.mesh === "jaw_soft")) {
        if (!jawGroup) {
          jawGroup = new THREE.Group();
          inner.add(jawGroup);
          view.jawPivot = { group: jawGroup, parent: inner, meshes: [] };
        }
        jawGroup.add(holder);
        view.jawPivot.meshes.push(holder);
      } else {
        inner.add(holder);
      }
      pending.push(
        loadMesh(geom.mesh).then((geometry) => {
          const mesh = new THREE.Mesh(geometry, into.ghost || materialFor(geom.rgba));
          mesh.castShadow = !into.ghost;
          mesh.receiveShadow = !into.ghost;
          holder.add(mesh);
        }),
      );
    }
    if (!into.ghost && body.name === "jaw_soft") view.headAnchor = inner;
    for (const child of body.children) build(child, inner, into);
  };
  build(model.trunk, trunkMount, { joints: view.joints });
  trunkMount.children[0].position.set(0, 0, 0);
  // The shadow duck: where Duck Core thinks it is and what it commands, drawn only while the
  // solid duck is the physics truth. Its own Z-up group, so it never counts in the fit.
  const shadowWorld = new THREE.Group();
  shadowWorld.rotation.x = -Math.PI / 2;
  scene.add(shadowWorld);
  const shadowMount = new THREE.Group();
  shadowMount.visible = false;
  shadowWorld.add(shadowMount);
  const ghost = new THREE.MeshStandardMaterial({ color: PLAN_COLOR, roughness: 1, metalness: 0, transparent: true, opacity: SHADOW_OPACITY, depthWrite: false });
  view.shadow = { mount: shadowMount, joints: {}, material: ghost };
  build(model.trunk, shadowMount, { joints: view.shadow.joints, ghost });
  shadowMount.children[0].position.set(0, 0, 0);
  buildPlanRing(scene);
  buildGapLine(scene);
  buildTrail(scene);

  try {
    await Promise.all(pending);
  } catch (error) {
    fail(T("page.duck3d.init.dfa1a2", { message: error.message }));
    return;
  }
  setupJawPivot();
  measurePoses(world);
  document.body.classList.add("has3d");
  if (note) {
    // (the model's credit replaces the page's 「正在載入…」 note: written here, in the viewer's language)
    delete note.dataset.i18n;
    note.textContent = F(model, "attribution");
  }
  window.duck3d = { applyPose, setExternalPose, view };
  if (window.duckTime) view.time = window.duckTime;
  if (window.duckLast) window.dispatchEvent(new CustomEvent("duck:state", { detail: window.duckLast }));
  startLive();
  if (new URLSearchParams(location.search).has("debug3d")) window.__duck3d = { view, camera, controls };

  const canvas = renderer.domElement;
  const clockBox = document.getElementById("duck3dClock");
  const resize = () => {
    const width = Math.max(1, Math.round(canvas.clientWidth || box.clientWidth || 320));
    const height = Math.max(1, Math.round(canvas.clientHeight || width * 0.8));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    refit();
  };
  new ResizeObserver(resize).observe(canvas);
  step(0, performance.now());
  resize();
  setupCloseUp();
  setupPointing(canvas, camera);

  const clock = new THREE.Clock();
  const tmp = new THREE.Vector3();
  const loop = () => {
    // First-order smoothing is stable at any step, so slow devices still converge.
    const dt = Math.min(clock.getDelta(), 1);
    const now = performance.now();
    step(dt, now);
    followDuck(dt);
    controls.update();
    tickPointing(now);
    renderer.render(scene, camera);
    const rect = canvas.getBoundingClientRect();
    const boxRect = box.getBoundingClientRect();
    if (captionEl) {
      // Words go over the floor at the bottom left, never over the duck's head.
      captionEl.style.top = `${Math.max(8, rect.bottom - boxRect.top - captionEl.offsetHeight - 10)}px`;
    }
    if (modeEl && !modeEl.hidden) {
      // Which duck is drawn: just under the clock.
      const below = clockBox && clockBox.offsetHeight ? clockBox.offsetTop + clockBox.offsetHeight + 8 : 10;
      modeEl.style.top = `${below}px`;
    }
    if (view.headAnchor && bubbleEl) {
      view.headAnchor.getWorldPosition(tmp);
      tmp.y += 0.07;
      tmp.project(camera);
      const bubbleX = (tmp.x * 0.5 + 0.5) * rect.width + rect.left - boxRect.left;
      const bubbleY = (-tmp.y * 0.5 + 0.5) * rect.height + rect.top - boxRect.top;
      const half = (bubbleEl.offsetWidth || 60) / 2 + 6;
      // Kept inside the view: above the head when there is room, else beside it.
      bubbleEl.style.left = `${Math.min(rect.width - half, Math.max(half, bubbleX))}px`;
      bubbleEl.style.top = `${Math.max((bubbleEl.offsetHeight || 28) * 1.1 + 6, bubbleY)}px`;
      bubbleEl.classList.toggle("show", now < view.bubbleUntil);
      if (zzzEl) {
        zzzEl.style.left = bubbleEl.style.left;
        zzzEl.style.top = bubbleEl.style.top;
        const asleep = view.neuro ? !!view.neuro.asleep : view.awake && view.napping;
        zzzEl.classList.toggle("show", asleep && now >= view.bubbleUntil);
      }
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

/* ---------------- the room ---------------- */

// Until the home arrives (and if it cannot be read): a plain wooden floor, Duck Core's 6 × 6 m.
function setupRoom(scene, renderer) {
  scene.background = new THREE.Color(0xe9eef0);
  scene.fog = new THREE.Fog(0xe9eef0, 5, 14);
  const size = ROOM_HALF * 2;
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(size, size),
    new THREE.MeshStandardMaterial({ map: woodTexture(renderer, [size / 1.2, size / 1.2]), roughness: 0.8 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  floor.userData.kind = "floor"; // (a click on it is a place on the floor: pointing.js)
  scene.add(floor);
  view.room.fallback = floor;
  view.renderer = renderer;
  // While the family sleeps it keeps close to the nest (≤ 1.5 m): a faint ring on the floor,
  // around Duck Core's origin (drawn here only).
  const leash = new THREE.Mesh(
    new THREE.RingGeometry(1.49, 1.51, 128),
    new THREE.MeshBasicMaterial({ color: 0x8a5cf5, transparent: true, opacity: 0.35, depthWrite: false }),
  );
  leash.rotation.x = -Math.PI / 2;
  leash.position.y = 0.003;
  leash.visible = false;
  scene.add(leash);
  view.room.leash = leash;
  // A dock against the wall: Duck Core still believes its nest is at its own origin until
  // phase 5 teaches it where the dock is; a ring in the plan's colour marks that believed nest.
  const belief = new THREE.Mesh(
    new THREE.RingGeometry(0.17, 0.2, 64),
    new THREE.MeshBasicMaterial({ color: PLAN_COLOR, transparent: true, opacity: 0.55, depthWrite: false, side: THREE.DoubleSide }),
  );
  belief.rotation.x = -Math.PI / 2;
  belief.position.y = 0.0035;
  belief.visible = false;
  scene.add(belief);
  view.room.belief = belief;
}

// The home the server draws (the layout the physics runs): built once per fingerprint.
async function loadHome(payload) {
  const room = view.room;
  if (!view.scene) return;
  let home = payload;
  if (!home) {
    if (room.loading) return;
    room.loading = true;
    try {
      home = await fetch("/api/home", { cache: "no-store" }).then((r) => r.json());
    } catch (error) {
      home = null;
    } finally {
      room.loading = false;
    }
  }
  if (!home || !home.layout || !home.fingerprint || home.fingerprint === room.fingerprint) return;
  disposeHome(view.scene, room.home);
  room.home = buildHome(view.scene, home, view.renderer);
  room.fingerprint = home.fingerprint;
  room.frame = home.layout.frame || { x_mm: 0, y_mm: 0, heading_mrad: 0 };
  if (room.fallback) room.fallback.visible = !room.home;
  // The room's lamp hangs where the home's first lamp is.
  const lamp = room.home && room.home.lamps[0];
  if (lamp && view.lamp) view.lamp.position.copy(lamp.three);
  const [lx, ly] = planToWorld(room.frame, 0, 0, 0);
  if (room.leash) room.leash.position.set(lx, 0.003, -ly);
  // (the believed nest, where the physical dock is not: default_home keeps Duck Core's frame)
  const dock = home.layout.nest && home.layout.nest.kind === "dock" ? home.layout.nest : null;
  const apart = dock && dock.seat ? Math.hypot(lx - dock.seat.x_mm / 1000, ly - dock.seat.y_mm / 1000) : 0;
  room.beliefApart = apart > 0.3;
  if (room.belief) {
    room.belief.position.set(lx, 0.0035, -ly);
    room.belief.visible = !!(room.beliefApart && view.hasNest);
  }
  // The camera can back off to see the whole home; the haze starts beyond it.
  const span = homeSpan(room.home);
  if (view.controls) view.controls.maxDistance = Math.max(view.controls.maxDistance, span * 0.9);
  if (view.scene.fog) {
    view.scene.fog.near = Math.max(5, span * 0.8);
    view.scene.fog.far = Math.max(14, span * 2.2);
  }
  if (view.world) setWorld(room.home, view.world, view.nestCharging);
  // The home this view drew (at start it comes from /api/home, and /api/body then carries no
  // home_json): card 09's toy chooser (app.js) takes the home's toys from here.
  window.dispatchEvent(new CustomEvent("duck:home", { detail: home }));
}

// Duck Core's own idea of how far off its position may be (pose_err_mm), as a ring on the
// floor around where it thinks it is.
function buildPlanRing(scene) {
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.994, 1.0, 96),
    new THREE.MeshBasicMaterial({ color: PLAN_COLOR, transparent: true, opacity: 0.7, depthWrite: false, side: THREE.DoubleSide }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = RING_Y;
  ring.visible = false;
  ring.userData.radius = 1;
  scene.add(ring);
  view.planRing = ring;
}

function showPlanRing(x, y) {
  const ring = view.planRing;
  if (!ring) return;
  const r = view.poseErr;
  ring.visible = r >= 0.005;
  if (!ring.visible) return;
  if (Math.abs(r - ring.userData.radius) > 0.002) {
    // A band about 1 cm wide whatever the radius.
    ring.geometry.dispose();
    ring.geometry = new THREE.RingGeometry(Math.max(0.001, r - 0.005), r + 0.005, 96);
    ring.userData.radius = r;
  }
  ring.position.set(x, RING_Y, -y); // MuJoCo (x, y) on the scene's floor
}

// From the solid duck to its shadow, on the floor: which way (and how far) the duck thinks it
// is, even when the shadow is out of view.
function buildGapLine(scene) {
  const strip = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({ color: PLAN_COLOR, transparent: true, opacity: 0.6, depthWrite: false, side: THREE.DoubleSide }),
  );
  strip.rotation.x = -Math.PI / 2; // flat on the floor, along the group's x
  const holder = new THREE.Group();
  holder.add(strip);
  holder.visible = false;
  scene.add(holder);
  view.gapLine = holder;
}

function showGapLine(from, to) {
  const line = view.gapLine;
  if (!line) return;
  const dx = from ? to[0] - from[0] : 0;
  const dy = from ? to[1] - from[1] : 0;
  const length = Math.hypot(dx, dy);
  line.visible = !!from && length > 0.05;
  if (!line.visible) return;
  // MuJoCo (x, y) is three.js (x, −z): the strip turns by atan2(dy, dx) about the vertical.
  line.position.set((from[0] + to[0]) / 2, RING_Y - 0.0002, -(from[1] + to[1]) / 2);
  line.rotation.y = Math.atan2(dy, dx);
  line.scale.set(length, 1, 0.008);
}

// The truth's last 10 minutes: a flat ribbon on the floor (a WebGL line is one pixel).
function buildTrail(scene) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL_MAX * 4 * 3), 3));
  const index = new Uint32Array(TRAIL_MAX * 6);
  for (let i = 0; i < TRAIL_MAX; i++) index.set([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 2, i * 4 + 1, i * 4 + 3], i * 6);
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.setDrawRange(0, 0);
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: TRUTH_COLOR, side: THREE.DoubleSide }));
  mesh.frustumCulled = false;
  mesh.visible = false;
  scene.add(mesh);
  view.trailMesh = mesh;
}

function showTrail(on, realNow) {
  const mesh = view.trailMesh;
  if (!mesh) return;
  mesh.visible = on;
  if (!on || (!view.trailDirty && realNow - view.trailBuilt < 5000) || realNow - view.trailBuilt < 250) return;
  view.trailDirty = false;
  view.trailBuilt = realNow;
  const segments = trailSegments(view.trail, Date.now()).slice(-TRAIL_MAX);
  const pos = mesh.geometry.attributes.position;
  segments.forEach(([x0, y0, x1, y1], i) => {
    const len = Math.hypot(x1 - x0, y1 - y0) || 1;
    const nx = (-(y1 - y0) / len) * TRAIL_HALF_W;
    const ny = ((x1 - x0) / len) * TRAIL_HALF_W;
    // three.js: MuJoCo (x, y) is (x, −z) on the floor.
    pos.setXYZ(i * 4, x0 + nx, TRAIL_Y, -(y0 + ny));
    pos.setXYZ(i * 4 + 1, x0 - nx, TRAIL_Y, -(y0 - ny));
    pos.setXYZ(i * 4 + 2, x1 + nx, TRAIL_Y, -(y1 + ny));
    pos.setXYZ(i * 4 + 3, x1 - nx, TRAIL_Y, -(y1 - ny));
  });
  pos.needsUpdate = true;
  mesh.geometry.setDrawRange(0, segments.length * 6);
}

// What Duck Core says about its nest (null: no nest): its charging light when the world does not
// say (no live physics), and the morning leash while the family sleeps. The nest itself is
// part of the home (home3d.js): a duck with a nest lives in a home with one.
function showNest(nest, day) {
  const room = view.room;
  view.nestCharging = !!(nest && nest.charging);
  if (room.belief) room.belief.visible = !!(room.beliefApart && nest);
  if (room.home && !(view.world && view.world.nest)) setWorld(room.home, view.world || {}, view.nestCharging);
  if (room.leash) {
    const solo = !!nest && day && day.phase === "solo";
    room.leash.visible = !!solo;
    if (solo) {
      const r = nest.leash_m || Math.min(1.5, nest.roam_m || 1.5); // Duck Core's own leash now
      room.leash.scale.set(r / 1.5, r / 1.5, 1);
    }
  }
}

// Keep the duck in view as it walks: move the orbit target and camera by how far it went,
// so the user's chosen angle and zoom stay; the sun's shadow box travels along. 「近看小鴨」
// (closeUp) instead keeps the camera close on its head.
function followDuck(dt = 0) {
  const { trunk, controls, camera, sun } = view;
  if (!trunk || !controls) return;
  const p = trunk.getWorldPosition(new THREE.Vector3());
  p.y = 0;
  if (view.closeup.on) closeUpFrame(dt, p);
  else if (view.follow) {
    const delta = p.clone().sub(view.follow);
    if (delta.lengthSq() > 1e-10) {
      controls.target.add(delta);
      camera.position.add(delta);
    }
  }
  view.follow = p;
  if (sun) {
    sun.position.set(p.x + 0.6, 1.2, p.z + 0.5);
    sun.target.position.copy(p);
  }
}

/* ---------------- 「近看小鴨」: a close camera on the head (fid:page; fid:fix) ---------------- */
// A hand's length from the head, from a three-quarter front (the side nearer the camera), a
// little above. (fid:fix 2026-09-28) The camera aims at a point fixed on the TRUNK — where the head
// sits on it, on average — and follows the trunk slowly, so the head's own snap, turn or tilt
// moves INSIDE the view (the review: aimed at the head itself and re-centred within ~0.3 s, a
// startle's jerk was followed out of the frame — the background moved instead — and a fall spun
// the view with the head). Only the trunk's heading turns the anchor (not its roll or pitch: a fall
// does not spin the view); framed for ±40° of head yaw. A drag or the wheel is the viewer's own
// for a few seconds.
const CLOSE_DIST = 0.52; // metres from the anchor (the duck is ~0.2 m tall: head, neck and chest)
const CLOSE_MIN = 0.18;
const CLOSE_MAX = 1.0;
const CLOSE_SIDE = 0.9; // rad off the duck's nose: a three-quarter front view (face and profile)
const CLOSE_ELEV = 0.3; // rad above the head
const CLOSE_AIM_UP = 0.005; // m above the anchor
const CLOSE_TARGET_S = 0.8; // the aim follows the TRUNK this slowly (the head moves in the view)
const CLOSE_ANCHOR_S = 6.0; // where the head sits on the trunk, on average, adapts this slowly
const CLOSE_ORBIT_S = 1.1; // the angle follows the duck's heading this slowly
const CLOSE_HANDS_OFF_MS = 5000; // after the viewer turns the view, it stays theirs this long
const CLOSE_KEY = "duckmind.closeup";
const UP = new THREE.Vector3(0, 1, 0);

// The trunk's place and heading only (its turn about "up": a roll or a pitch — a fall — does
// not turn the anchor).
function trunkHeading(trunk) {
  const pos = trunk.getWorldPosition(new THREE.Vector3());
  const nose = new THREE.Vector3(1, 0, 0).applyQuaternion(trunk.getWorldQuaternion(new THREE.Quaternion()));
  const az = nose.x * nose.x + nose.z * nose.z > 1e-8 ? Math.atan2(-nose.z, nose.x) : 0;
  return { pos, q: new THREE.Quaternion().setFromAxisAngle(UP, az) };
}

function closeUpFrame(dt, floor) {
  const { controls, camera, trunk } = view;
  const c = view.closeup;
  // the middle of the head (its shell's centre, measured once in the head's own frame)
  if (view.headAnchor && !c.centre) {
    view.headAnchor.updateWorldMatrix(true, true);
    const box = new THREE.Box3().setFromObject(view.headAnchor);
    c.centre = box.isEmpty() ? new THREE.Vector3() : view.headAnchor.worldToLocal(box.getCenter(new THREE.Vector3()));
  }
  const head = view.headAnchor ? view.headAnchor.localToWorld(c.centre.clone()) : floor.clone().setY(0.16);
  // the anchor: where the head sits on the trunk (the trunk's heading frame), averaged slowly
  const frame = trunkHeading(trunk);
  const local = head.clone().sub(frame.pos).applyQuaternion(frame.q.clone().invert());
  if (!c.anchor) c.anchor = local.clone();
  else c.anchor.lerp(local, 1 - Math.exp(-Math.max(0, dt) / CLOSE_ANCHOR_S));
  const target = c.anchor.clone().applyQuaternion(frame.q).add(frame.pos);
  target.y += CLOSE_AIM_UP;
  if (!c.aim) c.aim = target.clone();
  const k = 1 - Math.exp(-Math.max(0, dt) / CLOSE_TARGET_S);
  c.aim.lerp(target, dt > 0 ? k : 1);
  const offset = camera.position.clone().sub(controls.target);
  let dist = offset.length();
  if (!Number.isFinite(dist) || dist < 1e-4) dist = closeDist();
  dist = Math.min(CLOSE_MAX, Math.max(CLOSE_MIN, dist));
  const handsOn = performance.now() < c.userUntil;
  let dir = offset.normalize();
  if (!handsOn) {
    // the duck's nose on the floor (three.js: MuJoCo x is x, MuJoCo y is −z)
    const nose = new THREE.Vector3(1, 0, 0).applyQuaternion(frame.q);
    nose.y = 0;
    if (nose.lengthSq() > 1e-6) {
      nose.normalize();
      const now = Math.atan2(dir.x, dir.z);
      const noseAz = Math.atan2(nose.x, nose.z);
      const left = noseAz + CLOSE_SIDE;
      const right = noseAz - CLOSE_SIDE;
      const off = (a) => Math.abs(Math.atan2(Math.sin(a - now), Math.cos(a - now)));
      const goal = off(left) <= off(right) ? left : right;
      const turn = Math.atan2(Math.sin(goal - now), Math.cos(goal - now));
      const az = now + turn * (1 - Math.exp(-Math.max(0, dt) / CLOSE_ORBIT_S));
      const elev = Math.asin(Math.min(0.99, Math.max(-0.2, dir.y)));
      const el = elev + (CLOSE_ELEV - elev) * (1 - Math.exp(-Math.max(0, dt) / CLOSE_ORBIT_S));
      dir = new THREE.Vector3(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
    }
  }
  controls.target.copy(c.aim);
  camera.position.copy(c.aim).addScaledVector(dir, dist);
}

// A portrait view (a phone) is narrow: back off so the head and its turns stay in it.
function closeDist() {
  const aspect = view.camera ? view.camera.aspect : 1;
  return CLOSE_DIST / Math.max(0.5, Math.min(1, aspect));
}

function setCloseUp(on, save = true) {
  const { controls, camera } = view;
  const c = view.closeup;
  c.on = !!on;
  if (c.button) {
    c.button.setAttribute("aria-pressed", String(c.on));
    // (I18N.text / attr remember the key: a language switch rewrites them in the state's words)
    const textKey = c.on ? "page.duck3d.setCloseUp.d7d834" : "page.duck3d.setCloseUp.adff10";
    const titleKey = c.on ? "page.duck3d.setCloseUp.fe57f6" : "page.duck3d.setCloseUp.6daa49";
    if (I18N) {
      I18N.text(c.button, textKey);
      I18N.attr(c.button, "title", titleKey);
    } else {
      c.button.textContent = T(textKey);
      c.button.title = T(titleKey);
    }
  }
  if (save) {
    try {
      localStorage.setItem(CLOSE_KEY, c.on ? "1" : "0");
    } catch (error) {
      /* (a private window: not remembered) */
    }
  }
  if (!controls || !camera) return;
  if (c.on) {
    c.aim = null;
    c.anchor = null;
    c.userUntil = 0;
    controls.minDistance = CLOSE_MIN;
    controls.maxDistance = CLOSE_MAX;
    const offset = camera.position.clone().sub(controls.target);
    camera.position.copy(controls.target).addScaledVector(offset.normalize(), closeDist());
  } else {
    c.aim = null;
    c.anchor = null;
    view.fitted = false; // (back to the first view's angle and distance)
    refit();
  }
}

function setupCloseUp() {
  const c = view.closeup;
  c.button = document.getElementById("btnCloseup");
  if (view.controls) {
    // the viewer's own drag or zoom: theirs for a few seconds
    view.controls.addEventListener("start", () => {
      c.userUntil = performance.now() + CLOSE_HANDS_OFF_MS;
    });
    view.controls.addEventListener("end", () => {
      c.userUntil = performance.now() + CLOSE_HANDS_OFF_MS;
    });
  }
  if (!c.button) return;
  c.button.hidden = false;
  c.button.addEventListener("click", () => setCloseUp(!c.on));
  let saved = null;
  try {
    saved = localStorage.getItem(CLOSE_KEY);
  } catch (error) {
    saved = null;
  }
  setCloseUp(saved === "1", false);
}

/* ---------------- 點地板叫牠過去: a short click on the floor sends the duck there ----------------
 * A click that is not an orbit drag (pointing.js tells them apart) is a ray from the camera; the first solid of the
 * room it meets decides what it means (pointing.js resolvePick): a place on the floor in the WORLD's frame goes to
 * POST /api/neuro/gothere, furniture or a wall is "not floor", the duck itself is nothing. The duck walks there by
 * itself and may be distracted (neuro/errand.py); the page only follows what the telemetry's `errand` says: a small
 * marker in token colours stays on the spot until the errand ends (or 30 s), and the one line under the command row
 * says how it is going. Ignored while remote mode (remote.js) has the legs, and while the prototype's mode is off. */

const REFUSAL_WORDS = ["reason_zh", "error"]; // the fields of a refused answer that hold the server's own words
const POINT_WORDS = {
  hint: "page.point.hint_beta",
  hintRemote: "page.point.hint_remote",
  sent: "page.point.sent",
  walking: "page.point.walking",
  arrived: "page.point.arrived",
  gaveUp: "page.point.gave_up",
  gaveUpPlain: "page.point.gave_up.plain",
  notFloor: "page.point.notfloor",
  failed: "page.neuro.why",
};
// the duck's reason for giving up (errand.REASONS) → its words, spelled out so the page's key checks read them
const POINT_WHY = {
  stuck: "page.point.why.stuck",
  timeout: "page.point.why.timeout",
  asleep: "page.point.why.asleep",
  afraid: "page.point.why.afraid",
  low_battery: "page.point.why.low_battery",
  docking: "page.point.why.docking",
  busy: "page.point.why.busy",
  lifted: "page.point.why.lifted",
  remote: "page.point.why.remote",
  stopped: "page.point.why.stopped",
};
const pointing = {
  track: makeErrandTrack(),
  clicks: makeClickTracker(),
  busy: false, // a click is waiting for the server's answer
  marker: null, // the small thing on the floor (built with the scene)
  line: document.getElementById("cmdPoint"),
  shown: "",
  motionOk: !(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches),
};

const remoteHasLegs = () => !!(window.duckRemote && window.duckRemote.mode !== "idle");

function buildPointMarker(scene) {
  // a ring and a dot lying on the floor, about 12 cm across: the duck is 20 cm long
  const group = new THREE.Group();
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.045, 0.062, 40),
    new THREE.MeshBasicMaterial({ color: token("--orange", "#ff7a2f"), side: THREE.DoubleSide, depthWrite: false }),
  );
  const dot = new THREE.Mesh(
    new THREE.CircleGeometry(0.014, 20),
    new THREE.MeshBasicMaterial({ color: token("--magenta", "#ff2fa8"), side: THREE.DoubleSide, depthWrite: false }),
  );
  ring.rotation.x = dot.rotation.x = -Math.PI / 2;
  ring.renderOrder = dot.renderOrder = 4;
  group.add(ring, dot);
  group.visible = false;
  scene.add(group);
  pointing.marker = group;
}

function placePointMarker(point, height) {
  if (!pointing.marker) return;
  // the world's (x, y) on the scene's floor: pointing.js's one transform
  const [x, z] = sceneOfWorld(point.x_m, point.y_m);
  pointing.marker.position.set(x, height + 0.008, z);
}

// the first solid of the room a ray meets, and the duck if it meets that first
function pickSolids(raycaster) {
  const home = view.room.home;
  const solids = [];
  if (home) {
    // (a wall faded because it stands between the camera and the duck is see-through: a click goes past it)
    const seeThrough = new Set(home.walls.filter((w) => w.faded || w.opacity < 0.5).map((w) => w.mesh));
    for (const mesh of [...home.floors, ...home.meshes]) if (!seeThrough.has(mesh)) solids.push(mesh);
  } else if (view.room.fallback) {
    solids.push(view.room.fallback);
  }
  const shown = (object) => {
    for (let o = object; o; o = o.parent) if (!o.visible) return false;
    return true;
  };
  const kindOf = (object) => {
    for (let o = object; o; o = o.parent) if (o.userData && o.userData.kind) return o.userData.kind;
    return "thing";
  };
  let hit = null;
  for (const found of raycaster.intersectObjects(solids, false)) {
    const kind = kindOf(found.object);
    if (!shown(found.object) || OVERHEAD_KINDS.has(kind)) continue;
    hit = { distance: found.distance, x: found.point.x, y: found.point.y, z: found.point.z, kind };
    break;
  }
  let duck = null;
  if (view.trunk) {
    const found = raycaster.intersectObject(view.trunk, true).find((f) => shown(f.object));
    if (found) duck = { distance: found.distance };
  }
  return { hit, duck, drawn: solids.length > 0 };
}

async function sendGoThere(point, height) {
  pointing.busy = true;
  const track = pointing.track;
  try {
    const response = await fetch("/api/neuro/gothere", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ point }),
    });
    const data = await response.json();
    if (response.ok && data.ok && data.id != null) {
      placePointMarker(point, height);
      track.start(data.id, performance.now());
    } else {
      track.fail("refused", performance.now(), data); // (a refusal is a normal answer: its words are the server's)
    }
  } catch (error) {
    track.fail("failed", performance.now());
  } finally {
    pointing.busy = false;
  }
}

function floorClick(clientX, clientY, canvas, camera) {
  if (!view.neuro || !view.neuro.on || pointing.busy || remoteHasLegs()) return;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1), camera);
  const { origin, direction } = raycaster.ray;
  const { hit, duck, drawn } = pickSolids(raycaster);
  const pick = resolvePick([origin.x, origin.y, origin.z], [direction.x, direction.y, direction.z], hit, duck, drawn);
  if (pick.kind === "floor") sendGoThere(pick.point, pick.y);
  else if (pick.kind === "blocked") pointing.track.fail("notfloor", performance.now());
}

function setupPointing(canvas, camera) {
  buildPointMarker(view.scene);
  if (pointing.line) pointing.line.hidden = false;
  const touch = (event) => event.pointerType === "touch" || event.pointerType === "pen";
  canvas.addEventListener("pointerdown", (event) => {
    pointing.clicks.down(event.pointerId, event.clientX, event.clientY, event.timeStamp, {
      button: event.button,
      touch: touch(event),
      modifier: event.shiftKey || event.ctrlKey || event.metaKey || event.altKey,
    });
  });
  canvas.addEventListener("pointermove", (event) => pointing.clicks.move(event.pointerId, event.clientX, event.clientY));
  canvas.addEventListener("pointerup", (event) => {
    const click = pointing.clicks.up(event.pointerId, event.clientX, event.clientY, event.timeStamp);
    if (click) floorClick(click.x, click.y, canvas, camera);
  });
  canvas.addEventListener("pointercancel", (event) => pointing.clicks.cancel(event.pointerId));
}

// every frame: the marker (it breathes a little, unless motion is reduced) and the one line under the command row
function tickPointing(now) {
  const marker = pointing.marker;
  if (marker) {
    const on = pointing.track.marker(now);
    if (marker.visible !== on) marker.visible = on;
    if (on) marker.scale.setScalar(pointing.motionOk ? 1 + 0.12 * Math.sin(now / 260) : 1);
  }
  const el = pointing.line;
  if (!el || el.hidden) return;
  const line = pointing.track.line(now);
  const remote = remoteHasLegs();
  const key = `${line ? `${line.kind}|${line.seq}|${line.reason || ""}` : remote ? "remote" : "hint"}|${I18N ? I18N.lang : ""}`;
  if (key === pointing.shown) return;
  pointing.shown = key;
  const words = (name, params) => {
    if (I18N) I18N.text(el, name, params);
    else el.textContent = name;
  };
  el.dataset.state = line ? line.state : "hint";
  if (!line) words(remote ? POINT_WORDS.hintRemote : POINT_WORDS.hint);
  else if (line.kind === "sent") words(POINT_WORDS.sent);
  else if (line.kind === "walking") words(POINT_WORDS.walking);
  else if (line.kind === "arrived") words(POINT_WORDS.arrived);
  else if (line.kind === "gave_up") {
    if (POINT_WHY[line.reason]) words(POINT_WORDS.gaveUp, { why: T(POINT_WHY[line.reason]) });
    else words(POINT_WORDS.gaveUpPlain);
  } else if (line.kind === "notfloor") words(POINT_WORDS.notFloor);
  else if (line.kind === "refused") {
    const name = line.answer && REFUSAL_WORDS.find((field) => line.answer[field]);
    if (name && I18N) I18N.show(el, line.answer, name);
    else if (name) el.textContent = line.answer[name];
    else words(POINT_WORDS.failed);
  } else words(POINT_WORDS.failed);
}

// the telemetry says how the errand goes; a new duck or a restarted server starts the errands over
window.addEventListener("duck:telemetry", (event) => {
  const errand = event.detail && event.detail.errand;
  if (errand) pointing.track.update(errand, performance.now());
});
window.addEventListener("duck:reset", () => {
  pointing.track.clear();
  pointing.shown = "";
});

/* ---------------- live physics (official robotd --sim + MuJoCo) ---------------- */

const liveEls = {
  box: document.getElementById("duck3dLive"),
  toggle: document.getElementById("liveToggle"),
  status: document.getElementById("liveStatus"),
  reset: document.getElementById("liveReset"),
  demo: document.getElementById("liveDemo"),
  upright: document.getElementById("liveUpright"),
};

function liveText(info) {
  if (!info.enabled) return "";
  // (plain words; the program's own text is in the line's tooltip, see showLive)
  if (info.error) return T("page.duck3d.liveText.9a7698", { error_zh: (info.error_zh && F(info, "error_zh")) || info.error });
  if (info.starting || !info.available) return T("page.duck3d.liveText.34fa2a");
  const POLICY = { sit: "page.duck3d.liveText.2bb395", rise: "page.duck3d.liveText.6543ad", walk: "page.duck3d.liveText.e5dca5", stand: "page.duck3d.liveText.b52df7" };
  const policy = POLICY[info.policy] ? T(POLICY[info.policy]) : info.policy || "";
  const rt = info.realtime != null ? T("page.duck3d.liveText.80e047", { realtime: info.realtime }) : "";
  if (info.demo) return T("page.duck3d.liveText.403afa");
  if (info.held) return T("page.duck3d.liveText.438798");
  if (view.neuro) return `${info.fallen ? T("page.duck3d.liveText.85b23b") : T("page.duck3d.liveText.ed4d44")} · ${policy}${rt}`;
  if (!info.realtime_clock && !view.time.playing) return T("page.duck3d.liveText.5f8509");
  if (!info.realtime_clock) return T("page.duck3d.liveText.7217d5");
  const monitor = info.monitor || (info.frame && info.frame.monitor);
  const trust = T(monitor && monitor.trust === "suspect" ? "page.duck3d.liveText.94de65" : "page.duck3d.liveText.a732fe");
  return `${info.fallen ? T("page.duck3d.liveText.1b099d") : trust} · ${policy}${rt}`;
}

function showLive(info) {
  if (!liveEls.status) return;
  liveEls.status.textContent = liveText(info);
  liveEls.status.title = info.error || "";
}

// A world operation moved the physics body (x_place: the body started, or after a
// fast-forward): say why, since the drawn duck is then shown where it now is at once.
const PLACED_WHY_ZH = {
  start: "page.duck3d.PLACED_WHY_ZH.start",
  sync: "page.duck3d.PLACED_WHY_ZH.sync",
};
// Duck Core believes it walked into its nest (it did not see itself get there): only the
// shadow moves; the physics body never jumps (W05). A hand setting it down is different: the
// world's hand really carries the body there (HAND_ZH), a few seconds after the shadow.
const PLACED_ZH = "page.duck3d.PLACED_ZH";
const HAND_ZH = {
  pickup: "page.duck3d.HAND_ZH.pickup",
  place: "page.duck3d.HAND_ZH.place",
  upright: "page.duck3d.HAND_ZH.upright",
  cancel: "page.duck3d.HAND_ZH.cancel",
};
// Put in its nest with a dock in the home: the hand carries the body into the charging dock.
const HAND_DOCK_ZH = "page.duck3d.HAND_DOCK_ZH";

function noteTruth(frame) {
  const live = view.live;
  const truth = frame && frame.truth;
  if (!truth) return;
  if (live.placedSeen !== undefined && truth.placed_at !== live.placedSeen && PLACED_WHY_ZH[truth.placed_why]) {
    caption(T(PLACED_WHY_ZH[truth.placed_why]));
  }
  live.placedSeen = truth.placed_at;
  if (trailPush(view.trail, { x: truth.x_mm / 1000, y: truth.y_mm / 1000, a: truth.placed_at }, Date.now())) view.trailDirty = true;
}

// The world (/api/body's `world`): the home it runs, the hand, the family's bodies, the toys,
// the nest's contact; a new home to draw; what the hand was last asked to do, in words.
function noteWorld(info) {
  const world = (info && info.world) || null;
  view.world = world;
  const room = view.room;
  if (info && info.home_json) loadHome(info.home_json);
  const home = world && world.home;
  room.physical = !!(world && world.physical && home && home.fingerprint === room.fingerprint);
  if (world && world.frame && !room.home) room.frame = world.frame;
  if (room.home) setWorld(room.home, world || {}, view.nestCharging);
  const hand = world && world.hand_note;
  if (hand && hand.at_ms !== view.live.handSeen) {
    // (a put-down where the body already is needs no hand: nothing to say)
    if (view.live.handSeen !== undefined && !hand.skipped) {
      const docked = hand.do === "place" && hand.to === "nest" && room.home && room.home.nest && room.home.nest.kind === "dock";
      const why = FS(hand, "why_zh").join(T("fmt.sep.semicolon"));
      const said = hand.ok === false ? why : docked ? T(HAND_DOCK_ZH) : HAND_ZH[hand.do] ? T(HAND_ZH[hand.do]) : "";
      if (said) caption(hand.ok !== false && hand.why_zh && hand.why_zh.length ? T("page.duck3d.noteWorld.268b2f", { said, why_zh: why }) : said);
    }
    view.live.handSeen = hand.at_ms;
  } else if (!hand && view.live.handSeen === undefined) view.live.handSeen = null;
  const placed = world && world.placed;
  if (placed && placed.at_ms !== view.live.placedNoteSeen) {
    if (view.live.placedNoteSeen !== undefined && placed.why_zh && placed.why_zh.length) caption(FS(placed, "why_zh").join(T("fmt.sep.semicolon")), true);
    view.live.placedNoteSeen = placed.at_ms;
  }
  const demo = world && world.demo;
  if (demo && !demo.running && demo.at_ms && demo.at_ms !== view.live.demoSeen) {
    if (view.live.demoSeen !== undefined) {
      const moved = demo.moved_mm != null ? T("page.duck3d.noteWorld.b6c3cd", { moved_mm: Math.round(demo.moved_mm / 10) }) : "";
      caption(
        T("page.duck3d.noteWorld.5ed99b", {
          demo_zh: (demo.zh && F(demo, "zh")) || T("page.duck3d.noteWorld.31d93d"),
          ok: T(demo.ok ? "page.duck3d.noteWorld.55d92d" : "page.duck3d.noteWorld.a2cd76"),
          moved,
          why_zh: demo.why_zh && demo.why_zh.length ? T("page.duck3d.noteWorld.b12003", { why_zh: FS(demo, "why_zh").join(T("fmt.sep.semicolon")) }) : "",
        }),
      );
    }
    view.live.demoSeen = demo.at_ms;
  } else if (!demo && view.live.demoSeen === undefined) view.live.demoSeen = null;
}

// Every body sample of the answer (`frames`, compact, `frame_joints` their order) into the
// page's buffer, and the playing clock told when the newest one was read. A server without
// frames (an older one) leaves the buffer empty: the view then eases toward each frame instead.
function takeFrames(info) {
  const live = view.live;
  const raw = Array.isArray(info.frames) ? info.frames : null;
  if (!raw || !info.available) {
    if (!info.available) {
      live.frames = [];
      live.lastT = null;
    }
    return;
  }
  const names = info.frame_joints || [];
  // (prototype mode: the samples carry the beak and the calls; the bubble and the sound then
  // come with the sample that carried them, not when the brain's report arrives)
  live.frameCalls = !!info.frame_calls;
  if (raw.length) {
    live.frames = mergeFrames(live.frames, raw.map((f) => unpackFrame(f, names)));
    const newest = live.frames[live.frames.length - 1];
    live.lastT = newest.t;
    playoutArrive(live.play, newest.w, performance.now() / 1000);
    // (how many samples a second the server reads: the last second of them)
    const span = live.frames.filter((f) => newest.w - f.w <= 1.0);
    live.sampleHz = span.length > 1 ? (span.length - 1) / Math.max(1e-3, newest.w - span[0].w) : null;
  }
}

// The truth to draw now: the body sample at this moment of the playing clock (between two true
// samples), or null when there are none (an older server: the caller eases toward the frame).
function playFrames(dt, realNow) {
  const live = view.live;
  if (!live.frames.length) return null;
  const at = playoutTime(live.play, realNow / 1000, dt);
  if (at == null) return null;
  // the calls on the samples played since the last draw (none replayed after a long pause)
  if (live.playedW != null && at - live.playedW < 0.5) for (const call of callsBetween(live.frames, live.playedW, at)) voiceSampleCall(call);
  live.playedW = at;
  const { pose, state } = sampleFrames(live.frames, at);
  live.playState = state;
  if (!pose) return null;
  const placed = view.external && view.external.truth ? view.external.truth.placed_at ?? null : null;
  const drawn = framePose(pose, placed);
  live.played = drawn;
  return drawn;
}

async function pollLive() {
  const live = view.live;
  if (live.busy) return;
  live.busy = true;
  try {
    // (since_t: every body sample after the newest this page has — all of them get drawn)
    const since = live.lastT == null ? "" : String(live.lastT);
    const info = await fetch(`/api/body?home=${encodeURIComponent(view.room.fingerprint || "")}&since_t=${since}`, { cache: "no-store" }).then((r) => r.json());
    live.fails = 0;
    live.info = info;
    live.enabled = !!info.enabled;
    if (liveEls.box) liveEls.box.hidden = !live.enabled;
    showLive(info);
    takeFrames(info);
    // Physics runs in real time (and holds still while time is paused): while the duck's
    // clock is fast-forwarded the view plays Duck Core's own record of what happened
    // instead, and says so. The frame is kept whole: the solid duck is drawn from its truth.
    live.clockRealtime = !!info.realtime_clock;
    const shown = live.clockRealtime || !view.time.playing;
    if (live.enabled && live.on && shown && info.available && info.frame) {
      setExternalPose(info.frame);
      noteTruth(info.frame);
    } else {
      setExternalPose(null);
    }
    noteWorld(info);
    demoFrame(info);
    // The duck-eye window and the 「物理不可信」 badge (app.js) read the same answer.
    window.dispatchEvent(new CustomEvent("duck:body", { detail: info }));
  } catch (error) {
    // The page keeps animating on its own; the physics frames it had are not "now" any
    // more (the server is gone or restarting), so they stop being shown.
    live.fails = (live.fails || 0) + 1;
    if (live.fails >= 2) {
      setExternalPose(null);
      if (liveEls.status && live.enabled) liveEls.status.textContent = T("page.duck3d.pollLive");
    }
  } finally {
    live.busy = false;
  }
}

function startLive() {
  const physicsHead = document.getElementById("physicsHead");
  if (physicsHead) {
    view.live.physicsHead = physicsHead.checked;
    physicsHead.addEventListener("change", () => {
      view.live.physicsHead = physicsHead.checked;
    });
  }
  if (liveEls.toggle) {
    liveEls.toggle.addEventListener("change", () => {
      view.live.on = liveEls.toggle.checked;
      if (!view.live.on) setExternalPose(null);
    });
  }
  if (liveEls.demo) {
    liveEls.demo.addEventListener("click", async () => {
      liveEls.demo.disabled = true;
      caption(T("page.duck3d.startLive.e716d5"));
      try {
        await fetch("/api/call", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ method: "body_demo" }),
        });
      } finally {
        setTimeout(() => (liveEls.demo.disabled = false), 8000);
      }
    });
  }
  if (liveEls.upright) {
    // D2: the everyday recovery after a physics fall — the world's hand, not a new body.
    liveEls.upright.addEventListener("click", async () => {
      liveEls.upright.disabled = true;
      caption(T(view.neuro ? "page.duck3d.startLive.ce8d20" : "page.duck3d.startLive.d77402"));
      try {
        const data = await fetch("/api/call", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ method: "body_upright" }),
        }).then((r) => r.json());
        if (data && !data.ok) caption(T("page.duck3d.startLive.2de5ad", { error: (data.error && F(data, "error")) || T("page.duck3d.startLive.04846d") }));
        // the server says whether the duck feels it (the neural duck does, Duck Core does not)
        else if (data && data.result && data.result.note) caption(F(data.result, "note"));
      } catch (error) {
        caption(T("page.duck3d.startLive.f9da4a"));
      } finally {
        setTimeout(() => (liveEls.upright.disabled = false), 2500);
      }
    });
  }
  if (liveEls.reset) {
    liveEls.reset.addEventListener("click", async () => {
      liveEls.reset.disabled = true;
      caption(T("page.duck3d.startLive.ec0487"));
      try {
        await fetch("/api/call", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ method: "body_reset" }),
        });
      } finally {
        setTimeout(() => (liveEls.reset.disabled = false), 3000);
      }
    });
  }
  // (While the server does not answer: back off instead of asking 17 times a second.)
  const scheduleLive = () =>
    setTimeout(async () => {
      await pollLive();
      scheduleLive();
    }, view.live.fails ? Math.min(IDLE_POLL_MS, 500 * view.live.fails) : view.live.enabled ? LIVE_POLL_MS : IDLE_POLL_MS);
  // The first answer decides the pace: every frame with live physics, now and then without.
  pollLive().then(scheduleLive);
}

// "示範走一段": the physics body walks by itself, outside Duck Core. The solid duck shows it
// walking (it is the physics truth) and stays where the walk ended: Duck Core never knew.
function demoFrame(info) {
  const live = view.live;
  const on = !!(info && info.demo && info.frame && live.on);
  if (on && !live.demo) {
    live.demo = true;
    const show = info.world && info.world.demo && info.world.demo.running;
    const eye = show && info.world.demo.eye;
    caption(T(eye ? "page.duck3d.demoFrame.4951c2" : show ? "page.duck3d.demoFrame.7aac8e" : "page.duck3d.demoFrame.620151"));
  } else if (!on && live.demo) {
    live.demo = null;
    // 「用眼睛做」 ends with the duck's own report: keep it on the caption (not the generic end)
    const done = info && info.world && info.world.demo;
    if (done && !done.running && done.eye && done.why_zh && done.why_zh.length) {
      caption(
        T("page.duck3d.demoFrame.f504ad", {
          done_zh: (done.zh && F(done, "zh")) || T("page.duck3d.demoFrame.d0e7fb"),
          why_zh: FS(done, "why_zh").join(T("fmt.sep.semicolon")),
        }),
      );
      return;
    }
    caption(
      T(
        info && info.frame && "truth" in info.frame
          ? "page.duck3d.demoFrame.a841bb"
          : "page.duck3d.demoFrame.6ad94d", // (an older server sends no truth)
      ),
    );
  }
}

/* ---------------- poses ---------------- */

function legsOf(keyframe) {
  const legs = {};
  for (const [name, value] of Object.entries(keyframe || {})) if (!HEAD.includes(name)) legs[name] = value;
  return legs;
}

function makePoses(model) {
  const keyframes = model.keyframes || {};
  const heights = model.trunk_height || {};
  const sit = { legs: legsOf(keyframes.SIT), z: heights.SIT ?? 0.07, roll: 0 };
  const stand = { legs: legsOf(keyframes.STAND || keyframes.SIT), z: heights.STAND ?? 0.12, roll: 0 };
  const legs = {};
  for (const name of new Set([...Object.keys(sit.legs), ...Object.keys(stand.legs)])) {
    legs[name] = 0.5 * ((sit.legs[name] || 0) + (stand.legs[name] || 0));
  }
  // On its side (rolled about the forward axis); its height is measured once the meshes load.
  const fallen = { legs, z: sit.z, roll: Math.PI / 2 };
  return { sit, stand, fallen };
}

function blendPose(a, b, u) {
  const s = u * u * (3 - 2 * u);
  const legs = {};
  for (const name of new Set([...Object.keys(a.legs), ...Object.keys(b.legs)])) {
    legs[name] = (a.legs[name] || 0) + ((b.legs[name] || 0) - (a.legs[name] || 0)) * s;
  }
  return { legs, z: a.z + (b.z - a.z) * s, roll: a.roll + (b.roll - a.roll) * s };
}

// q = yaw about the world's up axis, then roll about the duck's own forward axis.
function yawRollQuat(yaw, roll) {
  const cy = Math.cos(yaw / 2);
  const sy = Math.sin(yaw / 2);
  const cr = Math.cos(roll / 2);
  const sr = Math.sin(roll / 2);
  return [cy * cr, cy * sr, sy * sr, sy * cr];
}

function headJoints() {
  const home = view.clips.home;
  const joints = {};
  for (const name of HEAD) joints[name] = view.rendered[name] ?? home[name] ?? 0;
  return joints;
}

/** Put the model in a pose: {joints, trunk: [x, y, z], quat: [w, x, y, z], mouth}. */
function applyPose(pose) {
  if (!view.trunk || !pose) return;
  const joints = pose.joints || {};
  for (const [name, joint] of Object.entries(view.joints)) {
    if (!(name in joints)) continue;
    const [lo, hi] = joint.range || [-Math.PI, Math.PI];
    joint.group.quaternion.setFromAxisAngle(joint.axis, Math.min(hi, Math.max(lo, joints[name])));
  }
  if (pose.trunk) view.trunk.position.set(pose.trunk[0], pose.trunk[1], pose.trunk[2]);
  if (pose.quat) view.trunk.quaternion.set(pose.quat[1], pose.quat[2], pose.quat[3], pose.quat[0]);
  if (pose.mouth != null && view.jawPivot && view.jawPivot.axis) {
    const travel = (view.clips && view.clips.mouth_travel) || DEFAULT_MOUTH_TRAVEL;
    view.jawPivot.group.quaternion.setFromAxisAngle(view.jawPivot.axis, pose.mouth * travel);
  }
}

/** A pose from outside (e.g. live physics), or null to go back to the animation. */
function setExternalPose(pose) {
  view.external = pose || null;
}

function measurePoses(world) {
  // How high the trunk sits when the duck lies on its side, and how much room standing needs.
  const fallen = view.poses.fallen;
  applyPose({ joints: { ...fallen.legs, ...headJoints() }, trunk: [0, 0, 0], quat: yawRollQuat(0, fallen.roll) });
  world.updateMatrixWorld(true);
  const lying = new THREE.Box3().setFromObject(world);
  if (!lying.isEmpty()) fallen.z = -lying.min.y + 0.002;
  const stand = view.poses.stand;
  applyPose({ joints: { ...stand.legs, ...headJoints() }, trunk: [0, 0, stand.z], quat: [1, 0, 0, 0] });
  world.updateMatrixWorld(true);
  const tall = new THREE.Box3().setFromObject(world);
  if (!tall.isEmpty()) view.fitBounds = tall;
}

// Fit the standing duck (plus head room for looking up) into the view. The first fit uses
// the official monitor's three-quarter angle; later ones keep the user's angle.
function refit() {
  const { camera, controls, fitBounds } = view;
  if (!camera || !fitBounds) return;
  if (view.closeup && view.closeup.on) {
    // 「近看小鴨」 keeps its own distance (a resize or the ⛶ button does not undo it)
    controls.minDistance = CLOSE_MIN;
    controls.maxDistance = CLOSE_MAX;
    return;
  }
  const size = fitBounds.getSize(new THREE.Vector3());
  const center = fitBounds.getCenter(new THREE.Vector3());
  // Room to turn: the footprint's diagonal, since the duck may face any way after a walk.
  const radius = 0.5 * Math.max(size.y * 1.12, 0.9 * Math.hypot(size.x, size.z));
  const vfov = THREE.MathUtils.degToRad(camera.fov);
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
  // Wider than the duck alone, so a few steps of floor around it are in view.
  // Close enough to see a head tilt; the wheel zooms out to the whole room.
  const distance = (radius / Math.sin(Math.min(vfov, hfov) / 2)) * 1.25;
  const dir = view.fitted
    ? camera.position.clone().sub(controls.target).normalize()
    : new THREE.Vector3(
        Math.cos(VIEW_ELEVATION) * Math.cos(VIEW_AZIMUTH),
        Math.sin(VIEW_ELEVATION),
        -Math.cos(VIEW_ELEVATION) * Math.sin(VIEW_AZIMUTH),
      );
  controls.target.copy(center).add(new THREE.Vector3(0, size.y * 0.04, 0));
  if (view.follow) controls.target.add(new THREE.Vector3(view.follow.x, 0, view.follow.z));
  camera.position.copy(controls.target).addScaledVector(dir, distance);
  controls.minDistance = distance * 0.4;
  // Far enough to see the whole home.
  controls.maxDistance = Math.max(distance * 3, 7, view.room.home ? homeSpan(view.room.home) * 0.9 : 0);
  controls.update();
  view.fitted = true;
}

function setupJawPivot() {
  const jaw = view.jawPivot;
  if (!jaw) return;
  // The MJCF jaw is a fixed geom while the real mouth is a servo, so the server reads the
  // hinge pin off the jaw mesh (head body frame: face −z, +x up, +y the duck's left).
  const hinge = view.model.jaw_hinge;
  if (hinge) {
    const pin = new THREE.Vector3(...hinge.pos);
    jaw.group.position.copy(pin);
    for (const holder of jaw.meshes) holder.position.sub(pin);
    jaw.axis = new THREE.Vector3(...hinge.axis).normalize();
    return;
  }
  // Fallback: hinge at the jaw's upper rear edge about +y (positive opens down).
  const bounds = new THREE.Box3();
  for (const holder of jaw.meshes) {
    holder.updateMatrix();
    const geo = holder.children[0]?.geometry;
    if (!geo) continue;
    geo.computeBoundingBox();
    const b = geo.boundingBox.clone().applyMatrix4(holder.matrix);
    bounds.union(b);
  }
  if (bounds.isEmpty()) return;
  const pivot = bounds.getCenter(new THREE.Vector3());
  pivot.z = bounds.max.z;
  pivot.x = bounds.max.x;
  jaw.group.position.copy(pivot);
  for (const holder of jaw.meshes) holder.position.sub(pivot);
  jaw.axis = new THREE.Vector3(0, 1, 0);
}

/* ---------------- head clips ---------------- */

function frameAt(frames, ms) {
  if (ms <= frames[0][0]) return frames[0][1];
  for (let i = 1; i < frames.length; i++) {
    const [t1, f1] = frames[i];
    const [t0, f0] = frames[i - 1];
    if (ms <= t1) {
      const u = (ms - t0) / Math.max(1, t1 - t0);
      const s = u * u * (3 - 2 * u);
      const out = {};
      for (const key of new Set([...Object.keys(f0), ...Object.keys(f1)])) out[key] = (f0[key] || 0) + ((f1[key] || 0) - (f0[key] || 0)) * s;
      return out;
    }
  }
  return frames[frames.length - 1][1];
}

function startClip(motion, now) {
  const frames = view.clips && view.clips.clips[motion.clip];
  if (!frames) {
    voiceCall(motion.calls); // (a move this view has no frames for still makes its sound)
    return;
  }
  const tempo = Math.max(0.5, (motion.tempo || 10000) / 10000);
  const duration = frames[frames.length - 1][0] / tempo;
  view.clip = { name: motion.clip, frames, start: now, duration, tempo, amplitude: (motion.amplitude || 10000) / 10000, weak: !!motion.weak };
  caption(motion.caption);
  if (motion.sounds && motion.sounds.length) say(motion.sounds.map((tag) => (SOUND_TAG[tag] ? T(SOUND_TAG[tag]) : tag)).join(T("fmt.sep.dot")), now);
  // Its sound, in the duck's own voice, as the move starts (a wheee is held while it lasts).
  voiceCall(motion.calls, duration);
}

// The duck's own voice (voice.js) for one of Duck Core's calls — from the 3D view only while it
// is the duck on screen (the 2D duck's calls are app.js's). `holdMs`: how long the move lasts.
function voiceCall(calls, holdMs = 0) {
  if (!calls || !calls.length || !window.duckVoice || !document.body.classList.contains("has3d")) return;
  window.duckVoice.call(calls, { holdMs });
}

// The duck's sound words on the bubble (its babble, as a person would write it down).
const SOUND_TAG = {
  chirp: "page.duck3d.startClip.2ec799",
  inquire: "page.duck3d.startClip.f9fb73",
  coo: "page.duck3d.startClip.3fb5b3",
  greet: "page.duck3d.startClip.9ac5f6",
  wheee: "page.duck3d.startClip.03aa74",
};

function say(text) {
  if (!bubbleEl || !text) return;
  bubbleEl.textContent = text;
  view.bubbleUntil = performance.now() + 2200;
}

// A call as the duck voices it: its words in the bubble, and its sound in the duck's own official
// voice (voice.js: the tag's bank, a random variant, robotd's one-voice rule), at the volume the
// viewer chose, or silently when muted. Played as the body sample that carried it is drawn — the
// sample whose robot.mouth command opened the beak for it — so a sound not ready by CALL_LATE_MS
// is dropped rather than heard after its beak (voice:neuro).
const CALL_ZH = {
  chirp: "page.duck3d.CALL_ZH.chirp",
  inquire: "page.duck3d.CALL_ZH.inquire",
  coo: "page.duck3d.CALL_ZH.coo",
  greet: "page.duck3d.CALL_ZH.greet",
  wheee: "page.duck3d.CALL_ZH.wheee",
  alarm: "page.duck3d.CALL_ZH.alarm",
};
const CALL_LATE_MS = 350;
function voiceSampleCall(call) {
  if (!call || !call.tag) return;
  // (an emotion's sound, emo_<name>: the page's 「動作試玩」 note names it; the bubble keeps its words)
  if (!String(call.tag).startsWith("emo_")) say(callWords(call));
  soundCall(call);
}
// A call's words for the bubble: a learned phrase or the song (the voice lineage) is said syllable by syllable.
function callWords(call) {
  if (Array.isArray(call.parts) && call.parts.length) {
    const word = (tag) => (CALL_ZH[tag] ? T(CALL_ZH[tag]) : tag);
    return call.parts.slice(0, 6).map(word).join(T("fmt.sep.babble"));
  }
  return CALL_ZH[call.tag] ? T(CALL_ZH[call.tag]) : call.tag;
}
function soundCall(call) {
  // (`lib`: the voice library the call comes from; the voice asks the server again when it is a newer one)
  if (window.duckVoice) window.duckVoice.call([String(call.tag)], { maxWaitMs: CALL_LATE_MS, lib: call.lib });
}

function fx(symbol) {
  if (!fxEl) return;
  const el = document.createElement("span");
  el.textContent = symbol;
  el.style.left = `${40 + Math.random() * 20}%`;
  fxEl.appendChild(el);
  setTimeout(() => el.remove(), 1600);
}

function headFrame(dt, now) {
  const home = view.clips.home;
  const env = view.clips.envelope;
  const side = view.clips.side || 1;
  const sign = view.clips.sign || DEFAULT_SIGN;

  // Clip frame (Duck Core's intent).
  let delta = {};
  if (view.clip) {
    const elapsed = (now - view.clip.start) * view.clip.tempo;
    if (elapsed >= view.clip.frames[view.clip.frames.length - 1][0] && !HOLDS.has(view.clip.name)) {
      view.clip = null;
      if (view.queue.length) startClip(view.queue.shift(), now);
    } else {
      delta = frameAt(view.clip.frames, Math.min(elapsed, view.clip.frames[view.clip.frames.length - 1][0]));
      for (const key of Object.keys(delta)) if (key !== "m") delta[key] *= view.clip.amplitude;
    }
  } else if (view.queue.length) {
    startClip(view.queue.shift(), now);
  }
  // Dozing or off: the sleep pose — except while a move Duck Core just did is playing (a
  // stir in its sleep, a greeting or a pat: hiding those looked broken).
  const stirring = view.clip && !HOLDS.has(view.clip.name);
  if ((view.napping || !view.awake) && !(stirring && view.awake)) {
    const pose = (view.clips && view.clips.nap_pose) || { np: 0.8, hp: -0.7 };
    delta = { np: pose.np * view.sleepAmp, hp: pose.hp * view.sleepAmp };
  }
  if (view.rest || now < view.holdUntil) delta = { np: 0.5, hp: -0.4 };

  // Breathing layer: a small neck motion awake; asleep a slow, deep breath you can see
  // (±2°, with the body rising and falling a few millimetres); switched off, nothing.
  if (view.awake && !view.napping && !view.rest) {
    if (now > view.drift.next) {
      view.drift.yaw = (Math.random() * 2 - 1) * 0.05;
      view.drift.next = now + 3000 + Math.random() * 3000;
    }
  } else {
    view.drift.yaw = 0;
  }
  const breathe = !view.awake
    ? 0
    : view.napping
      ? Math.sin((now / 1000) * Math.PI * 2 * 0.2) * 0.035
      : Math.sin((now / 1000) * Math.PI * 0.5) * 0.015;

  // Same mapping as duckmind.duckcore.content.head_targets (HOME + sign · envelope · delta).
  const target = {
    neck_pitch: home.neck_pitch + sign.neck_pitch * env.neck_pitch * (delta.np || 0) + breathe,
    head_pitch: home.head_pitch + sign.head_pitch * env.head_pitch * (delta.hp || 0),
    head_yaw: home.head_yaw + sign.head_yaw * (env.head_yaw * (delta.hy || 0) * side + view.drift.yaw),
    head_roll: home.head_roll + sign.head_roll * env.head_roll * (delta.hr || 0) * side,
  };
  // robotd smooths head targets (~1.78 Hz first-order).
  const alpha = 1 - Math.exp(-2 * Math.PI * view.clips.cutoff_hz * dt);
  for (const name of HEAD) {
    const current = view.rendered[name] ?? target[name];
    view.rendered[name] = current + (target[name] - current) * alpha;
  }
  // Mouth: an opening fraction of the servo's travel, as robot.mouth's `open` is — the clip's,
  // or the call being heard (voice.js: its loudness right now), whichever is wider.
  const voiceOpen = window.duckVoice ? window.duckVoice.mouth() : 0;
  let mouthTarget = Math.max(Math.min(1, Math.max(0, delta.m || 0)), voiceOpen) * env.mouth;
  if (view.neuro && view.live.drawnMouth != null) {
    // 神經鴨原型 with the body's own samples: the beak its driver commanded at that very sample,
    // on the same clock as the head (the physics model has no mouth: this is the command).
    view.mouth = Math.min(1, Math.max(0, view.live.drawnMouth));
    return;
  }
  if (view.neuro) {
    // 神經鴨原型: the beak is the neural duck's own robot.mouth (its 10 Hz peaks, replayed a
    // moment behind), not Duck Core's idle mouth.
    const q = view.neuroMouth || [];
    const at = performance.now();
    while (q.length > 1 && q[1].at <= at) q.shift();
    mouthTarget = q.length && q[0].at <= at ? q[0].open : 0;
    // (voice:neuro) no beak from its body at all (the made-up stub, an older server): it opens
    // with the call being heard (voice.js), as far as a middle-level call opens it (its motor:
    // 0.35 + 0.5 × level of the servo's travel)
    if (view.neuroMouthLast == null && window.duckVoice) mouthTarget = Math.max(mouthTarget, 0.6 * window.duckVoice.mouth());
    view.mouth += (mouthTarget - view.mouth) * Math.min(1, dt * 22);
    return;
  }
  view.mouth += (mouthTarget - view.mouth) * Math.min(1, dt * 14);
}

/* ---------------- body: posture, walks, falls ---------------- */

function posturePose(name) {
  return view.poses[name === "walk" ? "stand" : name] || view.poses.sit;
}

function currentPose(now) {
  const b = view.body;
  if (b.to && now < b.t0 + b.dur) return blendPose(b.from, posturePose(b.to), (now - b.t0) / b.dur);
  if (b.to) {
    b.shown = b.to;
    b.to = null;
  }
  return posturePose(b.shown);
}

function heading() {
  return view.body.to || view.body.shown;
}

function transition(target, now, ms) {
  const b = view.body;
  const from = currentPose(now);
  b.from = { legs: { ...from.legs }, z: from.z, roll: from.roll };
  b.to = target === "walk" ? "stand" : target;
  b.t0 = now;
  b.dur = ms;
}

function yawRateOf(walk) {
  if (walk.yaw_rate != null) return walk.yaw_rate;
  return (walk.vyaw || 0) * YAW_RATIO;
}

// Too far behind (time was fast-forwarded): apply an old action at once instead of showing it.
// Walk forward (+x of the duck) and turn back before the room's walls.
function travel(b, speed, dt) {
  const limit = ROOM_HALF - WALL_MARGIN;
  const nx = b.x + Math.cos(b.yaw) * speed * dt;
  const ny = b.y + Math.sin(b.yaw) * speed * dt;
  if (Math.abs(nx) > limit || Math.abs(ny) > limit) {
    b.yaw += Math.PI * 0.75; // it notices the wall and turns away
    return;
  }
  b.x = nx;
  b.y = ny;
}

// Where a walk has taken the duck after t seconds: Duck Core's path (constant ground speed
// and yaw rate from its `from` pose), ending exactly on its `to` pose.
function walkPose(walk, t) {
  const [x0, y0, h0] = walk.from;
  const v = walk.speed ?? 0;
  const w = walk.yawRate;
  const h = h0 + w * t;
  if (Math.abs(w) < 1e-6) return [x0 + v * t * Math.cos(h0), y0 + v * t * Math.sin(h0), h];
  return [x0 + (v / w) * (Math.sin(h) - Math.sin(h0)), y0 - (v / w) * (Math.cos(h) - Math.cos(h0)), h];
}

// ``pose`` [x, y] without a heading keeps the one it has (it sat down in its nest the way
// it walked in).
function placeAt(b, pose) {
  if (!pose) return;
  b.x = pose[0];
  b.y = pose[1];
  if (pose.length > 2 && pose[2] != null) b.yaw = pose[2];
}

function settle(item) {
  const b = view.body;
  if (item.note) return; // words only: nothing to apply
  if (item.walk && item.walk.to) placeAt(b, item.walk.to);
  else if (item.walk) {
    const seconds = item.walk.cut_after ?? item.walk.seconds ?? 0;
    const speed = item.walk.speed ?? 0;
    const rate = yawRateOf(item.walk);
    for (let t = 0; t < seconds; t += 0.1) {
      b.yaw += rate * 0.1;
      travel(b, speed, 0.1);
    }
  }
  else if (item.place) placeAt(b, item.place);
  else if (item.fall) {
    b.shown = "fallen";
    placeAt(b, item.at);
  }
  else if (item.posture) {
    b.shown = item.posture === "walk" ? "stand" : item.posture;
    if (item.posture === "fallen") placeAt(b, item.at); // v4_1: where it fell is on its posture
  }
  b.to = null;
}

// A hand or a fall cut the walk short. Within one feed batch the walk already carries the
// cut; across batches, shorten the walk being shown (or still queued) to where it stopped.
function cutWalk(cut, now) {
  const b = view.body;
  if (b.walk && !b.walk.from) {
    b.walk.end = now;
    return;
  }
  if (!cut) return;
  const queued = b.queue.find((item) => item.walk && item.walk.id === cut.walk);
  const walk = queued ? queued.walk : b.walk && b.walk.id === cut.walk ? b.walk : null;
  if (!walk || !walk.from || walk.cut_after != null) return;
  walk.cut_after = cut.after_s;
  walk.to = cut.pose;
  if (walk === b.walk) {
    walk.duration = Math.min(walk.duration, cut.after_s);
    walk.end = Math.min(walk.end, walk.start + cut.after_s * 1000);
  }
}

function busy(now) {
  const b = view.body;
  return (
    (b.to && now < b.t0 + b.dur) ||
    (b.walk && now < b.walk.end) ||
    (b.rock && now < b.rock.end) ||
    now < view.walkUntil
  );
}

// With nothing left to show, the view agrees with Duck Core's snapshot: posture, and the
// position whenever it is not mid-walk (resets, pickups, events the feed skipped).
function reconcile(now) {
  const b = view.body;
  const want = b.state === "walk" ? "stand" : b.state;
  if (want && want !== heading()) transition(want, now, want === "fallen" ? FALL_MS : POSE_MS);
  const p = b.position;
  if (p && b.state !== "walk") {
    const off = Math.hypot(p[0] - b.x, p[1] - b.y) > 0.01 || Math.abs(Math.atan2(Math.sin(p[2] - b.yaw), Math.cos(p[2] - b.yaw))) > 0.02;
    if (off) placeAt(b, p);
  }
}

function nextBodyAction(now) {
  const b = view.body;
  if (busy(now)) return;
  while (b.queue.length > QUEUE_MAX) settle(b.queue.shift());
  const item = b.queue.shift();
  if (!item) {
    if (!bodyPending()) reconcile(now);
    return;
  }
  if (item.note) {
    sayNote(item.note, item.kind);
    return;
  }
  caption(item.caption);
  if (item.place) {
    placeAt(b, item.place); // set down at its mat (or lifted off the floor)
    return;
  }
  if (item.righting) {
    if (heading() !== "fallen") transition("fallen", now, FALL_MS);
    b.rock = { start: now, end: now + RIGHTING_MS, ok: item.righting.ok };
    return;
  }
  if (item.walk) {
    if (heading() !== "stand") {
      b.queue.unshift(item); // it walks from standing: stand up first
      transition("stand", now, POSE_MS * 0.6);
      return;
    }
    const seconds = Math.max(0.5, Math.min(8, item.walk.cut_after ?? item.walk.seconds ?? 2));
    const walk = { ...item.walk, yawRate: yawRateOf(item.walk), start: now, end: now + seconds * 1000 };
    if (walk.from) {
      walk.duration = item.walk.cut_after ?? item.walk.seconds ?? seconds;
      placeAt(b, walk.from); // Duck Core's start pose (the same as the last walk's end)
    }
    b.walk = walk;
    return;
  }
  // Where it fell: v4 put it on body.fell, v4_1 on Duck Core's own duck.posture(fallen).
  if (item.fall || item.posture === "fallen") placeAt(b, item.at);
  const target = item.fall ? "fallen" : item.posture === "walk" ? "stand" : item.posture;
  if (target && target !== heading()) transition(target, now, target === "fallen" ? FALL_MS : POSE_MS);
}

function gait(legs, phase, amount) {
  for (const name of Object.keys(legs)) {
    const left = name.startsWith("left");
    const s = Math.sin(phase + (left ? 0 : Math.PI));
    if (name.endsWith("hip_pitch")) legs[name] += (left ? -1 : 1) * 0.28 * amount * s;
    if (name.endsWith("knee")) legs[name] += (left ? 1 : -1) * 0.35 * amount * Math.max(0, s);
    if (name.endsWith("ankle")) legs[name] += (left ? 1 : -1) * 0.12 * amount * s;
  }
}

function bodyFrame(dt, now) {
  const b = view.body;
  nextBodyAction(now);
  const replay = now < view.walkUntil; // gamepad play replay (illustrative)
  const walking = b.walk && now < b.walk.end;
  if (b.walk && !walking) {
    if (b.walk.to) placeAt(b, b.walk.to); // ends exactly where Duck Core says
    b.walk = null;
  }
  if (b.rock && now >= b.rock.end) b.rock = null;
  if (b.rock) {
    // Trying to roll back up: rock toward upright and fall back (a good try gets further;
    // the posture event that follows then sits it up).
    const u = (now - b.rock.start) / (b.rock.end - b.rock.start);
    const lift = Math.sin(Math.PI * Math.min(1, u * 1.15)) * (b.rock.ok ? 0.75 : 0.45);
    const rocking = Math.sin(u * Math.PI * 6) * 0.08 * (1 - u);
    const pose = currentPose(now);
    return { ...pose, roll: pose.roll * (1 - lift) + rocking, walking: false };
  }
  if (walking || replay) {
    const amount = walking && !(b.walk.vx > 0) ? 0.6 : 1; // turning in place: smaller steps
    const phase = (now / 1000) * Math.PI * 2 * GAIT_HZ;
    const legs = { ...view.poses.stand.legs };
    gait(legs, phase, amount);
    if (walking && b.walk.from) {
      // On the play clock, so a walk covers its full path whatever the playback speed.
      const t = Math.min(b.walk.duration, ((now - b.walk.start) / 1000) * (b.walk.duration / ((b.walk.end - b.walk.start) / 1000)));
      placeAt(b, walkPose(b.walk, t));
    } else if (walking) {
      const played = dt * view.rate; // the same distance whatever the playback speed
      b.yaw += b.walk.yawRate * played;
      travel(b, b.walk.speed ?? 0, played);
    }
    return { legs, z: view.poses.stand.z + Math.abs(Math.sin(phase)) * 0.004 * amount, roll: 0, walking: !!walking };
  }
  return { ...currentPose(now), walking: false };
}

// Every action Duck Core reports is shown, in order. When time runs faster than the view
// can play (fast-forward), the view plays its backlog faster instead of skipping it;
// positions still add up exactly, because walks cover their full distance either way.
function playbackRate() {
  const backlog = view.queue.length + view.body.queue.length;
  return Math.min(MAX_RATE, 1 + backlog * 0.75);
}

function clipZh(name) {
  const names = (view.clips && view.clips.names) || {};
  return names[name] ? F(names, name) : name;
}

// The duck's world keeps its own clock: its minute of the day, advancing between state
// updates while time runs in real time (fast-forward updates it every step anyway).
function clockMinute() {
  if (view.neuro && view.neuroDay) {
    // 神經鴨原型: the neural duck's own clock (its time of day runs at the real pace)
    return (view.neuroDay.minute + (performance.now() - view.neuroDay.at) / 60000) % 1440;
  }
  const c = view.clock;
  if (!c) return 12 * 60;
  const running = view.time && view.time.playing && view.time.realtime && !view.offline;
  const extra = running ? (performance.now() - c.at) / 60000 : 0;
  return (c.minute + extra) % 1440;
}

// Sunrise and sunset (minutes of the day) at the duck's home on that date, from the
// sun's declination and the equation of time — within a few minutes of an almanac.
function sunTimes(year, month, day) {
  const n = Math.round((Date.UTC(year, month - 1, day) - Date.UTC(year, 0, 0)) / 86400000);
  const g = (2 * Math.PI * (n - 1)) / 365;
  const decl =
    0.006918 - 0.399912 * Math.cos(g) + 0.070257 * Math.sin(g) - 0.006758 * Math.cos(2 * g) +
    0.000907 * Math.sin(2 * g) - 0.002697 * Math.cos(3 * g) + 0.00148 * Math.sin(3 * g);
  const eqt = 229.18 * (0.000075 + 0.001868 * Math.cos(g) - 0.032077 * Math.sin(g) -
    0.014615 * Math.cos(2 * g) - 0.040849 * Math.sin(2 * g));
  const lat = (HOME_LAT * Math.PI) / 180;
  const zenith = (90.833 * Math.PI) / 180; // the sun's rim at the horizon, with refraction
  const cosH = Math.cos(zenith) / (Math.cos(lat) * Math.cos(decl)) - Math.tan(lat) * Math.tan(decl);
  const half = (Math.acos(Math.min(1, Math.max(-1, cosH))) * 180) / Math.PI * 4;
  const noon = 720 - 4 * HOME_LON - eqt + HOME_UTC * 60;
  return { rise: noon - half, set: noon + half };
}

function smooth(x) {
  const t = Math.min(1, Math.max(0, x));
  return t * t * (3 - 2 * t);
}

// The light at the duck's time: the server's (world/lights.py, the one source the duck's-eye
// camera also uses), or — from an older server that sends none — the same rules here.
function lightTarget(m) {
  const s = view.serverLight;
  if (s && s.daylight != null) {
    return { daylight: s.daylight, glow: s.glow, lamps: s.lamps, elev: s.elevation, phase: s.phase };
  }
  const sun = (view.clock && view.clock.sun) || { rise: 360, set: 1110 };
  const dayLen = sun.set - sun.rise;
  const elev = m > sun.rise && m < sun.set ? Math.sin((Math.PI * (m - sun.rise)) / dayLen) : 0;
  // Civil twilight: the sky stays light for about half an hour past the horizon (rising in
  // the half hour before sunrise, fading in the half hour after sunset).
  const rim = m < sun.rise ? smooth(1 - (sun.rise - m) / 30) : m > sun.set ? smooth(1 - (m - sun.set) / 30) : 1;
  const daylight = Math.max(smooth(elev * 2.2), elev > 0 ? 0.12 : rim * 0.12);
  const glow = elev > 0 ? Math.exp(-(((elev - 0.12) / 0.13) ** 2)) : 0;
  const hour = m / 60;
  const lamps = daylight < 0.7 && hour >= 17 && hour < 23 ? 1 - daylight : 0;
  return { daylight, glow, lamps, elev, phase: elev > 0 ? (m - sun.rise) / dayLen : 0.5 };
}

// Light follows the duck's time of day: dawn, bright day, warm sunset, lamps in the
// evening, moonlight at night. Changes ease in so a fast-forward does not flicker.
function lightFrame(dt) {
  const m = clockMinute();
  const { daylight, glow, lamps, elev, phase } = lightTarget(m);
  const target = { daylight, glow, lamps };
  const k = Math.min(1, dt * 2);
  const cur = view.light || target;
  view.light = {
    daylight: cur.daylight + (target.daylight - cur.daylight) * k,
    glow: cur.glow + (target.glow - cur.glow) * k,
    lamps: cur.lamps + (target.lamps - cur.lamps) * k,
  };
  const L = view.light;
  if (view.hemi) {
    view.hemi.intensity = 0.28 + 0.97 * L.daylight + 0.25 * L.lamps;
    view.hemi.color.copy(NIGHT_LIGHT).lerp(DAY_LIGHT, Math.max(L.daylight, L.lamps * 0.8));
  }
  if (view.sun) {
    view.sun.intensity = 1.6 * L.daylight + 0.12;
    view.sun.color.copy(SUN_DAY).lerp(SUN_WARM, L.glow * 0.8);
    // The sun crosses the room from east (+x) to west (−x); the moon sits high.
    const f = view.follow || { x: 0, z: 0 };
    view.sun.position.set(f.x + Math.cos(Math.PI * phase) * 1.1, 0.35 + 1.1 * Math.max(elev, 0.35), f.z + 0.5);
  }
  if (view.fill) view.fill.intensity = 0.15 + 0.35 * L.daylight;
  if (view.lamp) view.lamp.intensity = 2.2 * L.lamps;
  if (view.scene && view.scene.background) {
    view.scene.background.copy(NIGHT_SKY).lerp(DAY_SKY, L.daylight).lerp(DUSK_SKY, L.glow * 0.55);
    if (view.scene.fog) view.scene.fog.color.copy(view.scene.background);
  }
  renderClock(m);
}

const clockEl = document.getElementById("duck3dClock");
let clockText = "";
function renderClock(m) {
  if (!clockEl || !view.clock) return;
  const c = view.clock;
  const hh = String(Math.floor(m / 60)).padStart(2, "0");
  const mm = String(Math.floor(m % 60)).padStart(2, "0");
  const t = view.time || {};
  const SPEED = { 1: "page.duck3d.renderClock.897bda", 5: "page.duck3d.renderClock.5c0a94", 15: "page.duck3d.renderClock.b0a122" };
  const mode = view.offline
    ? T("page.duck3d.renderClock.46e121")
    : view.neuro
    ? T("page.duck3d.renderClock.c71b00")
    : !t.playing
      ? T("page.duck3d.renderClock.783e33")
      : t.realtime
        ? T("page.duck3d.renderClock.968936")
        : T("page.duck3d.renderClock.8ea643", { speed: SPEED[t.speed] ? T(SPEED[t.speed]) : "" });
  const sun = c.sun || { rise: 360, set: 1110 };
  const icon = m >= sun.rise && m < sun.set ? "☀" : m >= 17 * 60 && m < 23 * 60 ? "🏠" : "🌙";
  // A duck that lives the family's day: where the day is at (the family asleep, out…).
  const phase = view.neuro ? view.neuro.who || "" : view.phaseZh && !view.offline ? F(view.phaseZh, "phase_zh") : "";
  // (繁體中文: the server's own weekday character; the other languages their own short name)
  const weekday = SOURCE_WORDS() ? c.weekday || T(WEEKDAY_ZH[c.day]) : I18N.weekday((c.day + 6) % 7);
  const when = view.neuro && view.neuroDay ? T("page.duck3d.renderClock.9ea917") : T("page.duck3d.renderClock.5f8ad2", { date: c.date, weekday });
  const text = `${when}${hh}:${mm} ${icon}|${mode}|${phase}`;
  if (text === clockText) return;
  clockText = text;
  const [main, sub, day] = text.split("|");
  const parts = [Object.assign(document.createElement("b"), { textContent: main }), Object.assign(document.createElement("span"), { textContent: sub })];
  if (day) parts.push(Object.assign(document.createElement("i"), { textContent: day }));
  clockEl.replaceChildren(...parts);
  clockEl.dataset.state = view.offline ? "offline" : view.neuro ? "realtime" : !t.playing ? "paused" : t.realtime ? "realtime" : "fast";
}

function step(dt, realNow) {
  if (!view.model || !view.clips) return;
  releasePending(realNow);
  if (!view.playNow) view.playNow = realNow;
  // Real time plays everything at its own speed, at the second it happens; only a
  // fast-forward backlog is played faster.
  view.rate = view.realtimeClock ? 1 : playbackRate();
  view.playNow = Math.max(view.playNow + dt * 1000 * view.rate, realNow);
  const now = view.playNow;
  lightFrame(dt);
  headFrame(dt, now);
  const body = bodyFrame(dt, now);
  // Lifted (a pickup's moment), or held up until someone puts it down (a duck with a nest
  // waits to be set back in its nest or on the floor).
  const liftTarget = realNow < view.liftUntil || view.held ? 0.06 : 0;
  view.lift += (liftTarget - view.lift) * Math.min(1, dt * 5);
  const wobble = realNow < view.liftUntil ? Math.sin(realNow / 300) * 0.05 : 0;
  const breath = view.awake && view.napping ? Math.sin((realNow / 1000) * Math.PI * 2 * 0.2) * 0.0025 : 0;
  // Duck Core's plan: where it thinks it is (its own frame, moved into the world by the home's
  // frame: its nest or mat), its posture, its moves and its head command.
  const [planX, planY, planYaw] = planToWorld(view.room.frame, view.body.x, view.body.y, view.body.yaw);
  const plan = {
    joints: { ...body.legs, ...headJoints() },
    trunk: [planX, planY, body.z + view.lift + breath],
    quat: yawRollQuat(planYaw, body.roll + wobble),
    mouth: view.mouth,
  };
  // Live physics: the solid duck is the physics truth only (the frame's truth, height, lean
  // and joints) and the plan moves the shadow duck; without truth (a fast-forward, no live
  // physics, the server gone) the solid duck is the plan, and the label says so.
  const target = view.external && !view.offline ? truthTarget(view.external) : null;
  // Every body sample at its own moment, between true samples only (fid:page: a head snap keeps
  // its speed and size); without frames (an older server) the view eases toward the frame.
  const played = target ? playFrames(dt, realNow) : null;
  view.live.shown = played || (target ? follow(view.live.shown, target, dt) : null);
  const hasPlayedMouth = !!played && Number.isFinite(played.mouth);
  view.live.drawnMouth = hasPlayedMouth ? played.mouth : null;
  view.live.drawnMouthSource = hasPlayedMouth ? played.mouthSource : "unavailable";
  const drawn = compose({ plan, truth: view.live.shown, physicsHead: view.live.physicsHead !== false });
  const truthMode = drawn.mode === "truth";
  view.drawnMode = drawn.mode;
  applyPose(drawn.solid);
  // 神經鴨原型 (neuro.js): its brain drives the body itself: no shadow duck, gap or ring.
  applyShadow(drawn.shadow && !view.neuro ? drawn.shadow : null, poseGap(drawn.shadow, drawn.solid));
  showGapLine(truthMode && !view.neuro ? drawn.solid.trunk : null, plan.trunk);
  // The home: translucent only when the physics does not have it (phase 0's flat floor);
  // the world's hand, people and toys move; walls between the camera and the duck fade (B4).
  const home = view.room.home;
  setGhost(home, truthMode && !view.room.physical);
  stepWorld(home, dt, truthMode ? drawn.solid : null);
  if (home && view.camera && view.trunk) fadeWalls(home, view.camera, view.trunk.getWorldPosition(new THREE.Vector3()), dt);
  if (view.neuro && view.planRing) view.planRing.visible = false;
  else showPlanRing(plan.trunk[0], plan.trunk[1]);
  showTrail(truthMode, realNow);
  renderMode(truthMode);

  if (readout) {
    const clipName = view.clip ? view.clip.name : null;
    const walk = view.body.walk;
    const physics = truthMode && view.live.info ? view.live.info.policy : null;
    const posture = physics
      ? { sit: "sit", rise: "rise", walk: "stand", stand: "stand" }[physics] || heading()
      : body.walking && walk ? "walk" : heading();
    let label = view.offline
      ? T("page.duck3d.step.a00bfc")
      : view.neuro
      ? T("page.duck3d.step.05430c", { lead: view.neuro.lead || "…" })
      : view.live.demo
      ? T("page.duck3d.step.9608bd")
      : !view.awake
      ? T("page.duck3d.step.177116")
      : view.napping
        ? clipName && clipName.startsWith("sleep_") ? T("page.duck3d.step.77715d", { clipName: clipZh(clipName) }) : T("page.duck3d.step.cfb5c3")
        : clipName && !HOLDS.has(clipName)
          ? clipZh(clipName)
          : now < view.walkUntil
            ? T("page.duck3d.step.304794")
            : view.rest || now < view.holdUntil
              ? T("page.duck3d.step.acd5fb")
              : T("page.duck3d.step.8f77cb");
    if (body.walking && walk) {
      const turning = Math.abs(yawRateOf(walk)) > 0.2;
      label +=
        walk.vx > 0
          ? T("page.duck3d.step.028128", { turning: T(turning ? "page.duck3d.step.cf83ae" : "page.duck3d.step.085168"), speed: walk.speed ?? 0 })
          : T("page.duck3d.step.b12124");
    }
    const r = drawn.solid.joints; // (the head as drawn: the physics head, or the command)
    const deg = (value) => `${Math.round(((value ?? 0) * 180) / Math.PI)}°`;
    // With a nest the mat is the nest: how far from it, and whether it sits in it. With the
    // truth drawn: where the body really is, and where the duck thinks it is.
    const base = T(view.hasNest ? "page.duck3d.step.f34f0e" : "page.duck3d.step.e0aafc");
    const planAway = Math.hypot(view.body.x, view.body.y);
    let where = T("page.duck3d.step.away", { base, away: planAway, nested: view.hasNest && view.nested ? T("page.duck3d.step.62aa02") : "" });
    if (truthMode) {
      const [tx, ty] = drawn.solid.trunk;
      const off = Math.hypot(tx - plan.trunk[0], ty - plan.trunk[1]);
      const [ux, uy] = worldToPlan(view.room.frame, tx, ty);
      // A dock against the wall: the body's distance from the real dock, and the duck's from
      // the nest it believes in (its origin) — the two are different places until phase 5.
      const dock = view.hasNest && view.room.home && view.room.home.layout.nest;
      const seat = dock && dock.kind === "dock" ? dock.seat : null;
      where = seat
        ? T("page.duck3d.step.6126b6", { y_mm: Math.hypot(tx - seat.x_mm / 1000, ty - seat.y_mm / 1000), planAway, off: Math.round(off * 100) })
        : T("page.duck3d.step.28a58b", { base, uy: Math.hypot(ux, uy), planAway, off: Math.round(off * 100) });
      // 神經鴨原型: only the body's own distance (Duck Core's idea of where it is is not its).
      if (view.neuro) {
        where = seat
          ? T("page.duck3d.step.body_dock", { y_mm: Math.hypot(tx - seat.x_mm / 1000, ty - seat.y_mm / 1000) })
          : T("page.duck3d.step.body_base", { base, uy: Math.hypot(ux, uy) });
      }
    }
    // 神經鴨原型: the beak is the command its driver sent for that very sample (the physics model has
    // no mouth joint): the readout says which it is (measured, commanded, or not given).
    const source = view.live.drawnMouthSource || "unavailable";
    const beakValue = source === "measured_sim" || source === "commanded_fallback" ? Math.round((view.live.drawnMouth ?? view.mouth) * 100) : null;
    const beak = truthMode ? beakCaption(window.I18N, source, beakValue) : T("page.duck3d.step.88a138", { mouth: Math.round(view.mouth * 100) });
    const truthNote = truthMode
      ? view.neuro
        ? T("page.duck3d.step.e38efc", { playedNote: playedNote() })
        : T("page.duck3d.step.6a7f61")
      : view.rate > 1.05
        ? T("page.duck3d.step.6fe76f", { rate: view.rate })
        : "";
    readout.textContent = T(view.neuro ? "page.duck3d.step.373d53" : "page.duck3d.step.4960ac", {
      posture: POSTURE_ZH[posture] ? T(POSTURE_ZH[posture]) : posture,
      label,
      neck_pitch: deg(r.neck_pitch),
      head_pitch: deg(r.head_pitch),
      head_yaw: deg(r.head_yaw),
      head_roll: deg(r.head_roll),
      mouth: Math.round(view.mouth * 100),
      beak,
      truthMode: truthNote,
      where,
      // (a phone's label over the view keeps to its title: its words are here)
      modeDetail: NARROW && NARROW.matches && modeDetail ? `\n${modeDetail}` : "",
    });
  }
}

// How the truth is drawn (fid:page): every body sample, so many a second, this far behind.
function playedNote() {
  const live = view.live;
  if (!live.played || !live.frames.length || live.sampleHz == null) return "";
  return T("page.duck3d.playedNote", { sampleHz: Math.round(live.sampleHz), delay: live.play.delay });
}

// The shadow duck: Duck Core's plan (null: hidden, the solid duck is the plan itself). Where
// it would lie right on the solid duck (`gap`, see poseGap) it fades out instead of tinting it.
function applyShadow(pose, gap) {
  const shadow = view.shadow;
  if (!shadow) return;
  const opacity = pose ? SHADOW_OPACITY * smooth((gap - 0.03) / 0.09) : 0;
  shadow.mount.visible = opacity > 0.01;
  if (!shadow.mount.visible) return;
  if (shadow.material) shadow.material.opacity = opacity;
  const joints = pose.joints || {};
  for (const [name, joint] of Object.entries(shadow.joints)) {
    if (!(name in joints)) continue;
    const [lo, hi] = joint.range || [-Math.PI, Math.PI];
    joint.group.quaternion.setFromAxisAngle(joint.axis, Math.min(hi, Math.max(lo, joints[name])));
  }
  shadow.mount.position.set(pose.trunk[0], pose.trunk[1], pose.trunk[2]);
  shadow.mount.quaternion.set(pose.quat[1], pose.quat[2], pose.quat[3], pose.quat[0]);
}

// Which duck is drawn, in words, just under the clock (see modeLabel in truth.js).
let modeText = "";
let modeDetail = "";
const NARROW = window.matchMedia ? window.matchMedia("(max-width: 760px)") : null;
function renderMode(truthMode) {
  if (!modeEl) return;
  const info = view.live.info || {};
  const label = modeLabel({
    truth: truthMode,
    enabled: view.live.enabled,
    on: view.live.on,
    available: !!info.available,
    // (time flows but the server's clock is not real time: the same test as pollLive's)
    fastForward: !!(view.time && view.time.playing && !view.live.clockRealtime),
    offline: view.offline,
    demo: !!info.demo,
    planFallen: heading() === "fallen",
    physicsFallen: !!info.fallen,
    broken: !!info.error,
    // (an older server's frames carry no truth at all, or the phase-0 anchored one)
    oldServer: !!(info.frame && (!("truth" in info.frame) || (info.frame.truth && "anchored_at" in info.frame.truth))),
    physicalHome: view.room.physical,
    held: !!info.held,
  });
  if (view.neuro) Object.assign(label, neuroMode(label, info));
  modeDetail = T(label.detail);
  const text = `${label.mode}|${label.title}|${label.detail}|${I18N ? I18N.lang : ""}`;
  if (text === modeText) return;
  modeText = text;
  const title = Object.assign(document.createElement("b"), { className: `chip ${label.mode}`, textContent: T(label.title) });
  const detail = Object.assign(document.createElement("span"), { textContent: modeDetail });
  modeEl.replaceChildren(title, detail);
  modeEl.dataset.mode = label.mode;
  modeEl.hidden = false;
}

// The prototype (neuro.js): the solid duck is the physics body its brain drives; there is no
// plan and no shadow duck. Without truth yet, the solid duck only waits at its start.
function neuroMode(label, info) {
  if (label.mode === "truth") {
    return { detail: info.fallen ? "page.duck3d.neuroMode.515726" : "page.duck3d.neuroMode.5a428e" };
  }
  return { title: "page.duck3d.neuroMode.acc137", detail: label.detail };
}

// Spots the duck learned to keep away from (fell there three times and could hardly get
// up): faint rings on the floor, exactly where Duck Core keeps them.
function drawSpots(spots) {
  if (!view.spots) return;
  const key = JSON.stringify(spots);
  if (key === view.spotsKey) return;
  view.spotsKey = key;
  view.spots.clear();
  for (const spot of spots) {
    const ring = new THREE.Mesh(
      new THREE.CircleGeometry(spot.r, 48),
      new THREE.MeshBasicMaterial({ color: 0xd9534f, transparent: true, opacity: 0.16, depthWrite: false }),
    );
    ring.rotation.x = -Math.PI / 2;
    const [sx, sy] = planToWorld(view.room.frame, spot.x, spot.y, 0); // Duck Core's frame → world
    ring.position.set(sx, 0.0025, -sy); // MuJoCo (x, y) on the scene's floor
    view.spots.add(ring);
  }
}

/* ---------------- page events ---------------- */

function isBody(motion) {
  return BODY_KEYS.some((key) => motion[key] != null && motion[key] !== false);
}

function bodyPending() {
  return view.pending.some((entry) => entry.item.motion && !entry.item.fidget && isBody(entry.item.motion));
}

function schedule(entry) {
  if (!view.model) {
    // 2D fallback: no frame loop to release it; its words at their moment all the same.
    if (!entry.item.fidget) setTimeout(() => play(entry.item), Math.max(0, entry.at - performance.now()));
    return;
  }
  view.pending.push(entry);
  view.pending.sort((a, b) => a.at - b.at);
}

function releasePending(realNow) {
  while (view.pending.length && view.pending[0].at <= realNow) play(view.pending.shift().item);
}

function queueItem(motion, text) {
  if (motion.walk) return { walk: motion.walk, caption: text };
  if (motion.place) return { place: motion.place, caption: text };
  if (motion.righting) return { righting: motion.righting, caption: text };
  if (motion.fall) return { fall: true, at: motion.at, caption: text };
  if (motion.posture) return { posture: motion.posture, at: motion.at, caption: text };
  return null;
}

// Finish at once everything the body still had to show (a walk in progress, the queue).
// Returns whether that moved the duck (so the page can say why it is somewhere else).
function flushBody() {
  const b = view.body;
  const from = [b.x, b.y];
  if (b.walk) {
    if (b.walk.to) placeAt(b, b.walk.to);
    b.walk = null;
  }
  b.rock = null;
  while (b.queue.length) settle(b.queue.shift());
  if (b.to) {
    b.shown = b.to;
    b.to = null;
  }
  return Math.hypot(b.x - from[0], b.y - from[1]) > 0.05;
}

// Apply what already happened without replaying it (fast-forwards, catch-ups).
function settleNow(item) {
  const motion = item.motion;
  if (motion.clip === "nap") view.sleepAmp = (motion.amplitude || 6000) / 10000;
  if (motion.power || motion.pickup) {
    view.clip = null;
    view.queue = [];
  }
  // Anything still waiting for its moment is older than this: its body part is applied
  // first, its moves are dropped.
  let older = false;
  for (const pending of view.pending) {
    const entry = pending.item.motion && !pending.item.fidget ? queueItem(pending.item.motion, pending.item.text) : null;
    if (entry) {
      view.body.queue.push(entry);
      older = true;
    }
  }
  view.pending = [];
  const entry = queueItem(motion, item.text);
  if (older || entry) flushBody();
  if (entry) settle(entry);
}

// Show one thing Duck Core did, now.
function play(item) {
  if (!view.clips) return;
  if (!view.model) {
    // The 2D fallback (no 3D model, nothing here animates: app.js plays the 2D duck): the
    // words still show, in order, and nothing piles up waiting for a 3D body.
    if (item.note) sayNote(item.text, item.kind);
    else if (item.motion && item.text && !item.fidget && !item.motion.weak) caption(item.text);
    return;
  }
  // Animation timing runs on the play clock; after a frozen tab it may lag real time.
  const now = Math.max(view.playNow, performance.now());
  const motion = item.motion ? { ...item.motion, bubble: item.bubble, caption: item.text, calls: item.calls } : null;
  if (!motion) {
    // What it is up to, in Duck Core's words: at once, or — when the body still has moves
    // to show (a fast-forward plays them one after another) — in its turn among them, so
    // the words never run ahead of the duck on screen.
    if (item.note && (busy(now) || view.body.queue.length)) view.body.queue.push({ note: item.text, kind: item.kind });
    else if (item.note) sayNote(item.text, item.kind);
    if (item.bubble) say(item.bubble, now);
    voiceCall(item.calls); // (a greeting, a bid, a babbled phrase: no move carries it)
    return;
  }
  const b = view.body;
  if (motion.clip) {
    if (motion.weak) {
      // The idle layer's glance: only when nothing Duck Core decided is playing. It takes
      // over from the glance before it (a duck looks at one thing, then at the next).
      const deciding = view.clip && !HOLDS.has(view.clip.name) && !view.clip.weak;
      if (deciding || view.queue.length || busy(now) || view.napping || !view.awake || view.rest) return;
      startClip(motion, now);
      return;
    }
    if (motion.clip === "nap") view.sleepAmp = (motion.amplitude || 6000) / 10000;
    // A held pose (the nap) and an idle glance give way to whatever the duck does next.
    if (!view.clip || HOLDS.has(view.clip.name) || view.clip.weak) startClip(motion, now);
    else {
      view.queue.push(motion);
      while (view.queue.length > CLIP_QUEUE_MAX) view.queue.shift();
    }
  } else if (motion.walk) {
    b.queue.push({ walk: motion.walk, caption: motion.caption });
  } else if (motion.place) {
    b.queue.push({ place: motion.place, caption: motion.caption });
    // (a lead line, queued with Duck Core's own notes: said once, never lost under a move)
    if (view.drawnMode === "truth" && motion.place.length < 3) setTimeout(() => sayNote(PLACED_ZH, "sim.placed"), 600);
  } else if (motion.righting) {
    b.queue.push({ righting: motion.righting, caption: motion.caption });
  } else if (motion.posture) {
    if (motion.stop) cutWalk(motion.cut, now); // a hand stopped it mid-walk
    b.queue.push({ posture: motion.posture, at: motion.at, caption: motion.caption });
  } else if (motion.fall) {
    cutWalk(motion.cut, now);
    b.queue.push({ fall: true, at: motion.at, caption: motion.caption });
    fx("!");
  } else if (motion.wake || motion.power) {
    if (view.clip && HOLDS.has(view.clip.name)) view.clip = null;
    let moved = false;
    if (motion.power) {
      // Switched off (or on): whatever was still to be shown is over.
      view.queue = [];
      const older = view.pending.filter((e) => !e.item.fidget);
      view.pending = [];
      for (const entry of older) if (entry.item.motion) settleNow(entry.item);
      moved = flushBody();
    }
    if (motion.power === "on") say(T("page.duck3d.play.7b1f2b"), now);
    caption(moved ? T("page.duck3d.play.12d26b", { caption: motion.caption }) : motion.caption);
  } else if (motion.battery) {
    fx("🔋");
    caption(motion.caption);
  } else if (motion.touch) {
    fx("♥");
    caption(motion.caption);
  } else if (motion.pickup) {
    view.pending = view.pending.filter((e) => e.item.fidget);
    flushBody(); // lifted: a walk being shown ends where it is picked up
    view.liftUntil = performance.now() + 2200;
    fx("↑");
    caption(motion.caption);
  } else if (motion.hold) {
    view.holdUntil = now + 2500;
    fx("✋");
    caption(motion.caption);
  } else if (motion.hand) {
    fx(motion.hand === "fast" ? "!" : "🖐");
    caption(motion.caption);
  } else if (motion.play) {
    view.walkUntil = now + 5200;
    say(T("page.duck3d.play.f9b34c"), now);
    caption(T("page.duck3d.play.8a0e20", { minutes: motion.minutes || "" }));
  }
  if (motion.bubble && !motion.clip && !motion.power) say(motion.bubble, now);
  if (!motion.clip) voiceCall(motion.calls); // (a clip's call sounds as the clip starts)
}

// Events that end whatever was still to be shown (it is applied at once, not played late).
const ENDS = new Set(["power.off", "sense.pickup", "sense.hold_head", "env.floor", "body.fell", "pad.play_start"]);

// Something someone just did plays now; whatever Duck Core did before it and the view has
// not shown yet (real time shows each move a moment late) comes first, at once: played in
// order, or, when this ends it (switched off, picked up, a hand on its head…), applied
// without being played — a walk never starts after the duck was lifted.
function catchUp(item) {
  const older = view.pending.filter((entry) => !entry.item.fidget);
  view.pending = [];
  if (!older.length) return;
  if (ENDS.has(item.kind)) {
    for (const entry of older) if (entry.item.motion) settleNow(entry.item);
  } else {
    for (const entry of older) play(entry.item);
  }
}

window.addEventListener("duck:feed", (event) => {
  const detail = Array.isArray(event.detail) ? { events: event.detail } : event.detail || {};
  if (detail.realtime != null) view.realtimeClock = !!detail.realtime;
  if (!view.clips) return;
  const nowMs = Date.now();
  const toPerf = (dueMs) => performance.now() + (dueMs - Date.now());
  let settled = 0;
  for (const raw of detail.events || []) {
    if (!raw.motion && !raw.bubble && !raw.note) continue;
    // (Duck Core's words in the viewer's language: 繁體中文 as they came)
    const item = !I18N || I18N.isSource() ? raw : { ...raw, text: F(raw, "text"), bubble: F(raw, "bubble") };
    // The first answer is history: only what is still to come (real time) is shown.
    if (detail.first && !(item.due_ms > nowMs)) continue;
    // A fast-forward step that answered after time was paused (or set to real time) is
    // stale: applied at once, never played as if it were happening now.
    if (item.settle || (detail.stale && !(item.due_ms > nowMs))) {
      if (item.motion) {
        settleNow(item);
        settled += 1;
      }
    } else if (item.due_ms && item.due_ms > nowMs + 40) {
      schedule({ at: toPerf(item.due_ms), item });
    } else {
      if (!item.due_ms) catchUp(item);
      play(item);
    }
  }
  for (const fidget of detail.fidgets || []) {
    if (view.seenFidgets.has(fidget.id)) continue;
    view.seenFidgets.add(fidget.id);
    if (fidget.due_ms > nowMs) schedule({ at: toPerf(fidget.due_ms), item: { motion: fidget, fidget: true } });
  }
  if (view.seenFidgets.size > 400) view.seenFidgets = new Set([...view.seenFidgets].slice(-200));
  if (settled >= 3) caption(T("page.duck3d.on_duck_feed", { settled }), true);
});

// Back from another tab (or a phone's frozen page): what happened meanwhile is applied at
// once and summed up, instead of playing stale moves late.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible" || !view.clips) return;
  const cutoff = performance.now() - 3000;
  const missed = [];
  const stale = view.pending.filter((entry) => entry.at < cutoff);
  view.pending = view.pending.filter((entry) => entry.at >= cutoff);
  for (const entry of stale) {
    if (entry.item.fidget) continue;
    if (entry.item.text) missed.push(entry.item.text);
    if (entry.item.motion) settleNow(entry.item);
  }
  for (const item of view.body.queue) if (item.caption) missed.push(item.caption);
  for (const motion of view.queue) if (motion.caption) missed.push(motion.caption);
  flushBody();
  view.queue = [];
  if (view.clip && !HOLDS.has(view.clip.name)) view.clip = null;
  view.playNow = performance.now();
  if (missed.length) caption(T("page.duck3d.on_visibilitychange.9e38fe", { missed: missed.slice(-3).join(T("fmt.sep.enum")) }));
});

// Leaving a fast-forward (paused, or real time): what it still had queued happened in the
// skipped minutes. It is applied at once (walks to where they ended, postures as they are)
// instead of playing on as if it were happening now. Real time's own moves, waiting for
// their second, are kept.
function flushBacklog() {
  const b = view.body;
  let skipped = view.queue.length + b.queue.filter((item) => !item.note).length + (b.walk ? 1 : 0);
  if (view.clip && !HOLDS.has(view.clip.name) && !view.clip.weak) {
    view.clip = null;
    skipped += 1;
  }
  view.queue = [];
  flushBody();
  view.playNow = performance.now();
  return skipped;
}

window.addEventListener("duck:time", (event) => {
  const before = view.time || {};
  view.time = event.detail || view.time;
  const wasFast = before.playing && !before.realtime;
  const isFast = view.time.playing && !view.time.realtime;
  if (wasFast && !isFast && view.clips) {
    const skipped = flushBacklog();
    // The words on screen (and those still waiting) were about the skipped minutes too.
    noteQueue.length = 0;
    clearTimeout(noteTimer);
    noteTimer = 0;
    captionLead.until = 0;
    captionMain.until = 0;
    renderCaption();
    if (skipped) {
      // A lead line, so the click's own words (⏸ / ▶) can show below it.
      captionLead.text = T("page.duck3d.on_duck_time", { skipped });
      captionLead.moment = false;
      captionLead.until = performance.now() + 4200;
      noteShownAt = performance.now();
      renderCaption();
    }
  }
  if (view.live.info) showLive(view.live.info);
});

// The server does not answer (stopped, restarting): the clock stops, the physics frames
// stop being shown, and the readout says so. Back again, all of it resumes.
window.addEventListener("duck:net", (event) => {
  const down = !!(event.detail && event.detail.down);
  if (down === view.offline) return;
  if (down && view.clock) {
    // Its clock stops at the last answer (not where real time would be by now).
    const c = view.clock;
    const running = view.time && view.time.playing && view.time.realtime;
    const lastOk = performance.now() - Math.max(0, Date.now() - (event.detail.since || Date.now()));
    const extra = running ? Math.max(0, lastOk - c.at) / 60000 : 0;
    view.clock = { ...c, minute: (c.minute + extra) % 1440, at: performance.now() };
  }
  view.offline = down;
  if (down) {
    setExternalPose(null);
    if (liveEls.status && view.live.enabled) liveEls.status.textContent = T("page.duck3d.on_duck_net");
  }
  clockText = "";
});

window.addEventListener("duck:caption", (event) => {
  const detail = event.detail;
  if (typeof detail === "string") caption(detail, true);
  else if (detail && detail.strong) caption(detail.text, false, false); // a click, right away
  else if (detail) caption(detail.text, true);
});

window.addEventListener("duck:state", (event) => {
  const st = event.detail;
  const when = /(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})(?::(\d{2}))?/.exec(String(st.clock.now || ""));
  if (when) {
    const minute = Number(when[4]) * 60 + Number(when[5]) + Number(when[6] || 0) / 60;
    const day = new Date(Number(when[1]), Number(when[2]) - 1, Number(when[3])).getDay();
    const date = `${when[2]}/${when[3]}`;
    const sun = view.clock && view.clock.date === date ? view.clock.sun : sunTimes(Number(when[1]), Number(when[2]), Number(when[3]));
    view.clock = { minute, at: performance.now(), date, sun, day, weekday: st.clock.weekday || T(WEEKDAY_ZH[day]) };
  }
  view.awake = st.body.awake;
  view.napping = st.body.napping;
  view.rest = ["stop", "lockout", "hot"].includes(st.body.mode);
  view.body.state = st.body.posture || "sit";
  view.body.position = st.body.position || null;
  view.poseErr = Math.max(0, Number(st.body.pose_err_mm) || 0) / 1000; // (the server adds it)
  drawSpots(st.body.avoid_spots || []);
  // A duck with a nest (Duck Core v4): the nest in the room, and the day's phase.
  const nest = st.body.nest || null;
  view.hasNest = !!nest;
  view.nested = !!(nest && nest.nested);
  view.held = !!(nest && nest.unplaced && st.body.awake);
  view.phaseZh = st.body.day || null; // (its phase_zh, shown in the viewer's language)
  showNest(nest, st.body.day);
  if (st.body.awake && !st.body.napping && view.clip && view.clip.name === "nap") view.clip = null;
  if (box) box.classList.toggle("off", !st.body.awake && !view.neuro);
});

window.addEventListener("duck:reset", async () => {
  const b = view.body;
  b.queue = [];
  b.walk = null;
  b.rock = null;
  b.position = null;
  b.to = null;
  b.shown = "sit";
  b.state = "sit";
  b.yaw = 0;
  b.x = 0;
  b.y = 0;
  view.pending = [];
  view.seenFidgets = new Set();
  view.queue = [];
  view.clip = null;
  view.live.shown = null;
  // Another duck (or another server): the old truth line is not this one's.
  view.trail = [];
  view.trailDirty = true;
  view.live.placedSeen = undefined;
  try {
    view.clips = await fetch("/api/clips").then((r) => r.json());
  } catch (error) {
    /* keep the old clips */
  }
});

// 神經鴨原型 (neuro.js, from /api/neuro/…): on or off, what the brain does now (the caption's
// lead line), a new reflex (said below it), a call (the bubble), asleep (Zzz), who it is.
window.addEventListener("duck:neuro", (event) => {
  const d = event.detail || {};
  if (!d.on) {
    view.neuro = null;
    clockText = "";
    return;
  }
  const was = view.neuro;
  view.neuro = { ...(was || {}), ...d };
  if (!was) clockText = "";
  const now = performance.now();
  if (d.moment) view.neuro.momentUntil = now + 3000; // (a startle: the magenta shadow a moment)
  if (d.lead != null) {
    captionLead.text = d.lead;
    captionLead.moment = now < (view.neuro.momentUntil || 0);
    captionLead.until = now + 60000;
  }
  // the brain's reflex as the brain says it (neuro.js marks it 「腦：」), then — a second or two
  // later — what the body measurably did (fid:page: 「身體：…」 or 「身體做不到：…」)
  if (d.reflex) caption(d.reflex, false, false);
  if (d.body) caption(d.body, false, false);
  // (the body's samples carry the calls when the server sends them: voiced with the beak)
  const spoken = d.say && !view.live.frameCalls;
  if (spoken) say(d.say);
  if (d.mouth && Array.isArray(d.mouth.t_ms) && d.mouth.t_ms.length) {
    // its own beak: each new 10 Hz sample played at its place in time, 250 ms behind
    const ts = d.mouth.t_ms;
    const newest = ts[ts.length - 1];
    const q = view.neuroMouth || (view.neuroMouth = []);
    const last = view.neuroMouthLast ?? -Infinity;
    if (newest < last) view.neuroMouthLast = -Infinity; // (a new duck: its clock starts over)
    ts.forEach((t, i) => {
      if (t <= (view.neuroMouthLast ?? -Infinity)) return;
      q.push({ at: now + (t - newest) + 250, open: Math.max(0, Math.min(1, Number(d.mouth.open[i]) || 0)) });
    });
    view.neuroMouthLast = newest;
    while (q.length > 60) q.shift();
  }
  // (voice:neuro) its sound with the beak that opened for it: the beak's track above plays 250 ms
  // behind its newest sample, so the call waits for its own moment on that track (no track: now)
  if (spoken && d.call && document.body.classList.contains("has3d")) {
    const at = Number(d.call.t_ms);
    const newest = view.neuroMouthLast;
    const wait = Number.isFinite(at) && Number.isFinite(newest) ? Math.max(0, Math.min(400, at - newest + 250)) : 0;
    if (wait > 0) setTimeout(() => soundCall(d.call), wait);
    else soundCall(d.call);
  }
  if (d.who != null && (!was || was.who !== d.who)) clockText = "";
  if (d.day != null && Number.isFinite(d.day)) view.neuroDay = { minute: d.day, at: now };
  renderCaption();
});

// The light at the duck's time, from the server (app.js passes on server.lights).
window.addEventListener("duck:lights", (event) => {
  view.serverLight = event.detail || null;
});

// The page may have had its first answer before this module loaded (three.js comes from a
// CDN): replay it, so the view starts from the duck's real state.
if (window.duckLights) view.serverLight = window.duckLights;
if (window.duckLast) window.dispatchEvent(new CustomEvent("duck:state", { detail: window.duckLast }));

// A language switch: the words drawn once (the mode label, the clock, the physics line) are
// written again; the readout and captions follow on the next frame / the next words.
if (I18N) {
  I18N.onChange(() => {
    modeText = "";
    clockText = "";
    pointing.shown = ""; // (the line under the command row is written again, in the new language)
    if (view.model && note && !note.dataset.i18n) note.textContent = F(view.model, "attribution");
    if (view.live.info) showLive(view.live.info);
  });
}

// (the words first: the catalogs load before the first thing this view says)
(I18N ? I18N.ready : Promise.resolve()).then(init, init);
