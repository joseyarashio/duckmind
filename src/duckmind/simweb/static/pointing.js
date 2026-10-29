/* Pointing at the floor (duck3d.js): a short click on the 3D view is a place on the floor, in the WORLD's
 * frame, the one /api/neuro/gothere takes (docs/design/neural-duck/REMOTE-GOTHERE-2026-10-23.md §2).
 *
 * The scene's frame: MuJoCo is Z-up and three.js Y-up, and duck3d.js turns the world group by −90° about x,
 * so a world point (x, y, z) in metres is the scene point (x, z, −y): the floor (z = 0) is the plane y = 0 and
 * the world's y is the scene's −z. `sceneOfWorld` and `floorPoint` are that one transform, both ways.
 *
 * Pure functions: no DOM, no three.js. duck3d.js uses them; tests run them in node. */

// layout element kinds the duck can stand on (world/homelayout.py SUPPORT_KINDS, and the floor-level decor)
export const SUPPORT_KINDS = new Set(["floor", "stairs", "platform", "pit_bottom", "nest_pad", "dock_plate", "rug", "mat", "floor_decor"]);
// kinds that are drawn but never in the way of a click on the floor (overhead or on the wall's upper part)
export const OVERHEAD_KINDS = new Set(["ceiling", "window", "wall_upper", "nest_tag"]);
export const FLOOR_LIMIT_M = 50; // a hit farther than this from the world's origin is the horizon, not a place
export const CLICK_MS = 500; // longer than this is a press, not a click
export const CLICK_PX = 6; // a mouse that moved more than this was dragging the camera
export const CLICK_PX_TOUCH = 10; // a finger is less steady
export const STALE_MS = 5000; // a pointer with no end this long is lost: it no longer counts
export const MARKER_MS = 30000; // the marker on the floor stays until the errand ends, or this long
export const OUTCOME_MS = 8000; // how long the line keeps saying how it went before it is the hint again

const round3 = (value) => Math.round(value * 1000) / 1000;

/** A world floor point (m) as the scene's [x, z] (the marker's place; its height is the floor's). */
export function sceneOfWorld(xM, yM) {
  return [xM, -yM];
}

/** Where a ray (origin and direction in scene units = metres) meets the floor plane y = 0, as the world's
 * {x_m, y_m}; null when it looks level or up (the sky), starts below the floor, or lands past FLOOR_LIMIT_M. */
export function floorPoint(origin, direction, limit = FLOOR_LIMIT_M) {
  const [ox, oy, oz] = origin;
  const [dx, dy, dz] = direction;
  if (![ox, oy, oz, dx, dy, dz].every(Number.isFinite)) return null;
  if (dy > -1e-4) return null;
  const k = -oy / dy;
  if (!(k > 0)) return null;
  const x = ox + dx * k;
  const z = oz + dz * k;
  if (Math.hypot(x, z) > limit) return null;
  return { x_m: round3(x), y_m: round3(-z) };
}

/** What a click on the 3D view means. `origin` / `direction` is the picking ray (scene units = metres); `hit` the
 * nearest solid of the room the ray meets ({distance, x, y, z, kind}: where, in scene coordinates, and the
 * layout element's kind) or null; `duck` the duck it meets ({distance}) or null; `drawn`: whether the room's
 * solids are in the scene at all (before the home loads, the floor plane y = 0 is the floor). Answers
 * `{kind: "floor", point, y}` (a place on the floor in the world's frame, and the height of the surface it was
 * on), `duck` (the duck itself: nothing to do), `blocked` (furniture, a wall or a toy is what was hit) or `sky`
 * (no floor there: past the walls, or above the horizon). */
export function resolvePick(origin, direction, hit = null, duck = null, drawn = true, limit = FLOOR_LIMIT_M) {
  if (duck && (!hit || duck.distance <= hit.distance)) return { kind: "duck" };
  if (!hit) {
    if (drawn) return { kind: "sky" };
    const point = floorPoint(origin, direction, limit);
    return point ? { kind: "floor", point, y: 0 } : { kind: "sky" };
  }
  if (!SUPPORT_KINDS.has(hit.kind)) return { kind: "blocked" };
  if (![hit.x, hit.y, hit.z].every(Number.isFinite) || Math.hypot(hit.x, hit.z) > limit) return { kind: "sky" };
  return { kind: "floor", point: { x_m: round3(hit.x), y_m: round3(-hit.z) }, y: hit.y };
}

/** What the telemetry's `errand` block says about the errand a click started (`id` is the one the server gave
 * back). `none`: the telemetry has not caught up yet (it still shows an older errand) or says nothing;
 * `waiting` / `walking`: under way; `arrived`; `gave_up` with the duck's `reason`; `replaced`: a newer errand
 * (another click, another page) took its place, so its own outcome is not shown any more. */
export function errandNews(errand, id) {
  if (!errand || id == null) return { kind: "none" };
  const seen = Number(errand.id);
  if (!Number.isFinite(seen) || seen < id) return { kind: "none" };
  if (seen > id) return { kind: "replaced" };
  switch (errand.state) {
    case "waiting":
    case "walking":
      return { kind: errand.state };
    case "arrived":
      return { kind: "arrived" };
    case "gave_up":
      return { kind: "gave_up", reason: errand.reason || null };
    default:
      return { kind: "none" };
  }
}

/** The errand a click started, as the page follows it: the marker on the floor and the one line under the command
 * row. `start(id, now)` after the server took the click; `update(errand, now)` with each telemetry `errand` block;
 * `fail(kind, now, answer)` when a click came to nothing (`refused`: the server's answer, which holds its words;
 * `failed`: no answer came; `notfloor`: the click hit furniture or a wall). A click that came to nothing never
 * ends an errand already under way, so it is kept apart (`note`) and only says so for OUTCOME_MS.
 * `marker(now)`: whether the marker shows (until the errand ends, or MARKER_MS). `line(now)`: what the line
 * says, `{state, kind, seq, ...}`, or null (the hint again). Pure and clock-injected, so node tests run it. */
export function makeErrandTrack() {
  let run = null; // {id, t0, news, endedAt}
  let note = null; // {kind, at, seq, answer}
  let seq = 0;
  return {
    start(id, now) {
      seq += 1;
      run = { id, t0: now, news: { kind: "sent" }, endedAt: null, seq };
      note = null;
    },
    fail(kind, now, answer = null) {
      seq += 1;
      note = { kind, at: now, seq, answer };
    },
    update(errand, now) {
      if (!run) return;
      const news = errandNews(errand, run.id);
      if (news.kind === "none") return;
      if (news.kind === "replaced") {
        run = null;
        return;
      }
      if (run.endedAt == null && (news.kind === "arrived" || news.kind === "gave_up")) {
        run.endedAt = now;
        note = null; // the end of the errand is the news now
      }
      run.news = news;
    },
    marker(now) {
      return !!run && run.endedAt == null && now - run.t0 < MARKER_MS;
    },
    line(now) {
      if (note) {
        if (now - note.at <= OUTCOME_MS) return { state: "refused", kind: note.kind, seq: note.seq, answer: note.answer };
        note = null;
      }
      if (!run) return null;
      const { news } = run;
      if (run.endedAt != null) {
        if (now - run.endedAt > OUTCOME_MS) {
          run = null;
          return null;
        }
        if (news.kind === "arrived") return { state: "done", kind: "arrived", seq: run.seq };
        return { state: "refused", kind: "gave_up", seq: run.seq, reason: news.reason };
      }
      if (news.kind === "walking") return { state: "walking", kind: "walking", seq: run.seq };
      if (now - run.t0 > MARKER_MS) {
        run = null; // the telemetry never said how it went: the line goes back to the hint
        return null;
      }
      return { state: "sent", kind: "sent", seq: run.seq };
    },
    clear() {
      run = null;
      note = null;
    },
  };
}

/** Tells a short click from the gestures that move the camera (a drag, a second finger, a long press, a
 * modified or non-primary button). `up` returns the click's {x, y} (client pixels) or null. Times are ms. */
export function makeClickTracker() {
  const active = new Map();
  let spoiled = false;
  return {
    down(id, x, y, t, options = {}) {
      for (const [other, p] of active) if (t - p.t > STALE_MS) active.delete(other);
      if (active.size === 0) spoiled = false;
      active.set(id, { x, y, t, limit: options.touch ? CLICK_PX_TOUCH : CLICK_PX });
      if (active.size > 1 || (options.button || 0) > 0 || options.modifier) spoiled = true;
    },
    move(id, x, y) {
      const p = active.get(id);
      if (p && Math.hypot(x - p.x, y - p.y) > p.limit) spoiled = true;
    },
    up(id, x, y, t) {
      const p = active.get(id);
      if (!p) return null;
      active.delete(id);
      const near = Math.hypot(x - p.x, y - p.y) <= p.limit;
      return !spoiled && active.size === 0 && near && t - p.t <= CLICK_MS ? { x, y } : null;
    },
    cancel(id) {
      if (active.delete(id)) spoiled = true;
    },
  };
}
