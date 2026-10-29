/* The remote hand's own arithmetic (remote.js): what the keys, the on-screen pad and a gamepad stick
 * mean as a twist, in the walk policy's own ranges (docs/design/neural-duck/REMOTE-GOTHERE-2026-10-23.md §1:
 * forward only, 0.35–0.55 m/s; an arc never faster than 0.40 m/s; |vyaw| ≤ 1.0 rad/s; + is left).
 * The server clamps every twist again (clamp_twist), so these numbers are what the page MEANS, never
 * the only guard.
 *
 * Pure functions: no DOM, no three.js. remote.js uses them; tests run them in node. */

export const WALK_VX_MIN = 0.35; // the slowest step the legs take
export const WALK_VX_MAX = 0.55;
export const ARC_VX_MAX = 0.4; // a turn (|vyaw| above ARC_VYAW) is an arc: never faster than this
export const ARC_VYAW = 0.3;
export const VYAW_MAX = 1.0;
export const KEY_FORWARD = 0.5; // a held key: half way up the walking range, 0.45 m/s
export const KEY_TURN = 0.8; // a held turn key: 0.8 rad/s
export const STICK_DEAD = 0.2; // the stick's dead zone (a worn stick never reads exactly 0)
const TURN_BELOW = 0.05; // a turn smaller than this rad/s is straight

// Physical keys (KeyboardEvent.code: the same place on every layout).
const FORWARD = ["ArrowUp", "KeyW"];
const BRAKE = ["ArrowDown", "KeyS"];
const LEFT = ["ArrowLeft", "KeyA"];
const RIGHT = ["ArrowRight", "KeyD"];
export const DRIVE_KEYS = new Set([...FORWARD, ...BRAKE, ...LEFT, ...RIGHT]);

const held = (codes, names) => names.some((name) => codes.has(name));
const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));
const round3 = (value) => Math.round(value * 1000) / 1000;

/** An input: `fwd` 0..1 (how hard forward), `turn` −1..1 (+ left), `brake` (stop whatever else is asked). */
export const NO_INPUT = Object.freeze({ fwd: 0, turn: 0, brake: false });

/** The arrow keys and WASD held now (a Set of KeyboardEvent.code). Down and S are the brake: the legs
 * never reverse (the server turns a negative speed into a stop), so there is nothing else for them to mean. */
export function keyInput(codes) {
  const left = held(codes, LEFT) ? 1 : 0;
  const right = held(codes, RIGHT) ? 1 : 0;
  return {
    fwd: held(codes, FORWARD) ? KEY_FORWARD : 0,
    turn: (left - right) * (KEY_TURN / VYAW_MAX),
    brake: held(codes, BRAKE),
  };
}

/** The pad's buttons held now: `{forward, left, right, stop}` booleans (same meaning as the keys). */
export function padInput(buttons) {
  const left = buttons.left ? 1 : 0;
  const right = buttons.right ? 1 : 0;
  return {
    fwd: buttons.forward ? KEY_FORWARD : 0,
    turn: (left - right) * (KEY_TURN / VYAW_MAX),
    brake: !!buttons.stop,
  };
}

const dead = (value) => (Math.abs(value) < STICK_DEAD ? 0 : (Math.sign(value) * (Math.abs(value) - STICK_DEAD)) / (1 - STICK_DEAD));

/** A gamepad's left stick: `x` right is +, `y` down is + (the Gamepad API's own signs). Pushed forward it
 * walks (how fast: how far), pushed sideways it turns (right is a negative vyaw). Back is nothing. */
export function stickInput(x, y) {
  const sx = Number(x);
  const sy = Number(y);
  if (!Number.isFinite(sx) || !Number.isFinite(sy)) return NO_INPUT;
  return { fwd: clamp(dead(-sy), 0, 1), turn: clamp(dead(-sx), -1, 1), brake: false };
}

/** Several hands at once: the hardest forward, the turns added (clamped), any brake wins. */
export function mergeInputs(...inputs) {
  let fwd = 0;
  let turn = 0;
  let brake = false;
  for (const input of inputs) {
    if (!input) continue;
    fwd = Math.max(fwd, input.fwd);
    turn += input.turn;
    brake = brake || !!input.brake;
  }
  return { fwd, turn: clamp(turn, -1, 1), brake };
}

/** The twist an input asks for: `{vx, vyaw}` (zero when nothing is asked, or the brake is held). Turning with
 * no forward asked is the slowest arc (the legs cannot turn on the spot); an arc is capped at ARC_VX_MAX. */
export function twistOf(input) {
  if (!input || input.brake) return { vx: 0, vyaw: 0 };
  const vyaw = Math.abs(input.turn * VYAW_MAX) < TURN_BELOW ? 0 : clamp(input.turn, -1, 1) * VYAW_MAX;
  if (input.fwd <= 0 && vyaw === 0) return { vx: 0, vyaw: 0 };
  let vx = input.fwd > 0 ? WALK_VX_MIN + (WALK_VX_MAX - WALK_VX_MIN) * clamp(input.fwd, 0, 1) : WALK_VX_MIN;
  if (Math.abs(vyaw) > ARC_VYAW) vx = Math.min(vx, ARC_VX_MAX);
  return { vx: round3(vx), vyaw: round3(vyaw) };
}

export const isZero = (twist) => !twist || (twist.vx === 0 && twist.vyaw === 0);

/** Whether a key press is going into a text field (then it is the person's typing, never a drive key). */
export function typing(target) {
  if (!target) return false;
  const tag = String(target.tagName || "").toUpperCase();
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || !!target.isContentEditable;
}

/** The epoch milliseconds of a key event (the server times key → legs from it). The event's own clock when it
 * agrees with the wall clock, else now (an old browser whose timeStamp is not relative to timeOrigin). */
export function epochOf(eventTimeStamp, timeOrigin, now) {
  const at = Number(timeOrigin) + Number(eventTimeStamp);
  return Number.isFinite(at) && Math.abs(now - at) < 1000 ? at : now;
}
