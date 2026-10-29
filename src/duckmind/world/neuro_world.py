"""The world side of the neural duck's senses (neuro:senses): ground truth for the world-side
sensor models, and the page's stimuli as real world events.

A simhome extension, loaded after the household and the sensors::

    DUCKMIND_SIMHOME_EXT=duckmind.world.worldkit,duckmind.world.sensors,duckmind.world.neuro_world

(:data:`EXTENSIONS`; ``LiveBody`` passes the environment on to the world server). The same part
runs inside :class:`duckmind.world.worldkit.Headless` and
:class:`duckmind.world.headless_duck.HeadlessDuck` for tests (:func:`attach`).

**Nothing here is a sense.** Everything this module computes is the world's truth — where things
are relative to the duck's head camera, what a ray from the lens hits first, how loud a clap is
at the duck's head, where a hand's fingers meet the duck's shell. It goes out only in the
``x_world`` section ``neuro`` (and ``x_neuro*`` ops): evaluator truth that feeds the world-side
sensor models of :mod:`duckmind.world.neuro_senses`, which decide what the duck's sensors could
honestly report. The brain (:mod:`duckmind.neuro`) never sees any of it.

What the section carries (all in world coordinates or physics seconds, for the models only):

* ``samples`` — the head camera's view at 10 Hz of physics time (:data:`SAMPLE_PERIOD_S`), the
  last :data:`KEEP_SAMPLES` of them. Per candidate (the visible hand, each person here, each
  thing, the dock's tag) the share of its sample points inside the delivered frame (family K and
  the lens's distortion, :mod:`duckmind.world.camera_model`) and, of those, the share a ray from
  the lens reaches first (``mj_ray`` against every seen geom: walls, furniture, people, the hand);
  the direction of the visible part from the eye in the duck's heading frame (bearing ``+`` =
  its left, elevation ``+`` = up), its angular size, distance, closing speed and own image
  motion, the light on it (``lights.HomeLightState.lux_at``) and, for things on the floor, the
  horizontal range to where it stands. Only candidates with a point inside the frame are listed.
* ``sounds`` — sound events near the duck's head (:class:`SoundEvent`): the page's claps, calls,
  voices, bangs, doors, music and TV, the family's footsteps (from their walking bodies), and the
  knocks of things (:mod:`duckmind.world.props` impacts). Each has its level at 1 m and at the
  duck's head (distance, the room's reverberation, a wall in between), when it starts and ends
  and its envelope. Its position is there for the page and the evaluators only: the duck's
  microphone is mono (spec §3.1), so the microphone model never turns it into a direction.
* ``touches`` — contact episodes between the duck's shell (collision meshes) and a hand, a
  person or a thing (``mj_geomDistance`` at 50 Hz while something is near), one per toucher
  (its site switches only after 60 ms elsewhere, a 60 ms gap does not end it: a mesh's nearest
  point jumps between neighbouring parts): the body site (head, beak, back, left, right, tail),
  the deepest press, the closing and sliding speeds from the two bodies' own motion.
* ``stim``, ``gesture``, ``tv``, ``music``, ``lamps_on`` — what the page's buttons set going
  (for drawing a sound's ripple or the hand's gesture on the page).

**Stimuli** (``x_neuro_stim``, :data:`STIMULI`): each page button becomes a world event, never a
message to the brain — a sound source at a person's hands or mouth (or, with nobody in the room,
somewhere 1.5–2.5 m away), the world's hand really moving in front of the lens or onto the shell
(:class:`Gesture`, driving the ``kit_hand`` mocap body only while the household's hand is idle), a
push as a real force on the trunk, a ball rolled or a thing put down by
:mod:`~duckmind.world.props`, a family member walking past (:mod:`~duckmind.world.people`), the
lamps switched (:mod:`~duckmind.world.sensors`' light clock), the duck picked up or set down by
the household's hand. The answer says in plain Chinese what the world did.

``say`` (live voice, LV0) is a family member saying a scripted line from a place — a sound event
that names the line (``utterance``), so the duck's microphone stream
(:mod:`duckmind.simweb.neuro_mic`) plays its audio while the hearing model hears the same event
(its level at the head, an onset, a class) as it hears any voice; with ``live: true`` the event
stands for a person speaking into a browser microphone.

Determinism: samples at fixed physics times, the stimulus choices from a seeded generator; the
same world, seed and request sequence give the same section.
"""

from __future__ import annotations

import math
import random
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from duckmind.i18n import join, lst, msg
from duckmind.world import camera_model as cm
from duckmind.world import worldkit as wk
from duckmind.world.neuro_utterances import LIVE_PREFIX, MAX_PLACE_MM, UtteranceLibrary

EXTENSIONS = "duckmind.world.worldkit,duckmind.world.sensors,duckmind.world.neuro_world"
SECTION = "neuro"
VERSION = 1
SAMPLE_PERIOD_S = 0.1  # the head camera's view, 10 Hz of physics time
KEEP_SAMPLES = 8  # the section carries the last 0.8 s (a 5 Hz reader misses nothing)
TOUCH_PERIOD_S = 0.02  # contact checks while something is near (50 Hz)
SELF_IMPACT_S = 2.0  # a thing the duck had against it this recently makes sounds the duck made
KEEP_EVENTS_S = 2.5  # sound events and contact episodes stay listed this long after they end
NEAR_M = 0.45  # a toucher's geom this close (centre to trunk, minus its size) is checked
JUMP_M_S = 4.0  # faster than anything moves at home: a placement, not a motion
TOUCH_GAP_M = 0.001  # a gap this small starts a contact (the hand collides with nothing)
TOUCH_END_GAP_M = 0.004  # …and one this small keeps it going (no flicker at rest)
SEEN_GROUPS = (1, 1, 1, 0, 0, 0)  # what the lens sees: world 0, nest details 1, robot visual 2
CAM = "head_camera"
PARK_Z_M = wk.PARK_Z_M
# The floor line's truth (duck-eye FL-v1's shape): five sectors across the picture, the clear
# floor to the nearest thing the body would meet (rays this high: a low toy, the chest), capped at
# FL-v1's range.
FLOOR_SECTORS = 5
# Bounded downward image search used only to associate a visible low silhouette with its floor.
# Two-pixel steps stay local to the silhouette; failure within this budget remains unknown.
# Consecutive rays that keep hitting the same candidate prove the march is still descending
# the object's own face (P1c A1 2026-09-30: the 16 px budget stranded on tall/near silhouettes
# whose face spans more than 16 px below the lowest sampled support point). Those carry steps
# may continue to the larger cap below. Off-frame rays, misses, unrelated hits, and
# over-tolerance floor transitions still end the association as unknown immediately.
FLOOR_ASSOCIATION_MAX_PX = 16
FLOOR_ASSOCIATION_CARRY_MAX_PX = 64
FLOOR_ASSOCIATION_STEP_PX = 2
FLOOR_MAX_M = 1.5
FLOOR_RAY_HEIGHTS_M = (0.02, 0.10)

# The delivered frame (family K at 1280×720, unrotated, through the family lens): a pinhole
# direction must first fall inside the frame's undistorted extents (the barrel polynomial folds
# back far outside the frame), then land inside it once distorted.
_K = cm.FAMILY
_DIST = cm.FAMILY_DISTORTION


def _frame_extents() -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    n = 48
    w, h = _K.width - 1, _K.height - 1
    for i in range(n + 1):
        for u, v in ((i * w / n, 0.0), (i * w / n, h), (0.0, i * h / n), (w, i * h / n)):
            x, y = cm.undistort_pixel(u, v, _K, _DIST)
            xs.append(x)
            ys.append(y)
    return min(xs), max(xs), min(ys), max(ys)


X_MIN, X_MAX, Y_MIN, Y_MAX = _frame_extents()


def in_frame(right: float, down: float, forward: float) -> bool:
    """Whether a point in the camera's own axes (OpenCV: right, down along the unrotated sensor,
    forward along the optical axis) lands inside the delivered frame."""
    if forward <= 1e-3:
        return False
    x, y = right / forward, down / forward
    if not (X_MIN - 0.02 <= x <= X_MAX + 0.02 and Y_MIN - 0.02 <= y <= Y_MAX + 0.02):
        return False
    xd, yd = cm.distort(x, y, _DIST)
    u = _K.cx + _K.fx * xd
    v = _K.cy + _K.fy * yd
    return 0.0 <= u <= _K.width - 1 and 0.0 <= v <= _K.height - 1


# ── sounds ──────────────────────────────────────────────────────────────────────

# Levels at 1 m, A-weighted "fast" (dB): a quiet flat is ~30 dB. Kinds are the contract's.
SOUND_DB_1M = {
    "clap": 82.0,
    "bang": 90.0,
    "door": 74.0,
    "knock": 72.0,
    "name": 70.0,  # a raised voice calling it
    "hello": 66.0,
    "voice": 62.0,
    "music": 66.0,
    "tv": 60.0,
    "footsteps": 52.0,
    "noise": 60.0,
}
IMPULSIVE = frozenset({"clap", "bang", "door", "knock", "footsteps"})
REVERB_DROP_DB = 11.0  # the room's diffuse field sits this far under the 1 m level
WALL_DB = 15.0  # a wall between the source and the duck
WALL_REVERB_DB = 8.0  # …and the other room's diffuse field reaches it this much weaker
STEP_EVERY_M = 0.31  # a footstep every half stride of a walking person (people.STRIDE_M / 2)
WALKING = frozenset({"walking", "leaving", "stepping_back"})


@dataclass
class SoundEvent:
    """One sound in the home (truth). ``db1m``: the source's level at 1 m; ``pos``: where it is
    (world metres); ``env``: its envelope (``impulse``: a sharp attack and a ~60 ms decay;
    ``speech``: syllables at ~4 Hz; ``steady``: on until ``t1``)."""

    id: int
    kind: str
    t0: float
    t1: float
    db1m: float
    pos: tuple[float, float, float]
    env: str
    src: str  # who or what made it (for the page and the evaluators)
    name: bool = False  # the utterance contains the duck's name
    db: float = 0.0  # at the duck's head, when last computed
    wall: bool = False
    # a scripted line (``neuro_utterances``: its id) or a person at a browser microphone
    # (``live:<who>``): the microphone stream plays its audio, the hearing model sees the event
    utterance: str | None = None
    # what made it, as a kind: "self" (the duck's own contact: it bumped something, something it had
    # just pushed or touched fell, a thing hit it), "thing" (a thing it did not touch), "person",
    # "page" (the page's bang or clap); "" = not told. Never an id (the hearing sense carries it on).
    by: str = ""

    def view(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "t0": round(self.t0, 3),
            "t1": round(self.t1, 3),
            "db1m": round(self.db1m, 1),
            "db": round(self.db, 1),
            "env": self.env,
            "name": self.name,
            "wall": self.wall,
            "src": self.src,
            "by": self.by,
            "pos_mm": wk.mm(self.pos),
        }
        if self.utterance is not None:
            out["utterance"] = self.utterance
        return out


def _finite(value: Any, default: float) -> float:
    """``value`` as a finite number, else ``default`` (a request's loose number)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def level_at(db1m: float, distance_m: float, *, wall: bool = False) -> float:
    """A source's level at ``distance_m``: the direct sound falling 6 dB per doubling plus the
    room's diffuse field (:data:`REVERB_DROP_DB` under the 1 m level); a wall in between takes
    :data:`WALL_DB` off the direct path and :data:`WALL_REVERB_DB` off the diffuse field."""
    direct = db1m - 20.0 * math.log10(max(distance_m, 0.1))
    reverb = db1m - REVERB_DROP_DB
    if wall:
        direct -= WALL_DB
        reverb -= WALL_REVERB_DB
    return 10.0 * math.log10(10.0 ** (direct / 10.0) + 10.0 ** (reverb / 10.0))


def impact_db(impulse_mns: float) -> float:
    """How loud a knock is at 1 m from its impulse (mN·s): a felt ball tapping the duck ~50 dB,
    a toy dropped from 35 cm ~63 dB, a stack of books falling ~85 dB (a rough fit, a knob)."""
    return max(40.0, min(92.0, 50.0 + 12.0 * math.log10(max(impulse_mns, 1.0) / 10.0)))


# ── touch sites ─────────────────────────────────────────────────────────────────

HEAD_BODIES = frozenset({"jaw_soft", "neck", "neck_pitch", "yaw_roll_motion", "bearing_roll"})
LEFT_LEG = frozenset({"hip_l", "upper_leg_left", "leg", "ankle_left"})
RIGHT_LEG = frozenset({"hip_l_2", "upper_leg_right", "leg_2", "ankle_right"})
BEAK_RADIUS_M = 0.035  # within this of the beak tip a head contact is the beak
# The trunk's collision hull in its own frame spans x −0.047…0.035, y ±0.032, z −0.044…0.042 m:
# a contact is classed by where it sits relative to the hull's middle (top: back, rear: tail).
TRUNK_CENTRE = (-0.006, 0.0, 0.0)
SITE_SWITCH_CHECKS = 3  # a contact's site changes after this many checks (60 ms) elsewhere
GAP_CHECKS = 3  # …and it ends after this many checks without contact


@dataclass
class Contact:
    """One contact episode between a toucher and a site of the duck (truth)."""

    key: str
    by: str
    site: str
    t0: float
    t1: float
    depth_mm: float = 0.0  # deepest press (the hand goes through: how far it would press)
    closing: float = 0.0  # fastest approach along the normal, m/s
    slide: float = 0.0  # fastest sliding along the surface, m/s
    now_depth_mm: float = 0.0
    now_slide: float = 0.0
    now_closing: float = 0.0
    active: bool = True
    pending_site: str = ""  # another site seen at the last checks (hysteresis)
    pending_n: int = 0
    missing: int = 0  # checks in a row without contact (a short gap does not end it)

    def view(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "by": self.by,
            "site": self.site,
            "t0": round(self.t0, 3),
            "t1": round(self.t1, 3),
            "active": self.active,
            "depth_mm": round(self.depth_mm, 1),
            "closing": round(self.closing, 3),
            "slide": round(self.slide, 3),
            "now_depth_mm": round(self.now_depth_mm, 1),
            "now_slide": round(self.now_slide, 3),
            "now_closing": round(self.now_closing, 3),
        }


# ── the hand's gestures ─────────────────────────────────────────────────────────


@dataclass
class GLeg:
    """One piece of a gesture. ``kind``: ``move`` (to ``to`` in the world), ``track`` (to a
    point of the duck plus ``offset`` in its heading frame, followed while it moves), ``press``
    (``track`` whose height is corrected each step until the hand just touches ``bodies`` —
    ``slide`` metres along the heading frame's x over the leg), ``wave`` (side to side around
    ``to`` by ``amp`` at ``hz``, across ``axis``)."""

    label: str
    kind: str
    seconds: float
    to: tuple[float, float, float] | None = None
    follow: str = "head"  # head | beak | trunk | camera
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    slide: float = 0.0
    press_mm: float = 3.0
    bodies: tuple[str, ...] = ()
    amp: float = 0.0
    hz: float = 0.0
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    quat: tuple[float, float, float, float] | None = None  # a fixed world orientation
    yaw_off: float = 0.0  # else: facing the duck's heading plus this (followed)
    press_dir: tuple[float, float, float] = (0.0, 0.0, 1.0)  # heading frame: the press backs off
    press_speed: float = 0.1  # how fast the press closes in (m/s): a pat 0.1, a poke 0.8
    force: tuple[float, float, float] | None = None  # a push on the trunk during the leg (N)
    t0: float | None = None
    p0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    q0: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class Gesture:
    stim: str
    legs: list[GLeg]
    index: int = 0
    started: float = 0.0
    done: bool = False
    why_zh: str = ""
    dz: float = 0.0  # the presses' correction along press_dir so far (kept across presses)


# ── what the page's buttons do ──────────────────────────────────────────────────

STIMULI: dict[str, str] = {
    "clap": "有人拍了一下手",
    "call_name": "有人叫牠的名字",
    "voice": msg("neuro.world.voice.default"),
    "hello": msg("neuro.world.hello.default"),
    "say": msg("neuro.world.say.default"),
    "bang": msg("neuro.world.bang.default"),
    "door": "開門的聲音",
    "knock": "敲門聲",
    "music": msg("neuro.world.music.default"),
    "tv": msg("neuro.world.tv.default"),
    "wave": "一隻手在牠前面揮一揮",
    "hand_loom": msg("neuro.world_fallback.hand_loom"),
    "pet_head": "一隻手摸摸牠的頭",
    "stroke_back": msg("neuro.world_fallback.stroke_back"),
    "poke": "有人戳牠一下",
    "push": "有人推牠一下",
    "roll_ball": "滾一顆球過去",
    "toy_drop": "丟一個玩具在牠旁邊",
    "novel_object": "在牠前面放一個東西",
    "walk_by": "一個家人從旁邊走過去",
    "sit_near": "一個家人走過來待在牠旁邊",
    "lights_off": "關燈",
    "lights_on": "開燈",
    "pick_up": "把牠抱起來",
    "set_down": "把牠放下來",
}
FAMILY_ORDER = ("owner", "partner", "child", "elder", "guest")
OFFSCREEN_R_M = (1.5, 2.5)  # nobody in the room: the sound comes from somewhere this far
HAND_H_M = 1.0  # a standing person's hands (a clap), m
MOUTH_H_M = 1.45  # a standing adult's mouth, m
SIT_NEAR_S = 20.0
STAND_AFTER_S = (
    25.0  # a family member who spoke from a place stays standing there this long (sim time)
)
# A family member steps into the room at the entry; with the duck sitting right there they would
# appear on top of it (live, neuro:fix: the snuggler resting by the door was knocked over twice),
# so then they come from the sofa instead (the home's other spot).
ENTRY_CLEAR_M = 0.8
PUSH_N = (2.0, 7.0)  # a push's force range (strength 0..1), for PUSH_S
PUSH_S = 0.15


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _vec(a: Sequence[float]) -> tuple[float, float, float]:
    return (float(a[0]), float(a[1]), float(a[2]))


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _angle(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return math.acos(max(-1.0, min(1.0, _dot(a, b) / (na * nb))))


def _heading_frame(yaw: float, v: Sequence[float]) -> tuple[float, float, float]:
    """A world vector in the heading frame (x forward, y left, z up) of a duck facing ``yaw``."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * v[0] + s * v[1], -s * v[0] + c * v[1], float(v[2]))


def _to_world(yaw: float, v: Sequence[float]) -> tuple[float, float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1], float(v[2]))


@dataclass
class Candidate:
    """Something the camera could see: its bodies, the geoms that are drawn, how it is named."""

    id: str
    kind: str  # hand | person | thing | nest
    bodies: tuple[int, ...]
    geoms: tuple[int, ...]
    cls: str = ""
    mocap: tuple[int, ...] = ()
    radius: float = 0.0  # a ball's radius (m)
    static: bool = False  # part of the home (the dock's tag): a ray must hit its own geom
    last: tuple[float, tuple[float, float, float], tuple[float, float, float]] | None = None
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


class NeuroWorld:
    """The neural duck's world-side truth and stimuli (see the module docstring). One per world.

    Parts come from the household (:meth:`bind`, or found through the simhome registry's
    sections): ``hand``, ``people``, ``props``, ``nest``; ``lights`` is a function returning the
    home's current :class:`~duckmind.world.lights.HomeLightState` (the sensors' light clock),
    ``switch_lamps(on)`` switches the lamps."""

    name = "neuro"

    def __init__(self, *, seed: int = 7) -> None:
        self.seed = int(seed)
        self.rng = random.Random(f"neuro-world:{self.seed}")
        self.registry: Any = None
        self.hand: Any = None
        self.people: Any = None
        self.props: Any = None
        self.nest: Any = None
        self.lights: Callable[[], Any] | None = None
        self.switch_lamps: Callable[[bool], Any] | None = None
        self.world: Any = None
        self.body: Any = None
        self.model: Any = None
        self.robot: Any = None
        self.cam = -1
        self.cam_body = -1
        self.head_site = -1
        self.beak_site = -1
        self.hand_mocap = -1
        self.hand_geoms: tuple[int, ...] = ()
        self.duck_geoms: tuple[int, ...] = ()  # collision meshes (touch)
        self.duck_body_names: dict[int, str] = {}
        self.candidates: list[Candidate] = []
        self._surface_cache: dict[int, tuple[tuple[float, float, float], ...]] = {}
        self.samples: deque[dict[str, Any]] = deque(maxlen=KEEP_SAMPLES)
        self.sounds: list[SoundEvent] = []
        self.contacts: dict[str, Contact] = {}
        self.ended: deque[Contact] = deque(maxlen=64)
        self.gesture: Gesture | None = None
        self.pending: list[tuple[float, str, dict[str, Any]]] = []  # (sim time, what, params)
        self.standing: dict[str, float] = {}  # a speaker's body stands here until this sim time
        self.stims: deque[dict[str, Any]] = deque(maxlen=24)
        self.tv: SoundEvent | None = None
        self.music: SoundEvent | None = None
        self.errors: list[str] = []
        self._utterances: UtteranceLibrary | None = None
        self._utterances_read = False
        self._next_sample = 0.0
        self._next_touch = 0.0
        self._sound_id = 0
        self._stim_id = 0
        self._impact_seq = 0
        self._duck_touched: dict[str, float] = {}  # thing key -> when it was last against the duck
        # thing key -> when a thing the duck moved last hit it (one step of chain, not carried on)
        self._chain_touched: dict[str, float] = {}
        self._walked: dict[str, tuple[float, float, float]] = {}  # who → (x, y, metres since step)
        self._lock = threading.RLock()  # the op thread, the pump thread and the step hooks
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lamps_on: bool | None = None
        self._ray_group: Any = None
        self._ray_hit: Any = None
        self._fromto: Any = None
        self._touch_seq = 0
        self._thing_bodies: set[int] = set()  # loose things and people: a floor line's "thing"
        # toucher → (time, position relative to the trunk, body, velocity, the one before)
        self._near: dict[
            str, tuple[float, tuple[float, ...], int, tuple[float, ...], tuple[float, ...]]
        ] = {}

    # ── wiring ───────────────────────────────────────────────────────────────

    def bind(
        self,
        *,
        hand: Any = None,
        people: Any = None,
        props: Any = None,
        nest: Any = None,
        lights: Callable[[], Any] | None = None,
        switch_lamps: Callable[[bool], Any] | None = None,
    ) -> NeuroWorld:
        self.hand, self.people, self.props, self.nest = hand, people, props, nest
        self.lights, self.switch_lamps = lights, switch_lamps
        return self

    def _from_registry(self) -> None:
        """The household's parts through the registry's ``x_world`` sections (bound methods)."""
        sections = getattr(self.registry, "sections", {}) or {}

        def part(section: str) -> Any:
            entry = sections.get(section)
            fn = getattr(entry, "fn", None)
            return getattr(fn, "__self__", None)

        self.hand = self.hand or part("hand")
        self.people = self.people or part("people")
        self.props = self.props or part("things")
        self.nest = self.nest or part("nest")
        suite = part("lights")
        if suite is not None and self.lights is None:
            self.lights = suite.recent_state

            def switch(on: bool, suite: Any = suite) -> Any:
                return suite.lights_op(self.body, {"lamps_on": bool(on)})

            self.switch_lamps = switch

    def ready(self, world: Any, body: Any = None) -> None:
        import mujoco
        import numpy as np

        if self.registry is not None:
            self._from_registry()
        self.world, self.body = world, body
        model = self.model = world.model
        self.robot = wk.RobotIds.of(model)
        self.cam = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM))
        if self.cam < 0:
            raise ValueError("the world has no head camera")
        self.cam_body = int(model.cam_bodyid[self.cam])
        self.head_site = wk.site_id(model, "head_imu")
        self.beak_site = wk.site_id(model, "mouth_tip")
        robot = self.robot.bodies
        names = {}
        for b in robot:
            names[int(b)] = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b)) or "")
        self.duck_body_names = names
        self.duck_geoms = tuple(
            g
            for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in robot and int(model.geom_contype[g]) != 0
        )
        hand = wk.body_id(model, "kit_hand")
        self.hand_mocap = int(model.body_mocapid[hand]) if hand >= 0 else -1
        self.hand_geoms = tuple(g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == hand)
        self.candidates = self._candidates(model)
        import mujoco

        def is_flat_floor(g: int) -> bool:
            name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
            named_floor = name == "floor" or name.startswith("floor_") or name.startswith("floor-")
            if not named_floor or int(model.geom_bodyid[g]) != 0:
                return False
            if float(world.data.geom_xmat[g][8]) < 0.98:
                return False  # walls or downward-facing surfaces are never support floor
            typ = int(model.geom_type[g])
            if typ == int(mujoco.mjtGeom.mjGEOM_PLANE):
                return True
            return typ == int(mujoco.mjtGeom.mjGEOM_BOX) and float(model.geom_size[g][2]) <= 0.02

        self._floor_geoms = frozenset(g for g in range(int(model.ngeom)) if is_flat_floor(g))
        self._surface_cache: dict[int, tuple[tuple[float, float, float], ...]] = {}
        self._thing_bodies = {
            int(b)
            for cand in self.candidates
            if cand.kind in ("thing", "person")
            for b in cand.bodies
        }
        self._ray_group = np.array(SEEN_GROUPS, dtype=np.uint8)
        self._ray_hit = np.zeros(1, dtype=np.int32)
        self._fromto = np.zeros(6)
        now = float(world.data.time)
        self._next_sample = math.floor(now / SAMPLE_PERIOD_S) * SAMPLE_PERIOD_S
        self._next_touch = now
        if self.props is not None:
            self._impact_seq = int(getattr(self.props, "seq", 0))

    def _candidates(self, model: Any) -> list[Candidate]:
        import mujoco

        def geoms_of(bodies: Sequence[int]) -> tuple[int, ...]:
            wanted = set(bodies)
            return tuple(
                g
                for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) in wanted and SEEN_GROUPS[int(model.geom_group[g])]
            )

        out: list[Candidate] = []
        hand = wk.body_id(model, "kit_hand")
        if hand >= 0:
            out.append(
                Candidate("hand", "hand", (hand,), geoms_of((hand,)), mocap=(self.hand_mocap,))
            )
        for who in FAMILY_ORDER:
            ids = [
                wk.body_id(model, f"person_{who}{suffix}") for suffix in ("", "_leg_l", "_leg_r")
            ]
            if min(ids) < 0:
                continue
            mocaps = tuple(int(model.body_mocapid[b]) for b in ids)
            out.append(
                Candidate(f"person:{who}", "person", tuple(ids), geoms_of(ids), mocap=mocaps)
            )
        things = getattr(self.props, "things", {}) or {}
        for key, tracked in things.items():
            thing = getattr(tracked, "thing", None)
            out.append(
                Candidate(
                    f"thing:{key}",
                    "thing",
                    (int(tracked.body),),
                    geoms_of((int(tracked.body),)),
                    cls=str(getattr(thing, "cls", "") or ""),
                    radius=float(getattr(tracked, "radius_m", 0.0) or 0.0),
                )
            )
        tag = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "home_nest_tag"))
        if tag >= 0:
            out.append(
                Candidate("nest", "nest", (int(model.geom_bodyid[tag]),), (tag,), static=True)
            )
        return out

    # ── the per-step hooks (the world's lock is held) ─────────────────────────

    def before_step(self, world: Any) -> None:
        gesture = self.gesture
        if gesture is None or gesture.done:
            return
        data = world.data
        if self._hand_busy():
            self._end_gesture(data, "家裡的手在做別的事，手勢停下")
            return
        self._drive(gesture, data, float(data.time))

    def after_step(self, world: Any) -> None:
        data = world.data
        now = float(data.time)
        if now + 1e-9 >= self._next_touch:
            self._next_touch = now + TOUCH_PERIOD_S
            try:
                self._touch(data, now)
            except Exception as error:  # a broken check never stops the world
                self._error(f"touch: {type(error).__name__}: {error}")
        if now + 1e-9 >= self._next_sample:
            while self._next_sample <= now + 1e-9:
                self._next_sample += SAMPLE_PERIOD_S
            try:
                self._footsteps(data, now)
                self._impacts(now)
                self.samples.append(self.sample(data))
            except Exception as error:
                self._error(f"sample: {type(error).__name__}: {error}")

    def _error(self, text: str) -> None:
        if len(self.errors) < 20:
            self.errors.append(text)

    # ── the camera's view ──────────────────────────────────────────────────────

    def camera(self, data: Any) -> tuple[tuple[float, float, float], list[tuple[float, ...]]]:
        """The lens's position and its OpenCV axes in the world (right, down, forward): the
        official ``head_camera`` faces backwards (camera_model's 180° fix), so the physics
        model's camera frame columns are right, down and forward as they are."""
        pos = _vec(data.cam_xpos[self.cam])
        m = [float(v) for v in data.cam_xmat[self.cam]]
        right = (m[0], m[3], m[6])
        down = (m[1], m[4], m[7])
        forward = (m[2], m[5], m[8])
        return pos, [right, down, forward]

    def _kinematic_center(self, data: Any, cand: Candidate) -> tuple[float, float, float]:
        """Truth-only center for motion estimates; independent of which surface points are visible."""
        points: list[tuple[float, float, float]] = []
        for g in cand.geoms:
            c = _vec(data.geom_xpos[g])
            points.append(c)
            if int(self.model.geom_type[g]) == 3:  # preserve the prior capsule weighting
                half = float(self.model.geom_size[g][1])
                axis = (
                    float(data.geom_xmat[g][2]),
                    float(data.geom_xmat[g][5]),
                    float(data.geom_xmat[g][8]),
                )
                points.extend(
                    (
                        (c[0] + axis[0] * half, c[1] + axis[1] * half, c[2] + axis[2] * half),
                        (c[0] - axis[0] * half, c[1] - axis[1] * half, c[2] - axis[2] * half),
                    )
                )
        if not points:
            return (0.0, 0.0, 0.0)
        return tuple(sum(point[k] for point in points) / len(points) for k in range(3))

    def _local_surface_points(self, geom: int) -> tuple[tuple[float, float, float], ...]:
        """Bounded deterministic surface support points (at most 26) in compiled geom coordinates."""
        cached = self._surface_cache.get(geom)
        if cached is not None:
            return cached
        import math

        typ = int(self.model.geom_type[geom])
        size = tuple(float(v) for v in self.model.geom_size[geom])
        directions = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    if x == y == z == 0:
                        continue
                    n = math.sqrt(x * x + y * y + z * z)
                    directions.append((x / n, y / n, z / n))
        out: list[tuple[float, float, float]] = []
        if typ == 7:  # compiled mesh vertices; choose one support vertex per direction
            mesh = int(self.model.geom_dataid[geom])
            if 0 <= mesh < int(self.model.nmesh):
                start = int(self.model.mesh_vertadr[mesh])
                count = int(self.model.mesh_vertnum[mesh])
                verts = self.model.mesh_vert[start : start + count]
                for d in directions:
                    if len(verts):
                        best = max(
                            verts,
                            key=lambda v: (
                                float(v[0]) * d[0] + float(v[1]) * d[1] + float(v[2]) * d[2]
                            ),
                        )
                        out.append(tuple(float(v) for v in best))
        elif typ == 6:  # 26 face, edge and corner support points
            for d in directions:
                out.append(
                    tuple((1.0 if v > 0 else -1.0) * size[k] if v else 0.0 for k, v in enumerate(d))
                )
        elif typ == 2:  # sphere
            out = [(d[0] * size[0], d[1] * size[0], d[2] * size[0]) for d in directions]
        elif typ == 4:  # ellipsoid support point
            for d in directions:
                denom = math.sqrt(sum((size[k] * d[k]) ** 2 for k in range(3)))
                if denom > 1e-12:
                    out.append(tuple(size[k] ** 2 * d[k] / denom for k in range(3)))
        elif typ == 3:  # capsule, local z axis
            for d in directions:
                sign = 1.0 if d[2] > 0 else -1.0 if d[2] < 0 else 0.0
                out.append((size[0] * d[0], size[0] * d[1], size[0] * d[2] + sign * size[1]))
        elif typ == 5:  # cylinder, local z axis
            for d in directions:
                radial = math.hypot(d[0], d[1])
                x, y = (
                    (size[0] * d[0] / radial, size[0] * d[1] / radial)
                    if radial > 1e-12
                    else (size[0], 0.0)
                )
                z = size[1] if d[2] > 0 else -size[1] if d[2] < 0 else 0.0
                out.append((x, y, z))
        unique = {tuple(round(v, 9) for v in point): tuple(point) for point in out}
        result = tuple(unique.values())[:26]
        self._surface_cache[geom] = result
        return result

    def _points(self, data: Any, cand: Candidate) -> list[tuple[tuple[float, float, float], int]]:
        """Transform bounded compiled-surface samples into world coordinates."""
        out = []
        for g in cand.geoms:
            pos = _vec(data.geom_xpos[g])
            m = [float(v) for v in data.geom_xmat[g]]
            for x, y, z in self._local_surface_points(g):
                out.append(
                    (
                        (
                            pos[0] + m[0] * x + m[1] * y + m[2] * z,
                            pos[1] + m[3] * x + m[4] * y + m[5] * z,
                            pos[2] + m[6] * x + m[7] * y + m[8] * z,
                        ),
                        g,
                    )
                )
        return out

    def _present(self, data: Any, cand: Candidate) -> bool:
        if cand.mocap:
            return any(m >= 0 and float(data.mocap_pos[m][2]) > PARK_Z_M / 2 for m in cand.mocap)
        return True

    def _visible_point(
        self, data: Any, cam: Sequence[float], point: Sequence[float], cand: Candidate
    ) -> tuple[tuple[float, float, float], int] | None:
        """Return the actual first candidate surface hit, never a hidden support point."""
        import mujoco
        import numpy as np

        vec = _sub(point, cam)
        dist = _norm(vec)
        if dist < 1e-6:
            return None
        unit = np.array([vec[0] / dist, vec[1] / dist, vec[2] / dist])
        self._ray_hit[0] = -1
        got = float(
            mujoco.mj_ray(
                self.model,
                data,
                np.array(cam, dtype=float),
                unit,
                self._ray_group,
                1,
                self.cam_body,
                self._ray_hit,
            )
        )
        hit = int(self._ray_hit[0])
        if got < 0 or hit < 0:
            return None
        if hit not in cand.geoms and (
            cand.static or int(self.model.geom_bodyid[hit]) not in cand.bodies
        ):
            return None
        actual = (
            float(cam[0] + unit[0] * got),
            float(cam[1] + unit[1] * got),
            float(cam[2] + unit[2] * got),
        )
        return actual, hit

    def _clear(
        self, data: Any, cam: Sequence[float], point: Sequence[float], cand: Candidate, g: int
    ) -> bool:
        """Compatibility predicate backed by a first-surface ray test."""
        return self._visible_point(data, cam, point, cand) is not None

    def sample(self, data: Any) -> dict[str, Any]:
        """The head camera's view now (see the module docstring's ``samples``)."""
        now = float(data.time)
        cam, (right, down, forward) = self.camera(data)
        robot = self.robot
        trunk = _vec(data.xpos[robot.trunk])
        yaw = wk.heading_of([float(v) for v in data.xquat[robot.trunk]])
        state = self._light_state()
        cam_lux = self._lux(state, (cam[0], cam[1], 0.0))
        backlight = bool(getattr(getattr(state, "backlight", None), "on", False))
        seen: list[dict[str, Any]] = []
        for cand in self.candidates:
            if not self._present(data, cand):
                cand.last = None
                continue
            points = self._points(data, cand)
            if not points:
                continue
            centre = self._kinematic_center(data, cand)
            prev = cand.last
            if prev is not None and now - prev[0] > 1e-6 and now - prev[0] < 0.6:
                dt = now - prev[0]
                cand.velocity = (
                    (centre[0] - prev[1][0]) / dt,
                    (centre[1] - prev[1][1]) / dt,
                    (centre[2] - prev[1][2]) / dt,
                )
                cam_v = (
                    (cam[0] - prev[2][0]) / dt,
                    (cam[1] - prev[2][1]) / dt,
                    (cam[2] - prev[2][2]) / dt,
                )
                # a jump no home thing makes (a thing put down by the page, the duck placed):
                # it appeared there, it did not fly there
                if _norm(cand.velocity) > JUMP_M_S:
                    cand.velocity = (0.0, 0.0, 0.0)
                if _norm(cam_v) > JUMP_M_S:
                    cam_v = (0.0, 0.0, 0.0)
            else:
                cand.velocity = (0.0, 0.0, 0.0)
                cam_v = (0.0, 0.0, 0.0)
            cand.last = (now, _vec(centre), cam)
            inside: list[tuple[tuple[float, float, float], int]] = []
            for p, g in points:
                rel = _sub(p, cam)
                if in_frame(_dot(rel, right), _dot(rel, down), _dot(rel, forward)):
                    inside.append((p, g))
            if not inside:
                continue
            visible_by_point: dict[
                tuple[int, tuple[float, float, float]], tuple[tuple[float, float, float], int]
            ] = {}
            for p, _g in inside:
                hit = self._visible_point(data, cam, p, cand)
                if hit is not None:
                    actual, hit_geom = hit
                    key = (hit_geom, tuple(round(v, 7) for v in actual))
                    visible_by_point[key] = hit
            visible = list(visible_by_point.values())
            entry: dict[str, Any] = {
                "id": cand.id,
                "kind": cand.kind,
                "cls": cand.cls,
                "n": len(points),
                "fov": len(inside),
                "vis": len(visible),
            }
            if not visible:
                seen.append(entry)  # evaluator metadata only; no hidden geometry reaches senses
                continue
            use = visible
            c = tuple(sum(p[0][k] for p in use) / len(use) for k in range(3))
            rel = _sub(c, cam)
            dist = max(_norm(rel), 1e-3)
            h = _heading_frame(yaw, rel)
            entry["b"] = round(math.atan2(h[1], h[0]), 4)
            entry["e"] = round(math.atan2(h[2], math.hypot(h[0], h[1])), 4)
            fwd = _dot(rel, forward)
            entry["cx"] = round(math.atan2(-_dot(rel, down), fwd), 4)  # upright image: + right
            entry["cy"] = round(math.atan2(_dot(rel, right), fwd), 4)  # upright image: + down
            spread = 0.0
            dirs = [_sub(p, cam) for p, _ in use]
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    spread = max(spread, _angle(dirs[i], dirs[j]))
            entry["size"] = round(spread, 4)
            entry["dist"] = round(dist, 3)
            unit = (rel[0] / dist, rel[1] / dist, rel[2] / dist)
            v_obj = cand.velocity
            v_rel = _sub(v_obj, cam_v)
            entry["vr"] = round(-_dot(v_rel, unit), 3)  # closing speed, + = coming nearer
            along = _dot(v_obj, unit)
            across = _sub(v_obj, (unit[0] * along, unit[1] * along, unit[2] * along))
            entry["om"] = round(_norm(across) / dist, 4)  # its own image motion, rad/s
            entry["lux"] = round(self._lux(state, c), 3)
            # In this upright image +right points down. Use the lowest visible silhouette point,
            # then a two-pixel adjacent image ray; never infer range from an occluded mesh bottom.
            low = max(
                use,
                key=lambda item: (
                    _dot(_sub(item[0], cam), right) / max(_dot(_sub(item[0], cam), forward), 1e-6)
                ),
            )[0]
            floor_point = self._floor_under(data, low, cam, right, down, forward, cand)
            entry["rng"] = (
                round(math.hypot(floor_point[0] - cam[0], floor_point[1] - cam[1]), 3)
                if floor_point is not None
                else None
            )
            if cand.kind == "nest":
                entry["glow"] = backlight
                entry["face"] = round(self._tag_facing(data, cand.geoms[0], cam), 4)
                # (dock:build 2026-09-28) the tag as duck-eye NB-v1's pose gives it: its centre
                # from the trunk origin in the level heading frame (m) and the heading of its
                # printed face's normal there (rad; π when it faces the duck square on) — truth
                # for the senses' tag-reader model only
                entry.update(self._tag_pose(data, cand.geoms[0], trunk, yaw, cam))
            seen.append(entry)
        try:
            floor_line = self._floor_line(data, cam, (right, down, forward), trunk, yaw)
        except Exception as error:  # a broken ray never stops the samples
            self._error(f"floor: {type(error).__name__}: {error}")
            floor_line = None
        return {
            "t": round(now, 4),
            "cam": {
                "pos_mm": wk.mm(cam),
                "fwd": [round(v, 4) for v in forward],
                "lux": round(cam_lux, 3),
                "backlight": backlight,
                "height_m": round(cam[2], 3),
            },
            "duck": {"yaw": round(yaw, 4), "trunk_mm": wk.mm(trunk)},
            "seen": seen,
            "floor": floor_line,
        }

    # ── the floor line's truth (duck-eye FL-v1's shape: five sectors across the picture) ──

    def _floor_line(
        self,
        data: Any,
        cam: Sequence[float],
        axes: Sequence[Sequence[float]],
        trunk: Sequence[float],
        yaw: float,
    ) -> dict[str, Any] | None:
        """How far the floor is clear in the five sectors across the delivered picture — what a
        perfect floor line would say (duck-eye FL-v1, :mod:`duckmind.eye.floorline`): per sector
        the nearest thing the duck's body would meet (rays at :data:`FLOOR_RAY_HEIGHTS_M` from
        the trunk origin along the sector's level direction; the duck's own parts skipped), from
        the trunk origin, level frame, capped at what the picture shows of the floor (its bottom
        edge's nearest floor, its top edge's farthest, FL-v1's 1.5 m). ``end`` is ``thing``
        (a loose thing, a person's legs), ``wall`` (walls, furniture, the nest) or ``unknown``
        (nothing in range). None when the picture is not upright enough to read the floor (the
        duck fell, the head rolled). World truth for the senses' model only."""
        right, down, forward = (_vec(a) for a in axes)
        if abs(right[2]) > 0.55 and abs(down[2]) > 0.55:
            return None  # the picture's rows are far from level (rolled): no floor line
        height = float(cam[2])
        if height < 0.03 or height > 0.6:
            return None
        # the picture's horizontal (upright) is −down; its vertical +right (camera_model's fix)
        cx_lo, cx_hi = -math.atan(Y_MAX), -math.atan(Y_MIN)  # + = the picture's right
        span = (cx_hi - cx_lo) / FLOOR_SECTORS

        def level_bearing(cx: float, row: float) -> tuple[float, float]:
            """The level direction of a picture column at a row (tan of the vertical angle):
            its bearing in the heading frame and how far below the horizon it looks (rad)."""
            y = -math.tan(cx)
            d = tuple(forward[k] + y * down[k] + row * right[k] for k in range(3))
            flat = math.hypot(d[0], d[1])
            bearing = _wrap(math.atan2(d[1], d[0]) - yaw)
            return bearing, -math.atan2(d[2], max(flat, 1e-9))

        ahead = (math.cos(yaw), math.sin(yaw))
        lens_ahead = (cam[0] - trunk[0]) * ahead[0] + (cam[1] - trunk[1]) * ahead[1]
        sectors: list[dict[str, Any]] = []
        near_all: list[float] = []
        for k in range(FLOOR_SECTORS):
            cx_mid = cx_lo + (k + 0.5) * span
            b_mid, _ = level_bearing(cx_mid, 0.0)
            _, dip_bottom = level_bearing(cx_mid, X_MAX)
            _, dip_top = level_bearing(cx_mid, X_MIN)
            # the nearest floor the picture shows (its bottom edge) and the farthest (its top)
            near = height / math.tan(dip_bottom) + lens_ahead if dip_bottom > 0.02 else FLOOR_MAX_M
            far = height / math.tan(dip_top) + lens_ahead if dip_top > 0.02 else FLOOR_MAX_M
            far = min(far, FLOOR_MAX_M)
            near_all.append(near)
            best = FLOOR_MAX_M
            end = "unknown"
            for frac in (-0.3, 0.0, 0.3):
                b, _ = level_bearing(cx_mid + frac * span, 0.0)
                hit, kind = self._floor_ray(data, trunk, yaw + b)
                if hit < best:
                    best, end = hit, kind
            if best > far:
                free, end = far, "unknown"
            elif best < near:
                free = near  # the thing fills the picture's bottom: no floor seen before it
            else:
                free = best
            b_left, _ = level_bearing(cx_mid - 0.5 * span, 0.0)
            b_right, _ = level_bearing(cx_mid + 0.5 * span, 0.0)
            sectors.append(
                {
                    "b": round(b_mid, 4),
                    "w": round(abs(_wrap(b_left - b_right)), 4),
                    "free": round(max(0.0, free), 3),
                    "end": end,
                    "hit": round(best, 3),
                }
            )
        return {"sectors": sectors, "near": round(min(near_all), 3)}

    def _floor_ray(self, data: Any, trunk: Sequence[float], heading: float) -> tuple[float, str]:
        """The nearest thing along a level heading from the trunk origin at the ray heights,
        skipping the duck's own parts: (distance m, "thing" | "wall"), or (FLOOR_MAX_M,
        "unknown")."""
        import mujoco
        import numpy as np

        unit = np.array([math.cos(heading), math.sin(heading), 0.0])
        robot = set(self.robot.bodies) if self.robot is not None else set()
        best = FLOOR_MAX_M
        kind = "unknown"
        for z in FLOOR_RAY_HEIGHTS_M:
            start = 0.0
            for _ in range(4):  # past the duck's own feet and chest
                pnt = np.array([trunk[0] + unit[0] * start, trunk[1] + unit[1] * start, z])
                self._ray_hit[0] = -1
                got = float(mujoco.mj_ray(self.model, data, pnt, unit, None, 1, -1, self._ray_hit))
                hit = int(self._ray_hit[0])
                if got < 0 or hit < 0:
                    break
                body = int(self.model.geom_bodyid[hit])
                if body in robot or hit in self.hand_geoms:
                    start += got + 0.01
                    continue
                dist = start + got
                if dist < best:
                    best = dist
                    kind = "thing" if body in self._thing_bodies else "wall"
                break
        return best, kind

    def _tag_pose(
        self, data: Any, geom: int, trunk: Sequence[float], yaw: float, cam: Sequence[float]
    ) -> dict[str, float]:
        """The tag's centre from the trunk origin (``tx``, ``ty``: level heading frame, m) and
        the heading of its printed face's normal in that frame (``tn``, rad): the normal is the
        tag's thinnest axis, turned toward the lens (a tag is read from its front only)."""
        size = [float(v) for v in self.model.geom_size[geom]]
        axis = min(range(3), key=lambda k: size[k])
        m = [float(v) for v in data.geom_xmat[geom]]
        normal = (m[axis], m[3 + axis], m[6 + axis])
        centre = _vec(data.geom_xpos[geom])
        if _dot(normal, _sub(cam, centre)) < 0.0:
            normal = (-normal[0], -normal[1], -normal[2])
        rel = _heading_frame(yaw, _sub(centre, trunk))
        n = _heading_frame(yaw, normal)
        return {
            "tx": round(rel[0], 4),
            "ty": round(rel[1], 4),
            "tn": round(math.atan2(n[1], n[0]), 4),
        }

    def _tag_facing(self, data: Any, geom: int, cam: Sequence[float]) -> float:
        """The angle between the tag's face normal and the direction to the lens (0 = head on)."""
        size = [float(v) for v in self.model.geom_size[geom]]
        axis = min(range(3), key=lambda k: size[k])
        m = [float(v) for v in data.geom_xmat[geom]]
        normal = (m[axis], m[3 + axis], m[6 + axis])
        to_cam = _sub(cam, _vec(data.geom_xpos[geom]))
        a = _angle(normal, to_cam)
        return min(a, math.pi - a)

    def _floor_under(
        self,
        data: Any,
        point: Sequence[float],
        cam: Sequence[float],
        right: Sequence[float],
        down: Sequence[float],
        forward: Sequence[float],
        cand: Candidate | None = None,
    ) -> tuple[float, float, float] | None:
        """Find floor immediately below a visible candidate silhouette; unknown support has no range.

        A support point can project inside a curved or sparsely sampled silhouette. March a small,
        calibrated number of pixels down from it, carrying forward only actual first-hit surfaces
        of this candidate. Any unrelated hit, off-frame ray, or exhausted budget stays unknown.
        """
        import mujoco
        import numpy as np

        vec = _sub(point, cam)
        dist = _norm(vec)
        if dist < 1e-6:
            return None
        fwd = _dot(vec, forward)
        if fwd <= 1e-6:
            return None
        # The delivered upright frame maps raw camera +right to screen-down: this is raw-u,
        # despite the source camera's unrotated axes. Invert lens distortion for every pixel.
        raw_x = _dot(vec, right) / fwd
        raw_y = _dot(vec, down) / fwd
        xd, yd = cm.distort(raw_x, raw_y, _DIST)
        u = _K.cx + _K.fx * xd
        v = _K.cy + _K.fy * yd
        last_surface = tuple(float(point[k]) for k in range(3))
        # P1c A2: the loop below returns on every non-carry outcome, so reaching a
        # pixel beyond FLOOR_ASSOCIATION_MAX_PX is only possible through consecutive
        # candidate carries. The extended cap therefore changes nothing except turning
        # long-carry-then-adjacent-floor into a floor hit; all other exits are identical.
        for pixel in range(
            FLOOR_ASSOCIATION_STEP_PX, FLOOR_ASSOCIATION_CARRY_MAX_PX + 1, FLOOR_ASSOCIATION_STEP_PX
        ):
            x2, y2 = cm.undistort_pixel(u + pixel, v, _K, _DIST)
            ray = [float(forward[k]) + x2 * float(right[k]) + y2 * float(down[k]) for k in range(3)]
            ray_norm = _norm(ray)
            ray = [component / ray_norm for component in ray]
            if not in_frame(_dot(ray, right), _dot(ray, down), _dot(ray, forward)):
                return None
            self._ray_hit[0] = -1
            got = float(
                mujoco.mj_ray(
                    self.model,
                    data,
                    np.array(cam, dtype=float),
                    np.array(ray),
                    self._ray_group,
                    1,
                    self.cam_body,
                    self._ray_hit,
                )
            )
            hit = int(self._ray_hit[0])
            if got < 0 or hit < 0:
                return None
            actual = (
                float(cam[0] + ray[0] * got),
                float(cam[1] + ray[1] * got),
                float(cam[2] + ray[2] * got),
            )
            if hit in self._floor_geoms:
                if _norm(_sub(actual, last_surface)) > 0.06:
                    return None
                return actual
            if cand is None:
                return None
            is_candidate = hit in cand.geoms or (
                not cand.static and int(self.model.geom_bodyid[hit]) in cand.bodies
            )
            if not is_candidate:
                return None
            last_surface = actual
        return None

    def _light_state(self) -> Any:
        fn = self.lights
        if fn is None:
            return None
        try:
            return fn()
        except Exception as error:
            self._error(f"lights: {type(error).__name__}: {error}")
            return None

    @staticmethod
    def _lux(state: Any, point: Sequence[float]) -> float:
        if state is None:
            return 300.0
        try:
            return float(state.lux_at((float(point[0]), float(point[1]), float(point[2]))))
        except Exception:
            return 300.0

    # ── sounds ───────────────────────────────────────────────────────────────

    def head(self, data: Any) -> tuple[float, float, float]:
        site = self.head_site if self.head_site >= 0 else None
        if site is not None:
            return _vec(data.site_xpos[site])
        return _vec(data.xpos[self.robot.trunk])

    def add_sound(
        self,
        kind: str,
        pos: Sequence[float],
        *,
        t0: float,
        seconds: float,
        db1m: float | None = None,
        env: str | None = None,
        src: str = "",
        name: bool = False,
        utterance: str | None = None,
        by: str = "",
    ) -> SoundEvent:
        with self._lock:
            self._sound_id += 1
            event = SoundEvent(
                id=self._sound_id,
                kind=kind,
                t0=float(t0),
                t1=float(t0) + max(0.02, float(seconds)),
                db1m=float(SOUND_DB_1M.get(kind, 60.0) if db1m is None else db1m),
                pos=_vec(pos),
                env=env
                or (
                    "impulse"
                    if kind in IMPULSIVE
                    else "speech"
                    if kind in ("voice", "name", "hello")
                    else "steady"
                ),
                src=src,
                name=name,
                utterance=utterance,
                by=by or ("person" if src.startswith("person:") else ""),
            )
            self.sounds.append(event)
            return event

    def _levels(self, data: Any, now: float) -> None:
        head = self.head(data)
        keep: list[SoundEvent] = []
        for event in self.sounds:
            if event.t1 + KEEP_EVENTS_S < now and event not in (self.tv, self.music):
                continue
            keep.append(event)
            distance = _norm(_sub(event.pos, head))
            event.wall = self._wall_between(data, event.pos, head)
            event.db = level_at(event.db1m, distance, wall=event.wall)
        self.sounds = keep

    def _wall_between(self, data: Any, a: Sequence[float], b: Sequence[float]) -> bool:
        """A static wall (a layout ``wall`` element) between two points."""
        import mujoco
        import numpy as np

        vec = _sub(b, a)
        dist = _norm(vec)
        if dist < 0.05:
            return False
        unit = np.array([vec[0] / dist, vec[1] / dist, vec[2] / dist])
        self._ray_hit[0] = -1
        got = float(
            mujoco.mj_ray(
                self.model,
                data,
                np.array(a, dtype=float),
                unit,
                np.array((1, 0, 0, 0, 0, 0), dtype=np.uint8),
                1,
                -1,
                self._ray_hit,
            )
        )
        hit = int(self._ray_hit[0])
        if got < 0 or hit < 0 or got >= dist - 0.05:
            return False
        name = wk.geom_name(self.model, hit)
        return "wall" in name

    def _footsteps(self, data: Any, now: float) -> None:
        people = self.people
        if people is None:
            return
        for who, walker in getattr(people, "walkers", {}).items():
            if not getattr(walker, "here", False):
                self._walked.pop(who, None)
                continue
            x, y = float(walker.pos[0]), float(walker.pos[1])
            last = self._walked.get(who)
            if last is None or walker.state not in WALKING:
                self._walked[who] = (x, y, 0.0)
                continue
            travelled = last[2] + math.hypot(x - last[0], y - last[1])
            while travelled >= STEP_EVERY_M:
                travelled -= STEP_EVERY_M
                self.add_sound("footsteps", (x, y, 0.02), t0=now, seconds=0.06, src=f"person:{who}")
            self._walked[who] = (x, y, travelled)

    def _impacts(self, now: float) -> None:
        props = self.props
        if props is None:
            return
        for item in list(getattr(props, "impacts", ())):
            seq = int(item.get("seq", 0))
            if seq <= self._impact_seq:
                continue
            self._impact_seq = seq
            if item.get("body") == "duck" and item.get("other") in (None, "floor"):
                continue  # its own feet and sitting down: part of its own body noise
            impulse = float(item.get("impulse_mns", 0))
            if impulse < 8.0:
                continue
            by = self._impact_by(item)
            pos = [v / 1000.0 for v in item.get("pos_mm", [0, 0, 0])]
            kind = "bang" if impulse >= 120.0 else "knock"
            self.add_sound(
                kind,
                pos,
                t0=float(item.get("t", now)),
                seconds=0.05,
                db1m=impact_db(impulse),
                src=f"thing:{item.get('body')}",
                by=by,
            )

    def _impact_by(self, item: Mapping[str, Any]) -> str:
        """What made an impact sound, from the world's own record: the duck's contact is its own
        ("self"): the duck against anything, a thing against the duck, or an impact in which either
        thing had the duck against it in the last :data:`SELF_IMPACT_S` (it pushed or knocked it, and
        it fell, hit a wall, or hit another thing). One step further counts too: a thing that a
        self-moved thing hit in that window makes its own knocks the duck's for the window (the
        chain stops there). A person's body is "person"; any other thing's knock is "thing"."""
        body, other = str(item.get("body") or ""), item.get("other")
        t = float(item.get("t", 0.0))
        if body == "duck" or other == "duck":
            return "self"
        if str(other or "").startswith("person"):
            return "person"
        keys = [f"thing:{body}"] + ([f"thing:{other}"] if other else [])
        for key in keys:
            last = self._duck_touched.get(key)
            if last is not None and t - last <= SELF_IMPACT_S:
                # the duck moved it: whatever it hits has been hit by something the duck moved
                for hit in keys:
                    if hit != key:
                        self._chain_touched[hit] = t
                return "self"
        for key in keys:
            last = self._chain_touched.get(key)
            if last is not None and t - last <= SELF_IMPACT_S:
                return "self"
        return "thing"

    # ── touch ──────────────────────────────────────────────────────────────────

    def _touchers(self, data: Any) -> list[tuple[str, tuple[int, ...]]]:
        out: list[tuple[str, tuple[int, ...]]] = []
        for cand in self.candidates:
            if cand.kind == "nest" or not self._present(data, cand):
                continue
            out.append((cand.id, cand.geoms))
        return out

    def _site(self, data: Any, body: int, point: Sequence[float]) -> str:
        name = self.duck_body_names.get(body, "")
        if name in HEAD_BODIES:
            if (
                self.beak_site >= 0
                and _norm(_sub(point, _vec(data.site_xpos[self.beak_site]))) < BEAK_RADIUS_M
            ):
                return "beak"
            return "head"
        if name in LEFT_LEG:
            return "left"
        if name in RIGHT_LEG:
            return "right"
        trunk = _vec(data.xpos[self.robot.trunk])
        m = [float(v) for v in data.xmat[self.robot.trunk]]
        rel = _sub(point, trunk)
        local = (
            m[0] * rel[0] + m[3] * rel[1] + m[6] * rel[2] - TRUNK_CENTRE[0],
            m[1] * rel[0] + m[4] * rel[1] + m[7] * rel[2] - TRUNK_CENTRE[1],
            m[2] * rel[0] + m[5] * rel[1] + m[8] * rel[2] - TRUNK_CENTRE[2],
        )
        ax, ay, az = abs(local[0]), abs(local[1]), local[2]
        if az > 0.02 and az >= 0.6 * max(ax, ay):
            return "back"
        if local[0] < 0.0 and ax >= ay:
            return "tail"
        if ay >= 0.015:
            return "left" if local[1] > 0 else "right"
        return "back"

    def _touch(self, data: Any, now: float) -> None:
        """Contacts now (50 Hz): per toucher its deepest point on the shell, one episode per
        toucher (its site switches only after :data:`SITE_SWITCH_CHECKS` checks elsewhere; it
        starts at a gap of :data:`TOUCH_GAP_M` and goes on up to :data:`TOUCH_END_GAP_M`, and
        a gap shorter than :data:`GAP_CHECKS` checks does not end it: a mesh's nearest point
        jumps between neighbouring parts, a standing duck sways against a toy). Speeds are the
        toucher's motion relative to the trunk since its previous check while near, so the
        first check of a contact already knows how fast it came."""
        import mujoco

        model = self.model
        trunk = _vec(data.xpos[self.robot.trunk])
        found: dict[str, tuple[str, float, tuple[float, float, float], int]] = {}
        near_now: dict[
            str, tuple[float, tuple[float, ...], int, tuple[float, ...], tuple[float, ...]]
        ] = {}
        for who, geoms in self._touchers(data):
            near = [
                g
                for g in geoms
                if _norm(_sub(_vec(data.geom_xpos[g]), trunk)) - float(model.geom_rbound[g])
                < NEAR_M
            ]
            if not near:
                continue
            best: tuple[float, int, int, tuple[float, ...]] | None = None
            for g in near:
                gpos = _vec(data.geom_xpos[g])
                grad = float(model.geom_rbound[g])
                for dg in self.duck_geoms:
                    reach = grad + float(model.geom_rbound[dg]) + 0.01
                    if _norm(_sub(_vec(data.geom_xpos[dg]), gpos)) > reach:
                        continue
                    d = float(mujoco.mj_geomDistance(model, data, g, dg, 0.02, self._fromto))
                    if best is None or d < best[0]:
                        best = (d, g, dg, tuple(float(v) for v in self._fromto))
            body = int(model.geom_bodyid[near[0]])
            rel_now = _sub(_vec(data.xpos[body]), trunk)
            last = self._near.get(who)
            v_prev = (0.0, 0.0, 0.0)
            if last is not None and last[2] == body and 1e-6 < now - last[0] < 0.2:
                dt = now - last[0]
                v_prev = tuple((rel_now[k] - last[1][k]) / dt for k in range(3))  # type: ignore[assignment]
            near_now[who] = (now, rel_now, body, v_prev, last[3] if last is not None else v_prev)
            limit = TOUCH_END_GAP_M if who in self.contacts else TOUCH_GAP_M
            if best is None or best[0] > limit:
                continue
            d, g, dg, ft = best
            duck_point = (ft[3], ft[4], ft[5])
            duck_body = int(model.geom_bodyid[dg])
            site = self._site(data, duck_body, duck_point)
            depth = max(0.0, -d) * 1000.0
            found[who] = (site, depth, duck_point, duck_body)
        for who, (site, depth, duck_point, duck_body) in found.items():
            if who.startswith("thing:"):
                self._duck_touched[who] = now  # (for the sounds this thing makes next)
            contact = self.contacts.get(who)
            if contact is None:
                self._touch_seq += 1
                contact = Contact(key=f"{who}#{self._touch_seq}", by=who, site=site, t0=now, t1=now)
                self.contacts[who] = contact
            _, _, _, v, v_before = near_now[who]
            out = _sub(duck_point, _vec(data.xpos[duck_body]))
            n = _norm(out)
            normal = (0.0, 0.0, 1.0) if n < 1e-6 else (out[0] / n, out[1] / n, out[2] / n)
            along = _dot(v, normal)
            closing = max(0.0, -along)
            slide = _norm(_sub(v, (normal[0] * along, normal[1] * along, normal[2] * along)))
            if who not in self.contacts:  # a new contact: how fast it came, before any bounce
                closing = max(closing, -_dot(v_before, normal))
            if site != contact.site:
                if site == contact.pending_site:
                    contact.pending_n += 1
                else:
                    contact.pending_site, contact.pending_n = site, 1
                if contact.pending_n >= SITE_SWITCH_CHECKS:
                    contact.site, contact.pending_site, contact.pending_n = site, "", 0
            else:
                contact.pending_site, contact.pending_n = "", 0
            contact.t1 = now
            contact.missing = 0
            contact.now_depth_mm = depth
            contact.now_slide = slide
            contact.now_closing = closing
            contact.depth_mm = max(contact.depth_mm, depth)
            contact.closing = max(contact.closing, closing)
            contact.slide = max(contact.slide, slide)
        self._near = near_now
        for who in [k for k in self.contacts if k not in found]:
            contact = self.contacts[who]
            contact.missing += 1
            contact.now_depth_mm = contact.now_slide = contact.now_closing = 0.0
            if contact.missing >= GAP_CHECKS:
                contact.active = False
                self.ended.append(self.contacts.pop(who))

    def snapshot(self, world: Any) -> dict[str, Any]:
        data = world.data
        now = float(data.time)
        with self._lock:
            self._levels(data, now)
            sounds = [
                e.view()
                for e in self.sounds
                if e.t1 + KEEP_EVENTS_S >= now or e in (self.tv, self.music)
            ]
            touches = [c.view() for c in self.contacts.values()]
            touches += [c.view() for c in self.ended if c.t1 + KEEP_EVENTS_S >= now]
            gesture = self.gesture
            return {
                "v": VERSION,
                "sim_time": round(now, 4),
                "samples": list(self.samples),
                "sounds": sounds,
                "touches": touches,
                "stim": list(self.stims)[-8:],
                "gesture": None
                if gesture is None or gesture.done
                else {
                    "stim": gesture.stim,
                    "leg": gesture.legs[min(gesture.index, len(gesture.legs) - 1)].label,
                },
                "tv": self.tv is not None,
                "music": self.music is not None,
                "lamps_on": self._lamps_on,
                "errors": list(self.errors[-5:]),
            }

    # ── gestures (the world's lock is held while driving) ────────────────────

    def _hand_busy(self) -> bool:
        hand = self.hand
        if hand is None:
            return False
        return getattr(hand, "script", None) is not None or bool(getattr(hand, "holding", False))

    def _point(self, data: Any, follow: str) -> tuple[tuple[float, float, float], float]:
        robot = self.robot
        yaw = wk.heading_of([float(v) for v in data.xquat[robot.trunk]])
        if follow == "head" and self.head_site >= 0:
            return _vec(data.site_xpos[self.head_site]), yaw
        if follow == "beak" and self.beak_site >= 0:
            return _vec(data.site_xpos[self.beak_site]), yaw
        if follow == "camera":
            return _vec(data.cam_xpos[self.cam]), yaw
        return _vec(data.xpos[robot.trunk]), yaw

    def _set_hand(self, data: Any, pos: Sequence[float], quat: Sequence[float]) -> None:
        data.mocap_pos[self.hand_mocap] = [float(v) for v in pos]
        data.mocap_quat[self.hand_mocap] = [float(v) for v in quat]

    def _park(self, data: Any) -> None:
        if self.hand_mocap < 0:
            return
        pos = [float(v) for v in data.mocap_pos[self.hand_mocap]]
        data.mocap_pos[self.hand_mocap] = [pos[0], pos[1], PARK_Z_M]
        data.mocap_quat[self.hand_mocap] = [1.0, 0.0, 0.0, 0.0]

    def _end_gesture(self, data: Any, why: str = "") -> None:
        gesture = self.gesture
        if gesture is None:
            return
        gesture.done = True
        if why:
            gesture.why_zh = why
        if not self._hand_busy():
            self._park(data)
        self._clear_force(data)

    def _clear_force(self, data: Any) -> None:
        try:
            data.xfrc_applied[self.robot.trunk] = [0.0] * 6
        except Exception:
            pass

    def _min_gap(self, data: Any, bodies: Sequence[str]) -> float:
        """The hand's nearest gap to these duck bodies' shells (m; negative: through them)."""
        import mujoco

        model = self.model
        wanted = {b for b, name in self.duck_body_names.items() if name in bodies}
        best = 1.0
        for g in self.hand_geoms:
            gpos = _vec(data.geom_xpos[g])
            for dg in self.duck_geoms:
                if int(model.geom_bodyid[dg]) not in wanted:
                    continue
                if (
                    _norm(_sub(_vec(data.geom_xpos[dg]), gpos))
                    > float(model.geom_rbound[g]) + float(model.geom_rbound[dg]) + 0.03
                ):
                    continue
                d = float(mujoco.mj_geomDistance(model, data, g, dg, 0.05, self._fromto))
                best = min(best, d)
        return best

    def _drive(self, gesture: Gesture, data: Any, now: float) -> None:
        guard = 0
        while not gesture.done and guard < 4:
            guard += 1
            if gesture.index >= len(gesture.legs):
                self._end_gesture(data)
                return
            leg = gesture.legs[gesture.index]
            if leg.t0 is None:
                leg.t0 = now
                if leg.kind != "press":
                    gesture.dz = 0.0  # a new press comes down from above again
                leg.p0 = _vec(data.mocap_pos[self.hand_mocap])
                q = [float(v) for v in data.mocap_quat[self.hand_mocap]]
                leg.q0 = (q[0], q[1], q[2], q[3])
            elapsed = now - leg.t0
            tau = 1.0 if leg.seconds <= 0 else min(1.0, elapsed / leg.seconds)
            s = wk.min_jerk(tau)
            base, yaw = self._point(data, leg.follow)
            quat = list(leg.quat) if leg.quat is not None else wk.yaw_quat(yaw + leg.yaw_off)
            if leg.kind == "move":
                goal = leg.to or leg.p0
                pos = [a + (b - a) * s for a, b in zip(leg.p0, goal, strict=True)]
                self._set_hand(data, pos, wk.slerp(list(leg.q0), quat, s))
            elif leg.kind in ("track", "press"):
                off = leg.offset
                if leg.kind == "press":
                    d = leg.press_dir
                    off = (
                        off[0] + leg.slide * tau + d[0] * gesture.dz,
                        off[1] + d[1] * gesture.dz,
                        off[2] + d[2] * gesture.dz,
                    )
                goal = _vec([b + o for b, o in zip(base, _to_world(yaw, off), strict=True)])
                pos = [a + (b - a) * s for a, b in zip(leg.p0, goal, strict=True)]
                self._set_hand(data, pos, wk.slerp(list(leg.q0), quat, s))
                if leg.kind == "press" and leg.bodies:
                    gap = self._min_gap(data, leg.bodies)
                    err = gap - (-leg.press_mm / 1000.0)  # + too far off, − pressing too deep
                    if err < 0.0:
                        gesture.dz = min(0.08, gesture.dz - err)  # back off at once
                    else:  # closing in only here, slowly (≤ 0.1 m/s): a pat, not a slap
                        step = leg.press_speed * 0.005  # per 5 ms physics step
                        gesture.dz = max(-0.15, gesture.dz - min(step, err * 0.5))
            elif leg.kind == "wave":
                centre = leg.to or leg.p0
                phase = math.sin(2.0 * math.pi * leg.hz * elapsed)
                ramp = min(1.0, elapsed / 0.3, max(0.0, (leg.seconds - elapsed) / 0.3))
                pos = [
                    c + a * leg.amp * phase * ramp for c, a in zip(centre, leg.axis, strict=True)
                ]
                self._set_hand(data, pos, quat)
            else:  # hold where it is
                self._set_hand(data, leg.p0, quat)
            if leg.force is not None:
                data.xfrc_applied[self.robot.trunk] = [*leg.force, 0.0, 0.0, 0.0]
            if tau < 1.0:
                return
            if leg.force is not None:
                self._clear_force(data)
            gesture.index += 1

    def _start_gesture(
        self, data: Any, stim: str, legs: list[GLeg], start: Sequence[float]
    ) -> None:
        if self.hand_mocap < 0:
            raise ValueError(msg("neuro.world_refusal.9eea94"))
        if self._hand_busy():
            raise ValueError(msg("neuro.world_refusal.563cc8"))
        if self.gesture is not None and not self.gesture.done:
            self._end_gesture(data, "換成新的手勢")
        if float(data.mocap_pos[self.hand_mocap][2]) < PARK_Z_M / 2:
            yaw = wk.heading_of([float(v) for v in data.xquat[self.robot.trunk]])
            self._set_hand(data, start, wk.yaw_quat(yaw))
        self.gesture = Gesture(stim=stim, legs=legs, started=float(data.time))

    def _hand_from(self, data: Any, who: str | None) -> tuple[float, float, float]:
        """Where a hand comes from: that person's hand when they are here, else above and a
        little behind the duck's head (the family reaching down from where it stands)."""
        people = self.people
        if who and people is not None and hasattr(people, "hand_point"):
            point = people.hand_point(who)
            if point is not None:
                return _vec(point)
        head, yaw = self._point(data, "head")
        back = _to_world(yaw, (-0.25, 0.1, 0.0))
        return (head[0] + back[0], head[1] + back[1], head[2] + 0.55)

    # ── the stimuli ──────────────────────────────────────────────────────────

    def stimulate(self, request: dict[str, Any]) -> dict[str, Any]:
        """One page button as a world event (call WITHOUT the world's lock)."""
        stim = str(request.get("stim") or request.get("kind") or "")
        if stim not in STIMULI:
            raise ValueError(
                msg("neuro.world_refusal.bc8d00", given_id=stim, known_ids=lst(STIMULI))
            )
        world = self.world
        if world is None:
            raise ValueError(msg("neuro.world_refusal.671851"))
        handler = getattr(self, f"_stim_{stim}")
        answer: dict[str, Any] = dict(handler(world, request))
        now = self._now(world)  # (never the world's lock inside this part's own lock)
        answer.setdefault("ok", True)
        answer.setdefault("note_zh", STIMULI[stim])
        with self._lock:
            self._stim_id += 1
            record = {
                "id": self._stim_id,
                "stim": stim,
                "zh": STIMULI[stim],
                "t": round(now, 3),
                **{k: v for k, v in answer.items() if k in ("where_mm", "who", "note_zh")},
            }
            self.stims.append(record)
            answer["id"] = self._stim_id
        answer["stim"] = stim
        return answer

    def _who_here(self) -> str | None:
        people = self.people
        if people is None:
            return None
        for who in FAMILY_ORDER:
            walker = getattr(people, "walkers", {}).get(who)
            if walker is not None and getattr(walker, "here", False):
                return who
        return None

    def _source(
        self, data: Any, request: dict[str, Any], height: float
    ) -> tuple[tuple[float, float, float], str]:
        """Where a sound comes from: a person here (their hands or mouth), else somewhere
        :data:`OFFSCREEN_R_M` away inside the room (the family is home, just not drawn)."""
        who = request.get("who") or self._who_here()
        people = self.people
        if who and people is not None:
            walker = getattr(people, "walkers", {}).get(str(who))
            if walker is not None and getattr(walker, "here", False):
                scale = float(walker.persona.height_m) / 1.72
                return (float(walker.pos[0]), float(walker.pos[1]), height * scale), f"person:{who}"
        trunk = _vec(data.xpos[self.robot.trunk])
        bounds = self._bounds()
        for _ in range(24):
            r = self.rng.uniform(*OFFSCREEN_R_M)
            a = self.rng.uniform(-math.pi, math.pi)
            x, y = trunk[0] + r * math.cos(a), trunk[1] + r * math.sin(a)
            if bounds is None or (
                bounds[0] + 0.3 < x < bounds[2] - 0.3 and bounds[1] + 0.3 < y < bounds[3] - 0.3
            ):
                return (x, y, height), "offscreen"
        return (trunk[0] + 1.5, trunk[1], height), "offscreen"

    def _bounds(self) -> tuple[float, float, float, float] | None:
        layout = getattr(self.people, "layout", None) or getattr(self.props, "layout", None)
        rect = getattr(layout, "bounds_mm", None)
        if not rect:
            return None
        return tuple(float(v) / 1000.0 for v in rect)  # type: ignore[return-value]

    def _now(self, world: Any) -> float:
        with world.lock:
            return float(world.data.time)

    def _sound(
        self,
        world: Any,
        request: dict[str, Any],
        kind: str,
        seconds: float,
        height: float,
        **kw: Any,
    ) -> dict[str, Any]:
        with world.lock:
            data = world.data
            pos, src = self._source(data, request, height)
            now = float(data.time)
        loud = request.get("db1m")
        event = self.add_sound(
            kind,
            pos,
            t0=now,
            seconds=seconds,
            db1m=float(loud) if loud is not None else None,
            src=src,
            by="person" if src.startswith("person:") else "page",
            **kw,
        )
        return {"where_mm": wk.mm(pos), "who": src, "sound": event.view()}

    def _stim_clap(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        answer = self._sound(world, request, "clap", 0.04, HAND_H_M)
        return {**answer, "note_zh": msg("neuro.world.clap.9923")}

    def _stim_call_name(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        answer = self._sound(world, request, "name", 0.7, MOUTH_H_M, name=True)
        return {**answer, "note_zh": msg("neuro.world.call_name.baf4")}

    def _stim_voice(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        seconds = float(request.get("seconds", 3.0))
        return self._sound(world, request, "voice", seconds, MOUTH_H_M)

    def _stim_hello(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        return self._sound(world, request, "hello", 0.6, MOUTH_H_M)

    def utterances(self) -> UtteranceLibrary | None:
        """The scripted lines (``neuro_utterances``), read once; None when there are none."""
        if not self._utterances_read:
            self._utterances_read = True
            try:
                self._utterances = UtteranceLibrary.default()
            except (OSError, ValueError, KeyError):
                self._utterances = None
        return self._utterances

    def _speaker_place(
        self, world: Any, request: dict[str, Any], who: str | None
    ) -> tuple[tuple[float, float, float], str]:
        """Where a person speaks from: the place asked for (``where_mm`` [x, y]), else where that
        person (or whoever is home) stands, else somewhere in the room (:meth:`_source`)."""
        where = request.get("where_mm")
        if where is not None:
            try:
                x, y = float(where[0]), float(where[1])
            except (TypeError, ValueError, IndexError, KeyError):
                raise ValueError(msg("neuro.world_refusal.say_where")) from None
            if not (math.isfinite(x) and math.isfinite(y)) or max(abs(x), abs(y)) > MAX_PLACE_MM:
                raise ValueError(msg("neuro.world_refusal.say_where"))
            scale = 1.0
            if who:
                try:
                    from duckmind.world.people import persona

                    scale = persona(who).height_m / 1.72
                except ValueError:
                    scale = 1.0  # (a speaker who is not one of the family: a grown-up's mouth)
            return (x / 1000.0, y / 1000.0, MOUTH_H_M * scale), f"person:{who or 'guest'}"
        with world.lock:
            return self._source(world.data, {"who": who} if who else {}, MOUTH_H_M)

    def _stim_say(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        """A family member says a scripted line (``utterance``: its id in
        :mod:`~duckmind.world.neuro_utterances`) from a place (``where_mm``, else where ``who``
        stands), or — ``live: true`` — speaks into a browser microphone for ``seconds`` (the
        microphone stream plays the audio that comes in). A sound event like the others (its
        level at the duck's head from the distance, a wall between), so the hearing model hears
        the onset as it hears any voice; ``duck_name``: the line calls the duck by name only
        when it is this one's (a name is data)."""
        who = str(request["who"]) if request.get("who") else None
        live = bool(request.get("live"))
        if live:
            seconds = min(3.0, max(0.2, _finite(request.get("seconds"), 1.0)))
            marker: str = f"{LIVE_PREFIX}{who or 'guest'}"
            kind, called, line = "voice", False, None
        else:
            library = self.utterances()
            if library is None:
                raise ValueError(msg("neuro.world_refusal.say_no_lines"))
            asked = str(request.get("utterance") or "")
            if asked not in library.lines:
                raise ValueError(
                    msg(
                        "neuro.world_refusal.say_unknown",
                        given_id=asked,
                        known_ids=lst(library.ids()),
                    )
                )
            line = library.get(asked)
            marker, seconds = line.id, line.duration_s
            duck = request.get("duck_name")
            called = bool(line.name) and (not duck or str(duck) == line.name)
            kind = "name" if called else "hello" if line.intent == "greet" else "voice"
        pos, src = self._speaker_place(world, request, who)
        now = self._now(world)
        if who and request.get("where_mm") is not None:
            pos = self._stand_speaker(world, who, pos, now)
        delay = min(10.0, max(0.0, _finite(request.get("delay_s"), 0.0)))
        loud = _finite(request.get("db1m"), math.nan)  # (nan: the kind's own level)
        event = self.add_sound(
            kind,
            pos,
            t0=now + delay,
            seconds=seconds,
            db1m=None if math.isnan(loud) else loud,
            env="speech",
            src=src,
            name=called,
            utterance=marker,
        )
        answer: dict[str, Any] = {"where_mm": wk.mm(pos), "who": src, "sound": event.view()}
        if line is not None:
            answer["utterance"] = line.view()
            answer["note_zh"] = msg("neuro.world.say.note", text=line.text)
        return answer

    def _stand_speaker(
        self, world: Any, who: str, pos: tuple[float, float, float], now: float
    ) -> tuple[float, float, float]:
        """The family member who speaks from a place is there in the room: their body stands at the
        place (:mod:`~duckmind.world.people`'s ``stand``: the nearest free place, apart from anyone
        else, facing the duck), so the duck's own camera sees them when it looks — the voice alone
        never put a person in front of its eyes. They stay :data:`STAND_AFTER_S` after the last
        thing they said and then walk out (:meth:`pump`). Returns the place the voice comes from
        (the body's, when it stands: it may be a step from the place asked); a world with no
        people, or a place no one can stand at, leaves the voice where it was asked."""
        people = self.people
        walker = getattr(people, "walkers", {}).get(who) if people is not None else None
        if walker is None or not hasattr(people, "command"):
            return pos
        try:
            answer = people.command(
                world,
                {"do": "stand", "who": who, "at": [round(pos[0] * 1000), round(pos[1] * 1000)]},
            )
        except (ValueError, RuntimeError):
            return pos
        if not answer.get("ok"):
            return pos
        with self._lock:
            self.standing[who] = now + STAND_AFTER_S
        with world.lock:
            if getattr(walker, "state", "") == "standing":
                return (float(walker.pos[0]), float(walker.pos[1]), pos[2])
        return pos

    def _stim_bang(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        return self._sound(world, request, "bang", 0.08, 0.05)

    def _entry(self) -> tuple[float, float]:
        spots = getattr(getattr(self.people, "layout", None), "agents", None) or {}
        for spot in spots.get("spots", []) if isinstance(spots, dict) else []:
            if spot.get("id") == "entry":
                return int(spot["x_mm"]) / 1000.0, int(spot["y_mm"]) / 1000.0
        return 2.5, -2.5

    def _stim_door(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        x, y = self._entry()
        now = self._now(world)
        event = self.add_sound("door", (x, y, 1.0), t0=now, seconds=0.5, src="door", by="page")
        return {
            "where_mm": wk.mm(event.pos),
            "sound": event.view(),
            "note_zh": msg("neuro.world.door.27ca"),
        }

    def _stim_knock(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        x, y = self._entry()
        now = self._now(world)
        for k in range(3):
            self.add_sound(
                "knock", (x, y, 1.2), t0=now + 0.28 * k, seconds=0.05, src="door", by="page"
            )
        return {"where_mm": wk.mm((x, y, 1.2)), "note_zh": msg("neuro.world.knock.906f")}

    def _toggle(self, world: Any, request: dict[str, Any], kind: str) -> dict[str, Any]:
        current = self.tv if kind == "tv" else self.music
        now = self._now(world)
        if current is not None:
            current.t1 = now
            if kind == "tv":
                self.tv = None
            else:
                self.music = None
            return {
                "on": False,
                "note_zh": msg("neuro.world.toggle.7b30")
                if kind == "tv"
                else msg("neuro.world.toggle.2725"),
            }
        where = request.get("where_mm")
        if isinstance(where, (list, tuple)) and len(where) >= 2:
            pos = (int(where[0]) / 1000.0, int(where[1]) / 1000.0, 0.8)
        else:
            bounds = self._bounds() or (-3.0, -3.0, 3.0, 3.0)
            pos = (bounds[0] + 0.4, (bounds[1] + bounds[3]) / 2, 0.8)
        event = self.add_sound(kind, pos, t0=now, seconds=3600.0, src=kind)
        if kind == "tv":
            self.tv = event
        else:
            self.music = event
        return {"on": True, "where_mm": wk.mm(pos), "sound": event.view()}

    def _stim_tv(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        return self._toggle(world, request, "tv")

    def _stim_music(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        return self._toggle(world, request, "music")

    def _stim_lights_off(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        return self._lamps(world, False)

    def _stim_lights_on(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        return self._lamps(world, True)

    def _lamps(self, world: Any, on: bool) -> dict[str, Any]:
        switch = self.switch_lamps
        if switch is None:
            raise ValueError(msg("neuro.world_refusal.58d07e"))
        switch(on)
        self._lamps_on = on
        state = self._light_state()
        with world.lock:
            head = self.head(world.data)
        lux = self._lux(state, (head[0], head[1], 0.0))
        note = msg("neuro.world.lamps.7240") if on else msg("neuro.world.lamps.ba63")
        if not on and lux >= 30.0:
            note = join(note, msg("neuro.world.lamps.e937", lux=lux))
        elif lux < 2.5:
            note = join(note, msg("neuro.world.lamps.2073", lux=lux))
        return {"lights_on": on, "lux_at_duck": round(lux, 2), "note_zh": note}

    # the hand

    def _stim_wave(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        with world.lock:
            data = world.data
            cam, yaw = self._point(data, "camera")
            reach = float(request.get("distance_m", 0.55))
            fwd = _to_world(yaw, (reach, 0.0, 0.0))
            centre = (cam[0] + fwd[0], cam[1] + fwd[1], max(0.18, cam[2] + 0.08))
            across = _to_world(yaw, (0.0, 1.0, 0.0))
            start = (centre[0] + fwd[0] * 0.6, centre[1] + fwd[1] * 0.6, centre[2] + 0.5)
            legs = [
                GLeg("come", "move", 0.6, to=centre, follow="camera", yaw_off=math.pi),
                GLeg(
                    "wave",
                    "wave",
                    float(request.get("seconds", 2.2)),
                    to=centre,
                    follow="camera",
                    amp=0.13,
                    hz=1.7,
                    axis=across,
                    yaw_off=math.pi,
                ),
                GLeg("leave", "move", 0.5, to=start, follow="camera", yaw_off=math.pi),
            ]
            self._start_gesture(data, "wave", legs, start)
        return {
            "where_mm": wk.mm(centre),
            "note_zh": msg("neuro.world.wave.0670"),
        }

    def _stim_hand_loom(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        with world.lock:
            data = world.data
            cam, yaw = self._point(data, "camera")
            beak, _ = self._point(data, "beak")
            far = _to_world(yaw, (0.75, 0.0, 0.0))
            start = (cam[0] + far[0], cam[1] + far[1], cam[2] + 0.06)
            near = _to_world(yaw, (0.09, 0.0, 0.0))
            stop = (beak[0] + near[0], beak[1] + near[1], beak[2] + 0.02)
            back = (start[0], start[1], start[2] + 0.5)
            seconds = float(request.get("seconds", 0.45))
            legs = [
                GLeg("appear", "hold", 0.25, follow="camera", yaw_off=math.pi),
                GLeg("loom", "move", seconds, to=stop, follow="camera", yaw_off=math.pi),
                GLeg("hold", "hold", 0.35, follow="camera", yaw_off=math.pi),
                GLeg("leave", "move", 0.6, to=back, follow="camera", yaw_off=math.pi),
            ]
            self._start_gesture(data, "hand_loom", legs, start)
            self._set_hand(data, start, wk.yaw_quat(yaw + math.pi))
        return {
            "where_mm": wk.mm(start),
            "note_zh": msg("neuro.world.hand_loom.02d8"),
        }

    def _stim_pet_head(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        strokes = max(1, min(4, int(request.get("strokes", 2))))
        head = tuple(self.duck_body_names.values())  # the press stops at any part of the duck
        with world.lock:
            data = world.data
            start = self._hand_from(data, request.get("who"))
            legs = [GLeg("reach", "track", 0.8, follow="head", offset=(0.02, 0.0, 0.05))]
            for k in range(strokes):
                legs.append(
                    GLeg(
                        "down",
                        "press",
                        1.0,  # a pat comes down slowly (≤ 0.1 m/s): gentle, not a poke
                        follow="head",
                        offset=(0.02, 0.0, 0.05),
                        bodies=head,
                        press_mm=1.5,
                    )
                )
                legs.append(
                    GLeg(
                        "stroke",
                        "press",
                        0.6,
                        follow="head",
                        offset=(0.02, 0.0, 0.05),
                        slide=-0.05,
                        bodies=head,
                        press_mm=2.0,
                    )
                )
                if k < strokes - 1:
                    legs.append(GLeg("lift", "track", 0.3, follow="head", offset=(0.02, 0.0, 0.05)))
            legs.append(GLeg("lift", "track", 0.25, follow="head", offset=(-0.02, 0.0, 0.08)))
            legs.append(GLeg("leave", "move", 0.6, to=start))
            self._start_gesture(data, "pet_head", legs, start)
        return {"note_zh": msg("neuro.world.pet_head.e252", strokes=strokes)}

    def _stim_stroke_back(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        body = tuple(self.duck_body_names.values())  # never through the neck or the head
        with world.lock:
            data = world.data
            start = self._hand_from(data, request.get("who"))
            legs = [
                GLeg("reach", "track", 1.0, follow="trunk", offset=(-0.06, 0.0, 0.12)),
                GLeg(
                    "down",
                    "press",
                    1.3,
                    follow="trunk",
                    offset=(-0.06, 0.0, 0.12),
                    bodies=body,
                    press_mm=2.0,
                ),
                GLeg(
                    "stroke",
                    "press",
                    0.9,
                    follow="trunk",
                    offset=(-0.06, 0.0, 0.12),
                    slide=-0.04,
                    bodies=body,
                    press_mm=3.0,
                ),
                GLeg("lift", "track", 0.3, follow="trunk", offset=(-0.08, 0.0, 0.14)),
                GLeg("leave", "move", 0.6, to=start),
            ]
            self._start_gesture(data, "stroke_back", legs, start)
        return {"note_zh": msg("neuro.world.stroke_back.4834")}

    def _side_legs(
        self, data: Any, request: dict[str, Any]
    ) -> tuple[list[GLeg], tuple[float, float, float], str]:
        """Reach the trunk's side: the hand upright beside it, its fingers toward the duck
        (the hand's fingers hang on its own right side: turned half round for the duck's right)."""
        side = str(request.get("side") or self.rng.choice(("left", "right")))
        sign = 1.0 if side == "left" else -1.0
        turn = 0.0 if side == "left" else math.pi
        trunk, yaw = self._point(data, "trunk")
        out = _to_world(yaw, (0.0, sign * 0.35, 0.12))
        start = (trunk[0] + out[0], trunk[1] + out[1], trunk[2] + out[2])
        legs = [
            GLeg(
                "reach",
                "track",
                0.35,
                follow="trunk",
                offset=(0.0, sign * 0.16, -0.03),
                yaw_off=turn,
            ),
            GLeg(
                "touch",
                "press",
                0.16,
                follow="trunk",
                offset=(0.0, sign * 0.16, -0.03),
                bodies=tuple(self.duck_body_names.values()),
                press_mm=4.0,
                yaw_off=turn,
                press_dir=(0.0, sign, 0.0),
                press_speed=0.8,  # a poke comes in fast: a sharp touch
            ),
        ]
        return legs, start, side

    def _stim_poke(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        with world.lock:
            data = world.data
            legs, start, side = self._side_legs(data, request)
            legs.append(GLeg("leave", "move", 0.3, to=start))
            self._start_gesture(data, "poke", legs, start)
        return {
            "side": side,
            "note_zh": msg(
                "neuro.world.poke.813b",
                side=msg("neuro.world.push.d2af")
                if side == "left"
                else msg("neuro.world.push.4d9c"),
            ),
        }

    def _stim_push(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        strength = max(0.0, min(1.0, float(request.get("strength", 0.4))))
        force = PUSH_N[0] + (PUSH_N[1] - PUSH_N[0]) * strength
        with world.lock:
            data = world.data
            legs, start, side = self._side_legs(data, request)
            _, yaw = self._point(data, "trunk")
            sign = 1.0 if side == "left" else -1.0
            push = _to_world(yaw, (0.0, -sign * force, 0.0))
            touch = legs[-1]
            legs.append(
                GLeg(
                    "push",
                    "press",
                    PUSH_S,
                    follow="trunk",
                    offset=touch.offset,
                    bodies=touch.bodies,
                    press_mm=6.0,
                    yaw_off=touch.yaw_off,
                    press_dir=touch.press_dir,
                    press_speed=0.8,
                    force=push,
                )
            )
            legs.append(GLeg("leave", "move", 0.4, to=start))
            self._start_gesture(data, "push", legs, start)
        where = msg("neuro.world.push.d2af") if side == "left" else msg("neuro.world.push.4d9c")
        return {
            "side": side,
            "force_n": round(force, 1),
            "note_zh": msg("neuro.world.push.c116", side=where),
        }

    def _stim_pick_up(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        hand = self.hand
        if hand is None:
            raise ValueError(msg("neuro.world_refusal.9eea94"))
        with world.lock:
            if self.gesture is not None and not self.gesture.done:
                self._end_gesture(world.data, "換成抱起來")
            answer = hand.command(world, {"do": "pickup"})
        return {"hand": answer, "note_zh": msg("neuro.world.pick_up.3de7")}

    def _stim_set_down(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        hand = self.hand
        if hand is None:
            raise ValueError(msg("neuro.world_refusal.9eea94"))
        with world.lock:
            if not getattr(hand, "holding", False):
                return {"ok": False, "note_zh": msg("neuro.world.set_down.f9cf")}
            answer = hand.command(world, {"do": "place", "to": "here"})
        return {"hand": answer, "note_zh": msg("neuro.world.set_down.3440")}

    # things

    def _stim_roll_ball(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        props = self.props
        if props is None:
            raise ValueError(msg("neuro.world_refusal.15c9c0"))
        params: dict[str, Any] = {
            "toward": "duck",
            "speed_mm_s": int(request.get("speed_mm_s", 900)),
        }
        if request.get("id"):
            params["id"] = str(request["id"])
        else:  # the bouncy ball reaches it; a felt ball stops short (real rolling friction)
            things = getattr(props, "things", {}) or {}
            fast = [k for k, t in things.items() if getattr(t.thing, "cls", "") == "ball_fast"]
            if fast:
                params["id"] = fast[0]
        with world.lock:
            answer = props.roll(world, params)
        return {
            "world": answer,
            "where_mm": answer.get("from_mm"),
            "note_zh": msg("neuro.world.roll_ball.2983"),
        }

    def _free_spot(self, world: Any, ahead: float, spread: float) -> list[tuple[float, float]]:
        with world.lock:
            data = world.data
            trunk, yaw = self._point(data, "trunk")
        out = []
        for k in range(12):
            a = yaw + self.rng.uniform(-spread, spread)
            r = ahead + self.rng.uniform(-0.1, 0.15) + 0.05 * k
            out.append((trunk[0] + r * math.cos(a), trunk[1] + r * math.sin(a)))
        return out

    def _pick_thing(self, request: dict[str, Any], *, prefer_toys: bool) -> Any:
        props = self.props
        things = getattr(props, "things", {}) or {}
        if request.get("id"):
            key = str(request["id"])
            if key not in things:
                raise ValueError(msg("neuro.world_refusal.68712d", given_id=key))
            return things[key]
        order = sorted(things)
        toys = [
            k
            for k in order
            if getattr(things[k].thing, "cls", "")
            in ("ball_slow", "ball_fast", "handle_toy", "light_toy")
        ]
        pool = toys if prefer_toys and toys else [k for k in order if k != "books"] or order
        if not pool:
            raise ValueError(msg("neuro.world_refusal.15c9c0"))
        return things[self.rng.choice(pool)]

    def _stim_novel_object(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        props = self.props
        if props is None:
            raise ValueError(msg("neuro.world_refusal.15c9c0"))
        tracked = self._pick_thing(request, prefer_toys=False)
        for x, y in self._free_spot(world, float(request.get("distance_m", 0.7)), 0.35):
            with world.lock:
                answer = props._put(
                    world, tracked, x, y, self.rng.uniform(-math.pi, math.pi), top=0.05
                )
            if answer.get("ok"):
                zh = getattr(getattr(tracked, "thing", None), "zh", tracked.id)
                return {
                    "world": answer,
                    "thing": tracked.id,
                    "where_mm": answer.get("pos_mm"),
                    "note_zh": msg("neuro.world.novel_object.8826", thing=zh),
                }
        raise ValueError(msg("neuro.world_refusal.0eb8ab"))

    def _stim_toy_drop(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        props = self.props
        if props is None:
            raise ValueError(msg("neuro.world_refusal.15c9c0"))
        tracked = self._pick_thing(request, prefer_toys=True)
        for x, y in self._free_spot(world, float(request.get("distance_m", 0.6)), 1.2):
            with world.lock:
                answer = props._put(world, tracked, x, y, 0.0, top=0.05)
                if answer.get("ok"):
                    data = world.data
                    data.qpos[tracked.qadr + 2] += float(request.get("height_m", 0.35))
                    data.qvel[tracked.dof : tracked.dof + 6] = 0.0
                    zh = getattr(getattr(tracked, "thing", None), "zh", tracked.id)
                    return {
                        "thing": tracked.id,
                        "where_mm": answer.get("pos_mm"),
                        "note_zh": msg("neuro.world.toy_drop.badf", thing=zh),
                    }
        raise ValueError(msg("neuro.world_refusal.7d8e37"))

    # people

    def _stim_walk_by(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        """Someone walks past in front of the duck: from the door to a point ahead of it on
        the door's side, across its front to the other side, and back out."""
        people = self.people
        if people is None:
            raise ValueError(msg("neuro.world_refusal.314e9b"))
        who = str(request.get("who") or self._free_person())
        with world.lock:
            trunk, yaw = self._point(world.data, "trunk")
        ex, ey = self._entry()
        door = _heading_frame(yaw, (ex - trunk[0], ey - trunk[1], 0.0))
        side = 1.0 if door[1] >= 0 else -1.0
        bounds = self._bounds() or (-3.0, -3.0, 3.0, 3.0)

        floor = getattr(people, "floor", None)

        def inside(x: float, y: float) -> bool:
            if not (bounds[0] + 0.45 < x < bounds[2] - 0.45):
                return False
            if not (bounds[1] + 0.45 < y < bounds[3] - 0.45):
                return False
            if floor is not None and hasattr(floor, "free"):
                return bool(floor.free(x, y, (trunk[0], trunk[1])))
            return True

        route: list[tuple[float, float]] | None = None
        for ahead in (1.3, 1.1, 1.6, 0.9, 1.9, 0.7):
            for lateral in (1.1, 0.9, 1.3, 0.7, 1.6):
                a = _to_world(yaw, (ahead, side * lateral, 0.0))
                b = _to_world(yaw, (ahead, -side * lateral, 0.0))
                pa = (trunk[0] + a[0], trunk[1] + a[1])
                pb = (trunk[0] + b[0], trunk[1] + b[1])
                if inside(*pa) and inside(*pb):
                    route = [pa, pb]
                    break
            if route is not None:
                break
        if route is None:  # no room in front: past it wherever it fits
            for turn in (0.45, -0.45, 0.8, -0.8, 1.2, -1.2, math.pi):
                heading = yaw + turn
                gx = trunk[0] + 1.3 * math.cos(heading)
                gy = trunk[1] + 1.3 * math.sin(heading)
                if inside(gx, gy):
                    route = [(gx, gy)]
                    break
        if route is None:
            raise ValueError(msg("neuro.world_refusal.1022d1"))
        first = [int(route[0][0] * 1000), int(route[0][1] * 1000)]
        start = self._come_from(world)
        answer = people.command(world, {"do": "come", "who": who, "to": first, "from": start})
        now = self._now(world)
        with self._lock:
            if len(route) > 1:
                to = [int(route[1][0] * 1000), int(route[1][1] * 1000)]
                params = {"who": who, "to": to, "give_up": now + 30.0}
                self.pending.append((now + 0.5, "come_when_there", params))
            else:
                params = {"who": who, "give_up": now + 30.0}
                self.pending.append((now + 0.5, "leave_when_there", params))
        return {
            "who": who,
            "people": answer,
            "where_mm": wk.mm((route[-1][0], route[-1][1], 0.0)),
            "note_zh": msg("neuro.world.walk_by.6176", person=answer.get("zh", who)),
        }

    def _stim_sit_near(self, world: Any, request: dict[str, Any]) -> dict[str, Any]:
        """A family member comes to the duck and stays beside it. ``to`` ([x_mm, y_mm] or
        ``spot:<id>``) makes them walk there after ``stay_s`` (default :data:`SIT_NEAR_S`) and stand
        there instead of leaving — the fair scene for a follow (the person stays, then walks a path);
        without it they leave as before."""
        people = self.people
        if people is None:
            raise ValueError(msg("neuro.world_refusal.314e9b"))
        who = str(request.get("who") or self._free_person())
        start = self._come_from(world)
        answer = people.command(world, {"do": "come", "who": who, "to": "duck", "from": start})
        now = self._now(world)
        to = request.get("to")
        if isinstance(to, str) and to.startswith("spot:"):
            x, y, _ = people._spot(to[5:])
            to = [int(x * 1000), int(y * 1000)]
        stay_s = float(request.get("stay_s") or SIT_NEAR_S)
        with self._lock:
            self.pending.append((now + 0.5, "talk_when_there", {"who": who, "give_up": now + 30.0}))
            if to is None:
                self.pending.append((now + stay_s, "leave", {"who": who}))
            else:
                self.pending.append((now + stay_s, "stand_when_there", {"who": who, "at": to}))
        return {
            "who": who,
            "people": answer,
            "to": to,
            "note_zh": msg("neuro.world.sit_near.a732", person=answer.get("zh", who)),
        }

    def _come_from(self, world: Any) -> str:
        """Where a family member steps in: the entry, unless the duck sits right there."""
        spots = getattr(getattr(self.people, "layout", None), "agents", None) or {}
        places = {
            str(spot.get("id")): (int(spot["x_mm"]) / 1000.0, int(spot["y_mm"]) / 1000.0)
            for spot in (spots.get("spots", []) if isinstance(spots, dict) else [])
        }
        with world.lock:
            trunk, _ = self._point(world.data, "trunk")
        entry = places.get("entry", self._entry())
        if math.dist(entry, (trunk[0], trunk[1])) >= ENTRY_CLEAR_M:
            return "entry"
        for name, xy in places.items():
            if name != "entry" and math.dist(xy, (trunk[0], trunk[1])) >= ENTRY_CLEAR_M:
                return name
        return "entry"

    def _free_person(self) -> str:
        people = self.people
        walkers = getattr(people, "walkers", {}) or {}
        for who in FAMILY_ORDER:
            walker = walkers.get(who)
            if walker is not None and not getattr(walker, "here", False):
                return who
        return FAMILY_ORDER[0]

    # ── follow-ups (people leave, talk) ──────────────────────────────────────

    def pump(self) -> None:
        """Run the follow-ups that are due (call WITHOUT the world's lock): a person who walked
        past goes back out, one who came to sit near talks a little and later leaves."""
        world, people = self.world, self.people
        if world is None or people is None:
            return
        now = self._now(world)
        with self._lock:
            due = [p for p in self.pending if p[0] <= now]
            self.pending = [p for p in self.pending if p[0] > now]
            gone = [who for who, until in self.standing.items() if until <= now]
            for who in gone:
                del self.standing[who]
        for who in gone:  # (a speaker who has said nothing for a while walks out)
            walker = getattr(people, "walkers", {}).get(who)
            if walker is not None and getattr(walker, "here", False):
                try:
                    people.command(world, {"do": "leave", "who": who})
                except (ValueError, RuntimeError) as error:
                    self._error(f"pump stand leave: {type(error).__name__}: {error}")
        for _, what, params in due:
            who = str(params.get("who"))
            walker = getattr(people, "walkers", {}).get(who)
            if walker is None:
                continue
            arrived = getattr(walker, "state", "") == "standing"
            waiting = ("leave_when_there", "talk_when_there", "come_when_there", "stand_when_there")
            if what in waiting and not arrived:
                if now < float(params.get("give_up", now)):
                    with self._lock:
                        self.pending.append((now + 0.3, what, params))
                    continue
            try:
                if what in ("leave_when_there", "leave"):
                    if getattr(walker, "here", False):
                        people.command(world, {"do": "leave", "who": who})
                elif what == "come_when_there":
                    people.command(world, {"do": "come", "who": who, "to": params["to"]})
                    later = {"who": who, "give_up": now + 30.0}
                    with self._lock:
                        self.pending.append((now + 0.5, "leave_when_there", later))
                elif what == "stand_when_there":
                    # (the fair follow scene) the person walks to the place and stands there
                    people.command(world, {"do": "stand", "who": who, "at": params["at"]})
                elif what == "talk_when_there":
                    scale = float(walker.persona.height_m) / 1.72
                    pos = (float(walker.pos[0]), float(walker.pos[1]), MOUTH_H_M * scale)
                    for k in range(3):
                        self.add_sound(
                            "voice",
                            pos,
                            t0=now + 0.5 + 4.0 * k,
                            seconds=1.6,
                            db1m=56.0,
                            src=f"person:{who}",
                        )
            except Exception as error:
                self._error(f"pump {what}: {type(error).__name__}: {error}")

    def start_thread(self) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.wait(0.1):
                try:
                    self.pump()
                except Exception as error:
                    self._error(f"pump: {type(error).__name__}: {error}")

        self._thread = threading.Thread(target=loop, name="neuro-world", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ── ops ─────────────────────────────────────────────────────────────────

    def ops(self) -> dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]]:
        return {"x_neuro_stim": self.stim_op, "x_neuro": self.status_op}

    def stim_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        answer = self.stimulate(request)
        self.pump()
        return answer

    def status_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        world = self.world
        if world is None:
            return {"ready": False}
        with world.lock:
            section = self.snapshot(world)
        return {"ready": True, "stimuli": dict(STIMULI), "section": section}


def register(registry: Any) -> NeuroWorld:
    """simhome's entry point: the part, its hooks, ops and the ``neuro`` section."""
    import os

    part = NeuroWorld(seed=int(os.environ.get("DUCKMIND_NEURO_SEED", "7") or 7))
    part.registry = registry
    wk.register_part(registry, part, section=SECTION)
    registry.on_ready(lambda world, body: part.start_thread(), name="neuro.pump")
    return part


def attach(
    host: Any,
    *,
    seed: int = 7,
    lights: Callable[[], Any] | None = None,
    switch_lamps: Callable[[bool], Any] | None = None,
) -> NeuroWorld:
    """The part on an in-process world (:class:`~duckmind.world.worldkit.Headless` or
    :class:`~duckmind.world.headless_duck.HeadlessDuck`): bound to its household, its hooks
    appended to the host's step lists, ready at once. Call :meth:`NeuroWorld.pump` yourself."""
    kit = host.kit
    part = NeuroWorld(seed=seed).bind(
        hand=kit.hand,
        people=kit.people,
        props=kit.props,
        nest=kit.nest,
        lights=lights,
        switch_lamps=switch_lamps,
    )
    part.ready(host.world, None)
    if hasattr(host, "before"):  # worldkit.Headless
        before, after = host.before, host.after
    else:  # headless_duck.HeadlessDuck
        before, after = host._before, host._after
    before.append(part.before_step)
    after.append(part.after_step)
    return part
