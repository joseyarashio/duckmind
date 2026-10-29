"""踢球、推球 — ball play (a:play, 2026-09-28; owner 「A B 全數」 item 5).

A ball, to this duck, is *a thing that rolled*: it saw it roll along the floor (someone rolled it
over, or it rolled away after the duck's own push), or it nudged a small thing and saw it go
(:class:`~duckmind.neuro.skill_common.PlayThing` ``rolled``). Nothing tells it which thing is a
ball. The play act (``play_ball``) wins the brain's competition from its play drive, a rolling
thing in view, a ball it knows nearby and how much fun ball play has been (``ball_fun``), against
fear and sleepiness — bold ducks play ball more and kick more, timid ones nudge gently with the
beak and are the first to be startled by a ball rolling at them (the looming reflex is the
brain's own), hatchlings mostly chase and bump.

What the body does (curiosity P1's verbs, with the measured numbers in
:mod:`duckmind.curiosity.calib` and this lane's rig, both in the integration report):

* **chase** — far away, the brain's own walk goes after it (its bursts, pauses and speed drift;
  up to 0.55 m/s), steering at where the ball will be (a rolling ball is led by ~0.6 s);
* **line up** — near (≤ 0.34 m) the fine approach (:class:`~duckmind.neuro.skill_common.
  Approach`): stop, look down at it, one calibrated creep or reverse arc, look again;
* **kick** (``kick_left`` / ``kick_right``, robotd's own 0.5-s kick networks) — the ball 65–90 mm
  ahead and 28–70 mm toward the foot (phase-3 live: rolls 0.4–0.9 m, 23/24 ≥ 30 cm; this rig
  12/12, 0.56–0.99 m);
* **beak push** (``ground_pick`` with the beak shut: the whole body dips the beak to 2 cm) — the
  ball 75–125 mm ahead: touched 8/8, pushed 24–45 mm (phase-3 live 8/8, 32–47 mm);
* **chest push** (a 1.0-s creep into it) — the ball 85–135 mm straight ahead: rolls 0.17–0.29 m;
* **watch** it go (the head follows where it went), then chase again — until the play drive is
  used up or the ball is lost (it rolled under the sofa, out of sight).

Every outcome is from its own eyes: the ball seen again ≥ 10 cm from where it was → it rolled
(fun: a chirp or a wheee, the mood up); not moved → a miss (the next line-up is more careful).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from duckmind.curiosity.calib import CALIB, Window
from duckmind.i18n import msg
from duckmind.neuro import signals as N
from duckmind.neuro.contract import Gaze, HeadPose, SensorFrame, Target
from duckmind.neuro.skill_common import (
    FF_VYAW,
    NEAR_M,
    ROLL_MIN_M,
    Approach,
    PlayOut,
    PlayThing,
    ThingBook,
    bearing_range,
    head_is_down,
    head_look,
    legs_still,
    rel_of,
    scan_gaze,
)

BALL_VERSION = "neuro-ball-1"
KICK_WINDOW = CALIB.kick_window  # (65, 90) ahead, (28, 70) toward the foot
BEAK_WINDOW = CALIB.peck  # (75, 125) ahead, ±20
CHEST_WINDOW = Window((85, 135), (-25, 25))  # this rig: 85 mm → 0.29 m in 1.0 s, 105 → 0.18 m
CHEST_MS = 1000
KICK_HOLD_MS = 1300  # the kick (0.5 s) and the legs settling after it
PICK_HOLD_MS = 3300  # the ground pick (2.8 s) and back to standing
WATCH_MS = (1600, 2600)  # (the camera's passes come ~2 Hz and the head needs ~0.4 s to get down:
# 0.9–1.6 s gave one pass at most — the final evaluation, 9 pushes and kicks, 0 rolls seen)
# Where to look after the act: this far beyond where the ball lay, along the heading (a kick rolls
# it 0.56–0.99 m, a chest push 0.17–0.29 m, a beak nudge 24–45 mm: measured, module doc) — the
# head's low looking pose when that is near: the eye sees nothing nearer than 0.17 m, and with
# the head level nothing nearer than ~0.4 m (this rig: after a chest push the ball lay 0.25 m
# ahead and no pass showed it with the old level look)
WATCH_AHEAD_M = {"kick": 0.5, "chest": 0.2, "beak": 0.1}
# A push whose outcome the watch could not see (the ball stopped nearer than the eye sees — a
# chest push left it 0.14 m ahead — or another act took over): the next look at that ball, within
# this long, tells whether it went (this rig, final evaluation: 9 pushes and kicks, 0 rolls seen,
# though 3 moved the ball 0.15–0.7 m)
PENDING_MS = 30_000
LOST_MS = 12000  # not seen this long (and not found where it was): it is gone
LEAD_S = 0.6  # a rolling ball is chased where it will be
SMALL_M = 0.095  # an untried thing this small may be nudged to see what it does
VERB_ZH = {
    "kick": msg("neuro.incoming.incoming_skill_ball.VERB_ZH.fd6e9e"),
    "beak": msg("neuro.incoming.incoming_skill_ball.VERB_ZH.3a1b0b"),
    "chest": msg("neuro.incoming.incoming_skill_ball.VERB_ZH.55c0d3"),
}
PHASE_ZH = {
    "idle": "",
    "chase": msg("neuro.incoming.incoming_skill_ball.PHASE_ZH.228623"),
    "line_up": msg("neuro.incoming.incoming_skill_ball.PHASE_ZH.fd6b6c"),
    "kick": msg("neuro.incoming.incoming_skill_ball.PHASE_ZH.6bb2ae"),
    "beak": msg("neuro.incoming.incoming_skill_ball.PHASE_ZH.0dd065"),
    "chest": msg("neuro.incoming.incoming_skill_ball.PHASE_ZH.a9b879"),
    "watch": msg("neuro.incoming.incoming_skill_ball.PHASE_ZH.118cfd"),
    "search": msg("neuro.incoming.incoming_skill_ball.PHASE_ZH.735b6a"),
}


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class BallStyle:
    """How this duck plays ball (from the genome's axes): how often it kicks and how carefully
    it lines up."""

    kick: float = 0.4
    beak: float = 0.35
    tol_pct: int = 100

    @classmethod
    def of(cls, axes: dict[str, float], young: float) -> BallStyle:
        bold = axes.get("boldness", 0.5)
        act = axes.get("activity", 0.5)
        wit = axes.get("wit", 0.5)
        kick = _clip(0.15 + 0.55 * bold + 0.2 * act - 0.35 * young, 0.05, 0.85)
        beak = _clip(0.25 + 0.35 * (1.0 - bold), 0.1, 0.7)
        # patience (curiosity §7.7): a quick-witted duck lines up tighter
        tol = int(round(130 - 50 * wit))
        return cls(kick=kick, beak=beak, tol_pct=tol)


@dataclass
class BallPlay:
    """The ball-play act's own state machine (one per duck; memory lives in the ThingBook)."""

    style: BallStyle = field(default_factory=BallStyle)
    seed: int = 0
    phase: str = "idle"
    phase_ms: int = 0
    target: int | None = None
    verb: str = ""
    foot: str = "left"
    approach: Approach | None = None
    skill_seq: int = 0
    before: tuple[float, float] | None = None  # where the ball was when it acted (odometry)
    acted_ms: int = 0
    watch_ms: int = 1200
    kicks: int = 0
    pushes: int = 0
    rolls_seen: int = 0
    misses: int = 0
    lost: int = 0
    rng: int = 0
    last_event: str | None = None
    # (thing key, where it lay when it acted, when the act ended): a push not yet seen to end
    pending: tuple[int, float, float, int] | None = None

    def __post_init__(self) -> None:
        self.rng = N.seed_from("neuro.play.ball", self.seed)

    # ── the choice ──────────────────────────────────────────────────────────

    def pick(
        self, book: ThingBook, odom: tuple[float, float, float], t_ms: int, fresh_s: float = 30.0
    ) -> PlayThing | None:
        """The ball to play with: the nearest thing it knows rolls (or rolling now) seen in the
        last half minute (``fresh_s``); else, for a bold duck in play, a small thing it has never
        nudged."""
        best, best_score = None, 0.0
        for th in book.things.values():
            age = (t_ms - th.seen_ms) / 1000.0
            if age > fresh_s or th.home or th.stuck >= 2:
                continue
            _, rng = bearing_range(odom, th.x, th.y)
            if th.rolls or th.moving(t_ms):
                score = 1.0 / (0.5 + rng) + 0.3 * th.fun
            elif th.nudges == 0 and th.widest <= SMALL_M and th.held == 0 and th.stuck == 0:
                score = (0.5 if th.axis is None else 0.25) / (0.5 + rng)
            else:
                continue
            score *= 1.0 if age < 5.0 else 0.6
            if score > best_score:
                best, best_score = th, score
        return best

    def start(self, t_ms: int) -> None:
        self.phase, self.phase_ms = "idle", t_ms
        self.target = None
        self.approach = None

    # ── the step ────────────────────────────────────────────────────────────

    def step(
        self,
        t_ms: int,
        frame: SensorFrame,
        book: ThingBook,
        odom: tuple[float, float, float],
        seen_now: list[PlayThing],
        *,
        halted: bool,
        standing: bool,
    ) -> PlayOut:
        self.last_event = None
        self.settle_pending(t_ms, book)
        thing = book.get(self.target)
        if thing is not None and self.phase in ("chase", "line_up") and t_ms - thing.seen_ms > 2000:
            other = self.pick(book, odom, t_ms, fresh_s=1.0)
            near_it = (
                other is not None
                and math.hypot(
                    other.x - thing.x - thing.vx * 1.5, other.y - thing.y - thing.vy * 1.5
                )
                < 0.6
            )
            if other is not None and other.key != thing.key and near_it:
                thing, self.target = other, other.key  # (likely the same ball, seen again)
                self._go("chase", t_ms)
                self.approach = None
        if self.phase == "search":
            # looking for a ball: the brain's own walk and gaze look about; a ball it knows rolls,
            # seen again (in the last 2 s), starts the chase again
            fresh = self.pick(book, odom, t_ms, fresh_s=2.0)
            if fresh is not None and not fresh.rolls:
                fresh = None
            if fresh is None:
                return PlayOut(phase="search", zh=PHASE_ZH["search"])
            thing, self.target = fresh, fresh.key
            self._go("chase", t_ms)
        if thing is None or self.phase == "idle":
            thing = self.pick(book, odom, t_ms)
            if thing is None:
                self._go("search", t_ms)
                return PlayOut(phase="search", zh=PHASE_ZH["search"])
            self.target = thing.key
            self._go("chase", t_ms)
        if self.phase in ("kick", "beak") and not halted:
            # (robotd's one-shot runs: its label is the skill's, not stand — go on)
            return self._acting(t_ms, frame, thing)
        if halted or not standing:
            # a reflex holds the body (or it is getting up): wait; a move under way is lost
            if self.approach is not None:
                self.approach.move = None
            return PlayOut(phase=self.phase, zh=PHASE_ZH.get(self.phase, ""), posture="stand")
        rel = rel_of(odom, *self._aim(thing, t_ms))
        seen_age = t_ms - thing.seen_ms
        if self.phase == "chest":
            return self._acting(t_ms, frame, thing)
        if self.phase == "watch":
            return self._watch(t_ms, thing, odom, seen_now)
        if seen_age > LOST_MS and self.phase != "line_up":
            self.lost += 1
            self.last_event = "lost"
            self._go("search", t_ms)
            return PlayOut(phase="search", zh=PHASE_ZH["search"], event="lost")
        dist = math.hypot(*rel)
        if self.phase == "chase":
            if dist > NEAR_M or thing.moving(t_ms):
                bearing = math.atan2(rel[1], rel[0])
                gaze, _ = head_look(rel[0], rel[1])
                return PlayOut(
                    far=True,
                    phase="chase",
                    zh=PHASE_ZH["chase"],
                    gaze=Gaze(bearing, max(gaze.elevation, -0.35), 0.9, 0.7),
                    target=Target("thing", bearing, dist, thing.track),
                )
            self._choose_verb(thing, rel)
            self._go("line_up", t_ms)
        if self.phase == "line_up":
            return self._line_up(t_ms, frame, thing, rel)
        self._go("chase", t_ms)
        return PlayOut(phase="chase", zh=PHASE_ZH["chase"])

    # ── pieces ──────────────────────────────────────────────────────────────

    def settle_pending(self, t_ms: int, book: ThingBook) -> bool:
        """A push the watch did not see end: the first later look at that ball says whether it
        went (≥ 10 cm from where it lay: it rolled — the fun of it, as in the watch). Called
        every tick (the play mind's senses), whichever act runs. True when it saw it had rolled."""
        if self.pending is None:
            return False
        key, px, py, since_ms = self.pending
        thing = book.get(key)
        if thing is None or t_ms - since_ms > PENDING_MS:
            self.pending = None
            return False
        if thing.seen_ms <= since_ms:
            return False
        self.pending = None
        if math.hypot(thing.x - px, thing.y - py) < ROLL_MIN_M:
            return False
        thing.rolled += 1
        thing.fun = min(1.0, thing.fun + 0.25)
        self.rolls_seen += 1
        self.last_event = "rolled"
        return True

    def _go(self, phase: str, t_ms: int) -> None:
        self.phase, self.phase_ms = phase, t_ms

    def _aim(self, thing: PlayThing, t_ms: int) -> tuple[float, float]:
        """Where to go for it: where it is — or, rolling, where it will be."""
        if thing.moving(t_ms):
            return thing.x + thing.vx * LEAD_S, thing.y + thing.vy * LEAD_S
        return thing.x, thing.y

    def _draw(self) -> float:
        self.rng, u = N.between(self.rng, 0.0, 1.0)
        return u

    def _choose_verb(self, thing: PlayThing, rel: tuple[float, float]) -> None:
        """Which verb this time: a kick (bold), a beak push (gentle), else a chest push; a thing
        it has never nudged gets a gentle push first."""
        self.foot = "left" if rel[1] >= 0.0 else "right"
        u = self._draw()
        if not thing.rolls:
            self.verb = "chest" if u < 0.6 else "beak"
        elif u < self.style.kick:
            self.verb = "kick"
        elif u < self.style.kick + self.style.beak * (1.0 - self.style.kick):
            self.verb = "beak"
        else:
            self.verb = "chest"
        if self.verb == "kick":
            window, sign = KICK_WINDOW, 1 if self.foot == "left" else -1
        elif self.verb == "beak":
            window, sign = BEAK_WINDOW, 1
        else:
            window, sign = CHEST_WINDOW, 1
        self.approach = Approach(window, sign, tol_pct=self.style.tol_pct)

    def _line_up(
        self, t_ms: int, frame: SensorFrame, thing: PlayThing, rel: tuple[float, float]
    ) -> PlayOut:
        ap = self.approach
        assert ap is not None
        if ap.phase == "look" and ap.moves == 0 and t_ms == self.phase_ms:
            ap.reset(t_ms)
        state, vx, vyaw = ap.step(
            t_ms, rel, thing.seen_ms, legs_still(frame), thing.fixes, head_is_down(frame)
        )
        gaze, head = head_look(rel[0], rel[1])
        words = msg(
            "neuro.incoming.incoming_skill_ball.BallPlay._line_up.b70630",
            line_up=PHASE_ZH["line_up"],
            verb=VERB_ZH[self.verb],
        )
        if state == "moving":
            blind = ap.move_name == "blind"  # (too close to see: the head at HOME, as measured)
            return PlayOut(
                drive=True,
                vx=vx,
                vyaw=vyaw,
                fine=True,
                reverse=vx < 0.0,
                head=HeadPose() if blind else head,
                head_blend=1.0 if blind else 0.7,
                gaze=None if blind else gaze,
                phase="line_up",
                zh=words,
                quiet=True,
            )
        if state == "far":
            self._go("chase", t_ms)
            return PlayOut(far=True, phase="chase", zh=PHASE_ZH["chase"])
        if state in ("lost", "gave_up"):
            self.misses += 1
            self._go("chase", t_ms)
            self.approach = None
            return PlayOut(
                phase="chase",
                zh=msg("neuro.incoming.incoming_skill_ball.BallPlay._line_up.de3143"),
                event="gave_up",
            )
        if state == "ready":
            self.before = (thing.x, thing.y)
            self.acted_ms = t_ms
            self.skill_seq += 1
            thing.nudges += 1
            self._go(self.verb, t_ms)
            return self._acting(t_ms, frame, thing)
        since = t_ms - ap.phase_ms
        if since > 1200 and (thing.seen_ms < ap.seen_after):
            # not seen where it should be: look about for it — around where it was, then all
            # across the front
            base = math.atan2(rel[1], max(rel[0], 0.05)) if since < 4200 else 0.0
            width = 0.5 if since < 4200 else 0.9
            gaze = scan_gaze(base, since - 1200, elevation=-0.35, width=width)
            return PlayOut(
                drive=True,
                gaze=gaze,
                phase="line_up",
                zh=msg("neuro.incoming.incoming_skill_ball.BallPlay._line_up.search"),
                quiet=True,
            )
        return PlayOut(
            drive=True, head=head, head_blend=0.8, gaze=gaze, phase="line_up", zh=words, quiet=True
        )

    def _acting(self, t_ms: int, frame: SensorFrame, thing: PlayThing) -> PlayOut:
        since = t_ms - self.acted_ms
        verb = self.phase
        if verb == "chest":
            if since < CHEST_MS:
                return PlayOut(
                    drive=True,
                    vx=0.40,
                    vyaw=FF_VYAW,
                    fine=True,
                    head=HeadPose(np=0.5, hp=-0.5),
                    head_blend=0.6,
                    phase="chest",
                    zh=PHASE_ZH["chest"],
                    quiet=True,
                    event="pushed" if since == 0 else None,
                )
            done = since >= CHEST_MS + 450
        else:
            hold = KICK_HOLD_MS if verb == "kick" else PICK_HOLD_MS
            done = since >= hold and legs_still(frame)
            if not done:
                skill = f"kick_{self.foot}" if verb == "kick" else "ground_pick"
                return PlayOut(
                    drive=True,
                    skill=skill,
                    skill_seq=self.skill_seq,
                    mouth=0.0,
                    head=HeadPose(),
                    head_blend=1.0,
                    phase=verb,
                    zh=PHASE_ZH[verb],
                    quiet=True,
                    vocal=(
                        "wheee" if verb == "kick" and since == 0 and self._draw() < 0.35 else None
                    ),
                    event=("kicked" if verb == "kick" else "pushed") if since == 0 else None,
                )
        if done:
            if verb == "kick":
                self.kicks += 1
            else:
                self.pushes += 1
            self.watch_ms = int(WATCH_MS[0] + (WATCH_MS[1] - WATCH_MS[0]) * self._draw())
            self._go("watch", t_ms)
            if self.before is not None:
                self.pending = (thing.key, self.before[0], self.before[1], t_ms)
        return PlayOut(drive=True, phase=verb, zh=PHASE_ZH[verb], quiet=True)

    def _watch(
        self, t_ms: int, thing: PlayThing, odom: tuple[float, float, float], seen: list[PlayThing]
    ) -> PlayOut:
        since = t_ms - self.phase_ms
        before = self.before or (thing.x, thing.y)
        moved = math.hypot(thing.x - before[0], thing.y - before[1])
        new_look = thing.seen_ms > self.phase_ms
        event = None
        if new_look:
            self.pending = None  # (the watch saw how it ended)
        if new_look and moved >= ROLL_MIN_M:
            thing.rolled += 1
            thing.fun = min(1.0, thing.fun + 0.25)
            self.rolls_seen += 1
            self.last_event = event = "rolled"
            self._go("chase", t_ms)
            return PlayOut(
                phase="watch",
                zh=msg("neuro.incoming.incoming_skill_ball.BallPlay._watch.f48f62"),
                event=event,
                vocal="wheee" if self._draw() < 0.4 else "chirp",
            )
        if since >= self.watch_ms:
            if new_look and moved < 0.015:
                thing.stuck += 1
                self.misses += 1
                self.last_event = "missed"
            elif not new_look:
                self.last_event = "unseen"
            self._go("chase", t_ms)
            return PlayOut(
                phase="watch",
                zh=msg("neuro.incoming.incoming_skill_ball.BallPlay._watch.9f30fc"),
                event=self.last_event,
            )
        # look where it went: along the push, a little way beyond where it lay (low when near)
        a, left = rel_of(odom, before[0], before[1])
        gaze, head = head_look(max(a, 0.0) + WATCH_AHEAD_M.get(self.verb, 0.2), left)
        return PlayOut(
            drive=True,
            head=head,
            head_blend=0.8,
            gaze=gaze,
            phase="watch",
            zh=PHASE_ZH["watch"],
        )

    def view(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "verb": self.verb,
            "target": self.target,
            "kicks": self.kicks,
            "pushes": self.pushes,
            "rolls_seen": self.rolls_seen,
            "misses": self.misses,
            "lost": self.lost,
            "moves": None if self.approach is None else self.approach.moves,
        }


def ball_features(book: ThingBook, odom: tuple[float, float, float], t_ms: int) -> dict[str, float]:
    """``ball`` (a thing it knows rolls, near and recently seen), ``ball_rolling`` (a thing on the
    floor rolling in view now) and ``ball_fun`` (how good ball play has felt), 0..1."""
    ball = rolling = fun = small = 0.0
    for th in book.things.values():
        age = (t_ms - th.seen_ms) / 1000.0
        if age > 20.0 or th.home:
            continue
        _, rng = bearing_range(odom, th.x, th.y)
        near = _clip(1.2 - 0.3 * rng)
        fresh = _clip(1.0 - age / 20.0)
        if th.moving(t_ms) and age < 1.5:
            rolling = max(rolling, _clip(1.3 - 0.25 * rng))
        if th.rolls:
            ball = max(ball, near * fresh)
            fun = max(fun, th.fun)
        elif th.nudges == 0 and th.widest <= SMALL_M and th.held == 0 and th.stuck == 0:
            # a small thing it has not pushed yet: round ones look like they would roll (the
            # picture shows no long side) — worth a nudge to see
            round_ = th.axis is None
            small = max(small, (0.6 if round_ else 0.3) * near * fresh)
    return {"ball": max(ball, small), "ball_rolling": rolling, "ball_fun": fun}


__all__ = ["BALL_VERSION", "BallPlay", "BallStyle", "ball_features"]
