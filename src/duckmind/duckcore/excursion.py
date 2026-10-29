"""Outings (Duck Core v4, part C): a small trip, not a lone step.

An outing is a few short legs (each still ≤ 6 s at the validated walking commands) with a
look around between them, toward a place it has not seen for a while, a place where good
things happened, or the family. It remembers the room as a coarse 0.5 m grid (when it last
saw each cell, and how good it was there), goes further as it grows up, never beyond the
family's roam radius (1.5 m by default) or, while the family sleeps, 1.5 m from the nest.

Personality only shapes the randomness: every duck sets off often (0.8–1.4× by
personality), what changes is where and how. Only the first leg of an outing is a logged
decision; the rest are ``duck.walk`` legs with the outing's id and a short look between
them. Curiosity has three sockets here (the look between legs, the pull of a cell where a
thing is, setting off for a toy: :mod:`duckmind.duckcore.curiosity`), all inert while its
home has no things. Integers only; draws through the core's counters.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from duckmind.duckcore import content, curiosity, daycycle, moments, nest, rng
from duckmind.duckcore.fixed import SCALE, clamp, cos_mrad, isqrt, power, sin_mrad

if TYPE_CHECKING:
    from duckmind.duckcore.engine import DuckCore

X: dict[str, Any] = content.EXCURSION
R: dict[str, Any] = content.RHYTHM
CELL_MM = int(X["cell_mm"])
# What one more leg, its pause and the minute's drain can cost at most (battery, fixed
# point): a 6 s leg and an 8 s look (~65), and two minutes' standing drain when the leg
# straddles two minutes' beats (~416).
LEG_MARGIN = 500
CELLS_PER_SIDE = 6000 // CELL_MM
FRONT_MM = (600, 0)  # nothing new worth the walk: a few steps in front of the nest
REASON_ZH = {
    "restless": "坐不住了",
    "curious": "去看看最近沒去的地方",
    "social": "去找家人",
    "play": "剛玩過，還想動",
    "zoomies": "開心到暴衝",
    "patrol": "巡一巡喜歡的地方",
    curiosity.TOY_REASON: "去找喜歡的玩具",
}


def _e() -> Any:
    from duckmind.duckcore import engine

    return engine


def cell_of(x: int, y: int) -> int:
    ix = clamp((x + 3000) // CELL_MM, 0, CELLS_PER_SIDE - 1)
    iy = clamp((y + 3000) // CELL_MM, 0, CELLS_PER_SIDE - 1)
    return ix * CELLS_PER_SIDE + iy


def cell_center(key: int) -> tuple[int, int]:
    ix, iy = divmod(key, CELLS_PER_SIDE)
    return ix * CELL_MM - 3000 + CELL_MM // 2, iy * CELL_MM - 3000 + CELL_MM // 2


CENTERS = [(key, *cell_center(key)) for key in range(CELLS_PER_SIDE * CELLS_PER_SIDE)]


def _minute_index(core: DuckCore) -> int:
    return int((core.s.now - core.s.hatched_at).total_seconds()) // 60


def leash_mm(core: DuckCore) -> int:
    """How far from the nest it may go right now: the family's roam radius, how far a duck
    of its age goes, 1.5 m while the family sleeps, less what it is unsure of its pose."""
    s = core.s
    # A bold duck ranges a little further at its age, a timid one a little less (0.8–1.2×).
    stage_radius = int(X["radius_mm"][core.stage().stage_id]) * core._axis("boldness", "leash")
    leash = min(s.roam_radius_mm, stage_radius // SCALE)
    if s.phase == "solo":
        leash = min(leash, int(R["solo_radius_mm"]))
    return max(500, leash - s.pose_err_mm)


def mark_seen(core: DuckCore, x: int, y: int) -> bool:
    """It looked around here: the cell is fresh again. True when it had not been there for
    a while (a new place, as far as today's duck is concerned)."""
    s = core.s
    key = cell_of(x, y)
    now = _minute_index(core)
    cell = s.cells.get(key)
    fresh = cell is None or now - int(cell[0]) >= 90
    if cell is None:
        s.cells[key] = [now, 0]
    else:
        cell[0] = now
    return fresh


def good_here(core: DuckCore, step: int = 500) -> None:
    """place-good-v1: something good happened where it is (touched, praised, played)."""
    s = core.s
    if not core.hw.nest:
        return
    key = cell_of(s.pos_x, s.pos_y)
    cell = s.cells.setdefault(key, [_minute_index(core), 0])
    cell[1] = min(SCALE, int(cell[1]) + step * (SCALE - int(cell[1])) // SCALE)


def relax_good(core: DuckCore) -> None:
    keep = int(X["good_relax_keep"])
    for cell in core.s.cells.values():
        cell[1] = int(cell[1]) * keep // SCALE


def family_spot(core: DuckCore) -> tuple[int, int] | None:
    """Where the family is most often (the best cell), or the way to the front door."""
    s = core.s
    leash = leash_mm(core) - 100
    best, best_good = None, 0
    for key, cell in sorted(s.cells.items()):
        x, y = cell_center(key)
        if int(cell[1]) > best_good and x * x + y * y <= leash * leash:
            best, best_good = (x, y), int(cell[1])
    if best is not None and best_good >= 1000:
        return best
    dx, dy = R["door_xy"]
    length = isqrt(int(dx) * int(dx) + int(dy) * int(dy))
    reach = min(leash, length)
    return int(dx) * reach // length, int(dy) * reach // length


# ---------------------------------------------------------------------------- starting


def outing_mult(core: DuckCore) -> int:
    """Personality (0.8–1.4×), age, mood, what the family taught, how tired, conservative."""
    E = _e()
    s = core.s
    low, high = X["style_band"]
    mult = clamp(core._style("outing", core._leaning()), int(low), int(high))
    mult = mult * content.MOTION_STAGE[core.stage().stage_id]["outing"] // SCALE
    mult = mult * clamp(8000 + s.arousal * 3 // 10 + s.valence // 5, 8000, 12500) // SCALE
    pref = s.prefs.get("household.verb:wander", 5000)
    mult = mult * (7500 + pref // 2) // SCALE
    mult = mult * core._axis("activity", "outing") // SCALE  # its temperament: 0.8–1.25×
    mult = mult * core._tiredness() // SCALE
    if s.conservative:
        mult = mult * E.CONSERVATIVE_OUTINGS // SCALE
    return mult


def can_start(core: DuckCore) -> str | None:
    """Why it may not set off on an outing right now (None: it may)."""
    s = core.s
    if not core.hw.nest:
        return "legacy"
    if s.excursion is not None or s.homing is not None:
        return "busy"
    if s.phase not in ("solo", "family"):
        return s.phase
    if s.family in ("away", "unknown"):
        return "alone"
    if (
        s.phase == "family"
        and not daycycle.recent(core, s.cue_logged_at, int(R["excursion_cue_min"]))
        and not core._present(int(R["excursion_cue_min"]))
    ):
        return "quiet"
    if s.door_hold_until is not None and s.now < s.door_hold_until:
        return "door"
    if s.phase == "solo" and s.solo_exc_s >= int(R["solo_excursion_max_min"]) * 60:
        return "solo_done"
    if s.nest_trouble and not daycycle.someone_here(core):
        return "nest_trouble"
    if core._until(s.stand_block_until):
        return "stand_backoff"
    why = nest._can_walk(core)
    if why is not None:
        return why
    if s.battery <= nest.threshold(core) + LEG_MARGIN + 200:
        return "going_home"
    if core._tiredness() == 0:
        return "tired"
    if s.posture == "fallen":
        return "fallen"
    return None


def pick_reason(core: DuckCore) -> str:
    """Why it sets off (drawn; exc.reason)."""
    s = core.s
    zoom: dict[str, Any] = X["zoomies"]
    played = s.last_play_at is not None and s.now - s.last_play_at <= timedelta(minutes=30)
    reunited = s.reunion_until is not None and s.now < s.reunion_until
    if (
        (played or reunited)
        and s.valence >= int(zoom["valence"])
        and s.arousal >= int(zoom["arousal"])
        and not s.conservative
        and s.battery >= max(5000, nest.threshold(core) + 2000)
        and s.energy >= 4000
    ):
        return "zoomies"
    if (
        s.family == "awake"
        and s.phase == "family"
        and (s.social >= 4000 or s.bid_pending_since is not None)
    ):
        return "social"
    if curiosity.wants_toy(core):  # (never without things, and never a draw then)
        return curiosity.TOY_REASON
    if (
        core.stage().stage_id == "adult"
        and sum(1 for cell in s.cells.values() if int(cell[1]) >= 1000) >= 3
        and rng.below(core._rand("exc.reason"), 3000)
    ):
        return "patrol"
    if played:
        return "play"
    return "curious" if rng.below(core._rand("exc.reason"), 5000) else "restless"


def maybe_start(core: DuckCore) -> bool:
    """outing-v2: seated on the floor, awake and free, it sets off every minute or two."""
    s = core.s
    if s.nested or s.posture != "sit" or core._until(s.napping_until):
        return False
    if can_start(core) is not None:
        return False
    urge = SCALE
    if s.last_excursion_end is not None:
        since = int((s.now - s.last_excursion_end).total_seconds()) // 60
        urge = min(SCALE, since * SCALE // max(1, int(X["refractory_min"])))
    chance = clamp(int(X["floor_hazard"]) * outing_mult(core) // SCALE * urge // SCALE)
    if chance <= 0:
        return False
    raw = core._rand("outing")
    if not rng.below(raw, chance):
        return False
    reason = pick_reason(core)
    if not somewhere_to_go(core, reason):
        return False  # its leash and the spots it avoids leave it nowhere to go: it stays
    decision = core._emit(
        "duck.decision",
        {
            "verb": "stand_up",
            "rule": "outing-v2",
            "trigger": reason,
            "from": "floor",
            "table": core.table,
            "counter": core.counters["outing"],
            "draw": rng.unit(raw),
            "inputs": core._decision_inputs(None),
            "utilities": {"stand_up": chance, "stay": SCALE - chance},
            "probabilities": {"stand_up": chance, "stay": SCALE - chance},
        },
    )
    return set_out(core, reason, parents=(decision.event_id,))


def set_out(core: DuckCore, reason: str, parents: tuple[str, ...] = ()) -> bool:
    """Stand up (off the nest, if it is on it) and walk the first leg of an outing."""
    s = core.s
    from_nest = s.nested
    if s.posture == "sit":
        core._perform("stand_up", reason="set_off", parents=parents, force_sound=())
        if s.posture != "stand":
            return False
    if reason == "zoomies" and not moments.zoomies(core):
        reason = "play"  # (a zoomies burst is one of its goofy moments)
    first_today = s.last_outing_day != s.now.date().isoformat()
    s.last_outing_day = s.now.date().isoformat()
    play = _play_spot(core)
    plan_ahead = moments.set_off(core, reason, first_today, play)
    if plan_ahead == "false_start":
        return False  # up, a look round, and straight back down
    stage = core.stage().stage_id
    low, high = X["legs"][stage]
    legs = int(low) + rng.pick_index(core._rand("exc.legs"), int(high) - int(low) + 1)
    if s.conservative:
        legs = max(1, legs - 1)
    if reason == "zoomies":
        legs = int(X["zoomies"]["legs"])
    s.excursions += 1
    s.day_excursions += 1
    ident = f"X{s.excursions}"
    s.excursion = {
        "id": ident,
        "reason": reason,
        "target": None,
        "legs_left": legs,
        "legs": 0,
        "started": s.now,
        "walk_s": 0,
        "cm": 0,
        "cells_new": 0,
        "from": "nest" if from_nest else "floor",
        "max_mm": nest.dist_mm(core),
        "pause": False,
        "phase": s.phase,
    }
    start = core._emit(
        "duck.excursion",
        {
            "phase": "start",
            "id": ident,
            "reason": reason,
            "zh": REASON_ZH.get(reason, reason),
            "from": s.excursion["from"],
            "legs_planned": legs,
            "family": s.family,
            "day_phase": s.phase,
        },
        parents=parents,
    )
    s.excursion["event"] = start.event_id
    if plan_ahead == "remembers_play_spot" and play is not None:
        s.excursion["target"] = cell_center(play)  # yesterday's play spot first
    if from_nest:
        _walk(core, _leave_plan(core), "leave")  # out of the nest's opening first
        return True
    next_leg(core)
    return True


# ---------------------------------------------------------------------------- legs


def _end_check(core: DuckCore) -> str | None:
    """Why the outing is over now (None: it goes on)."""
    E = _e()
    s = core.s
    exc = s.excursion
    assert exc is not None
    if s.phase not in ("solo", "family"):
        return {"alone": "family_away", "wind_down": "bedtime", "night": "bedtime"}.get(
            s.phase, s.phase
        )
    if s.family in ("away", "unknown"):
        return "family_away"  # (the app said so this minute: the phase follows at its beat)
    if (
        s.phase == "solo"
        and s.solo_exc_s + int((s.now - exc["started"]).total_seconds())
        >= int(R["solo_excursion_max_min"]) * 60
    ):
        return "solo_done"
    if s.battery <= nest.threshold(core) + LEG_MARGIN:
        # One more leg could take it below the battery kept for the way home: home now.
        return "battery"
    if s.energy < E.SIT_ENERGY + 500:
        return "tired"
    if core._must_sit() is not None:
        return "must_sit"
    why = nest._can_walk(core)
    if why is not None:
        return why
    if int(exc["legs_left"]) <= 0:
        return "legs"
    leash = leash_mm(core)
    if s.pos_x * s.pos_x + s.pos_y * s.pos_y > leash * leash:
        return "leash"  # (the leash shrank as it grew unsure of where it is): back home
    return None


def minute(core: DuckCore) -> bool:
    """An outing in progress: its legs run as time passes (True: nothing else to decide)."""
    s = core.s
    exc = s.excursion
    if exc is None:
        return False
    if s.posture in ("stand", "walk"):
        return True
    end(core, "sat", decide=False)
    return False


def next_leg(core: DuckCore) -> None:
    """After a leg: a short look around (one clip), then the next leg, until it is over."""
    s = core.s
    exc = s.excursion
    if exc is None or s.posture != "stand":
        return
    s.errand_step += 1
    if exc["pause"]:
        exc["pause"] = False
        key = cell_of(s.pos_x, s.pos_y)
        good = s.cells.get(key)
        if good is not None and int(good[1]) >= moments.CALM_SPOTS:
            exc.setdefault("good_seen", [])
            if key not in exc["good_seen"]:
                exc["good_seen"].append(key)  # one of its favourite places (its patrol)
        moments.arrived(core, key)
        if curiosity.at_pause(core):
            return  # a thing close by: curiosity has the pause (the outing goes on after it)
        leaving = exc["from"] == "nest" and int(exc["legs"]) == 1
        did = None
        if exc["reason"] != "zoomies":
            did = moments.pause(core, leaving=leaving)
        if did is None and exc["reason"] != "zoomies" and rng.below(core._rand("exc.stop"), 4000):
            clips = X["stop_clips"]
            clip = clips[rng.pick_index(core._rand("exc.stop"), len(clips))]
            core._express(clip, reason="pause", force_sound=())
        if did == "chases_tail":
            return  # it walks its little circle; the look round comes after
        low, high = X["stop_s"]
        s.leg_pause_s = int(low) + rng.pick_index(core._rand("exc.stop"), int(high) - int(low) + 1)
        if did is not None:
            s.leg_pause_s += moments.calm_s(core)  # …and then, for a while, nothing happens
        s.stand_idle_from = s.now
        if mark_seen(core, s.pos_x, s.pos_y):
            exc["cells_new"] = int(exc["cells_new"]) + 1
            core._self_note("explorer", str(exc["event"]), "new_place")
            # Somewhere it had not been for a while: finding it is its own small reward.
            E = _e()
            s.valence = min(SCALE, s.valence + E.EXPLORE_VALENCE)
            s.arousal = min(SCALE, s.arousal + E.EXPLORE_AROUSAL)
        return
    ended = _end_check(core)
    if ended is not None:
        end(core, ended)
        return
    target = exc.get("target")
    if target is None or _arrived(core, target):
        target = exc["target"] = pick_target(core, str(exc["reason"]))
        if target is None:
            end(core, "nowhere")  # nowhere left it may go: not a spin on the spot instead
            return
    purpose = "social" if exc["reason"] == "social" else "explore"
    vx = None
    if exc["reason"] == "zoomies":
        purpose, vx = "zoomies", int(X["zoomies"]["vx"])
    else:
        E = _e()
        low, high = E.WALK_VX_MM
        vx = low + rng.pick_index(core._rand("exc.vx"), high - low + 1)
    plan = core._plan_leg_toward(tuple(target), purpose, vx=vx, longest=_leg_seconds(core))
    _walk(core, plan, purpose)


def _leg_seconds(core: DuckCore) -> int:
    """How long this leg may be (exc.leg_s): younger ducks and conservative mode walk
    shorter legs, adults the longer ones."""
    E = _e()
    s = core.s
    stage_id = core.stage().stage_id
    scale = content.MOTION_STAGE[stage_id]["scale"] * (5000 if s.conservative else SCALE)
    longest = max(E.MIN_WANDER_S, E.WALK_S_MAX * scale // (SCALE * SCALE))
    shortest = max(E.MIN_WANDER_S, longest * 2 // 3)
    span = longest - shortest + 1
    seconds = shortest + rng.pick_index(core._rand("exc.leg_s"), span)
    if stage_id == "adult":
        seconds = max(seconds, shortest + rng.pick_index(core._rand("exc.leg_s"), span))
    return int(seconds)


def _arrived(core: DuckCore, target: Any) -> bool:
    s = core.s
    dx, dy = int(target[0]) - s.pos_x, int(target[1]) - s.pos_y
    return dx * dx + dy * dy <= int(X["arrive_mm"]) ** 2


def _walk(core: DuckCore, plan: dict[str, Any], purpose: str) -> None:
    s = core.s
    exc = s.excursion
    assert exc is not None
    exc["legs_left"] = int(exc["legs_left"]) - (0 if purpose == "leave" else 1)
    exc["legs"] = int(exc["legs"]) + 1
    exc["pause"] = True
    s.leg_pause_s = 0
    core._walk_leg(plan, reason="excursion", outing_id=str(exc["id"]))


def _clear_way(spots: list[dict[str, Any]], start: tuple[int, int], x: int, y: int) -> bool:
    """Whether the straight way from ``start`` to (x, y) mm keeps clear of every spot it
    learned to avoid, by the rule every path keeps: never closer to one than its radius
    (from inside one, only away from it)."""
    x0, y0 = start
    dx, dy = x - x0, y - y0
    length2 = dx * dx + dy * dy
    for spot in spots:
        sx, sy, radius = int(spot["x"]), int(spot["y"]), int(spot["r"])
        along = 0
        if length2:
            along = clamp(((sx - x0) * dx + (sy - y0) * dy) * SCALE // length2)
        px, py = x0 + dx * along // SCALE, y0 + dy * along // SCALE
        near2 = (px - sx) ** 2 + (py - sy) ** 2
        if near2 < radius * radius and near2 < (x0 - sx) ** 2 + (y0 - sy) ** 2:
            return False
    return True


def _can_go(
    core: DuckCore,
    spots: list[dict[str, Any]],
    start: tuple[int, int],
    x: int,
    y: int,
    leash: int,
) -> bool:
    """Whether (x, y) mm is somewhere it may head for from ``start``: inside ``leash`` (its
    own less 100 mm), not in (or just by) a spot it learned to avoid, the straight way there
    clear of them, and room to turn round toward home when it gets there."""
    if x * x + y * y > leash * leash:
        return False
    for spot in spots:
        if (x - int(spot["x"])) ** 2 + (y - int(spot["y"])) ** 2 < (int(spot["r"]) + 100) ** 2:
            return False
    if not _clear_way(spots, start, x, y):
        return False
    dx, dy = x - start[0], y - start[1]
    dist = isqrt(dx * dx + dy * dy)
    # (arriving facing out at the edge, with no room to turn home, is not somewhere to go)
    return dist == 0 or core._turn_home_fits(x, y, dx * SCALE // dist, dy * SCALE // dist)


def _choices(
    core: DuckCore, reason: str, pose: tuple[int, int, int]
) -> tuple[dict[str, int], dict[str, tuple[int, int]], tuple[int, int] | None]:
    """What ``pick_target`` draws from, seen from ``pose``: the cells worth the walk (their
    weights and centres) and, when there is none, where it goes instead (None: nowhere)."""
    s = core.s
    x0, y0, heading = pose
    spots = core.active_spots()
    leash = leash_mm(core) - 100
    if reason == "social":
        spot = family_spot(core)
        if (
            spot is not None
            and (spot[0] - x0) ** 2 + (spot[1] - y0) ** 2 > int(X["arrive_mm"]) ** 2
            and _can_go(core, spots, (x0, y0), *spot, leash)
        ):
            return {}, {}, spot
    now = _minute_index(core)
    keep = int(X["novelty_keep"])
    novelty = core._axis("exploration", "novelty")
    heading_cos, heading_sin = cos_mrad(heading), sin_mrad(heading)
    patrol = reason == "patrol"
    pull = curiosity.attractions(core)  # cells with a thing it is keen on (none: empty)
    weights: dict[str, int] = {}
    points: dict[str, tuple[int, int]] = {}
    least_bad: tuple[int, int, tuple[int, int]] | None = None  # (value, -key, cell)
    for key, x, y in CENTERS:
        if x * x + y * y > leash * leash:
            continue
        dx, dy = x - x0, y - y0
        dist = isqrt(dx * dx + dy * dy)
        if dist < CELL_MM or not _can_go(core, spots, (x0, y0), x, y, leash):
            continue
        cell = s.cells.get(key)
        if cell is None:
            fresh = SCALE
            good = 0
        else:
            age = min(2000, now - int(cell[0]))
            fresh = SCALE - power(keep, age)
            good = int(cell[1])
        facing = (dx * heading_cos + dy * heading_sin) // max(1, dist)  # −SCALE…SCALE
        fresh = fresh * novelty // SCALE  # its temperament: how much new places draw it
        value = fresh + good * (3 if patrol else 1) - dist * 2 - (SCALE - facing) * 3000 // SCALE
        if pull:
            value += pull.get(key, 0)
        if value > 0:
            weights[str(key)] = value
            points[str(key)] = (x, y)
        if least_bad is None or (value, -key) > least_bad[:2]:
            least_bad = (value, -key, (x, y))
    if weights:
        return weights, points, None
    if _can_go(core, spots, (x0, y0), *FRONT_MM, leash):
        return {}, {}, FRONT_MM
    return {}, {}, (least_bad[2] if least_bad is not None else None)


def pick_target(core: DuckCore, reason: str) -> tuple[int, int] | None:
    """excursion-target-v2: a cell it has not seen for a while, weighed against how good it
    was there and how far (and how far round) it is; anything at least a quarter as good as
    the best stays in the draw. Only somewhere it may head for (``_can_go``: never into, or
    through, a spot it learned to avoid). Nothing new worth the walk: a few steps in front
    of the nest, else the least bad place it can go; nowhere at all (its leash and its
    spots leave it no room): None, and the outing is over."""
    weights, points, instead = _choices(core, reason, core._pose())
    if not weights:
        return instead
    top = max(weights.values())
    kept = {key: value for key, value in weights.items() if value * 4 >= top}
    choice = rng.pick_weighted(core._rand("exc.target"), kept)
    return points[choice]


def somewhere_to_go(core: DuckCore, reason: str) -> bool:
    """Whether an outing would find anywhere to go (no draw): from where it stands, or off
    its nest from where its first leg out would end."""
    pose = core._pose()
    if core.s.nested:
        x, y, heading = _leave_plan(core)["to"]
        pose = (int(x), int(y), int(heading))
    weights, _, instead = _choices(core, reason, pose)
    return bool(weights) or instead is not None


def _leave_plan(core: DuckCore) -> dict[str, Any]:
    """The first leg out of the nest's opening: forward, and if it sat facing in (the way
    it walked in), round on a right arc (owner decision D1: never turning on the spot,
    never backing out)."""
    return core._plan_leg_toward(
        (700, core.s.pos_y), "leave", longest=int(content.NEST["leave_leg_s"]), side=-1
    )


def end(core: DuckCore, ended_by: str, *, decide: bool = True) -> None:
    """The outing is over: log it, then home to the nest or sit down here (home-base-v1)."""
    E = _e()
    s = core.s
    exc = s.excursion
    if exc is None:
        return
    s.excursion = None
    s.last_excursion_end = s.now
    s.last_outing = {"at": s.now, "cm": int(exc["cm"]), "id": exc["id"]}
    duration = int((s.now - exc["started"]).total_seconds())
    if exc.get("phase") == "solo":
        s.solo_exc_s += duration
    s.walk_sat = min(SCALE, s.walk_sat + E.WALK_SAT_WANDER)
    core._emit(
        "duck.excursion",
        {
            "phase": "end",
            "id": exc["id"],
            "reason": exc["reason"],
            "legs": exc["legs"],
            "walk_s": exc["walk_s"],
            "cm": exc["cm"],
            "cells_new": exc["cells_new"],
            "max_mm": exc["max_mm"],
            "seconds": duration,
            "ended_by": ended_by,
        },
        parents=(str(exc["event"]),),
    )
    moments.excursion_end(core, exc)
    if exc["reason"] == "zoomies" and s.posture in ("stand", "walk") and decide:
        moments.zoomies_end(core)  # …and plops down as if nothing had happened
        return
    if not decide or s.posture not in ("stand", "walk"):
        return
    home_base(core, ended_by)


def _play_spot(core: DuckCore) -> int | None:
    """Where the family played with it yesterday (a cell away from the nest), if it may
    still go there."""
    s = core.s
    if s.play_cell is None:
        return None
    cell, day = int(s.play_cell[0]), str(s.play_cell[1])
    yesterday = (s.now.date() - timedelta(days=1)).isoformat()
    if day != yesterday:
        return None
    x, y = cell_center(cell)
    if x * x + y * y < CELL_MM * CELL_MM:
        return None
    here = (s.pos_x, s.pos_y)
    if not _can_go(core, core.active_spots(), here, x, y, leash_mm(core) - 100):
        return None  # (past its leash, or where it learned not to walk)
    return cell


def home_base(core: DuckCore, ended_by: str) -> None:
    """home-base-v1: back to the nest (forced when the battery, the family, bedtime or
    tiredness says so; otherwise drawn, likelier the emptier the battery), or sit here."""
    E = _e()
    s = core.s
    thr = nest.threshold(core)
    forced = {
        "battery": "battery",
        "family_away": "alone",
        "bedtime": "bedtime",
        "tired": "tired",
        "solo_done": "solo_done",
        "leash": "leash",
    }.get(ended_by)
    if forced is None and s.battery <= thr:
        forced = "battery"
    if forced is None and s.energy < E.SIT_ENERGY + 500:
        forced = "tired"
    if forced is not None:
        if nest._can_walk(core) is None:
            nest.start_homing(core, forced)
        else:
            core._sit_down_now("excursion_end")
        return
    if ended_by not in ("legs", "must_sit", "nowhere") or s.posture != "stand":
        core._sit_down_now("excursion_end")
        return
    base = int(content.NEST["home_base_p"])
    ramp = content.NEST["ramp_hi"] or nest.leave_battery(core)
    ramp = int(ramp)
    lean = clamp((ramp - s.battery) * SCALE // max(1, ramp - thr))
    chance = base + (SCALE - base) * lean // SCALE
    raw = core._rand("nest.home")
    home = rng.below(raw, chance)
    decision = core._emit(
        "duck.decision",
        {
            "verb": "go_home" if home else "sit_down",
            "rule": "home-base-v1",
            "trigger": ended_by,
            "table": core.table,
            "counter": core.counters["nest.home"],
            "draw": rng.unit(raw),
            "inputs": {"battery": s.battery, "threshold": thr, "ramp": ramp},
            "utilities": {"go_home": chance, "sit_down": SCALE - chance},
            "probabilities": {"go_home": chance, "sit_down": SCALE - chance},
        },
    )
    if home:
        nest.start_homing(core, "home_base", parents=(decision.event_id,))
    else:
        core._perform("sit_down", reason="excursion_end", parents=(decision.event_id,))
