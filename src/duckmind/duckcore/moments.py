"""Clever and goofy moments (Duck Core v4, part F; phase P1).

*Clever* is a skill it really learned: it remembers when the family comes home, docks in
one go from far away, finds the family where they usually sit, walks round the spot where
it fell. It always guesses first (``duck.predict``) and the moment counts only when the
guess came true, with both events as its evidence: it never pretends to be clever.

*Goofy* is what real birds do: preening in the middle of something (displacement
activity), dabbling on a dry floor (vacuum activity), standing up and sitting straight
back down (an intention movement), looking twice. It happens only when the duck is fed,
rested and in no hurry (``relaxed``), so it never looks like a fault, and always with the
comic beat: the set-up, a freeze, a double take or a 嗯？, then a sheepish preen.

Every day has its form (ordinary, sharp or muddled); just woken, tired or over-excited
it is a little slower. Its wits on the moment decide which kind it is, and a clever guess
that went wrong turns into the goofy one (asleep while waiting at the door, looking the
wrong way when called). A moment the family laughs at on three growth days becomes a
running gag. How often: P = C·N·weight (RARE_TIERS clever and goofy; N the awake minutes
since the last one of that kind; the weight smaller where it comes round all the time),
a few a day of each kind, the same moment at most every 45 minutes.

Integers only; every draw through the core's counters (the day's form from the date, like
the idle glances); every decision and outcome is an L1 event. All motion goes through the
same gates as anything else it does.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from duckmind.duckcore import content, rng
from duckmind.duckcore.fixed import SCALE, clamp, cos_mrad, isqrt, sin_mrad

if TYPE_CHECKING:
    from duckmind.duckcore.engine import DuckCore
    from duckmind.provenance import Event

M: dict[str, Any] = content.MOMENT_RULES
BY_ID = content.MOMENT_BY_ID
BY_HOOK: dict[str, tuple[content.Moment, ...]] = {}
for _moment in content.MOMENTS:
    BY_HOOK[_moment.hook] = BY_HOOK.get(_moment.hook, ()) + (_moment,)
FORM_ZH = {"ordinary": "平常的一天", "sharp": "今天特別機靈", "muddled": "今天有點迷糊"}
URGENT_HOME = ("battery", "bedtime", "alone", "tired", "retry")
CALM_SPOTS = 1000  # a cell this good is one of its favourite places


def _e() -> Any:
    from duckmind.duckcore import engine

    return engine


# ---------------------------------------------------------------------------- wits


def _form_of(core: DuckCore, day: str) -> str:
    """A day's form, from its date (never a decision counter)."""
    u = rng.unit(rng.draw(core.seed, core.table, f"wits.form.{day}", 1))
    acc = 0
    for name, share in M["form"].items():
        acc += int(share)
        if u < acc:
            return str(name)
    return "muddled"


def day_form(core: DuckCore, *, log: bool = True) -> str:
    """Today's form: logged the first time it matters each day (duck.day_form). With
    ``log`` False (the page looking), nothing is written."""
    s = core.s
    today = s.now.date().isoformat()
    if s.form_day == today and s.form is not None:
        return s.form
    form = _form_of(core, today)
    if log:
        s.form_day, s.form = today, form
        core._emit("duck.day_form", {"form": form, "date": today, "zh": FORM_ZH[form]})
    return form


def wits(core: DuckCore, *, log: bool = True) -> int:
    """How sharp it is right now (spec 8.2): its wit, today's form, how long it has been
    awake, how tired and how excited it is, and its age. Always 15–85%: both kinds of
    moment stay possible."""
    s = core.s
    t: dict[str, Any] = content.TEMPERAMENT
    low, high = t["wits"]
    wit = int(s.temperament.get("wit", 5000))
    value = int(low) + (wit - 2000) * (int(high) - int(low)) // 6000
    since = None
    for moment in (s.woke_at, s.nap_ended_at, s.boot_at):
        if moment is not None and moment <= s.now:
            gap = int((s.now - moment).total_seconds()) // 60
            since = gap if since is None else min(since, gap)
    inertia = SCALE
    if since is not None:
        inertia = 6000 if since < 5 else 8000 if since < 15 else SCALE
    fatigue = 8500 if s.energy < 4000 else SCALE
    arousal = 11500 - abs(s.arousal - 5000) * 7 // 10
    stage = int(M["stage_mult"][core.stage().stage_id])
    for mult in (int(M["form_mult"][day_form(core, log=log)]), inertia, fatigue, arousal, stage):
        value = value * mult // SCALE
    lo, hi = M["wits_clamp"]
    return clamp(value, int(lo), int(hi))


def relaxed(core: DuckCore, hook: str) -> tuple[bool, dict[str, Any]]:
    """Whether a goofy moment may happen now (spec 8.4), and the inputs that said so."""
    E = _e()
    s = core.s
    line = E.MIN_MOTION_BATTERY
    if core.hw.nest:
        from duckmind.duckcore import nest

        line = nest.threshold(core)
    field: dict[str, Any] = {
        "battery": s.battery,
        "battery_min": max(5000, line + 2000),
        "energy": s.energy,
        "overload": s.overload,
        "phase": s.phase,
        "homing": (s.homing or {}).get("reason"),
    }
    phases = ["family", "solo"]
    if hook == "dawn":
        phases.append("dawn")
    if hook == "anticipate_miss":
        phases.append("alone")
    ok = (
        (s.docked or s.battery >= field["battery_min"])
        and s.energy >= 4000
        and s.overload < 5000
        and field["homing"] not in URGENT_HOME
        and not s.hot
        and not s.yielding
        and s.posture != "fallen"
        and s.righting is None
        and not core._until(s.stop_until)
        and not core._until(s.lockout_until)
        and not core._until(s.quiet_until)
        and s.stranded is None
        and not s.unplaced
        and not core._night()
        and (s.phase in phases if core.aware else True)
    )
    field["ok"] = ok
    return ok, field


# ---------------------------------------------------------------------------- the draw


def _ready(core: DuckCore, moment: content.Moment, ctx: dict[str, Any]) -> bool:
    """Whether this moment could happen here and now (its age, its gap, what it needs)."""
    s = core.s
    if s.gd < moment.gd_min:
        return False
    last = s.moment_last.get(moment.moment_id)
    if last is not None and s.now - last < timedelta(minutes=int(M["same_id_min"])):
        return False
    if moment.moment_id in s.moment_attempts:
        return False  # already guessing this one
    needs = moment.needs
    if needs == "ear":
        return core.hw.ear
    if needs == "tof":
        return core.hw.tof
    if needs == "name":
        return core.hw.name_ear and "learned_name" in s.milestones
    if needs == "depart_model":
        return bool(ctx.get("depart_due"))
    if needs == "far":
        return int(ctx.get("from_mm", 0)) >= int(M["far_mm"])
    if needs == "long_way":
        return bool(ctx.get("long_way"))
    if needs == "avoid_spot":
        return bool(core.active_spots())
    if needs == "family_spot":
        return bool(ctx.get("evening")) and _good_cells(core) >= 3
    if needs == "bedtime":
        return bool(ctx.get("bedtime"))
    if needs == "patrol":
        return ctx.get("reason") == "patrol"
    if needs == "play_spot":
        return ctx.get("play_spot") is not None
    return True


def _good_cells(core: DuckCore) -> int:
    return sum(1 for cell in core.s.cells.values() if int(cell[1]) >= CALM_SPOTS)


def _audience(core: DuckCore) -> bool:
    from duckmind.duckcore import daycycle

    if core.aware:
        return daycycle.someone_here(core)
    return core._present(15)


def _open(core: DuckCore, kind: str, hook: str) -> bool:
    """The kind's frequency gate, without drawing: its age, its gap, today's cap, and
    whether this place can have one of that kind at all."""
    s = core.s
    if _hook_weight(hook, kind) <= 0:
        return False
    spec = content.RARE_TIERS[kind]
    if s.gd < int(spec["gd_min"]):
        return False
    last = s.rare_last.get(kind)
    if last is not None and s.now - last < timedelta(minutes=int(spec["min_gap_min"])):
        return False
    return s.rare_today.get(kind, 0) < int(spec["per_day"])


def _hook_weight(hook: str, kind: str) -> int:
    clever, goofy = M["hook_rate"].get(hook, (SCALE, SCALE))
    return int(clever if kind == "clever" else goofy)


def _rate(core: DuckCore, kind: str, hook: str, audience: bool) -> tuple[bool, int, int]:
    """P = C·N·weight (the tier's pseudo-random distribution, N counting the awake minutes
    since its last moment of that kind), weighed by how often this place comes round
    (MOMENT_RULES hook_rate): (hit, draw, counter)."""
    s = core.s
    spec = content.RARE_TIERS[kind]
    n = max(1, s.rare_n.get(kind, 0))
    chance = int(spec["c_ppm"]) * n * _hook_weight(hook, kind) // SCALE
    chance = min(content.PPM, chance)
    if kind == "goofy" and not audience:
        chance = chance * int(M["audience"]) // SCALE  # goofy is for someone to see
    raw = core._rand(f"prd.{kind}")
    counter = core.counters[f"prd.{kind}"]
    if (raw * content.PPM) >> 64 >= chance:
        return False, rng.unit(raw), counter
    s.rare_n[kind] = 0
    return True, rng.unit(raw), counter


def _weight(core: DuckCore, moment: content.Moment, audience: bool) -> int:
    """pick weight = base × personality tint × what the family liked × fresh × gag."""
    s = core.s
    tints = dict(moment.tint)
    if s.branch is not None:
        tint = int(tints.get(s.branch, SCALE))
    else:
        tint = sum(s.nature[b] * int(tints.get(b, SCALE)) for b in sorted(s.nature)) // SCALE
    value = SCALE * tint // SCALE
    pref = s.prefs.get(f"household.moment:{moment.moment_id}", 5000)
    value = value * (7500 + pref // 2) // SCALE
    last = s.moment_last.get(moment.moment_id)
    if last is None or s.now - last >= timedelta(days=int(M["fresh_days"])):
        value = value * 15000 // SCALE
    if moment.moment_id in s.gags:
        value = value * 2
    del audience
    return max(1, value)


def at_hook(
    core: DuckCore,
    hook: str,
    feasible: Callable[[str], bool] | None = None,
    **ctx: Any,
) -> str | None:
    """Where a moment could happen: choose the kind by its wits, check the rate, pick one,
    and start it (a clever guess, or the goofy act itself). ``feasible`` says whether the
    chosen one can really be done here (a clear path for a walk); it is asked before
    anything is logged. Returns the moment's id when one started, else None."""
    s = core.s
    if not s.awake or core._until(s.napping_until):
        return None
    moments = BY_HOOK.get(hook, ())
    ready = [m for m in moments if _ready(core, m, ctx)]
    clever = [m for m in ready if m.kind == "clever" and _open(core, "clever", hook)]
    goofy = [m for m in ready if m.kind == "goofy" and _open(core, "goofy", hook)]
    if goofy and not relaxed(core, hook)[0]:
        goofy = []
    if clever and s.moment_attempts:
        clever = []  # one guess at a time
    if not clever and not goofy:
        return None
    sharp = wits(core)
    raw = core._rand("moment.kind")
    if clever and goofy:
        kind = "clever" if rng.below(raw, sharp) else "goofy"
    elif clever:
        kind = "clever" if rng.below(raw, sharp) else ""
    else:
        kind = "goofy" if rng.below(raw, SCALE - sharp) else ""
    if not kind:
        return None
    audience = _audience(core)
    hit, draw, counter = _rate(core, kind, hook, audience)
    if not hit:
        return None
    pool = clever if kind == "clever" else goofy
    weights = {m.moment_id: _weight(core, m, audience) for m in pool}
    choice = rng.pick_weighted(core._rand(f"moment.pick.{hook}"), weights)
    if feasible is not None and not feasible(choice):
        return None  # it cannot really do that here (nothing is logged)
    moment = BY_ID[choice]
    info = {
        "wits": sharp,
        "form": s.form,
        "draw": draw,
        "counter": counter,
        "audience": audience,
        "weights": weights,
    }
    if moment.kind == "clever":
        guess(core, moment, hook, info, ctx)
    else:
        info["field"] = relaxed(core, hook)[1]
        s.rare_last[kind] = s.now
        record(core, moment, hook, "done", info, ctx.get("parents", ()))
    return choice


# ---------------------------------------------------------------------------- clever guesses


def guess(
    core: DuckCore,
    moment: content.Moment,
    hook: str,
    info: dict[str, Any],
    ctx: dict[str, Any],
    *,
    predict: Event | None = None,
) -> Event:
    """A clever moment starts with a guess (duck.predict); it becomes a moment only if the
    guess comes true within its window."""
    s = core.s
    if predict is None:
        predict = core._emit(
            "duck.predict",
            {
                "model": moment.moment_id,
                "expect": ctx.get("expect"),
                "window_s": moment.window_s,
                "rule": "moment-v1",
                "hook": hook,
                "wits": info.get("wits"),
                "form": info.get("form"),
                "draw": info.get("draw"),
                "counter": info.get("counter"),
                "weights": info.get("weights"),
            },
            parents=tuple(ctx.get("parents", ())),
        )
    s.rare_last["clever"] = s.now
    until = s.now + timedelta(seconds=moment.window_s) if moment.window_s else None
    s.moment_attempts[moment.moment_id] = {
        "predict": predict.event_id,
        "since": s.now,
        "until": until,
        "hook": hook,
        "info": {k: v for k, v in info.items() if k != "weights"},
        "data": {k: v for k, v in ctx.items() if k not in ("parents",)},
    }
    return predict


def attempt(core: DuckCore, moment_id: str) -> dict[str, Any] | None:
    return core.s.moment_attempts.get(moment_id)


def resolve(
    core: DuckCore,
    moment_id: str,
    hit: bool,
    *,
    detail: dict[str, Any] | None = None,
    parents: tuple[str, ...] = (),
    outcome: Event | None = None,
) -> str | None:
    """The guess came true (a clever moment, with its evidence) or not (the goofy moment
    it turns into, when there is one and it is relaxed; otherwise just the outcome)."""
    s = core.s
    state = s.moment_attempts.pop(moment_id, None)
    if state is None:
        return None
    moment = BY_ID[moment_id]
    if outcome is None:
        outcome = core._emit(
            "duck.predict_outcome",
            {"model": moment_id, "hit": hit, **(detail or {}), "rule": "moment-v1"},
            parents=(str(state["predict"]), *parents),
        )
    info = dict(state["info"])
    info["predict_event"] = str(state["predict"])
    info["outcome_event"] = outcome.event_id
    if hit:
        spec = content.RARE_TIERS["clever"]
        if s.rare_today.get("clever", 0) >= int(spec["per_day"]):
            return None
        record(core, moment, str(state["hook"]), "hit", info, (outcome.event_id,))
        return moment_id
    if moment.fail_to is None:
        return None
    fail = BY_ID[moment.fail_to]
    ok, field = relaxed(core, fail.hook)
    if not ok or s.rare_today.get("goofy", 0) >= int(content.RARE_TIERS["goofy"]["per_day"]):
        return None
    last = s.moment_last.get(fail.moment_id)
    if last is not None and s.now - last < timedelta(minutes=int(M["same_id_min"])):
        return None  # (the same goofy moment at most every 45 minutes, however it comes)
    info["field"] = field
    info["from_guess"] = moment_id
    s.rare_last["goofy"] = s.now
    event = record(core, fail, fail.hook, "missed_guess", info, (outcome.event_id,))
    _play_miss(core, fail.moment_id, event.event_id)
    return fail.moment_id


def _play_miss(core: DuckCore, moment_id: str, parent: str) -> None:
    """The goofy moment a wrong guess turned into, with its beat."""
    s = core.s
    if moment_id == "wrong_way":
        beats(core, parent, setup="look_around")  # looking for them the other way
    elif moment_id == "stare_at_nothing":
        core._express("freeze", reason="moment", parents=(parent,), force_sound=(), tempo=2300)
        beats(core, parent)
    elif moment_id == "sits_beside_nest":
        beats(core, parent)
    elif moment_id == "door_nap":
        beats(core, parent, reaction=False, closer=False)  # still, eyes on the door…
        if not core._until(s.napping_until) and s.posture == "sit":
            from duckmind.duckcore import nest

            nest._nap(core, "away_rest")  # …and asleep, facing the door


HOMING_GUESSES = ("early_bed", "shortcut_home", "neat_dock")
OUTING_GUESSES = ("patrol", "remembers_play_spot")


def minute(core: DuckCore) -> None:
    """An awake minute: the chance of a moment of each kind grows (P = C·N); guesses whose
    time ran out, or whose trip was cut short, did not come true."""
    s = core.s
    for kind in ("clever", "goofy"):
        s.rare_n[kind] = s.rare_n.get(kind, 0) + 1
    for key in sorted(s.moment_attempts):
        state = s.moment_attempts.get(key)
        if state is None:
            continue
        until = state.get("until")
        if until is not None and s.now >= until:
            resolve(core, key, False, detail={"why": "time"})
        elif key in HOMING_GUESSES and s.homing is None:
            resolve(core, key, False, detail={"why": "cut_short"})
        elif key in OUTING_GUESSES and s.excursion is None:
            resolve(core, key, False, detail={"why": "cut_short"})
        elif s.now - state["since"] >= timedelta(hours=1):
            resolve(core, key, False, detail={"why": "time"})


def forget_guesses(core: DuckCore, why: str) -> None:
    """Switched off, lifted away, asleep for the night: open guesses are simply dropped."""
    for key in sorted(core.s.moment_attempts):
        resolve(core, key, False, detail={"why": why})


# ---------------------------------------------------------------------------- the moment itself


def record(
    core: DuckCore,
    moment: content.Moment,
    hook: str,
    outcome: str,
    info: dict[str, Any],
    parents: tuple[str, ...] = (),
) -> Event:
    """Log the moment (duck.moment) and play it: the comic beat, or the proud puff."""
    s = core.s
    first = moment.moment_id not in s.moment_album
    event = core._emit(
        "duck.moment",
        {
            "id": moment.moment_id,
            "kind": moment.kind,
            "zh": moment.zh,
            "hook": hook,
            "outcome": outcome,
            "wits": info.get("wits"),
            "form": info.get("form"),
            "draw": info.get("draw"),
            "counter": info.get("counter"),
            "rule": "moment-v1",
            "field": info.get("field"),
            "predict_event": info.get("predict_event"),
            "outcome_event": info.get("outcome_event"),
            "from_guess": info.get("from_guess"),
            "audience": info.get("audience"),
            "gag": moment.moment_id in s.gags,
            "first": first,
        },
        parents=tuple(parents),
    )
    s.moment_last[moment.moment_id] = s.now
    s.rare_today[moment.kind] = s.rare_today.get(moment.kind, 0) + 1
    s.moments_today[moment.kind] = s.moments_today.get(moment.kind, 0) + 1
    if s.adult_name is None:
        s.moment_counts[moment.moment_id] = s.moment_counts.get(moment.moment_id, 0) + 1
    if first:
        s.moment_album.append(moment.moment_id)
    s.last_moment = {"id": moment.moment_id, "at": s.now, "event": event.event_id, "seen": False}
    s.last_flag = {"kind": "moment", "id": moment.moment_id, "at": s.now, "event": event.event_id}
    if moment.kind == "clever":
        core._express(
            "proud_puff", reason="moment", parents=(event.event_id,), force_sound=_quiet(core)
        )
    return event


def _quiet(core: DuckCore, *sounds: str) -> tuple[str, ...]:
    """The moment's sounds, or none while the family sleeps (hush) or it must be quiet."""
    s = core.s
    if s.hush or core._until(s.quiet_until) or s.hot:
        return ()
    return sounds


def beats(
    core: DuckCore,
    parent: str,
    *,
    setup: str | None = None,
    setup_sounds: tuple[str, ...] = (),
    closer: bool = True,
    reaction: bool = True,
) -> None:
    """The comic beat (spec 8.4): the set-up, a freeze (600–1,400 ms), a reaction (a
    double take, or a tilt and 嗯？), then a sheepish preen (1.5–3 s). The clip player
    plays them one after another."""
    parents = (parent,)
    low, high = M["freeze_ms"]
    freeze = int(low) + rng.pick_index(core._rand("moment.beat"), int(high) - int(low) + 1)
    if setup is not None:
        core._express(
            setup, reason="moment", parents=parents, force_sound=_quiet(core, *setup_sounds)
        )
    core._express(
        "still", reason="moment", parents=parents, force_sound=(), tempo=1000 * SCALE // freeze
    )
    if reaction:
        if rng.below(core._rand("moment.beat"), 5000):
            core._express("double_take", reason="moment", parents=parents, force_sound=())
        else:
            core._express(
                "perk_tilt", reason="moment", parents=parents, force_sound=_quiet(core, "inquire")
            )
    if closer:
        low, high = M["closer_ms"]
        ms = int(low) + rng.pick_index(core._rand("moment.beat"), int(high) - int(low) + 1)
        core._express(
            "preen", reason="moment", parents=parents, force_sound=(), tempo=2900 * SCALE // ms
        )


def calm_s(core: DuckCore) -> int:
    """After a goofy moment, 5–20 s of nothing at all (the comic beat's rest)."""
    low, high = (int(ms) // 1000 for ms in M["calm_ms"])
    return low + rng.pick_index(core._rand("moment.beat"), high - low + 1)


def last_event(core: DuckCore) -> str:
    return core.log.events[-1].event_id


# ---------------------------------------------------------------------------- the family


REACTIONS = ("touch", "praise", "voice", "name")


def seen(core: DuckCore, by: str, evidence: Event | None) -> None:
    """Someone reacted (a touch, praise, a word, its name) within 10 s of a moment: it
    remembers they liked it, and three growth days of that make it a running gag."""
    s = core.s
    last = s.last_moment
    if by not in REACTIONS or last is None or last.get("seen"):
        return
    after = int((s.now - last["at"]).total_seconds())
    if after > int(M["reaction_s"]):
        return
    last["seen"] = True
    moment_id = str(last["id"])
    reaction = core._emit(
        "moment.reaction",
        {"id": moment_id, "after_s": after, "by": by},
        parents=(str(last["event"]),) + ((evidence.event_id,) if evidence is not None else ()),
    )
    core._learn_pref(f"moment:{moment_id}", +1, reaction)
    days = s.gag_days.setdefault(moment_id, [])
    if s.gd not in days:
        days.append(s.gd)
    moment = BY_ID[moment_id]
    ready = (
        moment.promotable
        and moment_id not in s.gags
        and len(s.gags) < int(M["gags_max"])
        and len(days) >= int(M["gag_days"])
        and (s.gag_gd is None or s.gd - s.gag_gd >= int(M["gag_every_gd"]))
    )
    if ready:
        s.gags.append(moment_id)
        s.gag_gd = s.gd
        core._emit(
            "moment.gag_promoted",
            {"id": moment_id, "zh": moment.zh, "days": list(days), "gd": s.gd},
            parents=(reaction.event_id,),
        )


def not_that(core: DuckCore, evidence: Event) -> bool:
    """「不是那個」 within 60 s of a moment: a little less of it."""
    s = core.s
    last = s.last_moment
    if last is None or s.now - last["at"] > timedelta(seconds=int(M["not_that_s"])):
        return False
    core._learn_pref(f"moment:{last['id']}", -1, evidence)
    return True


def epithet(core: DuckCore) -> str | None:
    """Its epithet when it grows up: its running gag (the latest), else the moment it had
    most often before it grew up (only moments that have an epithet)."""
    s = core.s
    for moment_id in reversed(s.gags):
        if moment_id in content.EPITHETS:
            return moment_id
    ranked = sorted(
        (key for key in s.moment_counts if key in content.EPITHETS),
        key=lambda key: (-s.moment_counts[key], key),
    )
    return ranked[0] if ranked else None


# ---------------------------------------------------------------------------- the hooks


def dawn(core: DuckCore, parent: str) -> None:
    """07:00, just woken (five_more_minutes): up, a look round, and straight back down for
    five more minutes."""
    if at_hook(core, "dawn", parents=(parent,)) != "five_more_minutes":
        return
    s = core.s
    event = last_event(core)
    stood = core._motion_block("stand_up") is None
    if stood:
        core._perform("stand_up", reason="moment", parents=(event,), force_sound=())
        stood = s.posture == "stand"
    beats(core, event, setup="look_around" if stood else "stretch_yawn", closer=False)
    if stood:
        core._perform("sit_down", reason="moment", parents=(event,), force_sound=())
    core._start_nap(False, minutes=5, reason="doze")
    if s.dawn_until is not None:
        s.dawn_until = max(s.dawn_until, s.now + timedelta(minutes=5))


def pause(core: DuckCore, *, leaving: bool) -> str | None:
    """A look between two legs of an outing: now and then something goofy instead (the
    first look after leaving the nest may greet it). Returns what it did."""
    chase: dict[str, Any] = {}

    def feasible(choice: str) -> bool:
        if choice != "chases_tail":
            return True
        plan = _chase_plan(core)
        chase["plan"] = plan
        return plan is not None

    choice = at_hook(core, "leave" if leaving else "pause", feasible=feasible)
    if choice is None:
        return None
    event = last_event(core)
    if choice == "greet_the_nest":
        beats(core, event, setup="glance_away", setup_sounds=("greet",))
    elif choice == "vacuum_dabble":
        beats(core, event, setup="dabble")
    elif choice == "sings_to_wall":
        beats(core, event, setup="little_song", setup_sounds=("chirp", "greet"))
    elif choice == "chases_tail":
        _chase(core, event, chase["plan"])
    return choice


def _chase_plan(core: DuckCore) -> dict[str, Any] | None:
    """Round a small circle after its own tail: one tight arc (the walked turn, 6 s), only
    where the arc is clear, it may walk now, and it can still turn home afterwards."""
    from duckmind.duckcore import excursion, nest

    E = _e()
    s = core.s
    if s.excursion is None or core._motion_block("turn_around") is not None:
        return None
    if s.battery <= nest.threshold(core) + excursion.LEG_MARGIN + 200:
        return None
    side = 1 if s.quirks.get("tilt_side") == "左" else -1
    plan = core._make_plan(
        "turn_around", E.ARC_VX_MM, side * E.TURN_VYAW_MRAD, E.WALK_S_MAX, "explore"
    )
    plan["moment"] = "chases_tail"
    end, why = core._along(plan, int(plan["seconds"]) * E.PATH_STEPS_PER_S, check=True)
    if why is not None or not core._pose_turns_home(end):
        return None
    return plan


def _chase(core: DuckCore, parent: str, plan: dict[str, Any]) -> None:
    s = core.s
    exc = s.excursion
    assert exc is not None
    core._express("glance_away", reason="moment", parents=(parent,), force_sound=())
    core._walk_leg(plan, reason="excursion", outing_id=str(exc["id"]))
    exc["pause"] = True  # a look round after the chase, then on
    s.leg_pause_s = 0
    beats(core, parent)


def home_start(core: DuckCore, reason: str, from_mm: int, parent: str) -> None:
    """Setting off home: a clever guess about the way (straight home after a roundabout
    trip, or straight to bed of its own accord), or confidently the wrong way first."""
    from duckmind.duckcore import nest

    s = core.s
    outing = s.last_outing or {}
    long_way = (
        outing.get("at") == s.now and from_mm >= 600 and int(outing.get("cm", 0)) * 8 >= from_mm
    )
    bedtime = reason in ("bedtime", "tired") and s.battery >= nest.threshold(core) + 1000
    if reason not in ("home_base", "leash", "solo_done") and not bedtime:
        return  # hurrying home (battery, the family gone out, tired): no moments
    wrong: dict[str, Any] = {}

    def feasible(choice: str) -> bool:
        if choice != "wrong_turn":
            return True
        wrong["plan"] = _wrong_turn_plan(core)
        return wrong["plan"] is not None

    choice = at_hook(
        core,
        "home_start",
        feasible=feasible,
        parents=(parent,),
        long_way=long_way,
        bedtime=bedtime,
        expect={"path_cm": int(outing.get("cm", 0)) * 8 // 10} if long_way else None,
    )
    if choice == "wrong_turn":
        _wrong_turn(core, last_event(core), wrong["plan"])


def _wrong_turn_plan(core: DuckCore) -> dict[str, Any] | None:
    """A few confident steps the way it faces, when that is away from the nest and there
    is room to turn back after."""
    E = _e()
    s = core.s
    if s.homing is None or s.posture != "stand":
        return None
    x, y = s.pos_x, s.pos_y
    dist = isqrt(x * x + y * y)
    if dist == 0:
        return None
    if x * cos_mrad(s.heading) + y * sin_mrad(s.heading) < 0:
        return None  # the nest is ahead of it: walking on would not be the wrong way
    plan = core._make_plan("wander", E.ARC_VX_MM, 0, E.MIN_WANDER_S, "home")
    plan["moment"] = "wrong_turn"
    end, why = core._along(plan, int(plan["seconds"]) * E.PATH_STEPS_PER_S, check=True)
    if why is not None or not core._pose_turns_home(end):
        return None  # only where there is room to turn back after
    return plan


def _wrong_turn(core: DuckCore, parent: str, plan: dict[str, Any]) -> None:
    """…then 嗯？ (the beat plays when that leg ends, before it turns for home)."""
    s = core.s
    h = s.homing
    assert h is not None
    low, high = content.NEST["home_pause_s"]
    s.leg_pause_s = int(low) + rng.pick_index(core._rand("nest.pause"), int(high) - int(low) + 1)
    h["beats"] = parent
    core._walk_leg(plan, reason="home")


def home_pause(core: DuckCore) -> bool:
    """Between two legs home: a displacement preen (it forgot where it was going), or the
    beat after a wrong turn. True when it paused for it (the next leg comes after)."""
    s = core.s
    h = s.homing
    if h is None:
        return False
    parent = h.pop("beats", None)
    if parent is not None:
        beats(core, str(parent), closer=False)
        return False
    if h.get("reason") in URGENT_HOME or int(h.get("legs", 0)) < 1:
        return False
    if at_hook(core, "home_pause") != "displacement_preen":
        return False
    beats(core, last_event(core), setup="preen", closer=False)
    s.stand_idle_from = s.now
    s.leg_pause_s = 3
    return True


def dock_leg(core: DuckCore, from_mm: int) -> None:
    """About to walk in: a guess that it will sit down onto the contacts at the first try."""
    at_hook(core, "dock_leg", from_mm=from_mm, expect={"tries": 1})


def dock(core: DuckCore) -> None:
    """About to sit down in the nest: now and then a look left and right first."""
    if at_hook(core, "dock") == "nest_circle":
        event = last_event(core)
        core._express("look_around", reason="moment", parents=(event,), force_sound=())
        core._express("still", reason="moment", parents=(event,), force_sound=(), tempo=SCALE)


def dock_result(core: DuckCore, ok: bool, tries: int, attempt_event: Event) -> None:
    """The contacts said whether it sat down onto them."""
    if attempt(core, "neat_dock") is not None:
        resolve(
            core,
            "neat_dock",
            ok and tries == 1,
            detail={"tries": tries},
            parents=(attempt_event.event_id,),
        )
        return
    if not ok and tries == 1 and at_hook(core, "dock_miss", parents=(attempt_event.event_id,)):
        beats(core, last_event(core), reaction=True)


def docked(core: DuckCore, by: str, tries: int, walked_cm: int, event: Event) -> None:
    """Home in the nest: the guesses about the way home come out."""
    if attempt(core, "early_bed") is not None:
        resolve(
            core,
            "early_bed",
            by == "self" and tries == 1,
            detail={"tries": tries},
            parents=(event.event_id,),
        )
    state = attempt(core, "shortcut_home")
    if state is not None:
        expect = int((state["data"].get("expect") or {}).get("path_cm", 0))
        resolve(
            core,
            "shortcut_home",
            by == "self" and walked_cm <= expect,
            detail={"walked_cm": walked_cm, "expect_cm": expect},
            parents=(event.event_id,),
        )


def homing_off(core: DuckCore) -> None:
    """The way home was cut short (lifted, stranded): its guesses about it are off."""
    for key in ("early_bed", "shortcut_home", "neat_dock"):
        if attempt(core, key) is not None:
            resolve(core, key, False, detail={"why": "cut_short"})


def set_off(core: DuckCore, reason: str, first_today: bool, play_spot: int | None) -> str | None:
    """Just stood up for an outing: a false start (and back down), or a clever plan (its
    patrol of favourite places, or yesterday's play spot first)."""
    choice = at_hook(
        core,
        "set_off",
        reason=reason,
        play_spot=play_spot if first_today else None,
        expect=play_spot,
    )
    if choice == "false_start":
        s = core.s
        event = last_event(core)
        beats(core, event, setup="look_around", closer=False)
        if s.posture == "stand":
            core._perform("sit_down", reason="moment", parents=(event,), force_sound=())
        core._express("preen", reason="moment", parents=(event,), force_sound=())
    return choice


def excursion_end(core: DuckCore, exc: dict[str, Any]) -> None:
    """An outing ended: did its patrol find its favourite places, and did it get to
    yesterday's play spot?"""
    if attempt(core, "patrol") is not None:
        good = len(exc.get("good_seen", []))
        resolve(core, "patrol", good >= 2, detail={"favourites": good})
    if attempt(core, "remembers_play_spot") is not None:
        resolve(core, "remembers_play_spot", False, detail={"why": "not_there"})


def arrived(core: DuckCore, cell: int) -> None:
    state = attempt(core, "remembers_play_spot")
    if state is not None and state["data"].get("play_spot") == cell:
        resolve(core, "remembers_play_spot", True, detail={"cell": cell})


def zoomies(core: DuckCore) -> bool:
    """A happy burst (played or reunited, bright and excited): the zoomies, if the moment
    says so (goofy, relaxed)."""
    return at_hook(core, "zoomies") == "zoomies"


def zoomies_end(core: DuckCore) -> None:
    """…and then it plops down as if nothing happened."""
    s = core.s
    last = s.last_moment
    parent = str(last["event"]) if last and last.get("id") == "zoomies" else last_event(core)
    if s.posture == "stand":
        core._perform("sit_down", reason="moment", parents=(parent,), force_sound=())
    beats(core, parent, reaction=False)


def righted(core: DuckCore, parent: str) -> None:
    """Rolled back up after a fall: preening as if it meant to do that."""
    if at_hook(core, "righted", parents=(parent,)) == "i_meant_to_do_that":
        beats(core, last_event(core), reaction=False)


def name_called(core: DuckCore, parent: str) -> bool:
    """Its name (it knows it): a guess that someone wants it, where they usually are, and
    off it goes to see. True when it went (no turning on the spot for the voice then)."""
    from duckmind.duckcore import excursion

    s = core.s
    spot = excursion.family_spot(core) if core.hw.nest else None
    choice = at_hook(core, "name", parents=(parent,), expect={"spot": list(spot) if spot else None})
    if choice != "comes_when_called":
        return False
    predict = last_event(core)
    if core.hw.nest and excursion.can_start(core) is None:
        return excursion.set_out(core, "social", parents=(predict,))
    core._express(
        "perk_tilt", reason="moment", parents=(predict,), force_sound=_quiet(core, "greet")
    )
    del s
    return False


def idle(core: DuckCore) -> None:
    """An ordinary awake minute: in the evening a clever trip to where the family usually
    sits; or a goofy stare at nothing (the TV gets greeted at the cue)."""
    from duckmind.duckcore import daycycle, excursion

    s = core.s
    if s.excursion is not None or s.homing is not None or s.posture not in ("sit", "stand"):
        return
    minute_of_day = daycycle.minute_of_day(s.now)
    evening = (
        core.hw.nest
        and 1020 <= minute_of_day < int(content.RHYTHM["wind_down_min"])
        and s.family == "awake"
        and daycycle.someone_here(core)
        and excursion.can_start(core) is None
    )
    spot = excursion.family_spot(core) if evening else None
    choice = at_hook(core, "idle", evening=evening, expect={"spot": list(spot) if spot else None})
    if choice == "family_spot":
        excursion.set_out(core, "social", parents=(last_event(core),))
    elif choice == "stare_at_nothing":
        event = last_event(core)
        core._express("freeze", reason="moment", parents=(event,), force_sound=(), tempo=2300)
        beats(core, event)


def presence(core: DuckCore, by: str, evidence: Event | None) -> None:
    """Someone paid it attention: the guesses that someone would come (its name, the
    family's spot, a sound) came true; and a moment just now was seen."""
    parents = (evidence.event_id,) if evidence is not None else ()
    for key in ("comes_when_called", "family_spot", "quick_orient"):
        state = attempt(core, key)
        if state is not None and core.s.now > state["since"]:
            resolve(core, key, True, detail={"by": by or "someone"}, parents=parents)
    seen(core, by, evidence)


def cue(core: DuckCore, bases: set[str], room: bool, event: Event) -> None:
    """A logged cue (env.cue): someone came into the room (a guess comes true); a sound
    from elsewhere in the house (a guess that someone is coming); the TV (goofy)."""
    from duckmind.duckcore import daycycle

    s = core.s
    if (room or "person" in bases) and attempt(core, "quick_orient") is not None:
        resolve(core, "quick_orient", True, detail={"by": "cue"}, parents=(event.event_id,))
        return
    if not s.awake or core._until(s.napping_until) or s.family != "awake":
        return
    if bases & {"voice", "steps"} and not room and "person" not in bases:
        if at_hook(core, "cue", parents=(event.event_id,), expect="someone") == "quick_orient":
            core._express("perk_tilt", reason="moment", parents=(last_event(core),), force_sound=())
        return
    if "door" in bases and s.phase == "family":
        model = s.depart_model.get(daycycle.daytype(s.now.date()))
        due = (
            model is not None
            and int(model[1]) >= int(content.RHYTHM["anticipate_min_samples"])
            and abs(daycycle.minute_of_day(s.now) - int(model[0])) <= 30
            and daycycle.minute_of_day(s.now) < 720
        )
        if (
            at_hook(core, "door", parents=(event.event_id,), depart_due=due, expect="away")
            == "sees_you_off"
        ):
            core._express(
                "perk_tilt",
                reason="moment",
                parents=(last_event(core),),
                force_sound=_quiet(core, "chirp"),
            )
    elif s.tv_on and "tv_voice" in bases:
        if at_hook(core, "tv", parents=(event.event_id,)) == "tv_greeting":
            beats(core, last_event(core), setup="perk_tilt", setup_sounds=("greet",))


def family(core: DuckCore, to: str, event: Event) -> None:
    """The family's belief changed: they went out as it guessed (it saw them off)."""
    if to in ("away", "unknown") and attempt(core, "sees_you_off") is not None:
        resolve(core, "sees_you_off", to == "away", detail={"to": to}, parents=(event.event_id,))


def anticipation(core: DuckCore, hit: bool, predict: str, outcome: Event) -> None:
    """Waiting for the family at the door (daycycle's own guess): home when it guessed
    (clever), or it dozed off waiting (goofy, when relaxed)."""
    s = core.s
    last = s.moment_last.get("waits_for_return")
    if last is not None and s.now - last < timedelta(minutes=int(M["same_id_min"])):
        return
    s.moment_attempts["waits_for_return"] = {
        "predict": predict,
        "since": s.now,
        "until": None,
        "hook": "anticipate",
        "info": {"wits": wits(core), "form": s.form, "audience": _audience(core)},
        "data": {},
    }
    resolve(core, "waits_for_return", hit, outcome=outcome)


def hand(core: DuckCore, event: Event, since_last_s: int | None) -> None:
    """A hand again (ToF): the guess where it shows up next came true, or a new guess."""
    if attempt(core, "peekaboo_ahead") is not None:
        resolve(
            core,
            "peekaboo_ahead",
            True,
            detail={"after_s": since_last_s},
            parents=(event.event_id,),
        )
        return
    if since_last_s is not None and since_last_s <= 30:
        if at_hook(core, "hand", parents=(event.event_id,), expect="hand") == "peekaboo_ahead":
            core._express(
                "glance_away", reason="moment", parents=(last_event(core),), force_sound=()
            )


def detour(core: DuckCore, walk: Event) -> None:
    """A leg that steers round the spot it fell at: a guess that it gets past safely."""
    at_hook(core, "detour", parents=(walk.event_id,), expect={"walk": walk.event_id})


def leg_done(core: DuckCore, walk_id: str, fell: bool) -> None:
    state = attempt(core, "smart_detour")
    if state is None:
        return
    expect = state["data"].get("expect") or {}
    if expect.get("walk") != walk_id:
        return
    resolve(core, "smart_detour", not fell, detail={"fell": fell})


def new_day(core: DuckCore) -> None:
    """Midnight: today's counts start afresh."""
    core.s.moments_today = {}


def title(core: DuckCore) -> str | None:
    """「迷糊的月光歌手」: its epithet and adult name (None before it grows up)."""
    s = core.s
    if s.adult_name is None:
        return None
    word = content.EPITHETS.get(s.epithet or "", "")
    return f"{word}{s.adult_name}"


def summary(core: DuckCore) -> dict[str, Any]:
    """For the page: today's moments, its wits and form, its gags and album."""
    s = core.s
    form = day_form(core, log=False)
    return {
        "today": dict(sorted(s.moments_today.items())),
        "wits": wits(core, log=False) * 100 // SCALE if s.awake else None,
        "form": form,
        "form_zh": FORM_ZH.get(form, ""),
        "gags": [BY_ID[key].short for key in s.gags],
        "album": len(s.moment_album),
        "title": title(core),
    }
