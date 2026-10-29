"""Growth G4: the 「成長」 view of a duck (docs/design/GROWTH-G4-2026-10-11.md).

One read-only picture of who a duck has become, as plain data for the 資料 page: the name, hatch
day, growth day, stage and branch; the temperament and quirks; what it learnt (what it likes, the
places it keeps away from); who it is close to; what it lived (the experience bridge's tally and
what was left out as not real); the nightly regrows; the voice lineage; and a short diary.

The same builder shows the duck the server lives (:class:`~duckmind.simweb.life_host.LifeHost`) and a
throwaway test duck of the fast-forward (:mod:`duckmind.life.fastforward`): both are a Duck Core
session, maybe a neural learner, and the numbers the caller has for the rest. Nothing here changes
anything: Duck Core is read (its memory is distilled from the log the way the Duck Core page does),
the learner is only looked at.

Words: Duck Core's own words (stage, branch, habits, calls, verbs) are catalog messages
(:func:`duckmind.i18n.content_msg`), so the page shows them in the viewer's language; the engine's
diary lines are plain 繁體中文 that the API boundary matches back to the engine's templates
(``i18nwire.growth_answer``). Names the family gave (the duck's, people's) are data. The page puts
its own words around all of it from its catalog.

Privacy: people's names are returned (the owner asked for them: 「names as data」); their looks (the
face vectors the learner keeps) never are.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from duckmind.duckcore import content, narrate, query
from duckmind.duckcore.fixed import SCALE
from duckmind.i18n import content_msg, lst, msg

API = "growth-view-v1"
TPE = timezone(timedelta(hours=8))

AXES = tuple(content.TEMPERAMENT["axes"])
AXIS_LOW, AXIS_HIGH = (int(v) for v in content.TEMPERAMENT["range"])
# the bridge's kinds (duckmind/duckcore/bridge.py::KINDS, frozen with bridge-v1, then bridge-v2's
# trick lessons, duckmind/duckcore/bridge_v2.py::NEW_KINDS), in the order the page lists them
KIND_ORDER = (
    "touch",
    "poke",
    "lift",
    "hand",
    "name",
    "voice",
    "hello",
    "noise",
    "company",
    "visit",
    "play",
    "fall",
    "sleep",
    "dock",
    "feed",
    "voiced",
    "explore",
    "teach_trick",
    "praise",
    "scold",
)
RECENT_EXPERIENCES = 12
DIARY_DAYS = 7
DIARY_LINES = 6
PREFS_SHOWN = 8
PEOPLE_SHOWN = 12
HISTORY_SHOWN = 8
# the neutral value of a preference (Duck Core: SCALE // 2)
NEUTRAL = SCALE // 2


# ── what the neural duck lived, counted from Duck Core's inputs ───────────────────────────────


class Tally:
    """The experiences Duck Core was given (its ``experience`` inputs, ``bridge-v1``): how many of
    each kind, how many a day, and the latest few. It reads the inputs once, from where it stopped
    (a life is thousands of them), so asking for it every few seconds costs nothing."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.seen = 0
        self.by_kind: Counter[str] = Counter()
        self.by_day: dict[str, Counter[str]] = {}
        self.recent: deque[dict[str, Any]] = deque(maxlen=RECENT_EXPERIENCES)
        self.first_t: int | None = None

    def update(self, inputs: Iterable[Any]) -> None:
        items = inputs if isinstance(inputs, (list, tuple)) else list(inputs)
        if len(items) < self.seen:  # (another session: it was reset or hatched anew)
            self.reset()
        for method, params in items[self.seen :]:
            if method != "experience" or not isinstance(params, Mapping):
                continue
            for event in params.get("events") or ():
                if not isinstance(event, Mapping):
                    continue
                kind = str(event.get("k", ""))
                t = event.get("t")
                if kind not in KIND_ORDER or type(t) is not int:
                    continue
                self.by_kind[kind] += 1
                day = datetime.fromtimestamp(t, TPE).strftime("%Y-%m-%d")
                self.by_day.setdefault(day, Counter())[kind] += 1
                if self.first_t is None or t < self.first_t:
                    self.first_t = t
                self.recent.append(_recent(kind, t, event))
        self.seen = len(items)


def _recent(kind: str, t: int, event: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": kind, "at": datetime.fromtimestamp(t, TPE).isoformat()}
    if kind in ("play", "sleep", "dock") and isinstance(event.get("minutes"), int):
        out["minutes"] = int(event["minutes"])
    if kind == "fall" and isinstance(event.get("outcome"), str):
        out["outcome"] = str(event["outcome"])
    if kind in ("teach_trick", "praise", "scold") and isinstance(event.get("trick"), str):
        out["trick"] = str(event["trick"])
    if kind == "teach_trick" and isinstance(event.get("outcome"), str):
        out["outcome"] = str(event["outcome"])
    return out


# ── the pieces ────────────────────────────────────────────────────────────────────────────────


def _pct(value: int) -> int:
    return int(value) * 100 // SCALE


def _stage_msg(stage_id: str) -> str:
    zh = next((s.zh for s in content.STAGES if s.stage_id == stage_id), stage_id)
    return content_msg(f"core.content.STAGES.{stage_id}_zh", zh)


def _branch_msg(branch_id: str | None) -> str | None:
    if not branch_id or branch_id not in content.BRANCH_BY_ID:
        return None
    return content_msg(f"core.content.BRANCHES.{branch_id}_zh", content.BRANCH_BY_ID[branch_id].zh)


def _tag_msg(tag: str) -> str:
    return content_msg(f"core.content.TAG_ZH.{tag}", content.TAG_ZH.get(tag, tag))


def _identity(
    core: Any,
    *,
    name: str,
    branch_id: str | None,
    pinned: bool,
    nature_branch: str | None,
    title_key: str | None,
) -> dict[str, Any]:
    s = core.s
    stage = core.stage()
    gd = int(s.gd)
    days = {st.stage_id: core._stage_day(st.stage_id) for st in content.STAGES}
    nxt = next(
        (
            {"id": st.stage_id, "label": _stage_msg(st.stage_id), "gd": days[st.stage_id]}
            for st in content.STAGES
            if days[st.stage_id] > gd
        ),
        None,
    )
    if nxt is not None:
        nxt["in_days"] = int(nxt["gd"]) - gd
    hatched = s.hatched_at
    age = max(0, (s.now - hatched).days)
    return {
        "name": name,
        "seed": core.seed,
        "hatched": hatched.isoformat(),
        "age_days": age,
        "growth_day": gd,
        "half_day": bool(s.gd2 % 2 == 1),
        "stage": {"id": stage.stage_id, "label": _stage_msg(stage.stage_id)},
        "next_stage": nxt,
        "stages": [
            {"id": st.stage_id, "label": _stage_msg(st.stage_id), "gd": days[st.stage_id]}
            for st in content.STAGES
        ],
        "branch": {
            "id": branch_id,
            "label": _branch_msg(branch_id),
            # who chose it: the family pinned it (蛋黃), the growth decided it (the care it got and
            # its nature), or it has not grown into one yet
            "source": "family" if pinned and branch_id else ("growth" if branch_id else "none"),
            "core_id": s.branch,
            "core_label": _branch_msg(s.branch),
            "nature_id": nature_branch,
        },
        "title": (msg(title_key) if title_key else None),
    }


def _temperament(core: Any) -> dict[str, Any]:
    out = []
    for axis in AXES:
        value = int(core.s.temperament.get(axis, 0))
        pct = max(0, min(100, (value - AXIS_LOW) * 100 // max(1, AXIS_HIGH - AXIS_LOW)))
        side = "high" if pct >= 50 else "low"
        out.append(
            {
                "axis": axis,
                "pct": pct,
                "name": msg(f"neuro.axis.{axis}.name"),
                "word": msg(f"neuro.axis.{axis}.{side}"),
            }
        )
    leaning = sorted(core.s.nature.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "axes": out,
        "nature": [
            {"id": branch_id, "label": _branch_msg(branch_id), "pct": _pct(int(v))}
            for branch_id, v in leaning
        ],
    }


def _quirks(core: Any) -> dict[str, Any]:
    s = core.s
    quirks: list[dict[str, Any]] = []
    if "tilt_side" in s.revealed:
        side = "left" if s.quirks.get("tilt_side") == "左" else "right"
        quirks.append({"key": "tilt_side", "side": msg(f"neuro.side.{side}_short")})
    if "favorite_tag" in s.revealed:
        quirks.append({"key": "favorite_tag", "tag": _tag_msg(str(s.quirks["favorite_tag"]))})
    if "habit" in s.revealed:
        habit = str(s.quirks.get("habit", ""))
        zh = dict(content.QUIRK_HABITS).get(habit, str(s.quirks.get("habit_text", habit)))
        quirks.append(
            {"key": "habit", "label": content_msg(f"core.content.QUIRK_HABITS.{habit}", zh)}
        )
    shown = {q["key"] for q in quirks}
    return {"items": quirks, "hidden": len({"tilt_side", "favorite_tag", "habit"} - shown)}


def _pref_item(key: str, phrases: Mapping[str, Any]) -> dict[str, Any] | None:
    """A learnt preference (Duck Core: ``household.verb:look_around``) as what it is a liking for."""
    item = key.split(".", 1)[1] if "." in key else key
    kind, _, ident = item.partition(":")
    if kind == "verb":
        verb = content.VERB_BY_ID.get(ident)
        if verb is None:
            return None
        return {"kind": "verb", "label": content_msg(f"core.content.VERBS.{ident}_zh", verb.zh)}
    if kind == "moment":
        moment = content.MOMENT_BY_ID.get(ident)
        if moment is None:
            return None
        return {
            "kind": "moment",
            "label": content_msg(f"core.content.MOMENT_SHORT.{ident}", moment.short),
        }
    if kind == "phrase":
        phrase = phrases.get(ident, {})
        return {"kind": "phrase", "label": str(phrase.get("text", ident))}
    if kind == "trick":
        words = narrate.TRICK_ZH.get(ident)
        if words is None:
            return None
        return {"kind": "trick", "label": content_msg(f"core.narrate.TRICK_ZH.{ident}", words)}
    return None


def _home(home: Mapping[str, Any] | None, places_found: int) -> dict[str, Any] | None:
    """(MAP-EXPLORE-2026-10-12) How much of its home the duck has come to know, from its own lasting map only (never the true
    layout): the floor it has seen clear (m²), the 1 m squares of it, a guess of the share of the floor it knows of, and how
    many new squares were told to Duck Core as finds this watch. None: no map (the map is off) or nothing worked out yet."""
    if not home:
        return None
    share = home.get("share")
    return {
        "known_m2": round(float(home.get("known_m2") or 0.0), 1),
        "places": int(home.get("places") or 0),
        "share_pct": None if share is None else int(round(float(share) * 100)),
        "found": int(places_found),
    }


def _learned(
    core: Any, learner: Any, home: Mapping[str, Any] | None = None, places_found: int = 0
) -> dict[str, Any]:
    s = core.s
    prefs: list[dict[str, Any]] = []
    for key, value in sorted(s.prefs.items(), key=lambda kv: (-kv[1], kv[0])):
        item = _pref_item(key, s.phrases)
        if item is None:
            continue
        prefs.append(
            {
                **item,
                "pct": _pct(int(value)),
                "liked": int(value) > NEUTRAL,
                "disliked": int(value) < NEUTRAL,
            }
        )
    spots = [
        {
            "x": round(sp["x"] / 1000, 2),
            "y": round(sp["y"] / 1000, 2),
            "r": round(sp["r"] / 1000, 2),
        }
        for sp in core.active_spots()
    ]
    out: dict[str, Any] = {
        "prefs": prefs[:PREFS_SHOWN],
        "prefs_total": len(prefs),
        "likes": sum(1 for p in prefs if p["liked"]),
        "dislikes": sum(1 for p in prefs if p["disliked"]),
        "avoid_spots": spots,
        "falls": int(s.falls),
        "touches": int(s.touches),
        "pickups": int(s.pickups),
        "name_calls": int(s.name_calls),
        "slow_hands": int(s.slow_hands),
        "walked_m": round(s.walked_cm / 100, 1),
        "mind": None,
        "home": _home(home, places_found),
        "tricks": _tricks(core, learner),
    }
    mind = _mind(learner)
    if mind is not None:
        out["mind"] = mind
    return out


def _tricks(core: Any, learner: Any) -> list[dict[str, Any]] | None:
    """The neural duck's trick book (its own proficiencies, ``learner.tricks``), one row per trick:
    how far it has learnt it (0–100), the counted tries today and ever, how many of them it did by
    itself (shaping), and whether the trick is open to it yet (``flip`` from the juvenile stage).
    Read from a copy: the book is only looked at (the brain's thread keeps it). None: no learner."""
    book = getattr(learner, "tricks", None)
    if book is None:
        return None
    from duckmind.neuro.tricks import TrickBook

    copy = None
    for _ in range(3):
        try:
            copy = TrickBook.from_state(book.to_state())
            break
        except RuntimeError:  # (the brain changed the table under us: look again)
            continue
    if copy is None:
        return None
    s = core.s
    locked = ("flip",) if int(s.gd) < core._stage_day("juvenile") else ()
    return [
        {
            "id": row["id"],
            "label": content_msg(f"core.narrate.TRICK_ZH.{row['id']}", narrate.TRICK_ZH[row["id"]]),
            "proficiency_pct": row["proficiency_pct"],
            "state": row["state"],
            "practised_today": row["tries_today"],
            "capped_today": row["capped_today"],
            "total": row["total"],
            "shaped": row["shaped"],
            "unlocked": row["unlocked"],
            "suggest": row["suggest"],
        }
        for row in copy.view(int(s.gd), locked)
    ]


def _mind(learner: Any) -> dict[str, Any] | None:
    """What the neural duck's own learner holds, as counts: places it knows, likes and keeps
    away from, the things it knows. (Looked at without a lock: tried again if the brain changes
    the table under us.)"""
    if learner is None:
        return None
    for _ in range(3):
        try:
            places = list(learner.places.values())
            things = list(learner.things.values())
            return {
                "places": len(places),
                "places_liked": sum(1 for p in places if p.value > 0.1),
                "places_disliked": sum(1 for p in places if p.value < -0.1),
                "places_avoided": sum(1 for p in places if p.hard_falls >= 3),
                "things": len(things),
                "things_liked": sum(1 for t in things if t.value > 0.1),
                "things_disliked": sum(1 for t in things if t.value < -0.1),
                "people": len(learner.people),
            }
        except RuntimeError:
            continue
    return None


def _bonds(core: Any, learner: Any) -> dict[str, Any]:
    people: list[dict[str, Any]] = []
    unnamed = 0
    if learner is not None:
        for _ in range(3):
            try:
                records = sorted(learner.people.values(), key=lambda p: (-p.bond, p.pid))
                break
            except RuntimeError:
                records = []
        else:
            records = []
        for person in records:
            if not person.name:
                unnamed += 1
                continue
            if len(people) < PEOPLE_SHOWN:
                people.append(
                    {
                        "name": str(person.name),
                        "pct": max(0, min(100, int(round(person.bond * 100)))),
                        "greeted": int(person.greeted),
                        "exchanges": int(person.exchanges),
                        "sightings": int(person.sightings),
                    }
                )
    return {
        # how close it is to the family (Duck Core's bond, 2000 at hatching up to 10000)
        "family_pct": _pct(int(core.s.bond_owner)),
        "family_raw": int(core.s.bond_owner),
        "people": people,
        "unnamed": unnamed,
    }


def _experiences(
    tally: Tally, ignored: Mapping[str, int] | None, delivered: Mapping[str, Any]
) -> dict[str, Any]:
    by_kind = {k: int(tally.by_kind.get(k, 0)) for k in KIND_ORDER if tally.by_kind.get(k, 0)}
    return {
        "total": int(sum(tally.by_kind.values())),
        "by_kind": by_kind,
        "recent": list(tally.recent)[::-1],
        # felt but not a hand (a thing, a person's leg): never told to Duck Core, counted since the
        # server started
        "ignored": {str(k): int(v) for k, v in sorted((ignored or {}).items())},
        "refused": int(delivered.get("refused", 0) or 0),
    }


def _voice(core: Any, live: Mapping[str, Any] | None) -> dict[str, Any]:
    from duckmind.life import voice_lineage as vl

    view = vl.view_of(core.s)
    phrases = {p.code: p for p in view.phrases}
    calls = []
    active = (live or {}).get("active") or {}
    by_clip = {c.get("clip"): c for c in (active.get("calls_learned") or [])}
    for code, how in vl.learned(view):
        phrase = phrases.get(code)
        tags = (
            list(phrase.tags)
            if phrase is not None
            else list(content.WORDS.get(view.first_word or "", ()))
        )
        clip = f"ph_{code}"
        known = by_clip.get(clip) or {}
        calls.append(
            {
                "clip": clip,
                "how": how,
                "words": lst([_tag_msg(t) for t in tags], sep="fmt.sep.babble"),
                "praise": int(phrase.praise) if phrase is not None else 0,
                "ms": known.get("ms"),
                "playable": clip in by_clip,
            }
        )
    first_word = None
    if view.first_word is not None and view.first_word in content.WORDS:
        first_word = lst(
            [_tag_msg(t) for t in content.WORDS[view.first_word]], sep="fmt.sep.babble"
        )
    song = None
    if view.song:
        parts = [
            lst([_tag_msg(t) for t in phrases[c].tags], sep="fmt.sep.babble")
            for c in view.song
            if c in phrases
        ]
        song = {
            "version": int(view.song_version),
            "words": lst(parts, sep="fmt.sep.slash"),
            "clip": "song",
            "playable": bool(active.get("song")),
        }
    return {
        "lineage_size": view.size,
        "first_word": first_word,
        "calls": calls,
        "song": song,
        "favorite_tag": _tag_msg(view.favorite_tag) if view.favorite_tag else None,
        "state": (live or {}).get("state"),
        "lib": active.get("key8"),
        "age": active.get("age"),
    }


def _regrow(status: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if status is None:
        return None
    history = [dict(h) for h in (status.get("history") or [])][-HISTORY_SHOWN:]
    return {
        "count": int(status.get("count") or 0),
        "growth_day": status.get("growth_day"),
        "applied_gd": status.get("applied_gd"),
        "due": bool(status.get("due")),
        "waiting_for": status.get("waiting_for"),
        "problem": status.get("problem"),
        "last": status.get("last"),
        "history": history[::-1],
        "genome": status.get("genome"),
    }


def _milestones(core: Any) -> list[dict[str, Any]]:
    out = []
    for mid, info in sorted(
        core.s.milestones.items(),
        key=lambda item: (
            item[1].get("debuted_gd", item[1].get("unlocked_gd", item[1]["gd"])),
            item[1]["order"],
            item[0],
        ),
    ):
        if info["status"] != "debuted":
            continue
        out.append(
            {
                "id": mid,
                "gd": info.get("debuted_gd", info.get("unlocked_gd")),
                "title": content_msg(f"core.content.CALENDAR.{mid}_title", str(info["title"])),
            }
        )
    return out


def _diary(core: Any, tally: Tally) -> list[dict[str, Any]]:
    """The last few days: Duck Core's own diary lines (memory, moments), and the day's tally of
    what the neural duck lived. Days with neither are left out."""
    days = query.diary(core, DIARY_DAYS * 2)
    out: list[dict[str, Any]] = []
    seen_dates = set()
    for day in days:
        lived = tally.by_day.get(day["date"], Counter())
        lines = [
            {"text": line["text"], "tone": line["tone"]} for line in day["lines"][:DIARY_LINES]
        ]
        more = max(0, len(day["lines"]) - DIARY_LINES)
        seen_dates.add(day["date"])
        out.append(
            {
                "date": day["date"],
                "gd": day["gd"],
                "positive": day["positive"],
                "negative": day["negative"],
                "lines": lines,
                "more": more,
                "lived": {k: int(lived[k]) for k in KIND_ORDER if lived.get(k)},
            }
        )
    for date in sorted(tally.by_day, reverse=True):
        if date in seen_dates:
            continue
        lived = tally.by_day[date]
        out.append(
            {
                "date": date,
                "gd": None,
                "positive": 0,
                "negative": 0,
                "lines": [],
                "more": 0,
                "lived": {k: int(lived[k]) for k in KIND_ORDER if lived.get(k)},
            }
        )
    out.sort(key=lambda d: d["date"], reverse=True)
    return out[:DIARY_DAYS]


# ── the whole view ────────────────────────────────────────────────────────────────────────────


def build(
    core: Any,
    *,
    tally: Tally,
    name: str | None = None,
    live: bool = True,
    branch_id: str | None = None,
    pinned: bool = False,
    nature_branch: str | None = None,
    learner: Any = None,
    home: Mapping[str, Any] | None = None,
    places_found: int = 0,
    regrow: Mapping[str, Any] | None = None,
    voice: Mapping[str, Any] | None = None,
    ignored: Mapping[str, int] | None = None,
    delivered: Mapping[str, Any] | None = None,
    calendar: str | None = None,
) -> dict[str, Any]:
    """The growth view of a duck whose Duck Core is ``core``. ``branch_id`` is the branch its body
    follows (Duck Core's own unless the record pins another: ``pinned``). ``live`` tells the page
    whether this is the duck the server lives or a test duck."""
    s = core.s
    title_key = None
    epithet = content.EPITHETS.get(s.epithet or "") if getattr(s, "epithet", None) else None
    if epithet:
        title_key = f"core.content.EPITHETS.{s.epithet}"
    identity = _identity(
        core,
        name=s.name if name is None else name,
        branch_id=branch_id,
        pinned=pinned,
        nature_branch=nature_branch,
        title_key=title_key,
    )
    return {
        "api": API,
        "live": live,
        "calendar": calendar or s.now.isoformat(),
        "identity": identity,
        "temperament": _temperament(core),
        "quirks": _quirks(core),
        "learned": _learned(core, learner, home, places_found),
        "bonds": _bonds(core, learner),
        "experiences": _experiences(tally, ignored, delivered or {}),
        "regrow": _regrow(regrow),
        "voice": _voice(core, voice),
        "milestones": _milestones(core),
        "diary": _diary(core, tally),
    }
