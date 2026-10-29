"""A dependency-free local HTTP server for the Duck Core simulator.

It serves the static page and a few JSON endpoints. The duck runs in this Python process:
the page never re-implements Duck Core logic, it only renders what ``duckmind.duckcore``
decides. Binds to 127.0.0.1 by default; pass ``--host`` to reach it over a tailnet.

With live physics (``--live``), ``/api/body`` carries the physics body's frames with their
truth in the world, the world itself (``world``: the home, the hand, the family's bodies, the
things, the world battery and the nest's contact; phase 2), the stall monitor and the
duck's-eye status, and ``/api/eye.jpg`` the latest duck's-eye picture (asking for it is what
keeps the world's camera window open). ``/api/home`` is the home the page draws (the same
layout the physics runs, :mod:`duckmind.simweb.homeview`); ``/api/body?home=<fingerprint>``
carries it too (``home_json``) whenever the page's copy is another one. Every answer to
``/api/call`` carries the light at the duck's time (``server.lights``, from
:mod:`duckmind.world.lights`) and which home (``server.home``).

Phase 2 buttons: a touch pressed as a family member (with a world that has the household)
first brings that person's body over and lets their hand do it; Duck Core hears about it a few
seconds later (:mod:`duckmind.simweb.family`, owner decision D3). ``body_upright`` is
「扶起物理身體」 (D2), ``body_show`` a curiosity demonstration (示範), ``body_roll`` and
``body_tidy`` the world's own buttons; none of these involve Duck Core.

The duck's voice (:mod:`duckmind.simweb.voice`, owner 2026-09-28): ``/api/voice/manifest`` and
``/api/voice/v5/<voice seed>/<tag>/<file>.wav`` serve the current duck's own official Microduck
voice bank (rendered on demand by upstream's ``sounds`` binary from ``--microduck``), and every
feed event that made a sound carries its tags as ``calls`` for the page to play.

The neural-duck prototype (``--brain neuro``, :mod:`duckmind.simweb.neuro_api` and
:mod:`duckmind.simweb.neuro_live`): the neural duck drives the live body instead of Duck Core's
motions. Everything below that belongs to it is gated on that mode and the normal mode is
unchanged: ``/api/neuro/…`` (the page's routes), ``/api/body?since_t=`` (every body sample since
the page's last), the voice route serves the neural duck on screen, and the flags ``--brain``,
``--neuro-duck``, ``--neuro-record``, ``--neuro-tof``, ``--neuro-start``, ``--neuro-time`` and
``--robotd-head-alpha``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld: a v4_1 duck lives in it)
from duckmind.duckcore.api import (
    ENGINE_REV,
    Config,
    SaveBridgeMismatch,
    SaveContentMismatch,
    Session,
    _motion,
    clips,
    dispatch,
    feed,
    open_session,
    restore,
    save_json,
    snapshot,
)
from duckmind.duckcore.bridge import BRIDGE_API
from duckmind.i18n import Msg, boundary, lst, msg, packs
from duckmind.life.clock import DuckClock
from duckmind.life.record import LifeError, LifeRecord, write_private
from duckmind.live.transport import TRANSCRIPTION_LANGUAGES
from duckmind.simweb import (
    battery_help,
    homeview,
    i18nwire,
    layouts,
    neuro_api,
    robot_model,
    skill_evidence,
    voice,
)
from duckmind.simweb.live_voice_api import LiveHub
from duckmind.world import layoutcheck as _layoutcheck
from duckmind.world import lights as world_lights

if TYPE_CHECKING:
    from duckmind.duckcore.api import AnySession
    from duckmind.simweb.family import FamilyVisits
    from duckmind.simweb.live import LiveBridge

STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
}
# The page's own fonts and libraries live in these folders of static/ (no CDN, owner decision Q7);
# every other page file sits at the top.
STATIC_FOLDERS = frozenset({"fonts", "vendor"})
MAX_BODY = 256 * 1024


class SimServer(ThreadingHTTPServer):
    daemon_threads = True
    # A reverse proxy such as `tailscale serve` opens every mesh request at once; the
    # stdlib default backlog of 5 turns that burst into refused connections (502s).
    request_queue_size = 128


MARKS_KEEP = 4000  # per-event playback notes kept for the page (the newest events)
SUMMARY_MINUTES = 15  # an advance longer than this is shown as its result, not replayed
FAST_MINUTES = 2  # from here up, only the newest walk and move of a step are played
# The page's contract with this server. The page checks it and says so when the server is
# an older process (started before the code changed) instead of misbehaving. 7: body frames
# carry their truth in the room, /api/body the monitor and the eye, /api/eye.jpg, the lights.
# 8 (phase 2): the truth in world coordinates (placed_at / placed_why instead of the anchor),
# /api/body's world and home_json, /api/home, server.home, family visits, the world buttons.
API_VERSION = 8
# Real time: Duck Core lives each second as it comes; the page (polling every second) and
# the live body show every event this long after it happened, at the same moment.
SHOW_DELAY_S = 2.5
CATCH_UP_S = 15 * 60  # further behind than this (the computer slept): catch up, don't replay
SAVE_EVERY_S = 30.0  # real time saves the duck at most this often (and on every input)
# A life (--life): Duck Core lives on to the calendar. A gap longer than this (times the clock's
# speed) is time the duck was switched off (the server was down, the computer slept): it counts
# the way the family's absence does (half growth days, never full ones), not as awake time.
OFFLINE_GAP_S = 300
# The code this process runs, watched so the page can say a restart is due.
WATCHED = ("duckcore", "simweb", "livebody.py", "bodyhost.py", "world", "eye")


def lights_payload(
    moment: datetime, home: str | None = None, at_m: tuple[float, float] | None = None
) -> dict[str, Any]:
    """The light in the duck's home at its time (world.lights, the one source): the page's
    factors, its three.js lights and a plain label. With ``home``, the light of the room the
    duck is in (``at_m``, world metres) from that home's layout — what the world's camera is
    lit by (spec §11.3); phase 0's one room otherwise, or when the home cannot be read."""
    state = homeview.room_light(home, moment, at_m) if home else None
    if state is None:
        state = world_lights.light_state(moment)
    return {
        "when": state.when,
        "key": state.key,
        "label": state.label,
        "lux": round(state.lux, 3),
        "daylight": round(state.daylight, 5),
        "glow": round(state.glow, 5),
        "lamps": round(state.lamps, 5),
        "elevation": round(state.elevation, 5),
        "phase": round(state.phase, 5),
        "sunrise": round(state.sunrise, 2),
        "sunset": round(state.sunset, 2),
        "three": state.three(),
    }


def _code_mtime() -> float:
    """Newest modification time of the code this server runs (Python and page files)."""
    root = Path(__file__).resolve().parent.parent
    newest = 0.0
    for name in WATCHED:
        path = root / name
        files = [path] if path.is_file() else [*path.rglob("*.py"), *path.rglob("*.js")]
        files += list(path.rglob("*.html")) + list(path.rglob("*.css")) if path.is_dir() else []
        for item in files:
            try:
                newest = max(newest, item.stat().st_mtime)
            except OSError:
                continue
    return newest


class SimState:
    def __init__(
        self,
        config: Config,
        live: LiveBridge | None = None,
        state_file: Path | None = None,
        *,
        voice_bank: voice.VoiceBank | None = None,
        life: LifeRecord | None = None,
        clock: DuckClock | None = None,
        resume_time: bool = False,
        attach_neural: bool = False,
    ) -> None:
        self.lock = threading.Lock()
        # A life (--life): the duck's one record, its calendar clock; Duck Core lives on to the
        # calendar (grow). None: the simulator as it always was (Duck Core's clock is its own).
        self.life = life
        self.clock = clock
        self.life_host: Any = None  # life_host.LifeHost, set by main()
        # the page's fast-forward of throwaway test ducks (growth G4), made on first use
        self.fastforward: Any = None
        # The layout editor (docs/design/REAL-HOME-2026-10-xx.md): the saved layouts, and the home this
        # server started with (what 「回到預設」 goes back to); set by main()
        self.layouts = layouts.LayoutStore(None)
        self.layout_default: str = "default_home"
        self.caps_file = config.caps_file
        self.state_file = state_file
        self.realtime = False
        self._resume_realtime = False
        # Real time: (wall-clock epoch, duck time) at one moment; the duck's clock runs on
        # from there at the wall clock's pace.
        self.anchor: tuple[float, datetime] | None = None
        # How the page should show each event (by its index in the log): at a wall-clock
        # moment (real time: {"due_ms"}), or applied at once without replaying it
        # ({"settle": True}: fast-forwards, catch-ups, a routine's weeks).
        self.marks: dict[int, dict[str, Any]] = {}
        # The idle layer's glances in real time, each at its wall moment.
        self.fidgets: list[dict[str, Any]] = []
        self._fidget_ids = itertools.count(1)
        self._saved_at = 0.0
        # Which server process this is: the page notices a restart (another boot) and says
        # so, and starts following the duck afresh.
        self.boot = f"{os.getpid()}-{int(time.time() * 1000):x}"
        self.started_code = _code_mtime()
        self._stale_checked = (0.0, False)
        self.session: AnySession = self._restore() or Session(config)
        if self.life is not None:
            if resume_time and self.clock is not None:
                # (tests with a fast clock or a chosen birth: the calendar starts where the duck is)
                self.clock.jump_to(self.session.core.s.now)
            fresh = not self.session.inputs  # (hatched just now: nothing lived yet)
            if attach_neural:
                # The neural duck lives this life (docs/design/GROWTH-G2-2026-10-10.md): Duck Core
                # stops simulating a body and experiences of its own, before anything else lives
                # (a backdated hatch's first days are then days of the calendar alone) and from the
                # moment the record is opened (an older record carries on from where its shadow
                # life stopped: one `bridge` input marks the seam).
                self.attach_neural()
            # (a duck that was away lives on to today before anything else; a backdated hatch lives
            # its first days awake, at creation: they are whole growth days, not days away)
            self.grow(lived=bool(self.life.backdated_days and fresh))
        # The duck's own official voice (its bank is rendered in the background if it is new).
        self.voice = voice_bank or voice.VoiceBank.off()
        self.voice.prepare(self.duck_seed())
        self.live = live
        if live is not None:
            live.clock = self.duck_now  # the duck's-eye light follows the duck's own time
        # Family members' touches with a body that walks over first (D3; live physics only).
        self.family: FamilyVisits | None = self._family()
        self.family_done: list[dict[str, Any]] = []  # the last visits' acts, for the page
        # the family's help with the neural duck's pack (Duck Core's battery_help setting)
        self.battery_help = battery_help.BatteryHelp()
        self._family_ids = itertools.count(1)
        self._saved_inputs: tuple[int, str, bool] | int = self._save_mark()
        if self.life is not None:
            # a life's time is the calendar's, never Duck Core's own real-time switch
            self._resume_realtime = False
            # and a duck just hatched (or lived on to today) is on disk from the first moment
            self._saved_inputs = -1
            self.save()
        if self._resume_realtime:
            self.catch_up_to_wall_clock()
            self.set_realtime(True)
        self._clock = threading.Thread(target=self._run_clock, daemon=True)
        self._clock.start()
        self._visits = threading.Thread(target=self._run_family, daemon=True)
        self._visits.start()

    def _family(self) -> FamilyVisits | None:
        if self.live is None:
            return None
        from duckmind.simweb.family import FamilyVisits

        return FamilyVisits(self.live.world_port(), seed=int(self.session.config.seed))

    def home(self) -> str:
        """The home the page draws (and the live body runs): see homeview.page_home."""
        return homeview.page_home(nest=bool(self.session.core.hw.nest))

    # ── the layout editor ───────────────────────────────────────────────────

    def layout_state(self, default: bool = False) -> dict[str, Any]:
        """The home that runs, as the editor holds it (placements), what to draw, its checks, what is
        saved. ``default``: the server's default home instead (the editor's 「重設」)."""
        home = self.home()
        if default:
            from duckmind.livebody import live_home

            home = (
                live_home(self.layout_default, nest=bool(self.session.core.hw.nest))
                or self.layout_default
            )
        layout = homeview.layout_of(home)
        out = layouts.state_payload(
            layout, home=home, default_home=self.layout_default, store=self.layouts
        )
        out["live"] = self.live is not None
        out["applying"] = bool(
            self.live is not None
            and (not self.live.ready or getattr(self.live, "restoring", False))
        )
        out["restored"] = getattr(self.live, "restored", None)
        return out

    def layout_apply(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        """Run this layout in the simulator (None: the server's own default home). The physics is reloaded;
        the duck's mind is kept (the life snapshot is taken first, the same neural driver goes on) and the
        duck is put back where it was if that spot is free in the new home, else at its spawn."""
        if raw is None:
            home = self.layout_default
            problems: list[dict[str, Any]] = []
        else:
            verdict = _layoutcheck.check(raw)
            problems = verdict["problems"]
            if not verdict["ok"]:
                raise layouts.LayoutProblem("invalid", problems=problems)
            home = str(self.layouts.applied_path(raw))
        pose = None
        if self.live is not None:
            truth = self.live.truth_now()
            if truth is not None:
                pose = (truth["x_mm"] / 1000.0, truth["y_mm"] / 1000.0, truth["yaw_mrad"] / 1000.0)
        if self.life_host is not None:
            try:
                self.life_host.snapshot_now(
                    "layout"
                )  # the mind first: the reload only moves the body
            except Exception:
                traceback.print_exc()
        homeview.set_home(home)
        self.layouts.remember(None if raw is None else home)
        if self.live is not None:
            self.live.restart(home=home, restore=pose)
        return {"ok": True, "home": home, "problems": problems, "live": self.live is not None}

    def duck_at(self) -> tuple[float, float] | None:
        """Where the duck is in its home's world (m), for the room whose light the page draws:
        the physics body with live physics, else Duck Core's plan moved into the home."""
        truth_now = getattr(self.live, "truth_now", None)
        truth = truth_now() if callable(truth_now) else None
        if truth is not None:
            return truth["x_mm"] / 1000.0, truth["y_mm"] / 1000.0
        pose = self.session.core.pose_now()
        return homeview.plan_in_world(self.home(), float(pose[0]), float(pose[1]))

    def sync_clock(self) -> None:
        """The world camera's light follows the duck's own clock (paused: it stands still)."""
        if self.live is not None:
            day = getattr(getattr(self.live, "neuro", None), "day_clock", None)
            if day is not None:  # prototype mode: the neural duck's own day clock
                self.live.set_clock(day.local(), day.rate)
                return
            self.live.set_clock(self.session.core.s.now, 1.0 if self.realtime else 0.0)

    def duck_now(self) -> datetime:
        """The duck's local time (read without the lock: one attribute of the current duck)."""
        return self.session.core.s.now

    def light_time(self) -> datetime:
        """The time the room's lights follow: the duck's clock. With the neural duck that is its
        day clock (the one its circadian drive, the world camera's light and the duck's-eye light
        follow too; before this the room followed Duck Core's paused clock and showed evening
        lamps while the eye saw the neural duck's morning); otherwise Duck Core's own."""
        day = getattr(getattr(self.live, "neuro", None), "day_clock", None)
        return day.local() if day is not None else self.session.core.s.now

    def attach_neural(self) -> bool:
        """(A life only.) Attach Duck Core to the neural duck (``bridge`` input): from here it is
        given what the neural duck lives and simulates nothing of its own. True: it was attached
        just now (an older record's seam); False: already."""
        if self.life is None or not isinstance(self.session, Session):
            return False
        if self.session.core.s.bridged:
            return False
        if self.state_file is not None and self.session.inputs:
            # (an older record: its file as it is now is kept beside it, once, before the seam)
            from duckmind.life.upgrade import backup

            backup(self.state_file, "pre-g2")
        dispatch(self.session, "bridge", {"api": BRIDGE_API})
        self._saved_inputs = -1
        return True

    def grow(self, *, lived: bool = False, until: datetime | None = None) -> int:
        """(Under the lock; a life only.) Duck Core lives on to the calendar's now (or to ``until``,
        a moment before it), so its growth day is the calendar's: each local midnight credits a day
        the way its rules say. Time the duck was away (the server down, the computer asleep) is
        lived switched off, which the rules count as the family's absence (``lived``: the whole gap
        is lived awake, a backdated hatch). Returns the seconds lived."""
        if self.life is None or self.clock is None:
            return 0
        core = self.session.core
        gap = int(((until or self.clock.calendar()) - core.s.now).total_seconds())
        if gap < 1:
            return 0  # (or something someone did put Duck Core a moment ahead: it catches up)
        before = len(core.log.events)
        away = not lived and gap > OFFLINE_GAP_S * max(1.0, self.clock.speed) and core.s.awake
        if away:
            dispatch(self.session, "act", {"action": "power_off", "params": {}})
        dispatch(self.session, "advance", {"seconds": gap})
        if away and not core.s.awake:
            dispatch(self.session, "act", {"action": "power_on", "params": {}})
        self.mark(before, "summary")  # (lived at once: the page does not replay a whole day)
        return gap

    def refuse_in_life(self, method: str, params: dict[str, Any]) -> None:
        """(A life only.) What would hatch the duck again or move its time is refused in plain
        words: its time is the calendar's, and the record on disk is one duck's whole life."""
        if method == "new":
            raise ValueError(msg("web.life.refuse_new", name=self.session.config.name))
        if getattr(self.session.core.s, "bridged", False) and self._is_hand(method, params):
            # (what happens to the neural duck is what it lives: a hand on Duck Core's own page
            # would be a second body)
            raise ValueError(msg("web.life.refuse_hand", name=self.session.config.name))
        jumps = method in {"advance", "live", "party", "realtime"}
        if method == "command":
            from duckmind.duckcore import commands

            kinds = {c.kind for c in commands.parse(str(params.get("text", "")), "owner")}
            jumps = bool(kinds & {"advance", "live"})
        if jumps:
            raise ValueError(msg("web.life.refuse_time", name=self.session.config.name))

    # What the family's phone says to a duck attached to the neural duck, and nothing else of the
    # hands: the owner's hello, praise, "not that" and "enough" are words to Duck Core, not a body.
    PHONE_WORDS = frozenset({"owner_hello", "owner_good", "owner_not_that", "owner_enough"})

    @classmethod
    def _is_hand(cls, method: str, params: dict[str, Any]) -> bool:
        from duckmind.duckcore import commands

        if method == "act":
            return str(params.get("action", "")) not in cls.PHONE_WORDS
        if method == "command":
            return any(
                c.kind == "act" and c.method not in cls.PHONE_WORDS
                for c in commands.parse(str(params.get("text", "")), "owner")
            )
        return method in {"set_setting"} and str(params.get("key")) in {"floor", "docked", "hot"}

    def duck_seed(self) -> int:
        """The current duck's seed, which its voice comes from (read without the lock)."""
        return int(self.session.config.seed)

    def code_changed(self) -> bool:
        """Whether the code on disk is newer than what this process runs (checked every 10 s)."""
        checked_at, changed = self._stale_checked
        now = time.time()
        if now - checked_at > 10:
            changed = _code_mtime() > self.started_code + 1
            self._stale_checked = (now, changed)
        return changed

    def _restore(self) -> AnySession | None:
        """The duck from the last run: its config and every input, replayed exactly.

        Duck Core is deterministic, so the inputs are the whole duck; today's expression
        caps apply to the replay (a newer sweep changes how big its moves were, never
        what it chose to do before the caps mattered).
        """
        if self.state_file is None or not self.state_file.is_file():
            return None
        saved = json.loads(self.state_file.read_text())
        # (its engine and its world come back with it: an old save without engine_rev is a v4
        # duck on the frozen engine; strictly, a field it no longer knows is an error)
        try:
            session = restore(saved, caps_file=self.caps_file)
        except SaveContentMismatch as error:
            # a save from another content version is never replayed (it would be another duck)
            # and never overwritten: the server does not start, and says why
            raise SystemExit(content_mismatch_words(error, self.state_file)) from error
        except SaveBridgeMismatch as error:
            raise SystemExit(bridge_mismatch_words(error, self.state_file)) from error
        self._resume_realtime = bool(saved.get("realtime", False))
        return session

    def save(self, soon: bool = False) -> None:
        """Write config + inputs (small) after anything changed the duck. ``soon``: real
        time's own seconds, saved at most every SAVE_EVERY_S."""
        if self.state_file is None:
            return
        mark = self._save_mark()
        if mark == self._saved_inputs:
            return
        if soon and time.time() - self._saved_at < SAVE_EVERY_S:
            return
        self._saved_at = time.time()
        payload = {
            **save_json(self.session),  # config, world header, inputs
            "realtime": self.realtime,  # a restart carries on living in real time
        }
        write_private(self.state_file, json.dumps(payload, ensure_ascii=False))
        self._saved_inputs = mark

    def _save_mark(self) -> tuple[int, str, bool]:
        """What was last saved. Real time's seconds grow the last input instead of adding
        one, so the last input is part of it."""
        inputs = self.session.inputs
        return (len(inputs), repr(inputs[-1]) if inputs else "", self.realtime)

    def set_realtime(self, on: bool) -> None:
        if on != self.realtime:
            self._saved_inputs = -1  # write the new mode even if nothing else changed
        self.realtime = on
        self.anchor = (time.time(), self.session.core.s.now) if on else None

    def offset_min(self) -> int:
        """How far the duck's time of day is ahead of the wall clock (minutes, −720…720)."""
        now = self.session.core.s.now
        wall = datetime.now(now.tzinfo)
        ahead = ((now.hour * 60 + now.minute) - (wall.hour * 60 + wall.minute)) % (24 * 60)
        return ahead - 24 * 60 if ahead > 12 * 60 else ahead

    def catch_up_to_wall_clock(self, align: bool = False) -> int:
        """Move the duck's clock forward to today's real time of day (never backward).

        Real time should mean the duck's morning is the family's morning: walks, naps and
        bedtime follow the hour. The skipped time is lived normally, just fast. A duck
        that is already ahead of the wall clock (by less than half a day: someone skipped
        to the morning, or fast-forwarded a little) keeps its own time and simply lives on
        at real speed; ``align`` catches up to the wall clock anyway (up to a day).
        """
        core = self.session.core
        now = core.s.now
        wall = datetime.now(now.tzinfo)
        gap = (wall.hour * 60 + wall.minute) - (now.hour * 60 + now.minute)
        minutes = gap % (24 * 60)
        if not minutes or (minutes > 12 * 60 and not align):
            return 0
        # Those hours pass on the charger, as they would while the family is out: a
        # battery run flat in the catch-up would leave the family a switched-off duck.
        docked, floor, awake = core.s.docked, core.s.floor, core.s.awake
        if not docked:
            dispatch(self.session, "set_setting", {"key": "docked", "value": True})
        dispatch(self.session, "advance", {"minutes": minutes})
        if not docked:
            dispatch(self.session, "set_setting", {"key": "docked", "value": False})
        if awake and not core.s.awake:
            dispatch(self.session, "act", {"action": "power_on", "params": {}})
        if floor and not core.s.floor:
            dispatch(self.session, "set_setting", {"key": "floor", "value": True})
        return minutes

    def _run_clock(self) -> None:
        """Real time: the duck lives on by itself, second by second, at the wall clock's
        pace, whether or not anyone has the page open."""
        while True:
            time.sleep(1.0 - time.time() % 1.0 + 0.02)
            if self.live is not None and self.live.take_fresh():
                # The physics body just (re)started, sitting at its home's spawn: tell it what
                # the duck is doing now, paused or not (otherwise a paused page shows a sitting
                # body while the duck stands), and put it where the duck is (x_place, "start").
                with self.lock:
                    try:
                        self.live.resync(self.session, "start")
                    except Exception:
                        traceback.print_exc()
            if self.live is not None:
                try:
                    self.sync_clock()
                except Exception:
                    traceback.print_exc()
            if self.life is not None:
                with self.lock:
                    try:
                        host = self.life_host
                        if host is not None:
                            host.deliver()  # what the neural duck lived, before the clock moves on
                        self.grow()
                        self.save(soon=True)
                    except Exception:
                        traceback.print_exc()
            if not self.realtime:
                continue
            with self.lock:
                try:
                    self.step_realtime()
                    self.save(soon=True)
                except Exception:
                    traceback.print_exc()

    def _run_family(self) -> None:
        """Family visits (D3): :meth:`family_step` a few times a second."""
        while True:
            time.sleep(0.15)
            self.family_step()

    def family_step(self) -> None:
        """One pass of the family: look at the neural duck's pack (:meth:`_household_pack`), then
        move each visit on (the world does the walking and the hand); an act whose moment has come
        goes to Duck Core then, and its hand events are not played on the body a second time
        (``done_by_hand``)."""
        family = self.family
        if family is None:
            return
        try:
            self._household_pack(family)
        except Exception:
            traceback.print_exc()
        if not family.visits:
            return
        try:
            dues = family.tick()
        except Exception:
            traceback.print_exc()
            return
        for due in dues:
            with self.lock:
                if family is not self.family:
                    break  # a new duck: the old duck's visits are over
                try:
                    self.act_due(due)
                except Exception:
                    traceback.print_exc()

    def _household_pack(self, family: FamilyVisits) -> None:
        """The family helps the neural duck with its battery when Duck Core's ``battery_help`` is
        on and it has no nest (:mod:`duckmind.simweb.battery_help`): the pack the world holds is
        swapped by the family's hand, as the page's button would, and said once on the page. What
        Duck Core hears of it is only what the neural duck lives (a pack that jumped up is its one
        ``feed`` experience), so nothing is dispatched here."""
        live, host = self.live, self.life_host
        if live is None or host is None:
            return
        helping = battery_help.helping(
            self.session.core, on_screen=host.on_screen(), world_ready=live.world_ready()
        )
        before = self.battery_help.step(
            family, helping=helping, pack=battery_help.pack_of(live.world_truth())
        )
        driver = getattr(live, "neuro", None)
        if before is not None and driver is not None:
            driver.household_note("battery_help", msg("neuro.world.battery_help", before=before))

    def act_due(self, due: Any) -> dict[str, Any]:
        """(Under the lock.) A family visit's act, now that its hand did it."""
        self.step_realtime()
        before = len(self.session.core.log.events)
        params = {"action": due.action, "persona": due.persona, "params": due.params}
        result = dispatch(self.session, "act", params)
        if self.live is not None and len(self.session.core.log.events) > before:
            self.live.after_call(self.session, before, done_by_hand=due.done_by_hand)
        self.family_done = [
            *self.family_done[-7:],
            {
                "id": next(self._family_ids),
                "visit": due.visit,
                "action": due.action,
                "persona": due.persona,
                "late_s": due.late_s,
                "note": result.get("note") if isinstance(result, dict) else None,
                "perceived": result.get("perceived") if isinstance(result, dict) else None,
                "at_ms": round(time.time() * 1000),
            },
        ]
        self.save()
        return result

    def visit(self, action: str, persona: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """(Under the lock.) A touch pressed as a family member, with live physics that has the
        household: that person walks over first and their hand does it; the act reaches Duck
        Core when the hand gets there. None: act now (not a touch, or no world to walk in)."""
        family = self.family
        if family is None or self.live is None or not self.live.world_ready():
            return None
        visit = family.request(action, persona, params)
        if visit is None:
            return None
        people = family.people()
        who = (people.get(str(visit.get("persona"))) or {}).get("zh") or msg(
            "web.server.SimState.visit.a12728"
        )
        eta = visit.get("eta_s")
        walk = (
            msg("web.server.SimState.visit.f2aa33", eta=max(1, round(float(eta))))
            if isinstance(eta, (int, float))
            else ""
        )
        return {
            "note": msg("web.server.SimState.visit.d7fd80", who=who, walk=walk),
            "visit": visit,
            "deferred": True,
        }

    def wall_of(self, moment: datetime) -> float:
        """The wall-clock epoch seconds of a moment on the duck's clock (real time)."""
        assert self.anchor is not None
        wall, duck = self.anchor
        return wall + (moment - duck).total_seconds()

    def step_realtime(self, wall: float | None = None) -> None:
        """Live the duck up to the wall clock (real time). Every event and idle glance gets
        its wall-clock moment plus SHOW_DELAY_S, so the page and the live body show it
        together, in order, a moment after it happened; nothing is decided ahead of time,
        so a hand, a pickup or switching off always comes before whatever follows it."""
        if not self.realtime:
            return
        core = self.session.core
        wall = time.time() if wall is None else wall
        if self.anchor is None:
            self.anchor = (wall, core.s.now)
        target = self.anchor[1] + timedelta(seconds=int(wall - self.anchor[0]))
        behind = int((target - core.s.now).total_seconds())
        if behind < -60:
            # Someone moved its clock ahead (skipped to the morning, a routine's weeks): it
            # lives on from its own time at the wall clock's pace.
            self.anchor = (wall, core.s.now)
            return
        if behind <= 0:
            return  # something someone did took a few seconds: the wall clock catches up
        before = len(core.log.events)
        if behind > CATCH_UP_S:
            # The computer slept (or the server was stopped): catch up, don't replay.
            dispatch(self.session, "advance", {"seconds": behind})
            self.mark(before, "summary")
            if self.live is not None:
                self.live.resync(self.session)
            return
        started = core.s.now
        glance = core.can_fidget()
        dispatch(self.session, "advance", {"seconds": behind})
        dues: dict[int, float] = {}
        for index in range(before, len(core.log.events)):
            dues[index] = self.wall_of(core.log.events[index].occurred_at) + SHOW_DELAY_S
            self.marks[index] = {"due_ms": round(dues[index] * 1000)}
        fidgets = []
        if glance and core.can_fidget():
            # (Any: a v4_0 duck's events are the frozen engine's own Event class)
            recent: list[Any] = core.log.events[max(0, len(core.log.events) - 60) :]
            for item in core.fidgets_between(started, core.s.now, recent):
                fidgets.append(
                    {
                        "id": next(self._fidget_ids),
                        "due_ms": round((self.wall_of(item["at"]) + SHOW_DELAY_S) * 1000),
                        "clip": item["clip"],
                        "amplitude": item["amplitude"],
                        "tempo": item["tempo"],
                        "weak": True,
                    }
                )
        horizon = (time.time() - 10) * 1000
        self.fidgets = [item for item in self.fidgets if item["due_ms"] > horizon] + fidgets
        self._trim_marks()
        if self.live is not None:
            self.live.after_call(self.session, before, dues=dues, fidgets=fidgets)

    def mark(self, before: int, how: str) -> None:
        """Note how the page should show the events a call just made (see ``marks``)."""
        events = self.session.core.log.events
        if how == "summary":
            for index in range(before, len(events)):
                self.marks[index] = {"settle": True}
        elif how == "fast":
            # A fast-forward step: show its newest walk and newest move, apply the rest.
            keep: set[int] = set()
            for want in ("walk", "clip"):
                for index in range(len(events) - 1, before - 1, -1):
                    motion = _motion(events[index])
                    if motion is not None and want in motion:
                        keep.add(index)
                        break
            for index in range(before, len(events)):
                if index not in keep:
                    self.marks[index] = {"settle": True}
        self._trim_marks()

    def _trim_marks(self) -> None:
        if len(self.marks) > MARKS_KEEP:
            floor = len(self.session.core.log.events) - MARKS_KEEP // 2
            self.marks = {index: mark for index, mark in self.marks.items() if index >= floor}

    def annotate(self, feed_payload: dict[str, Any]) -> dict[str, Any]:
        """How the page shows each event (``marks``), and the sounds it made (``calls``: the
        tags the page plays in the duck's own voice; an event that is only words has none)."""
        events = self.session.core.log.events
        for item in feed_payload["events"]:
            seq = int(item.get("seq", -1))
            mark = self.marks.get(seq)
            if mark:
                item.update(mark)
            if 0 <= seq < len(events):
                calls = voice.calls_of(str(item.get("kind", "")), events[seq].payload)
                if calls:
                    item["calls"] = calls
        return feed_payload

    def body(
        self, known_home: str | None = None, since_t: float | None = None, frames: bool = False
    ) -> dict[str, Any]:
        """What the official robotd + MuJoCo body is doing, for the 3D view. ``known_home``:
        the fingerprint of the home the page has (``?home=``); when it is not this one, the
        answer carries the home to draw (``home_json``, see homeview). ``frames``
        (``?since_t=``, empty for "from now"): also every body sample after ``since_t``, so the
        page plays the physics at its true speed (LiveBridge.frames_since)."""
        home = homeview.home_payload(self.home())
        out: dict[str, Any]
        if self.live is None:
            out = {
                "available": False,
                "enabled": False,
                "realtime_clock": self.realtime,
                "boot": self.boot,
                "world": None,
            }
        else:
            out = {
                "enabled": True,
                "realtime_clock": self.realtime,
                "boot": self.boot,
                **self.live.snapshot(since_t, frames=frames),
            }
            if self.family is not None and isinstance(out.get("world"), dict):
                # (what the duck made of a visit is Duck Core's words: matched to its engine)
                out["world"]["family"] = boundary.annotate(
                    {"visits": self.family.view(), "done": self.family_done},
                    boundary.engine_of(self.session),
                )
        layout = home.get("layout") or {}
        out["home"] = boundary.annotate_home(
            {"name": home["name"], "zh": layout.get("zh"), "fingerprint": home.get("fingerprint")},
            layout.get("id"),
        )
        if known_home is not None and known_home != home.get("fingerprint"):
            out["home_json"] = i18nwire.home_answer(home)
        return out

    def eye_image(self) -> tuple[bytes | None, str, str]:
        """The latest duck's-eye JPEG and its tag, or (None, "", why there is none)."""
        if self.live is None:
            return None, "", msg("web.server.SimState.eye_image")
        return self.live.eye_image()

    def ping(self) -> dict[str, Any]:
        """A cheap sign of life for the page (no lock: it only reads plain fields)."""
        return {
            "ok": True,
            "api": API_VERSION,
            "boot": self.boot,
            "realtime": self.realtime,
            "now_ms": round(time.time() * 1000),
        }

    def reset(self, params: dict[str, Any]) -> None:
        self.session = open_session(
            {
                "engine_rev": ENGINE_REV,
                "seed": int(params.get("seed", 7)),
                "name": str(params.get("name", "小鴨"))[:12] or "小鴨",
                "tof": bool(params.get("tof", False)),
                "pickup_detect": bool(params.get("pickup_detect", True)),
                "ear": bool(params.get("ear", True)),
                "name_ear": bool(params.get("name_ear", True)),
                "motion_validated": bool(params.get("motion_validated", True)),
                # (the world's: the new-duck dialog's clumsy duck falls more often)
                "fall_ppm_per_s": max(0, min(1_000_000, int(params.get("fall_ppm_per_s", 50)))),
                "caps_file": self.caps_file,
                "conservative": bool(params.get("conservative", True)),
                "battery_help": bool(params.get("battery_help", True)),
                # Owner decision 2026-09-26: the simulator hatches ducks with a charging nest.
                "nest": bool(params.get("nest", True)),
            }
        )
        self.marks = {}
        self.fidgets = []
        self._saved_inputs = -1  # a new duck replaces the saved one on the next save
        if self.realtime:
            self.anchor = (time.time(), self.session.core.s.now)
        # Another duck: its own family (the profile follows the seed); visits are over.
        self.family = self._family()
        self.family_done = []
        self.battery_help = battery_help.BatteryHelp()
        self.voice.prepare(self.duck_seed())  # and its own voice


def home_brief(home: str) -> dict[str, Any]:
    """Which home (every /api/call answer carries it), its name's message beside it."""
    brief = homeview.home_brief(home)
    return boundary.annotate_home(brief, brief.get("id"))


WORLD_BUTTONS = frozenset(
    {"body_upright", "body_show", "body_eye", "body_roll", "body_tidy", "body_place"}
)
SHOW_KINDS = {
    "bump": msg("web.server.SHOW_KINDS.bump"),
    "peck": msg("web.server.SHOW_KINDS.peck"),
    "kick": msg("web.server.SHOW_KINDS.kick"),
}


def world_button(live: LiveBridge, method: str, params: dict[str, Any]) -> dict[str, Any]:
    """The physics world's own buttons (never Duck Core's: it is not told, W05):
    「扶起物理身體」 (``body_upright``, owner decision D2: the world's hand), a curiosity
    demonstration (``body_show``: ``kind`` bump | peck | kick, ``thing`` a toy's id; 示範,
    truth-positioned), the same done by the duck's own eyes and body (``body_eye``, 「用眼睛做」,
    owner decision A6), rolling a ball to the duck (``body_roll``, ``thing`` optional) and
    tidying the toys into their corner (``body_tidy``)."""
    if method == "body_upright":
        answer = live.upright()
        eta = answer.get("eta_s")
        took = (
            msg("web.server.world_button.f2aa33", eta=float(eta))
            if isinstance(eta, (int, float))
            else ""
        )
        if getattr(live, "neuro", None) is not None:  # prototype mode: its own body senses it
            return {"note": msg("neuro.hook.upright_note", took=took), "hand": answer}
        return {
            "note": msg("web.server.world_button.1568dc", took=took),
            "hand": answer,
        }
    if method == "body_show":
        kind = str(params.get("kind", ""))
        if kind not in SHOW_KINDS:
            raise ValueError(
                msg("web.server.world_button.165078", SHOW_KINDS=lst(SHOW_KINDS.values()))
            )
        thing = str(params.get("thing") or "")
        if not thing:
            raise ValueError(msg("web.server.world_button.10ca9b"))
        side = params.get("side")
        started = live.demo(kind, thing, str(side) if side in ("left", "right") else None)
        return {
            "note": msg("web.server.world_button.ceaba0", kind=SHOW_KINDS[kind]),
            "demo": started,
        }
    if method == "body_eye":
        # 「用眼睛做」 (owner decision A6, phase 3): the duck's own eyes and body do it
        from duckmind.simweb.eyedemo import NOTE_ZH

        kind = str(params.get("kind", ""))
        if kind not in SHOW_KINDS:
            raise ValueError(
                msg("web.server.world_button.8e3d70", SHOW_KINDS=lst(SHOW_KINDS.values()))
            )
        thing = str(params.get("thing") or "")
        if not thing:
            raise ValueError(msg("web.server.world_button.10ca9b"))
        started = live.eye_demo(kind, thing)
        return {
            "note": msg("web.server.world_button.a9149b", kind=SHOW_KINDS[kind], NOTE_ZH=NOTE_ZH),
            "demo": started,
        }
    if method == "body_place":
        thing = str(params.get("thing") or "")
        if not thing:
            raise ValueError(msg("web.server.world_button.10ca9b"))
        try:
            x_mm = int(params["x_mm"])
            y_mm = int(params["y_mm"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(msg("web.server.world_button.place_xy")) from None
        on = str(params.get("on") or "floor")
        if on not in ("floor", "top"):
            raise ValueError(msg("web.server.world_button.place_on"))
        answer = live.place_prop(thing, x_mm, y_mm, on)
        return {"note": msg("web.server.world_button.place_ok"), "world": answer}
    if method == "body_roll":
        request: dict[str, Any] = {"toward": "duck"}
        if params.get("thing"):
            request["id"] = str(params["thing"])
        answer = live.world_action("x_roll_ball", **request)
        return {"note": msg("web.server.world_button.aee71d"), "world": answer}
    answer = live.world_action("x_tidy", to=str(params.get("to") or "toy_corner"))
    return {"note": msg("web.server.world_button.937b98"), "world": answer}


def _playback(method: str, params: dict[str, Any], result: dict[str, Any]) -> str:
    """How the page should show what a call did: play it, play only its newest moves
    (a fast-forward step), or show the result without replaying it (a summary)."""
    if method in {"live", "party"}:
        return "summary"
    if method == "advance":
        if params.get("until"):
            return "summary"
        minutes = int(params.get("minutes", 1))
        if minutes > SUMMARY_MINUTES:
            return "summary"
        return "fast" if minutes >= FAST_MINUTES else "play"
    if method == "command":
        kinds = {str(item.get("kind")) for item in result.get("results", [])}
        return "summary" if kinds & {"advance", "live"} else "play"
    return "play"


def _seconds(text: str) -> float | None:
    """A query's number of seconds (``since_t``), None when it is empty or not a number."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def make_handler(
    state: SimState,
    neuro: neuro_api.NeuroHub | None = None,
    voice_bank: voice.VoiceBank | None = None,
    live_hub: LiveHub | None = None,
) -> type[BaseHTTPRequestHandler]:
    static_root = resources.files("duckmind.simweb").joinpath("static")
    # The neural-duck prototype's page routes (/api/neuro/…, neuro_api): "off" without it.
    hub = neuro if neuro is not None else neuro_api.NeuroHub()
    # Sound routes are part of the default Neuro page, not a separate demo server.
    from duckmind.simweb.neuro_sound_live import SoundHub

    sound_hub = SoundHub(lambda: getattr(state.live, "neuro", None))
    # A browser microphone as a person in the home (live voice, LV0): no page for it yet.
    from duckmind.simweb.neuro_mic_api import MicHub

    # The talk panel's routes (live voice, LV5): "off" (``enabled: false``) without --live-voice.
    live_voice_hub = live_hub if live_hub is not None else LiveHub()
    mic_hub = MicHub(lambda: getattr(state.live, "neuro", None), on_speaker=live_voice_hub.speaker)
    # The duck's own official voice (voice.py): this one, else the hub's, else the state's.
    if voice_bank is None:
        voice_bank = hub.voice if hub.voice is not None else state.voice
    bank = voice_bank

    def voice_duck() -> int:
        """The duck on screen, whose voice is served: the neural duck in the prototype (the
        picker's), else Duck Core's duck (read without the lock: one attribute)."""
        return hub.voice_duck() if hub.enabled else state.duck_seed()

    growth_lock = threading.Lock()
    growth_cache: dict[str, Any] = {}

    def fast_forward() -> Any:
        """The page's fast-forward manager (one per server): its test ducks never use the folder of
        the duck this server lives."""
        with growth_lock:
            if state.fastforward is None:
                from duckmind.life.fastforward import FastForward

                state.fastforward = FastForward(
                    [state.life.folder] if state.life is not None else []
                )
            return state.fastforward

    class Handler(BaseHTTPRequestHandler):
        server_version = "DuckCoreSim/1"

        def log_message(self, format: str, *args: Any) -> None:  # quieter console
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, error: BaseException | str) -> None:
            """An error answer: its words (a Msg keeps its message) in the viewer's language."""
            said = (
                error.args[0]
                if isinstance(error, BaseException)
                and error.args
                and isinstance(error.args[0], Msg)
                else str(error)
            )
            self._json(status, i18nwire.call_answer({"ok": False, "error": said}, state.session))

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            self._send(
                status,
                json.dumps(payload, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )

        def _life_growth(self) -> None:
            """The 「成長」 picture of the duck this server lives (docs/design/GROWTH-G4-2026-10-11.md):
            read only. Its words get their ``_i18n`` maps outside the state's lock, once per picture."""
            host = state.life_host
            if host is None:
                self._json(HTTPStatus.OK, {"ok": True, "growth": None})
                return
            with state.lock:
                view = host.growth_view()
            with growth_lock:
                if growth_cache.get("view") is not view:
                    growth_cache["answer"] = i18nwire.growth_answer(
                        {"ok": True, "growth": view}, boundary.engine_of(state.session)
                    )
                    growth_cache["view"] = view
                answer = growth_cache["answer"]
            self._json(HTTPStatus.OK, answer)

        def _fastforward_post(self, post_path: str) -> None:
            """Start or cancel a fast-forward of throwaway test ducks. One at a time; the page's
            button names no folder (the ducks live in a temporary one, never a duck's own)."""
            from duckmind.life import fastforward as ff

            manager = fast_forward()
            try:
                if post_path == "/api/life/fastforward/cancel":
                    manager.cancel()
                    self._json(HTTPStatus.OK, {"ok": True, "fastforward": manager.status()})
                    return
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length > MAX_BODY:
                    self._json(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        {"ok": False, "error": "request too large"},
                    )
                    return
                request = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(request, dict):
                    raise ff.Invalid(msg("web.growth.ff.bad_days", max=ff.MAX_DAYS))
                answer = manager.start(
                    days=request.get("days", ff.DEFAULT_DAYS),
                    styles=request.get("styles", ["attentive"]),
                    seed=request.get("seed", ff.DEFAULT_SEED),
                )
                self._json(HTTPStatus.OK, {"ok": True, "fastforward": answer})
            except ff.Busy as error:
                self._error(HTTPStatus.CONFLICT, error)
            except ff.FastForwardError as error:
                self._error(HTTPStatus.BAD_REQUEST, error)
            except ValueError as error:  # (a body that is not JSON)
                self._error(HTTPStatus.BAD_REQUEST, error)

        def _layout_get(self, path: str, query: str = "") -> None:
            """The layout editor's reads: the home that runs (``?which=default``: the server's default
            home instead), the library, the saved layouts."""
            try:
                if path == "/api/layout":
                    which = parse_qs(query).get("which", [""])[0]
                    self._json(HTTPStatus.OK, state.layout_state(default=which == "default"))
                elif path == "/api/layout/library":
                    from duckmind.world import objlib

                    self._json(HTTPStatus.OK, {"ok": True, **objlib.library().page_json()})
                elif path == "/api/layout/list":
                    self._json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "saved": state.layouts.list(),
                            "can_save": state.layouts.enabled,
                        },
                    )
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            except layouts.LayoutProblem as problem:
                self._json(HTTPStatus.OK, {"ok": False, "code": problem.code, **problem.params})
            except Exception as error:  # surface bugs to the page instead of hanging it
                traceback.print_exc()
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": f"{type(error).__name__}: {error}"},
                )

        def _layout_post(self, path: str) -> None:
            """The layout editor's actions: check, save, load, delete, apply, reset. A refusal is a normal
            answer (``ok`` false with a ``code``); the page says it in the viewer's language."""
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > layouts.MAX_BYTES + 1024:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "code": "too_big"})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body")
            except ValueError:
                self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "code": "bad_request"})
                return
            verb = path[len("/api/layout/") :]
            try:
                if verb == "check":
                    raw = layouts.validate_upload(body.get("layout"))
                    self._json(HTTPStatus.OK, {"ok": True, **_layoutcheck.check(raw)})
                elif verb == "save":
                    raw = layouts.validate_upload(body.get("layout"))
                    verdict = _layoutcheck.check(raw)
                    if not verdict["ok"]:
                        raise layouts.LayoutProblem("invalid", problems=verdict["problems"])
                    name = str(body.get("name", "")).strip()
                    state.layouts.save(name, raw)
                    self._json(
                        HTTPStatus.OK, {"ok": True, "name": name, "saved": state.layouts.list()}
                    )
                elif verb == "load":
                    raw = state.layouts.read(str(body.get("name", "")))
                    self._json(
                        HTTPStatus.OK, {"ok": True, "layout": raw, "check": _layoutcheck.check(raw)}
                    )
                elif verb == "delete":
                    state.layouts.delete(str(body.get("name", "")))
                    self._json(HTTPStatus.OK, {"ok": True, "saved": state.layouts.list()})
                elif verb == "apply":
                    raw = layouts.validate_upload(body.get("layout"))
                    with state.lock:
                        self._json(HTTPStatus.OK, state.layout_apply(raw))
                elif verb == "reset":
                    with state.lock:
                        self._json(HTTPStatus.OK, state.layout_apply(None))
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            except layouts.LayoutProblem as problem:
                self._json(HTTPStatus.OK, {"ok": False, "code": problem.code, **problem.params})
            except Exception as error:
                traceback.print_exc()
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": f"{type(error).__name__}: {error}"},
                )

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            path, _, query = self.path.partition("?")
            if path == "/api/skill-evidence":
                status, body = skill_evidence.json_response()
                self._send(status, body, "application/json; charset=utf-8")
                return
            if path.startswith("/skill-evidence/"):
                status, body, headers = skill_evidence.clip_response(
                    path, self.headers.get("Range")
                )
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                if body:
                    self.wfile.write(body)
                return
            if path == "/api/robot":
                self._json(HTTPStatus.OK, i18nwire.body_answer(robot_model.load_model()))
                return
            if live_voice_hub.handles(path):
                live_voice_hub.get(self, path, query)
                return
            if mic_hub.handles(path):
                mic_hub.get(self, path, query)
                return
            if sound_hub.handles(path):
                sound_hub.get(self, path, query)
                return
            if hub.handles(path):
                hub.get(self, path, query)
                return
            if path == "/api/body":
                params = parse_qs(query, keep_blank_values=True)
                asked = params.get("home")
                known = asked[0] if asked else None
                # ``since_t``: the page plays every body sample (empty: from now)
                wants = params.get("since_t")
                since_t = _seconds(wants[0]) if wants else None
                self._json(
                    HTTPStatus.OK,
                    i18nwire.body_answer(state.body(known, since_t, frames=wants is not None)),
                )
                return
            if path.startswith("/api/layout"):
                self._layout_get(path, query)
                return
            if path == "/api/home":
                self._json(HTTPStatus.OK, i18nwire.home_answer(homeview.home_payload(state.home())))
                return
            if path == "/api/eye.jpg":
                image, tag, why = state.eye_image()
                if image is None:
                    # 202: the window just opened and the first picture is on its way (F4);
                    # 404: there is none to be had, and why. The words are the body (繁體中文,
                    # as before); their message rides in a header (ASCII JSON) for the page's
                    # other languages.
                    pending = tag == "pending"
                    status = HTTPStatus.ACCEPTED if pending else HTTPStatus.NOT_FOUND
                    said = i18nwire.body_answer({"text": why})
                    wire = (said.get("_i18n") or {}).get("text")
                    body = str(why).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    if wire is not None:
                        self.send_header("X-Duck-I18n", json.dumps(wire, ensure_ascii=True))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if tag and self.headers.get("If-None-Match") == tag:
                    self.send_response(HTTPStatus.NOT_MODIFIED)
                    self.send_header("ETag", tag)
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(image)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if tag:
                    self.send_header("ETag", tag)
                self.end_headers()
                self.wfile.write(image)
                return
            if path == "/api/ping":
                self._json(HTTPStatus.OK, state.ping())
                return
            if path == "/api/life":
                # The life this server lives (counts only: no names or looks leave the machine).
                host = state.life_host
                with state.lock:
                    life_view = host.status() if host is not None else None
                    if life_view is not None:
                        life_view["battery_help"] = state.battery_help.view()  # (numbers only)
                self._json(HTTPStatus.OK, {"ok": True, "life": life_view})
                return
            if path == "/api/life/growth":
                self._life_growth()
                return
            if path == "/api/life/fastforward":
                answer = fast_forward().status(results=parse_qs(query).get("results") == ["1"])
                self._json(HTTPStatus.OK, {"ok": True, "fastforward": answer})
                return
            if path == "/api/neuro":
                # Prototype mode (--brain neuro): the neural duck's telemetry since ``since``.
                driver = getattr(state.live, "neuro", None)
                if driver is None:
                    self._error(
                        HTTPStatus.NOT_FOUND, ValueError(msg("neuro.error.must_run.378a40"))
                    )
                    return
                raw = parse_qs(query).get("since", ["0"])[0]
                since = int(raw) if raw.isdigit() else 0
                self._json(HTTPStatus.OK, boundary.attach({"ok": True, **driver.telemetry(since)}))
                return
            # The duck's own voice: its manifest and its bank's whitelisted files (voice.py).
            reply = bank.handle(path, voice_duck())
            if reply is not None:
                voice.send(self, reply)
                return
            if path == "/api/clips":
                with state.lock:
                    self._json(
                        HTTPStatus.OK, i18nwire.call_answer(clips(state.session), state.session)
                    )
                return
            if path.startswith("/mesh/"):
                accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "")
                payload = robot_model.mesh_payload(path[len("/mesh/") :], accepts_gzip)
                if payload is None:
                    self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                    return
                body, encoding = payload
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "model/stl")
                self.send_header("Content-Length", str(len(body)))
                if encoding:
                    self.send_header("Content-Encoding", encoding)
                self.send_header("Vary", "Accept-Encoding")
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(body)
                return
            # (「資料」 is a view of the same page: /data serves the page, shell.js shows that view)
            if path in {"/", "/index.html", "/data", "/data/"}:
                path = "/index.html"
            name = path.lstrip("/")
            parts = name.split("/")
            if any(not part or part.startswith(".") for part in parts) or (
                len(parts) > 1 and parts[0] not in STATIC_FOLDERS
            ):
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
            target = static_root.joinpath(name)
            if suffix not in STATIC_TYPES or not target.is_file():
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            self._send(HTTPStatus.OK, target.read_bytes(), STATIC_TYPES[suffix])

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            post_path, _, post_query = self.path.partition("?")
            if live_voice_hub.handles(post_path):
                live_voice_hub.post(self, post_path)
                return
            if mic_hub.handles(post_path):
                mic_hub.post(self, post_path, post_query)
                return
            if sound_hub.handles(post_path):
                sound_hub.post(self, post_path)
                return
            if hub.handles(self.path):
                hub.post(self, post_path)
                return
            if post_path.startswith("/api/layout/"):
                self._layout_post(post_path)
                return
            if post_path in {"/api/life/fastforward", "/api/life/fastforward/cancel"}:
                self._fastforward_post(post_path)
                return
            if self.path != "/api/call":
                self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > MAX_BODY:
                self._json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "error": "request too large"}
                )
                return
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
                method = str(request.get("method", "state"))
                params = request.get("params") or {}
                cursor = request.get("cursor") or {}
                with state.lock:
                    # Real time: first live up to this second, so whatever this call does
                    # happens after everything the duck already did, never before it.
                    state.step_realtime()
                    if state.life is not None:
                        state.refuse_in_life(method, params)
                    before = len(state.session.core.log.events)
                    # A family member's own button (the interaction cards ask for it with
                    # "visit"; a banner's suggested steps never do: they stay in order).
                    asked_visit = method == "act" and bool(params.pop("visit", False))
                    visit = (
                        state.visit(
                            str(params.get("action", "")),
                            str(params.get("persona", "owner")),
                            dict(params.get("params") or {}),
                        )
                        if asked_visit
                        else None
                    )
                    if visit is not None:
                        # A family member walks over first: Duck Core hears it later (D3).
                        result: dict[str, Any] = visit
                    elif method == "new":
                        state.reset(params)
                        result = {"note": "重新孵化了"}
                        cursor = {}
                        before = 0
                        if state.live is not None:
                            # The new duck's own home (with or without its nest) and head.
                            config = state.session.config
                            state.live.restart(nest=bool(config.nest), tof=bool(config.tof))
                    elif method == "realtime":
                        on = bool(params.get("on", False))
                        # "catch_up": false — a banner button shows its action live from
                        # the duck's own clock instead of first living hours unseen.
                        catch_up = bool(params.get("catch_up", True)) or bool(
                            params.get("align", False)
                        )
                        caught_up = (
                            state.catch_up_to_wall_clock(bool(params.get("align", False)))
                            if on and catch_up
                            else 0
                        )
                        state.mark(before, "summary")
                        if state.live is not None and caught_up:
                            state.live.resync(state.session)
                        state.set_realtime(on)
                        result = {
                            "realtime": on,
                            "caught_up_minutes": caught_up,
                            "offset_min": state.offset_min(),
                        }
                    elif method == "body_demo":
                        if state.live is None:
                            raise ValueError("live physics is off")
                        state.live.demo_walk()
                        result = {
                            "note": "示範：物理身體自己站起來走一小圈（不是小鴨的決定，"
                            "牠的位置不算數；走完會回到小鴨自己的姿勢）"
                        }
                    elif method == "body_reset":
                        if state.live is None:
                            raise ValueError("live physics is off")
                        state.live.restart()
                        result = {"note": "物理身體重新放回起點坐好（小鴨自己沒有改變）"}
                    elif method in WORLD_BUTTONS:
                        if state.live is None:
                            raise ValueError(msg("web.server.make_handler.Handler.do_POST.fb5df2"))
                        result = world_button(state.live, method, params)
                    else:
                        result = dispatch(state.session, method, params)
                        how = _playback(method, params, result)
                        state.mark(before, how)
                        if state.realtime and method in {"advance", "live", "party", "command"}:
                            # Its clock was moved on: real time carries on from there.
                            state.anchor = (time.time(), state.session.core.s.now)
                        # Physics shows real time only; fast-forwarded minutes are not
                        # thrown at it (the page draws Duck Core's record instead).
                        changed = len(state.session.core.log.events) > before
                        if state.live is not None and changed and how == "play":
                            if state.realtime or method in {"act", "set_setting"}:
                                state.live.after_call(state.session, before)
                            elif method == "advance":
                                # A fast-forward minute the physics did not live: its world
                                # follows the plan (as after any fast-forward, see resync).
                                state.live.skipped(state.session)
                        elif state.live is not None and changed and how == "fast":
                            # The page's 5 and 15 min/s steps (once a second): one placement
                            # once they stop, not a synchronous x_place (which also cancels
                            # the hand and resets the joints) on every step.
                            state.live.skipped(state.session)
                        elif state.live is not None and changed:
                            state.live.resync(state.session)
                    if method == "new":
                        state.mark(0, "summary")
                    if method != "state":
                        state.save()
                    state.sync_clock()
                    horizon = (time.time() - 1) * 1000
                    shown = snapshot(state.session)
                    # Duck Core's own estimate of how far off its idea of where it is may be:
                    # the 3D view draws it as a ring around the plan (the shadow duck).
                    shown["body"]["pose_err_mm"] = int(state.session.core.s.pose_err_mm)
                    payload = {
                        "ok": True,
                        "result": result,
                        "server": {
                            "api": API_VERSION,
                            "boot": state.boot,
                            "code_changed": state.code_changed(),
                            "realtime": state.realtime,
                            "live": state.live is not None,
                            "offset_min": state.offset_min(),
                            "now_ms": round(time.time() * 1000),
                            "show_delay_ms": round(SHOW_DELAY_S * 1000),
                            "lights": lights_payload(
                                state.light_time(), state.home(), state.duck_at()
                            ),
                            "home": home_brief(state.home()),
                        },
                        "state": shown,
                        "feed": state.annotate(
                            feed(
                                state.session,
                                int(cursor.get("events", 0)),
                                int(cursor.get("journal", 0)),
                            )
                        ),
                        "fidgets": [item for item in state.fidgets if item["due_ms"] > horizon],
                    }
                    # (the words' messages next to them: Duck Core's by its engine's templates)
                    payload = i18nwire.call_answer(payload, state.session)
                self._json(HTTPStatus.OK, payload)
            except (ValueError, KeyError, TypeError) as error:
                self._error(HTTPStatus.BAD_REQUEST, error)
            except Exception as error:  # surface bugs to the page instead of hanging it
                traceback.print_exc()
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": f"{type(error).__name__}: {error}"},
                )

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Duck Core simulator (local web page)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--name",
        default=None,
        help="the duck's name (default 小鴨); with --brain neuro it is also the neural duck's "
        "name on the page, for the duck the server starts with",
    )
    parser.add_argument(
        "--expression-caps",
        default="",
        help="caps file from the expression sweep (per-clip amplitude limits)",
    )
    parser.add_argument(
        "--microduck-mjcf", default=None, help="folder with the official robot_walk.xml and assets/"
    )
    live = parser.add_argument_group(
        "live physics", "play every motion on the official robotd --sim + MuJoCo body too"
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path.home() / ".duckmind" / "simweb-duck.json",
        help="where the duck is kept between runs (its config and inputs); '' to not keep it",
    )
    live.add_argument("--live", action="store_true")
    live.add_argument("--microduck", type=Path, help="official microduck checkout (robotd)")
    live.add_argument("--microduck-rl", type=Path, help="official microduck_rl checkout (body)")
    live.add_argument("--policies", type=Path, help="official policy set (…/current)")
    live.add_argument("--robotd-bin", type=Path, help="robotd binary built for this machine")
    live.add_argument("--ort-lib", type=Path, help="ONNX Runtime library for robotd")
    neuro = parser.add_argument_group(
        "neural duck (prototype)",
        "--brain neuro: the neural duck (duckmind.neuro), not Duck Core's planner, drives the "
        "live body (needs --live)",
    )
    neuro.add_argument("--brain", choices=("duckcore", "neuro"), default="duckcore")
    neuro.add_argument(
        "--neuro-duck",
        default="",
        help="which neural duck: a seed ('7'), a seed at a growth day ('7:62') or a preset name; "
        "default: --seed",
    )
    neuro.add_argument(
        "--neuro-record", type=Path, default=None, help="JSONL of every tick's frame and command"
    )
    neuro.add_argument(
        "--neuro-tof",
        action="store_true",
        help="the neural duck's head has the ToF module (tofd's frames reach its senses)",
    )
    neuro.add_argument(
        "--neuro-start",
        choices=("mat", "nest"),
        default="mat",
        help="where the neural duck starts: seated on the mat in the room (default) or in its dock",
    )
    neuro.add_argument(
        "--neuro-time",
        default=None,
        help="TESTS ONLY: the neural duck's time of day at the start (HH:MM, today; its clock runs "
        "on from there at the real pace; the page can change it). Without it the duck lives by "
        "the real calendar (one clock: the neural duck's day and night, the room's lights and "
        "Duck Core's growth days)",
    )
    neuro.add_argument(
        "--clock-speed",
        type=float,
        default=1.0,
        help="TESTS ONLY: the duck's days pass this many times faster than the wall clock "
        "(a life resumes at its own time then, not at the real calendar)",
    )
    home_group = parser.add_argument_group(
        "the home (docs/design/REAL-HOME-2026-10-xx.md, docs/design/HOME-FORMAT.md)",
        "which home the physics runs and the page draws, and where the layout editor keeps its files",
    )
    home_group.add_argument(
        "--home",
        default=None,
        help="the home to run (--live): a name (default_home, real_home, hazard_home, apartment ...), "
        "'name,variant' or a layout file; without it the home the layout editor last applied, else "
        "$DUCKMIND_LIVE_HOME, else default_home",
    )
    home_group.add_argument(
        "--layouts-dir",
        type=Path,
        default=None,
        help="the layout editor's folder (saved layouts, the one applied last); default: the life's "
        "'layouts' folder with --life, else ~/.duckmind/layouts",
    )
    life_group = parser.add_argument_group(
        "a life (docs/design/GROWTH-G1-2026-10-10.md)",
        "--life FOLDER: this server lives one duck, kept in FOLDER (hatched now if the folder is "
        "new): its Duck Core save, its neural snapshot, its eye. --seed and --name are its seed and "
        "name when it hatches; --no-nest and --tof are the hardware it hatches with",
    )
    life_group.add_argument("--life", type=Path, default=None, help="the duck's folder")
    life_group.add_argument(
        "--no-nest", action="store_true", help="a new duck hatches without the charging nest"
    )
    life_group.add_argument(
        "--tof", action="store_true", help="a new duck's head has the ToF module"
    )
    life_group.add_argument(
        "--life-eye-from",
        type=Path,
        default=None,
        help="a new duck starts with a copy of this eye folder (the floor it learnt)",
    )
    life_group.add_argument(
        "--life-snapshot-every",
        type=float,
        default=600.0,
        help="seconds between snapshots of the neural duck's mind (and one every night, and on a "
        "clean stop); default 600",
    )
    life_group.add_argument(
        "--life-backdate-days",
        type=int,
        default=0,
        help="a new duck is hatched this many days ago and lives them (awake) at once: it starts at "
        "growth day N and grows on in real time from there (蛋黃: 48, the explorer)",
    )
    life_group.add_argument(
        "--life-birth",
        default=None,
        help="TESTS ONLY: a new duck's birth (ISO time with offset), default now",
    )
    neuro.add_argument(
        "--robotd-head-alpha",
        type=float,
        default=None,
        help="robotd's [control] head_alpha for this simulator's own robotd (written into its "
        "robotd.toml; nothing upstream changes). --brain neuro default 1.0 (owner 2026-09-28: "
        "「加快」, A100-validated); 0.2 = robotd's own default",
    )
    voice_group = parser.add_argument_group(
        "live voice (Gemini Live)",
        "--live-voice: the neural duck listens and looks through the cloud and the page gets its "
        "talk panel (the cloud switch is OFF at start; the page turns it on)",
    )
    voice_group.add_argument(
        "--live-voice",
        action="store_true",
        help="needs --brain neuro --live; the key is read from the Keychain "
        "(service duckmind-gemini-api-key) or $GEMINI_API_KEY when the cloud is switched on",
    )
    voice_group.add_argument(
        "--live-voice-name",
        default=None,
        help="the name the cloud brain knows the duck by (default: the duck's own name on the page)",
    )
    voice_group.add_argument(
        "--live-voice-languages",
        default=",".join(TRANSCRIPTION_LANGUAGES),
        help="the languages the transcriber is told the family speaks (BCP-47, the household's own "
        "first; default zh-TW,en-US). Without the hint the service guesses a language for each "
        "sentence (simplified characters, Spanish words); '' turns the hint off",
    )
    voice_group.add_argument(
        "--live-voice-cap-minutes",
        type=float,
        default=None,
        help="turn the cloud off by itself after this many minutes of microphone audio were sent",
    )
    neuro_api.add_arguments(parser)  # (--neuro-stub: the prototype's page, made-up numbers)
    args = parser.parse_args(argv)
    # (--name given: the neural duck's card shows it; not given: each duck keeps its own name)
    args.name_given = (args.name or "").strip()
    if args.name is None:
        args.name = "小鴨"
    if args.brain == "neuro" and not args.live:
        parser.error("--brain neuro drives the live body: add --live")
    if args.live_voice and not (args.brain == "neuro" and args.live):
        parser.error("--live-voice talks to the neural duck's live body: add --brain neuro --live")
    if args.live_voice_cap_minutes is not None and not args.live_voice_cap_minutes > 0:
        parser.error("--live-voice-cap-minutes must be above 0")
    if args.robotd_head_alpha is not None:
        if not 0.0 < args.robotd_head_alpha <= 1.0:
            parser.error("--robotd-head-alpha must be in (0, 1]")
        os.environ["DUCKMIND_ROBOTD_HEAD_ALPHA"] = repr(float(args.robotd_head_alpha))
    elif args.brain == "neuro":
        # the neural duck's head at robotd's fast setting (fid:motor; the motor shapes big moves)
        os.environ.setdefault("DUCKMIND_ROBOTD_HEAD_ALPHA", "1.0")
    if args.brain == "neuro" and args.live:
        # the trained motions (2026-10-05): the servo command delay the skill networks were trained
        # with, and robotd's table of them when its networks are the gated ones (skilltrain.motion_table)
        from duckmind.skilltrain import motion_table

        installed = motion_table.install_defaults()
        print(
            f"動作試玩：馬達指令延遲 {installed['delay_ms']} 毫秒；訓練過的動作表："
            + (installed["skills"] or f"沒有裝（{installed['why']}）"),
            file=sys.stderr,
        )
    if args.microduck_mjcf:
        os.environ["DUCKMIND_MICRODUCK_MJCF"] = args.microduck_mjcf
    if args.microduck and (args.brain == "neuro" or args.neuro_stub):
        os.environ["DUCKMIND_MICRODUCK"] = str(args.microduck)  # (the prototype's voice stream)
    bridge = _live_bridge(args) if args.live else None
    state_file = args.state_file if str(args.state_file) not in {"", "."} else None
    # A life (--life FOLDER): one duck's record, its calendar clock, Duck Core's save in the
    # folder and the neural duck's snapshot beside it (docs/design/GROWTH-G1-2026-10-10.md).
    life, new_config = _open_life(args, parser)
    if life is not None:
        state_file = life.core_path
        args.seed, args.name, args.name_given = life.seed, life.name, life.name
    # The home: --home, else the layout the editor applied last (kept in the layouts folder), else the
    # environment's, else default_home. What 「回到預設」 goes back to is the first and the last of those.
    layouts_dir = args.layouts_dir or (
        life.folder / "layouts" if life is not None else Path.home() / ".duckmind" / "layouts"
    )
    layout_store = layouts.LayoutStore(Path(layouts_dir).expanduser())
    home_default = args.home or os.environ.get("DUCKMIND_LIVE_HOME") or "default_home"
    home_asked = args.home or layout_store.remembered()
    homeview.set_home(home_asked)
    # the duck's one clock (real time unless a test says otherwise): the neural duck's day and
    # night, the room's lights, the world camera's and the duck's-eye light, Duck Core's growth days
    clock = _make_clock(args, neuro=args.brain == "neuro")
    # The duck's voice: upstream's `sounds` binary in the official checkout (live physics or
    # not), each duck's bank rendered once into the data folder's voices/ (owner 2026-09-28).
    voice_bank = voice.VoiceBank.from_microduck(
        args.microduck or os.environ.get("DUCKMIND_MICRODUCK") or None
    )
    # A new duck has its charging nest (owner decision 2026-09-26); a saved duck comes back
    # exactly as it was (an older save has no nest).
    state = SimState(
        new_config
        or Config(seed=args.seed, name=args.name, caps_file=args.expression_caps, nest=True),
        bridge,
        state_file,
        voice_bank=voice_bank,
        life=life,
        clock=clock,
        resume_time=bool(life is not None and (args.clock_speed != 1.0 or args.life_birth)),
        # the neural duck lives this life: Duck Core simulates nothing of its own
        attach_neural=bool(life is not None and bridge is not None and args.brain == "neuro"),
    )
    state.layouts = layout_store
    state.layout_default = home_default
    if bridge is not None:
        bridge.home_asked = home_asked
    if life is not None:
        # the neural duck on screen is the life duck: its seed at its growth day now
        args.neuro_duck = f"{life.seed}:{state.session.core.s.gd}"
    # The prototype's page (neuro_api): it reads the driver below once the body is up. Its
    # voice first, then the picker's four behind it, in the background.
    hub = neuro_api.setup(args, lambda: getattr(bridge, "neuro", None), voice_bank=voice_bank)
    duck_now = hub.voice_duck() if hub.enabled else state.duck_seed()
    if hub.enabled:
        hub.prepare_voices()
    if voice_bank.sounds_bin is not None:
        folder = voice_bank.bank_dir(voice.voice_seed(duck_now))
        print(
            f"叫聲：Pollen 的 Microduck 聲音（{voice_bank.sounds_bin}），這隻小鴨的聲音庫在 {folder}",
            file=sys.stderr,
        )
    else:
        print(f"叫聲：{voice_bank.problem}", file=sys.stderr)
    # (bound before the physics body starts: a port in use never leaves a body behind)
    live_hub = LiveHub()
    server = SimServer((args.host, args.port), make_handler(state, hub, voice_bank, live_hub))
    life_host: Any = None
    if bridge is not None and args.brain == "neuro":
        from duckmind.simweb.neuro_live import NeuroDriver, NeuroSpec, world_extensions

        # the world server also runs the neural duck's world side (its sensor models' truth)
        os.environ.setdefault("DUCKMIND_SIMHOME_EXT", world_extensions())
        spec = NeuroSpec(duck=args.neuro_duck, seed=int(args.seed), record=args.neuro_record)
        if life is not None:
            from duckmind.simweb.life_host import LifeHost

            life_host = LifeHost(life, state, clock, every_s=args.life_snapshot_every)
            state.life_host = life_host
            spec = life_host.spec(spec)
            # the picker offers the life duck itself, so the page can always come back to it
            hub.life_duck = life_host.picker_choice
            # the owner's word on a trick reaches Duck Core too (when the life duck is on screen)
            hub.core_feedback = life_host.feedback
        driver = NeuroDriver(spec)
        # the duck's time (its circadian drive, the room's daylight, the world camera's and the
        # duck's-eye light, Duck Core's growth days): the one calendar clock
        driver.day_clock = clock
        bridge.clock = lambda: driver.day_clock.local()
        driver.start_at = args.neuro_start
        bridge.neuro = driver
        if life_host is not None and life is not None:
            life_host.attach(driver)
            report = life_host.restore()  # what it learnt before the restart comes back
            print(_life_line(life, state, report, life_host.problem), file=sys.stderr)
            life_host.start()
            # the calls it learned, as its live voice library (docs/design/GROWTH-G5-2026-10-11.md):
            # built from Duck Core's voice lineage now, and again whenever it changes
            if life_host.attach_voice(voice_bank) is not None:
                life_host.refresh_voice("boot")
        print(
            f"神經鴨原型：{'、'.join(f'{k}={v}' for k, v in driver.lanes.names.items())}",
            file=sys.stderr,
        )
        if args.live_voice:
            # the live conversation (live voice, LV5): the cloud is OFF until the page turns it on
            from duckmind.simweb import live_voice as live_voice_host

            voice_host = live_voice_host.LiveVoice(
                driver,
                eye_source=bridge,
                name_of=lambda: str(hub.card().get("name") or "小鴨"),
                name=args.live_voice_name,
                caps=live_voice_host.caps_from_minutes(args.live_voice_cap_minutes),
                transcription_languages=[
                    code.strip() for code in args.live_voice_languages.split(",") if code.strip()
                ],
            )
            driver.live_voice = (
                voice_host  # the emotion layer asks it whether a voice was understood
            )
            hub.switch_hooks.append(voice_host.on_switch)
            live_hub.attach(voice_host)
            print(
                "說話與看見：雲端開關預設是關的（頁面上的「說話」面板才能打開）；"
                + (
                    "還沒有接上"
                    if not voice_host.session
                    else f"小鴨在雲端的名字：{voice_host.name}"
                ),
                file=sys.stderr,
            )
    if bridge is not None:
        # The body lives in this duck's home: with its nest or without (蛋黃: mat and desk,
        # owner decision A2), and its head with or without the ToF module.
        config = state.session.config
        tof = bool(config.tof) or (args.brain == "neuro" and args.neuro_tof)
        bridge.configure(nest=bool(config.nest), tof=tof)
        if life is not None:
            bridge.eye_dir = life.eye_dir  # 「用眼睛做」: the floor this duck learnt, kept with it
        elif state_file is not None:
            bridge.eye_dir = state_file.with_suffix(".eye")
        bridge.start()
        print(
            f"真實物理：上游的 robotd --sim + MuJoCo 身體在「{bridge.home or '平地'}」啟動中（通常幾秒鐘）",
            file=sys.stderr,
        )
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("注意：模擬器沒有登入保護，只在信任的網路（例如 tailnet）上開放。", file=sys.stderr)
    print(f"Duck Core 模擬器：http://{args.host}:{args.port}/  （Ctrl+C 結束）")
    # The page's other languages: Duck Core's words are matched back to its engines' templates
    # (the packs build in the background in about 2 s; docs/i18n.md).
    packs.warm()
    i18nwire.literal_index()
    try:
        serve_until_stopped(server, state, life, life_host)
    finally:
        server.server_close()
        if live_hub.voice is not None:  # (the cloud connection first: nothing leaves after this)
            live_hub.voice.close()
        if bridge is not None:
            bridge.stop()
        hub.stop()


def content_mismatch_words(error: SaveContentMismatch, path: Path | None) -> str:
    """Why a save is not opened, in plain words (printed when the server refuses to start)."""
    written, now = error.written, error.now
    return (
        f"存檔 {path} 是用另一個內容版本存的"
        f"（{written.get('engine_rev')} / {written.get('version')} / 表 {written.get('table_hash')}），"
        f"現在的程式是 {now.get('engine_rev')} / {now.get('version')} / 表 {now.get('table_hash')}。"
        "照存檔重播會得到另一隻鴨子，所以不重播，也不覆蓋存檔。"
        "請用存檔當時的程式打開，或先把舊引擎凍結成新的引擎版本（tools/freeze_engine.py）再改數字。"
    )


def serve_until_stopped(
    server: Any, state: SimState, life: LifeRecord | None, life_host: Any
) -> None:
    """Serve until Ctrl+C or SIGTERM (what ``stop-8791.sh`` sends to the server's PID). A life
    keeps what it learnt: the last snapshot of the mind and the last save of Duck Core are written
    before anything is torn down (``tests/test_life_stop_signal.py`` sends the real signal)."""
    if life is not None:
        signal.signal(signal.SIGTERM, _stop_on_term)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if life is not None:
            signal.signal(
                signal.SIGTERM, signal.SIG_IGN
            )  # (a second TERM must not cut the saving short)
        if state.fastforward is not None:
            state.fastforward.shutdown()  # (no test duck's worker outlives the server)
        if life_host is not None:
            life_host.stop()
        elif life is not None:
            with state.lock:
                state.save()


def bridge_mismatch_words(error: SaveBridgeMismatch, path: Path | None) -> str:
    """Why a save is not opened: it was fed with a newer neural-duck vocabulary."""
    return (
        f"存檔 {path} 是用神經鴨輸入版本 {error.written} 存的，現在的程式只認得 {error.now}。"
        "照存檔重播會漏掉或讀錯牠的經歷，所以不重播，也不覆蓋存檔。請用存檔當時（或更新）的程式打開。"
    )


def _stop_on_term(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def _make_clock(args: argparse.Namespace, *, neuro: bool) -> DuckClock:
    """The duck's one clock: the real calendar, or (tests) ``--neuro-time`` today at that time,
    at ``--clock-speed``."""
    speed = float(args.clock_speed)
    if not speed > 0.0:
        raise SystemExit("--clock-speed must be above 0")
    clock: DuckClock
    if neuro:
        from duckmind.simweb.neuro_live import DayClock, parse_day_time

        minute = parse_day_time(args.neuro_time) if args.neuro_time else None
        clock = DayClock(minute, speed=speed)
    else:
        clock = DuckClock(speed=speed)
    return clock


def _open_life(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[LifeRecord | None, Config | None]:
    """The duck record of ``--life`` (hatched now when the folder is new) and the Config to
    start its Duck Core with (None: not a life)."""
    if args.life is None:
        return None, None
    if args.neuro_time:
        parser.error("--neuro-time is for tests; a life lives by the calendar (use --clock-speed)")
    folder = Path(args.life).expanduser()
    try:
        if (folder / "life.json").is_file():
            life = LifeRecord.load(folder)
            if args.name_given and args.name_given != life.name:
                print(
                    f"注意：這個資料夾的鴨子叫「{life.name}」，--name {args.name_given} 不用。",
                    file=sys.stderr,
                )
            if int(args.seed) != life.seed and "--seed" in sys.argv:
                print(
                    f"注意：這隻鴨子的 seed 是 {life.seed}，--seed {args.seed} 不用。",
                    file=sys.stderr,
                )
            if (not args.no_nest) != life.nest or bool(args.tof) != life.tof:
                print(
                    "注意：窩和 ToF 是牠孵化時定的（存在 life.json），這次的 --no-nest／--tof 不改牠。",
                    file=sys.stderr,
                )
        else:
            from duckmind.simweb.life_host import BRANCH_FROM_DUCK_CORE

            birth = datetime.fromisoformat(args.life_birth) if args.life_birth else None
            life = LifeRecord.create(
                folder,
                name=args.name_given or args.name or "小鴨",
                seed=int(args.seed),
                birth=birth,
                nature=_nature_of(int(args.seed), str(args.neuro_duck or "")),
                nest=not args.no_nest,
                tof=bool(args.tof),
                eye_from=args.life_eye_from,
                backdate_days=int(args.life_backdate_days),
                # (attached from its first moment: Duck Core's branch is its real care's)
                branch_from_core=bool(
                    getattr(args, "brain", "") == "neuro" and BRANCH_FROM_DUCK_CORE
                ),
            )
            print(
                f"新的一生：{life.name}（seed {life.seed}）在 {life.birth.isoformat()} 孵化，資料夾 {folder}",
                file=sys.stderr,
            )
    except LifeError as error:
        raise SystemExit(f"--life：{error}") from error
    config = Config(
        seed=life.seed,
        name=life.name,
        tof=life.tof,
        nest=life.nest,
        caps_file=args.expression_caps,
        start=life.birth.isoformat(),
    )
    return life, config


def _nature_of(seed: int, named: str) -> str:
    """The character a new life grows from (its seed's nature; the preset name when one was given)."""
    if named:
        return named
    from duckmind.neuro.genome import Genome

    return Genome.from_seed(seed, gd=0).primary


def _life_line(
    life: LifeRecord, state: SimState, report: dict[str, Any] | None, problem: str | None
) -> str:
    core = state.session.core
    text = (
        f"一生：{life.name}，成長日 {core.s.gd}（{core.stage().stage_id}），"
        f"孵化 {life.birth.isoformat()}，Duck Core 時間 {core.s.now.isoformat()}"
    )
    if report is None:
        text += "；神經鴨的快照：沒有（乾淨的心）"
    else:
        text += (
            f"；神經鴨的快照接回來了：{report['people']} 個人、{report['places']} 個地點、"
            f"{report['things']} 樣東西、{report['avoided']} 個要避開的地方，離開了 {report['down_s']} 秒"
        )
    if problem:
        text += f"；快照有問題（已收到一旁，沒有覆蓋）：{problem}"
    return text


def _live_bridge(args: argparse.Namespace) -> LiveBridge:
    from duckmind.livebody import LivePaths
    from duckmind.simweb.live import LiveBridge

    paths = LivePaths.from_env()
    if args.microduck:
        if args.ort_lib:
            ort = args.ort_lib
        else:
            from duckmind.sim_live import _installed_ort_library

            ort = _installed_ort_library()
        paths = LivePaths(args.microduck, args.microduck_rl, args.policies, args.robotd_bin, ort)
    if paths is None:
        raise SystemExit("--live needs --microduck, --microduck-rl, --policies and --robotd-bin")
    missing = paths.missing()
    if missing:
        raise SystemExit("--live is missing: " + ", ".join(missing))
    # The page reads the same home layout the body's world runs; a home built on an official
    # scene (the apartment) needs the official MJCF folder here too: the live checkout's, when
    # --microduck-mjcf did not name one.
    from duckmind.simweb.live import official_mjcf

    folder = official_mjcf(paths)
    if not os.environ.get("DUCKMIND_MICRODUCK_MJCF") and folder.is_dir():
        os.environ["DUCKMIND_MICRODUCK_MJCF"] = str(folder)
    return LiveBridge(paths)  # (started once the duck is known: its home and head)


if __name__ == "__main__":
    main()
