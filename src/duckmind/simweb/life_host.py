"""The server's side of a duck's life (docs/design/GROWTH-G1-2026-10-10.md).

``--life FOLDER`` makes this server live one duck, kept in that folder (:mod:`duckmind.life.record`):

* the **clock** is the duck's calendar clock (:mod:`duckmind.life.clock`), real time unless a test
  says otherwise;
* **Duck Core** is the duck's growth: its save is replayed on a restart and it lives on to the
  calendar (:meth:`~duckmind.simweb.server.SimState.grow`), so its growth day is the calendar's;
* the **neural duck** is built from Duck Core's state at that growth day
  (:meth:`duckmind.neuro.genome.Genome.from_core_state`), and what it has learnt is saved as a
  snapshot every night (at the calendar's midnight), every ten minutes and when the server stops
  cleanly, and brought back on the next start;
* people's names and looks live in the folder only (mode 0700 / 0600) and are never sent anywhere.

While the page shows another duck of its picker (a demo duck), nothing is saved over the life duck's
snapshot: the life duck's mind is saved at the moment the page leaves it and is brought back when the
page returns to it (the seed and growth day of the life duck, picked again).

Growth G3 (docs/design/GROWTH-G3-2026-10-11.md): the neural duck's body grows with Duck Core. The
genome it runs is built from a **basis** (:class:`duckmind.life.regrow.Basis`: the growth day, the
branch its body follows, the bond and the preferences it has learnt). When a growth day has passed
a new basis is offered; it is put into the brain **while the duck sleeps calmly, never awake**
(:meth:`regrow_tick`, called by the driver before the brain steps). The basis in force is written
into the mind's snapshot, so a restart builds exactly the genome the duck had.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from collections.abc import Callable
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from duckmind.duckcore.api import dispatch
from duckmind.i18n import msg
from duckmind.life import growth_view as gv
from duckmind.life import regrow as rg
from duckmind.life import snapshot as snap
from duckmind.life.clock import DuckClock
from duckmind.life.experience import Experiences
from duckmind.life.record import LifeRecord
from duckmind.simweb import neuro_api

if TYPE_CHECKING:
    from duckmind.simweb.neuro_live import NeuroDriver, NeuroSpec
    from duckmind.simweb.server import SimState

# G1 (docs/design/GROWTH-G1-2026-10-10.md, section 7): Duck Core lived a shadow life of its own (no
# family, no care, naps of its own), and the "own experience" it counted from that life (naps →
# dreamer) would decide the branch at growth day ~48 instead of the neural duck's nature. G2
# (docs/design/GROWTH-G2-2026-10-10.md) ends the shadow life: Duck Core is given what the neural duck
# lives, and its branch is nature plus the care it really got (the care gate passed: the same seed
# grows into different branches under different care). So a life hatched from now on follows Duck
# Core's branch: this is the default of a NEW record (``LifeRecord.branch_from_core``). A record that
# already exists keeps what it says (absent: False, the branch its seed's nature draws, explorer for
# 蛋黃, as Genome.from_seed draws it) until the planner changes that record
# (``python -m duckmind.life.upgrade FOLDER --branch-from-core on``).
BRANCH_FROM_DUCK_CORE = True

SNAPSHOT_EVERY_S = 600.0  # every ten minutes
WATCH_EVERY_S = 2.0


class LifeHost:
    """One duck's life on a server: the record, the clock, Duck Core (``state``) and the neural
    driver, and the snapshot of the driver's mind."""

    def __init__(
        self,
        record: LifeRecord,
        state: SimState,
        clock: DuckClock,
        *,
        every_s: float = SNAPSHOT_EVERY_S,
        wall: Callable[[], float] = time.time,
        watch_s: float = WATCH_EVERY_S,
    ) -> None:
        self.record = record
        self.state = state
        self.clock = clock
        self.every_s = float(every_s)
        self.watch_s = float(watch_s)
        self._wall = wall
        self.driver: NeuroDriver | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_saved_wall = 0.0
        self._last_day: date | None = None
        # a snapshot file that could not be used, so nothing may overwrite it (set aside instead)
        self.restored: dict[str, Any] | None = None
        self.problem: str | None = None
        self.saves = 0
        self.last_why = ""
        # why the mind is not being saved, when it is not (the page shows a demo duck, or a save
        # failed): kept for /api/life and said once in the log, never silent
        self._problem_kind: str | None = None  # None | "off_screen" | "save"
        self._off_screen_since: float | None = None
        self._fail_logged_wall = -1e18
        self._retry_at_wall = 0.0
        self._picker_gd = -1
        self.saved_calendar: str | None = None  # the duck's calendar at the last save
        # what the neural duck lived, on its way to Duck Core (docs/design/GROWTH-G2-2026-10-10.md)
        self.experiences = Experiences()
        self._obs_lock = threading.Lock()
        self._pending: list[dict[str, Any]] = []
        self._pending_lock = threading.Lock()
        self.delivered = 0
        self.refused = 0
        self.deliver_problem: str | None = None
        # growth G3: the basis of the genome on screen, what is waiting to be grown into, and what
        # the last regrows were
        with state.lock:
            self._static = rg.Static.of(state.session.core.s)
        self._nature_branch: str | None = None
        self._stamp: rg.Basis | None = self._read_stamp()
        self.regrower: rg.Regrower | None = None
        self.regrows = 0
        self.regrow_history: list[dict[str, Any]] = []
        self._regrow_snap = ""  # a snapshot owed after a regrow (the watch thread writes it)
        self._regrow_said_wait = 0.0
        self.regrow_problem: str | None = None
        # the lineage voice (docs/design/GROWTH-G5-2026-10-11.md): the calls the duck learned, as the
        # clips the page plays; made again when the lineage changes (boot, a day change, the nightly regrow)
        self.voice_lineage: Any = None
        # the growth page's picture (growth G4): the experiences counted from Duck Core's inputs
        # (read once, from where it stopped), and the last picture made (a page asks every ~15 s)
        self._tally = gv.Tally()
        self._growth_cache: tuple[float, dict[str, Any]] | None = None

    # ── who the neural duck is ───────────────────────────────────────────────

    def growth_day(self) -> int:
        return int(self.state.session.core.s.gd)

    def spec_string(self) -> str:
        return f"{self.record.seed}:{self.growth_day()}"

    # ── the genome: Duck Core's state as a basis, grown into while asleep ───────────────────

    def _read_stamp(self) -> rg.Basis | None:
        """The basis the mind on disk was running when it was last saved (None: none, or it is
        not one this build trusts)."""
        try:
            data = snap.read(self.record.neural_path)
        except snap.SnapshotUnusable:
            return None
        return rg.Basis.from_json((data or {}).get("regrow"))

    def nature_branch(self) -> str | None:
        """The branch its seed's nature draws (what 蛋黃's body follows while its record pins it)."""
        if self._nature_branch is None:
            from duckmind.neuro.genome import Genome

            # (drawn as the grown duck: before the juvenile stage Genome.from_seed gives no branch)
            self._nature_branch = Genome.from_seed(self.record.seed, gd=400).branch or ""
        return self._nature_branch or None

    def live_basis(self) -> rg.Basis:
        """Duck Core's state now as a basis. The branch is the state's own for a record that
        follows Duck Core (``branch_from_core``), else the one its nature draws (蛋黃: explorer)."""
        with self.state.lock:
            core_state = self.state.session.core.s
            branch = core_state.branch if self.record.branch_from_core else self.nature_branch()
            return rg.basis_of(core_state, branch=branch)

    def build_basis(self) -> tuple[rg.Basis, str]:
        """The basis a brain built now is given, and why: the one the mind was saved with when it
        is still of today's growth day (a restart gives back the genome the duck had: no jump
        between last night's regrow and this boot), else Duck Core's state now (a night passed
        while it was away: the regrow it missed)."""
        live = self.live_basis()
        stamp = self._stamp
        current = self.regrower.applied if self.regrower is not None else None
        for held, why in ((current, "kept"), (stamp, "snapshot")):
            if held is None or held.gd != live.gd:
                continue
            if not self.record.branch_from_core and held.branch != live.branch:
                continue  # (the record's pin changed: the body follows the record)
            return held, why
        return live, "state"

    def genome_builder(self) -> Callable[[], Any]:
        """The genome the neural brain is built from (called once the skills are installed, in
        :func:`~duckmind.simweb.neuro_live.load_lanes`): its basis as above."""

        def build() -> Any:
            basis, why = self.build_basis()
            genome = self._static.genome(basis)
            self.regrower = rg.Regrower(self._static, basis)
            self._regrow_snap = ""
            if why == "state" and self._stamp is not None and self._stamp.gd != basis.gd:
                self._say_regrow(
                    f"開機：睡著時的重新長出錯過了（上次存的是成長日 {self._stamp.gd}，現在是 {basis.gd}），"
                    f"用 Duck Core 現在的狀態長出成長日 {basis.gd} 的身體。"
                )
            self._register_card(genome)
            return genome

        return build

    def _register_card(self, genome: Any) -> None:
        neuro_api.register_life_summary(genome.seed, genome.gd, genome.summary())

    def picker_choice(self) -> Any:
        """The life duck as the page's picker offers it: its seed at its growth day now, under its
        own name; its card is built from Duck Core's state (so it says what the brain that runs
        is). Choosing it brings the life duck back on screen, with its mind restored."""
        from duckmind.simweb.neuro_api import LIFE_PRESET, DuckChoice

        gd = self.growth_day()
        if self._picker_gd != gd:
            # (registers its card at this growth day: the genome it would have now)
            self._register_card(self._static.genome(self.build_basis()[0]))
            self._picker_gd = gd
        return DuckChoice(
            self.record.seed, gd, self.record.name, LIFE_PRESET, msg("neuro.preset.life.title")
        )

    def is_life_duck(self, seed: int | None, gd: int | None) -> bool:
        """Is this seed at this growth day the life duck (the page picked it again)?"""
        return seed == self.record.seed and gd == self.growth_day()

    def spec(self, base: NeuroSpec) -> NeuroSpec:
        """``base`` made the life duck's: its seed at the growth day now, its genome from Duck
        Core's state."""
        from dataclasses import replace

        return replace(
            base,
            duck=self.spec_string(),
            seed=self.record.seed,
            gd=self.growth_day(),
            genome_builder=self.genome_builder(),
            life=True,
        )

    def attach(self, driver: NeuroDriver) -> None:
        self.driver = driver
        driver.life = self

    # ── the voice lineage ────────────────────────────────────────────────────────────────

    def lineage_view(self) -> Any:
        """Duck Core's voice lineage now (copied out under its lock): what the live voice library is
        made from. Read only."""
        from duckmind.life import voice_lineage

        with self.state.lock:
            return voice_lineage.view_of(self.state.session.core.s)

    def attach_voice(self, bank: Any, *, root: Any = None) -> Any:
        """Give the duck a lineage voice on top of ``bank`` (a :class:`~duckmind.simweb.voice.VoiceBank`):
        its learned calls are laid over the bank's in the page's manifest. Returns it (None: numpy or
        the voice cache is missing, the bank's own voice then)."""
        try:
            from duckmind.simweb import lineage_voice
        except ImportError:
            return None
        lineage = lineage_voice.LineageVoice(
            bank,
            self.record.seed,
            self.lineage_view,
            calendar=lambda: self.clock.calendar().isoformat(),
            root=root or lineage_voice.env_root(),
        )
        bank.lineage = lineage
        self.voice_lineage = lineage
        return lineage

    def refresh_voice(self, why: str = "night", *, wait: bool = False) -> str:
        """Look at the voice lineage again: if it sounds different from the library the page has, a
        new library is built and put in between two calls (never in the middle of one). The one call
        for anything that changes the lineage's sound: the nightly regrow calls it with ``"regrow"``
        after the new growth day's numbers are in. Safe from any thread; as often as wanted (a
        lineage that sounds the same builds nothing). Returns ``building`` | ``unavailable``."""
        lineage = self.voice_lineage
        if lineage is None:
            return "unavailable"
        try:
            return str(lineage.refresh(why, wait=wait))
        except Exception:  # (the voice never stops the life)
            traceback.print_exc()
            return "unavailable"

    # ── the experiences ──────────────────────────────────────────────────────────────────

    def on_tick(self, t_ms: int, frame: Any, cmd: Any, tel: Any) -> None:
        """One brain tick of the neural duck on screen (called by the driver on its own thread):
        what really happened to it is queued for Duck Core. A demo duck on screen is not the life
        duck: nothing it lives is given."""
        driver = self.driver
        if driver is None or not getattr(driver.spec, "life", False):
            return
        if not getattr(self.state.session.core.s, "bridged", False):
            return
        t_s = int(self.clock.calendar().timestamp())
        with self._obs_lock:
            events = self.experiences.observe(t_s, frame, cmd, tel)
        if events:
            with self._pending_lock:
                self._pending.extend(events)

    def deliver(self) -> int:
        """(Under the state's lock.) Give Duck Core what the neural duck lived since the last
        time, each at its second, before the clock moves on. A stall (a computer that slept) is
        lived first as the time away it was. Returns the events taken in."""
        with self._pending_lock:
            events, self._pending = self._pending, []
        if not events:
            return 0
        state = self.state
        core = state.session.core
        tz = core.s.now.tzinfo
        taken = 0
        index = 0
        while index < len(events):
            first = int(events[index]["t"])
            if first - int(core.s.now.timestamp()) > 30:
                state.grow(until=datetime.fromtimestamp(first - 1, tz))
            batch = []
            while index < len(events) and int(events[index]["t"]) - first <= 30:
                batch.append(events[index])
                index += 1
            try:
                dispatch(state.session, "experience", {"events": batch})
                taken += len(batch)
            except Exception as error:  # never the body loop's: the events are dropped, said once
                self.refused += len(batch)
                self.deliver_problem = f"{type(error).__name__}: {error}"
                traceback.print_exc()
        self.delivered += taken
        return taken

    # the owner's word on a trick (``POST /api/neuro/feedback``), as the family phone says it to
    # Duck Core: 「乖」, a treat (the handset's 「乖」) and 「不對」
    FEEDBACK_ACTS = {"praise": "owner_good", "treat": "pad_good", "scold": "owner_not_that"}

    def feedback(self, kind: str, trick: str) -> dict[str, Any] | None:
        """The owner's 「乖」, treat or 「不對」 for ``trick`` (the brain has just taken it): Duck Core
        hears the phone's word (:data:`FEEDBACK_ACTS`) and bridge-v2's ``praise`` / ``scold`` for the
        trick, at the calendar's second, after what the neural duck lived until now. None: the life
        duck is not the one on screen (a demo duck's lessons are not its) or Duck Core is not
        attached. The answer is codes, not words (what each input made of it). The brain has stepped
        the trick's proficiency already, so the 「成長」 picture kept for a few seconds is dropped: the
        next ask shows the new proficiency at once."""
        if self._brain() is None:
            return None
        self._growth_cache = None
        state = self.state
        with state.lock:
            core = state.session.core
            if not getattr(core.s, "bridged", False):
                return None
            self.deliver()
            state.grow()
            before = len(core.log.events)
            action = self.FEEDBACK_ACTS[kind]
            said = dispatch(state.session, "act", {"action": action, "params": {}})
            event = {
                "k": "scold" if kind == "scold" else "praise",
                "t": int(core.s.now.timestamp()),
                "trick": trick,
            }
            taken = dispatch(state.session, "experience", {"events": [event]})["taken"][0]
            state.mark(before, "play")
            state.save(soon=True)
        return {
            "act": action,
            "perceived": bool(said.get("perceived")),
            "event": event["k"],
            "taken": bool(taken.get("taken")),
            "result": taken.get("note"),
        }

    def finish_experiences(self, *, restart: bool = False) -> None:
        """The end of a watch (a clean stop, the page leaving the life duck): what is still open
        is given too. ``restart``: a new watch begins (another brain will be on screen)."""
        t_s = int(self.clock.calendar().timestamp())
        with self._obs_lock:
            events = self.experiences.flush(t_s)
            if restart:
                fresh = Experiences()
                fresh.sent = self.experiences.sent
                self.experiences = fresh
        if events:
            with self._pending_lock:
                self._pending.extend(events)
        with self.state.lock:
            self.deliver()

    def body_replaced(self) -> None:
        """The neural duck has a new physics body (the layout editor's Apply): what was open on the
        old one (a contact, a sleep, a stay at the charger, a fall) ends at its last moment, and the
        new body's first readings (its battery, its sensors) are not compared with the old one's.
        What it lived goes to Duck Core with the next second's news (no lock needed here)."""
        t_s = int(self.clock.calendar().timestamp())
        with self._obs_lock:
            events = self.experiences.flush(t_s)
        if events:
            with self._pending_lock:
                self._pending.extend(events)

    def context(self) -> dict[str, Any]:
        """Where the body starts: places and the nest are in its odometry frame, which is the same
        only when it starts from the same home and place."""
        driver = self.driver
        return {
            "home": self.state.home(),
            "start": getattr(driver, "start_at", "mat") if driver is not None else "mat",
        }

    # ── growing in its sleep (growth G3) ────────────────────────────────────────

    def _stamp_now(self) -> dict[str, Any] | None:
        """What the snapshot records of the genome on screen: its basis, so a restart builds the
        same one (and the regrows so far, for the status)."""
        regrower = self.regrower
        if regrower is None:
            return None
        out = regrower.applied.to_json()
        out["regrows"] = self.regrows
        if self.regrow_history:
            out["last"] = self.regrow_history[-1]
        return out

    def _say_regrow(self, text: str) -> None:
        print(f"成長：{text}", file=sys.stderr, flush=True)

    def regrow_watch(self) -> None:
        """(The watch thread, every couple of seconds.) Is a growth day waiting to be grown into?
        The basis is Duck Core's state now; the body loop never takes the state's lock."""
        regrower = self.regrower
        if regrower is None or self._brain() is None:
            return
        live = self.live_basis()
        if live.gd == regrower.applied.gd:
            regrower.offer(
                regrower.applied
            )  # (nothing to grow into: a day's care waits for the night)
            return
        regrower.offer(live)

    def _driver_busy(self, driver: Any, state: Any) -> str | None:
        """What the driver itself has going that a swap must not cut (a motion trial, an emotion,
        a skill of the body link, a flight)."""
        if getattr(driver, "_motion_run", None) is not None or getattr(driver, "_play_req", None):
            return "motion"
        if getattr(getattr(driver, "emotions", None), "run", None) is not None:
            return "emotion"
        if str(getattr(state, "policy", "sit")) not in ("sit", "stand", "walk"):
            return "skill"
        link = getattr(driver, "_link", None)
        if link is not None and (
            getattr(link, "flight_t0_ns", None) is not None or getattr(link, "pose_on", False)
        ):
            return "flight"
        return None

    def regrow_tick(self, driver: Any, lanes: Any, frame: Any, state: Any, t_ms: int) -> None:
        """(The driver's tick thread, before the brain steps; the cheap path costs one attribute
        read.) Put the waiting basis into the sleeping brain, once it has slept calmly for a
        while. Anything that goes wrong here is said once and never stops the body loop."""
        regrower = self.regrower
        if regrower is None or not regrower.due:
            return
        if not getattr(driver.spec, "life", False):
            return  # (a demo duck is on screen: the life duck's body is not that brain)
        brain = lanes.brain
        if not hasattr(brain, "regrow"):
            return
        try:
            report = regrower.tick(
                brain,
                frame,
                t_ms,
                cmd=getattr(driver, "_last_cmd", None),
                also_calm=self._driver_busy(driver, state),
                calendar=self.clock.calendar().isoformat(),
                why="night",
            )
            if report is not None:
                self._after_regrow(driver, lanes, brain, report)
        except Exception as error:
            self.regrow_problem = f"{type(error).__name__}: {error}"
            self._say_regrow(f"重新長出失敗（沒有換，牠照舊）：{self.regrow_problem}")
            traceback.print_exc()
            regrower._offer = None  # (not every tick: the next growth day tries again)

    def _after_regrow(self, driver: Any, lanes: Any, brain: Any, report: rg.Report) -> None:
        """What follows a swap: the motor's profile, the genome the page shows, the log, the
        snapshot that makes it the duck's."""
        from dataclasses import replace

        genome = brain.genome
        motor = getattr(lanes.motor, "base", lanes.motor)
        regrow_motor = getattr(motor, "regrow", None)
        if callable(regrow_motor):
            regrow_motor(genome)
        lanes.genome = genome.summary()
        self._register_card(genome)
        self._picker_gd = genome.gd
        driver.spec = replace(driver.spec, duck=f"{self.record.seed}:{genome.gd}", gd=genome.gd)
        self.regrows += 1
        self.regrow_problem = None
        view = report.view()
        self.regrow_history.append(view)
        del self.regrow_history[:-8]
        self._regrow_snap = "regrow"
        self.refresh_voice("regrow")  # (growth G5: the lineage may sound different on the new day)
        change = report.change
        self._say_regrow(
            f"睡著的時候重新長出了：成長日 {change['gd'][0]} → {change['gd'][1]}，"
            f"階段 {change['stage'][0]} → {change['stage'][1]}，分支 {change['branch'][0]} → {change['branch'][1]}；"
            f"新解鎖 {len(change['unlocked'])} 個動作、拿掉 {len(change['dropped'])} 個；"
            f"{change['params_moved']} 個參數、{change['gains_moved']} 個動作的力道變了；"
            f"親近度 {change['care']['bond'][0]} → {change['care']['bond'][1]}，學到的偏好 {change['care']['prefs']} 項；"
            f"等了 {view['waited_s']} 秒才睡得夠安穩。"
        )

    # ── the snapshot ─────────────────────────────────────────────────────────

    def _brain(self) -> Any | None:
        driver = self.driver
        if driver is None or not getattr(driver.spec, "life", False):
            return None  # (the page shows a demo duck: the life duck's mind is not on screen)
        brain = getattr(driver.lanes, "brain", None)
        return brain if hasattr(brain, "learn") and hasattr(brain, "hab") else None

    def on_screen(self) -> bool:
        """Is the life duck's own mind the one on screen (not a demo duck of the page's picker)?"""
        return self._brain() is not None

    def _talk(self) -> float | None:
        voice = getattr(self.driver, "live_voice", None)
        bridge = getattr(voice, "bridge", None)
        talk = getattr(bridge, "talk", None)
        return None if talk is None else float(talk.value)

    def _set_talk(self, value: float) -> None:
        voice = getattr(self.driver, "live_voice", None)
        talk = getattr(getattr(voice, "bridge", None), "talk", None)
        if talk is not None:
            talk.value = min(1.0, max(0.0, float(value)))

    def restore(self) -> dict[str, Any] | None:
        """Bring the saved mind back into the brain on screen (a fresh brain). The file that
        cannot be used is set aside, never overwritten; the duck then starts with a clean mind."""
        driver = self.driver
        brain = self._brain()
        if driver is None or brain is None:
            return None
        with driver._tick_lock, self._lock:
            data: dict[str, Any] | None = None
            try:
                data = snap.read(self.record.neural_path)
            except snap.SnapshotUnusable as error:
                self.problem = str(error)
                prev = None
                try:
                    prev = snap.read(self.record.neural_prev_path)
                except snap.SnapshotUnusable:
                    pass
                snap.set_aside(self.record.neural_path, str(error))
                data = prev
            if data is None:
                return None
            try:
                report = snap.apply(
                    brain,
                    data,
                    seed=self.record.seed,
                    context=self.context(),
                    wall=self._wall(),
                    set_talk=self._set_talk,
                )
            except snap.SnapshotUnusable as error:
                self.problem = str(error)
                if self.record.neural_path.is_file():
                    snap.set_aside(self.record.neural_path, str(error))
                return None
            self.restored = report
            return report

    def snapshot_now(self, why: str) -> bool:
        """Save the mind of the life duck on screen. False: not now — a demo duck is shown (the
        life duck's mind is not on screen: said in :meth:`status` and in the log), or no brain yet,
        or the save failed (said the same way). Safe to call from any thread."""
        driver = self.driver
        brain = self._brain()
        if driver is None or brain is None:
            self._not_on_screen(why)
            return False
        try:
            with driver._tick_lock, self._lock:
                calendar = self.clock.calendar()
                data = snap.capture(
                    brain,
                    name=self.record.name,
                    seed=self.record.seed,
                    gd=self.growth_day(),
                    calendar=calendar,
                    context=self.context(),
                    why=why,
                    wall=self._wall(),
                    talk=self._talk(),
                    regrow=self._stamp_now(),
                )
                snap.write(self.record.neural_path, data, self.record.neural_prev_path)
                if self.regrower is not None:
                    self._stamp = self.regrower.applied
                self._last_saved_wall = self._wall()
                self.saved_calendar = calendar.isoformat()
                self.saves += 1
                self.last_why = why
        except Exception as error:
            self._save_failed(why, error)
            return False
        self._saved_ok()
        return True

    def _last_saved_text(self) -> str:
        """When the mind on disk was saved (the duck's calendar), for the words."""
        return self.saved_calendar or str((self.restored or {}).get("saved_calendar") or "還沒有")

    def _say(self, text: str) -> None:
        print(f"神經鴨快照：{text}", file=sys.stderr, flush=True)

    def _not_on_screen(self, why: str) -> None:
        """The mind to save is not on screen. Not an error (a demo duck's mind is never the life
        duck's), but never silent: the status says so from the first moment and the log once."""
        name = self.record.name
        if self._problem_kind != "off_screen":
            self._off_screen_since = self._wall()
            self._say(
                f"頁面現在顯示的不是{name}（是示範鴨或還沒有腦），所以現在存不了{name}的心（要存的是「{why}」）；"
                f"最後一次存是 {self._last_saved_text()}（{self.last_why or '-'}）。"
                f"從選單選回「{name}」就會接著存。"
            )
        self._problem_kind = "off_screen"
        self.problem = self._off_screen_words()

    def _off_screen_words(self) -> str:
        name = self.record.name
        return (
            f"頁面顯示的不是{name}，牠的心沒有在存；最後一次存是 {self._last_saved_text()}"
            f"（{self.last_why or '-'}）。從選單選回「{name}」。"
        )

    def _save_failed(self, why: str, error: Exception) -> None:
        text = f"{type(error).__name__}: {error}"
        self._problem_kind = "save"
        self.problem = f"存心智快照失敗（{why}）：{text}"
        self._retry_at_wall = self._wall() + min(30.0, self.every_s)
        if self._wall() - self._fail_logged_wall >= 600.0:
            self._fail_logged_wall = self._wall()
            self._say(f"存不下去（{why}）：{text}；每 30 秒再試一次。")
            traceback.print_exc()

    def _saved_ok(self) -> None:
        if self._problem_kind is not None:
            if self._problem_kind == "off_screen":
                self._say(f"{self.record.name}回到畫面上了，心又接著存。")
            else:
                self._say("又存得下去了。")
            self._problem_kind = None
            self.problem = None
        self._off_screen_since = None
        self._retry_at_wall = 0.0

    def _problem_now(self) -> str | None:
        """What /api/life says is wrong with the saving: the recorded problem, or — whatever the
        watch thread is doing — a mind on screen that has not been saved for three cycles."""
        if self.problem:
            return self.problem
        if self._thread is not None and self._brain() is None:
            return self._off_screen_words()
        if self._brain() is not None and self._thread is not None:
            late = self._wall() - self._last_saved_wall
            if late > 3 * self.every_s + 60.0:
                return f"{self.record.name}的心已經 {int(late)} 秒沒有存了（每 {int(self.every_s)} 秒該存一次）。"
        return None

    def before_switch(self) -> None:
        """The page is about to leave the life duck: keep its mind."""
        try:
            self.finish_experiences(restart=True)
        except Exception:
            traceback.print_exc()
        if self.snapshot_now("switch"):
            self._not_on_screen("switch")  # (said at once: from here the page shows another duck)

    def after_switch(self) -> None:
        """The page came back to the life duck (a fresh brain): bring its mind back."""
        try:
            self.restore()
        except Exception:
            traceback.print_exc()
        if self._brain() is not None and self._problem_kind == "off_screen":
            self._saved_ok()  # (back on screen: the saving goes on; its mind was just read from disk)
            self._last_saved_wall = self._wall()

    # ── the background watch ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the watch. A duck just hatched has its (empty) snapshot on disk from the first
        moment, so the record is whole: name, birth, Duck Core's save and the mind's file."""
        self._last_day = self.clock.calendar().date()
        self._last_saved_wall = self._wall()
        if not self.record.neural_path.exists():
            try:
                self.snapshot_now("hatch")
            except Exception:
                traceback.print_exc()
        self._thread = threading.Thread(target=self._run, name="life-snapshots", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.watch_s):
            try:
                self.regrow_watch()
                owed, self._regrow_snap = self._regrow_snap, ""
                if owed:
                    self.snapshot_now(owed)
                today = self.clock.calendar().date()
                if self._last_day is not None and today != self._last_day:
                    self._last_day = today
                    self.snapshot_now("night")
                    self.refresh_voice("night")
                elif (
                    self._wall() - self._last_saved_wall >= self.every_s
                    and self._wall() >= self._retry_at_wall
                ):
                    if not self.snapshot_now("periodic"):
                        # (not saved: try again in a while, not every two seconds)
                        self._retry_at_wall = max(
                            self._retry_at_wall, self._wall() + min(30.0, self.every_s)
                        )
            except Exception:
                self._save_failed("watch", sys.exc_info()[1] or RuntimeError("watch"))  # type: ignore[arg-type]

    def stop(self) -> None:
        """A clean stop: the last snapshot of the mind and the last save of Duck Core."""
        self._stop.set()
        try:
            self.finish_experiences()
        except Exception:
            traceback.print_exc()
        if not self.snapshot_now("stop"):
            self._say(f"乾淨關機時沒有存到{self.record.name}的心：{self.problem or '沒有原因'}")
        try:
            with self.state.lock:
                self.state.save()
        except Exception:
            traceback.print_exc()

    # ── what the page may ask ────────────────────────────────────────────────

    def _regrow_status(self) -> dict[str, Any]:
        """(Under the state's lock: reads only what the host holds.) Plain numbers."""
        regrower = self.regrower
        applied = None if regrower is None else regrower.applied
        brain = self._brain()
        genome = getattr(brain, "genome", None)
        return {
            "rev": rg.GROWTH_REV,
            "count": self.regrows,
            "growth_day": self.growth_day(),
            "applied_gd": None if applied is None else applied.gd,
            "due": bool(regrower is not None and regrower.due),
            "waiting_for": None if regrower is None or not regrower.due else regrower.refused,
            "problem": self.regrow_problem,
            "last": self.regrow_history[-1] if self.regrow_history else None,
            "history": list(self.regrow_history),
            # the genome the body runs now, in numbers
            "genome": None if genome is None else rg.genome_numbers(genome),
        }

    GROWTH_CACHE_S = 8.0

    def growth_view(self) -> dict[str, Any]:
        """The 「成長」 picture of the duck (docs/design/GROWTH-G4-2026-10-11.md): read only, the
        same shape as a test duck's. Called under the state's lock (Duck Core is read); the last
        picture is kept for a few seconds, a page asks every ~15."""
        now = self._wall()
        cached = self._growth_cache
        if cached is not None and now - cached[0] < self.GROWTH_CACHE_S:
            return cached[1]
        core = self.state.session.core
        pinned = not self.record.branch_from_core
        body_branch = self.nature_branch() if pinned else core.s.branch
        brain = self._brain()
        self._tally.update(self.state.session.inputs)
        with self._obs_lock:
            ignored = dict(self.experiences.ignored)
            places_found = int(self.experiences.places_found)
        known = getattr(brain, "home_known", None)
        home = known() if callable(known) else None
        voice = None if self.voice_lineage is None else self.voice_lineage.status()
        view = gv.build(
            core,
            tally=self._tally,
            name=self.record.name,
            live=True,
            branch_id=body_branch,
            pinned=pinned,
            nature_branch=self.nature_branch(),
            learner=getattr(brain, "learn", None),
            home=home,
            places_found=places_found,
            regrow=self._regrow_status(),
            voice=voice,
            ignored=ignored,
            delivered={"delivered": self.delivered, "refused": self.refused},
            calendar=self.clock.calendar().isoformat(),
        )
        view["on_screen"] = bool(brain is not None)
        self._growth_cache = (now, view)
        return view

    def status(self) -> dict[str, Any]:
        with self._obs_lock:  # (the driver's thread feeds these while the page asks)
            seen = dict(sorted(self.experiences.sent.items()))
            contacts = list(self.experiences.contacts)
            ignored = dict(sorted(self.experiences.ignored.items()))
        core = self.state.session.core
        stage = core.stage()
        return {
            "name": self.record.name,
            "seed": self.record.seed,
            "birth": self.record.birth.isoformat(),
            "calendar": self.clock.calendar().isoformat(),
            "speed": self.clock.speed,
            "growth_day": self.growth_day(),
            "stage": stage.stage_id,
            "duck_core_now": core.s.now.isoformat(),
            "on_screen": bool(self._brain() is not None),
            # (counts only: what the neural duck lived, and gave to Duck Core)
            "bridge": {
                "attached": bool(getattr(core.s, "bridged", False)),
                "delivered": self.delivered,
                "refused": self.refused,
                "problem": self.deliver_problem,
                "seen": seen,
                "contacts": contacts,
                # contacts that were not a hand (things, a person's leg): felt, never told
                "ignored": ignored,
            },
            "regrow": self._regrow_status(),
            "snapshot": {
                "saves": self.saves,
                "last_why": self.last_why,
                "every_s": self.every_s,
                "problem": self._problem_now(),
                # when the mind was last saved (the duck's calendar) and whether it is on screen to
                # be saved now: a demo duck on screen is not a mind to save
                "saved_calendar": self.saved_calendar,
                "on_screen": bool(self._brain() is not None),
                "watching": bool(self._thread is not None and self._thread.is_alive()),
            },
            # the calls it learned, as the live voice library: lineage size, what was learned, the
            # library the page has and when it was last updated (docs/design/GROWTH-G5-2026-10-11.md)
            "voice": None if self.voice_lineage is None else self.voice_lineage.status(),
            # (counts only: the people's names and looks are never put on the wire)
            "restored": None
            if self.restored is None
            else {k: v for k, v in self.restored.items() if k != "names"},
        }
