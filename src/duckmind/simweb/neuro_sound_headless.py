"""a:sound — the whole neural duck headless, with the sound skills (tests and evidence).

The live driver's tick without its sockets and threads, on the simulation clock (fid:fix's
``headless_neuro`` rig, plus a:sound): the official MJCF in a composed home through
:class:`~duckmind.world.headless_duck.HeadlessDuck` (robotd's control path, the shipped velstand);
the world side of the senses (:mod:`~duckmind.world.neuro_world`) and of the sound skills
(:mod:`~duckmind.world.neuro_sound`: music with a tempo, the hovering hand, the ride); the ToF
module when the duck has one (:class:`~duckmind.world.tof_model.TofSensor` at tofd's 15 Hz: its
raw frames to the senses' DS-v1 summary and to robotd's theremin); the senses, the brain with the
skills (:class:`~duckmind.neuro.skill_sound.SkillBrain`), the motor, the body link and robotd's
streamed voice (:class:`~duckmind.simweb.neuro_voice_stream.VoiceEngine`, its samples kept in a
stream for the evaluators).

Truth (the trunk, the joints, the hand, the song's beats) is read only by :meth:`SoundDuck.truth`
and the evaluators, never by the duck.
"""

from __future__ import annotations

import math
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from duckmind import paths

TOF_PERIOD_S = 1.0 / 15.0  # tofd --hz 15


class SoundDuck:
    """One neural duck with the sound skills in a headless home (see the module docstring)."""

    def __init__(
        self,
        genome: Any,
        seed: int,
        *,
        x: float = 0.0,
        y: float = -1.5,
        yaw: float = math.pi / 2,
        home: str = "default_home",
        keyframe: str = "STAND",
        tof: bool = False,
        day_minute: float = 660.0,
        engine: bool = True,
        voice_rng: int | None = None,
    ) -> None:
        from duckmind.neuro import motor as M
        from duckmind.neuro import skill_sound as SS
        from duckmind.neuro.bodylink import BodyLink, config_for_motor
        from duckmind.neuro.brain import NeuroBrain
        from duckmind.simweb import neuro_voice_stream as V
        from duckmind.world import headless_duck as HD
        from duckmind.world import lights as L
        from duckmind.world import neuro_senses as ns
        from duckmind.world import neuro_sound as nsd
        from duckmind.world import neuro_world as nw
        from duckmind.world import tof_model as T

        if not SS.installed():
            raise RuntimeError("skill_sound.install() must run before the genome is made")
        mjcf = str(paths.microduck_mjcf_dir())
        self.duck = HD.HeadlessDuck(home, mjcf_dir=mjcf, keyframe=keyframe, x=x, y=y, yaw=yaw)
        d = self.duck
        home_l = L.HomeLights.from_layout(d.layout)
        when = datetime(2026, 12, 20, int(day_minute // 60), int(day_minute % 60))
        self._light = L.home_light_state(when, home_l)
        self.world_part = nw.attach(d, seed=seed, lights=lambda: self._light)
        self.sound_part = nsd.attach(d, self.world_part, seed=seed)
        ok, _ = d.stand(timeout=10.0, hold=1.0)
        self.stood = ok
        self.hub = ns.SenseHub(seed=seed)
        self.brain = SS.SkillBrain(NeuroBrain(genome, seed=seed))
        profile = M.profile_from_genome(genome, head_alpha=float(HD.HEAD_ALPHA))
        self.motor = M.MotorSystem(profile, seed)
        self._pose_to_offsets = M.pose_to_offsets
        self.link = BodyLink(config_for_motor(M.COMMAND_MAX))
        self.tof = T.TofSensor(d.model, seed=seed) if tof else None
        self.tof_frame: dict[str, Any] | None = None
        self._next_tof = 0.0
        self.stream = V.SoundStream()
        self.engine = (
            V.VoiceEngine(duck_seed=int(genome.seed), stream=self.stream, rng_seed=voice_rng)
            if engine
            else None
        )
        self.last: Any = None
        self.section: Any = None
        self.sound_section: Any = None
        self.ticks = 0
        self.day_minute = day_minute
        self.t0_ns: int | None = None
        self.cmd: Any = None
        self.tel: Any = None
        self.out: Any = None
        self.tick_out: Any = None
        self.vocals: list[tuple[float, str, str]] = []  # (sim s, tag, behaviour)
        self.dropped = 0

    # ── the world's events (as the page's buttons make them) ──

    def world(self, do: str, **params: Any) -> dict[str, Any]:
        return self.sound_part.op(None, {"do": do, **params})

    def stim(self, stim: str, **params: Any) -> dict[str, Any]:
        return self.world_part.stimulate({"stim": stim, **params})

    # ── one tick ──

    def step(self) -> None:
        from duckmind.duckbody.robotd_client import RobotState
        from duckmind.neuro.bodylink import BodyRequest
        from duckmind.simweb.neuro_live import _events_of

        d = self.duck
        d.tick()
        now = d.now()
        if self.section is None or self.ticks % 10 == 0:  # the world's truth at 5 Hz, as live
            self.world_part.pump()
            with d.world.lock:
                self.section = self.world_part.snapshot(d.world)
                self.sound_section = self.sound_part.snapshot(d.world)
        if self.tof is not None and now + 1e-9 >= self._next_tof:
            self._next_tof = now + TOF_PERIOD_S
            raw = dict(self.tof.capture(d.world))
            raw["t_ns"] = d.t_ns()
            self.tof_frame = raw
        state = RobotState.from_wire(d.state())
        if self.t0_ns is None:
            self.t0_ns = state.t_ns
        t_ms = (state.t_ns - self.t0_ns) // 1_000_000
        doors = SimpleNamespace(
            t_ms=t_ms,
            state=state,
            link=self.last,
            events=_events_of(self.last),
            depth=None,
            tof=self.tof_frame,
            pg=None,
            day_minute=self.day_minute + t_ms / 60000.0,
            world={"neuro": self.section, "neuro_sound": self.sound_section},
        )
        frame = self.hub(doors)
        self.brain.light = _light(self.last)
        cmd, tel = self.brain.step(t_ms, frame)
        out = self.motor.step(cmd, frame)
        head = tuple(float(v) for v in out.head)
        sound = self.brain.last
        if sound is not None and sound.dancing:
            bob = self._pose_to_offsets(sound.bob)
            head = tuple(a + b for a, b in zip(head, bob, strict=True))
        request = BodyRequest(
            vx=float(out.vx),
            vyaw=float(out.vyaw),
            head=(head[0], head[1], head[2], head[3]),
            mouth=float(out.mouth),
            posture=str(out.posture),
            approach=bool(cmd.approach),
            docking=bool(cmd.docking),
            back_arc=bool(cmd.back_arc),
        )
        lo = self.link.tick(state, request, depth=None, battery_pct=None)
        self.last = lo
        mouth_override: float | None = None
        if self.engine is not None and sound is not None:
            tof_age = None
            if self.tof_frame is not None:
                tof_age = max(0.0, (state.t_ns - int(self.tof_frame["t_ns"])) / 1e9)
            owned = self.engine.last is not None and self.engine.last.owns_pcm
            if out.sound and owned:
                self.dropped += 1  # robotd: a stream has the PCM, the one-shot is dropped
            elif out.sound:
                self.vocals.append((now, str(out.sound), cmd.behaviour))
            music = (self.sound_section or {}).get("music")
            self.tick_out = self.engine.tick(
                t_s=t_ms / 1000.0,
                sim_s=now,
                w=now,
                wheee_hold=sound.wheee_hold,
                theremin_up=sound.theremin_up,
                tof=self.tof_frame,
                tof_age_s=tof_age,
                room=music,
            )
            mouth_override = self.tick_out.mouth
        elif out.sound:
            self.vocals.append((now, str(out.sound), cmd.behaviour))
        for intent in lo.intents:
            method, p = intent.method, dict(intent.params or {})
            if method == "robot.move":
                d.move(float(p.get("vx", 0.0)), float(p.get("vy", 0.0)), float(p.get("vyaw", 0.0)))
            elif method == "robot.head":
                names = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
                d.head([float(p.get(n, 0.0)) for n in names])
            elif method == "robot.mouth" and mouth_override is None:
                d.mouth(float(p.get("open", 0.0)))
            elif method == "robot.do":
                d.do(str(p.get("skill", "")))
        if mouth_override is not None:
            d.mouth(mouth_override)
        self.cmd, self.tel, self.out = cmd, tel, lo
        self.ticks += 1

    def run(self, seconds: float, each: Any = None) -> None:
        for _ in range(int(round(seconds * 50))):
            self.step()
            if each is not None and each(self):
                return

    # ── truth, for the evaluation only ──

    def truth(self) -> dict[str, Any]:
        return dict(self.duck.truth())

    def head_joints(self) -> tuple[float, float, float, float]:
        """The measured head joints (rad offsets from HOME: robot.state's own reading)."""
        from duckmind.duckbody.robotd_client import RobotState

        state = RobotState.from_wire(self.duck.state())
        return tuple(float(v) for v in state.head_offsets)  # type: ignore[return-value]

    def fallen(self) -> bool:
        return bool(self.duck.safety.fallen) or self.duck.tilt_deg() > 40.0


def _light(link: Any) -> bool:
    """duck-body's unsure pickup states in the body link's last output: the legs unloaded, or
    swinging free (its ``airborne`` gate)."""
    if link is None:
        return False
    view = link.prox.view
    return bool(view.unloaded or view.lifted or "airborne" in link.gate)


__all__ = ["SoundDuck"]
