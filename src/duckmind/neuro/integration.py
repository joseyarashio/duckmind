"""The default single-brain composition for social, play and sound skills."""

from __future__ import annotations

import threading
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Any, Mapping

from duckmind.neuro import command_brain as CB
from duckmind.neuro import contract as C
from duckmind.neuro import genome as G
from duckmind.neuro import select as S
from duckmind.neuro import skill_play as SP
from duckmind.neuro import skill_social_brain as SB
from duckmind.neuro import skill_sound as SS


@dataclass(frozen=True, slots=True)
class SkillRegistry:
    """Stable registry identity captured once before any integrated brain is created."""

    behaviour_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_index: Mapping[str, int]
    feature_map_identity: int
    builder_names: tuple[str, ...]
    weighted_names: tuple[str, ...]
    satiation_names: tuple[str, ...]


_INSTALL_LOCK = threading.Lock()
_REGISTRY: SkillRegistry | None = None


def install_default_skills() -> SkillRegistry:
    """Install all default contributions once; never unload a live registry."""
    global _REGISTRY
    _reject_legacy_fixture_overlap()
    if _REGISTRY is not None:
        _validate_snapshot(_REGISTRY)
        return _REGISTRY
    with _INSTALL_LOCK:
        _reject_legacy_fixture_overlap()
        if _REGISTRY is not None:
            _validate_snapshot(_REGISTRY)
            return _REGISTRY
        _validate_contributions()
        SB.register()
        SP.install_permanent()
        SS.install()
        from duckmind.neuro import repertoire as P

        registry = SkillRegistry(
            behaviour_names=tuple(C.BEHAVIOURS),
            feature_names=tuple(S.FEATURES),
            feature_index=MappingProxyType(dict(S.F)),
            feature_map_identity=id(S.F),
            builder_names=tuple(P.BUILDERS),
            weighted_names=tuple(S.W),
            satiation_names=tuple(S.SATIATE_S),
        )
        _validate_snapshot(registry)
        _REGISTRY = registry
        return registry


def _reject_legacy_fixture_overlap() -> None:
    """Do not let a one-way registry be captured by an active reversible test fixture."""
    active = (
        SP.legacy_context_active(),
        SB.legacy_context_active(),
        _REGISTRY is None and SS.installed(),
    )
    if any(active):
        raise RuntimeError(
            "install_default_skills cannot run inside legacy reversible skill fixtures; "
            f"exit play/social/sound fixture contexts and retry (active={active})"
        )


def _validate_contributions() -> None:
    behaviour_sets = (
        set(SB.SOCIAL_BEHAVIOURS),
        set(SP.PLAY_BEHAVIOURS),
        set(SS.BEHAVIOURS),
    )
    feature_sets = (
        set(SB.SOCIAL_FEATURES),
        set(SP.PLAY_FEATURES),
        set(SS.FEATURES),
    )
    for label, sets in (("behaviour", behaviour_sets), ("feature", feature_sets)):
        seen: set[str] = set()
        for names in sets:
            overlap = seen & names
            if overlap:
                raise RuntimeError(f"duplicate skill {label} contribution: {sorted(overlap)}")
            seen |= names


def _validate_snapshot(registry: SkillRegistry) -> None:
    from duckmind.neuro import repertoire as P

    if (
        tuple(S.FEATURES) != registry.feature_names
        or tuple(C.BEHAVIOURS) != registry.behaviour_names
    ):
        raise RuntimeError("default skill registry changed after integrated brains were created")
    if id(S.F) != registry.feature_map_identity:
        raise RuntimeError("selector feature-index object was replaced after registry installation")
    if len(S.F) != len(S.FEATURES) or any(S.F.get(name) != i for i, name in enumerate(S.FEATURES)):
        raise RuntimeError("selector feature index does not match the installed registry")
    if dict(S.F) != dict(registry.feature_index):
        raise RuntimeError("selector feature index contents changed after registry installation")
    if tuple(P.BUILDERS) != registry.builder_names or tuple(S.W) != registry.weighted_names:
        raise RuntimeError(
            "skill builder or selector weight table changed after registry installation"
        )
    if tuple(S.SATIATE_S) != registry.satiation_names:
        raise RuntimeError("selector satiation table changed after registry installation")
    if any(
        name not in C.BEHAVIOURS
        for name in (*SB.SOCIAL_BEHAVIOURS, *SP.PLAY_BEHAVIOURS, *SS.BEHAVIOURS)
    ):
        raise RuntimeError("an installed skill behaviour is missing from the contract")
    from duckmind.neuro import repertoire as P

    actions = set(SB.SOCIAL_BEHAVIOURS) | set(SP.PLAY_BEHAVIOURS) | set(SS.BEHAVIOURS)
    missing_builders = actions - set(P.BUILDERS)
    missing_satiation = actions - set(S.SATIATE_S)
    selectable = actions - set(C.MOMENTS) - set(S.REFLEX_ONLY)
    missing_weights = selectable - set(S.W)
    if missing_builders or missing_satiation or missing_weights:
        raise RuntimeError(
            "incomplete skill registry: "
            f"builders={sorted(missing_builders)}, "
            f"satiation={sorted(missing_satiation)}, weights={sorted(missing_weights)}"
        )


def _integrated_genome(genome: G.Genome) -> G.Genome:
    """Add skill unlocks/gains to an already-created genome without redrawing its identity."""
    genome = SB.with_social(genome)
    gains = dict(genome.gains)
    unlocked = set(genome.unlocked)
    # Re-evaluate the ordinary unlock rule against the installed repertoire, preserving the
    # caller's exact age, pace, branch and stage instead of reconstructing a genome from its seed.
    unlocked.update(G._unlocked(genome.gd, genome.pace, genome.branch, genome.stage_days))
    style = G._Style(genome.nature, genome.branch)
    expanded_gains = G._gains(genome.axes, genome.stage, style, genome.branch, genome.primary)
    for act in SP.PLAY_ACTS:
        gains[act] = expanded_gains[act]
    gains.update(SS.gains_for(genome))
    unlocked.update(SS.BEHAVIOURS)
    # (G3) what the family's care taught it goes in last: the acts the skills added have gains of
    # their own, so the care is applied on the integrated genome, once
    return replace(genome, unlocked=frozenset(unlocked), gains=gains).with_care_applied()


class RegrowRefused(RuntimeError):
    """A regrow was asked while the duck was not asleep (it is never done awake)."""


def describe_change(before: G.Genome, after: G.Genome) -> dict[str, Any]:
    """What a regrow changed, in plain numbers (for the log and /api/life): the growth day, stage
    and branch, how many acts were unlocked or dropped, and the neural parameters and act gains
    that moved (count, and the largest relative moves)."""
    moved: list[tuple[float, str]] = []
    for field_ in fields(after.p):
        name = field_.name
        a, b = getattr(before.p, name), getattr(after.p, name)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a != b:
            moved.append((abs(b - a) / max(abs(a), 1e-9), name))
    gains_moved: list[tuple[float, str]] = []
    for name, b in after.gains.items():
        a = before.gains.get(name)
        if a is not None and a != b:
            gains_moved.append((abs(b - a) / max(abs(a), 1e-9), name))
    moved.sort(reverse=True)
    gains_moved.sort(reverse=True)
    return {
        "gd": [before.gd, after.gd],
        "stage": [before.stage, after.stage],
        "branch": [before.branch, after.branch],
        "unlocked": sorted(after.unlocked - before.unlocked),
        "dropped": sorted(before.unlocked - after.unlocked),
        "params_moved": len(moved),
        "gains_moved": len(gains_moved),
        "largest_params": [[name, round(rel, 3)] for rel, name in moved[:5]],
        "largest_gains": [[name, round(rel, 3)] for rel, name in gains_moved[:5]],
        "care": {
            "bond": [before.care.bond, after.care.bond],
            "prefs": len(after.care.prefs),
        },
    }


class IntegratedBrain(SP.PlayBrain, SB.SocialBrain):
    """One NeuroBrain with cooperative social/play hooks and a sound pre/post facade."""

    def __init__(self, genome: G.Genome, seed: int = 0, **kw: Any) -> None:
        install_default_skills()
        genome = _integrated_genome(genome)
        super().__init__(genome, seed, **kw)
        self.sound_skills = SS.SoundSkills(genome, seed)
        self.sound_out: SS.SoundOut | None = None
        self.light = False
        # (live voice, LV2) the bridge between the cloud brain's proposals and this brain: it adds
        # candidates to the competition before the step and voices a won call after it; None (the
        # default) changes nothing. A fault in it is counted there and never stops the brain.
        self.live: Any = None
        # what a person told the duck (the page's command strip, ``POST /api/neuro/command``): taken in
        # at the start of a tick, obeyed or refused from the duck's own state, never a motor command
        self.commands = CB.CommandBrain(self)
        # The base Selector was created from this same, individual genome.
        for index, name in enumerate(self.sel.names):
            if name in self.sound_skills.gains:
                self.sel._gain[index] = self.sound_skills.gains[name]

    def trick_feedback(self, kind: str, trick: str | None = None) -> dict[str, Any]:
        """The owner's word on a trick (``praise`` / ``treat`` / ``scold``): the one named, else the one
        it did last. See :meth:`duckmind.neuro.command_brain.CommandBrain.trick_feedback`."""
        return self.commands.trick_feedback(self, kind, trick)

    def regrow(self, genome: G.Genome) -> dict[str, Any]:
        """(Growth G3, docs/design/GROWTH-G3-2026-10-11.md.) The duck has grown: give it the
        parameters of ``genome`` (Duck Core's state at its growth day) and keep everything it has
        lived. Only while it sleeps, never awake: refused (``RegrowRefused``) otherwise.

        The new genome is built into a throw-away brain of the same seed that never steps; its
        genome-derived holders replace this brain's (``_adopt_params`` in each class that has
        any), the sleeping act keeps the body. Returns what changed, in numbers."""
        if str(self.behaviour) not in S.SLEEP:
            raise RegrowRefused(f"it is not asleep (it is doing {self.behaviour!r})")
        before = self.genome
        fresh = IntegratedBrain(genome, self.seed)
        self._adopt_params(fresh)
        self.sound_skills.genome = fresh.sound_skills.genome
        self.sound_skills.gains = fresh.sound_skills.gains
        self.sound_skills.dance.songbird = fresh.sound_skills.dance.songbird
        self.sound_skills.ride.young = fresh.sound_skills.ride.young
        return describe_change(before, self.genome)

    def step(self, t_ms: int, frame: C.SensorFrame) -> tuple[C.MotorCommand, C.BrainTelemetry]:
        commands = self.commands
        try:
            commands.pre(self, t_ms, frame)
        except Exception as error:  # (a fault in the intake never stops the brain)
            commands.fault("pre", error)
        live = self.live
        if live is not None:
            try:
                live.pre(self, t_ms, frame)
            except Exception as error:
                live.fault("pre", error)
        base = self
        active = getattr(getattr(base, "_reflex", None), "active", None)
        fright = active is not None and getattr(active, "reflex", "") in ("startle", "withdraw")
        from duckmind.neuro import select as Sel

        features_now = {
            name: float(base._feat[Sel.F[name]])
            for name in ("play", "curiosity", "social", "happy", "music", "arousal")
            if name in Sel.F and Sel.F[name] < len(base._feat)
        }
        seen, sound_features = self.sound_skills.pre(
            t_ms,
            frame,
            behaviour=str(base.behaviour),
            play=float(base.drives.play),
            fear=float(base.drives.fear),
            arousal=float(base.drives.arousal),
            asleep=str(base.behaviour) in S.SLEEP,
            fright=fright,
            light=bool(getattr(self, "light", False)),
            drives=features_now,
        )
        for name, value in sound_features.items():
            self._feat[Sel.F[name]] = float(value)
        vocal_seq = int(self._vocal_seq)
        cmd, tel = super().step(t_ms, seen)
        cmd = self.escape_command(cmd)  # (an escape's back arc outranks a skill's own walk)
        self._req_vx = float(cmd.vx)  # … and what it asked of its legs is the final command
        cmd, out = self.sound_skills.post(cmd, bout=int(self.bout), vocal_seq=vocal_seq)
        if cmd.vocal_seq > self._vocal_seq:
            self._vocal_seq = cmd.vocal_seq
        ride = out.ride
        if ride.started:
            if ride.frightened:
                self.drives.fear = min(1.0, self.drives.fear + 0.15)
            else:
                self.drives.valence = min(1.0, self.drives.valence + 0.1)
                self.drives.play = min(1.0, self.drives.play + 0.1)
        self.sound_out = out
        if live is not None:
            try:
                cmd = live.post(self, cmd, t_ms)
            except Exception as error:
                live.fault("post", error)
        try:
            cmd = commands.post(self, cmd, t_ms)
        except Exception as error:
            commands.fault("post", error)
        learned = dict(tel.learned)
        learned["sound"] = out.view()
        if live is not None:
            learned["live"] = live.view()
        learned["command"] = commands.view()
        if cmd.behaviour != tel.behaviour:
            tel = replace(
                tel,
                behaviour=cmd.behaviour,
                behaviour_zh=C.behaviour_zh(cmd.behaviour),
                reason_zh=C.behaviour_zh(cmd.behaviour),
            )
        tel = replace(tel, learned=learned)
        return cmd, tel


def make_brain(genome: G.Genome, seed: int = 0, **kw: Any) -> IntegratedBrain:
    """Build the default single brain from an existing identity (seed/GD are never redrawn)."""
    install_default_skills()
    return IntegratedBrain(genome, seed, **kw)


__all__ = ["IntegratedBrain", "SkillRegistry", "install_default_skills", "make_brain"]
