"""The duck's genome: Duck Core's diversity design turned into neural parameters.

Nothing here invents a new generic duck. A genome starts from Duck Core's own hatching draws
for the seed (``DuckCore(seed).s``: the seven temperament axes of ``content.TEMPERAMENT``
drawn in ``engine._temperament``, the four-branch nature and its primary, the quirks of
``engine._hatch`` — innate tilt side, favourite sound, habit, tempo — and the duck's own
growth pace), then maps them, together with the growth stage and the branch, onto the brain's
gains, thresholds, time constants, bout statistics and motor style.

The mapping reuses Duck Core's tables wherever they exist:

* ``TEMPERAMENT`` ranges (``engine._axis``): leash (boldness), outing (activity), social
  (sociability), novelty (exploration), wits (wit), pace (tempo);
* ``MOTION_STYLE`` (the branch's row, or the nature-weighted blend before the branch, as
  ``engine._style``) and ``MOTION_STAGE`` for standing, walking, turning, sitting, outings,
  greeting reflexes, alert-standing at noises and settling into a touch;
* ``engine.FAVOURED_FITS`` (a branch favours its fits ×1.2) and the branch signatures;
* ``SOCIAL`` (how fast a duck of this stage misses the family; each contact's weight),
  ``engine._style_social`` (a snuggler misses you more, an explorer less);
* ``STAGES`` (amplitude cap, phrase length), ``EXCURSION`` radius (how far it roams),
  unlock days at the duck's own pace (``engine._stage_day`` / ``_calendar_gd`` without jitter).

Pure: :mod:`duckmind.duckcore` (content and, for the hatching draws, the engine) only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from duckmind.duckcore import content
from duckmind.i18n import content_msg, msg
from duckmind.neuro.contract import BEHAVIOURS

if TYPE_CHECKING:
    from duckmind.duckcore.engine import CoreState

AXES: tuple[str, ...] = tuple(content.TEMPERAMENT["axes"])
BRANCH_IDS: tuple[str, ...] = tuple(branch.branch_id for branch in content.BRANCHES)
STAGE_IDS: tuple[str, ...] = tuple(stage.stage_id for stage in content.STAGES)
FAVOURED_FITS: dict[str, frozenset[str]] = {  # engine.FAVOURED_FITS
    "songbird": frozenset({"happy", "social"}),
    "explorer": frozenset({"curious"}),
    "snuggler": frozenset({"calm"}),
    "dreamer": frozenset({"tired", "calm"}),
}
STYLE_SOCIAL = {"snuggler": 1.2, "songbird": 1.0, "dreamer": 0.9, "explorer": 0.8}
# (fid:fix 2026-09-28) Duck Core's quirk words, said as this body performs them (its wake ritual
# is a head raised with a yawn: the body cannot stretch)
HABIT_ZH_BODY = {"stretch_after_nap": msg("neuro.duckcore.quirk_habit.stretch_after_nap.neuro")}
WANDER_GAIN_FLOOR = 0.95
WANDER_GAIN_SLOPE = 0.5
FIRST_STEPS_GD = int(content.FIRST_STEPS_GD)

# ── what the family's care adds to a genome (docs/design/GROWTH-G3-2026-10-11.md) ──────────────
# Duck Core keeps what the duck has *learnt* (a bond to the family, what it likes to do): small,
# slow, bounded numbers (a preference moves at most 0.0105 a day, the bond three steps a day). The
# neural duck chooses ten times a second, so it reads them stronger than Duck Core's own once a
# minute decision does: these are the G3 gains (genome.py only; Duck Core's tables are untouched).
BOND_HATCH = 2000  # CoreState.bond_owner at hatching, and what it fades to without the family
BOND_MAX = 10_000
BOND_SOCIAL = 0.6  # bond 1.0 (of its range) makes the social acts and the missing of people ×1.6
BOND_GREET = 0.5  # … and the greeting reflex ×1.5 (never above 1)
PREF_NEUTRAL = 5000  # a preference nobody has touched
PREF_GAIN = 3.0  # a preference moved by one fifth of its range makes the act ×1.6
PREF_FACTOR_RANGE = (0.6, 1.8)  # how likely, never whether
PREF_VERB_PREFIX = "household.verb:"
# the acts a bond to people is about (the ones whose gains already scale with sociability)
BOND_ACTS: tuple[str, ...] = (
    "follow",
    "follow_mom",
    "greet",
    "call_out",
    "snuggle",
    "nuzzle_tilt",
    "call_response",
)


def care_gain_factor(care: Care, behaviour: str) -> float:
    """What the family's care does to one act's gain: the bond for the acts about people, the
    preference for an act that is one of Duck Core's verbs; exactly 1 for everything else."""
    factor = 1.0
    u = care.bond_unit
    if u > 0.0 and behaviour in BOND_ACTS:
        factor *= 1.0 + BOND_SOCIAL * u
    spec = BEHAVIOURS.get(behaviour)
    if spec is not None and spec.verb is not None:
        factor *= pref_factor(care.pref(spec.verb))
    return factor


@dataclass(frozen=True, slots=True)
class Care:
    """What Duck Core learnt from the family, as far as the body can use it: the bond to the
    owner (CoreState.bond_owner, 2000–10000) and the preferences for what it does by itself
    (``household.verb:<verb>`` → 0–10000, 5000 neutral; only the ones that moved). Neutral = the
    duck nobody has cared for: a genome with it is exactly what it was without."""

    bond: int = BOND_HATCH
    prefs: tuple[tuple[str, int], ...] = ()

    @classmethod
    def from_state(cls, state: Any) -> Care:
        prefs = tuple(
            sorted(
                (str(key)[len(PREF_VERB_PREFIX) :], int(value))
                for key, value in dict(getattr(state, "prefs", {}) or {}).items()
                if str(key).startswith(PREF_VERB_PREFIX) and int(value) != PREF_NEUTRAL
            )
        )
        return cls(bond=int(getattr(state, "bond_owner", BOND_HATCH)), prefs=prefs)

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> Care:
        data = data or {}
        prefs = tuple(sorted((str(k), int(v)) for k, v in dict(data.get("prefs") or {}).items()))
        return cls(bond=int(data.get("bond", BOND_HATCH)), prefs=prefs)

    def to_json(self) -> dict[str, Any]:
        return {"bond": self.bond, "prefs": {verb: value for verb, value in self.prefs}}

    @property
    def neutral(self) -> bool:
        return self.bond == BOND_HATCH and not self.prefs

    @property
    def bond_unit(self) -> float:
        """The bond as 0 (hatching, or faded back) … 1 (the most it can be)."""
        return max(0.0, min(1.0, (self.bond - BOND_HATCH) / (BOND_MAX - BOND_HATCH)))

    def pref(self, verb: str) -> int:
        return dict(self.prefs).get(verb, PREF_NEUTRAL)


def pref_factor(pref: int) -> float:
    """A learnt preference as the salience factor of the act: exactly 1 for neutral."""
    if pref == PREF_NEUTRAL:
        return 1.0
    low, high = PREF_FACTOR_RANGE
    return max(low, min(high, 1.0 + PREF_GAIN * (pref - PREF_NEUTRAL) / PREF_NEUTRAL))


# How the fit of an ethogram act maps onto Duck Core's fits (for FAVOURED_FITS).
ETHOGRAM_FIT = {
    "peck": "curious",
    "forage": "curious",
    "nibble": "curious",
    "investigate": "curious",
    "watch": "calm",
    "head_shake": "calm",
    "body_shake": "calm",
    "follow": "social",
    "greet": "social",
    "call_out": "social",
    "retreat": "overwhelmed",
    "flee": "overwhelmed",
    "doze": "tired",
    "rest_sit": "calm",
    "return_dock": "tired",
    "zoomies": "happy",
}


def _unit(raw: int) -> float:
    """A Duck Core axis value (2000–8000) as 0..1."""
    low, high = (int(value) for value in content.TEMPERAMENT["range"])
    return max(0.0, min(1.0, (raw - low) / (high - low)))


def _span(key: str, a: float) -> float:
    """``engine._axis``: the TEMPERAMENT[key] range at axis value ``a`` (0..1), as a float ×."""
    low, high = (int(value) / 10_000 for value in content.TEMPERAMENT[key])
    return low + a * (high - low)


@dataclass(frozen=True, slots=True)
class NeuralParams:
    """The brain's constants for one duck (floats; seconds, 0..1 levels, multipliers)."""

    # arousal ("neuromodulator") and mood
    tonic_arousal: float
    arousal_tau_s: float
    # startle, fear and sensitisation
    startle_threshold: float
    startle_gain: float
    fear_gain: float
    fear_tau_s: float
    sens_gain: float
    sens_tau_s: float
    alert_stand: float  # MOTION_STYLE alert: standing up at a loud noise
    settle: float  # MOTION_STYLE settle: sitting down into a gentle touch
    greet_reflex: float  # MOTION_STYLE reflex × MOTION_STAGE reflex: standing up to greet
    # habituation
    hab_rate: float
    hab_recover_tau_s: float
    hab_floor: float
    thing_hab_rate: float  # how fast a seen thing stops being new
    # drives
    curiosity_tau_s: float
    novelty_gain: float
    social_tau_s: float
    social_contact: dict[str, float]
    restless_tau_s: float
    walk_consume_s: float  # walking time that satisfies a full locomotor urge
    fatigue_walk_s: float  # walking time that tires it out completely
    rest_recover_s: float
    sleep_rise_s: float
    sleep_fall_s: float
    chrono_shift_min: float
    play_tau_s: float
    # locomotor rhythm
    burst_hz: float
    burst_duty: float
    speed_bias: float
    curve_sigma: float
    curve_tau_s: float
    side_bias: float
    leash_m: float
    # selection
    commitment: float
    switch_margin: float
    sel_noise: float
    sel_noise_tau_s: float
    satiation_scale: float
    # learning
    learn_rate: float
    place_gain: float
    # motor style
    motor_tempo: float
    amplitude_cap: float
    head_alive: float
    phrase_len: int
    vocal_rate: float
    hatchling_wobble: float
    leave_battery: float  # NEST.leave for its stage: charged enough to leave the nest
    # the maintenance needs (integration 2026-09-28; pet:etho's budgets: comfort ~17–19 %,
    # foraging ~16 %, alert scanning): the time each builds in (s), from the temperament
    groom_tau_s: float = 240.0  # the itch to preen, shake and stretch
    peck_tau_s: float = 200.0  # the urge to peck and forage at the floor
    vigil_walk_s: float = 18.0  # walking this long makes it want to stop and look around


@dataclass(frozen=True)
class Genome:
    """One duck: Duck Core's hatching draws + its age + its branch → neural parameters."""

    seed: int
    gd: int
    axes: dict[str, float]  # 0..1
    axes_raw: dict[str, int]  # Duck Core fixed point (2000–8000)
    nature: dict[str, float]  # branch weights, sum 1
    primary: str
    branch: str | None
    stage: str
    stage_days: dict[str, int]
    pace: float
    side: int  # +1: its habitual side is its left (Duck Core 左), −1 right
    favorite_tag: str
    habit: str
    quirk_tempo: float
    unlocked: frozenset[str]
    p: NeuralParams
    gains: dict[str, float]  # behaviour → salience multiplier
    vocal_weights: dict[str, float]
    source: str = "duckcore"  # "duckcore": the seed's own draws; "axes": constructed
    notes: tuple[str, ...] = field(default_factory=tuple)
    # (G3) what the family's care taught it, and whether ``p`` / ``gains`` already carry it
    care: Care = field(default_factory=Care)
    care_applied: bool = False

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_seed(
        cls, seed: int, *, gd: int = 60, branch: str | None = "auto", name: str = "小鴨"
    ) -> Genome:
        """The duck Duck Core hatches from ``seed`` (same seed → same temperament, nature,
        quirks and pace), at growth day ``gd``. ``branch``: ``"auto"`` = the branch Duck Core's
        own lottery gives it with no care history yet (only from the juvenile stage on); a
        branch id = that branch (the family's care raised it); None = not grown into one."""
        from duckmind.duckcore.engine import DuckCore  # hatching only; never at step time

        core = DuckCore(seed=seed, name=name)
        s = core.s
        raw = {axis: int(s.temperament[axis]) for axis in AXES}
        nature = {key: int(value) / 10_000 for key, value in s.nature.items()}
        primary = str(s.branch_detail.get("primary", max(nature, key=lambda k: nature[k])))
        stage_days = {key: int(value) for key, value in s.stage_days.items()}
        stage = _stage_at(gd, stage_days)
        chosen: str | None = None
        if branch == "auto":
            if STAGE_IDS.index(stage) >= STAGE_IDS.index("juvenile"):
                core._choose_branch()  # Duck Core's own draw (branch.pick), nature-only
                chosen = s.branch
        elif branch is not None:
            chosen = branch
        quirks = s.quirks
        return cls.build(
            raw_axes=raw,
            nature=nature,
            primary=primary,
            branch=chosen,
            gd=gd,
            side=1 if quirks.get("tilt_side") == "左" else -1,
            favorite_tag=str(quirks.get("favorite_tag", "chirp")),
            habit=str(quirks.get("habit", "double_look")),
            quirk_tempo=int(quirks.get("tempo", "10000")) / 10_000,
            seed=seed,
            stage_days=stage_days,
            source="duckcore",
        )

    @classmethod
    def from_core_state(
        cls,
        state: CoreState,
        *,
        branch: str | None = "state",
        gd: int | None = None,
        care: Care | None = None,
    ) -> Genome:
        """The duck Duck Core's own state says it is *now*: its temperament, nature, quirks,
        pace and stage days (the hatching draws), and its growth day and branch as they stand
        (the branch the family's care gave it, ``None`` until it grew into one). This is the seam
        of the growth integration (docs/design/GROWTH-G1): at growth day 0 it is the hatchling
        of the seed; later the same call gives the duck it has grown into. Unlike
        :meth:`from_seed` it never draws: whatever the state holds is what the genome is.
        ``branch``: ``"state"`` = the branch the state holds; a branch id or ``None`` puts that
        one in its place (the growth integration's G1 uses it until the duck's own experience
        reaches Duck Core: see ``duckmind.simweb.life_host``).

        G3 (docs/design/GROWTH-G3-2026-10-11.md): ``gd`` and ``care`` stand in for the state's own
        growth day and learning (a regrow builds the genome of the *basis* it recorded, so a restart
        gives back exactly the genome the duck had); ``care`` None = what the state holds
        (``Care.from_state``). The care is carried on the genome and put into its ``p`` and
        ``gains`` by :meth:`with_care_applied`, which the integrated brain calls last (the acts
        the skills add have gains of their own)."""
        raw = {axis: int(state.temperament[axis]) for axis in AXES}
        nature = {key: int(value) / 10_000 for key, value in state.nature.items()}
        primary = str(state.branch_detail.get("primary", max(nature, key=lambda k: nature[k])))
        quirks = state.quirks
        genome = cls.build(
            raw_axes=raw,
            nature=nature,
            primary=primary,
            branch=state.branch if branch == "state" else branch,
            gd=int(state.gd) if gd is None else int(gd),
            side=1 if quirks.get("tilt_side") == "左" else -1,
            favorite_tag=str(quirks.get("favorite_tag", "chirp")),
            habit=str(quirks.get("habit", "double_look")),
            quirk_tempo=int(quirks.get("tempo", "10000")) / 10_000,
            seed=int(state.seed),
            stage_days={key: int(value) for key, value in state.stage_days.items()},
            source="duckcore",
        )
        return replace(genome, care=Care.from_state(state) if care is None else care)

    @classmethod
    def from_axes(
        cls,
        axes: Mapping[str, float],
        *,
        primary: str,
        branch: str | None = None,
        gd: int = 60,
        nature: Mapping[str, float] | None = None,
        side: int = 1,
        favorite_tag: str = "chirp",
        habit: str = "double_look",
        quirk_tempo: float = 1.0,
    ) -> Genome:
        """A duck with a chosen temperament (axes 0..1; missing ones 0.5), for comparisons."""
        raw = {axis: int(round(2000 + 6000 * float(axes.get(axis, 0.5)))) for axis in AXES}
        if nature is None:
            base = {b: 0.15 for b in BRANCH_IDS}
            base[primary] = 0.55
            nature = base
        total = sum(nature.values()) or 1.0
        return cls.build(
            raw_axes=raw,
            nature={k: float(v) / total for k, v in nature.items()},
            primary=primary,
            branch=branch,
            gd=gd,
            side=side,
            favorite_tag=favorite_tag,
            habit=habit,
            quirk_tempo=quirk_tempo,
            seed=-1,
            stage_days=None,
            source="axes",
        )

    @classmethod
    def build(
        cls,
        *,
        raw_axes: Mapping[str, int],
        nature: Mapping[str, float],
        primary: str,
        branch: str | None,
        gd: int,
        side: int,
        favorite_tag: str,
        habit: str,
        quirk_tempo: float,
        seed: int,
        stage_days: Mapping[str, int] | None,
        source: str,
    ) -> Genome:
        a = {axis: _unit(int(raw_axes[axis])) for axis in AXES}
        pace = _span("pace", a["tempo"])
        if stage_days is None:
            stage_days = {
                stage.stage_id: int(round(stage.gd_min * pace)) for stage in content.STAGES
            }
        stage = _stage_at(gd, stage_days)
        if branch is not None and STAGE_IDS.index(stage) < STAGE_IDS.index("juvenile"):
            branch = None  # a duck grows into its branch only as a juvenile
        style = _Style(nature, branch)
        p = _params(a, stage, style, quirk_tempo)
        unlocked = _unlocked(gd, pace, branch, stage_days)
        gains = _gains(a, stage, style, branch, primary)
        vocal = _vocal_weights(favorite_tag, branch, primary)
        notes: list[str] = []
        if branch is None and STAGE_IDS.index(stage) >= STAGE_IDS.index("juvenile"):
            notes.append("juvenile without a branch: the nature blend stands in")
        return cls(
            seed=seed,
            gd=gd,
            axes=a,
            axes_raw={k: int(v) for k, v in raw_axes.items()},
            nature=dict(nature),
            primary=primary,
            branch=branch,
            stage=stage,
            stage_days=dict(stage_days),
            pace=pace,
            side=1 if side >= 0 else -1,
            favorite_tag=favorite_tag,
            habit=habit,
            quirk_tempo=quirk_tempo,
            unlocked=unlocked,
            p=p,
            gains=gains,
            vocal_weights=vocal,
            source=source,
            notes=tuple(notes),
        )

    # ── what the care taught ──────────────────────────────────────────────────

    def with_care_applied(self) -> Genome:
        """This genome with its :attr:`care` put into its parameters and gains (once: a genome
        that carries it already, or has no care to carry, is returned as it is).

        * the **bond** to the family (``Care.bond_unit`` u, 0 … 1): it misses people sooner
          (``social_tau_s`` ÷ (1 + 0.6 u)), greets more readily (``greet_reflex`` × (1 + 0.5 u),
          never above 1) and the acts about people are likelier (× (1 + 0.6 u)) — the same lever
          sociability already pulls;
        * a **preference** for something it does by itself (Duck Core's ``household.verb:X``):
          that act's salience × ``pref_factor`` (1 for neutral, ×1.6 for a fifth of the range
          up, ×0.6 … down; bounded 0.6 … 1.8) — how likely, never whether.

        Neutral care changes nothing, not a bit."""
        care = self.care
        if self.care_applied or care.neutral:
            return self
        u = care.bond_unit
        p = self.p
        if u > 0.0:
            p = replace(
                p,
                social_tau_s=p.social_tau_s / (1.0 + BOND_SOCIAL * u),
                greet_reflex=min(1.0, p.greet_reflex * (1.0 + BOND_GREET * u)),
            )
        gains = dict(self.gains)
        for behaviour in gains:
            factor = care_gain_factor(care, behaviour)
            if factor != 1.0:
                gains[behaviour] *= factor
        return replace(self, p=p, gains=gains, care_applied=True)

    # ── views ─────────────────────────────────────────────────────────────────

    @property
    def style_branch(self) -> str:
        """The branch whose style leads: its branch, else its primary nature."""
        return self.branch or self.primary

    def summary(self) -> dict[str, Any]:
        # (Duck Core's own words for its stages, branches and habits, as catalog messages: the
        # text is the same 繁體中文, the page shows them in the viewer's language)
        zh_branch = (
            content_msg(
                f"core.content.BRANCHES.{self.branch}_zh", content.BRANCH_BY_ID[self.branch].zh
            )
            if self.branch
            else None
        )
        stage_zh = content_msg(
            f"core.content.STAGES.{self.stage}_zh",
            next(s.zh for s in content.STAGES if s.stage_id == self.stage),
        )
        habit_zh = content_msg(
            f"core.content.QUIRK_HABITS.{self.habit}",
            dict(content.QUIRK_HABITS).get(self.habit, self.habit),
        )
        # (fid:fix) said as this body does it: it cannot stretch its body (Duck Core's words)
        habit_zh = HABIT_ZH_BODY.get(self.habit, habit_zh)
        return {
            "seed": self.seed,
            "source": self.source,
            "gd": self.gd,
            "stage": self.stage,
            "stage_zh": stage_zh,
            "branch": self.branch,
            "branch_zh": zh_branch,
            "primary": self.primary,
            "primary_zh": content_msg(
                f"core.content.BRANCHES.{self.primary}_zh", content.BRANCH_BY_ID[self.primary].zh
            ),
            "axes": {k: round(v, 3) for k, v in self.axes.items()},
            "axes_raw": dict(self.axes_raw),
            "nature": {k: round(v, 3) for k, v in sorted(self.nature.items())},
            "side": msg("neuro.side.left_short")
            if self.side > 0
            else msg("neuro.side.right_short"),
            "favorite_tag": self.favorite_tag,
            "habit": self.habit,
            "habit_zh": habit_zh,
            "quirk_tempo": round(self.quirk_tempo, 3),
            "pace": round(self.pace, 3),
            "unlocked": sorted(self.unlocked),
            "notes": list(self.notes),
            **({} if self.care.neutral else {"care": self.care.to_json()}),
        }


# The page's four contrasting ducks (simweb/neuro_api.PRESETS: Duck Core seeds found over 6,000
# hatchings): a preset is only a seed and a growth day — the genome is Duck Core's own draw.
PRESETS: dict[str, tuple[int, int]] = {
    "explorer": (4107, 60),
    "snuggler": (1728, 20),
    "dreamer": (4670, 200),
    "hatchling": (3341, 7),  # a toddling hatchling: its first steps unlock at GD 5 × its pace
}


def preset(name: str) -> Genome:
    """A preset duck by id (``PRESETS``): ``Genome.from_seed(seed, gd=gd)``."""
    if name not in PRESETS:
        raise ValueError(f"no preset {name!r}; known: {sorted(PRESETS)}")
    seed, gd = PRESETS[name]
    return Genome.from_seed(seed, gd=gd)


def _stage_at(gd: int, stage_days: Mapping[str, int]) -> str:
    current = STAGE_IDS[0]
    for stage in content.STAGES:
        if gd >= int(stage_days.get(stage.stage_id, stage.gd_min)):
            current = stage.stage_id
    return current


class _Style:
    """``engine._style``: the branch's MOTION_STYLE row, or the nature-weighted blend."""

    def __init__(self, nature: Mapping[str, float], branch: str | None) -> None:
        self.nature = dict(nature)
        self.branch = branch

    def weight(self, table: Mapping[str, float]) -> float:
        if self.branch is not None:
            return float(table[self.branch])
        return sum(self.nature.get(b, 0.0) * float(table[b]) for b in BRANCH_IDS)

    def motion(self, key: str) -> float:
        return self.weight({b: content.MOTION_STYLE[b][key] / 10_000 for b in BRANCH_IDS})

    def lean(self, branch: str) -> float:
        """How much this duck is of ``branch`` (1 for its grown branch)."""
        if self.branch is not None:
            return 1.0 if self.branch == branch else 0.0
        return self.nature.get(branch, 0.0)


def _stage_row(stage: str) -> dict[str, float]:
    return {k: v / 10_000 for k, v in content.MOTION_STAGE[stage].items() if k != "stand_s"}


def _params(a: Mapping[str, float], stage: str, style: _Style, quirk_tempo: float) -> NeuralParams:
    st = _stage_row(stage)
    young = {"hatchling": 1.0, "duckling": 0.6, "juvenile": 0.25, "adult": 0.0}[stage]
    stage_obj = next(s for s in content.STAGES if s.stage_id == stage)
    bold, act, soc = a["boldness"], a["activity"], a["sociability"]
    expl, wit, chrono = a["exploration"], a["wit"], a["chronotype"]
    outing = _span("outing", act) * style.motion("outing") * st["outing"]
    social_mult = _span("social", soc) * style.weight(STYLE_SOCIAL)
    rise_alone = int(content.SOCIAL["rise_alone"][stage])  # per minute, fixed point
    contact = {k: int(v) / 10_000 for k, v in content.SOCIAL["contact"].items()}
    radius_m = int(content.EXCURSION["radius_mm"][stage]) / 1000
    dreamy = style.lean("dreamer") + 0.8 * style.lean("snuggler") + 0.5 * style.lean("songbird")
    return NeuralParams(
        tonic_arousal=0.22 + 0.2 * act + 0.06 * (1 - bold) + 0.05 * young,
        arousal_tau_s=12.0 + 18.0 * (1 - bold),
        startle_threshold=0.3 + 0.45 * bold - 0.08 * young,
        startle_gain=1.25 - 0.5 * bold + 0.2 * young,
        fear_gain=1.05 - 0.55 * bold + 0.25 * young,
        fear_tau_s=25.0 + 95.0 * (1 - bold),
        sens_gain=0.65 - 0.35 * bold + 0.2 * young,
        sens_tau_s=120.0 + 240.0 * (1 - bold),
        alert_stand=style.motion("alert"),
        settle=style.motion("settle"),
        greet_reflex=style.motion("reflex") * st["reflex"],
        hab_rate=(0.13 + 0.14 * wit) * (1 - 0.3 * young),
        # (integration 2026-09-28: pet:etho's recovery time constants are 10–30 min for
        # orienting, 5–20 min for a startle; 2.5–5 min let the 8-s clap train recover between
        # claps and the gate — 10th clap ≤ half the 1st in ≥ 90 % of runs — failed, 91/130)
        hab_recover_tau_s=300.0 + 400.0 * wit,
        hab_floor=0.12 + 0.08 * (1 - wit),
        thing_hab_rate=0.08 + 0.1 * expl,
        curiosity_tau_s=420.0 / _span("novelty", expl),
        novelty_gain=_span("novelty", expl) * (0.85 + 0.3 * style.lean("explorer")),
        social_tau_s=3600.0 * 30 / max(1, rise_alone) / social_mult,
        social_contact=contact,
        # (integration 2026-09-28: the least active ducks built the urge to walk so slowly they
        # walked ~5 % of the day — the owner: every duck wanders often, personality only shapes
        # how much; the explorer keeps its 120 s)
        restless_tau_s=60.0 + 105.0 / outing,
        # (integration 2026-09-28: walking satisfies the urge a little sooner — the bold
        # explorer walked 62 % of the day in the synthetic room; pet:etho's budget is 10–20 %)
        # (fix 2026-09-28: calm ducks walked 7–9 % of the day, about 蛋黃's 7 % — the owner's
        # 走太少; every duck wanders often, activity shapes how much: the urge is satisfied by
        # about the same walking for everyone)
        walk_consume_s=(34.0 + 12.0 * act) * (0.5 + 0.5 * st["scale"]),
        fatigue_walk_s=(900.0 + 900.0 * act) * (0.6 + 0.4 * (1 - young)),
        rest_recover_s=420.0 - 120.0 * act,
        sleep_rise_s=(6500.0 - 3200.0 * young) * (1.1 - 0.25 * style.lean("dreamer")),
        sleep_fall_s=600.0 + 480.0 * young,
        chrono_shift_min=(chrono - 0.5) * 150.0,
        play_tau_s=900.0 / (0.7 + 0.6 * act),
        burst_hz=(0.09 + 0.06 * act) * (1.0 + 0.4 * young) * quirk_tempo,
        burst_duty=0.5 + 0.2 * act + 0.1 * style.lean("explorer") - 0.12 * young,
        speed_bias=0.25 + 0.35 * act + 0.15 * bold - 0.1 * style.lean("dreamer"),
        curve_sigma=0.28 + 0.2 * expl + 0.1 * young,
        curve_tau_s=2.5 + 2.0 * (1 - act),
        side_bias=0.06,
        leash_m=radius_m * _span("leash", bold),
        commitment=0.3 + 0.15 * dreamy + 0.1 * (1 - act),
        switch_margin=0.04 + 0.04 * dreamy,
        sel_noise=0.05 + 0.06 * expl + 0.02 * young,
        sel_noise_tau_s=3.0 + 3.0 * (1 - act),
        satiation_scale=(0.8 + 0.4 * act) * (1.0 + 0.5 * young),
        learn_rate=_span("wits", wit),
        place_gain=0.15 + 0.1 * wit,
        motor_tempo=quirk_tempo * (0.92 + 0.16 * act) * (1.0 + 0.1 * young),
        amplitude_cap=int(stage_obj.amplitude_cap) / 10_000,
        head_alive=0.35 + 0.3 * expl + 0.1 * style.lean("explorer") - 0.1 * style.lean("dreamer"),
        phrase_len=int(stage_obj.phrase_len),
        vocal_rate=0.7 + 0.5 * style.lean("songbird") + 0.2 * soc,
        hatchling_wobble=young,
        leave_battery=int(content.NEST["leave"][stage]) / 10_000,
        # comfort-minded ducks (dreamy, snuggly, calm) itch sooner; curious ones peck sooner;
        # timid ones stop to look around after shorter walks
        groom_tau_s=300.0 / (0.75 + 0.5 * dreamy / 1.8 + 0.35 * (1 - act)),
        peck_tau_s=200.0 / (0.7 + 0.5 * expl + 0.2 * young),
        vigil_walk_s=(10.0 + 16.0 * bold) * (0.8 + 0.4 * act),
    )


def _unlocked(
    gd: int, pace: float, branch: str | None, stage_days: Mapping[str, int]
) -> frozenset[str]:
    """Behaviours this duck can do at ``gd``: unlock days at its own pace (Duck Core's
    calendar without the ±2 jitter; stage days exact), branch signatures for its branch."""
    out = set()
    for behaviour, spec in BEHAVIOURS.items():
        if spec.branch is not None and spec.branch != branch:
            continue
        day = spec.unlock_gd
        if behaviour == "flip":
            day = int(stage_days.get("juvenile", 45))  # taught from the juvenile stage on (its own day)
        elif spec.branch is not None:
            # 45 = becoming a juvenile (its branch); 75 = the second signature, at its pace.
            day = int(stage_days.get("juvenile", 45)) if day == 45 else int(round(day * pace))
        elif spec.locomotes or behaviour == "body_shake":
            day = (
                int(round(FIRST_STEPS_GD * pace))
                if day <= FIRST_STEPS_GD
                else int(round(day * pace))
            )
        elif day > 0:
            day = int(round(day * pace))
        if gd >= day:
            out.add(behaviour)
    return frozenset(out)


def _gains(
    a: Mapping[str, float], stage: str, style: _Style, branch: str | None, primary: str
) -> dict[str, float]:
    """Salience multipliers per behaviour: Duck Core's style and stage rows, favoured fits,
    and the temperament axes (how likely, never whether)."""
    st = _stage_row(stage)
    bold, act, soc = a["boldness"], a["activity"], a["sociability"]
    expl, wit = a["exploration"], a["wit"]
    young = {"hatchling": 1.0, "duckling": 0.6, "juvenile": 0.25, "adult": 0.0}[stage]
    gains: dict[str, float] = {}
    for behaviour, spec in BEHAVIOURS.items():
        g = 1.0
        fit = None
        if spec.verb is not None:
            fit = content.VERB_BY_ID[spec.verb].fit
        else:
            fit = ETHOGRAM_FIT.get(behaviour)
        if fit is not None:
            lean = style.weight({b: (0.2 if fit in FAVOURED_FITS[b] else 0.0) for b in BRANCH_IDS})
            g *= 1.0 + lean  # favoured fits ×1.2 for the grown branch (blended before)
        if spec.branch is not None:
            g *= 1.2
        gains[behaviour] = g
    for verb in ("stand_up", "sit_down", "wander", "turn_around"):
        gains[verb] *= style.motion(verb) * st[verb]
    outing = _span("outing", act)
    novelty = _span("novelty", expl)
    social = _span("social", soc) * style.weight(STYLE_SOCIAL)
    gains["wander"] *= outing
    # compressed (fix 2026-09-28): Duck Core's style × stage × outing spread the urge to wander
    # 5-fold (explorer 3.0, dreamer 0.6) and the calm ducks walked like 蛋黃 (7 %); the owner:
    # personality shapes how much, never nearly still
    gains["wander"] = max(gains["wander"], WANDER_GAIN_FLOOR + WANDER_GAIN_SLOPE * gains["wander"])
    gains["forage"] *= 0.6 + 0.4 * novelty * style.motion("wander")
    gains["zoomies"] *= (0.4 + 0.8 * act) * (0.7 + 0.6 * style.lean("explorer"))
    for behaviour in ("investigate", "peck", "nibble", "survey", "look_around"):
        gains[behaviour] *= novelty
    for behaviour in ("follow", "greet", "call_out", "snuggle", "nuzzle_tilt", "call_response"):
        gains[behaviour] *= social
    gains["follow"] *= 1.0 + 0.6 * young  # a young duck follows its family everywhere
    gains["call_out"] *= 1.0 + 0.5 * young  # and peeps for them
    # (fix 2026-09-28: the review found ordinary ducklings hard to tell apart — 9/16 identified
    # from their own second run at GD 20; each axis now leads to its own acts more strongly)
    for behaviour in ("retreat", "flee", "freeze", "tuck_away"):
        gains[behaviour] *= 1.45 - 0.9 * bold
    for behaviour in ("double_take", "glance", "perk_tilt"):
        gains[behaviour] *= 0.7 + 0.6 * wit
    for behaviour in ("dabble", "still", "proud_puff", "head_shake"):
        gains[behaviour] *= 1.25 - 0.5 * wit  # goofy moments: more of them in a less sharp duck
    for behaviour in ("rest_sit", "doze", "watch", "moon_gaze"):
        gains[behaviour] *= 1.3 - 0.6 * act
    for behaviour in ("happy_sway", "little_dance", "peekaboo", "zoomies"):
        gains[behaviour] *= 0.7 + 0.3 * (act + soc)  # playful: the lively and the sociable
    for behaviour in ("preen", "stretch_yawn", "body_shake"):
        gains[behaviour] *= 1.2 - 0.4 * act  # comfort acts: the calm take their time over them
    gains["melt"] *= 0.6 + style.motion("settle") * 0.8
    return gains


def _vocal_weights(favorite: str, branch: str | None, primary: str) -> dict[str, float]:
    """Engine: its favourite tag weighs most, then its branch's favourite tags."""
    weights = {tag: 1.0 for tag in content.TAGS}
    weights[favorite] = 3.0
    for tag in content.BRANCH_BY_ID[branch or primary].favorite_tags:
        weights[tag] += 1.0
    return weights
