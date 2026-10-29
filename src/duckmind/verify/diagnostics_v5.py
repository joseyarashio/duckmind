"""Facts that explain the verdicts of MACHINE-VERIFY-v5.md but are not part of them (the spec gives them no threshold): per cell of a
stimulus with a place, how the two arms came out and what the numbers behind them look like — which selections were already there
at the stimulus tick and so left out, whether the object ever had a labelled track, how often the duck held any thing at all, what
the control's hits were and how near, and what the head-direction rule of v4 and v3 would have said on the same pairs. Pure
functions over pair outcomes; the report quotes them, no verdict reads them. The exclusions, placement and hand summaries are
v4's (``diagnostics_v4``): the recordings are v4's.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from duckmind.verify import outcomes_v5 as O5
from duckmind.verify import outcomes_v6 as O6
from duckmind.verify import outcomes_v7 as O7
from duckmind.verify import spec_v5 as S
from duckmind.verify.diagnostics_v4 import (
    _spread,
    exclusions,
    loom_summary,
    placement_summary,
    rule_effect,
)
from duckmind.verify.outcomes_v2 import PairOutcomeV2

ALL = "ALL"

__all__ = ["ALL", "exclusions", "loom_summary", "object_cells", "placement_summary", "rule_effect"]


def _alt(t: O5.Target) -> bool:
    """The reading that is only reported: a stretch that starts AT the stimulus tick also counts when the tick before did not match
    (so only a selection that was there before the stimulus is left out). In the brain layer the new object is in the world at the
    stimulus tick and the brain answers in that same tick, so a same-tick start is a reaction there, not a carry-over."""
    return bool(t.value) or (
        t.at_stimulus and not t.before and t.carried_s >= S.TARGET_HOLD_S - S.EPS
    )


def object_cells(
    outcomes: Iterable[PairOutcomeV2], *, merge_presets: bool
) -> dict[str, dict[str, Any]]:
    """For every cell of a stimulus with a place (key ``scope|stimulus``), over its pairs that were in view and read on both arms."""
    cells: dict[str, dict[str, Any]] = {}
    nearest: dict[str, list[float]] = {}
    for o in outcomes:
        if not isinstance(o, O5.PairOutcomeV5) or not o.valid or o.stimulus not in S.DIRECTIONAL:
            continue
        key = f"{ALL if merge_presets else o.preset}|{o.stimulus}"
        c = cells.setdefault(
            key,
            {
                "valid": 0,
                "in_view": 0,
                "out_of_view": 0,
                "view_unknown": 0,
                "unknown": 0,
                "read": 0,
                "stim": 0,
                "ctrl": 0,
                "wins": 0,
                "both": 0,
                "only_ctrl": 0,
                "neither": 0,
                "stim_at_stimulus": 0,
                "stim_at_stimulus_then_hit": 0,
                "ctrl_at_stimulus": 0,
                "ctrl_at_stimulus_then_hit": 0,
                "stim_alt": 0,
                "ctrl_alt": 0,
                "wins_alt": 0,
                "stim_before": 0,
                "ctrl_before": 0,
                "label_exists": 0,
                "stim_thing_windows": 0,
                "ctrl_thing_windows": 0,
                "stim_flick_only": 0,
                "ctrl_flick_only": 0,
                "stim_unlabelled_ticks": 0,
                "ctrl_unlabelled_ticks": 0,
                "stim_thing_ticks": 0,
                "ctrl_thing_ticks": 0,
                "ctrl_hit_objects": Counter(),
                "esc_stim_reflex": 0,
                "esc_stim_flee": 0,
                "esc_stim_either": 0,
                "esc_ctrl_reflex": 0,
                "esc_ctrl_flee": 0,
                "esc_ctrl_either": 0,
                "v4_stim": 0,
                "v4_ctrl": 0,
                "v3_stim": 0,
                "v3_ctrl": 0,
                "v3_n": 0,
            },
        )
        c["valid"] += 1
        if o.perceivable is False:
            c["out_of_view"] += 1
            continue
        if o.perceivable is None:
            c["view_unknown"] += 1
            continue
        c["in_view"] += 1
        ts, tc = o.tgt_stim, o.tgt_sham
        if ts is None or tc is None or ts.value is None or tc.value is None:
            c["unknown"] += 1
            continue
        c["read"] += 1
        if isinstance(ts, O6.TargetV6) and isinstance(tc, O6.TargetV6):
            # (v6 and v7 only: the keys appear only when v6 or v7 read the pair, so the v5 and v5.1 diagnostics stay as they were)
            # v6's own reading of each arm (v7's value also counts a reflex)
            ts6 = ts.v6_value if isinstance(ts, O7.TargetV7) else ts.value
            tc6 = tc.v6_value if isinstance(tc, O7.TargetV7) else tc.value
            for name, add in (
                ("stim_thing_only", bool(ts.thing_only)),
                ("ctrl_thing_only", bool(tc.thing_only)),
                ("wins_thing_only", bool(ts.thing_only) and not tc.thing_only),
                ("stim_by_threat", bool(ts6) and not ts.thing_only),
                ("ctrl_by_threat", bool(tc6) and not tc.thing_only),
                ("stim_threat_windows", ts.threat_ticks > 0),
                ("ctrl_threat_windows", tc.threat_ticks > 0),
                ("stim_threat_ticks", ts.threat_ticks),
                ("ctrl_threat_ticks", tc.threat_ticks),
                ("stim_threat_selected_ticks", ts.threat_selected_ticks),
                ("ctrl_threat_selected_ticks", tc.threat_selected_ticks),
            ):
                c[name] = c.get(name, 0) + int(add)
            if isinstance(ts, O7.TargetV7) and isinstance(tc, O7.TargetV7):
                # (v7 only: what the reflex reading adds to v6's)
                for name, add in (
                    ("stim_v6", bool(ts6)),
                    ("ctrl_v6", bool(tc6)),
                    ("wins_v6", bool(ts6) and not tc6),
                    ("stim_by_reflex", bool(ts.value) and not ts6),
                    ("ctrl_by_reflex", bool(tc.value) and not tc6),
                    ("stim_reflex_hit", bool(ts.reflex_value)),
                    ("ctrl_reflex_hit", bool(tc.reflex_value)),
                    ("stim_reflex_windows", ts.reflex_onsets > 0),
                    ("ctrl_reflex_windows", tc.reflex_onsets > 0),
                    ("stim_reflex_onsets", ts.reflex_onsets),
                    ("ctrl_reflex_onsets", tc.reflex_onsets),
                    ("stim_reflex_on_target_onsets", ts.reflex_on_target_onsets),
                    ("ctrl_reflex_on_target_onsets", tc.reflex_on_target_onsets),
                    ("recordings_with_reflex_track", ts.has_reflex_track and tc.has_reflex_track),
                ):
                    c[name] = c.get(name, 0) + int(add)
                for arm, t7 in (("stim", ts), ("ctrl", tc)):
                    kinds = c.setdefault(f"{arm}_reflex_kinds", Counter())
                    kinds.update(t7.reflex_kinds)
        c["stim"] += bool(ts.value)
        c["ctrl"] += bool(tc.value)
        c["wins"] += bool(ts.value) and not tc.value
        c["both"] += bool(ts.value) and bool(tc.value)
        c["only_ctrl"] += bool(tc.value) and not ts.value
        c["neither"] += not ts.value and not tc.value
        c["stim_at_stimulus"] += ts.at_stimulus
        c["stim_at_stimulus_then_hit"] += ts.at_stimulus and bool(ts.value)
        c["ctrl_at_stimulus"] += tc.at_stimulus
        c["ctrl_at_stimulus_then_hit"] += tc.at_stimulus and bool(tc.value)
        alt_stim, alt_ctrl = _alt(ts), _alt(tc)
        c["stim_alt"] += alt_stim
        c["ctrl_alt"] += alt_ctrl
        c["wins_alt"] += alt_stim and not alt_ctrl
        c["stim_before"] += ts.before
        c["ctrl_before"] += tc.before
        c["label_exists"] += ts.labelled_object
        c["stim_thing_windows"] += ts.thing_ticks > 0
        c["ctrl_thing_windows"] += tc.thing_ticks > 0
        c["stim_flick_only"] += ts.matched_ticks > 0 and not ts.value
        c["ctrl_flick_only"] += tc.matched_ticks > 0 and not tc.value
        c["stim_unlabelled_ticks"] += ts.unlabelled_ticks
        c["ctrl_unlabelled_ticks"] += tc.unlabelled_ticks
        c["stim_thing_ticks"] += ts.thing_ticks
        c["ctrl_thing_ticks"] += tc.thing_ticks
        if tc.value:
            c["ctrl_hit_objects"].update(tc.objects)
            if tc.nearest_m is not None:
                nearest.setdefault(key, []).append(round(tc.nearest_m, 3))
        for arm, esc in (("stim", o.esc_stim), ("ctrl", o.esc_sham)):
            if esc is not None:
                c[f"esc_{arm}_reflex"] += esc.reflex
                c[f"esc_{arm}_flee"] += esc.flee
                c[f"esc_{arm}_either"] += esc.either
        if o.fix_stim is not None and o.fix_sham is not None:
            c["v4_stim"] += bool(o.fix_stim.value)
            c["v4_ctrl"] += bool(o.fix_sham.value)
        if o.turn_stim is not None and o.turn_sham is not None:
            c["v3_n"] += 1
            c["v3_stim"] += bool(o.turn_stim)
            c["v3_ctrl"] += bool(o.turn_sham)
    for key, c in cells.items():
        for arm in ("stim", "ctrl"):
            if f"{arm}_reflex_kinds" in c:
                c[f"{arm}_reflex_kinds"] = dict(c[f"{arm}_reflex_kinds"].most_common())
        c["ctrl_hit_objects"] = dict(c["ctrl_hit_objects"].most_common())
        c["ctrl_hit_nearest_m"] = sorted(nearest.get(key, []))
    return dict(sorted(cells.items()))


def hold_summary(outcomes: Iterable[PairOutcomeV2]) -> dict[str, Any]:
    """How long the stimulated arm's qualifying stretches were (all cells together): the 0.3 s line is crossed by a wide margin or not."""
    longest = [
        o.tgt_stim.longest_s
        for o in outcomes
        if isinstance(o, O5.PairOutcomeV5) and o.tgt_stim is not None and o.tgt_stim.value
    ]
    return {"hits": len(longest), "longest_s": _spread(longest)}
