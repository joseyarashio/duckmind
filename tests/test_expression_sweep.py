import math

import pytest

from duckmind.duckcore import content
from duckmind.expression_sweep import envelope
from duckmind.expression_sweep.clips import (
    AMPLITUDES,
    CONTROL_HZ,
    SIDES,
    TEMPOS,
    Variant,
    duration_s,
    peak_offsets,
    trajectory,
    variants,
)


def test_frame_interpolation_hits_keyframes_and_eases_between_them():
    frames = [(0, {}), (1000, {"hp": 1.0}), (2000, {})]
    assert content.frame_at(frames, 1000) == {"hp": 1.0}
    assert content.frame_at(frames, 500)["hp"] == pytest.approx(0.5)
    assert content.frame_at(frames, 250)["hp"] == pytest.approx(0.15625)  # smoothstep(0.25)
    assert content.frame_at(frames, 5000) == {}


def test_every_clip_amplitude_tempo_and_side_is_swept():
    grid = variants()
    assert len(grid) == len(content.CLIPS) * len(AMPLITUDES) * len(TEMPOS) * len(SIDES)
    assert len({variant.key for variant in grid}) == len(grid)


def test_streams_start_and_end_at_home_and_follow_the_clip():
    variant = Variant("moon_gaze", 1.0, 1.0, 1)
    stream = trajectory(variant, lead_s=0.5, tail_s=1.0)
    ticks = round((0.5 + duration_s(variant) + 1.0) * CONTROL_HZ)
    assert all(len(values) == ticks for values in stream.values())
    for name in ("neck_pitch", "head_pitch", "head_yaw", "head_roll"):
        assert stream[name][0] == 0.0 and stream[name][-1] == 0.0
    assert max(stream["neck_pitch"]) > 0  # leans back...
    assert min(stream["head_pitch"]) < 0  # ...and looks up (negative head_pitch is up)


def test_amplitude_scales_the_head_but_tempo_only_changes_timing():
    full = peak_offsets(Variant("look_around", 1.0, 1.0, 1))
    half = peak_offsets(Variant("look_around", 0.5, 1.0, 1))
    fast = peak_offsets(Variant("look_around", 1.0, 1.4, 1))
    assert half["head_yaw"] == pytest.approx(full["head_yaw"] / 2)
    assert fast["head_yaw"] == pytest.approx(full["head_yaw"], rel=0.02)
    assert duration_s(Variant("look_around", 1.0, 1.4, 1)) < duration_s(
        Variant("look_around", 1.0, 1.0, 1)
    )


def _stats(falls=0, tilt=5.0, torque=0.3, contacts=0, track=0.9, unsettled=0):
    return {
        "episodes": 32,
        "settle_failures": unsettled,
        "falls": falls,
        "tilt_rise_deg_p95": tilt,
        "torque_ratio_max": torque,
        "self_contacts": contacts,
        "track_ratio": track,
    }


def _result(clip: str, fail_at: float | None, posture_fail: str = "stand"):
    variants_ = {}
    for amplitude in AMPLITUDES:
        for tempo in TEMPOS:
            for side in SIDES:
                key = Variant(clip, amplitude, tempo, side).key
                bad = fail_at is not None and amplitude >= fail_at and tempo == max(TEMPOS)
                variants_[key] = {
                    "sit": _stats(),
                    "stand": _stats(falls=2) if bad and posture_fail == "stand" else _stats(),
                }
    from duckmind.expression_sweep.clips import clip_digest

    return {
        "format": envelope.RESULT_FORMAT,
        "source": {
            "content_table": content.table_hash(),
            "clip_digests": {clip: clip_digest(clip)},
        },
        "variants": variants_,
    }


def test_a_clip_is_capped_below_the_first_failing_amplitude():
    capped = envelope.caps(_result("look_around", fail_at=0.8))
    assert capped == {"look_around": {"sit": 1.0, "stand": 0.6}}


def test_an_unswept_amplitude_is_never_assumed_safe():
    result = _result("look_around", fail_at=None)
    del result["variants"][Variant("look_around", 0.8, 1.0, -1).key]
    assert envelope.caps(result)["look_around"] == {"sit": 0.6, "stand": 0.6}


def test_stale_results_are_detected():
    result = _result("look_around", fail_at=None)
    assert envelope.is_current(result)
    result["source"]["content_table"] = "different"
    assert not envelope.is_current(result)
    assert envelope.fidelity(result, "look_around", "sit") == pytest.approx(0.9)


def _amplitudes(core, verb: str) -> list[int]:
    return [
        event.payload["amplitude"]
        for event in core.log.events
        if event.kind == "duck.expression" and event.payload.get("verb") == verb
    ]


def test_validated_caps_limit_each_clip_and_unswept_clips_stay_conservative():
    from duckmind.duckcore.engine import DuckCore

    core = DuckCore(seed=7, conservative=False, expression_caps={"look_around": 2500})
    core.advance(240)
    looks = _amplitudes(core, "look_around")
    others = [
        event.payload["amplitude"]
        for event in core.log.events
        if event.kind == "duck.expression" and event.payload.get("verb") != "look_around"
    ]
    assert looks and max(looks) <= 2500
    assert others and max(others) <= 6000
    hatched = next(event for event in core.log.events if event.kind == "sim.hatched")
    assert hatched.payload["expression_caps"] is not None


def test_a_caps_file_round_trips_and_is_ignored_when_stale(tmp_path):
    result = _result("look_around", fail_at=0.8)
    path = tmp_path / "caps.json"
    payload = envelope.write_caps(result, path)
    expected = {"look_around": 6000, "look_around@sit": 10000, "look_around@stand": 6000}
    assert payload["engine"] == expected
    assert envelope.read_caps(path) == expected
    from duckmind.expression_sweep.clips import clip_digest

    stale = path.read_text().replace(clip_digest("look_around"), "0" * 16)
    path.write_text(stale)
    assert envelope.read_caps(path) is None  # the clip changed since it was swept
    assert envelope.read_caps(tmp_path / "missing.json") is None


def test_the_seated_envelope_stays_inside_what_the_networks_were_trained_on():
    from duckmind.expression_sweep import robotd_model

    for joint, trained in robotd_model.TRAINED_HEAD_RANGE.items():
        assert content.SEATED_ENVELOPE[joint] <= trained, joint
    assert content.SEATED_ENVELOPE["neck_pitch"] <= robotd_model.NECK_BACK_LIMIT


def test_robotd_slew_matches_the_simulator_smoothing():
    from duckmind.expression_sweep import robotd_model

    step = robotd_model.slew([1.0] * 50)
    assert step[0] == pytest.approx(0.2)
    assert step[13] > 0.95  # ~0.27 s to 95%
    # The 3D view's 1.78 Hz first-order cutoff is the same filter at 50 Hz.
    alpha = 1 - math.exp(-2 * math.pi * content.ROBOTD_HEAD_CUTOFF_HZ / robotd_model.CONTROL_HZ)
    assert alpha == pytest.approx(robotd_model.HEAD_ALPHA, abs=0.01)
    assert robotd_model.mouth_angle(0.0) == pytest.approx(math.radians(-5))
    assert robotd_model.mouth_angle(2.0) == pytest.approx(math.radians(30))


def test_sweep_statistics_combine_randomizations_per_variant():
    from duckmind.expression_sweep.runner import _percentile, aggregate

    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assigned = [Variant("nod", 1.0, 1.0, 1)] * 3 + [Variant("nod", 0.6, 1.0, 1)]
    metrics = {
        "settle_bad": [False, False, False, False],
        "fell": [False, True, False, False],
        "tilt": [5.0, 60.0, 7.0, 3.0],
        "tilt_rise": [1.0, 55.0, 2.0, 0.5],
        "torque_ratio": [0.3, 1.0, 0.4, 0.2],
        "strain_frac": [0.0, 0.5, 0.0, 0.0],
        "contacts": [0, 0, 2, 0],
        "track_ratio": [0.9, 0.3, 0.9, 1.0],
        "rms_error": [0.02, 0.4, 0.03, 0.01],
        "drift": [0.001, 0.2, 0.002, 0.0],
    }
    out = aggregate(assigned, metrics)
    full = out[Variant("nod", 1.0, 1.0, 1).key]
    assert (full["episodes"], full["falls"], full["self_contacts"]) == (3, 1, 1)
    assert full["torque_ratio_max"] == 1.0
    assert full["settle_failures"] == 0
    assert out[Variant("nod", 0.6, 1.0, 1).key]["falls"] == 0


def test_a_duck_that_had_not_settled_does_not_count_against_the_clip():
    from duckmind.expression_sweep.runner import aggregate

    assigned = [Variant("nod", 1.0, 1.0, 1)] * 3
    metrics = {
        # The second episode was already leaning back when the clip began.
        "settle_bad": [False, True, False],
        "fell": [False, True, False],
        "tilt": [2.0, 62.0, 3.0],
        "tilt_rise": [0.5, 0.2, 1.0],
        "torque_ratio": [0.3, 1.0, 0.4],
        "strain_frac": [0.0, 0.9, 0.0],
        "contacts": [0, 5, 0],
        "track_ratio": [0.9, 0.2, 0.8],
        "rms_error": [0.02, 0.4, 0.03],
        "drift": [0.001, 0.0, 0.002],
    }
    stats = aggregate(assigned, metrics)[Variant("nod", 1.0, 1.0, 1).key]
    assert (stats["episodes"], stats["settle_failures"]) == (3, 1)
    assert (stats["falls"], stats["self_contacts"], stats["torque_ratio_max"]) == (0, 0, 0.4)
    assert envelope.passes(stats, envelope.Criteria())
    # With no settled episode at all the clip is judged on everything, never waved through.
    metrics["settle_bad"] = [True, True, True]
    assert not envelope.passes(
        aggregate(assigned, metrics)[Variant("nod", 1.0, 1.0, 1).key], envelope.Criteria()
    )


def test_settle_failures_are_reported_per_posture():
    result = _result("look_around", fail_at=None)
    first = next(iter(result["variants"].values()))
    first["sit"]["settle_failures"] = 3
    failures, episodes = envelope.settle_failures(result, "sit")
    assert failures == 3 and episodes == 32 * len(result["variants"])
    assert envelope.caps(result)["look_around"]["sit"] == 1.0


def test_walk_plans_cover_the_locomotion_bounds():
    from duckmind.expression_sweep.runner import WALK_PLANS

    low, high = content.LOCOMOTION["walk_vx"]
    speeds = {vx for vx, _, _ in WALK_PLANS if vx > 0}
    assert low in speeds and high in speeds
    turns = {abs(vyaw) for vx, vyaw, _ in WALK_PLANS if vx == 0}
    assert content.LOCOMOTION["turn_vyaw"] in turns
    assert max(seconds for _, _, seconds in WALK_PLANS) == content.LOCOMOTION["walk_s_max"]


def test_caps_survive_new_clips_but_not_edited_ones(monkeypatch):
    result = _result("look_around", fail_at=None)
    assert envelope.current_clips(result) == {"look_around"}
    frames = [(0, {}), (500, {"hy": 0.5}), (1000, {})]
    monkeypatch.setitem(content.CLIPS, "brand_new_clip", frames)
    assert envelope.current_clips(result) == {"look_around"}  # adding a clip keeps evidence
    monkeypatch.setitem(content.CLIPS, "look_around", frames)
    assert envelope.current_clips(result) == set()  # editing the swept clip voids it


def test_the_engine_uses_the_cap_of_the_posture_it_is_in():
    from duckmind.duckcore.engine import DuckCore

    caps = {"look_around": 0, "look_around@sit": 10000, "look_around@stand": 0}
    core = DuckCore(seed=7, conservative=False, expression_caps=caps)
    core.advance(240)
    seated = [
        event.payload["amplitude"]
        for event in core.log.events
        if event.kind == "duck.expression"
        and event.payload.get("verb") == "look_around"
        and event.payload.get("posture", "sit") == "sit"
    ]
    assert seated and max(seated) > 0  # the stand-only limit does not silence it seated


def test_a_brief_load_spike_is_not_a_failure_but_sustained_strain_is():
    spike = _stats(torque=1.0) | {"strain_frac_p95": 0.0}
    strain = _stats(torque=0.7) | {"strain_frac_p95": 0.2}
    assert envelope.passes(spike, envelope.Criteria())
    assert not envelope.passes(strain, envelope.Criteria())
    assert envelope.passes(_stats(contacts=1), envelope.Criteria())
    assert not envelope.passes(_stats(contacts=2), envelope.Criteria())


# ---------------------------------------------------------------- v3: formats and looks

EVIDENCE = __import__("pathlib").Path(__file__).resolve().parents[1] / "docs/validation/evidence"


def test_format_2_results_and_caps_stay_readable(tmp_path):
    import json

    old = _result("look_around", fail_at=0.8) | {"format": 2}
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(old))
    assert envelope.load(path)["format"] == 2
    payload = envelope.write_caps(old, tmp_path / "caps.json")
    assert payload["result_format"] == 2
    assert envelope.read_caps(tmp_path / "caps.json") == payload["engine"]
    for unreadable in (1, 4):
        path.write_text(json.dumps(old | {"format": unreadable}))
        with pytest.raises(ValueError):
            envelope.load(path)
        caps = json.loads((tmp_path / "caps.json").read_text()) | {"result_format": unreadable}
        (tmp_path / "old_caps.json").write_text(json.dumps(caps))
        assert envelope.read_caps(tmp_path / "old_caps.json") is None
    assert envelope.RESULT_FORMAT == 3 and 2 in envelope.READABLE_RESULT_FORMATS


def test_the_v2_evidence_still_loads_and_its_caps_file_still_applies():
    result_path = EVIDENCE / "expression-sweep-v2.json"
    caps_path = EVIDENCE / "expression_caps.json"
    if not (result_path.is_file() and caps_path.is_file()):
        pytest.skip("v2 evidence not present")
    import json

    from duckmind.expression_sweep.clips import clip_digest

    result = envelope.load(result_path)
    assert result["format"] == 2 and len(envelope.caps(result)) == 27
    payload = json.loads(caps_path.read_text())
    unchanged = {c for c, d in payload["clip_digests"].items() if clip_digest(c) == d}
    expected = {k: v for k, v in payload["engine"].items() if k.split("@")[0] in unchanged}
    assert envelope.read_caps(caps_path) == (expected or None)


def test_looks_are_defined_only_in_the_sweep_from_the_spec_angles():
    from duckmind.expression_sweep import looks, robotd_model
    from duckmind.expression_sweep.runner import head_offsets

    assert set(looks.CLIPS).isdisjoint(content.CLIPS)
    assert not any(name in content.VERB_BY_ID for name in looks.CLIPS)
    # half down = np +0.5, hp −0.5 (spec §4.1): the survey's neck −0.125 / head +0.175 rad
    assert head_offsets(looks.HALF_DOWN) == pytest.approx([-0.125, 0.175, 0.0, 0.0])
    assert head_offsets(looks.UP) == pytest.approx([0.0, -0.35, 0.0, 0.0])  # up +20°
    assert content.frame_at(looks.CLIPS["look.glance_down"], 500) == looks.HALF_DOWN
    assert content.frame_at(looks.CLIPS["look.up"], 800) == looks.UP
    assert content.frame_at(looks.CLIPS["look.seated"], 300) == {}  # the seated HOME head
    assert looks.LOOKS == ("look.glance_down", "look.up", "look.seated")
    for name in looks.CLIPS:
        for _, frame in looks.CLIPS[name]:
            assert all(abs(value) <= 1.0 for value in frame.values()), name
        peaks = peak_offsets(Variant(name, 1.0, 1.0, 1))
        for joint, trained in robotd_model.TRAINED_HEAD_RANGE.items():
            assert peaks[joint] <= trained, (name, joint)
        # Every frame is taken after the command has held still for the stillness window.
        for ms in looks.CAPTURES_MS[name]:
            steady = ms - looks.STILL_S * 1000
            assert content.frame_at(looks.CLIPS[name], ms) == content.frame_at(
                looks.CLIPS[name], steady
            ), (name, ms)


def test_looks_use_the_clip_machinery_without_touching_duck_core_digests():
    import hashlib
    import json

    from duckmind.expression_sweep import looks
    from duckmind.expression_sweep.clips import capture_ticks, clip_digest, frames

    for clip in ("look_around", "moon_gaze", "nod"):  # digests are the pre-looks recipe
        blob = json.dumps(
            {
                "frames": content.CLIPS[clip],
                "envelope": content.SEATED_ENVELOPE,
                "sign": content.JOINT_SIGN,
                "home": content.HEAD_HOME,
            },
            sort_keys=True,
        ).encode()
        assert clip_digest(clip) == hashlib.sha256(blob).hexdigest()[:16]
    assert frames("look.up") is looks.CLIPS["look.up"]
    with pytest.raises(KeyError):
        frames("no_such_clip")
    assert len(variants(looks.LOOKS)) == 3 * len(AMPLITUDES) * len(TEMPOS) * len(SIDES)
    assert duration_s(Variant("look.glance_down", 1.0, 1.0, 1)) == pytest.approx(1.3)
    assert capture_ticks(Variant("look.glance_down", 1.0, 1.0, 1)) == (25, 50)
    assert capture_ticks(Variant("look.glance_down", 1.0, 0.7, 1)) == (36, 71)
    assert capture_ticks(Variant("nod", 1.0, 1.0, 1)) == ()
    stream = trajectory(Variant("look.glance_down", 1.0, 1.0, -1), lead_s=0.0, tail_s=0.0)
    assert min(stream["head_yaw"]) == pytest.approx(-0.54)  # aimed to the other side


def test_looks_never_change_the_engine_caps_of_duck_core_clips(tmp_path):
    from duckmind.expression_sweep import looks

    result = _result("look_around", fail_at=0.8)
    plain = envelope.write_caps(result, tmp_path / "a.json")["engine"]
    with_looks = result | {
        "looks": {
            Variant(name, amplitude, tempo, side).key: {"sit": _stats(), "stand": _stats()}
            for name in looks.CLIPS
            for amplitude in AMPLITUDES
            for tempo in TEMPOS
            for side in SIDES
        }
    }
    assert envelope.write_caps(with_looks, tmp_path / "b.json")["engine"] == plain
    capped = envelope.look_caps(with_looks)
    assert set(capped) == set(looks.LOOKS)  # the HOME-hold control is not a look
    assert capped["look.up"] == {"sit": 1.0, "stand": 1.0}


# ------------------------------------------------------------ v3: counting and falling


def test_wilson_intervals_and_the_shape_of_every_rate():
    from duckmind.expression_sweep.stats import describe, proportion, wilson

    assert wilson(0, 10) == pytest.approx((0.0, 0.27753), abs=1e-4)
    assert wilson(5, 10) == pytest.approx((0.23659, 0.76341), abs=1e-4)
    assert wilson(10, 10) == pytest.approx((0.72247, 1.0), abs=1e-4)
    assert wilson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson(3, 2)
    rate = proportion(12, 256)
    assert rate["n"] == 256 and rate["count"] == 12 and rate["rate"] == pytest.approx(0.04688)
    assert rate["wilson95"][0] < 12 / 256 < rate["wilson95"][1]
    spread = describe([1.0, 2.0, 3.0, float("nan")])
    assert spread["n"] == 3 and spread["mean"] == 2.0 and spread["p50"] == 2.0
    assert describe([])["n"] == 0


def test_robotd_fall_verdict_needs_200_ms_past_60_degrees():
    from duckmind.expression_sweep import falls

    assert math.degrees(math.acos(-falls.FALL_GRAVITY_Z)) == pytest.approx(falls.FALL_TILT_DEG)
    assert falls.debounce_ticks(50) == 10
    assert not any(falls.robotd_fallen([70.0] * 9))
    assert falls.robotd_fallen([70.0] * 10)[-1]
    # One upright sample clears the accumulator, exactly as robotd.
    verdict = falls.robotd_fallen([70.0] * 9 + [59.0] + [70.0] * 9)
    assert not any(verdict)
    assert not any(falls.robotd_fallen([60.0] * 20))  # past, not at, the threshold
    assert falls.next_run(4, True) == 5 and falls.next_run(4, False) == 0


def test_the_posture_schedule_is_robotds_sit_machine_and_the_v2_cycle():
    from duckmind.expression_sweep import falls

    s = falls.PostureSchedule()
    assert (s.rise_at, s.stand_at, s.rise_end, s.sit_at) == (100, 150, 200, 550)
    assert (s.sit_end, s.cycle_end, s.end) == (700, 800, 1700)
    for nxt in range(1, s.end):
        seated_net = nxt < s.stand_at or nxt >= s.sit_at  # v2 run_posture
        sit_flag = nxt < s.rise_at or nxt >= s.sit_at
        mode = s.scripted_mode(nxt)
        assert (mode != "up") == seated_net and (mode == "sit") == sit_flag, nxt
    assert [s.phase(t) for t in (99, 100, 199, 200, 549, 550, 699, 700, 799, 800)] == [
        "settle",
        "rise",
        "rise",
        "stand",
        "stand",
        "sit",
        "sit",
        "seated",
        "seated",
        "after",
    ]
    assert falls.PostureSchedule.from_dict(s.describe()) == s


def _posture_row(**fields):
    row = {
        "scenario": "posture",
        "task": "T",
        "reset": False,
        "settle_tilt_deg": 3.0,
        "settle_fallen": False,
        "fall_tick": None,
        "fall_mode": None,
        "fall_gravity": None,
        "landed_gravity": None,
        "tilt45_tick": None,
        "upright_tick": None,
        "righted_tick": None,
        "z_righted": None,
        "refell_tick": None,
    }
    return row | fields


def test_phase_falls_count_each_duck_once_among_those_still_on_script():
    from duckmind.expression_sweep import falls

    s = falls.PostureSchedule()
    rows = [_posture_row() for _ in range(6)]
    rows += [
        _posture_row(fall_tick=160, tilt45_tick=150),  # rise (during robotd's 1 s rise)
        _posture_row(fall_tick=195, tilt45_tick=190),  # still rise: 1 s after the handover
        _posture_row(fall_tick=300, tilt45_tick=290),  # stand
        _posture_row(tilt45_tick=600),  # sit: a near-fall only
        _posture_row(fall_tick=720, tilt45_tick=705),  # a late seated fall
        _posture_row(settle_tilt_deg=62.0, fall_tick=400),  # tipped before any action
        _posture_row(settle_fallen=True),
        _posture_row(reset=True, fall_tick=300),  # a nan reset: not a duck's number
    ]
    # The runner's own verdict wins over a rounded tilt that sits right at the threshold.
    assert falls.settle_bad(_posture_row(settle_tilt_deg=30.0, settle_bad=True))
    assert not falls.settle_bad(_posture_row(settle_tilt_deg=30.0))
    out = falls.summarize_posture(rows, s)
    assert out["episodes_total"] == 14 and out["resets"] == 1 and out["settled"] == 11
    assert out["settle_tilt"]["count"] == 2 and out["settle_tilt"]["n"] == 13
    phases = out["phases"]
    assert (phases["rise"]["robotd"]["count"], phases["rise"]["robotd"]["n"]) == (2, 11)
    assert (phases["stand"]["robotd"]["count"], phases["stand"]["robotd"]["n"]) == (1, 9)
    assert (phases["sit"]["robotd"]["count"], phases["sit"]["robotd"]["n"]) == (0, 8)
    assert (phases["seated"]["robotd"]["count"], phases["seated"]["robotd"]["n"]) == (1, 8)
    assert (phases["sit"]["tilt45"]["count"], phases["sit"]["tilt45"]["n"]) == (1, 8)
    assert phases["any"]["robotd"]["count"] == 4 and phases["any"]["tilt45"]["count"] == 5
    assert phases["rise"]["robotd"]["per_cycle_rate"] == pytest.approx(2 / 11, abs=1e-4)


def test_righting_is_counted_per_try_with_the_pause_in_between():
    from duckmind.expression_sweep import falls

    s = falls.PostureSchedule()
    hz = s.hz

    def fallen(after_s, *, landed, up_after_s=None):
        fall = 300
        up = None if up_after_s is None else fall + round(up_after_s * hz)
        return _posture_row(
            fall_tick=fall,
            tilt45_tick=fall - 12,
            landed_gravity=landed,
            righted_tick=None if after_s is None else fall + round(after_s * hz),
            upright_tick=up,
            z_righted=0.11,
        )

    rows = [
        fallen(3.0, landed=[0.9, 0.1, -0.2], up_after_s=2.0),  # try 1, face down
        fallen(9.0, landed=[-0.9, 0.0, 0.2], up_after_s=8.0),  # up during the pause
        fallen(12.0, landed=[-0.8, 0.3, 0.1], up_after_s=11.0),  # try 2, face up
        fallen(19.0, landed=[0.1, 0.9, 0.0], up_after_s=18.0),  # too late: not up
        fallen(None, landed=[0.2, -0.9, 0.0]),  # never
    ]
    out = falls.righting(rows, s)
    assert out["falls"] == 5
    assert out["try1"]["count"] == 1 and out["by_try2"]["count"] == 3
    assert (out["try2_after_try1_failed"]["count"], out["try2_after_try1_failed"]["n"]) == (2, 4)
    assert out["upright_after_s"]["n"] == 3 and out["upright_after_s"]["p50"] == 8.0
    assert out["by_landing"]["back"]["falls"] == 2 and out["by_landing"]["side"]["falls"] == 2
    assert out["by_landing"]["front"]["try1"]["count"] == 1
    assert out["by_phase"]["stand"]["falls"] == 5
    assert out["trunk_above_9cm_when_up"]["count"] == 3
    assert falls.landing(None) is None


def test_look_summary_gates_against_the_home_hold_control():
    from duckmind.expression_sweep import looks, summary

    def row(clip, drift, *, fell=False, amplitude=1.0):
        return {
            "posture": "stand",
            "clip": clip,
            "amplitude": amplitude,
            "variant": Variant(clip, amplitude, 1.0, 1).key,
            "settle_bad": False,
            "reset": False,
            "fell": fell,
            "fell_robotd": fell,
            "tilt": 8.0,
            "tilt_rise": 2.0,
            "drift": drift,
            "rest_pitch": -12.0,
            "rest_height": 0.233,
            "cam_pitch": [-25.0, -24.0],
            "cam_height": [0.22, 0.22],
            "cam_still": [True, False],
        }

    rows = [row(looks.CONTROL, 0.02) for _ in range(20)]
    rows += [row("look.up", 0.025) for _ in range(20)]
    rows += [row("look.glance_down", 0.05) for _ in range(20)]
    rows += [row("look.seated", 0.02) for _ in range(19)] + [row("look.seated", 0.02, fell=True)]
    rows += [row("look.up", 0.3, fell=True, amplitude=0.35)]  # outside the design pose
    out = summary.summarize_looks(rows)["look.up"]["stand"]
    assert out["gate"]["passes"] and out["falls_robotd"]["n"] == 20
    assert out["whole_grid"]["falls_robotd"]["count"] == 1
    assert out["captures"][0]["pitch_deg"]["mean"] == -25.0
    assert out["captures"][1]["still"]["count"] == 0
    stand = summary.summarize_looks(rows)
    assert not stand["look.glance_down"]["stand"]["gate"]["passes"]  # drifts 3 cm more
    assert not stand["look.seated"]["stand"]["gate"]["passes"]  # one fall
    assert "gate" not in stand[looks.CONTROL]["stand"]


def test_head_walk_summary_compares_each_pose_with_level():
    from duckmind.expression_sweep import summary

    def row(head, speed, yaw, *, fell=False):
        return {
            "head": head,
            "vx": 0.3,
            "vyaw": 0.0,
            "seconds": 6.0,
            "head_offsets_rad": [],
            "reset": False,
            "fell": fell,
            "fell_robotd": fell,
            "tilt": 10.0,
            "distance_m": speed * 6,
            "speed_mps": speed,
            "steady_speed_mps": speed * 1.3,
            "yaw_rate_dps": yaw,
            "steady_yaw_rate_dps": yaw,
            "overshoot_m": 0.05,
            "cam_pitch_deg": -12.0,
            "cam_height_m": 0.23,
        }

    rows = [row("level", 0.11 + 0.001 * (i % 3), -8.0) for i in range(30)]
    rows += [row("half_down", 0.055 + 0.001 * (i % 3), -4.0) for i in range(30)]
    rows += [row("half_down", 0.0, 0.0, fell=True)]
    out = summary.summarize_head_walk(rows)
    assert list(out)[:2] == ["level", "half_down"]
    assert out["half_down"]["falls_robotd"]["count"] == 1
    assert out["half_down"]["speed_mps"]["n"] == 30  # a fallen walk has no speed
    compare = out["compare_to_level"]["half_down"]["speed_mps"]
    assert compare["diff"] == pytest.approx(-0.055, abs=1e-6)
    assert compare["ratio_of_means"] == pytest.approx(0.056 / 0.111, abs=1e-3)


def test_the_trials_file_alone_recomputes_every_summary(tmp_path):
    from duckmind.expression_sweep import falls, summary

    schedule = falls.PostureSchedule()
    header = {"scenario": "header", "schedule": schedule.describe()}
    rows = [_posture_row(task="A", fall_tick=300, righted_tick=450, upright_tick=401)]
    rows += [_posture_row(task="B") for _ in range(3)]
    look = {
        "scenario": "look",
        "posture": "sit",
        "clip": "look.up",
        "amplitude": 1.0,
        "tempo": 1.0,
        "side": 1,
        "variant": Variant("look.up", 1.0, 1.0, 1).key,
        "settle_bad": False,
        "fell": False,
        "fell_robotd": False,
        "tilt": 3.0,
        "tilt_rise": 0.5,
        "torque_ratio": 0.2,
        "strain_frac": 0.0,
        "contacts": 0,
        "track_ratio": 0.6,
        "rms_error": 0.02,
        "drift": 0.001,
        "reset": False,
        "rest_pitch": 0.1,
        "rest_height": 0.19,
        "cam_pitch": [14.0],
        "cam_height": [0.2],
        "cam_still": [True],
    }
    rows += [look, look | {"cam_pitch": [16.0]}]
    path = tmp_path / "run.trials.jsonl"
    summary.write_rows(path, [header], mode="w")
    summary.write_rows(path, rows)
    rebuilt = summary.recompute(path)
    assert rebuilt == summary.summarize(header, rows)
    assert set(rebuilt["posture"]) == {"A", "B"}
    assert rebuilt["posture"]["A"]["righting"]["try1"]["count"] == 1
    stats = rebuilt["looks"][Variant("look.up", 1.0, 1.0, 1).key]["sit"]
    assert stats["episodes"] == 2 and stats["captures"][0]["pitch_deg"] == 15.0
    assert stats["falls_robotd"] == 0 and stats["resets"] == 0


def test_a_reset_episode_never_counts_for_or_against_a_clip():
    from duckmind.expression_sweep.runner import aggregate

    assigned = [Variant("nod", 1.0, 1.0, 1)] * 3
    metrics = {
        "settle_bad": [False, False, False],
        "reset": [False, True, False],
        "fell": [False, True, False],
        "fell_robotd": [False, True, False],
        "tilt": [2.0, 90.0, 3.0],
        "tilt_rise": [0.5, 80.0, 1.0],
        "torque_ratio": [0.3, 1.0, 0.4],
        "strain_frac": [0.0, 0.9, 0.0],
        "contacts": [0, 5, 0],
        "track_ratio": [0.9, 0.2, 0.8],
        "rms_error": [0.02, 0.4, 0.03],
        "drift": [0.001, 1.0, 0.002],
    }
    stats = aggregate(assigned, metrics)[Variant("nod", 1.0, 1.0, 1).key]
    assert (stats["falls"], stats["falls_robotd"], stats["resets"]) == (0, 0, 1)
    assert stats["track_ratio"] == pytest.approx(0.85)  # over the judged episodes only


def test_the_plan_reproduces_v2_and_sizes_the_v3_run():
    import argparse

    from duckmind.expression_sweep import runner

    v2 = [runner.Batch("v2", envs, ticks) for envs, ticks in runner.V2_BATCHES]
    assert runner.estimate_seconds(v2, runner.V2_BUILDS)["seconds"] == pytest.approx(519.2)
    args = runner.parser().parse_args(["--plan", "--scenarios", "look,posture,headwalk"])
    batches, builds = runner.plan(args)
    by = {}
    for batch in batches:
        by.setdefault(batch.scenario, []).append((batch.envs, batch.ticks))
    assert by["look"] == [(74 * 96, 275)] * 2
    assert by["posture"] == [(4096, 1700)] * 2
    assert by["headwalk"] == [(1024, 500)]
    assert builds == 5
    estimate = runner.estimate_seconds(batches, builds)
    assert estimate["low"] <= estimate["seconds"] <= estimate["high"] < 900
    assert isinstance(args, argparse.Namespace)
    with pytest.raises(SystemExit):
        runner.plan(runner.parser().parse_args(["--scenarios", "look,dance"]))
