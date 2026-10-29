"""fid:page (2026-09-28): the page says what the body measurably did after each reflex.

The owner judges only what the physical body visibly does. :mod:`duckmind.simweb.neuro_witness`
watches robotd's measured joints, robotd's own head command and the physics truth after every
reflex the brain reports, and says it in plain words — or 「身體做不到」 when the body did not do
what it was asked. Synthetic ticks at 50 Hz, no physics.
"""

from __future__ import annotations

import math

from duckmind.simweb.neuro_witness import HALTING, Tick, Witness, head_words

DEG = math.pi / 180.0


def _ramp(t: float, start: float, rise: float) -> float:
    """0 before ``start``, a smooth rise to 1 over ``rise`` seconds."""
    if t <= start:
        return 0.0
    u = min(1.0, (t - start) / rise)
    return u * u * (3 - 2 * u)


def _run(
    *,
    reflex: str = "startle",
    cmd_pitch_deg: float = 0.0,
    got_pitch_deg: float = 0.0,
    cmd_yaw_deg: float = 0.0,
    got_yaw_deg: float = 0.0,
    speed_before: float = 0.0,
    speed_after: float | None = None,
    stop_cmd: bool = True,
    seconds: float = 3.5,
    onset_s: float = 1.0,
    holding: str | None = None,
) -> tuple[Witness, list[dict]]:
    """A duck at 50 Hz; at ``onset_s`` the brain reports ``reflex``. The head command
    (nose up by leaning the neck back, and a turn) rises over 0.25 s; the measured head rises
    to ``got_*`` over 0.3 s; the trunk walks at ``speed_before`` and after the reflex slows
    to ``speed_after`` (default: the same)."""
    witness = Witness()
    closed: list[dict] = []
    x = 0.0
    after = speed_before if speed_after is None else speed_after
    for i in range(int(seconds * 50)):
        t = i * 0.02
        t_ms = int(round(t * 1000))
        c = _ramp(t, onset_s, 0.25)
        g = _ramp(t, onset_s + 0.05, 0.3)
        # nose up = neck back (+) and head_pitch up (−): half each
        cmd = (
            0.5 * cmd_pitch_deg * DEG * c,
            -0.5 * cmd_pitch_deg * DEG * c,
            cmd_yaw_deg * DEG * c,
            0.0,
        )
        head = (
            0.5 * got_pitch_deg * DEG * g,
            -0.5 * got_pitch_deg * DEG * g,
            got_yaw_deg * DEG * g,
            0.0,
        )
        blend = _ramp(t, onset_s, 0.5)
        v = speed_before + (after - speed_before) * blend
        x += v * 0.02
        walking_cmd = 0.4 if speed_before > 0 else 0.0
        twist = (0.0 if (stop_cmd and t >= onset_s) else walking_cmd, 0.0)
        tick = Tick(
            t_ms=t_ms,
            head=head,
            head_cmd=cmd,
            twist_cmd=twist,
            policy="walk" if speed_before > 0 else "stand",
            trunk=(x, 0.0, 0.12, 0.0),
            trunk_t=t,
            holding=holding if t >= onset_s else None,
        )
        events = []
        if t >= onset_s:
            events = [{"t_ms": int(onset_s * 1000), "reflex": reflex, "stimulus": "sound:clap"}]
        closed += witness.feed(tick, events)
    return witness, closed


def test_a_startle_the_body_performs_is_said_with_its_size_and_speed():
    _, audits = _run(cmd_pitch_deg=20, got_pitch_deg=18)
    assert len(audits) == 1
    audit = audits[0]
    assert audit["verdict"] == "done"
    assert audit["head"]["pitch"]["deg"] == 18.0
    # fid:audit's gesture index: visible (V ≥ 1) within 0.40 s of the reflex
    assert audit["head_ok"] == "done" and audit["v_peak"] >= 1.0 and audit["v_reach_s"] <= 0.40
    # the neck leaned back 9°: the head was pulled back (縮)
    assert audit["zh"].startswith("身體：頭往後縮 18°（0.")
    assert audit["head"]["pitch"]["reach_s"] <= 0.35


def test_a_startle_the_body_cannot_do_says_so():
    _, audits = _run(cmd_pitch_deg=20, got_pitch_deg=2)
    audit = audits[0]
    assert audit["verdict"] == "not_done"
    assert audit["zh"].startswith("身體做不到：")
    assert "只動了 2°（要牠動 20°）" in audit["zh"]


def test_half_of_it_is_a_part_and_the_page_says_how_much():
    _, audits = _run(cmd_pitch_deg=20, got_pitch_deg=6)
    audit = audits[0]
    assert audit["verdict"] == "partial"
    assert audit["zh"].startswith("身體只做到一部分：")
    assert "6°（動得很小，要牠動 20°）" in audit["zh"]


def test_a_walking_duck_that_stops_is_said_to_stop_and_one_that_walks_on_is_not():
    _, stopped = _run(cmd_pitch_deg=20, got_pitch_deg=18, speed_before=0.12, speed_after=0.0)
    audit = stopped[0]
    # still (chord speed < 3 cm/s over 0.3 s) within 0.8 s, held ≥ 0.8 s: fid:audit's stop
    assert audit["legs"]["walking"] and audit["legs"]["ok"] == "done"
    assert audit["legs"]["still_s"] <= 0.8 and audit["legs"]["held_s"] >= 0.8
    assert audit["verdict"] == "done" and "停下來（0." in audit["zh"]
    _, walked_on = _run(cmd_pitch_deg=20, got_pitch_deg=18, speed_before=0.12, speed_after=0.12)
    audit = walked_on[0]
    assert audit["legs"]["still_s"] is None and audit["legs"]["ok"] == "not_done"
    assert audit["verdict"] == "not_done"
    assert "腳沒停下來" in audit["zh"] and audit["zh"].startswith("身體做不到：")
    # the brain's motor never zeroed the twist: the page says the legs were not told to stop
    _, untold = _run(
        cmd_pitch_deg=20, got_pitch_deg=18, speed_before=0.12, speed_after=0.12, stop_cmd=False
    )
    assert "腳沒收到停的指令" in untold[0]["zh"]


def test_an_orienting_turn_says_the_side_and_what_it_was_doing_anyway_does_not_count():
    _, audits = _run(reflex="orient", cmd_yaw_deg=-15, got_yaw_deg=-14)
    audit = audits[0]
    assert audit["reflex"] == "orient" and audit["verdict"] == "done"
    assert "頭轉到右邊 14°" in audit["zh"]  # where the head ended up pointing
    # the brain says it oriented, but robotd was not even asked to move the head: said so
    _, quiet = _run(reflex="orient", got_yaw_deg=1)
    assert quiet[0]["verdict"] == "not_done"
    assert quiet[0]["zh"] == "身體做不到：頭沒有看得出來的動作（也沒有叫頭動）"


def test_now_says_what_the_body_does_and_flags_a_halt_the_legs_ignore():
    witness, _ = _run(
        reflex="freeze",
        speed_before=0.12,
        speed_after=0.12,
        stop_cmd=False,
        holding="freeze",
        seconds=2.5,
    )
    now = witness.now()
    assert now is not None and now["posture"] == "walk"
    assert "走路 0.12 公尺/秒" in now["zh"]
    assert now["mismatch_zh"] == "腦說要停下來，腳還在走"
    standing, _ = _run(reflex="orient", cmd_yaw_deg=20, got_yaw_deg=20, seconds=1.2)
    now = standing.now()
    assert now["posture"] == "stand" and now["mismatch_zh"] is None
    assert "頭朝左邊" in now["zh"] and "頭在動" in now["zh"]


def test_each_reflex_is_audited_once_and_numbered():
    witness, audits = _run(cmd_pitch_deg=20, got_pitch_deg=18)
    assert [a["seq"] for a in audits] == [1]
    assert witness.since(0) == audits and witness.since(1) == []


def test_words():
    assert head_words("pitch", 12.4, neck=6) == "頭往後縮 12°"
    assert head_words("pitch", 12.4, neck=1) == "頭往後仰 12°"
    assert head_words("pitch", -9) == "低頭 9°"
    assert head_words("yaw", 7) == "頭往左轉 7°"
    assert head_words("roll", -5) == "頭往右歪 5°"
    assert "freeze" in HALTING and "orient" not in HALTING


def test_a_head_already_turned_past_what_it_looks_at_is_said_to_turn_back_toward_it():
    """The brain says 「頭轉向右邊」 (the thing is on its right); the head was already turned 40°
    right, so it turns 18° LEFT to face the thing 22° right: the row says where the head ended
    up, and which way it turned to get there."""
    witness = Witness()
    audits: list[dict] = []
    for i in range(150):
        t = i * 0.02
        u = _ramp(t, 1.05, 0.3)
        yaw = (-40.0 + 18.0 * u) * DEG
        cmd = (-40.0 + 18.0 * _ramp(t, 1.0, 0.25)) * DEG
        tick = Tick(
            t_ms=int(round(t * 1000)), head=(0.0, 0.0, yaw, 0.0), head_cmd=(0.0, 0.0, cmd, 0.0)
        )
        events = (
            [{"t_ms": 1000, "reflex": "orient", "stimulus": "visual:thing"}] if t >= 1.0 else []
        )
        audits += witness.feed(tick, events)
    audit = audits[0]
    assert audit["verdict"] == "done"
    assert "頭轉到右邊 22°（往左轉了 18°，0.2 秒內）" in audit["zh"]


def test_the_gesture_index_is_fid_audits():
    """V = max(beak-tip travel / 1.2 cm, head rotation / 10°), levers 6.5 / 8.4 / 6.0 cm."""
    from duckmind.simweb.neuro_witness import gesture_index

    zero = (0.0, 0.0, 0.0, 0.0)
    assert round(gesture_index((0.0, 0.0, 10 * DEG, 0.0), zero), 6) == 1.0  # 10° of yaw
    assert round(gesture_index((0.0, 0.0, 0.0, 10 * DEG), zero), 6) == 1.0  # roll: by rotation
    neck_only = 1.2 / 8.4  # rad: the beak moves 1.2 cm
    assert round(gesture_index((neck_only, 0.0, 0.0, 0.0), zero), 6) == 1.0


def _startle_late(reach_s: float) -> dict:
    witness = Witness()
    audits: list[dict] = []
    for i in range(200):
        t = i * 0.02
        g = _ramp(t, 1.0 + reach_s - 0.15, 0.3)  # V passes 1 about reach_s after the reflex
        head = (0.5 * 20 * DEG * g, -0.5 * 20 * DEG * g, 0.0, 0.0)
        cmd = (0.5 * 20 * DEG * _ramp(t, 1.0, 0.2), -0.5 * 20 * DEG * _ramp(t, 1.0, 0.2), 0.0, 0.0)
        events = [{"t_ms": 1000, "reflex": "startle", "stimulus": "sound:clap"}] if t >= 1.0 else []
        audits += witness.feed(Tick(t_ms=int(round(t * 1000)), head=head, head_cmd=cmd), events)
    return audits[0]


def test_a_startle_that_comes_late_is_only_a_part_of_one():
    on_time = _startle_late(0.2)
    assert on_time["verdict"] == "done"
    late = _startle_late(0.5)  # visible, but after 0.40 s: fid:audit's partial
    assert late["head_ok"] == "partial" and late["verdict"] == "partial"
    assert "但慢了" in late["zh"]


def _freeze(wobble_deg: float) -> dict:
    witness = Witness()
    audits: list[dict] = []
    for i in range(200):
        t = i * 0.02
        yaw = wobble_deg * DEG * 0.5 * math.sin(2 * math.pi * 1.5 * t) if t > 1.3 else 0.0
        events = [{"t_ms": 1000, "reflex": "freeze", "stimulus": "sound:clap"}] if t >= 1.0 else []
        audits += witness.feed(Tick(t_ms=int(round(t * 1000)), head=(0.0, 0.0, yaw, 0.0)), events)
    return audits[0]


def test_a_freeze_is_judged_by_how_still_the_head_holds():
    assert "頭定住不動" in _freeze(1.0)["zh"]
    wobbly = _freeze(8.0)
    assert wobbly["verdict"] == "not_done" and "頭沒定住（晃了 8°）" in wobbly["zh"]


def test_looking_straight_ahead_needs_no_turn():
    witness = Witness()
    audits: list[dict] = []
    for i in range(140):
        t = i * 0.02
        events = (
            [{"t_ms": 1000, "reflex": "orient", "stimulus": "visual:thing", "zh": "看正前方"}]
            if t >= 1.0
            else []
        )
        audits += witness.feed(
            Tick(
                t_ms=int(round(t * 1000)), head=(0.0, 0.0, 0.0, 0.0), head_cmd=(0.0, 0.0, 0.0, 0.0)
            ),
            events,
        )
    assert audits[0]["verdict"] is None and audits[0]["zh"] == "身體：看不出有動"


def test_a_startle_that_turns_but_never_pulls_back_says_the_pull_back_is_missing():
    """The head turned 21° in 0.2 s (visible in time: fid:audit's flinch), but the retract
    robotd was asked for (25° nose-up by the neck) never came: the row says so."""
    witness = Witness()
    audits: list[dict] = []
    for i in range(200):
        t = i * 0.02
        g = _ramp(t, 1.05, 0.2)
        c = _ramp(t, 1.0, 0.2)
        head = (0.0, 0.0, -21 * DEG * g, 0.0)
        cmd = (0.5 * 25 * DEG * c, -0.5 * 25 * DEG * c, -20 * DEG * c, 0.0)
        events = [{"t_ms": 1000, "reflex": "startle", "stimulus": "sound:clap"}] if t >= 1.0 else []
        audits += witness.feed(Tick(t_ms=int(round(t * 1000)), head=head, head_cmd=cmd), events)
    audit = audits[0]
    assert audit["missing"] == ["pitch"] and audit["verdict"] == "partial"
    assert audit["zh"].startswith(
        "身體只做到一部分：頭往後縮沒有做出來（要牠動 25°）、頭轉到右邊 21°"
    )
