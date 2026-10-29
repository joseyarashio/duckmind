"""s1b:fix: the page's witness judges a trained skill window by the skill's own detectors.

While robotd runs a trained skill it ignores the head command; the witness used to grade such a
window against the motor's head command and the brain's named parts, so a real 27 mm crouch read
「身體只做到一部分：頭往右轉沒有做出來」 (tr:gate live, 4 of 6). Synthetic ticks at 50 Hz."""

from __future__ import annotations

from duckmind.simweb.neuro_witness import Tick, Witness


def _ramp(t: float, start: float, dur: float) -> float:
    return min(max((t - start) / dur, 0.0), 1.0)


def _run(*, skill: str | None, dip_m: float, neck_rad: float, seconds: float = 4.0):
    """At 1.0 s the brain reports a startle naming a neck pull-back and a turn to the right
    (which robotd is asked for); robotd runs ``skill`` from 1.2 s (0.5 s + 0.8 s unwind) and the
    trunk dips ``dip_m`` with the neck pulled back ``neck_rad``, the head never turning."""
    w = Witness()
    closed: list[dict] = []
    for i in range(int(seconds * 50)):
        t = i * 0.02
        s = 1.2
        down = _ramp(t, s + 0.02, 0.2) - _ramp(t, s + 0.5, 0.8)
        policy = "walk"
        if skill is not None and s <= t < s + 0.5:
            policy = skill
        elif skill is not None and s + 0.5 <= t < s + 1.3:
            policy = f"{skill}:unwind"
        head = (neck_rad * down, 0.0, 0.0, 0.0)
        cmd = (0.2 * _ramp(t, 1.0, 0.25), -0.1 * _ramp(t, 1.0, 0.25), -0.3 * _ramp(t, 1.0, 0.25),
               0.0)  # fmt: skip
        tick = Tick(
            t_ms=int(round(t * 1000)),
            head=head,
            head_cmd=cmd,
            policy=policy,
            trunk=(0.0, 0.0, 0.117 - dip_m * down, 0.0),
            trunk_t=t,
            holding="startle" if 1.0 <= t < 1.6 else None,
            behaviour="startle" if 1.0 <= t < 1.6 else "stand_still",
        )
        events = []
        if t >= 1.0:
            events = [{"t_ms": 1000, "reflex": "startle", "stimulus": "sound:clap",
                       "zh": "嚇一跳", "parts": ["neck", "yaw_right"]}]  # fmt: skip
        closed += w.feed(tick, events)
    return w, closed


def test_a_trained_crouch_is_done_by_its_own_detectors() -> None:
    w, audits = _run(skill="startle_4", dip_m=0.027, neck_rad=0.44)
    startle = [a for a in audits if a.get("reflex") == "startle"]
    assert len(startle) == 1
    a = startle[0]
    assert a["verdict"] == "done", a["zh"]
    assert a["skill"]["skill"]["entry"] == "startle_4"
    assert "蹲低 2.7 公分" in a["zh"] and "脖子往後縮" in a["zh"]
    assert "頭轉向右邊" in a["zh"]  # said as information: the skill does not turn the head
    assert a["head_ok_without_skill"] in ("partial", "not_done")  # the old verdict, for the record


def test_a_skill_that_barely_moved_is_not_done() -> None:
    _, audits = _run(skill="startle_4", dip_m=0.004, neck_rad=0.05)
    a = [x for x in audits if x.get("reflex") == "startle"][0]
    assert a["verdict"] == "not_done" and a["zh"].startswith("身體做不到")


def test_without_a_skill_the_witness_judges_as_before() -> None:
    _, audits = _run(skill=None, dip_m=0.0, neck_rad=0.0)
    a = [x for x in audits if x.get("reflex") == "startle"][0]
    assert "skill" not in a and a["verdict"] in ("partial", "not_done")


def test_now_names_the_trained_skill() -> None:
    w = Witness()
    for i in range(60):
        t = i * 0.02
        w.feed(Tick(t_ms=int(t * 1000), head=(0.3, 0.0, 0.0, 0.0), policy="freeze_3",
                    trunk=(0.0, 0.0, 0.095, 0.0), trunk_t=t))  # fmt: skip
    now = w.now()
    assert now is not None and "蹲低定住" in now["zh"] and now["posture"] == "skill"
