"""P3: mediad's auto-exposure port against every unit test in ``mediad/src/exposure.rs``.

Each test below is the Rust test of the same name, line for line (spec §11.7 P3). Pure Python:
no numpy, no MuJoCo.
"""

from dataclasses import replace

from duckmind.world import camera_model as cm
from duckmind.world.camera_model import (
    ABSENT,
    CAPTURE_FORMAT,
    DEADBAND,
    HARD_LINES,
    MAX_ANALOGUE,
    MAX_DIGITAL,
    SOFT_LINES,
    TARGET_Y,
    UNKNOWN,
    Ae,
    AeLoop,
    Controls,
    mean_luma,
    told,
    worth_reading_back,
    worth_writing_gain,
)


def frame(luma: int) -> bytes:
    # UYVY, so half the bytes are chroma; they are set to something that would be a *different*
    # mean if the walk read them by mistake.
    return bytes(luma if index % 2 == 1 else 200 for index in range(1280 * 720 * 2))


def brightness(c: Controls) -> float:
    return float(c.exposure) * float(c.analogue_gain) * float(c.gain)


def test_luma_comes_from_the_luma_bytes():
    assert mean_luma(frame(64)) == 64.0


def test_a_format_we_cannot_read_is_not_guessed_at():
    assert mean_luma(frame(64), "NV12") is None
    assert mean_luma(frame(64), CAPTURE_FORMAT) == 64.0


def test_a_picture_at_the_setpoint_is_left_alone():
    ae = Ae.starting_at(600, 1024)
    assert ae.step(TARGET_Y) is None
    # And within the deadband, either side.
    assert ae.step(TARGET_Y * 1.1) is None
    assert ae.step(TARGET_Y / 1.1) is None


def test_a_dark_picture_gets_more_light_and_a_bright_one_less():
    ae = Ae.starting_at(600, 1024)
    before = ae.controls()
    darker = ae.step(20.0)
    assert darker is not None, "a step out of the deadband"
    assert brightness(darker) > brightness(before)

    ae = Ae.starting_at(600, 1024)
    before = ae.controls()
    brighter = ae.step(220.0)
    assert brighter is not None, "a step out of the deadband"
    assert brightness(brighter) < brightness(before)


def test_light_is_spent_on_the_shutter_before_the_gains():
    # From the dimmest state there is, the first thing that grows is the shutter, and neither
    # gain moves off 1x until the soft cap is reached.
    ae = Ae(exposure=4.0, analogue=1.0, digital=1.0, written=None)
    step = ae.step(2.0)
    assert step is not None
    assert step.exposure > 4
    assert step.exposure <= SOFT_LINES
    assert (step.analogue_gain, step.gain) == (256, 256)


def test_a_room_darker_than_the_sensor_can_reach_stops_being_written_to():
    ae = Ae.starting_at(600, 1024)
    steps = 0
    for _ in range(200):
        if ae.step(1.0) is not None:
            steps += 1
    assert steps < 10, f"still writing after settling at the ceiling: {steps} writes"
    # And it is at the ceiling, not merely quiet.
    pinned = ae.controls()
    assert float(pinned.exposure) == HARD_LINES
    assert float(pinned.analogue_gain) == MAX_ANALOGUE * 256.0
    # Light returns: the next step must write again rather than stay stuck on "unchanged".
    assert ae.step(TARGET_Y * 4.0) is not None, "a bright scene must move it"


def test_the_re_assert_writes_the_same_values_and_restarts_the_heartbeat():
    ae = Ae.starting_at(600, 1024)
    for _ in range(50):
        ae.step(1.0)
    quiet = ae.controls()
    assert ae.step(1.0) is None, "settled"
    # What the heartbeat sends: the values as they stand, not a recomputation.
    assert ae.current() == quiet
    # And it counts as written, so the next quiet tick is quiet again.
    assert ae.step(1.0) is None


def test_the_shutter_never_asks_for_a_longer_frame_than_the_sensor_has():
    ae = Ae.starting_at(600, 1024)
    for _ in range(200):
        ae.step(1.0)
    pinned = ae.controls()
    assert float(pinned.exposure) <= HARD_LINES, pinned
    assert float(pinned.analogue_gain) <= MAX_ANALOGUE * 256.0, pinned
    assert float(pinned.gain) <= MAX_DIGITAL * 256.0, pinned


def test_the_digital_gain_is_written_on_the_way_back_down():
    assert worth_writing_gain(4096, told(256)), "putting it up"
    assert worth_writing_gain(256, told(4096)), "and taking it back down"
    assert not worth_writing_gain(256, told(256)), "1x it was already told"


def test_a_run_that_has_said_nothing_yet_states_the_gain():
    assert worth_writing_gain(256, UNKNOWN), "1x is a claim"
    assert worth_writing_gain(4096, UNKNOWN)


def test_the_heartbeat_forgets_a_value_but_not_a_missing_control():
    forgotten = told(256).forget()
    assert forgotten == UNKNOWN
    assert worth_writing_gain(256, forgotten), "so 1x goes out again"
    assert ABSENT.forget() == ABSENT, "a node does not grow the control"


def test_a_node_without_a_digital_gain_is_not_written_to():
    assert not worth_writing_gain(256, ABSENT)
    assert not worth_writing_gain(4096, ABSENT)


def test_a_room_that_gets_brighter_asks_for_1x_digital_gain_back():
    ae = Ae.starting_at(600, 1024)
    put_up = False
    for _ in range(50):
        step = ae.step(1.0)
        if step is not None:
            put_up |= step.gain > 256
    assert put_up, "a dark room must reach for digital gain"
    handed_back = False
    for _ in range(50):
        step = ae.step(250.0)
        if step is not None:
            handed_back |= step.gain == 256
    assert handed_back, "a bright room must ask for 1x back"


def test_the_sensor_is_only_believed_about_a_value_it_was_not_already_at():
    assert not worth_reading_back(False, 600, 600)
    assert worth_reading_back(False, 601, 600)
    assert not worth_reading_back(True, 900, 600)


def test_the_loop_converges_rather_than_ringing():
    # A crude model of the sensor: luma is proportional to the product of the controls, through
    # the gamma curve the damping term exists to account for. The picture starts eight times
    # too dark; what matters is that it settles and stays settled.
    reference = 600.0 * 4.0

    def luma(c: Controls) -> float:
        product = float(c.exposure) * (c.analogue_gain / 256.0) * (c.gain / 256.0)
        return (TARGET_Y / 8.0) * (product / reference) ** (1.0 / 0.45)

    ae = Ae.starting_at(600, 1024)
    seen = ae.controls()
    steps = 0
    for _ in range(60):
        after = ae.step(luma(seen))
        if after is None:
            break
        seen = after
        steps += 1
    settled = luma(seen)
    assert abs(settled - TARGET_Y) / TARGET_Y < DEADBAND, f"settled at {settled:.0f} after {steps}"
    assert steps < 30, f"took {steps} steps to settle"


# ── beyond the Rust file: the port's own arithmetic and the simulated thread ─────────


def test_step_arithmetic_matches_the_rust_order_of_operations():
    # One step from the pinned start at a quarter of the target, by hand from exposure.rs:
    # budget = 600 * 4 * 1 * 4^0.6, then shutter to 600, analogue 11 max, shutter again, digital.
    ae = Ae.starting_at(600, 1024)
    step = ae.step(TARGET_Y / 4)
    budget = 600.0 * 4.0 * 1.0 * 4.0**0.6  # 5513.75: the shutter stays at 600, analogue 9.19x
    assert step == Controls(exposure=600, analogue_gain=int(budget / 600.0 * 256.0), gain=256)
    assert step.analogue_gain == 2352
    assert abs(ae.exposure * ae.analogue * ae.digital - budget) < 1e-9


def test_nan_and_zero_luma_count_as_the_darkest_reading():
    a, b, c = Ae.starting_at(600, 1024), Ae.starting_at(600, 1024), Ae.starting_at(600, 1024)
    assert a.step(float("nan")) == b.step(0.0) == c.step(1.0)


def test_the_simulated_thread_writes_what_the_sensor_then_holds_and_reasserts():
    loop = AeLoop()
    assert loop.sensor == Controls(600, 1024, 256), "the pipeline pins shutter and analogue gain"
    assert loop.tick(None) is None and loop.writes == 0, "no frame, no step"
    wrote = loop.tick(20.0)
    assert wrote is not None and loop.sensor == wrote
    # At the setpoint: quiet ticks, then the heartbeat re-states the same values.
    for _ in range(cm.REASSERT_TICKS):
        assert loop.tick(TARGET_Y) is None
    assert loop.settled()
    again = loop.tick(TARGET_Y)
    assert again == wrote and loop.sensor == wrote
    assert loop.tick(TARGET_Y) is None, "the heartbeat restarts"


def test_controls_are_frozen_values():
    c = Controls(600, 1024, 256)
    assert replace(c, gain=512).gain == 512 and c.gain == 256
