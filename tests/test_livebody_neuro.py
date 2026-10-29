"""Prototype-mode hooks of the live body (duckmind.livebody), ported from the neural-duck
prototype: the 3D beak reads the simulator's measured jaw joint only when the world confirms a
jaw, else the commanded beak, else nothing. Pure fakes: no physics needed."""

import pytest


def _mouth_frame_body(*, jaw_present=True, fresh=True, command=0.6):
    import threading
    import time
    from collections import deque

    from duckmind.livebody import FRAMES_KEEP_S, LiveBody

    body = LiveBody.__new__(LiveBody)
    body.world_state = {"neuro": {"capabilities": {"jaw_present": jaw_present}}}
    body.world_at = time.monotonic() if fresh else time.monotonic() - 10.0
    body.mouth_cmd = command
    body._frames = deque(maxlen=int(FRAMES_KEEP_S * 60) + 20)
    body._frames_lock = threading.Lock()
    body._calls = []
    body._mouth_measured_rad_for_test = None
    return body


def test_3d_beak_uses_measured_wire_joint_nine_only_with_fresh_confirmed_jaw():
    import math

    from duckmind.livebody import FRAME_JOINTS

    body = _mouth_frame_body(jaw_present=True)
    base = {"sim_time": 1.0, "positions": [0.0] * 15}
    base["positions"][9] = math.radians(12.5)
    measured, source = body._mouth_readback(base)
    assert measured == pytest.approx(math.radians(12.5))
    assert source == "measured_sim"
    body._keep_frame(
        {
            "t": 1.0,
            "joints": {name: 0.0 for name in FRAME_JOINTS},
            "trunk": [0, 0, 0.1],
            "quat": [1, 0, 0, 0],
            "mouth_measured_rad": measured,
            "mouth_source": source,
        },
        1.0,
    )
    assert body._frames[-1]["m"] == pytest.approx(0.5, abs=0.001)
    assert body._frames[-1]["m_source"] == "measured_sim"


@pytest.mark.parametrize(
    "jaw_present,fresh,positions",
    [
        (False, True, [0.0] * 15),
        (None, True, [0.0] * 15),
        (True, False, [0.0] * 15),
        (True, True, [0.0] * 14),
    ],
)
def test_3d_beak_marks_command_fallback_when_jaw_readback_is_not_verified(
    jaw_present, fresh, positions
):
    from duckmind.livebody import FRAME_JOINTS

    body = _mouth_frame_body(jaw_present=jaw_present, fresh=fresh, command=0.37)
    measured, source = body._mouth_readback({"positions": positions})
    assert measured is None
    assert source == "commanded_fallback"
    body._keep_frame(
        {
            "t": 1.0,
            "joints": {name: 0.0 for name in FRAME_JOINTS},
            "trunk": [0, 0, 0.1],
            "quat": [1, 0, 0, 0],
            "mouth_measured_rad": measured,
            "mouth_source": source,
        },
        1.0,
    )
    assert body._frames[-1]["m"] == pytest.approx(0.37)
    assert body._frames[-1]["m_source"] == "commanded_fallback"


def test_3d_beak_marks_unavailable_when_neither_verified_readback_nor_command_exists():
    from duckmind.livebody import FRAME_JOINTS

    body = _mouth_frame_body(jaw_present=False, command=None)
    body._keep_frame(
        {
            "t": 1.0,
            "joints": {name: 0.0 for name in FRAME_JOINTS},
            "trunk": [0, 0, 0.1],
            "quat": [1, 0, 0, 0],
            "mouth_measured_rad": None,
            "mouth_source": "commanded_fallback",
        },
        1.0,
    )
    assert "m" not in body._frames[-1]
    assert body._frames[-1]["m_source"] == "unavailable"
