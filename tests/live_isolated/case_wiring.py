"""Live voice, LV2: one call plugs the bridge into a running duck (``duckmind.live.wiring``): the
microphone's frames go up the session and to the duck's own voice detection, a stop cuts the duck's
call in the microphone, and the brain steps through the bridge. No network."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from live_bridge_rig import make_brain, run_brain

from duckmind.live.bridge import LiveBridge
from duckmind.live.wiring import attach_live
from duckmind.simweb.neuro_mic import MicStream

pytestmark = pytest.mark.usefixtures("no_network")


class Session:
    """What the wiring needs of a session: attach_mic, and the bridge's reads."""

    def __init__(self, clock: Any) -> None:
        self.clock = clock
        self.enabled = True
        self.state = "online"
        self.fed = 0

    def attach_mic(self, mic: Any) -> None:
        mic.listen(self._on_frame)

    def _on_frame(self, frame: Any) -> None:
        self.fed += 1

    def take_calls(self, limit: int | None = None) -> list[Any]:
        return []


def test_attach_live_connects_the_microphone_the_detector_the_stop_and_the_brain() -> None:
    brain, clock = make_brain()
    mic = MicStream(seed=7, wall=lambda: 0.0)
    lanes = SimpleNamespace(brain=brain, sound=SimpleNamespace(mic=mic, engine=None))
    session = Session(clock)
    bridge = attach_live(lanes, session)
    assert isinstance(bridge, LiveBridge) and brain.live is bridge and bridge.session is session
    for k in range(10):
        clock.set(round(0.02 * (k + 1), 6))
        mic.tick(clock.t, [], wall=clock.t)
    assert session.fed == 10 and bridge.vad is not None and bridge.vad.frames == 10
    # the stop reaches the microphone: the duck's own call in it ends now
    mic.tick(0.22, [], calls=[{"tag": "coo", "level": 0.8}], wall=0.22)
    assert mic.vocal_windows()[-1].t1 > 0.23
    assert bridge.on_stop is not None
    bridge.on_stop(None)  # type: ignore[arg-type]
    assert mic.vocal_windows()[-1].t1 <= 0.25
    run_brain(brain, clock, 1.0)
    assert bridge.faults == 0


class SessionWithEvidence(Session):
    """A session that takes the duck's own detector for its echo gate (``LiveSession`` does)."""

    evidence: Any = None

    def set_human_evidence(self, evidence: Any) -> None:
        self.evidence = evidence


def test_attach_live_hands_the_echo_gate_the_ducks_own_voice_detection() -> None:
    """A transcript that overlaps the duck's own call is the person's, not its echo, when the duck
    itself heard that person: the gate asks the same question the grounding gate asks about a call."""
    brain, clock = make_brain()
    mic = MicStream(seed=7, wall=lambda: 0.0)
    lanes = SimpleNamespace(brain=brain, sound=SimpleNamespace(mic=mic, engine=None))
    session = SessionWithEvidence(clock)
    bridge = attach_live(lanes, session)
    assert callable(session.evidence)
    assert session.evidence(0.0, 1.0) is False  # (nobody has spoken)
    bridge._spans.append([0.5, 0.8])  # the detector heard a person from 0.5 to 0.8 (the clock's s)
    assert session.evidence(0.0, 1.0) is True
    assert session.evidence(0.7, 0.75) is True
    assert session.evidence(2.0, 3.0) is False


def test_a_duck_with_no_microphone_stream_says_so() -> None:
    brain, clock = make_brain()
    lanes = SimpleNamespace(brain=brain, sound=SimpleNamespace(mic=None, engine=None))
    with pytest.raises(ValueError, match="microphone"):
        attach_live(lanes, Session(clock))
    with pytest.raises(ValueError, match="microphone"):
        attach_live(SimpleNamespace(brain=brain), Session(clock))


def test_attach_live_tells_the_cloud_the_ducks_state_and_drives_every_few_seconds() -> None:
    from live_bridge_rig import MOM_LOOK, frame, person_sight

    from duckmind.live import soul as S
    from duckmind.live.wiring import CONTEXT_EVERY_S

    class Talking(Session):
        def __init__(self, clock: Any) -> None:
            super().__init__(clock)
            self.pushed: list[Any] = []

        def set_context(self, context: Any) -> None:
            self.pushed.append(context)

    brain, clock = make_brain()
    mic = MicStream(seed=7, wall=lambda: 0.0)
    lanes = SimpleNamespace(brain=brain, sound=SimpleNamespace(mic=mic, engine=None))
    session = Talking(clock)
    bridge = attach_live(lanes, session)
    assert bridge.config.context_every_s == CONTEXT_EVERY_S
    run_brain(
        brain,
        clock,
        12.0,
        frames=lambda t: frame(
            int(round(t * 1000)), person_sight(1, look=MOM_LOOK, bearing=0.0, range_m=1.4)
        ),
    )
    assert len(session.pushed) in (3, 4)
    last = session.pushed[-1]
    assert isinstance(last, S.ContextSnapshot)
    assert "talk" in last.drives and "bond" in last.drives
    text = last.text()
    assert "想說話" in text and "親近" in text
    # (a config of the host's own is its own: nothing is pushed unless it asks)
    quiet = attach_live(lanes, Talking(clock), config=type(bridge.config)())
    assert quiet.config.context_every_s is None
