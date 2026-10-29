"""Plugging the live bridge into a running duck (live voice, LV2).

One call for the host (the simulator's driver, a test, later the talk panel's server):
:func:`attach_live` takes the duck's lanes (its integrated brain and the sound lane with the
microphone and the voice engine) and a :class:`~duckmind.live.session.LiveSession`, and makes

* the microphone's frames go up the session (``session.attach_mic``: audio, echo flag, pump),
* the same frames go to the duck's own voice detection (with the duck's voice bank as the
  reference for its own echo), which is what stops a call when a person speaks over it,
* the echo gate's rescue of a transcript the duck's own detector heard a person in,
* a stop cut the duck's own call in the microphone (``MicStream.cut_calls``; the page's sound
  lane reads the same ``vocal_seq`` bump),
* the duck's state goes to the cloud brain every :data:`CONTEXT_EVERY_S` seconds (what it is doing, its
  mood and drives, among them 想說話 and its closeness to the person it sees, whom it knows and how close it
  is to each, what its eyes see: :meth:`LiveBridge.snapshot`), unless the config says otherwise, and
* the brain step through the bridge (``brain.live``).

Nothing here starts a thread or a connection: the session's transport is the caller's.
"""

from __future__ import annotations

from typing import Any

from duckmind.live.bridge import BridgeConfig, LiveBridge
from duckmind.live.vad import BankReferences, VoiceDetector

CONTEXT_EVERY_S = 5.0


def attach_live(lanes: Any, session: Any, *, config: BridgeConfig | None = None) -> LiveBridge:
    """Wire ``session`` to the duck in ``lanes`` (see the module doc); returns the bridge. With no
    ``config`` the duck's state goes to the cloud every :data:`CONTEXT_EVERY_S` seconds."""
    sound = getattr(lanes, "sound", None)
    mic = getattr(sound, "mic", None)
    if mic is None:
        raise ValueError(
            "the duck has no microphone stream (DUCKMIND_LIVE_MIC=0?): nothing to listen with"
        )
    engine = getattr(sound, "engine", None)
    bank = getattr(engine, "bank", None)
    refs = BankReferences(bank, int(getattr(engine, "voice_seed", 0))) if bank is not None else None
    bridge = LiveBridge(
        session,
        config=config or BridgeConfig(context_every_s=CONTEXT_EVERY_S),
        vad=VoiceDetector(refs),
        on_stop=lambda _request: mic.cut_calls(),
    )
    session.attach_mic(mic)
    set_evidence = getattr(session, "set_human_evidence", None)
    if callable(set_evidence):
        # a transcript that overlaps the duck's own call is the person's, not its echo, when the
        # duck's own ear heard that person (the same evidence the grounding gate rescues a call with)
        set_evidence(bridge._human_between)
    mic.listen(lambda frame: bridge.hear(frame, mic.vocal_windows()))
    bridge.attach(lanes.brain)
    return bridge


__all__ = ["CONTEXT_EVERY_S", "attach_live"]
