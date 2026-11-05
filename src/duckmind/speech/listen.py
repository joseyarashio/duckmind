"""One spoken clip → what the duck should take it as: the one path the server and the measurements share.

1. :meth:`duckmind.speech.asr.Recognizer.hear` writes down the words, on this computer;
2. :func:`duckmind.neuro.commands.understand` reads them: a command, the owner's praise / scold, or nothing;
3. words that are none of those (and not a negated command) are compared by how they sound with the
   table's phrases (:func:`duckmind.speech.nearest.nearest`): 「按我」 is taken as 「看我」 when nothing
   else is about as close.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from duckmind.neuro.commands import Understood, understand
from duckmind.speech import asr
from duckmind.speech.nearest import nearest

SOUNDED_CONFIDENCE = 0.5  # (a meaning found by sound is less sure than one read from the words)


@dataclass(frozen=True, slots=True)
class Spoken:
    """A clip as heard (``transcript``) and understood; ``by_sound``: the table phrase it was taken as when the
    words themselves were not a command (None when they were, or when nothing was close)."""

    transcript: asr.Transcript
    understood: Understood
    by_sound: str | None = None

    @property
    def meaning(self) -> str:
        """``<verb>``, ``praise``, ``scold`` or ``none``."""
        return self.understood.verb or self.understood.feedback or "none"


def hear_command(
    recognizer: asr.Recognizer, pcm: bytes, lang: str, duck_name: str | None = None
) -> Spoken:
    """Hear ``pcm`` (16 kHz mono 16-bit) in ``lang`` and say what it means. Raises :class:`asr.Refused`."""
    heard = recognizer.hear(pcm, lang, duck_name)
    return read_words(heard, lang, duck_name)


def read_words(heard: asr.Transcript, lang: str, duck_name: str | None = None) -> Spoken:
    """What a transcript means (steps 2 and 3 above)."""
    if not heard.text:
        return Spoken(heard, Understood(lang=lang, reason=heard.empty or "no_speech"))
    got = understand(heard.text, lang=lang, duck_name=duck_name)
    if got.verb or got.feedback or got.reason == "negated":
        return Spoken(heard, got)
    near = nearest(heard.text, lang, duck_name)
    if near is None:
        return Spoken(heard, got)
    meant = understand(near.phrase, lang=lang)
    confidence = min(meant.confidence, SOUNDED_CONFIDENCE) * (1.0 - near.distance)
    return Spoken(heard, replace(meant, confidence=round(confidence, 3)), near.phrase)
