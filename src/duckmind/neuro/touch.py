"""Shared touch-shape classification for perception and fast reflexes."""

STROKE_THRESHOLD = 0.2


def is_stroking(stroke: float) -> bool:
    """Return whether measured sliding contact is a stroke, not a point poke.

    Keep the strict boundary aligned with Perceiver's gentle-stroke contract: exactly 0.2 is
    still a poke; a sliding contact above it is a stroke.
    """
    return stroke > STROKE_THRESHOLD
