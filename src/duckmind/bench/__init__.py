"""The measuring tools of the repository: fixed questions, fixed seeds, one row schema, paired statistics.

``python -m duckmind.bench gothere …`` measures the walk to a place somebody pointed at (docs/validation/bench.md):
a versioned click set (:mod:`.clicks`), the headless runner on the real brain and the real physics (:mod:`.runner`),
the paired comparison of two source trees (:mod:`.compare`, :mod:`.stats`) and a live mode for the last confirmation
(:mod:`.live`). The truth planner (:mod:`.truthplan`) exists only here: it measures what a perfect route would reach on
this body, and is never part of the duck.
"""
