"""Expression sweep: what the real duck does with Duck Core's head clips.

On the robot, head targets ride in the official policies' command block and the
network decides the actual joint motion while keeping balance. This package turns every
clip into the command stream robotd would build (:mod:`.clips`), runs them through the
official networks in the official physics at scale (``notebooks/expression_sweep_colab.ipynb``),
and reads the results back as per-clip safe amplitudes (:mod:`.envelope`).

The same runs measure what the camera-first design needs before any behaviour changes:
the looks (:mod:`.looks`), the sit ↔ stand cycle per phase by robotd's own fall rule with
self-righting (:mod:`.falls`), and walking with the head posed. Every result carries n,
the count and a Wilson interval (:mod:`.stats`), and the raw per-duck rows sit next to it
so :mod:`.summary` can recompute every number.
"""
