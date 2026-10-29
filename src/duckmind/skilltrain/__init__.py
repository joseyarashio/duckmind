"""Stage 1 of the reopened motion training (owner 「Do it」, 2026-09-28): the Mac side.

The A100 trains new one-shot robotd skills (training plan §3.1–3.2: the startle family —
flinch, crouch-and-freeze, hop-back as quick back-steps — and the comfort family — shake-off,
stretch standing and seated). Nothing here trains; this package is what a trained skill must
pass on the Mac before it may reach the owner's simulator, and how it gets there:

* :mod:`.table` — robotd's ``[[policy.skill]]`` table (``robotd-params`` ``SkillDef``): the
  Stage-1 entries, their validation against robotd's own rules, and the ``robotd.toml``
  section our live launcher writes into ``/private/tmp/dm-live-*/`` (no upstream edits).
* :mod:`.robotd_port` — ``robotd/src/control.rs`` (the controller: windows, unwind, chain,
  pre-emption, the sit machine, scale/gain, low-pass) and the ``main.rs`` tick around it
  (requests once per tick, command smoothing, deadman, voltage adaptation), in Python.
* :mod:`.servo` / :mod:`.dr` — the two servo models every gate runs on (``mjcf``: the MJCF's XML
  servos, what the owner's live simulator runs; ``bam``: the BAM XL330 model training uses) and
  the per-duck domain randomisation.
* :mod:`.skillduck` — :class:`~duckmind.world.headless_duck.HeadlessDuck` on the official flat
  scene, driven by the port, recording the truth every gate reads.
* :mod:`.metrics` — the plan's visibility and safety gates (§6), computed from recorded traces
  only (the A100 tier can feed the same functions).
* :mod:`.programs` / :mod:`.gate` — trial programs (standing, walking, seated; the skill, or a
  v5 baseline) and the resumable gate runner with Wilson 95 % summaries.
* :mod:`.guard` — the duck-body side: which postures may start which skill (and how fast a
  walk, how settled), the hourly wear caps, the held crouches, and the brain's act → table
  entry picker (pure standard library, board-safe).
* :mod:`.gt_wiring` / :mod:`.live_judge` — what an installed table tells the simulator (servo
  delay, guard rules, pinned networks) and how the page's witness judges a trained skill by
  its own detectors (pure standard library).
* :mod:`.gt_gate` / :mod:`.gt_latency` / :mod:`.gt_verdict` / :mod:`.gt_report` — the gate
  runs with a servo command delay, the latency map, and the verdicts and honest reports.

Heavy imports (numpy, MuJoCo, onnxruntime) happen inside the modules that need them.
"""

from __future__ import annotations

__all__ = [
    "dr",
    "gate",
    "guard",
    "gt_gate",
    "gt_latency",
    "gt_live",
    "gt_report",
    "gt_verdict",
    "gt_wiring",
    "live_judge",
    "metrics",
    "programs",
    "robotd_port",
    "servo",
    "skillduck",
    "table",
]
