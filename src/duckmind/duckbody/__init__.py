"""duck-body: the process between Duck Core and the robot's daemons (spec §5, §6; phase 3).

It turns what the sensors publish (``robot.state``, ``tof.frame``, the eye's results) into
percepts and runs the body's own reflexes. The same code runs on the board and beside the
simulator: it reads sockets, never the simulator's truth, and imports neither
:mod:`duckmind.world`, :mod:`duckmind.simweb` nor MuJoCo.

Named ``duckbody`` because :mod:`duckmind.body` (the backend-neutral body port Duck Core
already imports) owns the shorter name (spec appendix B said ``src/duckmind/body/``).

Phase 3 builds the process: ``python -m duckmind.duckbody`` (:mod:`.server`, the only place with
sockets, threads and clocks) runs :class:`~duckmind.duckbody.bodyloop.BodyLoop` — the posture tap,
body sense, righting, reflexes and the forbidden-command guard, the act executors (walkexec,
dockexec, lookplan's looks, the curiosity Act Runner) — and serves ``body.sock``
(:mod:`.protocol`). Phase 2's :mod:`.tof_reproject` is the Python port of the official
``kinematics::tof::Reprojector`` (P4).
"""
