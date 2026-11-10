"""Bringing up a real Microduck on the bench (docs/deploy/bring-up.md): record its body over the remote-body link
through the staged poses, fit its body-sense calibration (:mod:`duckmind.duckbody.calibration`), write the report.

``python -m duckmind.bringup`` — :mod:`.record` (the guided recording), :mod:`.fit` (the leg-load bands and their
replay check), :mod:`.tofcal` (the ToF touch band), :mod:`.simperson` (a rehearsal's hands, for ``sim-body``).
"""
