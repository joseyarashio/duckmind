"""The simulated world around the duck: its light and what its head camera makes of it.

Phase 0 of the physical-world spec (§11.2, §11.3, §12.3): :mod:`.lights` is the one light
state (the page, the MuJoCo camera and later HomeWorld read it), :mod:`.camera_model` is the
real head camera's geometry, auto-exposure and sensor applied to a rendered frame. This is
the world side: Duck Core never imports it, and nothing here decides anything for the duck.
"""
