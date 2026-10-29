"""The dock-tag contract between duck-eye and duck-body, end to end on a rendered dock (phase-3
review, critical 3: duck-body read a nest result shape duck-eye never sends, and the two sides
gave the tag's yaw in mirrored conventions — every tag fix was lost, and a fixed shape alone would
have put the duck behind the dock).

Render tier: the default home's dock through the world's head camera (``eye_vision_scene``);
duck-eye's own ``NestTask`` reads the frame; the result goes through ``lookplan.nest_fix`` (as
every ``eye.result`` does in duck-body) into ``dockexec.pose_from_fix``; the pose must match the
duck's true pose relative to the tag (evaluator truth) within 6 cm and 3°.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from duckmind.duckbody import dockexec, lookplan  # noqa: E402


@pytest.mark.parametrize(
    "dist_m,off_deg,turn_deg",
    [(1.0, 0.0, 0.0), (1.5, 20.0, 8.0), (1.2, -25.0, -10.0)],
)
def test_a_rendered_tag_fix_puts_the_duck_where_it_truly_is(
    dist_m: float, off_deg: float, turn_deg: float
) -> None:
    pytest.importorskip("pupil_apriltags")
    import eye_vision_scene as ev

    from duckmind.eye import nesttag

    scene = ev.Scene("default_home")
    try:
        gid = scene.geom("home_nest_tag")
        tag = np.array(scene.d.geom_xpos[gid])
        normal = np.array(scene.d.geom_xmat[gid]).reshape(3, 3)[:, 2]
        axis = math.atan2(normal[1], normal[0])  # the dock's axis, out of the wall into the room
        a = axis + math.radians(off_deg)
        cx, cy = tag[0] + dist_m * math.cos(a), tag[1] + dist_m * math.sin(a)
        facing = math.atan2(tag[1] - cy, tag[0] - cx) + math.radians(turn_deg)
        scene.pose_camera_at(cx, cy, facing)
        planes, cam, luma_of = ev.eye_inputs(scene.capture("day"))
        nest = nesttag.NestTask().read(luma_of, planes, cam, scene.camera_pose(), level="bright")
        fix = lookplan.nest_fix({"look_id": "D1.fix", "status": "ok", "results": {"nest": nest}})
        assert fix is not None and fix["src"] == "tag" and fix["tag"] == 17, nest
        geo = dockexec.GEOMETRY["forward"]
        pose = dockexec.pose_from_fix(fix, geo)
        assert pose is not None
        # the truth, in the dock's frame (x along the axis, y its left, heading π = facing it)
        tx, ty, yaw = scene.trunk_xy_yaw()
        dx, dy = tx - tag[0], ty - tag[1]
        along = dx * math.cos(axis) + dy * math.sin(axis)
        left = -dx * math.sin(axis) + dy * math.cos(axis)
        heading = dockexec.wrap(yaw - axis)
        assert abs((pose.x - geo.face_mm) - along * 1000) <= 60, (pose, along)
        assert abs(pose.y - left * 1000) <= 60, (pose, left)
        assert abs(dockexec.wrap(pose.h - heading)) <= math.radians(3.0), (pose, heading)
        # the sure facing is what the pose used (the mirror pose would put it behind the dock)
        assert fix["yaw_sure"] is True and pose.x > geo.face_mm
    finally:
        scene.close()
