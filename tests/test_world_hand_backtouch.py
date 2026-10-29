"""Back-touch hooks in the world's hand (duckmind.world.hand), ported from the neural-duck
prototype: the rear-first approach, trunk-local offsets and relative orientation. Pure maths
and fake data only: no MuJoCo needed."""

from __future__ import annotations

import math
import types

import pytest

from duckmind.world import hand as hand_mod
from duckmind.world import worldkit


def test_back_touch_local_offsets_follow_full_body_rotation_but_legacy_offsets_do_not():
    roll = [math.cos(0.19), math.sin(0.19), 0.0, 0.0]
    pitch = [math.cos(0.13), 0.0, math.sin(0.13), 0.0]
    quat = worldkit.quat_mul(worldkit.yaw_quat(0.7), worldkit.quat_mul(roll, pitch))
    offset = [0.004, -0.031, 0.21]
    expected = worldkit.rotate(quat, offset)

    assert hand_mod._local_offset_world(quat, offset, True) == pytest.approx(expected)
    assert hand_mod._local_offset_world(quat, offset, False) == offset


def test_back_touch_clearance_projects_every_collider_through_full_trunk_rotation():
    origin = [0.7, -0.4, 0.2]
    roll = [math.cos(0.21), math.sin(0.21), 0.0, 0.0]
    pitch = [math.cos(0.17), 0.0, math.sin(0.17), 0.0]
    quat = worldkit.quat_mul(worldkit.yaw_quat(-0.8), worldkit.quat_mul(roll, pitch))
    local_centers = [[0.08, -0.03, 0.24], [-0.11, 0.02, 0.31], [0.02, 0.04, 0.12]]
    radii = [0.03, 0.015, 0.08]
    world_centers = [
        [a + b for a, b in zip(origin, worldkit.rotate(quat, center), strict=True)]
        for center in local_centers
    ]

    projected = hand_mod._projected_geom_top(origin, quat, world_centers, radii)

    assert projected == pytest.approx(
        max(c[2] + r for c, r in zip(local_centers, radii, strict=True))
    )
    with pytest.raises(ValueError, match="collision geometry"):
        hand_mod._projected_geom_top(origin, quat, [], [])


def test_back_touch_hand_envelope_covers_the_finger_tip_and_forearm():
    assert hand_mod.Hand._hand_low_local() == pytest.approx(0.011)
    assert hand_mod.Hand._hand_radius() >= 0.44


def test_back_touch_candidate03_uses_thumb_anchor_and_reverses_low_approach_before_rising():
    hand = object.__new__(hand_mod.Hand)
    hand.robot = types.SimpleNamespace(trunk=0)
    hand.mocap = 0
    hand._duck_top_local = lambda _data: 0.14
    data = types.SimpleNamespace(
        xpos=[[0.0, 0.0, 0.0]],
        xquat=[[1.0, 0.0, 0.0, 0.0]],
        mocap_quat=[[1.0, 0.0, 0.0, 0.0]],
    )

    legs = hand._touch_back(data, [0.0, -0.4, 0.5], 1.0, "touch")

    expected = [-0.100, 0.010, 0.021186137105937256]
    rear = [-0.160, 0.010, expected[2]]
    assert legs[1].offset == pytest.approx(
        [-0.160, 0.010, 0.14 + 0.03 + hand_mod.Hand._hand_radius()]
    )
    assert legs[1].relative_orientation == pytest.approx(worldkit.yaw_quat(-math.pi / 2))
    assert legs[2].offset[0:2] == pytest.approx([-0.160, 0.010])
    assert legs[3].offset == pytest.approx(rear)  # descend behind the duck
    assert legs[4].offset == pytest.approx(expected)  # then approach horizontally
    assert legs[5].offset == pytest.approx(expected)
    assert legs[5].follow_live is True
    assert legs[5].pat == pytest.approx(0.002)
    assert legs[6].offset == pytest.approx(rear)  # withdraw before rising
    assert legs[7].offset[0:2] == pytest.approx([-0.160, 0.010])


def test_back_dwell_tracks_current_trunk_pose_and_rotates_the_pat_normal():
    hand = object.__new__(hand_mod.Hand)
    hand.robot = types.SimpleNamespace(trunk=0)
    hand.mocap = 0
    quat = worldkit.yaw_quat(math.pi / 2)
    data = types.SimpleNamespace(
        xpos=[[0.5, -0.2, 0.3]],
        xquat=[quat],
        mocap_pos=[[0.5, -0.2, 0.3]],
        mocap_quat=[[1.0, 0.0, 0.0, 0.0]],
    )
    leg = hand_mod.Leg(
        "touch",
        "track",
        2.0,
        offset=[0.1, 0.0, 0.0],
        local_offset=True,
        follow_live=True,
        pat=0.002,
        t0=0.0,
        p0=[-2.0, -2.0, 0.0],
        q0=[1.0, 0.0, 0.0, 0.0],
    )

    hand._drive(leg, data, 0.5)

    # At 0.5 s the 1.5 Hz pat is halfway through its 2 mm inward excursion.
    expected = [0.5, -0.1, 0.299]
    assert data.mocap_pos[0] == pytest.approx(expected)
    assert data.mocap_quat[0] == pytest.approx(quat)


def test_back_track_relative_orientation_composes_full_body_quaternion_without_rotating_offset():
    roll = [math.cos(0.19), math.sin(0.19), 0.0, 0.0]
    pitch = [math.cos(-0.14), 0.0, math.sin(-0.14), 0.0]
    trunk_quat = worldkit.quat_mul(worldkit.yaw_quat(0.61), worldkit.quat_mul(roll, pitch))
    relative = worldkit.yaw_quat(-math.pi / 2)
    base = [0.4, -0.3, 0.2]
    offset = [-0.16, 0.01, 0.42]
    hand = object.__new__(hand_mod.Hand)
    hand.mocap = 0
    hand._point = lambda _follow, _data: (base, trunk_quat)
    data = types.SimpleNamespace(mocap_pos=[[0.0, 0.0, 0.0]], mocap_quat=[[1.0, 0.0, 0.0, 0.0]])
    leg = hand_mod.Leg(
        "back-clearance",
        "track",
        0.5,
        offset=offset,
        local_offset=True,
        follow_live=True,
        relative_orientation=relative,
        t0=0.0,
        p0=[0.0, 0.0, 0.0],
        q0=[1.0, 0.0, 0.0, 0.0],
    )

    hand._drive(leg, data, 0.25)

    expected_position = [
        a + b for a, b in zip(base, worldkit.rotate(trunk_quat, offset), strict=True)
    ]
    expected_quat = worldkit.quat_mul(trunk_quat, relative)
    assert data.mocap_pos[0] == pytest.approx(expected_position)
    assert data.mocap_quat[0] == pytest.approx(expected_quat)

    # Default None preserves the prior track orientation and is opt-in for back legs only.
    leg.relative_orientation = None
    hand._drive(leg, data, 0.25)
    assert data.mocap_quat[0] == pytest.approx(trunk_quat)
