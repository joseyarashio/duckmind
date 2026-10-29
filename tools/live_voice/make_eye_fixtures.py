"""Render the duck-eye pictures the live-voice scenarios send up (``tests/fixtures/live/eye``).

Two pictures from the simulator's own head camera (the world's MuJoCo scene through the camera
model: lens, auto-exposure, white balance, sensor noise; ``tests/eye_vision_scene.py``), the duck
standing in the default home facing the same way: ``eye_toy.jpg`` with a red ball on the floor
half a metre in front of it, ``eye_empty.jpg`` the same view with nothing there. They are what the
scenarios ``what_do_you_see`` and ``what_do_you_see_empty`` send, and what their cassettes were
recorded against (the cassette key holds each picture's hash).

This is the render tier: it needs MuJoCo 3.10, numpy, pytest (for the scene helper) and the
official microduck checkout, and one physics process::

    DUCKMIND_MICRODUCK_MJCF=/path/to/microduck_rl/src/mjlab_microduck/robot/microduck \\
    PYTHONPATH=src uv run --no-project --python 3.12 --with numpy --with mujoco==3.10.0 --with pytest \\
        python tools/live_voice/make_eye_fixtures.py [--out tests/fixtures/live/eye] [--step 2]

The pictures are the simulator's computed sensor reading (spec §4.8), not camera recordings; no
network is touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

SPOT = (-1.5, -0.5, 0.0)  # where the duck stands in the default home, and its heading
BALL_AHEAD_M = 0.55
BALL_RADIUS_M = 0.035
BALL_RGBA = (0.85, 0.12, 0.10, 1.0)


def add_ball(spec: Any, name: str = "eyefix_ball") -> None:
    """A mocap ball (no collision: only the camera sees it), parked below the floor."""
    import mujoco

    body = spec.worldbody.add_body()
    body.name = name
    body.mocap = True
    body.pos = [0, 0, -2]
    geom = body.add_geom()
    geom.name = name + "_geom"
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size = [BALL_RADIUS_M, 0, 0]
    geom.pos = [0, 0, BALL_RADIUS_M]
    geom.rgba = list(BALL_RGBA)
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = 1


def render(with_ball: bool, step: int, quality: int) -> bytes:
    import eye_vision_scene as ev

    from duckmind.eye import frames
    from duckmind.simweb import jpeg

    scene = ev.Scene("default_home", edit=add_ball if with_ball else None)
    try:
        x, y, heading = SPOT
        scene.pose(x, y, heading)
        if with_ball:
            body = scene.mujoco.mj_name2id(scene.m, scene.mujoco.mjtObj.mjOBJ_BODY, "eyefix_ball")
            mocap = int(scene.m.body_mocapid[body])
            wx, wy = scene.to_world(BALL_AHEAD_M, 0.0)
            scene.d.mocap_pos[mocap] = [wx, wy, 0.0]
            scene.mujoco.mj_forward(scene.m, scene.d)
        cap = scene.capture("day")
        rgb = frames.upright_rgb(cap.uyvy(), cap.width, cap.height, cap.rotate, step=step)
        data, _how = jpeg.encode(rgb, quality)
        print(
            f"  {'toy' if with_ball else 'empty'}: {rgb.shape[1]}x{rgb.shape[0]}, {len(data)} bytes"
        )
        return bytes(data)
    finally:
        scene.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=str(ROOT / "tests" / "fixtures" / "live" / "eye"))
    parser.add_argument("--step", type=int, default=2, help="every n-th pixel (2: 640x360)")
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, with_ball in (("eye_empty", False), ("eye_toy", True)):
        (out / f"{name}.jpg").write_bytes(render(with_ball, args.step, args.quality))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
