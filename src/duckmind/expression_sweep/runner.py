"""The sweep itself, in the official mjlab physics (Colab A100; tiny CPU runs on a Mac).

Scenarios, each a batch of up to thousands of parallel ducks with domain randomization:

* ``head`` — every Duck Core clip × amplitude × tempo × side, seated (SitStand task
  physics, ``alpha_sitstand`` with the sit flag) and standing (VelStand task physics,
  ``velstand`` at zero twist);
* ``look`` — the looks the camera-first design needs (:mod:`.looks`) the same way, with a
  HOME-hold control, reading where the head camera points at each frame a look takes;
* ``posture`` — sit → rise (flag 0 for robotd's 1 s) → stand → sit in both task physics,
  falls counted per phase by robotd's own rule, and after a fall velstand left at zero
  twist to right itself (:mod:`.falls`);
* ``walk`` — slow twists from ``content.LOCOMOTION``, then stop;
* ``headwalk`` — walking at vx 0.30 with the head level or half down (the glance_down pose).

The head command stream is robotd's (offsets from HOME, slewed at α 0.2 per 50 Hz tick)
and, by default, so is the action path: ``HOME + scale·a`` times the voltage factor,
then robotd's target low-pass, with the raw network output fed back as the last action.
Pushes, prone spawns and curricula are removed; randomization is set explicitly.

Every scenario but ``walk`` also writes one raw row per simulated duck to
``<out>.trials.jsonl``; :func:`.summary.recompute` rebuilds the result's summaries from it.

The official v5 ONNX files take a fixed batch of one, so the MLP is rebuilt in torch
from their initializers (checked against onnxruntime when it is installed).
Needs the microduck_rl environment (``uv sync --locked``) with duckmind on the path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from duckmind.duckcore import content
from duckmind.expression_sweep import envelope, falls, looks, robotd_model, summary
from duckmind.expression_sweep.clips import (
    HEAD_JOINTS,
    SIDES,
    Variant,
    capture_ticks,
    clip_digest,
    duration_s,
    trajectory,
    variants,
)
from duckmind.expression_sweep.stats import percentile

SIT_TASK = "Mjlab-SitStand-Flat-MicroDuck"
STAND_TASK = "Mjlab-VelStand-Flat-MicroDuck"
HOME = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": -0.0049,
    "left_ankle": 0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0049,
    "right_ankle": -0.4530,
}
ACTION_JOINTS = tuple(HOME)  # the networks' 14 outputs, in this order
STALL_NM = 0.64  # XL330 stall with the firmware current limit (BAM model)
FALL_TILT_DEG = falls.LEGACY_FALL_TILT_DEG  # the v2 count; robotd's own rule lives in .falls
# A duck already tipped this far when a clip begins did not settle into the posture;
# such episodes say something about sitting down, not about the clip, and are counted
# apart (the first A100 sweep: ~3% of seated starts leaned back ~62° before any motion).
SETTLE_BAD_TILT_DEG = falls.SETTLE_BAD_TILT_DEG
SETTLE_S = 2.0
TAIL_S = 1.5  # HOME after a clip: does the head come back and the body stay up
RECORD_HZ = 25
SCENARIOS = ("head", "look", "posture", "walk", "headwalk")
# Walking with the head posed (spec §10.2): the slowest walk the network reliably takes,
# straight, for the longest leg Duck Core plans; the head level or at the glance_down pose.
_LOCOMOTION: dict[str, Any] = dict(content.LOCOMOTION)
HEAD_WALK: tuple[float, float, float] = (
    float(_LOCOMOTION["walk_vx"][0]),
    0.0,
    float(_LOCOMOTION["walk_s_max"]),
)
HEAD_WALK_POSES: dict[str, dict[str, float]] = {"level": {}, "half_down": dict(looks.HALF_DOWN)}

aggregate = summary.aggregate  # per-variant statistics (kept here for older callers)
_percentile = percentile


@dataclass(frozen=True)
class DomainRandomization:
    """Final-stage training ranges; ``nominal`` keeps only the always-on noise."""

    trunk_com_m: float = 0.015
    head_com_m: float = 0.010

    @classmethod
    def level(cls, name: str) -> DomainRandomization:
        return {"final": cls(), "nominal": cls(0.0, 0.0)}[name]


def release(env: Any) -> None:
    """Close an environment and hand its GPU memory back before building the next one."""
    import gc

    env.close()
    del env
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_policy(path: Path, device: str) -> Any:
    """The v5 MLP as a batched torch function: (obs − mean)/std → 3×ELU → 14."""
    import onnx
    import torch
    from onnx import numpy_helper

    weights = {
        item.name: torch.from_numpy(numpy_helper.to_array(item).copy()).to(device)
        for item in onnx.load(str(path)).graph.initializer
    }
    std_key = next(key for key in weights if key.startswith("onnx::Div"))
    mean, std = weights["obs_normalizer._mean"], weights[std_key]

    @torch.no_grad()
    def policy(obs: Any) -> Any:
        x = (obs - mean) / std
        for layer in ("mlp.0", "mlp.2", "mlp.4"):
            x = torch.nn.functional.elu(x @ weights[layer + ".weight"].T + weights[layer + ".bias"])
        return x @ weights["mlp.6.weight"].T + weights["mlp.6.bias"]

    return policy


def check_parity(path: Path, policy: Any, device: str, samples: int = 64) -> float:
    """Max |torch − onnxruntime| over random observations; 0.0 when ORT is absent."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return 0.0
    import torch

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(samples):
        obs = rng.normal(0, 1, (1, 61)).astype(np.float32)
        expected = session.run(None, {name: obs})[0]
        got = policy(torch.from_numpy(obs).to(device)).cpu().numpy()
        worst = max(worst, float(np.abs(got - expected).max()))
    return worst


def build_env(
    task: str, envs: int, device: str, *, seated: bool, dr: DomainRandomization, episode_s: float
) -> Any:
    import mjlab.tasks  # noqa: F401 — registers the tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = envs
    cfg.seed = 0
    cfg.episode_length_s = episode_s
    cfg.curriculum = {}
    for name in ("push_robot", "topple_push", "random_prone_init"):
        cfg.events.pop(name, None)
    for name, command in cfg.commands.items():
        command.resampling_time_range = (1e9, 1e9)
        if name in ("head_pose", "body_pose"):
            command.ranges = ((0.0, 0.0),) * len(command.ranges)
    twist = cfg.commands["twist"]
    if hasattr(twist, "sit_prob"):
        twist.sit_prob = 1.0 if seated else 0.0
    else:
        twist.rel_standing_envs = 0.0
        twist.rel_heading_envs = twist.rel_forward_envs = twist.rel_turn_in_place_envs = 0.0
        twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0.0, 0.0)
    if "set_ground_state" not in cfg.events:
        cfg.events["set_ground_state"] = deepcopy(
            load_env_cfg(SIT_TASK, play=True).events["set_ground_state"]
        )
    cfg.events["set_ground_state"].params.update(
        sitting_prob=1.0 if seated else 0.0, standing_prob=0.0 if seated else 1.0
    )
    if "randomize_com" in cfg.events:
        cfg.events["randomize_com"].params["ranges"] = (-dr.trunk_com_m, dr.trunk_com_m)
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["ranges"] = (-dr.head_com_m, dr.head_com_m)
    # Falls are measured here, never by resetting the duck mid-clip.
    for name in list(cfg.terminations):
        if name != "nan_state":
            cfg.terminations.pop(name)
    sensors = [
        ContactSensorCfg(
            name="jaw_trunk",
            primary=ContactMatch(mode="body", pattern="^jaw_soft$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="^trunk_base$", entity="robot"),
            fields=("found",),
            reduce="netforce",
            num_slots=1,
        )
    ]
    cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(sensors)
    return ManagerBasedRlEnv(cfg, device=device)


class Episode:
    """Steps one batch with robotd's command and action paths and measures everything."""

    def __init__(self, env: Any, device: str, *, robotd_actions: bool) -> None:
        import torch

        self.torch = torch
        self.env = env
        self.robot = env.scene["robot"]
        self.device = device
        self.robotd_actions = robotd_actions
        ids, _ = self.robot.find_joints(list(ACTION_JOINTS), preserve_order=True)
        self.joint_ids = ids
        self.head_ids = [ids[ACTION_JOINTS.index(name)] for name in HEAD_JOINTS]
        self.home = torch.tensor([HOME[name] for name in ACTION_JOINTS], device=device)
        self.alpha = torch.tensor(
            [
                robotd_model.ACTION_LOWPASS["head" if name in HEAD_JOINTS else "legs"]
                for name in ACTION_JOINTS
            ],
            device=device,
        )
        self.head = env.command_manager.get_term("head_pose").command
        self.twist = env.command_manager.get_term("twist").command
        self.previous_target: Any = None
        self.torques: list[Any] = []
        # A mid-episode reset (nan_state) makes an env's numbers not a duck's: flag it.
        self.resets = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        sites, _ = self.robot.find_sites(["head_camera"], preserve_order=True)
        self.camera_site = sites[0] if sites else None
        # Wrap the scene's own update once per env, not once per batch: nested wrappers
        # would keep every earlier batch's torque list growing.
        original = getattr(env, "_sweep_scene_update", None)
        if original is None:
            original = env.scene.update
            env._sweep_scene_update = original

        def capture(dt: float) -> None:  # per physics substep
            original(dt)
            force = self.robot.data.actuator_force[:, self.head_ids].abs()
            self.torques.append(force.max(dim=1).values)

        env.scene.update = capture

    def shape(self, raw: Any, scale: Any) -> Any:
        """robotd's action path: HOME + scale·a·(7.4 V / battery), then its target low-pass."""
        if not self.robotd_actions:
            return raw
        volts = self.robot.actuators[0].vin_tensor.reshape(-1, 1)
        target = self.home + scale * raw * (7.4 / volts.clamp(6.0, 9.5))
        if self.previous_target is None:
            filtered = target
        else:
            filtered = self.alpha * target + (1.0 - self.alpha) * self.previous_target
        self.previous_target = filtered
        return filtered - self.home

    def act(self, policy: Any, obs: Any, scale: Any) -> Any:
        raw = policy(obs["actor"])
        return self.shape(raw, scale), raw

    def step(self, action: Any, raw: Any) -> Any:
        obs, _reward, terminated, time_outs, _extras = self.env.step(action)
        self.resets |= terminated | time_outs
        if self.robotd_actions:
            obs["actor"][:, 34:48] = raw  # robotd feeds back the network's own output
        return obs

    def tilt_deg(self) -> Any:
        gravity = self.robot.data.projected_gravity_b
        return self.torch.rad2deg(self.torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0)))

    def head_offsets(self) -> Any:
        return (
            self.robot.data.joint_pos[:, self.head_ids]
            - self.home[[ACTION_JOINTS.index(n) for n in HEAD_JOINTS]]
        )

    def camera(self) -> tuple[Any, Any]:
        """Where the head camera points: optical-axis pitch (degrees, + up) and height (m).

        The MJCF's ``head_camera`` site has its x axis on the true optical axis (the camera
        element itself faces backwards; sim/camera.py turns it round the same way).
        """
        torch = self.torch
        if self.camera_site is None:
            nan = torch.full((self.env.num_envs,), math.nan, device=self.device)
            return nan, nan
        pose = self.robot.data.site_pose_w[:, self.camera_site]
        w, x, y, z = pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6]
        axis_z = (2.0 * (x * z - w * y)).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.asin(axis_z)), pose[:, 2]

    def still(self) -> Any:
        """Spec §4.2 rule 2 for one sample: head joints and the body gyro both quiet."""
        data = self.robot.data
        head = data.joint_vel[:, self.head_ids].abs().max(dim=1).values
        gyro = data.root_link_ang_vel_b.norm(dim=1)
        return (head < looks.STILL_HEAD_RAD_S) & (gyro < looks.STILL_GYRO_RAD_S)


def _head_streams(
    assigned: list[Variant], ticks: int, device: str
) -> tuple[Any, Any, Any, set[int]]:
    """Slewed offsets [envs, ticks, 4], the clip window [envs, ticks], and for looks the
    frame index taken at each tick [envs, ticks] (−1: none) with the ticks that hold any."""
    import torch

    commands = torch.zeros(len(assigned), ticks, 4)
    active = torch.zeros(len(assigned), ticks, dtype=torch.bool)
    frame_index = torch.full((len(assigned), ticks), -1, dtype=torch.int8)
    frame_ticks: set[int] = set()
    cache: dict[Variant, tuple[list[list[float]], int, int]] = {}
    lead_ticks = round(SETTLE_S * robotd_model.CONTROL_HZ)
    for index, variant in enumerate(assigned):
        if variant not in cache:
            stream = trajectory(variant, lead_s=0.0, tail_s=TAIL_S)
            columns = [robotd_model.slew(stream[name]) for name in HEAD_JOINTS]
            clip_ticks = round(duration_s(variant) * robotd_model.CONTROL_HZ)
            cache[variant] = (columns, len(columns[0]), clip_ticks)
        columns, length, clip_ticks = cache[variant]
        span = min(length, ticks - lead_ticks)
        for joint in range(4):
            commands[index, lead_ticks : lead_ticks + span, joint] = torch.tensor(
                columns[joint][:span]
            )
        active[index, lead_ticks : lead_ticks + min(clip_ticks, span)] = True
        for k, offset in enumerate(capture_ticks(variant)):
            if lead_ticks + offset < ticks:
                frame_index[index, lead_ticks + offset] = k
                frame_ticks.add(lead_ticks + offset)
    return commands.to(device), active.to(device), frame_index.to(device), frame_ticks


def run_head(
    policy: Any,
    env: Any,
    device: str,
    assigned: list[Variant],
    posture: str,
    *,
    robotd_actions: bool,
    record: set[int],
) -> tuple[dict[str, Any], dict[int, Any]]:
    """One batch: env i plays assigned[i]. Returns per-env metrics and recorded tracks."""
    torch = __import__("torch")
    hz = robotd_model.CONTROL_HZ
    longest = max(duration_s(variant) for variant in assigned)
    ticks = round((SETTLE_S + longest + TAIL_S) * hz)
    commands, active, frame_index, frame_ticks = _head_streams(assigned, ticks, device)
    episode = Episode(env, device, robotd_actions=robotd_actions)
    obs, _ = env.reset(seed=0)
    n = len(assigned)
    scale = robotd_model.ACTION_SCALE[posture]
    settle_ticks = round(SETTLE_S * hz)
    debounce = falls.debounce_ticks(hz)
    still_needed = looks.still_ticks(hz)
    # Safety is judged from the settled moment on: falls, tilt, load and contact while
    # the duck drops into its posture after reset belong to the posture, not the clip.
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    fell_robotd = torch.zeros(n, dtype=torch.bool, device=device)
    down_run = torch.zeros(n, dtype=torch.long, device=device)
    still_run = torch.zeros(n, dtype=torch.long, device=device)
    tilts = torch.zeros(n, device=device)
    settled_tilt = torch.zeros(n, device=device)
    tilt_rise = torch.zeros(n, device=device)
    measured = 0
    torque = torch.zeros(n, device=device)
    strain = torch.zeros(n, device=device)
    contacts = torch.zeros(n, device=device)
    dot = torch.zeros(n, 4, device=device)
    norm = torch.zeros(n, 4, device=device)
    err2 = torch.zeros(n, device=device)
    count = torch.zeros(n, device=device)
    start_xy = None
    rest_pitch = torch.full((n,), math.nan, device=device)
    rest_height = torch.full((n,), math.nan, device=device)
    frames = looks.MAX_CAPTURES
    cam_pitch = torch.full((n, frames), math.nan, device=device)
    cam_height = torch.full((n, frames), math.nan, device=device)
    cam_still = torch.zeros(n, frames, dtype=torch.bool, device=device)
    # The head droops a little at a zero command; measure motion relative to that rest.
    baseline = torch.zeros(n, 4, device=device)
    recorded = sorted(record)
    record_ids = torch.tensor(recorded, dtype=torch.long, device=device)
    samples: list[Any] = []
    every = max(1, round(hz / RECORD_HZ))
    acted = torch.zeros(n, 4, device=device)
    substep_hook_missed = 0
    for tick in range(ticks):
        if episode.twist.shape[1] == 3 and posture == "sit":
            episode.twist[:] = 0.0
            episode.twist[:, 0] = 1.0
        elif posture == "stand":
            episode.twist[:] = 0.0
        next_command = commands[:, min(tick + 1, ticks - 1)]
        action, raw = episode.act(policy, obs, scale)
        episode.head[:] = next_command
        episode.torques.clear()
        obs = episode.step(action, raw)
        tilt = episode.tilt_deg()
        down_run = falls.next_run(down_run, tilt > falls.FALL_TILT_DEG)
        still_run = falls.next_run(still_run, episode.still())
        if tick + 1 == settle_ticks:
            start_xy = episode.robot.data.root_link_pos_w[:, :2].clone()
            baseline = episode.head_offsets().clone()
            settled_tilt = tilt.clone()
            rest_pitch, rest_height = episode.camera()
        if tick + 1 >= settle_ticks:
            measured += 1
            fell |= tilt > FALL_TILT_DEG
            fell_robotd |= down_run >= debounce
            tilts = torch.maximum(tilts, tilt)
            tilt_rise = torch.maximum(tilt_rise, tilt - settled_tilt)
            if episode.torques:
                peak = torch.stack(episode.torques).max(dim=0).values
            else:
                # A captured CUDA graph can skip the Python substep hook: never report
                # zero load for that — fall back to the force after the last substep.
                force = episode.robot.data.actuator_force[:, episode.head_ids].abs()
                peak = force.max(dim=1).values
                substep_hook_missed += 1
            torque = torch.maximum(torque, peak)
            strain += (peak > robotd_model.HEAD_STRAIN_NM).float()
            contacts += (
                (env.scene.sensors["jaw_trunk"].data.found.reshape(n, -1) > 0).any(dim=1).float()
            )
        if tick in frame_ticks:
            pitch, height = episode.camera()
            quiet = still_run >= still_needed
            for k in range(frames):
                here = frame_index[:, tick] == k
                cam_pitch[:, k] = torch.where(here, pitch, cam_pitch[:, k])
                cam_height[:, k] = torch.where(here, height, cam_height[:, k])
                cam_still[:, k] = torch.where(here, quiet, cam_still[:, k])
        window = active[:, tick]
        actual = episode.head_offsets() - baseline
        dot += torch.where(window[:, None], actual * acted, 0.0)
        norm += torch.where(window[:, None], acted * acted, 0.0)
        err2 += torch.where(window, ((actual - acted) ** 2).mean(dim=1), 0.0)
        count += window.float()
        acted = next_command
        if tick % every == 0 and recorded:
            data = episode.robot.data
            samples.append(
                torch.cat(
                    [
                        data.joint_pos[record_ids][:, episode.joint_ids],
                        data.root_link_quat_w[record_ids],
                        data.root_link_pos_w[record_ids, 2:3],
                    ],
                    dim=1,
                )
            )
    significant = norm > 1e-3
    ratio = torch.where(significant, dot / norm.clamp_min(1e-9), torch.zeros_like(dot))
    track_ratio = ratio.sum(dim=1) / significant.sum(dim=1).clamp_min(1)
    drift = (
        (episode.robot.data.root_link_pos_w[:, :2] - start_xy).norm(dim=1)
        if start_xy is not None
        else torch.zeros(n, device=device)
    )
    has_frames = [bool(capture_ticks(variant)) for variant in assigned]
    pitch_rows = cam_pitch.cpu().tolist()
    height_rows = cam_height.cpu().tolist()
    still_rows = cam_still.cpu().tolist()

    def per_frame(values: list[float], flags: list[bool] | None = None) -> list[Any]:
        return [
            None if math.isnan(value) else (flags[k] if flags is not None else value)
            for k, value in enumerate(values)
        ]

    metrics = {
        "settle_bad": (settled_tilt > SETTLE_BAD_TILT_DEG).cpu().tolist(),
        "settled_tilt": settled_tilt.cpu().tolist(),
        "fell": fell.cpu().tolist(),
        "fell_robotd": fell_robotd.cpu().tolist(),
        "tilt": tilts.cpu().tolist(),
        "tilt_rise": tilt_rise.cpu().tolist(),
        "torque_ratio": (torque / STALL_NM).cpu().tolist(),
        "strain_frac": (strain / max(1, measured)).cpu().tolist(),
        "contacts": contacts.cpu().tolist(),
        "track_ratio": track_ratio.cpu().tolist(),
        "rms_error": torch.sqrt(err2 / count.clamp_min(1)).cpu().tolist(),
        "drift": drift.cpu().tolist(),
        "substep_torque": [substep_hook_missed == 0] * n,
        "reset": episode.resets.cpu().tolist(),
        "rest_pitch": rest_pitch.cpu().tolist(),
        "rest_height": rest_height.cpu().tolist(),
        "cam_pitch": [
            per_frame(row) if has_frames[i] else None for i, row in enumerate(pitch_rows)
        ],
        "cam_height": [
            per_frame(row) if has_frames[i] else None for i, row in enumerate(height_rows)
        ],
        "cam_still": [
            per_frame(pitch_rows[i], still_rows[i]) if has_frames[i] else None for i in range(n)
        ],
    }
    tracks: dict[int, list[list[float]]] = {}
    if samples:
        stacked = torch.stack(samples, dim=1).cpu()  # [recorded, frames, 19]
        for row, index in enumerate(recorded):
            tracks[index] = [[round(float(v), 4) for v in frame] for frame in stacked[row].tolist()]
    return metrics, tracks


def _clip_seconds(variant: Variant) -> float:
    return duration_s(variant)


def _rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, digits)
    if isinstance(value, (list, tuple)):
        return [_rounded(item, digits) for item in value]
    return value


def look_grid() -> list[Variant]:
    """Every look on the full amplitude × tempo × side grid, plus the HOME-hold control."""
    grid = variants(looks.LOOKS)
    grid += [Variant(looks.CONTROL, 1.0, 1.0, side) for side in SIDES]
    return grid


def sweep_head(
    policies: Path,
    device: str,
    *,
    randomizations: int,
    envs: int,
    dr: DomainRandomization,
    robotd_actions: bool,
    clips: tuple[str, ...] | None,
    log: Any,
    checkpoint: Any = None,
    grid: list[Variant] | None = None,
    scenario: str = "head",
    record_if: Callable[[Variant], bool] | None = None,
    rows_out: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every variant × ``randomizations`` in both postures → {variant: {posture: stats}}.

    Raw per-episode rows are appended to ``rows_out`` when given; each posture is
    aggregated once over all its episodes, so a variant split across batches is exact.
    """
    results: dict[str, dict[str, Any]] = {}
    recorded: dict[str, Any] = {}
    grid = grid if grid is not None else variants(clips)
    record_if = record_if or (lambda v: v.amplitude == 0.6 and v.tempo == 1.0)
    for posture, task, network in (
        ("sit", SIT_TASK, "alpha_sitstand.onnx"),
        ("stand", STAND_TASK, "velstand.onnx"),
    ):
        policy = load_policy(policies / network, device)
        parity = check_parity(policies / network, policy, device)
        log(f"{scenario} {posture}: {network} torch/ORT parity {parity:.2e}")
        jobs = [variant for variant in grid for _ in range(randomizations)]
        size = min(envs, len(jobs))
        longest = max(duration_s(v) for v in grid)
        # One environment per posture: rebuilding for a smaller last batch has run out of
        # GPU memory on an A100, so the last batch is padded and its padding discarded.
        env = build_env(
            task, size, device, seated=posture == "sit", dr=dr, episode_s=SETTLE_S + longest + 10
        )
        posture_rows: list[dict[str, Any]] = []
        for start in range(0, len(jobs), size):
            batch = jobs[start : start + size]
            real = len(batch)
            batch = batch + [jobs[0]] * (size - real)
            # Record the first randomization of the chosen variants.
            record = {
                i
                for i, v in enumerate(batch[:real])
                if record_if(v) and (start + i) % randomizations == 0
            }
            began = time.monotonic()
            metrics, tracks = run_head(
                policy, env, device, batch, posture, robotd_actions=robotd_actions, record=record
            )
            log(
                f"{scenario} {posture}: envs {start}-{start + real} of {len(jobs)} "
                f"in {time.monotonic() - began:.1f}s"
            )
            for i in range(real):
                variant = batch[i]
                row = {
                    "scenario": scenario,
                    "posture": posture,
                    "variant": variant.key,
                    "clip": variant.clip,
                    "amplitude": variant.amplitude,
                    "tempo": variant.tempo,
                    "side": variant.side,
                    "randomization": (start + i) % randomizations,
                    "env": i,
                }
                for name, values in metrics.items():
                    row[name] = _rounded(values[i])
                posture_rows.append(row)
            for index, track in tracks.items():
                recorded.setdefault(batch[index].key, {})[posture] = track
        for key, stats in summary.aggregate_rows(posture_rows).items():
            results.setdefault(key, {})[posture] = stats
        if rows_out is not None:
            rows_out.extend(posture_rows)
        release(env)
        if checkpoint is not None:
            checkpoint(results, recorded)
    return results, recorded


def _phase_policy(sit_policy: Any, stand_policy: Any, use_sit: Any, obs: Any) -> Any:
    torch = __import__("torch")
    return torch.where(use_sit[:, None], sit_policy(obs["actor"]), stand_policy(obs["actor"]))


def run_posture(
    sit_policy: Any,
    stand_policy: Any,
    env: Any,
    device: str,
    *,
    schedule: falls.PostureSchedule,
    robotd_actions: bool,
    task: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The sit → rise → stand → sit cycle for every env, one robotd controller per duck.

    Each env runs robotd's Sit machine (:class:`.falls.PostureSchedule`) until robotd's own
    verdict says it fell; from then on it is left to right itself (:mod:`.falls`). Returns
    the v2 statistics (same definitions, over every episode) and one raw row per episode.
    """
    torch = __import__("torch")
    episode = Episode(env, device, robotd_actions=robotd_actions)
    obs, _ = env.reset(seed=0)
    n = env.num_envs
    hz = schedule.hz
    rise_ticks = round(schedule.rise_s * hz)
    debounce = falls.debounce_ticks(hz)
    hold = falls.hold_ticks(hz)
    every = max(1, round(hz / falls.TRACE_HZ))
    sit, rise, up = (falls.MODES.index(name) for name in ("sit", "rise", "up"))

    def filled(value: Any, dtype: Any) -> Any:
        return torch.full((n,), value, dtype=dtype, device=device)

    long, flag, real = torch.long, torch.bool, torch.float32
    down_run = filled(0, long)
    fallen_before = filled(False, flag)
    settle_fallen = filled(False, flag)
    settle_tilt = filled(0.0, real)
    settle_bad = filled(False, flag)
    fall_tick = filled(-1, long)
    fall_mode = filled(-1, long)
    tilt45_tick = filled(-1, long)
    fall_gravity = torch.full((n, 3), math.nan, device=device)
    landed_gravity = torch.full((n, 3), math.nan, device=device)
    in_recovery = filled(False, flag)
    recovery_rise_end = filled(0, long)
    up_run = filled(0, long)
    upright_tick = filled(-1, long)
    righted_tick = filled(-1, long)
    z_righted = filled(math.nan, real)
    refell_tick = filled(-1, long)
    fell45 = filled(False, flag)
    max_tilt = filled(0.0, real)
    rise_ok = filled(False, flag)
    sit_ok = filled(False, flag)
    drift = filled(0.0, real)
    start_xy = None
    trace_tilt: list[Any] = []
    trace_z: list[Any] = []
    for tick in range(schedule.end):
        nxt = tick + 1
        scripted = falls.MODES.index(schedule.scripted_mode(nxt))
        mode = torch.where(
            in_recovery, torch.where(recovery_rise_end > nxt, rise, up), scripted
        ).to(long)
        use_sit = mode != up
        episode.twist[:] = 0.0
        episode.twist[:, 0] = (mode == sit).float()  # the posture flag rides twist vx
        episode.head[:] = 0.0
        raw = _phase_policy(sit_policy, stand_policy, use_sit, obs)
        scale = torch.where(
            use_sit, robotd_model.ACTION_SCALE["sit"], robotd_model.ACTION_SCALE["stand"]
        )
        action = episode.shape(raw, scale.unsqueeze(1))
        episode.torques.clear()
        obs = episode.step(action, raw)
        tilt = episode.tilt_deg()
        data = episode.robot.data
        z = data.root_link_pos_w[:, 2]
        down_run = falls.next_run(down_run, tilt > falls.FALL_TILT_DEG)
        fallen = down_run >= debounce
        rising = fallen & ~fallen_before
        fallen_before = fallen
        if nxt < schedule.rise_at:
            settle_fallen |= fallen
            if nxt == schedule.rise_at - 1:
                settle_tilt = tilt.clone()
                settle_bad = settle_fallen | (settle_tilt > falls.SETTLE_BAD_TILT_DEG)
        elif nxt < schedule.cycle_end:
            first = rising & (fall_tick < 0)
            fall_tick = torch.where(first, nxt, fall_tick)
            fall_mode = torch.where(first, mode, fall_mode)
            fall_gravity = torch.where(first[:, None], data.projected_gravity_b, fall_gravity)
            # duck-body's answer (spec §5.4): seated → "stand" (1 s rise, then velstand);
            # rising → robotd finishes the rise; standing → velstand keeps driving.
            begin = first & ~settle_bad
            recovery_rise_end = torch.where(
                begin & (mode == sit),
                nxt + 1 + rise_ticks,
                torch.where(
                    begin & (mode == rise),
                    schedule.stand_at,
                    torch.where(begin, 0, recovery_rise_end),
                ),
            )
            in_recovery |= begin
            first45 = (tilt > falls.LEGACY_FALL_TILT_DEG) & (tilt45_tick < 0)
            tilt45_tick = torch.where(first45, nxt, tilt45_tick)
        landed = (fall_tick >= 0) & (fall_tick + hz == nxt)
        landed_gravity = torch.where(landed[:, None], data.projected_gravity_b, landed_gravity)
        upright = in_recovery & (mode == up) & (tilt < falls.UPRIGHT_TILT_DEG) & (righted_tick < 0)
        up_run = falls.next_run(up_run, upright)
        confirmed = up_run >= hold
        righted_tick = torch.where(confirmed, nxt, righted_tick)
        upright_tick = torch.where(confirmed, nxt - hold + 1, upright_tick)
        z_righted = torch.where(confirmed, z, z_righted)
        again = rising & (righted_tick >= 0) & (righted_tick < nxt) & (refell_tick < 0)
        refell_tick = torch.where(again, nxt, refell_tick)
        # The v2 numbers, same definitions (tilt past 45° anywhere in the scripted cycle).
        if nxt <= schedule.sit_end:
            fell45 |= tilt > FALL_TILT_DEG
            max_tilt = torch.maximum(max_tilt, tilt)
        if nxt == schedule.rise_end:  # one second after handing over to velstand
            rise_ok = (z > 0.10) & (tilt < 30.0) & ~fell45
            start_xy = data.root_link_pos_w[:, :2].clone()
        if nxt == schedule.sit_at and start_xy is not None:
            drift = (data.root_link_pos_w[:, :2] - start_xy).norm(dim=1)
        if nxt == schedule.sit_end:
            sit_ok = (z < 0.08) & (tilt < FALL_TILT_DEG)
        if nxt % every == 0:
            trace_tilt.append(tilt.clone())
            trace_z.append(z.clone())
    traced = (settle_fallen | (fall_tick >= 0) | (refell_tick >= 0)).cpu().tolist()
    tilt_trace = torch.stack(trace_tilt, dim=1).cpu().tolist()
    z_trace = torch.stack(trace_z, dim=1).cpu().tolist()
    cols = {
        name: tensor.cpu().tolist()
        for name, tensor in {
            "reset": episode.resets,
            "settle_tilt": settle_tilt,
            "settle_fallen": settle_fallen,
            "settle_bad": settle_bad,
            "fall_tick": fall_tick,
            "fall_mode": fall_mode,
            "fall_gravity": fall_gravity,
            "landed_gravity": landed_gravity,
            "tilt45_tick": tilt45_tick,
            "max_tilt": max_tilt,
            "fell45": fell45,
            "rise_ok": rise_ok,
            "sit_ok": sit_ok,
            "drift": drift,
            "upright_tick": upright_tick,
            "righted_tick": righted_tick,
            "z_righted": z_righted,
            "refell_tick": refell_tick,
        }.items()
    }

    def tick_or_none(value: int) -> int | None:
        return None if value < 0 else int(value)

    def vector_or_none(values: list[float]) -> list[float] | None:
        return None if any(math.isnan(v) for v in values) else [round(v, 4) for v in values]

    rows = []
    for i in range(n):
        row: dict[str, Any] = {
            "scenario": "posture",
            "task": task,
            "episode": i,
            "reset": bool(cols["reset"][i]),
            "settle_tilt_deg": round(cols["settle_tilt"][i], 2),
            "settle_fallen": bool(cols["settle_fallen"][i]),
            "settle_bad": bool(cols["settle_bad"][i]),
            "fall_tick": tick_or_none(cols["fall_tick"][i]),
            "fall_mode": falls.MODES[cols["fall_mode"][i]] if cols["fall_mode"][i] >= 0 else None,
            "fall_gravity": vector_or_none(cols["fall_gravity"][i]),
            "landed_gravity": vector_or_none(cols["landed_gravity"][i]),
            "tilt45_tick": tick_or_none(cols["tilt45_tick"][i]),
            "max_tilt_deg": round(cols["max_tilt"][i], 2),
            "fell45_cycle": bool(cols["fell45"][i]),
            "rise_ok": bool(cols["rise_ok"][i]),
            "sit_ok": bool(cols["sit_ok"][i]),
            "stand_drift_m": round(cols["drift"][i], 4),
            "upright_tick": tick_or_none(cols["upright_tick"][i]),
            "righted_tick": tick_or_none(cols["righted_tick"][i]),
            "z_righted": None
            if math.isnan(cols["z_righted"][i])
            else round(cols["z_righted"][i], 4),
            "refell_tick": tick_or_none(cols["refell_tick"][i]),
        }
        if traced[i]:
            row["trace"] = {
                "hz": falls.TRACE_HZ,
                "first_tick": every,
                "tilt_deg": [round(v, 1) for v in tilt_trace[i]],
                "z_m": [round(v, 3) for v in z_trace[i]],
            }
        rows.append(row)
    legacy = {
        "episodes": n,
        "rise_ok": int(sum(cols["rise_ok"])),
        "falls": int(sum(cols["fell45"])),
        "sit_ok": int(sum(cols["sit_ok"])),
        "stand_drift_m_p95": round(percentile(cols["drift"], 0.95), 4),
        "tilt_deg_p95": round(percentile(cols["max_tilt"], 0.95), 2),
    }
    return legacy, rows


def sweep_posture(
    policies: Path,
    device: str,
    *,
    episodes: int,
    envs: int,
    dr: DomainRandomization,
    robotd_actions: bool,
    schedule: falls.PostureSchedule,
    log: Any,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """The posture cycle in both task physics, ``episodes`` ducks each (one batch)."""
    sit_policy = load_policy(policies / "alpha_sitstand.onnx", device)
    stand_policy = load_policy(policies / "velstand.onnx", device)
    n = min(episodes, envs)
    if n < episodes:
        log(f"posture: {episodes} episodes do not fit one batch of {envs}; running {n}")
    legacy: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for task in (SIT_TASK, STAND_TASK):
        env = build_env(
            task, n, device, seated=True, dr=dr, episode_s=schedule.end / schedule.hz + 10
        )
        began = time.monotonic()
        legacy[task], task_rows = run_posture(
            sit_policy,
            stand_policy,
            env,
            device,
            schedule=schedule,
            robotd_actions=robotd_actions,
            task=task,
        )
        release(env)
        rows.extend(task_rows)
        log(f"posture in {task}: {legacy[task]} in {time.monotonic() - began:.1f}s")
    return legacy, rows


# Straight and curving walks across content.LOCOMOTION, plus turns in place; the
# threshold points (0.15, 0.2) show how reliably the network starts to walk.
WALK_PLANS = tuple(
    (vx, vyaw, seconds)
    for vx in (0.15, 0.2, 0.3, 0.35, 0.4)
    for vyaw in (0.0, 0.5, -0.5)
    for seconds in (2.0, 4.0, 6.0)
) + tuple((0.0, vyaw, seconds) for vyaw in (1.0, -1.0) for seconds in (2.0, 4.0))


def head_offsets(pose: dict[str, float]) -> list[float]:
    """A clip-semantics pose as robot.head offsets from HOME (radians), in HEAD_JOINTS order."""
    targets = content.head_targets(pose, 1)
    return [targets[name] - content.HEAD_HOME[name] for name in HEAD_JOINTS]


def run_walk(
    stand_policy: Any,
    env: Any,
    device: str,
    plans: list[tuple[float, float, float]],
    *,
    robotd_actions: bool,
    heads: list[list[float]] | None = None,
) -> dict[str, list[Any]]:
    """Standing → plan (vx, vyaw, seconds) → stop for 2 s; env i follows plans[i].

    ``heads`` holds each env's head offsets (radians from HOME): robotd slews the head
    there from 1 s before the walk, so it has settled when the legs start, and keeps it.
    """
    torch = __import__("torch")
    episode = Episode(env, device, robotd_actions=robotd_actions)
    obs, _ = env.reset(seed=0)
    n = len(plans)
    hz = robotd_model.CONTROL_HZ
    go = round(SETTLE_S * hz)
    head_at = go - hz
    debounce = falls.debounce_ticks(hz)
    longest = max(seconds for _, _, seconds in plans)
    end = go + round((longest + 2.0) * hz)
    plan = torch.tensor(plans, device=device)
    stop_at = go + (plan[:, 2] * hz).round()
    mid_at = go + (plan[:, 2] * hz / 2).round()
    # The ticks in which any env is in the second half of its walk (host side: no sync).
    steady_ticks = range(int(mid_at.min().item()) + 1, int(stop_at.max().item()) + 1)
    target = torch.zeros(n, 4, device=device)
    if heads is not None:
        target = torch.tensor(heads, dtype=torch.float32, device=device)
    head = torch.zeros(n, 4, device=device)
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    fell_robotd = torch.zeros(n, dtype=torch.bool, device=device)
    down_run = torch.zeros(n, dtype=torch.long, device=device)
    tilts = torch.zeros(n, device=device)
    start_xy = start_yaw = None
    stop_xy = torch.zeros(n, 2, device=device)
    mid_xy = torch.zeros(n, 2, device=device)
    turned_walk = torch.zeros(n, device=device)
    turned_steady = torch.zeros(n, device=device)
    pitch_sum = torch.zeros(n, device=device)
    height_sum = torch.zeros(n, device=device)
    samples = torch.zeros(n, device=device)
    previous_yaw = None
    for tick in range(end):
        nxt = tick + 1
        now = torch.tensor(float(nxt), device=device)
        moving = (nxt >= go) & (now < stop_at)
        episode.twist[:] = 0.0
        episode.twist[:, 0] = torch.where(moving, plan[:, 0], 0.0)
        episode.twist[:, 2] = torch.where(moving, plan[:, 1], 0.0)
        if nxt >= head_at:
            head += robotd_model.HEAD_ALPHA * (target - head)
        episode.head[:] = head
        action, raw = episode.act(stand_policy, obs, robotd_model.ACTION_SCALE["stand"])
        episode.torques.clear()
        obs = episode.step(action, raw)
        tilt = episode.tilt_deg()
        fell |= tilt > FALL_TILT_DEG
        down_run = falls.next_run(down_run, tilt > falls.FALL_TILT_DEG)
        fell_robotd |= down_run >= debounce
        tilts = torch.maximum(tilts, tilt)
        xy = episode.robot.data.root_link_pos_w[:, :2]
        yaw = episode.robot.data.heading_w
        if nxt == go:
            start_xy = xy.clone()
            start_yaw = yaw.clone()
        if previous_yaw is not None:
            step = torch.atan2(torch.sin(yaw - previous_yaw), torch.cos(yaw - previous_yaw))
            walking = (now > go) & (now <= stop_at)
            steady = (now > mid_at) & (now <= stop_at)
            turned_walk += torch.where(walking, step, 0.0)
            turned_steady += torch.where(steady, step, 0.0)
            if nxt in steady_ticks:
                pitch, height = episode.camera()
                pitch_sum += torch.where(steady, pitch, 0.0)
                height_sum += torch.where(steady, height, 0.0)
                samples += steady.float()
        previous_yaw = yaw.clone()
        mid_xy = torch.where((now == mid_at)[:, None], xy, mid_xy)
        stop_xy = torch.where((now == stop_at)[:, None], xy, stop_xy)
    final_xy = episode.robot.data.root_link_pos_w[:, :2]
    yaw = episode.robot.data.heading_w
    turned = torch.atan2(torch.sin(yaw - start_yaw), torch.cos(yaw - start_yaw))
    seconds = plan[:, 2]
    return {
        "fell": fell.cpu().tolist(),
        "fell_robotd": fell_robotd.cpu().tolist(),
        "reset": episode.resets.cpu().tolist(),
        "tilt": tilts.cpu().tolist(),
        "distance": (stop_xy - start_xy).norm(dim=1).cpu().tolist(),
        "steady_distance": (stop_xy - mid_xy).norm(dim=1).cpu().tolist(),
        "overshoot": (final_xy - stop_xy).norm(dim=1).cpu().tolist(),
        "turned": turned.cpu().tolist(),
        "yaw_rate_dps": torch.rad2deg(turned_walk / seconds).cpu().tolist(),
        "steady_yaw_rate_dps": torch.rad2deg(turned_steady / (seconds / 2)).cpu().tolist(),
        "cam_pitch_deg": (pitch_sum / samples.clamp_min(1)).cpu().tolist(),
        "cam_height_m": (height_sum / samples.clamp_min(1)).cpu().tolist(),
    }


def sweep_walk(
    policies: Path,
    device: str,
    *,
    randomizations: int,
    envs: int,
    dr: DomainRandomization,
    robotd_actions: bool,
    log: Any,
) -> dict[str, Any]:
    """The v2 walk table over WALK_PLANS (same statistics as before)."""
    stand_policy = load_policy(policies / "velstand.onnx", device)
    plans = [plan for plan in WALK_PLANS for _ in range(randomizations)]
    walks: dict[str, dict[str, Any]] = {}
    for start in range(0, len(plans), envs):
        batch = plans[start : start + envs]
        env = build_env(STAND_TASK, len(batch), device, seated=False, dr=dr, episode_s=60)
        metrics = run_walk(stand_policy, env, device, batch, robotd_actions=robotd_actions)
        release(env)
        for index, (vx, vyaw, seconds) in enumerate(batch):
            entry = walks.setdefault(
                f"vx{vx:g}/wz{vyaw:g}/{seconds:g}s",
                {
                    "episodes": 0,
                    "falls": 0,
                    "distance": [],
                    "overshoot": [],
                    "turned": [],
                    "tilt": [],
                },
            )
            entry["episodes"] += 1
            entry["falls"] += int(metrics["fell"][index])
            for name in ("distance", "overshoot", "turned", "tilt"):
                entry[name].append(metrics[name][index])
        log(f"walk: plans {start}-{start + len(batch)} of {len(plans)}")
    out = {}
    for key, entry in walks.items():
        vx_text, wz_text, seconds_text = key.split("/")
        out[key] = {
            "episodes": entry["episodes"],
            "falls": entry["falls"],
            "commanded_m": round(float(vx_text[2:]) * float(seconds_text[:-1]), 3),
            "distance_m_mean": round(sum(entry["distance"]) / entry["episodes"], 3),
            "overshoot_m_p95": round(percentile(entry["overshoot"], 0.95), 3),
            "turned_rad_mean": round(sum(entry["turned"]) / entry["episodes"], 3),
            "commanded_rad": round(float(wz_text[2:]) * float(seconds_text[:-1]), 3),
            "tilt_deg_p95": round(percentile(entry["tilt"], 0.95), 2),
        }
    return out


def sweep_head_walk(
    policies: Path,
    device: str,
    *,
    episodes: int,
    envs: int,
    dr: DomainRandomization,
    robotd_actions: bool,
    log: Any,
) -> list[dict[str, Any]]:
    """HEAD_WALK with each head pose, ``episodes`` ducks per pose, poses interleaved."""
    stand_policy = load_policy(policies / "velstand.onnx", device)
    vx, vyaw, seconds = HEAD_WALK
    labels = [label for _ in range(episodes) for label in HEAD_WALK_POSES]
    offsets = {label: head_offsets(pose) for label, pose in HEAD_WALK_POSES.items()}
    rows: list[dict[str, Any]] = []
    for start in range(0, len(labels), envs):
        batch = labels[start : start + envs]
        env = build_env(STAND_TASK, len(batch), device, seated=False, dr=dr, episode_s=60)
        began = time.monotonic()
        metrics = run_walk(
            stand_policy,
            env,
            device,
            [(vx, vyaw, seconds)] * len(batch),
            robotd_actions=robotd_actions,
            heads=[offsets[label] for label in batch],
        )
        release(env)
        for i, label in enumerate(batch):
            rows.append(
                {
                    "scenario": "headwalk",
                    "head": label,
                    "head_offsets_rad": [round(v, 4) for v in offsets[label]],
                    "vx": vx,
                    "vyaw": vyaw,
                    "seconds": seconds,
                    "env": i,
                    "reset": bool(metrics["reset"][i]),
                    "fell": bool(metrics["fell"][i]),
                    "fell_robotd": bool(metrics["fell_robotd"][i]),
                    "tilt": round(metrics["tilt"][i], 2),
                    "distance_m": round(metrics["distance"][i], 4),
                    "speed_mps": round(metrics["distance"][i] / seconds, 4),
                    "steady_speed_mps": round(metrics["steady_distance"][i] / (seconds / 2), 4),
                    "yaw_rate_dps": round(metrics["yaw_rate_dps"][i], 2),
                    "steady_yaw_rate_dps": round(metrics["steady_yaw_rate_dps"][i], 2),
                    "overshoot_m": round(metrics["overshoot"][i], 4),
                    "cam_pitch_deg": round(metrics["cam_pitch_deg"][i], 2),
                    "cam_height_m": round(metrics["cam_height_m"][i], 4),
                }
            )
        log(
            f"headwalk: envs {start}-{start + len(batch)} of {len(labels)} in {time.monotonic() - began:.1f}s"
        )
    return rows


# ------------------------------------------------------------------------ planning
# The second A100 run (docs/validation/evidence/expression-sweep-v2.json) took 519.2 s of
# simulation for these batches — (envs, control ticks): the head sweep (27 clips × 24
# variants × 32, three padded batches per posture, 568 ticks), the posture cycle
# (256 ducks, 14 s, each task physics) and the walk table (1632 ducks, 10 s) — and five
# environment builds.
V2_SECONDS = 519.2
V2_BATCHES = ((8192, 568),) * 6 + ((256, 700),) * 2 + ((1632, 500),)
V2_BUILDS = 5
# Relative cost of one control tick of a batch (a fixed part plus a part that grows with
# the ducks in it) and of one environment build; scaled so the v2 work comes to 519.2 s.
_TICK_FIXED, _TICK_PER_8192, _BUILD = 0.020, 0.045, 30.0


@dataclass(frozen=True)
class Batch:
    scenario: str
    envs: int
    ticks: int


def plan(args: argparse.Namespace) -> tuple[list[Batch], int]:
    """The batches ``run`` will simulate for these arguments, and its environment builds."""
    hz = robotd_model.CONTROL_HZ
    scenarios = _scenarios(args.scenarios)
    batches: list[Batch] = []
    builds = 0

    def clip_batches(name: str, grid: list[Variant], randomizations: int) -> None:
        nonlocal builds
        jobs = len(grid) * randomizations
        size = min(args.envs, jobs)
        ticks = round((SETTLE_S + max(duration_s(v) for v in grid) + TAIL_S) * hz)
        for _posture in ("sit", "stand"):
            builds += 1
            batches.extend(Batch(name, size, ticks) for _ in range(math.ceil(jobs / size)))

    if "head" in scenarios:
        clip_batches("head", variants(_clips(args)), args.randomizations)
    if "look" in scenarios:
        clip_batches("look", look_grid(), args.look_randomizations)
    if "posture" in scenarios:
        schedule = falls.PostureSchedule(stand_s=args.stand_s)
        for _task in (SIT_TASK, STAND_TASK):
            builds += 1
            batches.append(Batch("posture", min(args.posture_episodes, args.envs), schedule.end))
    if "walk" in scenarios:
        plans = len(WALK_PLANS) * args.randomizations
        ticks = round(SETTLE_S * hz) + round((max(p[2] for p in WALK_PLANS) + 2.0) * hz)
        for start in range(0, plans, args.envs):
            builds += 1
            batches.append(Batch("walk", min(args.envs, plans - start), ticks))
    if "headwalk" in scenarios:
        ducks = len(HEAD_WALK_POSES) * args.headwalk_episodes
        ticks = round(SETTLE_S * hz) + round((HEAD_WALK[2] + 2.0) * hz)
        for start in range(0, ducks, args.envs):
            builds += 1
            batches.append(Batch("headwalk", min(args.envs, ducks - start), ticks))
    return batches, builds


def estimate_seconds(batches: list[Batch], builds: int) -> dict[str, float]:
    """A100 simulation time, scaled from the v2 run: a point estimate and a plausible range.

    The range brackets the two extremes the single v2 total cannot tell apart: time that
    grows with every simulated duck-tick, and time that is per batch tick whatever its size.
    """

    def mixed(work: list[tuple[int, int]], builds_: int) -> float:
        return sum(ticks * (_TICK_FIXED + _TICK_PER_8192 * envs / 8192) for envs, ticks in work) + (
            builds_ * _BUILD
        )

    v2 = list(V2_BATCHES)
    work = [(batch.envs, batch.ticks) for batch in batches]
    point = mixed(work, builds) * V2_SECONDS / mixed(v2, V2_BUILDS)
    per_duck_tick = sum(e * t for e, t in work) / sum(e * t for e, t in v2) * V2_SECONDS
    per_batch_tick = sum(t for _, t in work) / sum(t for _, t in v2) * V2_SECONDS
    return {
        "seconds": round(point, 1),
        "low": round(min(point, per_duck_tick, per_batch_tick), 1),
        "high": round(max(point, per_duck_tick, per_batch_tick), 1),
    }


def _scenarios(text: str) -> list[str]:
    names = [name for name in text.split(",") if name]
    unknown = sorted(set(names) - set(SCENARIOS))
    if unknown:
        raise SystemExit(f"unknown scenarios {unknown}; choose from {', '.join(SCENARIOS)}")
    return names


def _clips(args: argparse.Namespace) -> tuple[str, ...] | None:
    return tuple(args.clips.split(",")) if args.clips else None


def rules(schedule: falls.PostureSchedule) -> dict[str, Any]:
    """The definitions every number in a v3 result was counted with."""
    return {
        "fall_robotd": {
            "tilt_deg": round(falls.FALL_TILT_DEG, 3),
            "gravity_z": falls.FALL_GRAVITY_Z,
            "debounce_s": falls.FALL_DEBOUNCE_S,
            "note": "duck-control Safety::observe with robotd-params defaults; one upright sample clears it",
        },
        "fall_legacy_tilt_deg": falls.LEGACY_FALL_TILT_DEG,
        "settle_bad": {
            "tilt_deg": falls.SETTLE_BAD_TILT_DEG,
            "note": "tipped before any action (or robotd fallen then): counted apart, never a fall",
        },
        "righting": {
            "controller": "velstand at zero twist and HOME head; seated falls first get robotd's 1 s rise",
            "upright_tilt_deg": falls.UPRIGHT_TILT_DEG,
            "hold_s": falls.UPRIGHT_HOLD_S,
            "try_s": falls.TRY_S,
            "pause_s": falls.PAUSE_S,
            "window_s": falls.RIGHTING_WINDOW_S,
            "note": "velstand keeps running through the pause (robotd has no pause short of torque-off)",
        },
        "posture_schedule": schedule.describe(),
        "look": {
            "captures_ms_at_tempo_1": {
                name: list(times) for name, times in looks.CAPTURES_MS.items()
            },
            "still": {
                "head_rad_s": looks.STILL_HEAD_RAD_S,
                "gyro_rad_s": looks.STILL_GYRO_RAD_S,
                "for_s": looks.STILL_S,
            },
            "gate_drift_margin_m": looks.DRIFT_MARGIN_M,
            "design_camera": {name: list(value) for name, value in looks.DESIGN_CAMERA.items()},
        },
        "head_walk": {
            "vx": HEAD_WALK[0],
            "vyaw": HEAD_WALK[1],
            "seconds": HEAD_WALK[2],
            "poses": HEAD_WALK_POSES,
            "head_moves_s_before_walk": 1.0,
            "steady": "second half of the walk",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the requested scenarios, writing the result after every phase.

    A failure in one phase is recorded in ``errors`` with its traceback instead of losing
    what already ran; the file is always valid JSON with ``complete`` telling the truth.
    """
    import traceback

    started = time.monotonic()
    policies = Path(args.policies)
    out = Path(args.out)
    trials_path = Path(args.trials) if args.trials else out.with_suffix(".trials.jsonl")

    def log(message: str) -> None:
        print(f"[{time.monotonic() - started:7.1f}s] {message}", flush=True)

    dr = DomainRandomization.level(args.dr)
    clips = _clips(args)
    scenarios = _scenarios(args.scenarios)
    schedule = falls.PostureSchedule(stand_s=args.stand_s)
    batches, builds = plan(args)
    created = datetime.now(UTC).isoformat(timespec="seconds")
    result: dict[str, Any] = {
        "format": envelope.RESULT_FORMAT,
        "created": created,
        "complete": False,
        "errors": [],
        "scenarios": scenarios,
        "source": {
            "content_table": content.table_hash(),
            "clip_digests": {},
            "look_digests": {},
            "policies": {
                name: hashlib.sha256((policies / name).read_bytes()).hexdigest()
                for name in ("alpha_sitstand.onnx", "velstand.onnx")
            },
            "microduck_rl": _git_head(Path(args.microduck_rl)) if args.microduck_rl else None,
            "action_path": "training" if args.training_actions else "robotd",
            "domain_randomization": args.dr,
            "randomizations": args.randomizations,
            "look_randomizations": args.look_randomizations,
            "posture_episodes": min(args.posture_episodes, args.envs),
            "headwalk_episodes": args.headwalk_episodes,
            "device": args.device,
            "trials": {"file": trials_path.name, "rows": 0, "sha256": None},
        },
        "rules": rules(schedule),
        "plan": {
            "batches": [[b.scenario, b.envs, b.ticks] for b in batches],
            "builds": builds,
            "estimate_s": estimate_seconds(batches, builds),
        },
        "variants": {},
        "looks": {},
        "looks_summary": {},
        "motion": {},
        "head_walk": {},
        "tracks": {
            "hz": RECORD_HZ,
            "joints": list(ACTION_JOINTS),
            "extra": ["qw", "qx", "qy", "qz", "z"],
            "by_variant": {},
            "by_look": {},
        },
    }
    header = {
        "scenario": "header",
        "format": envelope.RESULT_FORMAT,
        "created": created,
        "schedule": schedule.describe(),
        "rules": result["rules"],
        "source": {key: value for key, value in result["source"].items() if key != "trials"},
    }
    rows_written = summary.write_rows(trials_path, [header], mode="w") - 1

    def keep(rows: list[dict[str, Any]]) -> None:
        nonlocal rows_written
        rows_written += summary.write_rows(trials_path, rows)

    def save() -> None:
        result["source"]["seconds"] = round(time.monotonic() - started, 1)
        result["source"]["trials"]["rows"] = rows_written
        out.write_text(json.dumps(result, ensure_ascii=False))

    def attempt(phase: str, work: Callable[[], None]) -> None:
        try:
            work()
        except Exception:
            result["errors"].append({"phase": phase, "traceback": traceback.format_exc()})
            log(f"{phase} failed:\n" + traceback.format_exc())
        save()

    log(
        f"plan: {len(batches)} batches, {builds} builds, A100 estimate {result['plan']['estimate_s']}"
    )

    def head() -> None:
        swept = clips or tuple(sorted(content.CLIPS))
        rows: list[dict[str, Any]] = []

        def checkpoint(variants_so_far: dict[str, Any], tracks_so_far: dict[str, Any]) -> None:
            result["variants"] = variants_so_far
            result["tracks"]["by_variant"] = tracks_so_far
            save()
            log(f"checkpoint written to {out}")

        variants_result, tracks = sweep_head(
            policies,
            args.device,
            randomizations=args.randomizations,
            envs=args.envs,
            dr=dr,
            robotd_actions=not args.training_actions,
            clips=clips,
            log=log,
            checkpoint=checkpoint,
            rows_out=rows,
        )
        result["variants"] = variants_result
        result["tracks"]["by_variant"] = tracks
        result["source"]["clip_digests"] = {clip: clip_digest(clip) for clip in swept}
        keep(rows)

    def look() -> None:
        rows: list[dict[str, Any]] = []
        looks_result, tracks = sweep_head(
            policies,
            args.device,
            randomizations=args.look_randomizations,
            envs=args.envs,
            dr=dr,
            robotd_actions=not args.training_actions,
            clips=None,
            log=log,
            grid=look_grid(),
            scenario="look",
            record_if=lambda v: v.amplitude == 1.0 and v.tempo == 1.0,
            rows_out=rows,
        )
        result["looks"] = looks_result
        result["looks_summary"] = summary.summarize_looks(rows)
        result["tracks"]["by_look"] = tracks
        result["source"]["look_digests"] = {name: clip_digest(name) for name in looks.CLIPS}
        keep(rows)

    def posture() -> None:
        legacy, rows = sweep_posture(
            policies,
            args.device,
            episodes=args.posture_episodes,
            envs=args.envs,
            dr=dr,
            robotd_actions=not args.training_actions,
            schedule=schedule,
            log=log,
        )
        result["motion"]["posture"] = {
            task: stats
            | falls.summarize_posture([row for row in rows if row["task"] == task], schedule)
            for task, stats in legacy.items()
        }
        keep(rows)

    def walk() -> None:
        result["motion"]["walk"] = sweep_walk(
            policies,
            args.device,
            randomizations=args.randomizations,
            envs=args.envs,
            dr=dr,
            robotd_actions=not args.training_actions,
            log=log,
        )

    def headwalk() -> None:
        rows = sweep_head_walk(
            policies,
            args.device,
            episodes=args.headwalk_episodes,
            envs=args.envs,
            dr=dr,
            robotd_actions=not args.training_actions,
            log=log,
        )
        result["head_walk"] = summary.summarize_head_walk(rows)
        keep(rows)

    phases: dict[str, Callable[[], None]] = {
        "head": head,
        "look": look,
        "posture": posture,
        "walk": walk,
        "headwalk": headwalk,
    }
    for name in SCENARIOS:  # a fixed order, whatever order they were asked in
        if name in scenarios:
            attempt(name, phases[name])
    result["source"]["trials"]["sha256"] = hashlib.sha256(trials_path.read_bytes()).hexdigest()
    result["complete"] = not result["errors"]
    save()
    log(f"wrote {out} and {trials_path} (complete={result['complete']})")
    return result


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Duck Core expression sweep in the official mjlab physics"
    )
    p.add_argument("--policies", help="v5 policy folder (velstand.onnx, alpha_sitstand.onnx)")
    p.add_argument("--out", help="result JSON; the raw rows go next to it (.trials.jsonl)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--envs", type=int, default=8192, help="parallel ducks per batch")
    p.add_argument("--randomizations", type=int, default=32)
    p.add_argument("--dr", choices=("final", "nominal"), default="final")
    p.add_argument("--clips", default="", help="comma-separated subset (default: all)")
    p.add_argument("--scenarios", default="head,posture,walk", help=f"any of {','.join(SCENARIOS)}")
    p.add_argument(
        "--training-actions", action="store_true", help="skip robotd's scale and low-pass"
    )
    p.add_argument("--microduck-rl", default="")
    p.add_argument("--trials", default="", help="raw rows (default: <out> with .trials.jsonl)")
    p.add_argument("--look-randomizations", type=int, default=96)
    p.add_argument("--posture-episodes", type=int, default=4096, help="ducks per task physics")
    p.add_argument("--stand-s", type=float, default=8.0, help="standing time in the posture cycle")
    p.add_argument("--headwalk-episodes", type=int, default=512, help="ducks per head pose")
    p.add_argument(
        "--plan", action="store_true", help="print the batches and the A100 estimate; run nothing"
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.plan:
        batches, builds = plan(args)
        totals: dict[str, list[int]] = {}
        for batch in batches:
            entry = totals.setdefault(batch.scenario, [0, 0, 0])
            entry[0] += 1
            entry[1] = max(entry[1], batch.envs)
            entry[2] = max(entry[2], batch.ticks)
        for name, (count, envs, ticks) in totals.items():
            print(f"{name:9s} {count} batch(es) of up to {envs} ducks × {ticks} ticks")
        print(f"builds    {builds}")
        print(
            f"A100 simulation estimate (scaled from v2's {V2_SECONDS} s): {estimate_seconds(batches, builds)}"
        )
        return
    if not args.policies or not args.out:
        raise SystemExit("--policies and --out are required (or --plan)")
    run(args)


if __name__ == "__main__":
    main()
