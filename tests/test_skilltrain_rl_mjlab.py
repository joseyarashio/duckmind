"""Stage-1 training tasks on the official physics (tr:tasks): the ONNX importer's parity, the
export round trip, robotd's action path, the observation contract, reward signs, the skill
window and the kinematic tables against the official MJCF.

Needs the official stack (mjlab 1.3.0, microduck_rl cb70b79, torch, onnx, onnxruntime) — the
repository's ``.venv-official-cpu``; skipped where it is absent (the main venv):

    PYTHONPATH=<pytest>:src .venv-official-cpu/bin/python -m pytest tests/test_skilltrain_rl_mjlab.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
pytest.importorskip("mjlab_microduck")
pytest.importorskip("onnxruntime")

from duckmind import paths  # noqa: E402
from duckmind.skilltrain import rl_contract as C  # noqa: E402
from duckmind.skilltrain import rl_onnx as O  # noqa: E402

V5_DIR = paths.v5_policy_dir()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _v5(name: str) -> Path:
    path = V5_DIR / name
    if not path.is_file() or O.sha256(path) != C.V5_SHA256[name]:
        pytest.skip(f"the validated v5 {name} is not at {V5_DIR}")
    return path


# The export round trip's bar. 1e-5 on the Mac (macOS arm64: torch and onnxruntime round the same way); 2e-5 on
# Linux aarch64 (the DGX Spark, GB10: 1.5e-5 measured on velstand, 4.8e-6 on alpha_sitstand), where each float32 path
# is as far from the float64 reference as on the Mac (about 1.2e-5, 6 ulp at the output scale 17) but they differ
# the other way. float32 accumulation order, not the model: see tools/parity_numerics.py and BUGS-2026-10-09.md.
ROUND_TRIP_BAR = 1e-5 if sys.platform == "darwin" else 2e-5


def _parity_rows(path: Path):  # type: ignore[no-untyped-def]
    import numpy as np

    w = O.read_onnx(path)
    rng = np.random.default_rng(0)
    return (w["mean"] + 3.0 * w["div"] * rng.standard_normal((64, w["obs_dim"]))).astype(np.float32)


def test_import_reproduces_the_onnx_and_export_round_trips(tmp_path: Path) -> None:
    for name in ("velstand.onnx", "alpha_sitstand.onnx"):
        path = _v5(name)
        actor = O.build_actor()
        info = O.load_into_actor(actor, path, init_std=0.25)
        assert info["frozen_normalizer"]
        par = O.parity(path, lambda x, a=actor: O.actor_forward(a, x), samples=512)
        assert par["max_abs"] <= 1e-4, par  # the task's bar (typically ~5e-6)
        assert float(actor.distribution.std_param.mean()) == pytest.approx(0.25)
        # PPO's running update cannot move the imported statistics
        before = actor.obs_normalizer._mean.clone()
        from tensordict import TensorDict

        actor.train()
        actor.update_normalization(
            TensorDict({"actor": torch.randn(8192, 61) * 5}, batch_size=[8192])
        )
        assert float((actor.obs_normalizer._mean - before).abs().max()) < 1e-6
        out = O.export_actor(actor, tmp_path / f"rt_{name}", {"test": "round trip"})
        assert O.robotd_problems(out) == []
        round_trip = O.parity(out, O.torch_policy(path, "cpu"), samples=512)
        assert round_trip["max_abs"] <= ROUND_TRIP_BAR, round_trip
        # why the bar is per platform: each float32 path is within the same ~1e-5 of the float64 truth everywhere; on
        # the Mac torch and onnxruntime err the same way (1.9e-6 apart), on the Spark's aarch64 BLAS/MLAS the other
        # way (1.5e-5 apart). The model is the same; tools/parity_numerics.py shows the three distances.
        import numpy as np

        ref = O.reference64(out, _parity_rows(out))
        for label, fn in (("torch", O.torch_policy(path, "cpu")),):
            got = fn(torch.from_numpy(_parity_rows(out))).numpy().astype(np.float64)
            assert float(np.abs(got - ref).max()) <= 2e-5, label
        meta = O.read_onnx(out)
        assert meta["inputs"] == ["obs"] and meta["outputs"] == ["actions"]
        assert meta["hidden_dims"] == (512, 256, 128)


@pytest.fixture(scope="module")
def startle_env():
    import mjlab.tasks  # noqa: F401
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from duckmind.skilltrain import rl_tasks as T

    cells = [
        ("startle_4", "stand"),
        ("hop_back", "walk"),
        ("startle_release", "crouched"),
        ("freeze_3", "coast"),
    ]
    cfg = T.make_env_cfg("startle", eval_cells=cells)
    cfg.scene.num_envs = 4
    cfg.seed = 9
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset(seed=9)
    yield env
    env.close()


def test_observation_contract(startle_env) -> None:
    env = startle_env
    obs = env.observation_manager.compute()
    assert obs["actor"].shape == (4, C.OBS_LEN)
    assert env.observation_manager.active_terms["actor"] == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command",
        "head_command", "body_command",
    ]  # fmt: skip
    a, b = C.OBS_SLICES["twist"]
    plan = env.skill_plan
    assert torch.allclose(obs["actor"][:, a:b], plan.command)  # the hold shows the table's code
    h0, b1 = C.OBS_SLICES["head"][0], C.OBS_SLICES["body"][1]
    assert torch.all(obs["actor"][:, h0:b1] == 0)  # robotd zeroes head and body in a skill
    assert obs["critic"].shape[1] > C.OBS_LEN
    assert plan.command[0].tolist() == [1.0, 0.0, 0.0]
    assert plan.command[1].tolist() == [1.0, 1.0, 0.0]
    assert plan.hold_ticks.tolist() == [25, 40, 40, 100]
    assert plan.unwind_ticks.tolist() == [40, 30, 0, 40]
    kp = env.scene["robot"].actuators[0].kp_scale
    assert torch.allclose(kp, torch.full_like(kp, C.pinned("startle")[1]))


def test_action_path_is_robotds(startle_env) -> None:
    env = startle_env
    term = env.action_manager.get_term("joint_pos")
    prev, has_prev = term.filter_state
    prev0 = prev.clone()
    raw = torch.full((4, 14), 0.1)
    term.process_actions(raw)
    vf = term.voltage_factor()
    assert torch.all((vf > 0.85) & (vf < 1.2))
    home = torch.tensor(C.HOME)
    unfiltered = home + C.pinned("startle")[0] * vf * raw
    alpha = torch.tensor([C.LOWPASS_HEAD if i in C.HEAD_IDX else C.LOWPASS_LEGS for i in range(14)])
    expected = torch.where(
        has_prev.unsqueeze(1), alpha * unfiltered + (1 - alpha) * prev0, unfiltered
    )
    assert torch.allclose(term._processed_actions, expected, atol=1e-6)
    assert torch.all(has_prev)  # every entry carried robotd's filter state
    # the entry's last raw action is what the first observation fed back (robotd shares it)
    assert torch.allclose(env.action_manager.action, env.skill_entry.last_raw)
    from duckmind.skilltrain.rl_action import lowpass_cutoff_hz

    # exact −3 dB of the discrete filter at 50 Hz (the plan quotes ≈ 5.5 / 9.6 Hz)
    assert lowpass_cutoff_hz(0.5) == pytest.approx(5.75, abs=0.05)
    assert lowpass_cutoff_hz(0.7) == pytest.approx(11.03, abs=0.05)


def test_rewards_have_the_declared_signs_and_the_window_ends(startle_env) -> None:
    env = startle_env
    from duckmind.skilltrain import rl_mdp as M
    from duckmind.skilltrain import rl_tasks as T

    costs = set(T.cost_terms(env.cfg))
    rm = env.reward_manager
    seen_done = torch.zeros(4, dtype=torch.bool)
    for _ in range(70):
        _obs, _rew, term, trunc, _ = env.step(torch.randn(4, 14) * 0.3)
        seen_done |= term | trunc
        for i, name in enumerate(rm.active_terms):
            cfg = rm.get_term_cfg(name)
            if cfg.weight == 0:
                continue
            value = rm._step_reward[:, i] / cfg.weight
            if name in costs:
                assert torch.all(value >= -1e-6), (name, value)
            elif name not in T.SIGNED_TERMS:
                assert torch.all(value >= -1e-6), (name, value)
    assert bool(seen_done[0]) and bool(seen_done[2])  # 0.5 + 0.8 s = 65 ticks; the 0.8 s release
    assert M.PHASES[M.HOLD] == "hold"


def test_kinematic_table_matches_the_official_mjcf() -> None:
    import mujoco
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_XML

    m = mujoco.MjModel.from_xml_path(str(MICRODUCK_ALLCOLLISIONS_XML))
    d = mujoco.MjData(m)

    def feet(q):
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        for name, v in zip(C.JOINTS, q, strict=True):
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = v
        mujoco.mj_kinematics(m, d)
        out = []
        for side in ("left_foot", "right_foot"):
            i = m.site(side).id
            out.append((d.site_xpos[i].copy(), d.site_xmat[i].reshape(3, 3).copy()))
        return out

    home = feet(C.HOME)
    for dip in (-0.008, 0.01, 0.025):
        got = feet(C.home_dip_pose(dip))
        for (p0, r0), (p1, r1) in zip(home, got, strict=True):
            dz = (p1[2] - p0[2]) - dip  # the feet rise by the dip in the trunk frame
            assert abs(dz) < 5e-4 and abs(p1[0] - p0[0]) < 5e-4, (dip, p1 - p0)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, (float((r0.T @ r1).trace()) - 1) / 2))))
            assert ang < 0.2, (dip, ang)
