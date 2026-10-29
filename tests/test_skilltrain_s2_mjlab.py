"""Stage-2 tasks on the official physics (s2:build): the observation contract, robotd's pose / twist /
head semantics in the command block, the servo-delay model, reward signs, the re-conditioned v5
import, the critic warm-up, the trainer end to end with export, and the battery.

Needs the official stack (mjlab 1.3.0, microduck_rl cb70b79, torch, onnx, onnxruntime) — the
repository's ``.venv-official-cpu``; skipped where it is absent:

    PYTHONPATH=src .venv-official-cpu/bin/python -m pytest tests/test_skilltrain_s2_mjlab.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
pytest.importorskip("mjlab_microduck")
pytest.importorskip("onnxruntime")

from duckmind import paths  # noqa: E402
from duckmind.skilltrain import rl_onnx as O  # noqa: E402
from duckmind.skilltrain import s2_contract as C  # noqa: E402
from duckmind.skilltrain.s2_battery import NEURO_SRC_DEFAULT  # noqa: E402

V5_DIR = paths.v5_policy_dir()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
REPO = Path(__file__).resolve().parents[1]


def _v5(name: str = "velstand.onnx") -> Path:
    path = V5_DIR / name
    if not path.is_file() or O.sha256(path) != C.V5_SHA256[name]:
        pytest.skip(f"the validated v5 {name} is not at {V5_DIR}")
    return path


@pytest.fixture(scope="module")
def pose_env():
    import mjlab.tasks  # noqa: F401
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from duckmind.skilltrain import s2_tasks as T

    cfg = T.make_env_cfg("pose")
    cfg.scene.num_envs = 8
    cfg.seed = 5
    env = ManagerBasedRlEnv(cfg, device="cpu")
    obs, _ = env.reset(seed=5)
    policy = O.torch_policy(_v5(), "cpu")
    yield env, obs, policy
    env.close()


def _roll(env, obs, policy, ticks):
    for _ in range(ticks):
        obs, *_ = env.step(policy(obs["actor"]))
    return obs


def test_observation_contract_and_robotds_command_block(pose_env) -> None:
    from duckmind.skilltrain import s2_mdp as M

    env, obs, policy = pose_env
    assert tuple(obs["actor"].shape) == (8, 61) and tuple(obs["critic"].shape) == (8, 100)
    st = M.state_of(env)
    seen_pose = False
    for _ in range(12):
        obs = _roll(env, obs, policy, 25)
        x = obs["actor"]
        assert torch.allclose(x[:, 48:51], st.twist_eff, atol=1e-6)
        assert torch.allclose(x[:, 51:55], st.head_ema, atol=1e-6)
        assert torch.allclose(x[:, 57:60], st.body_ema, atol=1e-6)
        assert float(x[:, [55, 56, 60]].abs().max()) == 0.0
        if bool(st.pose_on.any()):
            seen_pose = True
            assert float(x[st.pose_on, 48:51].abs().max()) == 0.0  # pose mode zeroes the twist
        assert float(x[~st.pose_on, 57:60].abs().max()) == 0.0  # released: the body block is 0
    assert seen_pose, "no env entered pose mode in 6 s"


def test_the_servo_delay_model(pose_env) -> None:
    env, obs, policy = pose_env
    act = env.action_manager.get_term("joint_pos")
    assert int(act.lag.min()) >= 0 and int(act.lag.max()) <= 8
    buf = [a for a in env.scene["robot"].actuators if getattr(a, "_delay_buffer", None) is not None]
    assert buf, "the BAM actuator has no delay buffer"
    before = act.lag.clone()
    _roll(env, obs, policy, 3)
    lags = buf[0]._delay_buffer.current_lags
    assert torch.equal(lags.cpu(), act.lag.cpu())  # only the action term sets the lags
    jit = act.jitter
    if bool(jit.any()):
        assert bool(((act.lag[jit] >= 3) & (act.lag[jit] <= 6)).all())
    assert torch.equal(before[~jit], act.lag[~jit])  # a held delay stays for the episode


def test_reward_signs_and_finite_values(pose_env) -> None:
    from duckmind.skilltrain import s2_mdp as M
    from duckmind.skilltrain import s2_tasks as T

    env, obs, policy = pose_env
    _roll(env, obs, policy, 10)
    rm = env.reward_manager
    costs = set(T.cost_terms(env.cfg))
    for name in rm._term_names:
        cfg = rm.get_term_cfg(name)
        v = cfg.func(env, **cfg.params)
        assert torch.isfinite(v).all(), name
        if name in costs and name not in T.SIGNED_TERMS:
            assert float(v.min()) >= 0.0, f"cost {name} returned {float(v.min())} < 0"
    for fn in M.COST_FUNCS:
        assert fn.__name__ in {"steps_standing", "stand_leg_torque", "release_rate", "cam_smear", "head_torque",
                               "head_vel", "neck_accel"}  # fmt: skip
    assert {
        "pose_track",
        "posture",
        "upright",
        "stand_leg_torque",
        "release_rate",
        "head_fine",
    } <= set(rm._term_names)
    assert "pose" not in rm._term_names and "body_pose_tracking" not in rm._term_names


def test_the_reconditioned_import_keeps_v5s_function() -> None:
    from duckmind.skilltrain import s2_bc as B

    path = _v5()
    actor = O.build_actor()
    info = B.import_v5(actor, path)
    assert info["parity"]["max_abs"] <= 1e-4
    std = actor.obs_normalizer._std.reshape(-1)
    for slot, value in C.RECONDITION_STD.items():
        assert float(std[slot]) == pytest.approx(value, rel=1e-5)
    x = torch.from_numpy(B.pose_observations(path, n=64))
    ref = O.torch_policy(path, "cpu")(x)
    got = O.actor_forward(actor, x)
    assert float((got - ref).abs().max()) <= 1e-4
    teacher = B.teacher_from_onnx(actor, path)
    xz = B.teacher_input(x)
    assert (
        float((O.actor_forward(teacher, xz) - O.torch_policy(path, "cpu")(xz)).abs().max()) <= 1e-4
    )


def _train(args: list[str], out: Path) -> dict:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-m", "duckmind.skilltrain.s2_train", *args, "--out", str(out)], env=env,
                       capture_output=True, text=True, timeout=1800)  # fmt: skip
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads((out / "metrics.json").read_text())


def test_training_end_to_end_warmup_bc_export_and_r4_from_r3(tmp_path: Path) -> None:
    _v5()
    common = [
        "--envs",
        "8",
        "--device",
        "cpu",
        "--v5",
        str(V5_DIR),
        "--print-every",
        "1",
        "--save-every",
        "1",
    ]
    out = tmp_path / "pose"
    m = _train(["--stage", "pose", "--iterations", "3", "--critic-warmup", "2", *common], out)
    assert m["stop_reason"] == "iterations" and m["warm"]["import"]["parity"]["max_abs"] <= 1e-4
    rows = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
    assert [r["warmup"] for r in rows] == [1, 1, 0]
    ck0 = torch.load(out / "checkpoints" / "model_0.pt", map_location="cpu", weights_only=False)
    ck1 = torch.load(out / "checkpoints" / "model_1.pt", map_location="cpu", weights_only=False)
    for k, v in ck0["actor_state_dict"].items():
        assert torch.equal(v, ck1["actor_state_dict"][k]), (
            f"the critic warm-up moved the actor ({k})"
        )
    assert any(
        not torch.equal(v, ck1["critic_state_dict"][k]) for k, v in ck0["critic_state_dict"].items()
    )
    exp = m["export"]
    assert exp["parity"]["max_abs"] <= 1e-5 and exp["robotd_problems"] == []
    man = json.loads((out / "final" / "manifest.json").read_text())
    assert man["slot"] == "walk" and man["kind"] == "perpetual" and man["duckmind"]["pose_capable"]
    assert "bc_loss" in rows[-1]["losses"]
    gaze = tmp_path / "gaze"
    m2 = _train(["--stage", "gaze", "--iterations", "2", "--init", str(out / "checkpoints" / "latest.pt"),
                 "--anchor-onnx", str(out / "final" / C.WALK_FILE), "--critic-warmup", "1", *common], gaze)  # fmt: skip
    assert m2["stop_reason"] == "iterations" and "init" in m2["warm"]
    assert m2["export"]["robotd_problems"] == []
    assert (
        json.loads((gaze / "final" / "manifest.json").read_text())["duckmind"]["gaze_stable"]
        is True
    )


def test_the_battery_runs_and_scores_the_gates(tmp_path: Path) -> None:
    neuro = Path(os.environ.get("DUCKMIND_NEURO_A100_SRC") or NEURO_SRC_DEFAULT)
    if not (neuro / "neuro_a100").is_dir():
        pytest.skip("the neuro_a100 harness is not available")
    _v5()
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO / "src"),
        "CUDA_VISIBLE_DEVICES": "",
        "DUCKMIND_NEURO_A100_SRC": str(neuro),
    }
    out = tmp_path / "battery.json"
    r = subprocess.run([sys.executable, "-m", "duckmind.skilltrain.s2_battery", "--walk", str(_v5()), "--policies",
                        str(V5_DIR), "--out", str(out), "--mode", "quick", "--n", "1", "--envs", "8", "--device", "cpu",
                        "--only", "S.head.micro,S.lean.roll+0.2"], env=env, capture_output=True, text=True,
                       timeout=1800)  # fmt: skip
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads(out.read_text())
    assert res["complete"] and "cand" in res["arms"]
    g = res["gates"]["j3-6"]
    assert (
        g["lean_roll_deg"]["verdict"] in ("PASS", "FAIL")
        and g["yaw_micro_gain"]["value"] is not None
    )
    # v5 itself does not lean (the "before" of the gate)
    assert abs(g["lean_roll_deg"]["value"]) < 2.0
