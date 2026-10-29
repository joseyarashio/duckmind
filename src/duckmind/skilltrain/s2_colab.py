"""The Stage-2 runner on the Colab A100 (s2:build, 2026-09-28): one detached job per session.

Runs ON the VM with the system Python (orchestration only); the work runs in the official
microduck_rl environment (``/content/microduck_rl/.venv``). Started by a short ``colab exec`` cell
(``colab/make_launch_cell.py`` in the Stage-2 data folder) and polled by the Mac through
``/content/out/s2_status.json`` with ``colab download`` (the file API, never a second ``exec`` on a
busy session — the Stage-1 RUNBOOK lessons: one ``colab exec`` per session at a time, hard timeouts,
detached long runs, a keep-alive cell every ≤ 3 min).

    python3 -m duckmind.skilltrain.s2_colab job        # the job named by S2_JOB (below)
    python3 -m duckmind.skilltrain.s2_colab status     # print the status file
    python3 -m duckmind.skilltrain.s2_colab collect    # tar what exists

Jobs (``S2_JOB``):

* ``calibrate`` (S0-2, ~30–40 min): setup; the v5 import's GPU parity (with the body slots
  re-conditioned); the **v5 control battery** (the Stage-2 programs on v5, n = 512 — the baselines the
  stored A100 run lacks; kept for every later gate); seconds per iteration of R3 at 4,096 and 8,192
  envs and of R4 at 4,096 (40 / 30 / 30 it); the **warm-start sanity run** (R3 from v5: 100 critic-only
  + 100 actor iterations; its quick battery must keep v5's walking within 5 %); the export round trip;
  the quick battery's wall time → ``calibration.json`` with the unit estimate for the chosen sizes.
* ``train``: one attempt of one stage (``S2_STAGE``): the trainer detached; every new checkpoint
  gets the **quick battery** in a separate process (n ≈ 48 per condition, the core gates) while the
  training continues; at the end the export and its quick battery; a results tar.
* ``battery``: the V2 battery of an exported network (``S2_WALK``): ``full`` at the recipe's delay +
  the core set at 0 / 20 / 40 ms, n = 512 (2,048 pooled stop-go programs). ``S2_V5CTL_FIRST=1`` runs
  the v5 control battery first when the bundle has none (s2:train moved it out of S0).
* ``chain`` (s2:train, 2026-09-28): ``calibrate`` (with ``S2_CTL_DELAYS=none`` ≤ 5 units), the
  automatic S0 go / no-go (:func:`go_no_go`), then ``train`` of ``S2_STAGE`` with the time left — one
  session, so the A100 never idles while the Mac reads the calibration and the lock is not lost
  between S0 and R3.

Control files (uploaded by the Mac): ``/content/out/STOP`` (graceful: the trainer saves and
exports), ``/content/out/ABORT`` (now). The job also stops itself before ``S2_DEADLINE_MIN``
(SIGINT at −7 min, SIGKILL at −1 min). ``S2_DRYRUN=1`` rehearses a job on the Mac (CPU, tiny sizes,
the repository's code and ``.venv-official-cpu``; no setup).

Settings (env): S2_JOB, S2_STAGE (pose|gaze), S2_TAG, S2_ITERS, S2_ENVS (4096), S2_BUDGET_MIN,
S2_DEADLINE_MIN, S2_INIT (v5 | a checkpoint path on the VM), S2_ANCHOR_ONNX, S2_RESUME, S2_SEED,
S2_SAVE_EVERY (250), S2_EVAL_N (48), S2_EXTRA (more trainer flags), S2_WALK (battery), S2_DELAYS,
S2_BATTERY_MODE (full), S2_BATTERY_N (512), S2_CAL_ITERS ("40,30,30"), S2_SANITY_ITERS ("100,100"),
S2_SANITY_EXTRA (more flags for the sanity run, e.g. ``--topple-scale 0.5``), S2_CTL_N (512),
S2_CTL_DELAYS ("j3-6,d0,d20,d40": the v5 control battery's delays; "none" skips it in S0),
S2_V5CTL_FIRST (0), S2_EVAL_ENVS (quick batteries: 1024 ducks per world instead of 7,168),
S2_THEN_BATTERY (chain: 0 | 1, the V2 battery of the export after the training), S2_V2_RESERVE_MIN (60).
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

T0 = time.time()
DRY = os.environ.get("S2_DRYRUN", "0") == "1"
ROOT = Path(
    os.environ.get("S2_ROOT")
    or ("/content" if Path("/content").is_dir() and not DRY else os.getcwd())
)
STAGE_DIR = ROOT / "stage2"
OUT = ROOT / "out"
MRL = ROOT / "microduck_rl"
PY = Path(os.environ.get("S2_PY") or (MRL / ".venv" / "bin" / "python"))
STATUS = OUT / "s2_status.json"
STOP, ABORT = OUT / "STOP", OUT / "ABORT"
MICRODUCK_RL_COMMIT = "cb70b792312d559a4da09064d92009079671815f"
MICRODUCK_RL_GIT = "https://github.com/pollen-robotics/microduck_rl.git"
UNITS_PER_HOUR = 6.77
V5 = {
    "velstand.onnx": "1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45",
    "alpha_sitstand.onnx": "c6c40e35e726eabd803d633e090d112994f469921152448367953fbaf9799bc8",
    "alpha_stand.onnx": "1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b",
    "manifest.json": "622048c2c23ea58942023f66fd16b189a875fd169e88d85beb16ebbe63b20c94",
}
CKPT_EVAL_TIMEOUT_S = 15 * 60
WALK_ONNX = "velstand_duck.onnx"  # s2_contract.WALK_FILE (the runner imports nothing of ours)


def env(name: str, default: str = "") -> str:
    return os.environ.get("S2_" + name.upper(), default)


JOB = env("job", "status")
STAGE = env("stage", "pose")
TAG = env("tag", "p1")
DEVICE = env("device", "cpu" if DRY else "cuda:0")
DEADLINE_MIN = float(env("deadline_min", "120"))
STATE: dict[str, Any] = {
    "format": "duckmind-s2-status-1",
    "job": JOB,
    "stage": STAGE,
    "tag": TAG,
    "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "deadline_min": DEADLINE_MIN,
    "step": None,
    "done": False,
    "checkpoints": [],
    "evals": {},
    "history": [],
}


def elapsed_min() -> float:
    return (time.time() - T0) / 60.0


def say(msg: str) -> None:
    line = f"[s2 {JOB}/{STAGE}-{TAG} {time.strftime('%H:%M:%S')} +{elapsed_min():.1f}m] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"s2_{JOB}_{STAGE}-{TAG}.log").open("a") as f:
        f.write(line + "\n")


def save_status() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE["t_min"] = round(elapsed_min(), 2)
    STATE["updated"] = time.strftime("%H:%M:%S")
    STATE["units_estimate"] = round(elapsed_min() / 60.0 * UNITS_PER_HOUR, 2)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(STATE, indent=1, default=str) + "\n")
    tmp.replace(STATUS)


def sh(
    cmd: str,
    cwd: Path | None = None,
    check: bool = True,
    tail: int = 2000,
    envx: dict | None = None,
) -> str:
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, env=envx)
    text = (r.stdout + r.stderr)[-tail:]
    if check and r.returncode:
        print(text, flush=True)
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")
    return text


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child_env() -> dict[str, str]:
    e = dict(os.environ)
    src = str(
        (STAGE_DIR / "src")
        if not DRY
        else Path(env("dry_src", str(Path(__file__).resolve().parents[2])))
    )
    if not DRY:
        neuro = str(STAGE_DIR / "neuro")
    else:
        from duckmind.skilltrain.s2_battery import NEURO_SRC_DEFAULT

        neuro = str(Path(env("dry_neuro", str(NEURO_SRC_DEFAULT))))
    e.update(PYTHONPATH=src, DUCKMIND_NEURO_A100_SRC=neuro, WANDB_MODE="disabled", PYTHONUNBUFFERED="1",
             HF_HUB_DISABLE_TELEMETRY="1")  # fmt: skip
    if sys.platform.startswith("linux"):
        e["MUJOCO_GL"] = "egl"
    if DEVICE == "cpu":
        e.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    return e


def policies_dir() -> Path:
    if not DRY:
        return STAGE_DIR / "policies"
    from duckmind import paths

    return Path(env("dry_policies", str(paths.v5_policy_dir())))


def baselines_dir() -> Path:
    return (
        (STAGE_DIR / "baselines") if not DRY else Path(env("dry_baselines", str(OUT / "baselines")))
    )


def gpu() -> str:
    try:
        return sh(
            "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader",
            check=False,
        ).strip()
    except Exception:  # noqa: BLE001
        return "no nvidia-smi"


# ── setup ────────────────────────────────────────────────────────────────────────────────────


def setup() -> None:
    STATE["step"] = "setup"
    save_status()
    if DRY:
        say("dry run: setup skipped")
        return
    t = time.time()
    bundle = ROOT / "stage2-bundle.zip"
    if not (STAGE_DIR / "MANIFEST.json").exists():
        if not bundle.exists():
            raise SystemExit("upload stage2-bundle.zip to /content first")
        with zipfile.ZipFile(bundle) as z:
            z.extractall(ROOT)
    manifest = json.loads((STAGE_DIR / "MANIFEST.json").read_text())
    bad = [n for n, d in manifest["sha256"].items() if sha256(STAGE_DIR / n) != d]
    if bad:
        raise SystemExit(f"stage2-bundle.zip differs from its manifest: {bad[:5]} (re-upload it)")
    for name, digest in V5.items():
        got = sha256(STAGE_DIR / "policies" / name)
        if got != digest:
            raise SystemExit(f"{name}: sha256 {got[:16]} is not the validated v5")
    if not MRL.exists():
        r = subprocess.run(["git", "clone", "-q", MICRODUCK_RL_GIT, str(MRL)])
        if r.returncode:
            backup = ROOT / "microduck_rl-src.zip"
            if not backup.exists():
                raise SystemExit("git clone failed: upload microduck_rl-src.zip to /content")
            with zipfile.ZipFile(backup) as z:
                z.extractall(ROOT)
    if (MRL / ".git").exists():
        sh(f"git -C {MRL} checkout -q {MICRODUCK_RL_COMMIT}")
        head = sh(f"git -C {MRL} rev-parse HEAD").strip()
        if head != MICRODUCK_RL_COMMIT:
            raise SystemExit(f"microduck_rl at {head}, not {MICRODUCK_RL_COMMIT}")
    if not PY.exists():
        say("installing the official environment (uv sync --locked)")
        sh("pip -q install uv")
        sh("uv sync --locked 2>&1 | tail -3", cwd=MRL)
        sh("uv pip install onnxruntime 2>&1 | tail -1", cwd=MRL)
    check = sh(f"{PY} -c \"import torch, mjlab, onnxruntime; print('torch', torch.__version__, 'cuda', "
               f"torch.cuda.is_available())\"", envx=child_env())  # fmt: skip
    STATE["setup_s"] = round(time.time() - t, 1)
    STATE["gpu"] = gpu()
    STATE["bundle"] = manifest.get("built")
    say(f"setup done in {STATE['setup_s']} s on {STATE['gpu']}: {check.strip().splitlines()[-1]}")


# ── processes ────────────────────────────────────────────────────────────────────────────────


def spawn(args: list[str], log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("a")
    return subprocess.Popen([str(PY), *args], cwd=str(ROOT), env=child_env(), stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)  # fmt: skip


def killpg(p: subprocess.Popen | None, sig: int) -> None:
    if p is None:
        return
    try:
        os.killpg(p.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def wait_proc(p: subprocess.Popen, poll_s: float = 15.0) -> int | None:
    """Wait for a child while refreshing the status file (the Mac supervisor calls a status that
    stops changing a stall: 12 min warns, 25 min stops the session — a 15-min sanity run must not
    look like one); ABORT or the deadline − 2 min kill it (s2:train, 2026-09-28)."""
    while p.poll() is None:
        if ABORT.exists() or elapsed_min() > DEADLINE_MIN - 2:
            killpg(p, signal.SIGKILL)
            p.wait()
            break
        save_status()
        time.sleep(poll_s)
    save_status()
    return p.returncode


def trainer_args(
    stage: str, out: Path, iters: int, envs: int, budget_min: float, init: str, extra: list[str]
) -> list[str]:
    args = ["-m", "duckmind.skilltrain.s2_train", "--stage", stage, "--envs", str(envs), "--iterations", str(iters),
            "--device", DEVICE, "--out", str(out), "--v5", str(policies_dir()), "--init", init,
            "--time-budget-min", str(budget_min), "--save-every", env("save_every", "250"), "--print-every", "10",
            "--seed", env("seed", "1")]  # fmt: skip
    if env("anchor_onnx"):
        args += ["--anchor-onnx", env("anchor_onnx")]
    if env("resume"):
        args += ["--resume", env("resume")]
    return args + extra


def battery_args(walk: str, out: Path, mode: str, n: int, delays: str, stage: str) -> list[str]:
    args = ["-m", "duckmind.skilltrain.s2_battery", "--policies", str(policies_dir()), "--out", str(out), "--mode", mode,
            "--n", str(n), "--delays", delays, "--device", DEVICE, "--stage", stage,
            "--v5-results", str(baselines_dir() / "results.json")]  # fmt: skip
    ctl = baselines_dir() / "v5ctl.json"
    if ctl.exists():
        args += ["--v5-ctl", str(ctl)]
    if walk:
        args += ["--walk", walk]
    if DRY:
        args += ["--envs", "16", "--only", env("dry_only", "S.head.micro,C.idle.v5.quiet")]
    elif mode == "quick" or env("eval_envs"):
        # a quick battery runs beside the training on the same GPU: size its world to its ducks
        # (18 conditions × 48 = 864) instead of neuro_a100's 7,168 (s2:train, 2026-09-28)
        args += ["--envs", env("eval_envs", "1024")]
    return args


def last_progress(run: Path) -> dict | None:
    p = run / "progress.jsonl"
    if not p.exists():
        return None
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    try:
        return json.loads(lines[-1]) if lines else None
    except ValueError:
        return None


BRIEF = (
    "Episode_Reward/pose_track", "Episode_Reward/pose_composite", "Episode_Reward/gaze_yaw", "Episode_Reward/cam_smear",
    "Episode_Reward/track_linear_velocity", "Episode_Metrics/fallen45", "Episode_Metrics/lean_deg",
    "Episode_Metrics/pose_err_deg", "Episode_Metrics/z_err_mm", "Episode_Metrics/cam_perp_dps",
    "Episode_Metrics/stand_leg_rms", "Episode_Metrics/steps_standing", "Episode_Metrics/neck_dev",
)  # fmt: skip


def brief(row: dict | None) -> dict:
    if not row:
        return {}
    out = {
        "it": row.get("it"),
        "s": row.get("s"),
        "std": row.get("std"),
        "lr": row.get("lr"),
        "warmup": row.get("warmup"),
    }
    for k in BRIEF:
        if k in row:
            out[k.split("/", 1)[1]] = row[k]
    losses = row.get("losses") or {}
    for k in ("bc_loss", "bc_anchor_frac", "bc_fallen_frac", "value"):
        if k in losses:
            out[k] = losses[k]
    if row.get("sign_problems"):
        out["sign_problems"] = row["sign_problems"]
    return out


def gates_brief(path: Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}
    out = {}
    for delay, g in (d.get("gates") or {}).items():
        out[delay] = {k: [v.get("value"), v.get("verdict")] for k, v in g.items()}
    return {
        "gates": out,
        "complete": d.get("complete"),
        "seconds": (d.get("meta") or {}).get("seconds"),
    }


def watch_training(trainer: subprocess.Popen, run: Path, stage: str, eval_n: int) -> None:
    """Poll the trainer; each new checkpoint → a quick battery in its own process (one at a time)."""
    STATE["step"] = "train"
    save_every = int(env("save_every", "250"))
    queue: list[int] = []
    queued: set[int] = set()
    ev: tuple[subprocess.Popen, Path, float, int] | None = None
    sigint_at = None
    last_hist = -1
    while True:
        rc = trainer.poll()
        row = last_progress(run)
        if row:
            STATE["last"] = brief(row)
            it = int(row.get("it", 0))
            if it // 25 > last_hist:
                last_hist = it // 25
                STATE["history"].append(brief(row))
        its = sorted(int(p.stem.removeprefix("model_")) for p in (run / "checkpoints").glob("model_*.pt")
                     if p.stem.removeprefix("model_").isdigit())  # fmt: skip
        STATE["checkpoints"] = [f"model_{i}.pt" for i in its]
        for i in its:
            if i not in queued and (i + 1) % save_every == 0:
                queue.append(i)
                queued.add(i)
        if ev is not None:
            p, out, t_ev, it_ev = ev
            if p.poll() is not None or time.time() - t_ev > CKPT_EVAL_TIMEOUT_S:
                if p.poll() is None:
                    killpg(p, signal.SIGKILL)
                entry = {
                    "rc": p.returncode,
                    "wall_s": round(time.time() - t_ev, 1),
                    **(gates_brief(out) if out.exists() else {"error": "no output"}),
                }
                STATE["evals"][str(it_ev)] = entry
                say(f"checkpoint battery it {it_ev}: rc {p.returncode} in {entry['wall_s']} s")
                ev = None
        if ev is None and queue and sigint_at is None and not ABORT.exists():
            i = queue.pop(0)
            out = run / "evals" / f"battery_{i}.json"
            proc = spawn(battery_args(str(run / "checkpoints" / f"model_{i}.pt"), out, "quick", eval_n, "j3-6", stage),
                         run / "evals" / f"battery_{i}.log")  # fmt: skip
            ev = (proc, out, time.time(), i)
            say(f"checkpoint battery it {i} started (n = {eval_n})")
        STATE["eval_running"] = ev[3] if ev else None
        STATE["eval_queue"] = list(queue)
        if ABORT.exists():
            say("ABORT: killing the trainer and the battery")
            killpg(trainer, signal.SIGKILL)
            if ev:
                killpg(ev[0], signal.SIGKILL)
            STATE["aborted"] = "ABORT file"
            break
        if STOP.exists() and sigint_at is None and rc is None:
            say("STOP: SIGINT to the trainer (it saves and exports)")
            killpg(trainer, signal.SIGINT)
            sigint_at = time.time()
        if rc is None and sigint_at is None and elapsed_min() > DEADLINE_MIN - 7:
            say("deadline − 7 min: SIGINT to the trainer")
            killpg(trainer, signal.SIGINT)
            sigint_at = time.time()
        if rc is None and elapsed_min() > DEADLINE_MIN - 1:
            killpg(trainer, signal.SIGKILL)
        if rc is not None:
            STATE["trainer_rc"] = rc
            if ev is not None and elapsed_min() < DEADLINE_MIN - 4:
                save_status()
                time.sleep(10)
                continue
            if ev is not None:
                killpg(ev[0], signal.SIGKILL)
            break
        save_status()
        time.sleep(10)
    metrics = run / "metrics.json"
    if metrics.exists():
        m = json.loads(metrics.read_text())
        STATE["result"] = {k: m.get(k) for k in ("stop_reason", "iterations_done", "last_iteration", "wall_min",
                                                   "seconds_per_iteration", "a100_units_estimate", "reward_rev")}  # fmt: skip
        exp = m.get("export") or {}
        STATE["result"]["export"] = {k: exp.get(k) for k in ("onnx", "sha256", "robotd_problems")}
        STATE["result"]["export_parity"] = (exp.get("parity") or {}).get("max_abs")


def final_battery(run: Path, stage: str, n: int) -> None:
    onnx = run / "final" / "velstand_duck.onnx"
    if not onnx.exists() or elapsed_min() > DEADLINE_MIN - 5:
        return
    STATE["step"] = "final battery"
    save_status()
    out = run / "final" / "battery_quick.json"
    proc = spawn(
        battery_args(str(onnx), out, "quick", n, "j3-6", stage), run / "final" / "battery_quick.log"
    )
    while proc.poll() is None and elapsed_min() < DEADLINE_MIN - 2 and not ABORT.exists():
        time.sleep(10)
        save_status()
    if proc.poll() is None:
        killpg(proc, signal.SIGKILL)
    if out.exists():
        STATE["final_battery"] = gates_brief(out)


# ── jobs ─────────────────────────────────────────────────────────────────────────────────────


def job_train(budget_cap: float | None = None) -> None:
    run = OUT / f"{STAGE}-{TAG}"
    iters = int(env("iters", "1500"))
    envs = int(env("envs", "4096"))
    budget = float(env("budget_min", str(max(10.0, DEADLINE_MIN - 25))))
    if budget_cap is not None:
        budget = min(budget, budget_cap)
    init = env("init", "v5")
    extra = env("extra").split()
    trainer = spawn(trainer_args(STAGE, run, iters, envs, budget, init, extra), run / "stdout.log")
    STATE["trainer"] = {
        "pid": trainer.pid,
        "iters": iters,
        "envs": envs,
        "budget_min": budget,
        "init": init,
        "extra": extra,
    }
    say(
        f"trainer started (pid {trainer.pid}): {STAGE} {iters} it at {envs} envs, budget {budget} min, init {init}"
    )
    save_status()
    watch_training(trainer, run, STAGE, int(env("eval_n", "48")))
    final_battery(run, STAGE, int(env("eval_n", "48")))
    STATE["run"] = str(run)


def v5ctl_battery(n_ctl: int, ctl_delays: str, cdir: Path) -> dict:
    """The v5 control battery (the Stage-2 programs + the regression conditions the relative gates
    compare with, on v5, one reference per servo delay) → ``cdir/v5ctl.json`` and the baselines."""
    STATE["step"] = "v5 control battery"
    save_status()
    cdir.mkdir(parents=True, exist_ok=True)
    ctl = cdir / "v5ctl.json"
    t = time.time()
    proc = spawn(battery_args("", ctl, "v5ctl", n_ctl, ctl_delays, "pose"), cdir / "v5ctl.log")
    while proc.poll() is None:
        if ABORT.exists() or elapsed_min() > DEADLINE_MIN - 2:
            killpg(proc, signal.SIGKILL)
            break
        time.sleep(15)
        save_status()
    info = {
        "rc": proc.returncode,
        "wall_s": round(time.time() - t, 1),
        "file": str(ctl),
        "n": n_ctl,
        "delays": ctl_delays,
    }
    if ctl.exists():
        bdir = baselines_dir()
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "v5ctl.json").write_text(ctl.read_text())
        info["complete"] = json.loads(ctl.read_text()).get("complete")
    say(
        f"v5 control battery rc {proc.returncode} in {info['wall_s']} s ({ctl_delays}, n = {n_ctl})"
    )
    STATE["v5ctl"] = info
    return info


def job_battery(walk: str | None = None, v5ctl_first: bool | None = None) -> None:
    walk = walk if walk is not None else env("walk")
    mode = env("battery_mode", "full")
    n = int(env("battery_n", "512"))
    delays = env("delays", "j3-6,d0,d20,d40")
    out = OUT / f"battery-{TAG}.json"
    if v5ctl_first is None:
        v5ctl_first = env("v5ctl_first", "0") == "1"
    if v5ctl_first and not (baselines_dir() / "v5ctl.json").exists():
        # s2:train: the per-delay v5 references run here, in the first V2 session, not in S0 (S0 ≤ 5 units),
        # at the candidate battery's own delays (S2_CTL_DELAYS may be "none": that only skipped it in S0)
        v5ctl_battery(
            int(env("ctl_n", "512" if not DRY else "1")),
            env("v2_ctl_delays", delays),
            OUT / "calib",
        )
        if ABORT.exists() or elapsed_min() > DEADLINE_MIN - 5:
            STATE["aborted"] = "ABORT or deadline after the v5 control battery"
            return
    STATE["step"] = f"battery {mode}"
    save_status()
    args = battery_args(walk if mode != "v5ctl" else "", out, mode, n, delays, STAGE)
    proc = spawn(args, OUT / f"battery-{TAG}.log")
    while proc.poll() is None:
        if ABORT.exists() or elapsed_min() > DEADLINE_MIN - 2:
            killpg(proc, signal.SIGKILL)
            STATE["aborted"] = "ABORT or deadline"
            break
        time.sleep(15)
        save_status()
    STATE["battery_rc"] = proc.returncode
    if out.exists():
        STATE["battery"] = gates_brief(out)


def job_calibrate() -> None:
    """S0-2: parity, the v5 control battery, s/it, the warm-start sanity run, the export, the
    quick battery's time → calibration.json."""
    cal: dict[str, Any] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gpu": gpu()}
    cdir = OUT / "calib"
    cdir.mkdir(parents=True, exist_ok=True)
    STATE["step"] = "calibrate: parity"
    save_status()
    script = cdir / "parity.py"
    script.write_text(
        "import json\n"
        "from duckmind.skilltrain import rl_onnx as O, s2_bc as B\n"
        f"p = '{policies_dir()}/velstand.onnx'\n"
        f"a = O.build_actor().to('{DEVICE}')\n"
        "info = B.import_v5(a, p)\n"
        "print(json.dumps({'parity': info['parity'], 'recondition': info['recondition']}))\n"
    )
    txt = sh(f"{PY} {script}", envx=child_env(), tail=4000).strip().splitlines()[-1]
    cal["import"] = json.loads(txt)
    say(f"import parity {cal['import']['parity']['max_abs']:.2e}")
    # the v5 control battery: the Stage-2 programs on v5 (kept as a baseline for every gate); every
    # servo delay the gates are judged at gets its own v5 reference (WORK-RULES §0: 0/20/40 ms).
    # S2_CTL_DELAYS=none skips it here (s2:train: it runs in the first V2 session, S2_V5CTL_FIRST=1)
    ctl_delays = env("ctl_delays", "j3-6,d0,d20,d40")
    if ctl_delays.strip().lower() in ("", "none", "0", "skip"):
        cal["v5ctl"] = {"skipped": "S2_CTL_DELAYS=none (runs with the first V2 battery)"}
        say("v5 control battery skipped here")
    else:
        cal["v5ctl"] = v5ctl_battery(int(env("ctl_n", "512" if not DRY else "1")), ctl_delays, cdir)
    # seconds per iteration
    iters = [int(x) for x in env("cal_iters", "40,30,30" if not DRY else "2,2,2").split(",")]
    sizes = [("pose", 4096, iters[0]), ("pose", 8192, iters[1]), ("gaze", 4096, iters[2])]
    if DRY:
        sizes = [("pose", 16, iters[0]), ("pose", 32, iters[1]), ("gaze", 16, iters[2])]
    cal["timing"] = {}
    for stage, envs, its in sizes:
        STATE["step"] = f"calibrate: timing {stage} {envs}"
        save_status()
        out = cdir / f"time-{stage}-{envs}"
        proc = spawn(trainer_args(stage, out, its, envs, 15.0, "v5", ["--no-export", "--critic-warmup", "5", "--no-kill-flat"]),
                     out / "stdout.log")  # fmt: skip
        wait_proc(proc)
        m = (
            json.loads((out / "metrics.json").read_text())
            if (out / "metrics.json").exists()
            else {}
        )
        cal["timing"][f"{stage}-{envs}"] = {"s_per_it": m.get("seconds_per_iteration"), "build_s": m.get("build_s"),
                                             "stop": m.get("stop_reason"), "rc": proc.returncode}  # fmt: skip
        say(f"timing {stage} @ {envs}: {m.get('seconds_per_iteration')}")
    # the warm-start sanity run (plan S0 item 2): R3 from v5, critic warm-up + actor iterations
    w, a = (int(x) for x in env("sanity_iters", "100,100" if not DRY else "1,2").split(","))
    STATE["step"] = "calibrate: warm-start sanity"
    save_status()
    run = cdir / "sanity-pose"
    envs = int(env("envs", "4096" if not DRY else "16"))
    sanity_extra = [
        "--critic-warmup",
        str(w),
        "--no-kill-flat",
        "--save-every",
        str(w + a),
        *env("sanity_extra").split(),
    ]
    trainer = spawn(
        trainer_args("pose", run, w + a, envs, 30.0, "v5", sanity_extra), run / "stdout.log"
    )
    wait_proc(trainer)
    m = json.loads((run / "metrics.json").read_text()) if (run / "metrics.json").exists() else {}
    cal["sanity"] = {"stop": m.get("stop_reason"), "s_per_it": m.get("seconds_per_iteration"),
                     "final": {k: v for k, v in (m.get("final_values") or {}).items()
                               if k.startswith(("Episode_Reward/pose", "Episode_Reward/track", "Episode_Metrics/"))},
                     "export": (m.get("export") or {}).get("parity")}  # fmt: skip
    onnx = run / "final" / "velstand_duck.onnx"
    if onnx.exists():
        STATE["step"] = "calibrate: quick battery"
        save_status()
        t = time.time()
        qb = cdir / "sanity-quick.json"
        proc = spawn(battery_args(str(onnx), qb, "quick", int(env("eval_n", "48" if not DRY else "1")), "j3-6", "pose"),
                     cdir / "sanity-quick.log")  # fmt: skip
        wait_proc(proc)
        cal["sanity"]["quick_battery"] = {
            "wall_s": round(time.time() - t, 1),
            **(gates_brief(qb) if qb.exists() else {}),
        }
    cal["recommendation"] = recommend(cal)
    cal["session_min"] = round(elapsed_min(), 1)
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=1, ensure_ascii=False) + "\n")
    STATE["calibration"] = cal["recommendation"]
    say("calibration: " + json.dumps(cal["recommendation"]))


def recommend(cal: dict) -> dict:
    """The sizes and the unit estimate for the rest of Stage 2 from the measured speeds."""
    t = cal.get("timing") or {}

    def med(key: str) -> float | None:
        v = (t.get(key) or {}).get("s_per_it") or {}
        return v.get("median") if isinstance(v, dict) else None

    s4, s8, g4 = med("pose-4096"), med("pose-8192"), med("gaze-4096")
    envs = 4096
    if s4 and s8 and s8 / s4 < 1.7:  # 8,192 envs give ≥ 1.18× env-steps per second
        envs = 8192
    sp = {
        "pose": (s8 if envs == 8192 else s4) or 4.0,
        "gaze": (g4 * (s8 / s4 if envs == 8192 and s4 and s8 else 1.0)) if g4 else 4.2,
    }
    iters = {"pose": 1500 if envs == 4096 else 900, "gaze": 1500 if envs == 4096 else 900}
    quick_s = ((cal.get("sanity") or {}).get("quick_battery") or {}).get("wall_s") or 120.0
    minutes = {k: iters[k] * sp[k] / 60.0 for k in iters}
    one = {k: round(v / 60 * UNITS_PER_HOUR, 2) for k, v in minutes.items()}
    return {"envs": envs, "s_per_it": sp, "iterations": iters, "train_min": {k: round(v, 1) for k, v in minutes.items()},
            "one_attempt_units": one, "quick_battery_s": quick_s,
            "note": "checkpoint batteries run beside the training (no extra wall time)"}  # fmt: skip


def go_no_go(cal: dict) -> dict:
    """The S0 go / no-go before a chained training (RUNBOOK §1, plan §5 S0), automatic so the A100
    never idles while the Mac decides (s2:train, 2026-09-28). ``go`` only when: the v5 import's GPU
    parity ≤ 1e-4; the sanity run ended by its iterations (no error, no sign stop); its export parity
    ≤ 1e-5 with no robotd problems; its quick battery (n = 48, the recipe's delay) FAILs none of the
    walking / stop-go / righting / sit-stand safety gates and keeps the walking speed within ±10 % of
    v5; the share of fallen time over the last 20 actor iterations ≤ 1.5 × the critic-only
    iterations' (v5's own actor) + 0.02."""
    why: list[str] = []
    imp = ((cal.get("import") or {}).get("parity") or {}).get("max_abs")
    if imp is None or imp > 1e-4:
        why.append(f"import parity {imp}")
    s = cal.get("sanity") or {}
    stop = str(s.get("stop"))
    if s.get("stop") is None or stop.startswith(("error", "sign", "flat")):
        why.append(f"sanity stop: {stop}")
    ex = (s.get("export") or {}).get("max_abs") if isinstance(s.get("export"), dict) else None
    if ex is None or ex > 1e-5:
        why.append(f"export parity {ex}")
    g = ((s.get("quick_battery") or {}).get("gates") or {}).get("j3-6") or {}
    if not g:
        why.append("no sanity quick battery")
    # n = 48 per condition cannot resolve the 0.1 % / 0.3 % gates: "clearly worse than v5" here is
    # ≥ 3 events (v5: 0.057 % walking falls → 0.16 expected in 288 programs; 0.15 % stop-go → 0.07 in 48)
    for key in ("walk_falls", "stopgo_near_falls"):
        v = (g.get(key) or [None])[0]
        if isinstance(v, (list, tuple)) and len(v) == 2 and int(v[0]) >= 3:
            why.append(f"sanity {key} {v[0]}/{v[1]}")
    v = g.get("push_righting_8s")
    if v and len(v) > 1 and v[1] == "FAIL":
        why.append(f"sanity push_righting_8s FAIL ({v[0]})")
    sp = (g.get("walk_speed_ratio") or [None])[0]
    if sp is not None and abs(float(sp) - 1.0) > 0.10:
        why.append(f"sanity walk speed ratio {sp}")
    fallen = {}
    prog = OUT / "calib" / "sanity-pose" / "progress.jsonl"
    if prog.exists():
        rows = []
        for ln in prog.read_text().splitlines():
            try:
                rows.append(json.loads(ln))
            except ValueError:
                continue
        key = "Episode_Metrics/fallen45"
        warm = [r[key] for r in rows if r.get("warmup") and key in r][-40:]
        last = [r[key] for r in rows if not r.get("warmup") and key in r][-20:]
        if warm and last:
            fallen = {
                "critic_only": round(sum(warm) / len(warm), 4),
                "last_actor": round(sum(last) / len(last), 4),
            }
            if fallen["last_actor"] > 1.5 * fallen["critic_only"] + 0.02:
                why.append(f"fallen share rose {fallen}")
    out = {"go": not why, "reasons": why, "fallen": fallen, "speed_ratio": sp}
    say("S0 go/no-go: " + json.dumps(out))
    return out


def job_chain() -> None:
    """S0 then R3 in one session (s2:train): the calibration; the go / no-go; if go, the pose
    training (``S2_STAGE``/``S2_TAG``) with the time left before the deadline (− 18 min for the export,
    the final quick battery and the results tar); with ``S2_THEN_BATTERY=1`` then the V2 battery of
    the export (the v5 control battery first when the bundle has none), ``S2_V2_RESERVE_MIN`` (60)
    kept for it — skipped after a STOP or an ABORT (a stopped run is not worth a V2)."""
    job_calibrate()
    cal = (
        json.loads((OUT / "calibration.json").read_text())
        if (OUT / "calibration.json").exists()
        else {}
    )
    decision = go_no_go(cal)
    STATE["go_no_go"] = decision
    cal["go_no_go"] = decision
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=1, ensure_ascii=False) + "\n")
    if not decision["go"]:
        STATE["no_go"] = decision["reasons"]
        return
    if ABORT.exists() or STOP.exists():
        STATE["aborted"] = "control file after the calibration"
        return
    train_then_v2()


def train_then_v2() -> None:
    """The training with the time left (− 18 min for the export, the final quick battery and the
    results tar, − ``S2_V2_RESERVE_MIN`` with ``S2_THEN_BATTERY=1``), then that V2 battery of the
    export — skipped after a STOP or an ABORT (a stopped run is not worth a V2)."""
    then_battery = env("then_battery", "0") == "1"
    reserve = float(env("v2_reserve_min", "60")) if then_battery else 0.0
    left = DEADLINE_MIN - elapsed_min() - 18.0 - reserve
    if left < 20:
        STATE["no_go"] = [f"only {left:.0f} min left for the training"]
        say(STATE["no_go"][0])
        return
    job_train(budget_cap=left)
    onnx = OUT / f"{STAGE}-{TAG}" / "final" / WALK_ONNX
    if then_battery:
        if STOP.exists() or ABORT.exists() or not onnx.exists():
            STATE["v2"] = (
                f"skipped: stop {STOP.exists()} abort {ABORT.exists()} export {onnx.exists()}"
            )
            say("V2 battery " + STATE["v2"])
            return
        say(
            f"V2 battery of {onnx.name} ({DEADLINE_MIN - elapsed_min():.0f} min before the deadline)"
        )
        job_battery(walk=str(onnx), v5ctl_first=True)


def collect() -> Path:
    STATE["step"] = "collect"
    save_status()
    tar = OUT / f"results-{JOB}-{STAGE}-{TAG}.tar.gz"
    skip_suffix = (".pt",) if env("with_checkpoints", "latest") != "all" else ()
    with tarfile.open(tar, "w:gz") as t:
        for p in sorted(OUT.rglob("*")):
            if not p.is_file() or p.name.startswith("results-"):
                continue
            if p.suffix in skip_suffix and p.name != "latest.pt" and "_final" not in p.name:
                continue
            t.add(p, arcname=str(p.relative_to(OUT)))
    STATE["tar"] = str(tar)
    STATE["tar_mb"] = round(tar.stat().st_size / 1e6, 2)
    say(f"packed {tar} ({STATE['tar_mb']} MB)")
    return tar


def main() -> int:
    mode = (
        sys.argv[1]
        if len(sys.argv) > 1 and sys.argv[1] in ("job", "status", "collect", "setup")
        else "job"
    )
    if mode == "status":
        print(STATUS.read_text() if STATUS.exists() else "{}")
        return 0
    if mode == "collect":
        collect()
        return 0
    if mode == "setup":
        setup()
        return 0
    say(f"start: job {JOB}, stage {STAGE}, tag {TAG}, deadline {DEADLINE_MIN} min, dry {DRY}")
    save_status()
    try:
        setup()
        if ABORT.exists() or STOP.exists():
            STATE["aborted"] = "control file before the job"
        elif JOB == "train":
            train_then_v2()
        elif JOB == "battery":
            job_battery()
        elif JOB == "calibrate":
            job_calibrate()
        elif JOB == "chain":
            job_chain()
        else:
            raise SystemExit(f"unknown job {JOB!r}")
    except BaseException as exc:  # noqa: BLE001
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        say("ERROR " + STATE["error"])
    finally:
        try:
            collect()
        except Exception as exc:  # noqa: BLE001
            STATE["tar_error"] = str(exc)
        STATE["done"] = True
        STATE["step"] = None
        save_status()
        say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
