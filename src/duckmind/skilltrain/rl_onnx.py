"""ONNX in and out of the trainer (tr:tasks).

* **Import** (:func:`actor_state_dict_from_onnx`): a microduck policy ONNX — ``obs[1,61]`` →
  ``(obs − mean) / (std + eps)`` → 61-512-256-128-14 ELU MLP → ``actions[1,14]``, the layout
  every v5 file has (``mjlab.rl`` exports rsl_rl's ``MLPModel`` this way) — becomes an rsl_rl
  5.0.1 ``MLPModel`` actor state dict: the baked normaliser into ``obs_normalizer`` (``_std`` =
  the ``Div`` constant − eps, ``count`` huge so PPO's running update cannot move it: frozen at
  the ONNX statistics), the four linear layers into ``mlp``, and a fresh Gaussian ``std``. The
  critic is not in the ONNX: a warm start trains a fresh one (training plan §4). Only the v5
  ONNX files exist (no Pollen checkpoints), so this is the only warm-start source.
* **Parity** (:func:`parity`): max |torch actor − onnxruntime| over random and supplied
  observations; the import must reproduce the ONNX to ≤ 1e-4 (the A100 run reached 1.3e-5).
* **Export** (:func:`export_actor`): the trained actor to robotd's format — exactly v5's graph
  (input ``obs`` float32 [1, 61], one output ``actions`` [1, 14], normaliser baked, opset 18,
  mjlab's legacy exporter), with our metadata; :func:`robotd_problems` re-checks the file the way
  ``duck-control``'s loader does.
* :func:`torch_policy` — a batched torch function of an ONNX file (v5 velstand / sitstand for the
  entry-state recorder and the evaluator's hand-back).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.skilltrain import rl_contract as C

NORMALIZER_EPS = 1e-2  # rsl_rl EmpiricalNormalization default eps (baked into the Div constant)
FROZEN_COUNT = 10**12  # PPO's running update moves the statistics by batch/count ≈ 1e-7
LAYERS = ("mlp.0", "mlp.2", "mlp.4", "mlp.6")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_onnx(path: str | Path) -> dict[str, Any]:
    """The weights and shape facts of a microduck policy ONNX (numpy)."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    g = model.graph
    init = {i.name: numpy_helper.to_array(i).copy() for i in g.initializer}
    div = [k for k in init if k.startswith("onnx::Div") or k.endswith("_std_plus_eps")]
    if "obs_normalizer._mean" not in init or len(div) != 1:
        raise ValueError(f"{path}: no baked normaliser (mean + one Div constant)")
    for layer in LAYERS:
        if f"{layer}.weight" not in init:
            raise ValueError(f"{path}: missing {layer}.weight (not the 4-layer microduck MLP)")
    ops = [n.op_type for n in g.node]
    if ops.count("Elu") != 3:
        raise ValueError(f"{path}: expected 3 ELU activations, got {ops}")
    dims = [init[f"{layer}.weight"].shape for layer in LAYERS]
    return {
        "mean": init["obs_normalizer._mean"].reshape(1, -1).astype(np.float32),
        "div": init[div[0]].reshape(1, -1).astype(np.float32),
        "layers": {
            layer: (
                init[f"{layer}.weight"].astype(np.float32),
                init[f"{layer}.bias"].astype(np.float32),
            )
            for layer in LAYERS
        },
        "hidden_dims": tuple(d[0] for d in dims[:-1]),
        "obs_dim": int(dims[0][1]),
        "action_dim": int(dims[-1][0]),
        "inputs": [i.name for i in g.input],
        "outputs": [o.name for o in g.output],
        "opset": [(o.domain, o.version) for o in model.opset_import],
        "metadata": {p.key: p.value for p in model.metadata_props},
    }


def actor_state_dict_from_onnx(
    path: str | Path,
    *,
    init_std: float = 0.25,
    eps: float = NORMALIZER_EPS,
    frozen: bool = True,
) -> OrderedDict:
    """An rsl_rl ``MLPModel`` (obs_normalization=True, GaussianDistribution scalar std) actor
    state dict reproducing the ONNX exactly: (x − mean)/(std + eps) with std = Div − eps."""
    import torch

    w = read_onnx(path)
    std = w["div"] - eps
    if float(std.min()) < 0.0:
        raise ValueError(f"{path}: Div constant below eps={eps} (not an rsl_rl normaliser?)")
    sd: OrderedDict = OrderedDict()
    sd["obs_normalizer._mean"] = torch.from_numpy(w["mean"])
    sd["obs_normalizer._var"] = torch.from_numpy(std**2)
    sd["obs_normalizer._std"] = torch.from_numpy(std)
    sd["obs_normalizer.count"] = torch.tensor(FROZEN_COUNT if frozen else 0, dtype=torch.long)
    for layer in LAYERS:
        weight, bias = w["layers"][layer]
        sd[f"{layer}.weight"] = torch.from_numpy(weight)
        sd[f"{layer}.bias"] = torch.from_numpy(bias)
    sd["distribution.std_param"] = torch.full((w["action_dim"],), float(init_std))
    return sd


def build_actor(
    obs_dim: int = C.OBS_LEN,
    action_dim: int = C.ACTION_LEN,
    hidden=(512, 256, 128),
    init_std: float = 1.0,
):
    """A bare rsl_rl ``MLPModel`` actor with the microduck recipe's architecture (for tests and
    the importer's proof; training builds its own through the runner)."""
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    obs = TensorDict({"actor": torch.zeros(1, obs_dim)}, batch_size=[1])
    return MLPModel(
        obs,
        {"actor": ["actor"]},
        "actor",
        action_dim,
        hidden_dims=hidden,
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": init_std,
            "std_type": "scalar",
        },
    )


def load_into_actor(
    actor: Any, path: str | Path, *, init_std: float = 0.25, frozen: bool = True
) -> dict:
    """Load an ONNX into an existing rsl_rl actor (strict: every key, every shape)."""
    sd = actor_state_dict_from_onnx(path, init_std=init_std, frozen=frozen)
    own = actor.state_dict()
    missing = [k for k in own if k not in sd]
    extra = [k for k in sd if k not in own]
    if missing or extra:
        raise ValueError(f"actor/ONNX key mismatch: missing {missing}, extra {extra}")
    for k, v in sd.items():
        if tuple(own[k].shape) != tuple(v.shape):
            raise ValueError(f"{k}: actor {tuple(own[k].shape)} vs ONNX {tuple(v.shape)}")
    device = next(actor.parameters()).device
    actor.load_state_dict(
        {k: v.to(device=device, dtype=own[k].dtype) for k, v in sd.items()}, strict=True
    )
    if frozen:
        freeze_normalizer(actor)
    return {
        "source": str(path),
        "sha256": sha256(path),
        "init_std": init_std,
        "frozen_normalizer": frozen,
    }


def freeze_normalizer(actor: Any) -> None:
    """Stop rsl_rl's running update exactly (``until`` ≤ ``count``). ``until`` is not in the
    state dict, so a resumed warm-started run must call this again (the trainer does); the huge
    ``count`` alone would still let a nearly-constant slot drift over a full run."""
    norm = actor.obs_normalizer
    if hasattr(norm, "count"):
        norm.until = int(norm.count.item())


def actor_forward(actor: Any, obs: Any) -> Any:
    """Deterministic actor output for a flat (N, 61) observation tensor."""
    import torch
    from tensordict import TensorDict

    groups = list(actor.obs_groups)
    if len(groups) != 1:
        raise ValueError(f"expected one actor observation group, got {groups}")
    td = TensorDict({groups[0]: obs}, batch_size=[obs.shape[0]])
    with torch.no_grad():
        return actor(td)


def parity(
    path: str | Path, fn: Any, *, samples: int = 256, obs: np.ndarray | None = None, seed: int = 0
) -> dict:
    """max |fn(obs) − onnxruntime(obs)| over ``samples`` random observations (scaled like real
    ones: N(mean, 3·std) of the ONNX's own normaliser) plus any supplied ``obs`` rows."""
    import onnxruntime as ort
    import torch

    w = read_onnx(path)
    rng = np.random.default_rng(seed)
    rows = [
        (w["mean"] + 3.0 * w["div"] * rng.standard_normal((samples, w["obs_dim"]))).astype(
            np.float32
        ),
        rng.standard_normal((max(8, samples // 8), w["obs_dim"])).astype(np.float32),
    ]
    if obs is not None and len(obs):
        rows.append(np.asarray(obs, dtype=np.float32).reshape(-1, w["obs_dim"]))
    x = np.concatenate(rows, axis=0)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    batch_ok = sess.get_inputs()[0].shape[0] not in (1,)
    if batch_ok:
        expected = sess.run(None, {name: x})[0]
    else:
        expected = np.concatenate([sess.run(None, {name: x[i : i + 1]})[0] for i in range(len(x))])
    got = fn(torch.from_numpy(x))
    got = got.detach().cpu().numpy() if hasattr(got, "detach") else np.asarray(got)
    diff = np.abs(got - expected)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rows": int(len(x)),
        "output_scale": float(np.abs(expected).max()),
    }


def reference64(path: str | Path, obs: np.ndarray) -> np.ndarray:
    """The ONNX's network evaluated in float64 with numpy (normalise, ELU layers, linear head): what the float32
    evaluations (torch, onnxruntime) are close to, within their own accumulation-order noise (bug 6, 2026-10-09:
    ~1.2e-5 at an output scale of 17, on the Mac and on the Spark alike)."""
    w = read_onnx(path)
    h = (np.asarray(obs, dtype=np.float64) - w["mean"].astype(np.float64)) / w["div"].astype(
        np.float64
    )
    layers = list(w["layers"].values())
    for i, (wt, b) in enumerate(layers):
        h = h @ wt.astype(np.float64).T + b.astype(np.float64)
        if i < len(layers) - 1:
            h = np.where(h > 0, h, np.expm1(np.minimum(h, 0.0)))
    return h


def torch_policy(path: str | Path, device: str) -> Any:
    """A batched, no-grad torch function of the ONNX (the neuro-validation reconstruction)."""
    import torch

    w = read_onnx(path)
    mean = torch.from_numpy(w["mean"]).to(device)
    div = torch.from_numpy(w["div"]).to(device)
    layers = [
        (torch.from_numpy(wt).to(device), torch.from_numpy(b).to(device))
        for wt, b in w["layers"].values()
    ]

    @torch.no_grad()
    def policy(obs: Any) -> Any:
        x = (obs.to(device) - mean) / div
        for i, (wt, b) in enumerate(layers):
            x = x @ wt.T + b
            if i < len(layers) - 1:
                x = torch.nn.functional.elu(x)
        return x

    policy.sha256 = sha256(path)  # type: ignore[attr-defined]
    return policy


def export_actor(actor: Any, out_path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    """Write the actor as robotd loads a policy: v5's exact graph shape (mjlab's legacy exporter:
    ``dynamo=False``, opset 18, fixed batch 1, names ``obs`` / ``actions``), plus metadata."""
    import onnx
    import torch

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # rsl_rl's wrapper deep-copies only the normaliser and the MLP (the actor's distribution
    # holds non-leaf tensors from the last rollout and cannot be deep-copied)
    wrapper = actor.as_onnx(verbose=False)
    wrapper.to("cpu")
    wrapper.eval()
    torch.onnx.export(
        wrapper,
        wrapper.get_dummy_inputs(),
        str(out),
        export_params=True,
        opset_version=18,
        verbose=False,
        input_names=wrapper.input_names,
        output_names=wrapper.output_names,
        dynamic_axes={},
        dynamo=False,
    )
    if metadata:
        m = onnx.load(str(out))
        for k, v in metadata.items():
            entry = onnx.StringStringEntryProto()
            entry.key = str(k)
            entry.value = (
                v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)
            )
            m.metadata_props.append(entry)
        onnx.save(m, str(out))
    return out


def robotd_problems(path: str | Path) -> list[str]:
    """What duck-control's ``policy::open`` would refuse (a feed-forward policy): the input
    named ``obs`` float32 [1 or dynamic, 61], exactly one output float32 [1 or dynamic, 14],
    finite actions on a zero observation (robotd warms every network up on zeros)."""
    import onnxruntime as ort

    found = []
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return [f"onnxruntime cannot load it: {exc}"]
    ins, outs = sess.get_inputs(), sess.get_outputs()
    if len(ins) != 1 or len(outs) != 1:
        found.append(f"{len(ins)} inputs / {len(outs)} outputs: robotd wants obs → actions")
    names = [i.name for i in ins]
    if "obs" not in names:
        found.append(f"no input named 'obs' (got {names})")
    for io, width in (
        (ins[0] if ins else None, C.OBS_LEN),
        (outs[0] if outs else None, C.ACTION_LEN),
    ):
        if io is None:
            continue
        if io.type != "tensor(float)":
            found.append(f"{io.name}: type {io.type}, robotd wants float32")
        shape = list(io.shape)
        ok = (
            len(shape) == 2
            and shape[1] == width
            and (shape[0] in (1, None) or isinstance(shape[0], str))
        )
        if not ok:
            found.append(f"{io.name}: shape {shape}, robotd wants [1 or dynamic, {width}]")
    if not found:
        out = sess.run(None, {"obs": np.zeros((1, C.OBS_LEN), dtype=np.float32)})[0]
        if not np.all(np.isfinite(out)):
            found.append("non-finite actions on the zero observation (robotd's warm-up)")
    return found


def describe(path: str | Path) -> dict[str, Any]:
    w = read_onnx(path)
    return {
        "file": Path(path).name,
        "sha256": sha256(path),
        "bytes": Path(path).stat().st_size,
        "inputs": w["inputs"],
        "outputs": w["outputs"],
        "hidden_dims": list(w["hidden_dims"]),
        "obs_dim": w["obs_dim"],
        "action_dim": w["action_dim"],
        "opset": w["opset"],
        "normaliser_min_div": float(w["div"].min()),
        "problems": robotd_problems(path),
    }


def max_abs(a: Any, b: Any) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if len(np.asarray(a)) else 0.0


def isfinite_all(x: Any) -> bool:
    return bool(np.all(np.isfinite(np.asarray(x))))


def clean_float(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


__all__ = [
    "actor_forward",
    "freeze_normalizer",
    "actor_state_dict_from_onnx",
    "build_actor",
    "describe",
    "export_actor",
    "load_into_actor",
    "parity",
    "read_onnx",
    "robotd_problems",
    "sha256",
    "torch_policy",
]
