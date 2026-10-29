"""Item ``style``: personality gaits (normal, sneak, drag, strut) in one network; the style code rides in obs[55:57].

Per style the reward asks for trunk height and lean (``qt_mdp.ex_trunk`` reads the style's targets), a forward-speed band, the swing
height at landing (``StyleClearance``) and the step frequency (``style_cadence``). ``normal`` is v5 itself (the walk anchor holds it).
A skill file for one style is the same network with its code folded into the first layer (:func:`bake_style`): robotd's skill block
carries no body slots.
"""

from __future__ import annotations

from typing import Any

import torch
from mjlab.managers.metrics_manager import MetricsTermCfg

from duckmind.skilltrain import qt_contract as C
from duckmind.skilltrain import qt_mdp as M

STYLE_PROBS = (0.25, 0.25, 0.25, 0.25)
TWIST_MIX = (
    (0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False),  # standing
    (
        0.82,
        C.STYLE_TWIST_VX[0],
        C.STYLE_TWIST_VX[1],
        0.0,
        0.0,
        0.0,
        0.4,
        True,
    ),  # walking, slight arcs
    (0.10, None, None, None, None, None, None, False),  # upstream's distribution
)


def _style_id(env: Any) -> torch.Tensor:
    return env.command_manager.get_term("body_pose").style_id


def _tab(env: Any, key: str, default: float) -> torch.Tensor:
    return torch.tensor(
        [
            C.STYLE_TARGETS[s][key] if C.STYLE_TARGETS[s][key] is not None else default
            for s in C.STYLES
        ],
        device=env.device,
    )


def _walking(env: Any) -> torch.Tensor:
    t = M._twist(env)
    return (t[:, :2].norm(dim=1) + t[:, 2].abs()) > 0.2


def style_speed(env: Any, soft: float = 0.04) -> torch.Tensor:
    """exp(-(distance outside the style's speed band)^2 / soft^2) on the forward speed while walking (not for normal)."""
    lo = _tab_band(env, 0)[_style_id(env)]
    hi = _tab_band(env, 1)[_style_id(env)]
    vx = torch.nan_to_num(M._robot(env).data.root_link_lin_vel_b[:, 0])
    out = (lo - vx).clamp(min=0.0) + (vx - hi).clamp(min=0.0)
    banded = (_style_id(env) != 0) & _walking(env) & (M._twist(env)[:, 0] > 0.25)
    return torch.exp(-((out / soft) ** 2)) * banded * M._upright(env, 40.0)


def _tab_band(env: Any, side: int) -> torch.Tensor:
    return torch.tensor(
        [
            C.STYLE_TARGETS[s]["speed"][side] if C.STYLE_TARGETS[s]["speed"] else 0.0
            for s in C.STYLES
        ],
        device=env.device,
    )


class StyleClearance:
    """Reward at landing: exp(-(peak foot height of the swing - the style's target)^2 / sigma^2) per foot."""

    def __init__(self, cfg: Any, env: Any) -> None:
        sensor = env.scene[cfg.params["height_sensor_name"]]
        self.peak = torch.zeros((env.num_envs, sensor.num_frames), device=env.device)
        self.step_dt = env.step_dt

    def __call__(
        self, env: Any, sensor_name: str, height_sensor_name: str, sigma: float = 0.006
    ) -> torch.Tensor:
        contact = env.scene[sensor_name]
        heights = env.scene[height_sensor_name].data.heights
        in_air = contact.data.found == 0
        self.peak = torch.where(in_air, torch.maximum(self.peak, heights), self.peak)
        first = contact.compute_first_contact(dt=self.step_dt)
        target = _tab(env, "clear", 0.02)[_style_id(env)].unsqueeze(1)
        r = torch.exp(-(((self.peak - target) / sigma) ** 2)) * first.float()
        self.peak = torch.where(first, torch.zeros_like(self.peak), self.peak)
        return r.sum(dim=1) * _walking(env) * M._upright(env, 40.0)


def style_cadence(env: Any, sensor_name: str, sigma: float = 0.3) -> torch.Tensor:
    """At each landing: exp(-(1/(air + stance) - the style's step frequency)^2 / sigma^2) (not for normal)."""
    contact = env.scene[sensor_name]
    first = contact.compute_first_contact(dt=env.step_dt)
    period = (contact.data.last_air_time + contact.data.last_contact_time).clamp(min=0.05)
    f_t = _tab(env, "freq", 0.0)[_style_id(env)].unsqueeze(1)
    r = torch.exp(-(((1.0 / period - f_t) / sigma) ** 2)) * first.float()
    on = (_style_id(env) != 0) & _walking(env)
    return r.sum(dim=1) * on * M._upright(env, 40.0)


def style_terms(rev: str) -> dict[str, tuple[Any, float, dict]]:
    return {
        "ex_trunk": (M.ex_trunk, 5.0, {"pitch_std": 0.045, "z_std": 0.007, "roll_std": 0.06}),
        "ex_upright": (M.ex_upright, 2.0, {}),
        "ex_posture": (M.posture, 1.0, {}),
        "style_speed": (style_speed, 4.0, {"soft": 0.04}),
        "style_clear": (
            StyleClearance,
            10.0,
            {
                "sensor_name": "feet_ground_contact",
                "height_sensor_name": "foot_height_scan",
                "sigma": 0.006,
            },
        ),
        "style_cadence": (
            style_cadence,
            10.0,
            {"sensor_name": "feet_ground_contact", "sigma": 0.3},
        ),
        "fall_event": (M.fall_event, -10.0, {"tilt_deg": 60.0}),
    }


def _mk(style: int, what: str):
    def fn(env: Any) -> torch.Tensor:
        on = (_style_id(env) == style) & _walking(env) & (M._twist(env)[:, 0] > 0.25)
        if what == "n":
            return on.float()
        if what == "vx":
            v = torch.nan_to_num(M._robot(env).data.root_link_lin_vel_b[:, 0])
        elif what == "z":
            v = M.S2M._trunk_z(env) * 1000.0
        elif what == "pitch":
            v = torch.rad2deg(M._rp(env)[1])
        else:
            raise ValueError(what)
        return torch.where(on, v, torch.zeros_like(v))

    fn.__name__ = f"m_style_{C.STYLES[style]}_{what}"
    return fn


def style_metrics() -> dict[str, MetricsTermCfg]:
    out = {}
    for k, name in enumerate(C.STYLES):
        for what in ("n", "vx", "z", "pitch"):
            out[f"sty_{name}_{what}"] = MetricsTermCfg(func=_mk(k, what))
    return out


def bake_style(onnx_in: str, onnx_out: str, style: str) -> dict:
    """Fold a style code (obs[55:57]) into the first layer of a robotd ONNX: the output file ignores slots 55 and 56 and behaves as
    if they carried the code. Exact: W' = W with the columns zeroed, b' = b + W[:, s] * (code - mean_s) / div_s."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(onnx_in)
    inits = {i.name: i for i in m.graph.initializer}
    mean = numpy_helper.to_array(inits["obs_normalizer._mean"]).reshape(-1)
    div_name = next(
        n
        for n, i in inits.items()
        if n.startswith("onnx::Div") or n.endswith("_std") or n.endswith("Div")
    )
    div = numpy_helper.to_array(inits[div_name]).reshape(-1)
    w = numpy_helper.to_array(inits["mlp.0.weight"]).copy()
    b = numpy_helper.to_array(inits["mlp.0.bias"]).copy()
    code = C.STYLE_CODE[style]
    for slot, c in zip((C.SLOT_X, C.SLOT_Y), code, strict=True):
        b = b + w[:, slot] * ((c - mean[slot]) / div[slot])
        w[:, slot] = 0.0
    inits["mlp.0.weight"].CopyFrom(numpy_helper.from_array(w, "mlp.0.weight"))
    inits["mlp.0.bias"].CopyFrom(numpy_helper.from_array(b, "mlp.0.bias"))
    onnx.save(m, onnx_out)
    return {"style": style, "div": div_name}
