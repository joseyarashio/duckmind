"""What the bridge on a real duck tells the brain about it, and how the brain uses it (real duck prep
2026-11-08): the walk and sit-stand networks it runs (by the sha256 the brain's measured walk and
seat tables are keyed by), its two-layer robotd.toml and its camera."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from duckmind.livebody import LiveBody
from duckmind.neuro import seatcal, walkcal
from duckmind.remotebody.__main__ import describe_body
from duckmind.remotebody.bridge import real_body_info
from duckmind.simweb import neuro_live

V8_WALK = "2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492"
V8_SEAT = "a6f102d9fd655782ba85bfffe18175c34621da719cffcdd673be7982a06fc254"
V9_WALK = "ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a"  # v9 = v7's velstand


def _duck(root: Path, *, user: str | None) -> tuple[Path, Path, Path]:
    """A provisioned duck's files: the official policy set and both robotd.toml layers."""
    policies = root / "opt/robot/policies/current"
    policies.mkdir(parents=True)
    manifest = {
        "model_api": 1, "obs_len": 61, "action_len": 14,
        "robot": {"model": "microduck", "control_hz": 50},
        "policies": [
            {"file": "velstand.onnx", "kind": "perpetual", "slot": "walk", "entry_pose": "standing"},
            {"file": "alpha_sitstand.onnx", "name": "sitstand", "kind": "scripted"},
        ],
    }  # fmt: skip
    (policies / "manifest.json").write_text(json.dumps(manifest))
    (policies / ".source").write_text("repo=pollen-robotics/microduck-policies\nversion=v8\n")
    (policies / "velstand.onnx").write_bytes(b"walk")
    (policies / "alpha_sitstand.onnx").write_bytes(b"seat")
    etc = root / "etc/robot"
    etc.mkdir(parents=True)
    (etc / "robotd.toml").write_text('[board]\nversion = "face-board-v1"\n')
    var = root / "var/lib/robot"
    var.mkdir(parents=True)
    if user is not None:
        (var / "robotd.toml").write_text(user)
    return policies / "velstand.onnx", etc / "robotd.toml", var / "robotd.toml"


def test_the_real_bridge_reports_the_networks_and_the_config_from_the_ducks_files(tmp_path) -> None:
    info = real_body_info(*_duck(tmp_path, user="[pickup]\nenabled = false\n"))
    assert info["backend"] == "real"
    assert info["policies"] == {
        "set": "v8",
        "walk_sha256": hashlib.sha256(b"walk").hexdigest(),
        "sitstand_sha256": hashlib.sha256(b"seat").hexdigest(),
    }
    assert info["robotd_config"] == {
        "board": "face-board-v1",
        "idle": True,
        "pickup": False,
        "limp_fall": True,
        "layers": {"base": True, "user": True},
    }


def test_check_says_whether_the_brain_measured_the_ducks_walk_and_what_to_change() -> None:
    config = {"board": "face-board-v1", "idle": True, "pickup": True, "limp_fall": True,
              "layers": {"base": True, "user": False}}  # fmt: skip
    v8 = describe_body(
        {
            "backend": "real",
            "robotd_config": config,
            "policies": {"set": "v8", "walk_sha256": V8_WALK, "sitstand_sha256": V8_SEAT},
        }  # fmt: skip
    )
    text = "\n".join(v8)
    assert "v8 velstand" in text and "measured in the simulator" in text
    assert "v8 alpha_sitstand" in text
    assert "[idle] enabled = false" in text
    assert "gc2093" in text and "--board face-board-v1" in text
    v9 = "\n".join(
        describe_body(
            {"policies": {"set": "v9", "walk_sha256": V9_WALK, "sitstand_sha256": V8_SEAT}}
        )
    )
    assert "v9 velstand" in v9 and "measured in the simulator" in v9 and "NOT measured" not in v9
    other = "\n".join(
        describe_body(
            {"policies": {"set": "v10", "walk_sha256": "0" * 64, "sitstand_sha256": V8_SEAT}}
        )
    )
    assert "NOT measured" in other
    quiet = describe_body({"robotd_config": {**config, "idle": False, "board": "zero3"}})
    assert not any("change on the duck" in line for line in quiet)
    assert any("imx219" in line for line in quiet)


def test_a_remote_body_is_calibrated_to_the_walk_its_bridge_reported_not_this_machines(
    tmp_path,
) -> None:
    # this machine's policy folder holds v5 (or nothing); the remote duck runs v8
    local = tmp_path / "local-policies"
    local.mkdir()
    remote = SimpleNamespace(
        _link=SimpleNamespace(
            info={"policies": {"walk_sha256": V8_WALK, "sitstand_sha256": V8_SEAT}}
        ),
        paths=SimpleNamespace(policies=local),
    )
    hashes = LiveBody.policy_hashes(remote)  # type: ignore[arg-type]
    assert hashes == {"walk_sha256": V8_WALK, "sitstand_sha256": V8_SEAT}
    body = SimpleNamespace(policy_hashes=lambda: hashes, collision="allcollisions_backlash")
    walk = neuro_live._walk_map(body, 20.0)
    assert walk is not None and walk.loaded.walk_sha256 == V8_WALK
    assert walkcal.load().map_for(V8_WALK, "allcollisions_backlash", 20.0) is not None
    seat = neuro_live._seat_proprio(body, neuro_live.ProprioConfig(), 20.0)
    assert seat is not None and "v8" in seat[1]
    assert seatcal.load().seat_for(V8_SEAT, "allcollisions_backlash", 20.0) is not None
    # a local body: the files in its own policy folder, as before
    alone = SimpleNamespace(_link=None, paths=SimpleNamespace(policies=local))
    assert LiveBody.policy_hashes(alone) == {"walk_sha256": None, "sitstand_sha256": None}  # type: ignore[arg-type]
