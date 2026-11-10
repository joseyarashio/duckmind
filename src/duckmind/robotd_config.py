"""What a real Microduck's robotd runs with, read from its two-layer ``robotd.toml`` (read only).

Since daemon 0.16 (pollen-robotics/microduck #483) robotd's configuration is two files:
``/etc/robot/robotd.toml`` is the base the image (or installer) wrote, replaced by every OS update
on an A/B robot, and ``/var/lib/robot/robotd.toml`` holds only the keys changed on this robot and
survives OS updates. ``robotd_params::merge_layers`` merges them for every reader: tables key by
key, anything else (lists included) replaced whole; the sections renamed in 2026-09 are normalised
first. This module does the same merge and reads the few keys Duck Mind cares about:

* ``[board] version`` — which board the robot is built on ("zero3": the Radxa Zero 3W and its IMX219
  camera turned a quarter; "beta" / "face-board-v1": Pollen's production main board, its GC2093
  camera upright). Unset is "zero3", as robotd reads it.
* ``[idle] enabled`` (0.16.1+, on by default) — robotd's own look-around and yawn: after 2 s with no
  motion and no ``robot.head`` it moves the head and opens the beak itself (a yawn can drop what the
  beak holds). Duck Mind's simulator runs it off (``livebody.ROBOTD_UPSTREAM_CHOICES``); on a real
  duck the owner layer should say ``enabled = false`` (the brain also re-sends ``robot.head`` at
  least every 0.5 s, which keeps it from starting).
* ``[pickup] enabled`` (0.15.2+, on by default) — robotd pauses the policy while it thinks a hand
  holds the duck (``policy`` reads ``held``, ``safety.picked_up`` is true). The simulator runs it
  off; with it on, the brain takes robotd's ``safety.picked_up`` as a lift
  (:mod:`duckmind.duckbody.proprio`), so either works.
* ``[safety] limp_fall`` (0.16.2+, on by default) — robotd goes limp in a fall, then poses the duck
  back; Duck Mind's righting was measured with it on and off (no change kept), so the board's choice
  stands.

Nothing here writes either file: the owner changes keys with ``sudo robotctl configure`` (which
validates against the base before it writes the user layer) and restarts robotd.
https://github.com/pollen-robotics/microduck/blob/daemon-v0.18.5/docs/design/architecture.md
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_PATH = Path("/etc/robot/robotd.toml")
USER_PATH = Path("/var/lib/robot/robotd.toml")
# robotd_params::registry::RENAMED_SECTIONS (old name → new name)
RENAMED_SECTIONS = (("imu_head", "pad_imu_head_control"), ("detect", "duck_detector"))
MAX_LAYER_BYTES = 512 * 1024  # the base with every comment is ~60 kB
# What robotd takes a robot's board to be when its robotd.toml names none (robotd's BoardParams).
# This is robotd's reading of a real robot's file, not the simulator's choice of camera: a
# simulator started without ``--board`` carries camera_model.DEFAULT_SIM_BOARD ("beta").
ROBOTD_UNSET_BOARD = "zero3"


def merge_layers(base: Mapping[str, Any], user: Mapping[str, Any]) -> dict[str, Any]:
    """``robotd_params::merge_layers``: the user layer over the base, tables key by key, anything
    else (lists too) replaced whole; renamed sections normalised in both layers first."""
    out = _normalised(base)
    _merge_into(out, _normalised(user))
    return out


def _normalised(table: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: (_copy(v)) for k, v in table.items()}
    for old, new in RENAMED_SECTIONS:
        if new not in out and old in out:
            out[new] = out.pop(old)
    return out


def _copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _copy(v) for k, v in value.items()}
    return value


def _merge_into(base: dict[str, Any], user: Mapping[str, Any]) -> None:
    for key, value in user.items():
        below = base.get(key)
        if isinstance(below, dict) and isinstance(value, Mapping):
            _merge_into(below, value)
        else:
            base[key] = _copy(value)


def read_layer(path: Path) -> dict[str, Any] | None:
    """One layer as a table; None when the file is absent, unreadable, oversized or not TOML (robotd
    sets a broken user layer aside and runs on the base: so does this)."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_LAYER_BYTES:
            return None
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return None


@dataclass(frozen=True)
class RobotdConfig:
    """The few keys of a board's merged ``robotd.toml`` Duck Mind reads; ``base_found`` /
    ``user_found``: whether each layer was there and readable."""

    board: str
    idle: bool
    pickup: bool
    limp_fall: bool
    base_found: bool
    user_found: bool

    @classmethod
    def from_table(
        cls, table: Mapping[str, Any], *, base_found: bool = True, user_found: bool = False
    ) -> RobotdConfig:
        def flag(section: str, key: str, default: bool) -> bool:
            value = (table.get(section) or {}).get(key, default)
            return value if isinstance(value, bool) else default

        board = (table.get("board") or {}).get("version", ROBOTD_UNSET_BOARD)
        pickup = flag("pickup", "enabled", True)
        if (table.get("pickup") or {}).get("model") == "none":
            pickup = False  # robotd: the literal "none" disables the classifier
        return cls(
            board=board if isinstance(board, str) else ROBOTD_UNSET_BOARD,
            idle=flag("idle", "enabled", True),
            pickup=pickup,
            limp_fall=flag("safety", "limp_fall", True),
            base_found=base_found,
            user_found=user_found,
        )

    def findings(self) -> list[str]:
        """What the owner should change for Duck Mind, plain English; empty when nothing."""
        out = []
        if self.idle:
            out.append(
                "[idle] enabled is on: robotd's own look-around and yawn take the head and open "
                "the beak when the brain is quiet (turn it off: sudo robotctl configure)"
            )
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "board": self.board,
            "idle": self.idle,
            "pickup": self.pickup,
            "limp_fall": self.limp_fall,
            "layers": {"base": self.base_found, "user": self.user_found},
        }


def board_config(base: Path = BASE_PATH, user: Path = USER_PATH) -> RobotdConfig | None:
    """The board's merged configuration (robotd's defaults for unset keys); None when the base layer
    is not there (not a provisioned Microduck, or not readable by this user)."""
    below = read_layer(base)
    if below is None:
        return None
    above = read_layer(user)
    merged = merge_layers(below, above or {})
    return RobotdConfig.from_table(merged, base_found=True, user_found=above is not None)


__all__ = [
    "BASE_PATH",
    "ROBOTD_UNSET_BOARD",
    "USER_PATH",
    "RobotdConfig",
    "board_config",
    "merge_layers",
    "read_layer",
]
