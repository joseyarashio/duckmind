"""Simulator-only validation oracle kept outside cognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True, slots=True)
class SimulatorTruth:
    monotonic_s: float
    trunk_xyz: tuple[float, float, float]
    trunk_z: float
    values: Mapping[str, object]


class SimulatorOracle(Protocol):
    """Read-only ground truth for test evaluators, never injected into Duck Self."""

    def snapshot(self) -> SimulatorTruth: ...


class FaultInjector(Protocol):
    """Test harness control plane, isolated from the production BodyPort."""

    def disconnect_body(self, duration_s: float) -> None: ...

    def perturb(self, name: str, value: float) -> None: ...
