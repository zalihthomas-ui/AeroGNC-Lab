"""Generic deterministic staging for fictional civilian research vehicles."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np

from aerognc.vehicle.propulsion import ThrustCurve

StagePhase = Literal["attached", "burning", "spent", "separated"]
StageEventKind = Literal["ignition", "burnout", "separation"]


@dataclass(frozen=True, slots=True)
class StageDefinition:
    """One ordered stage with a local-time motor and jettisonable dry mass."""

    name: str
    dry_mass_kg: float
    propulsion: ThrustCurve
    ignition_time_s: float
    separation_time_s: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stage name cannot be empty")
        if not np.isfinite(self.dry_mass_kg) or self.dry_mass_kg <= 0.0:
            raise ValueError("stage dry_mass_kg must be positive and finite")
        if not np.isfinite(self.ignition_time_s) or self.ignition_time_s < 0.0:
            raise ValueError("stage ignition_time_s must be non-negative and finite")
        if self.separation_time_s is not None:
            if not np.isfinite(self.separation_time_s):
                raise ValueError("stage separation_time_s must be finite")
            if self.separation_time_s < self.burnout_time_s:
                raise ValueError("stage separation cannot occur before burnout")
        if abs(self.propulsion.thrust_n[0]) > 1.0e-12:
            raise ValueError("stage thrust curve must start at zero thrust")
        if abs(self.propulsion.thrust_n[-1]) > 1.0e-12:
            raise ValueError("stage thrust curve must end at zero thrust")

    @property
    def burnout_time_s(self) -> float:
        """Global burnout epoch [s]."""
        duration_s = self.propulsion.burnout_time_s - self.propulsion.ignition_time_s
        return self.ignition_time_s + duration_s

    def motor_time_s(self, time_s: float) -> float:
        """Convert global vehicle time to the motor table's local time."""
        return self.propulsion.ignition_time_s + time_s - self.ignition_time_s

    def phase_at_time(self, time_s: float) -> StagePhase:
        """Return the deterministic discrete stage phase."""
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if self.separation_time_s is not None and time_s >= self.separation_time_s:
            return "separated"
        if time_s < self.ignition_time_s:
            return "attached"
        if time_s < self.burnout_time_s:
            return "burning"
        return "spent"

    def propellant_remaining_kg(self, time_s: float) -> float:
        """Return attached stage propellant, including full pre-ignition loading."""
        if self.phase_at_time(time_s) == "separated":
            return 0.0
        if time_s <= self.ignition_time_s:
            return self.propulsion.propellant_mass_kg
        return self.propulsion.propellant_remaining_kg(self.motor_time_s(time_s))

    def mass_kg(self, time_s: float) -> float:
        """Return stage contribution to current vehicle mass [kg]."""
        if self.phase_at_time(time_s) == "separated":
            return 0.0
        return self.dry_mass_kg + self.propellant_remaining_kg(time_s)

    def thrust_n(self, time_s: float) -> float:
        """Return axial stage thrust [N] at global time."""
        if self.phase_at_time(time_s) in {"attached", "spent", "separated"}:
            return 0.0
        return self.propulsion.thrust_at_time_n(self.motor_time_s(time_s))


@dataclass(frozen=True, slots=True)
class StageEvent:
    """Named discrete staging event."""

    stage_name: str
    kind: StageEventKind
    time_s: float
    mass_after_kg: float


@dataclass(frozen=True, slots=True)
class StageContinuityReport:
    """Exact checks around thrust endpoints and allowed mass discontinuities."""

    thrust_endpoints_zero: bool
    maximum_separation_mass_residual_kg: float
    minimum_dry_mass_margin_kg: float

    @property
    def passed(self) -> bool:
        """Return true when every staging invariant holds."""
        return (
            self.thrust_endpoints_zero
            and self.maximum_separation_mass_residual_kg <= 1.0e-10
            and self.minimum_dry_mass_margin_kg >= -1.0e-10
        )


class MultistageVehicle:
    """Ordered stage mass/thrust accounting with an always-retained payload."""

    def __init__(self, *, payload_mass_kg: float, stages: tuple[StageDefinition, ...]) -> None:
        if not np.isfinite(payload_mass_kg) or payload_mass_kg <= 0.0:
            raise ValueError("payload_mass_kg must be positive and finite")
        if not stages:
            raise ValueError("at least one stage is required")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("stage names must be unique")
        for previous, current in pairwise(stages):
            if current.ignition_time_s < previous.burnout_time_s:
                raise ValueError("stage ignitions must be ordered after prior burnout")
            if (
                previous.separation_time_s is not None
                and current.ignition_time_s < previous.separation_time_s
            ):
                raise ValueError("next-stage ignition cannot precede prior-stage separation")
        self.payload_mass_kg = float(payload_mass_kg)
        self.stages = stages

    def mass_kg(self, time_s: float) -> float:
        """Return payload plus every currently attached stage [kg]."""
        return self.payload_mass_kg + sum(stage.mass_kg(time_s) for stage in self.stages)

    def retained_dry_mass_floor_kg(self, time_s: float) -> float:
        """Return payload plus attached stage dry mass [kg]."""
        return self.payload_mass_kg + sum(
            stage.dry_mass_kg for stage in self.stages if stage.phase_at_time(time_s) != "separated"
        )

    def thrust_n(self, time_s: float) -> float:
        """Return total axial thrust from all stages [N]."""
        return float(sum(stage.thrust_n(time_s) for stage in self.stages))

    def active_stage_name(self, time_s: float) -> str | None:
        """Return the unique burning stage name, if any."""
        burning = [stage.name for stage in self.stages if stage.phase_at_time(time_s) == "burning"]
        if len(burning) > 1:
            raise RuntimeError("multiple stages are burning despite ordered-stage validation")
        return burning[0] if burning else None

    def events(self) -> tuple[StageEvent, ...]:
        """Return ignition, burnout, and separation records in deterministic order."""
        priority: dict[StageEventKind, int] = {"burnout": 0, "separation": 1, "ignition": 2}
        raw: list[tuple[float, int, str, StageEventKind]] = []
        for stage in self.stages:
            raw.append((stage.ignition_time_s, priority["ignition"], stage.name, "ignition"))
            raw.append((stage.burnout_time_s, priority["burnout"], stage.name, "burnout"))
            if stage.separation_time_s is not None:
                raw.append(
                    (
                        stage.separation_time_s,
                        priority["separation"],
                        stage.name,
                        "separation",
                    )
                )
        raw.sort(key=lambda item: (item[0], item[1], item[2]))
        return tuple(
            StageEvent(stage_name, kind, time_s, self.mass_kg(time_s))
            for time_s, _priority, stage_name, kind in raw
        )

    def continuity_report(self) -> StageContinuityReport:
        """Evaluate exact endpoint and dry-floor accounting invariants."""
        thrust_zero = all(
            abs(stage.propulsion.thrust_n[0]) <= 1.0e-12
            and abs(stage.propulsion.thrust_n[-1]) <= 1.0e-12
            for stage in self.stages
        )
        separation_residuals: list[float] = []
        sample_times = {0.0}
        for stage in self.stages:
            sample_times.update((stage.ignition_time_s, stage.burnout_time_s))
            if stage.separation_time_s is not None:
                before = np.nextafter(stage.separation_time_s, -np.inf)
                actual_drop = self.mass_kg(before) - self.mass_kg(stage.separation_time_s)
                separation_residuals.append(abs(actual_drop - stage.dry_mass_kg))
                sample_times.update((before, stage.separation_time_s))
        margins = [
            self.mass_kg(time_s) - self.retained_dry_mass_floor_kg(time_s)
            for time_s in sample_times
        ]
        return StageContinuityReport(
            thrust_endpoints_zero=thrust_zero,
            maximum_separation_mass_residual_kg=max(separation_residuals, default=0.0),
            minimum_dry_mass_margin_kg=float(min(margins)),
        )
