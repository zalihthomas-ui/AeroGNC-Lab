"""Public-safe deployable drag recovery and vertical-descent verification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerognc.mathematics.integrators import EventOccurrence, EventSpec, integrate_fixed_step


@dataclass(frozen=True, slots=True)
class RecoveryDevice:
    """Two-stage reefed-to-full drag-area schedule."""

    trigger_time_s: float
    deployment_delay_s: float
    reefing_time_s: float
    reefed_hold_time_s: float
    inflation_time_s: float
    reefed_area_m2: float
    full_area_m2: float
    drag_coefficient: float

    def __post_init__(self) -> None:
        nonnegative = np.array(
            [self.trigger_time_s, self.deployment_delay_s, self.reefed_hold_time_s]
        )
        positive = np.array(
            [
                self.reefing_time_s,
                self.inflation_time_s,
                self.reefed_area_m2,
                self.full_area_m2,
                self.drag_coefficient,
            ]
        )
        if not np.all(np.isfinite(nonnegative)) or np.any(nonnegative < 0.0):
            raise ValueError("recovery trigger, delay, and hold times must be non-negative")
        if not np.all(np.isfinite(positive)) or np.any(positive <= 0.0):
            raise ValueError("recovery times, areas, and drag coefficient must be positive")
        if self.full_area_m2 < self.reefed_area_m2:
            raise ValueError("full_area_m2 cannot be smaller than reefed_area_m2")

    @property
    def deployment_time_s(self) -> float:
        """Start of reefed inflation [s]."""
        return self.trigger_time_s + self.deployment_delay_s

    @property
    def reefed_time_s(self) -> float:
        """Time at which reefed area is reached [s]."""
        return self.deployment_time_s + self.reefing_time_s

    @property
    def full_inflation_start_time_s(self) -> float:
        """Start of the reefed-to-full ramp [s]."""
        return self.reefed_time_s + self.reefed_hold_time_s

    @property
    def fully_inflated_time_s(self) -> float:
        """Time at which full drag area is reached [s]."""
        return self.full_inflation_start_time_s + self.inflation_time_s

    def drag_area_m2(self, time_s: float) -> float:
        """Return continuous projected recovery area [m^2]."""
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if time_s <= self.deployment_time_s:
            return 0.0
        if time_s < self.reefed_time_s:
            fraction = (time_s - self.deployment_time_s) / self.reefing_time_s
            return float(fraction * self.reefed_area_m2)
        if time_s <= self.full_inflation_start_time_s:
            return self.reefed_area_m2
        if time_s < self.fully_inflated_time_s:
            fraction = (time_s - self.full_inflation_start_time_s) / self.inflation_time_s
            return float(self.reefed_area_m2 + fraction * (self.full_area_m2 - self.reefed_area_m2))
        return self.full_area_m2

    def drag_force_down_n(
        self, time_s: float, air_velocity_down_mps: float, density_kgpm3: float
    ) -> float:
        """Return signed drag force along positive-down vertical axis [N]."""
        values = np.array([time_s, air_velocity_down_mps, density_kgpm3])
        if not np.all(np.isfinite(values)) or density_kgpm3 < 0.0:
            raise ValueError("recovery load inputs must be finite and density non-negative")
        magnitude_n = (
            0.5
            * density_kgpm3
            * air_velocity_down_mps**2
            * self.drag_coefficient
            * self.drag_area_m2(time_s)
        )
        return float(-np.sign(air_velocity_down_mps) * magnitude_n)


@dataclass(frozen=True, slots=True)
class RecoveryEvent:
    """One deployment-state transition."""

    name: str
    time_s: float


@dataclass(frozen=True, slots=True)
class RecoveryDescentResult:
    """Vertical descent history and opening/touchdown evidence."""

    time_s: np.ndarray
    altitude_m: np.ndarray
    velocity_down_mps: np.ndarray
    drag_area_m2: np.ndarray
    opening_load_n: np.ndarray
    events: tuple[RecoveryEvent, ...]
    ground_contact: EventOccurrence

    @property
    def maximum_opening_load_n(self) -> float:
        """Maximum recovery drag magnitude [N]."""
        return float(np.max(self.opening_load_n))

    @property
    def touchdown_speed_mps(self) -> float:
        """Absolute vertical speed at ground contact [m/s]."""
        return abs(float(self.velocity_down_mps[-1]))


def simulate_vertical_recovery(
    device: RecoveryDevice,
    *,
    initial_altitude_m: float,
    initial_velocity_down_mps: float,
    mass_kg: float,
    density_kgpm3: float = 1.225,
    gravity_mps2: float = 9.80665,
    step_s: float = 0.01,
    maximum_time_s: float = 600.0,
) -> RecoveryDescentResult:
    """Integrate a transparent constant-density vertical recovery benchmark."""
    values = np.array(
        [
            initial_altitude_m,
            initial_velocity_down_mps,
            mass_kg,
            density_kgpm3,
            gravity_mps2,
            step_s,
            maximum_time_s,
        ]
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("vertical recovery inputs must be finite")
    if initial_altitude_m <= 0.0 or mass_kg <= 0.0 or density_kgpm3 < 0.0:
        raise ValueError("altitude and mass must be positive; density must be non-negative")
    if gravity_mps2 <= 0.0 or step_s <= 0.0 or maximum_time_s <= 0.0:
        raise ValueError("gravity, step, and maximum time must be positive")

    def derivative(time_s: float, state: np.ndarray) -> np.ndarray:
        _altitude_m, velocity_down_mps = state
        drag_down_n = device.drag_force_down_n(time_s, float(velocity_down_mps), density_kgpm3)
        return np.array([-velocity_down_mps, gravity_mps2 + drag_down_n / mass_kg])

    integration = integrate_fixed_step(
        derivative,
        [initial_altitude_m, initial_velocity_down_mps],
        (0.0, maximum_time_s),
        step_s,
        events=(
            EventSpec(
                "ground_contact",
                lambda _time_s, state: float(state[0]),
                direction=-1,
                terminal=True,
            ),
        ),
    )
    if not integration.events:
        raise RuntimeError("vertical recovery did not reach ground before maximum_time_s")
    drag_area = np.array([device.drag_area_m2(float(value)) for value in integration.time_s])
    opening_load = np.array(
        [
            abs(device.drag_force_down_n(float(time_s), float(velocity_down_mps), density_kgpm3))
            for time_s, velocity_down_mps in zip(
                integration.time_s, integration.state[:, 1], strict=True
            )
        ]
    )
    event_candidates = (
        RecoveryEvent("deployment_start", device.deployment_time_s),
        RecoveryEvent("reefed", device.reefed_time_s),
        RecoveryEvent("full_inflation_start", device.full_inflation_start_time_s),
        RecoveryEvent("fully_inflated", device.fully_inflated_time_s),
    )
    return RecoveryDescentResult(
        time_s=integration.time_s,
        altitude_m=integration.state[:, 0],
        velocity_down_mps=integration.state[:, 1],
        drag_area_m2=drag_area,
        opening_load_n=opening_load,
        events=tuple(event for event in event_candidates if event.time_s <= integration.time_s[-1]),
        ground_contact=integration.events[0],
    )
