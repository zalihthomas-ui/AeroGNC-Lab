"""Navigation state providers: perfect-truth and seeded noisy estimation.

The GNC loop reads its state through a :class:`NavigationProvider` so the
controller never touches simulator truth directly in estimated mode. Two
providers are supplied here:

* :class:`PerfectStateProvider` — passes truth through unchanged (debugging,
  controller bring-up).
* :class:`NoisyStateProvider` — adds seeded, reproducible Gaussian sensor noise
  and can model a GPS dropout window (position/velocity marked invalid), which is
  enough to exercise the safety manager's navigation-validity logic.

Wiring the project's existing EKF / error-state INS filters
(`gnc/error_state_ekf.py`, `gnc/strapdown_ins.py`) as a full estimated backend is
tracked separately (TODO 7.3).
"""

from abc import ABC, abstractmethod

import numpy as np

from aerognc.navigation.state import NavigationState


class NavigationProvider(ABC):
    """Turns simulator/sensor truth into the state the controller consumes."""

    @abstractmethod
    def update(self, truth: NavigationState, dt_s: float) -> NavigationState:
        """Return the (possibly estimated) navigation state for this step."""

    def reset(self) -> None:  # noqa: B027 - intentional no-op default
        """Reset internal state (no-op for memoryless providers)."""


class PerfectStateProvider(NavigationProvider):
    """Ideal navigation: returns the truth state verbatim."""

    def update(self, truth: NavigationState, dt_s: float) -> NavigationState:
        return truth


class NoisyStateProvider(NavigationProvider):
    """Seeded Gaussian-noise estimator with an optional GPS dropout window."""

    def __init__(
        self,
        *,
        seed: int = 0,
        position_sigma_m: float = 1.5,
        velocity_sigma_mps: float = 0.3,
        airspeed_sigma_mps: float = 0.4,
        gps_dropout_window_s: tuple[float, float] | None = None,
    ) -> None:
        if position_sigma_m < 0.0 or velocity_sigma_mps < 0.0 or airspeed_sigma_mps < 0.0:
            raise ValueError("noise sigmas must be nonnegative")
        self._seed = seed
        self.position_sigma_m = position_sigma_m
        self.velocity_sigma_mps = velocity_sigma_mps
        self.airspeed_sigma_mps = airspeed_sigma_mps
        self.gps_dropout_window_s = gps_dropout_window_s
        self._rng = np.random.default_rng(seed)
        self._time_s = 0.0

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._time_s = 0.0

    def update(self, truth: NavigationState, dt_s: float) -> NavigationState:
        self._time_s += dt_s
        dropped = False
        if self.gps_dropout_window_s is not None:
            start_s, end_s = self.gps_dropout_window_s
            dropped = start_s <= self._time_s <= end_s
        position = truth.position_ned_m + self._rng.normal(0.0, self.position_sigma_m, size=3)
        velocity = truth.velocity_ned_mps + self._rng.normal(0.0, self.velocity_sigma_mps, size=3)
        airspeed = max(
            0.0, truth.airspeed_mps + float(self._rng.normal(0.0, self.airspeed_sigma_mps))
        )
        return NavigationState(
            position_ned_m=position,
            velocity_ned_mps=velocity,
            quaternion_nb=truth.quaternion_nb,
            angular_rate_body_radps=truth.angular_rate_body_radps,
            airspeed_mps=airspeed,
            valid=not dropped,
        )
