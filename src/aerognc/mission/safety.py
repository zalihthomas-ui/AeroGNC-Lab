"""Safety manager: envelope, geofence, and integrity monitoring.

The safety manager is deliberately separate from guidance, control, and mission
logic. Each step it inspects the navigation state against configured limits and
returns a :class:`SafetyVerdict` carrying every triggered
:class:`SafetyEvent` (timestamp, trigger, value, threshold, response) plus the
single most-severe recommended :class:`~aerognc.mission.mission_manager.SafetyResponse`.
The mission manager, not the safety manager, actuates the response.

Response mapping (most severe wins):

* minimum-altitude breach (ground proximity) -> ABORT
* geofence breach / maximum-altitude breach / navigation invalid (e.g. GPS loss)
  / excessive cross-track -> RETURN_HOME
* bank / pitch / airspeed envelope exceedance -> LIMIT (advisory; controllers
  already bound the commands)

Non-finite (NaN/Inf) states cannot reach here: :class:`NavigationState` rejects
them at construction and the backend raises ``FloatingPointError`` on a non-finite
integration step, which the runner maps to an EMERGENCY/TERMINATE stop.
"""

from dataclasses import dataclass, field

import numpy as np

from aerognc.mission.mission_manager import SafetyResponse
from aerognc.navigation.state import NavigationState

_SEVERITY = {
    SafetyResponse.NONE: 0,
    SafetyResponse.LIMIT: 1,
    SafetyResponse.LOITER: 2,
    SafetyResponse.RETURN_HOME: 3,
    SafetyResponse.ABORT: 4,
    SafetyResponse.TERMINATE: 5,
}


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    """Configured safety envelope (SI units)."""

    min_airspeed_mps: float = 12.0
    max_airspeed_mps: float = 45.0
    max_bank_rad: float = float(np.deg2rad(60.0))
    max_pitch_rad: float = float(np.deg2rad(45.0))
    min_altitude_m: float = 5.0
    max_altitude_m: float = 3000.0
    geofence_radius_m: float = 5000.0
    max_cross_track_m: float = 500.0

    def __post_init__(self) -> None:
        values = [
            self.min_airspeed_mps,
            self.max_airspeed_mps,
            self.max_bank_rad,
            self.max_pitch_rad,
            self.max_altitude_m,
            self.geofence_radius_m,
            self.max_cross_track_m,
        ]
        if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0.0):
            raise ValueError("safety limits must be positive and finite")
        if self.max_airspeed_mps <= self.min_airspeed_mps:
            raise ValueError("max_airspeed_mps must exceed min_airspeed_mps")


@dataclass(frozen=True, slots=True)
class SafetyEvent:
    """A single triggered safety condition."""

    time_s: float
    trigger: str
    value: float
    threshold: float
    response: SafetyResponse


@dataclass(frozen=True, slots=True)
class SafetyVerdict:
    """Result of one safety evaluation."""

    response: SafetyResponse
    events: tuple[SafetyEvent, ...] = field(default_factory=tuple)

    @property
    def triggered(self) -> bool:
        return bool(self.events)


class SafetyManager:
    """Monitors the vehicle state and recommends a safety response."""

    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits or SafetyLimits()
        self._events: list[SafetyEvent] = []

    @property
    def events(self) -> tuple[SafetyEvent, ...]:
        """Return every safety event recorded so far."""
        return tuple(self._events)

    def reset(self) -> None:
        self._events.clear()

    def check(
        self,
        vehicle_state: NavigationState,
        time_s: float,
        *,
        cross_track_error_m: float | None = None,
    ) -> SafetyVerdict:
        """Evaluate the state and return the verdict (most-severe response)."""
        events: list[SafetyEvent] = []
        limits = self.limits
        altitude = vehicle_state.altitude_m
        if altitude < limits.min_altitude_m:
            events.append(
                SafetyEvent(
                    time_s, "min_altitude", altitude, limits.min_altitude_m, SafetyResponse.ABORT
                )
            )
        if altitude > limits.max_altitude_m:
            events.append(
                SafetyEvent(
                    time_s,
                    "max_altitude",
                    altitude,
                    limits.max_altitude_m,
                    SafetyResponse.RETURN_HOME,
                )
            )

        geofence_range_m = float(np.linalg.norm(vehicle_state.position_ned_m[:2]))
        if geofence_range_m > limits.geofence_radius_m:
            events.append(
                SafetyEvent(
                    time_s,
                    "geofence",
                    geofence_range_m,
                    limits.geofence_radius_m,
                    SafetyResponse.RETURN_HOME,
                )
            )

        if not vehicle_state.valid:
            events.append(
                SafetyEvent(time_s, "navigation_invalid", 0.0, 1.0, SafetyResponse.RETURN_HOME)
            )

        if cross_track_error_m is not None and abs(cross_track_error_m) > limits.max_cross_track_m:
            events.append(
                SafetyEvent(
                    time_s,
                    "cross_track",
                    abs(cross_track_error_m),
                    limits.max_cross_track_m,
                    SafetyResponse.RETURN_HOME,
                )
            )

        airspeed = vehicle_state.airspeed_mps
        if airspeed < limits.min_airspeed_mps:
            events.append(
                SafetyEvent(
                    time_s, "min_airspeed", airspeed, limits.min_airspeed_mps, SafetyResponse.LIMIT
                )
            )
        if airspeed > limits.max_airspeed_mps:
            events.append(
                SafetyEvent(
                    time_s, "max_airspeed", airspeed, limits.max_airspeed_mps, SafetyResponse.LIMIT
                )
            )
        if abs(vehicle_state.roll_rad) > limits.max_bank_rad:
            events.append(
                SafetyEvent(
                    time_s,
                    "bank",
                    abs(vehicle_state.roll_rad),
                    limits.max_bank_rad,
                    SafetyResponse.LIMIT,
                )
            )
        if abs(vehicle_state.pitch_rad) > limits.max_pitch_rad:
            events.append(
                SafetyEvent(
                    time_s,
                    "pitch",
                    abs(vehicle_state.pitch_rad),
                    limits.max_pitch_rad,
                    SafetyResponse.LIMIT,
                )
            )
        return self._finalize(events)

    def _finalize(self, events: list[SafetyEvent]) -> SafetyVerdict:
        self._events.extend(events)
        response = SafetyResponse.NONE
        for event in events:
            if _SEVERITY[event.response] > _SEVERITY[response]:
                response = event.response
        return SafetyVerdict(response=response, events=tuple(events))
