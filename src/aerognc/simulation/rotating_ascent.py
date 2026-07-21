"""Configured point-mass research ascent in a rotating oblate-body frame."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import numpy.typing as npt

from aerognc.configuration.rotating_flight_loader import RotatingAscentConfiguration
from aerognc.mathematics.coordinates import launch_direction_ned
from aerognc.mathematics.geodesy import (
    GeodeticPosition,
    dcm_ecef_to_ned,
    ecef_position_to_ned,
    ecef_to_geodetic,
    geodetic_to_ecef,
)
from aerognc.mathematics.integrators import EventOccurrence, EventSpec, integrate_fixed_step
from aerognc.mathematics.vectors import FloatArray, as_vector
from aerognc.simulation.logging import SimulationResult


@dataclass(frozen=True, slots=True)
class RotatingAscentDiagnostics:
    """Instantaneous rotating-frame loads and navigation quantities."""

    geodetic: GeodeticPosition
    position_ned_m: FloatArray
    velocity_ned_mps: FloatArray
    wind_ned_mps: FloatArray
    acceleration_ned_mps2: FloatArray
    airspeed_mps: float
    total_speed_mps: float
    mach: float
    dynamic_pressure_pa: float
    mass_kg: float
    thrust_n: float
    drag_n: float


class RotatingAscentModel:
    """Compose the existing fictional vehicle with rotating ECEF dynamics."""

    def __init__(self, configuration: RotatingAscentConfiguration) -> None:
        self.configuration = configuration
        self.base = configuration.base_configuration
        self.planet = configuration.planet
        self.site = configuration.launch_site
        self.site_ecef_m = geodetic_to_ecef(self.site.geodetic, self.planet.ellipsoid)
        self.dcm_ne_site = dcm_ecef_to_ned(
            self.site.geodetic.latitude_rad, self.site.geodetic.longitude_rad
        )
        self.launch_direction_ned = launch_direction_ned(
            self.site.elevation_rad, self.site.azimuth_rad
        )
        self.launch_direction_ecef = self.dcm_ne_site.T @ self.launch_direction_ned

    def initial_state(self) -> FloatArray:
        """Return stationary-on-ground ECEF state plus configured launch speed."""
        velocity_ecef_mps = self.base.launch.initial_speed_mps * self.launch_direction_ecef
        return np.concatenate((self.site_ecef_m, velocity_ecef_mps))

    def _loads(
        self, time_s: float, state: npt.ArrayLike
    ) -> tuple[RotatingAscentDiagnostics, FloatArray]:
        values = as_vector(state, 6, name="state")
        position_ecef_m = values[:3]
        velocity_ecef_mps = values[3:]
        geodetic = ecef_to_geodetic(position_ecef_m, self.planet.ellipsoid)
        dcm_ne = dcm_ecef_to_ned(geodetic.latitude_rad, geodetic.longitude_rad)
        velocity_ned_mps = dcm_ne @ velocity_ecef_mps
        position_ned_m = ecef_position_to_ned(
            position_ecef_m, self.site.geodetic, self.planet.ellipsoid
        )
        altitude_above_site_m = geodetic.altitude_m - self.site.geodetic.altitude_m
        atmosphere = self.base.environment.atmosphere.properties(altitude_above_site_m)
        wind_ned_mps = self.base.environment.wind.velocity_ned_mps(time_s, altitude_above_site_m)
        wind_ecef_mps = dcm_ne.T @ wind_ned_mps
        air_velocity_ecef_mps = velocity_ecef_mps - wind_ecef_mps
        airspeed_mps = float(np.linalg.norm(air_velocity_ecef_mps))
        total_speed_mps = float(np.linalg.norm(velocity_ecef_mps))
        mach = airspeed_mps / atmosphere.speed_of_sound_mps
        dynamic_pressure_pa = 0.5 * atmosphere.density_kgpm3 * airspeed_mps**2
        drag_n = self.base.vehicle.aerodynamics.drag_force_n(dynamic_pressure_pa, mach)
        mass_kg = self.base.vehicle.mass_properties.at_time(time_s).mass_kg
        thrust_n = self.base.vehicle.propulsion.thrust_at_time_n(time_s)
        if self.base.launch.thrust_alignment == "velocity" and total_speed_mps > 1.0e-9:
            thrust_direction_ecef = velocity_ecef_mps / total_speed_mps
        else:
            thrust_direction_ecef = self.launch_direction_ecef
        drag_force_ecef_n = (
            np.zeros(3)
            if airspeed_mps <= 1.0e-9
            else -drag_n * air_velocity_ecef_mps / airspeed_mps
        )
        specific_force_ecef_mps2 = (thrust_n * thrust_direction_ecef + drag_force_ecef_n) / mass_kg
        acceleration_ecef_mps2 = (
            specific_force_ecef_mps2
            + self.planet.apparent_acceleration_ecef_mps2(position_ecef_m, velocity_ecef_mps)
        )
        diagnostics = RotatingAscentDiagnostics(
            geodetic=geodetic,
            position_ned_m=position_ned_m,
            velocity_ned_mps=velocity_ned_mps,
            wind_ned_mps=wind_ned_mps,
            acceleration_ned_mps2=dcm_ne @ acceleration_ecef_mps2,
            airspeed_mps=airspeed_mps,
            total_speed_mps=total_speed_mps,
            mach=mach,
            dynamic_pressure_pa=dynamic_pressure_pa,
            mass_kg=mass_kg,
            thrust_n=thrust_n,
            drag_n=drag_n,
        )
        return diagnostics, acceleration_ecef_mps2

    def diagnostics(self, time_s: float, state: npt.ArrayLike) -> RotatingAscentDiagnostics:
        """Return deterministic diagnostics at a state."""
        diagnostics, _ = self._loads(time_s, state)
        return diagnostics

    def derivative(self, time_s: float, state: FloatArray) -> FloatArray:
        """Return ECEF derivative ordered ``[velocity, acceleration]``."""
        values = as_vector(state, 6, name="state")
        _, acceleration_ecef_mps2 = self._loads(time_s, values)
        return np.concatenate((values[3:], acceleration_ecef_mps2))

    def events(self) -> tuple[EventSpec, EventSpec, EventSpec]:
        """Return burnout, local-vertical apogee, and ellipsoid-impact events."""
        return (
            EventSpec(
                "burnout",
                lambda time_s, _state: time_s - self.base.vehicle.propulsion.burnout_time_s,
                direction=1,
            ),
            EventSpec(
                "apogee",
                lambda time_s, state: self.diagnostics(time_s, state).velocity_ned_mps[2],
                direction=1,
            ),
            EventSpec(
                "ground_impact",
                lambda time_s, state: (
                    self.diagnostics(time_s, state).geodetic.altitude_m
                    - self.site.geodetic.altitude_m
                ),
                direction=-1,
                terminal=True,
            ),
        )


def _maximum_record(time_s: FloatArray, values: FloatArray, unit: str) -> dict[str, float | str]:
    index = int(np.argmax(values))
    return {"value": float(values[index]), "unit": unit, "time_s": float(time_s[index])}


def _event_summary(
    model: RotatingAscentModel,
    events: tuple[EventOccurrence, ...],
) -> tuple[dict[str, float | str], ...]:
    records: list[dict[str, float | str]] = []
    for event in events:
        diagnostic = model.diagnostics(event.time_s, event.state)
        records.append(
            {
                "name": event.name,
                "time_s": event.time_s,
                "altitude_m": diagnostic.geodetic.altitude_m - model.site.geodetic.altitude_m,
                "ground_range_m": float(np.linalg.norm(diagnostic.position_ned_m[:2])),
                "speed_mps": diagnostic.total_speed_mps,
            }
        )
    return tuple(records)


def simulate_rotating_ascent(configuration: RotatingAscentConfiguration) -> SimulationResult:
    """Run the deterministic rotating-oblate-planet ascent scenario."""
    model = RotatingAscentModel(configuration)
    start = perf_counter()
    integration = integrate_fixed_step(
        model.derivative,
        model.initial_state(),
        (0.0, configuration.base_configuration.simulation.maximum_time_s),
        configuration.base_configuration.simulation.step_s,
        events=model.events(),
    )
    execution_time_s = perf_counter() - start
    sample_count = integration.time_s.size
    names = (
        "latitude_deg",
        "longitude_deg",
        "ellipsoid_altitude_m",
        "altitude_m",
        "north_m",
        "east_m",
        "down_m",
        "velocity_north_mps",
        "velocity_east_mps",
        "velocity_down_mps",
        "total_velocity_mps",
        "airspeed_mps",
        "acceleration_north_mps2",
        "acceleration_east_mps2",
        "acceleration_down_mps2",
        "mach",
        "dynamic_pressure_pa",
        "mass_kg",
        "thrust_n",
        "drag_n",
    )
    columns = {name: np.empty(sample_count, dtype=np.float64) for name in names}
    columns.update(
        {
            "ecef_x_m": integration.state[:, 0].copy(),
            "ecef_y_m": integration.state[:, 1].copy(),
            "ecef_z_m": integration.state[:, 2].copy(),
            "velocity_ecef_x_mps": integration.state[:, 3].copy(),
            "velocity_ecef_y_mps": integration.state[:, 4].copy(),
            "velocity_ecef_z_mps": integration.state[:, 5].copy(),
        }
    )
    for index, (time_s, state) in enumerate(
        zip(integration.time_s, integration.state, strict=True)
    ):
        diagnostic = model.diagnostics(float(time_s), state)
        columns["latitude_deg"][index] = np.rad2deg(diagnostic.geodetic.latitude_rad)
        columns["longitude_deg"][index] = np.rad2deg(diagnostic.geodetic.longitude_rad)
        columns["ellipsoid_altitude_m"][index] = diagnostic.geodetic.altitude_m
        columns["altitude_m"][index] = (
            diagnostic.geodetic.altitude_m - model.site.geodetic.altitude_m
        )
        columns["north_m"][index], columns["east_m"][index], columns["down_m"][index] = (
            diagnostic.position_ned_m
        )
        (
            columns["velocity_north_mps"][index],
            columns["velocity_east_mps"][index],
            columns["velocity_down_mps"][index],
        ) = diagnostic.velocity_ned_mps
        columns["total_velocity_mps"][index] = diagnostic.total_speed_mps
        columns["airspeed_mps"][index] = diagnostic.airspeed_mps
        (
            columns["acceleration_north_mps2"][index],
            columns["acceleration_east_mps2"][index],
            columns["acceleration_down_mps2"][index],
        ) = diagnostic.acceleration_ned_mps2
        columns["mach"][index] = diagnostic.mach
        columns["dynamic_pressure_pa"][index] = diagnostic.dynamic_pressure_pa
        columns["mass_kg"][index] = diagnostic.mass_kg
        columns["thrust_n"][index] = diagnostic.thrust_n
        columns["drag_n"][index] = diagnostic.drag_n
    ground_range_m = np.hypot(columns["north_m"], columns["east_m"])
    maximum_summary = {
        "altitude": _maximum_record(integration.time_s, columns["altitude_m"], "m"),
        "ground_range": _maximum_record(integration.time_s, ground_range_m, "m"),
        "speed": _maximum_record(integration.time_s, columns["total_velocity_mps"], "m/s"),
        "mach": _maximum_record(integration.time_s, columns["mach"], "1"),
        "dynamic_pressure": _maximum_record(
            integration.time_s, columns["dynamic_pressure_pa"], "Pa"
        ),
    }
    return SimulationResult(
        scenario_name="rotating_planet_ascent",
        time_s=integration.time_s,
        columns=columns,
        events=integration.events,
        event_summary=_event_summary(model, integration.events),
        maximum_summary=maximum_summary,
        execution_time_s=execution_time_s,
    )
