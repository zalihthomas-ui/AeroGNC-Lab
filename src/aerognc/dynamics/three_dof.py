"""Three-dimensional point-mass flight mechanics in local NED coordinates."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.configuration.models import (
    EnvironmentDefinition,
    LaunchConfiguration,
    VehicleDefinition,
)
from aerognc.dynamics.state import ThreeDofState
from aerognc.environment.atmosphere import dynamic_pressure_pa, mach_number
from aerognc.mathematics.coordinates import launch_direction_ned
from aerognc.mathematics.vectors import FloatArray, as_vector


def point_mass_derivative(
    state: npt.ArrayLike,
    *,
    mass_kg: float,
    applied_force_ned_n: npt.ArrayLike,
    gravity_ned_mps2: npt.ArrayLike,
) -> FloatArray:
    """Return point-mass derivative for explicit force, mass, and gravity inputs."""
    point_state = ThreeDofState.from_array(state)
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive and finite")
    force_n = as_vector(applied_force_ned_n, 3, name="applied_force_ned_n")
    gravity_n = as_vector(gravity_ned_mps2, 3, name="gravity_ned_mps2")
    acceleration_ned_mps2 = force_n / mass_kg + gravity_n
    return np.concatenate((point_state.velocity_ned_mps, acceleration_ned_mps2))


@dataclass(frozen=True, slots=True)
class ThreeDofDiagnostics:
    """Instantaneous point-mass quantities in SI units."""

    altitude_m: float
    ground_range_m: float
    vertical_velocity_up_mps: float
    total_velocity_mps: float
    airspeed_mps: float
    acceleration_ned_mps2: FloatArray
    acceleration_magnitude_mps2: float
    mach: float
    dynamic_pressure_pa: float
    mass_kg: float
    thrust_n: float
    drag_n: float
    wind_velocity_ned_mps: FloatArray
    flight_path_angle_rad: float


class PointMassModel:
    """Configured fictional research-rocket point-mass model."""

    def __init__(
        self,
        vehicle: VehicleDefinition,
        environment: EnvironmentDefinition,
        launch: LaunchConfiguration,
        *,
        thrust_misalignment_pitch_rad: float = 0.0,
        thrust_misalignment_yaw_rad: float = 0.0,
    ) -> None:
        self.vehicle = vehicle
        self.environment = environment
        self.launch = launch
        if not np.isfinite([thrust_misalignment_pitch_rad, thrust_misalignment_yaw_rad]).all():
            raise ValueError("thrust misalignment angles must be finite")
        self.thrust_misalignment_pitch_rad = float(thrust_misalignment_pitch_rad)
        self.thrust_misalignment_yaw_rad = float(thrust_misalignment_yaw_rad)
        self.launch_direction_ned = launch_direction_ned(
            np.deg2rad(launch.elevation_deg), np.deg2rad(launch.azimuth_deg)
        )

    def initial_state(self) -> FloatArray:
        """Return configured initial point-mass state."""
        velocity = self.launch.initial_speed_mps * self.launch_direction_ned
        return ThreeDofState(
            np.asarray(self.launch.position_ned_m, dtype=np.float64), velocity
        ).as_array()

    def _loads(
        self, time_s: float, state: ThreeDofState
    ) -> tuple[FloatArray, float, float, float, float, FloatArray]:
        altitude_m = -float(state.position_ned_m[2])
        atmosphere = self.environment.atmosphere.properties(altitude_m)
        wind_ned_mps = self.environment.wind.velocity_ned_mps(time_s, altitude_m)
        air_velocity_ned_mps = state.velocity_ned_mps - wind_ned_mps
        airspeed_mps = float(np.linalg.norm(air_velocity_ned_mps))
        mach = mach_number(airspeed_mps, atmosphere.speed_of_sound_mps)
        dynamic_pressure = dynamic_pressure_pa(atmosphere.density_kgpm3, airspeed_mps)
        drag_n = self.vehicle.aerodynamics.drag_force_n(dynamic_pressure, mach)
        if airspeed_mps > 1.0e-9:
            drag_force_ned_n = -drag_n * air_velocity_ned_mps / airspeed_mps
        else:
            drag_force_ned_n = np.zeros(3)

        thrust_n = self.vehicle.propulsion.thrust_at_time_n(time_s)
        speed_mps = float(np.linalg.norm(state.velocity_ned_mps))
        if self.launch.thrust_alignment == "velocity" and speed_mps > 1.0e-6:
            thrust_direction_ned = state.velocity_ned_mps / speed_mps
        else:
            thrust_direction_ned = self.launch_direction_ned
        thrust_direction_ned = self._misaligned_thrust_direction(thrust_direction_ned)
        applied_force_ned_n = thrust_n * thrust_direction_ned + drag_force_ned_n
        return applied_force_ned_n, thrust_n, drag_n, mach, dynamic_pressure, wind_ned_mps

    def _misaligned_thrust_direction(self, nominal_direction_ned: FloatArray) -> FloatArray:
        if self.thrust_misalignment_pitch_rad == 0.0 and self.thrust_misalignment_yaw_rad == 0.0:
            return nominal_direction_ned
        down = np.array([0.0, 0.0, 1.0])
        lateral = np.cross(down, nominal_direction_ned)
        if np.linalg.norm(lateral) < 1.0e-9:
            lateral = np.array([0.0, 1.0, 0.0])
        lateral /= np.linalg.norm(lateral)
        normal = np.cross(nominal_direction_ned, lateral)
        direction = (
            nominal_direction_ned
            + np.tan(self.thrust_misalignment_pitch_rad) * normal
            + np.tan(self.thrust_misalignment_yaw_rad) * lateral
        )
        return np.asarray(direction / np.linalg.norm(direction), dtype=np.float64)

    def derivative(self, time_s: float, state_array: FloatArray) -> FloatArray:
        """Return configured point-mass derivative."""
        state = ThreeDofState.from_array(state_array)
        altitude_m = -float(state.position_ned_m[2])
        mass_kg = self.vehicle.mass_properties.at_time(time_s).mass_kg
        applied_force, _thrust, _drag, _mach, _q, _wind = self._loads(time_s, state)
        gravity = self.environment.gravity.acceleration_ned_mps2(altitude_m)
        return point_mass_derivative(
            state_array,
            mass_kg=mass_kg,
            applied_force_ned_n=applied_force,
            gravity_ned_mps2=gravity,
        )

    def diagnostics(self, time_s: float, state_array: npt.ArrayLike) -> ThreeDofDiagnostics:
        """Evaluate logged engineering quantities at an accepted state."""
        state = ThreeDofState.from_array(state_array)
        altitude_m = -float(state.position_ned_m[2])
        horizontal_speed_mps = float(np.linalg.norm(state.velocity_ned_mps[:2]))
        total_velocity_mps = float(np.linalg.norm(state.velocity_ned_mps))
        applied_force, thrust_n, drag_n, mach, dynamic_pressure, wind = self._loads(time_s, state)
        mass_kg = self.vehicle.mass_properties.at_time(time_s).mass_kg
        gravity = self.environment.gravity.acceleration_ned_mps2(altitude_m)
        acceleration = applied_force / mass_kg + gravity
        flight_path_angle = float(np.arctan2(-state.velocity_ned_mps[2], horizontal_speed_mps))
        return ThreeDofDiagnostics(
            altitude_m=altitude_m,
            ground_range_m=float(np.linalg.norm(state.position_ned_m[:2])),
            vertical_velocity_up_mps=-float(state.velocity_ned_mps[2]),
            total_velocity_mps=total_velocity_mps,
            airspeed_mps=float(np.linalg.norm(state.velocity_ned_mps - wind)),
            acceleration_ned_mps2=acceleration,
            acceleration_magnitude_mps2=float(np.linalg.norm(acceleration)),
            mach=mach,
            dynamic_pressure_pa=dynamic_pressure,
            mass_kg=mass_kg,
            thrust_n=thrust_n,
            drag_n=drag_n,
            wind_velocity_ned_mps=wind,
            flight_path_angle_rad=flight_path_angle,
        )
