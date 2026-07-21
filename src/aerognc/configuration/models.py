"""Typed configuration records used after YAML validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aerognc.environment.atmosphere import StandardAtmosphere1976
from aerognc.environment.gravity import GravityModel
from aerognc.environment.wind import WindModel
from aerognc.vehicle.actuators import ActuatorAllocator, ActuatorLimits
from aerognc.vehicle.aerodynamics import AerodynamicModel
from aerognc.vehicle.mass_properties import MassPropertiesModel
from aerognc.vehicle.propulsion import ThrustCurve


@dataclass(frozen=True, slots=True)
class VehicleDefinition:
    """Validated fictional vehicle configuration and constructed subsystem models."""

    name: str
    description: str
    fictional: bool
    propulsion: ThrustCurve
    mass_properties: MassPropertiesModel
    aerodynamics: AerodynamicModel
    actuator_limits: ActuatorLimits
    actuator_allocator: ActuatorAllocator


@dataclass(frozen=True, slots=True)
class LaunchConfiguration:
    """Initial point-mass launch condition."""

    position_ned_m: tuple[float, float, float]
    initial_speed_mps: float
    elevation_deg: float
    azimuth_deg: float
    thrust_alignment: Literal["velocity", "launch_axis"]


@dataclass(frozen=True, slots=True)
class SimulationConfiguration:
    """Integration and output settings."""

    name: str
    step_s: float
    maximum_time_s: float
    random_seed: int
    output_directory: Path


@dataclass(frozen=True, slots=True)
class EnvironmentDefinition:
    """Constructed environment models."""

    atmosphere: StandardAtmosphere1976
    gravity: GravityModel
    wind: WindModel


@dataclass(frozen=True, slots=True)
class ThreeDofConfiguration:
    """Complete validated configuration for the point-mass ascent."""

    source_path: Path
    safety_scope: str
    simulation: SimulationConfiguration
    launch: LaunchConfiguration
    environment: EnvironmentDefinition
    vehicle: VehicleDefinition
