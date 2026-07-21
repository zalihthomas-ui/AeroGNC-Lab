"""Strict YAML boundary for the fictional multistage/recovery demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aerognc.configuration.loader import (
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _number_tuple,
    _sequence,
    _string,
)
from aerognc.vehicle.propulsion import ThrustCurve
from aerognc.vehicle.recovery import RecoveryDevice
from aerognc.vehicle.staging import MultistageVehicle, StageDefinition


@dataclass(frozen=True, slots=True)
class MultistageRecoveryConfiguration:
    """Validated one-axis staging and recovery scenario."""

    source_path: Path
    name: str
    safety_scope: str
    vehicle: MultistageVehicle
    recovery: RecoveryDevice
    initial_altitude_m: float
    initial_velocity_down_mps: float
    density_kgpm3: float
    gravity_mps2: float
    body_drag_area_m2: float
    body_drag_coefficient: float
    step_s: float
    maximum_time_s: float
    output_directory: Path


def load_multistage_recovery_configuration(
    path: str | Path,
) -> MultistageRecoveryConfiguration:
    """Load synthetic stages, recovery schedule, and vertical benchmark settings."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "multistage_recovery",
        required={"metadata", "vehicle", "recovery", "simulation"},
    )
    metadata = _mapping(root["metadata"], "multistage_recovery.metadata")
    _keys(metadata, "multistage_recovery.metadata", required={"name", "safety_scope", "fictional"})
    if metadata["fictional"] is not True:
        raise ValueError("multistage_recovery.metadata.fictional must be true")
    safety_scope = _string(metadata["safety_scope"], "multistage_recovery.metadata.safety_scope")
    if "fictional" not in safety_scope.lower() or "civilian" not in safety_scope.lower():
        raise ValueError("multistage recovery safety_scope must state fictional and civilian")

    vehicle_data = _mapping(root["vehicle"], "multistage_recovery.vehicle")
    _keys(vehicle_data, "multistage_recovery.vehicle", required={"payload_mass_kg", "stages"})
    stage_rows = _sequence(vehicle_data["stages"], "multistage_recovery.vehicle.stages")
    stages: list[StageDefinition] = []
    for index, raw_stage in enumerate(stage_rows):
        context = f"multistage_recovery.vehicle.stages[{index}]"
        stage = _mapping(raw_stage, context)
        _keys(
            stage,
            context,
            required={
                "name",
                "dry_mass_kg",
                "propellant_mass_kg",
                "ignition_time_s",
                "separation_time_s",
                "thrust_time_s",
                "thrust_n",
            },
        )
        separation_raw = stage["separation_time_s"]
        separation_time_s = (
            None
            if separation_raw is None
            else _number(separation_raw, f"{context}.separation_time_s", nonnegative=True)
        )
        propulsion = ThrustCurve(
            _number_tuple(stage["thrust_time_s"], f"{context}.thrust_time_s"),
            _number_tuple(stage["thrust_n"], f"{context}.thrust_n"),
            _number(stage["propellant_mass_kg"], f"{context}.propellant_mass_kg", positive=True),
        )
        stages.append(
            StageDefinition(
                name=_string(stage["name"], f"{context}.name"),
                dry_mass_kg=_number(stage["dry_mass_kg"], f"{context}.dry_mass_kg", positive=True),
                propulsion=propulsion,
                ignition_time_s=_number(
                    stage["ignition_time_s"], f"{context}.ignition_time_s", nonnegative=True
                ),
                separation_time_s=separation_time_s,
            )
        )
    vehicle = MultistageVehicle(
        payload_mass_kg=_number(
            vehicle_data["payload_mass_kg"],
            "multistage_recovery.vehicle.payload_mass_kg",
            positive=True,
        ),
        stages=tuple(stages),
    )

    recovery_data = _mapping(root["recovery"], "multistage_recovery.recovery")
    recovery_keys = {
        "trigger_time_s",
        "deployment_delay_s",
        "reefing_time_s",
        "reefed_hold_time_s",
        "inflation_time_s",
        "reefed_area_m2",
        "full_area_m2",
        "drag_coefficient",
    }
    _keys(recovery_data, "multistage_recovery.recovery", required=recovery_keys)
    recovery = RecoveryDevice(
        trigger_time_s=_number(
            recovery_data["trigger_time_s"],
            "multistage_recovery.recovery.trigger_time_s",
            nonnegative=True,
        ),
        deployment_delay_s=_number(
            recovery_data["deployment_delay_s"],
            "multistage_recovery.recovery.deployment_delay_s",
            nonnegative=True,
        ),
        reefing_time_s=_number(
            recovery_data["reefing_time_s"],
            "multistage_recovery.recovery.reefing_time_s",
            positive=True,
        ),
        reefed_hold_time_s=_number(
            recovery_data["reefed_hold_time_s"],
            "multistage_recovery.recovery.reefed_hold_time_s",
            nonnegative=True,
        ),
        inflation_time_s=_number(
            recovery_data["inflation_time_s"],
            "multistage_recovery.recovery.inflation_time_s",
            positive=True,
        ),
        reefed_area_m2=_number(
            recovery_data["reefed_area_m2"],
            "multistage_recovery.recovery.reefed_area_m2",
            positive=True,
        ),
        full_area_m2=_number(
            recovery_data["full_area_m2"],
            "multistage_recovery.recovery.full_area_m2",
            positive=True,
        ),
        drag_coefficient=_number(
            recovery_data["drag_coefficient"],
            "multistage_recovery.recovery.drag_coefficient",
            positive=True,
        ),
    )

    simulation = _mapping(root["simulation"], "multistage_recovery.simulation")
    simulation_keys = {
        "initial_altitude_m",
        "initial_velocity_down_mps",
        "density_kgpm3",
        "gravity_mps2",
        "body_drag_area_m2",
        "body_drag_coefficient",
        "step_s",
        "maximum_time_s",
        "output_directory",
    }
    _keys(simulation, "multistage_recovery.simulation", required=simulation_keys)
    return MultistageRecoveryConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "multistage_recovery.metadata.name"),
        safety_scope=safety_scope,
        vehicle=vehicle,
        recovery=recovery,
        initial_altitude_m=_number(
            simulation["initial_altitude_m"],
            "multistage_recovery.simulation.initial_altitude_m",
            nonnegative=True,
        ),
        initial_velocity_down_mps=_number(
            simulation["initial_velocity_down_mps"],
            "multistage_recovery.simulation.initial_velocity_down_mps",
        ),
        density_kgpm3=_number(
            simulation["density_kgpm3"],
            "multistage_recovery.simulation.density_kgpm3",
            nonnegative=True,
        ),
        gravity_mps2=_number(
            simulation["gravity_mps2"],
            "multistage_recovery.simulation.gravity_mps2",
            positive=True,
        ),
        body_drag_area_m2=_number(
            simulation["body_drag_area_m2"],
            "multistage_recovery.simulation.body_drag_area_m2",
            positive=True,
        ),
        body_drag_coefficient=_number(
            simulation["body_drag_coefficient"],
            "multistage_recovery.simulation.body_drag_coefficient",
            positive=True,
        ),
        step_s=_number(
            simulation["step_s"], "multistage_recovery.simulation.step_s", positive=True
        ),
        maximum_time_s=_number(
            simulation["maximum_time_s"],
            "multistage_recovery.simulation.maximum_time_s",
            positive=True,
        ),
        output_directory=Path(
            _string(
                simulation["output_directory"],
                "multistage_recovery.simulation.output_directory",
            )
        ),
    )
