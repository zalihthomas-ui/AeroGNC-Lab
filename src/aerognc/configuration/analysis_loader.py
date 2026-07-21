"""Validated configuration for linear flight-control engineering analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

from aerognc.configuration.loader import (
    _integer,
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _number_tuple,
    _string,
)


@dataclass(frozen=True, slots=True)
class FlightControlAnalysisConfiguration:
    """Synthetic plant, LQR, schedule, frequency, and SIL benchmark inputs."""

    source_path: Path
    name: str
    safety_scope: str
    nominal_inertia_kgm2: float
    passive_damping_nms: float
    trim_disturbance_moment_nm: float
    state_weight_diagonal: tuple[float, float]
    input_weight: float
    scheduling_points: tuple[float, ...]
    scheduled_inertia_kgm2: tuple[float, ...]
    minimum_frequency_radps: float
    maximum_frequency_radps: float
    frequency_sample_count: int
    sil_sample_count: int
    sil_repeat_count: int
    sil_deadline_s: float
    random_seed: int
    output_directory: Path


def load_flight_control_analysis_configuration(
    path: str | Path,
) -> FlightControlAnalysisConfiguration:
    """Load and strictly validate one linear-analysis YAML file."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "flight_control_analysis",
        required={
            "metadata",
            "plant",
            "lqr",
            "gain_schedule",
            "frequency_analysis",
            "sil_timing",
            "output_directory",
        },
    )
    metadata = _mapping(root["metadata"], "flight_control_analysis.metadata")
    _keys(metadata, "flight_control_analysis.metadata", required={"name", "safety_scope"})
    safety_scope = _string(metadata["safety_scope"], "flight_control_analysis.safety_scope")
    folded = safety_scope.casefold()
    if not all(word in folded for word in ("fictional", "civilian", "synthetic")):
        raise ValueError(
            "flight-control analysis safety scope must say fictional, civilian, synthetic"
        )
    plant = _mapping(root["plant"], "flight_control_analysis.plant")
    _keys(
        plant,
        "flight_control_analysis.plant",
        required={
            "nominal_inertia_kgm2",
            "passive_damping_nms",
            "trim_disturbance_moment_nm",
        },
    )
    lqr = _mapping(root["lqr"], "flight_control_analysis.lqr")
    _keys(lqr, "flight_control_analysis.lqr", required={"state_weight_diagonal", "input_weight"})
    weights = cast(
        tuple[float, float],
        _number_tuple(
            lqr["state_weight_diagonal"],
            "flight_control_analysis.lqr.state_weight_diagonal",
            length=2,
        ),
    )
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("LQR state weights must be positive")
    schedule = _mapping(root["gain_schedule"], "flight_control_analysis.gain_schedule")
    _keys(
        schedule,
        "flight_control_analysis.gain_schedule",
        required={"scheduling_points", "inertia_kgm2"},
    )
    points = _number_tuple(
        schedule["scheduling_points"],
        "flight_control_analysis.gain_schedule.scheduling_points",
    )
    inertias = _number_tuple(
        schedule["inertia_kgm2"],
        "flight_control_analysis.gain_schedule.inertia_kgm2",
    )
    if any(inertia <= 0.0 for inertia in inertias):
        raise ValueError("scheduled inertias must be positive")
    if (
        len(points) < 2
        or len(points) != len(inertias)
        or any(after <= before for before, after in pairwise(points))
    ):
        raise ValueError("gain schedule points must be increasing and match inertia count")
    frequency = _mapping(root["frequency_analysis"], "flight_control_analysis.frequency")
    _keys(
        frequency,
        "flight_control_analysis.frequency",
        required={"minimum_radps", "maximum_radps", "sample_count"},
    )
    minimum_frequency = _number(
        frequency["minimum_radps"], "flight_control_analysis.frequency.minimum_radps", positive=True
    )
    maximum_frequency = _number(
        frequency["maximum_radps"], "flight_control_analysis.frequency.maximum_radps", positive=True
    )
    if maximum_frequency <= minimum_frequency:
        raise ValueError("maximum analysis frequency must exceed minimum frequency")
    timing = _mapping(root["sil_timing"], "flight_control_analysis.sil_timing")
    _keys(
        timing,
        "flight_control_analysis.sil_timing",
        required={"sample_count", "repeat_count", "deadline_s", "random_seed"},
    )
    return FlightControlAnalysisConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "flight_control_analysis.metadata.name"),
        safety_scope=safety_scope,
        nominal_inertia_kgm2=_number(
            plant["nominal_inertia_kgm2"],
            "flight_control_analysis.plant.nominal_inertia_kgm2",
            positive=True,
        ),
        passive_damping_nms=_number(
            plant["passive_damping_nms"],
            "flight_control_analysis.plant.passive_damping_nms",
            nonnegative=True,
        ),
        trim_disturbance_moment_nm=_number(
            plant["trim_disturbance_moment_nm"],
            "flight_control_analysis.plant.trim_disturbance_moment_nm",
        ),
        state_weight_diagonal=weights,
        input_weight=_number(
            lqr["input_weight"], "flight_control_analysis.lqr.input_weight", positive=True
        ),
        scheduling_points=points,
        scheduled_inertia_kgm2=inertias,
        minimum_frequency_radps=minimum_frequency,
        maximum_frequency_radps=maximum_frequency,
        frequency_sample_count=_positive_integer(
            frequency["sample_count"], "flight_control_analysis.frequency.sample_count", 100
        ),
        sil_sample_count=_positive_integer(
            timing["sample_count"], "flight_control_analysis.sil_timing.sample_count", 1
        ),
        sil_repeat_count=_positive_integer(
            timing["repeat_count"], "flight_control_analysis.sil_timing.repeat_count", 1
        ),
        sil_deadline_s=_number(
            timing["deadline_s"], "flight_control_analysis.sil_timing.deadline_s", positive=True
        ),
        random_seed=_integer(
            timing["random_seed"],
            "flight_control_analysis.sil_timing.random_seed",
            nonnegative=True,
        ),
        output_directory=Path(
            _string(root["output_directory"], "flight_control_analysis.output_directory")
        ),
    )


def _positive_integer(value: object, context: str, minimum: int) -> int:
    parsed = _integer(value, context)
    if parsed < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return parsed
