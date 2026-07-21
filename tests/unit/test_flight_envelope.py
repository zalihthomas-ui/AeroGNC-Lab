from pathlib import Path

import numpy as np
import pytest

from aerognc.configuration import load_flight_envelope_configuration
from aerognc.gnc.flight_envelope import (
    ScheduledStateFeedback,
    analyze_flight_envelope,
    make_operating_point,
)
from aerognc.verification.flight_envelope import assess_flight_envelope


def test_envelope_grid_trims_and_scheduled_control_meet_requirements() -> None:
    configuration = load_flight_envelope_configuration("configs/flight_envelope.yaml")
    result = analyze_flight_envelope(configuration)

    assert len(result.analyses) == 36
    assert result.all_trim_converged
    assert all(item.controllability_rank == 2 for item in result.analyses)
    assert all(item.observability_rank == 2 for item in result.analyses)
    assert all(
        np.linalg.norm(item.linear_model.derivative_at_trim, ord=np.inf) < 1.0e-8
        for item in result.analyses
    )
    assert result.schedule_verification.stable_point_count == 12
    assert result.robustness_verification.stable_sample_count == 120
    assert assess_flight_envelope(result).all_pass

    first = result.analyses[0]
    point = first.operating_point
    np.testing.assert_allclose(
        result.gain_schedule.gain(point.mach, point.altitude_m, point.mass_kg),
        first.lqr.gain,
        rtol=1.0e-12,
    )
    command = result.gain_schedule.command(
        [1.0, 1.0],
        mach=point.mach,
        altitude_m=point.altitude_m,
        mass_kg=point.mass_kg,
        command_limit_rad=0.05,
    )
    assert abs(command) <= 0.05


def test_mass_property_schedule_and_uncertainty_are_reproducible() -> None:
    configuration = load_flight_envelope_configuration("configs/flight_envelope.yaml")
    dry = make_operating_point(
        configuration,
        configuration.mach_points[0],
        configuration.altitude_points_m[0],
        configuration.mass_points_kg[0],
    )
    wet = make_operating_point(
        configuration,
        configuration.mach_points[-1],
        configuration.altitude_points_m[-1],
        configuration.mass_points_kg[-1],
    )
    model = configuration.base_scenario.vehicle.mass_properties
    assert dry.pitch_inertia_kgm2 == pytest.approx(model.dry_inertia_body_kgm2[1, 1])
    assert wet.pitch_inertia_kgm2 == pytest.approx(model.wet_inertia_body_kgm2[1, 1])

    first = analyze_flight_envelope(configuration).robustness_verification
    second = analyze_flight_envelope(configuration).robustness_verification
    assert first == second


def test_gain_schedule_and_configuration_reject_invalid_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gain_values"):
        ScheduledStateFeedback([0.5, 1.0], [0.0, 1000.0], [20.0, 30.0], np.zeros((2, 2)))

    invalid = tmp_path / "flight_envelope.yaml"
    source = Path("configs/flight_envelope.yaml").read_text(encoding="utf-8")
    scenario_path = Path("configs/three_dof_aero_database.yaml").resolve().as_posix()
    source = source.replace(
        "base_scenario: three_dof_aero_database.yaml",
        f'base_scenario: "{scenario_path}"',
    )
    invalid.write_text(source.replace("sample_count: 120", "sample_count: 5"), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 10"):
        load_flight_envelope_configuration(invalid)
