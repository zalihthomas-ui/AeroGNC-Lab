from dataclasses import replace

import numpy as np
import pytest

from aerognc.configuration import load_aircraft_configuration
from aerognc.vehicle.fixed_wing import (
    AircraftControlCommand,
    FixedWingFlightModel,
    aerodynamic_state,
    aircraft_initial_state,
    aircraft_stall_speed_mps,
    initial_tangent_displacement_ned_m,
    longitudinal_trim_command,
    longitudinal_trim_elevator_rad,
)
from aerognc.verification.aircraft_aerodynamics import (
    compare_aircraft_aerodynamic_backends,
    write_aircraft_aerodynamic_comparison,
)


def _air_velocity(alpha_deg: float, speed_mps: float = 80.0) -> np.ndarray:
    alpha_rad = np.deg2rad(alpha_deg)
    return speed_mps * np.array([np.cos(alpha_rad), 0.0, np.sin(alpha_rad)])


def test_aerodynamic_model_has_linear_lift_then_post_stall_loss_and_drag_rise() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")

    low = aerodynamic_state(_air_velocity(5.0), np.zeros(3), np.zeros(3), 1.0, 340.0, configuration)
    near_stall = aerodynamic_state(
        _air_velocity(14.0), np.zeros(3), np.zeros(3), 1.0, 340.0, configuration
    )
    post_stall = aerodynamic_state(
        _air_velocity(40.0), np.zeros(3), np.zeros(3), 1.0, 340.0, configuration
    )

    assert near_stall.lift_coefficient > low.lift_coefficient
    assert post_stall.lift_coefficient < near_stall.lift_coefficient
    assert post_stall.drag_coefficient > 5.0 * near_stall.drag_coefficient
    assert not low.stalled
    assert post_stall.stalled


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_post_stall_lift_is_continuous_at_both_signed_boundaries(sign: float) -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    stall_deg = np.rad2deg(configuration.aerodynamics.stall_angle_rad)
    inside = aerodynamic_state(
        _air_velocity(sign * (stall_deg - 1.0e-4)),
        np.zeros(3),
        np.zeros(3),
        1.0,
        340.0,
        configuration,
    )
    outside = aerodynamic_state(
        _air_velocity(sign * (stall_deg + 1.0e-4)),
        np.zeros(3),
        np.zeros(3),
        1.0,
        340.0,
        configuration,
    )

    assert outside.lift_coefficient == pytest.approx(inside.lift_coefficient, abs=2.0e-5)


def test_cl_cd_cm_and_mass_each_change_the_equations_of_motion() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    state = aircraft_initial_state(configuration)
    command = longitudinal_trim_command(configuration)
    baseline = FixedWingFlightModel(configuration).derivative(0.0, state, command)

    lower_cl = replace(
        configuration,
        aerodynamics=replace(
            configuration.aerodynamics,
            cl_alpha_per_rad=0.7 * configuration.aerodynamics.cl_alpha_per_rad,
        ),
    )
    higher_cd = replace(
        configuration,
        aerodynamics=replace(
            configuration.aerodynamics,
            cd_zero=2.0 * configuration.aerodynamics.cd_zero,
        ),
    )
    different_cm = replace(
        configuration,
        aerodynamics=replace(
            configuration.aerodynamics,
            pitch_alpha_per_rad=0.5 * configuration.aerodynamics.pitch_alpha_per_rad,
        ),
    )
    lighter_state = state.copy()
    lighter_state[13] = configuration.mass.dry_mass_kg + 20.0

    cl_derivative = FixedWingFlightModel(lower_cl).derivative(0.0, state, command)
    cd_derivative = FixedWingFlightModel(higher_cd).derivative(0.0, state, command)
    cm_derivative = FixedWingFlightModel(different_cm).derivative(0.0, state, command)
    mass_derivative = FixedWingFlightModel(configuration).derivative(0.0, lighter_state, command)

    assert not np.isclose(cl_derivative[3:6], baseline[3:6]).all()
    assert not np.isclose(cd_derivative[3:6], baseline[3:6]).all()
    assert not np.isclose(cm_derivative[10:13], baseline[10:13]).all()
    assert not np.isclose(mass_derivative[3:6], baseline[3:6]).all()


def test_control_surface_command_has_lag_rate_limit_and_expected_pitch_sign() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    model = FixedWingFlightModel(configuration)
    state = aircraft_initial_state(configuration)
    derivative = model.derivative(
        0.0,
        state,
        AircraftControlCommand(pitch=1.0, throttle=configuration.initial_throttle),
    )

    assert derivative[15] < 0.0
    assert abs(derivative[15]) <= configuration.geometry.control_rate_limit_radps


def test_initial_state_starts_at_zero_body_rate_with_trimmed_elevator() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    state = aircraft_initial_state(configuration)

    np.testing.assert_array_equal(state[10:13], np.zeros(3))
    assert state[14] == 0.0
    assert state[15] == pytest.approx(
        longitudinal_trim_elevator_rad(
            configuration.initial.angle_of_attack_rad, configuration
        )
    )
    assert state[16] == 0.0


def test_positive_yaw_command_produces_positive_initial_yaw_acceleration() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    model = FixedWingFlightModel(configuration)
    state = aircraft_initial_state(configuration)
    baseline = model.derivative(
        0.0, state, AircraftControlCommand(throttle=configuration.initial_throttle)
    )
    commanded_state = state.copy()
    commanded_state[16] = -configuration.geometry.rudder_limit_rad
    commanded = model.derivative(
        0.0,
        commanded_state,
        AircraftControlCommand(yaw=1.0, throttle=configuration.initial_throttle),
    )

    neutral_derivative = model.derivative(
        0.0,
        state,
        AircraftControlCommand(yaw=1.0, throttle=configuration.initial_throttle),
    )
    assert neutral_derivative[16] < 0.0
    assert commanded[12] > baseline[12]


def test_planet_fixed_local_displacement_removes_surface_rotation() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    initial = aircraft_initial_state(configuration)[:3]
    time_s = 90.0
    angle = configuration.planet.rotation_rate_radps * time_s
    cosine = np.cos(angle)
    sine = np.sin(angle)
    co_rotating_position = np.array(
        [
            cosine * initial[0] - sine * initial[1],
            sine * initial[0] + cosine * initial[1],
            initial[2],
        ]
    )

    displacement = initial_tangent_displacement_ned_m(
        co_rotating_position,
        time_s,
        initial,
        configuration.planet.rotation_rate_radps,
    )

    np.testing.assert_allclose(displacement, np.zeros(3), atol=1.0e-9)


def test_stall_speed_obeys_mass_density_area_and_clmax_relation() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    speed = aircraft_stall_speed_mps(4_200.0, 1.225, configuration)
    heavier = aircraft_stall_speed_mps(5_000.0, 1.225, configuration)

    assert speed == pytest.approx(41.93, rel=0.02)
    assert heavier > speed


def test_optional_static_table_backend_loads_and_has_provenance(tmp_path) -> None:
    analytic = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    table_configuration = load_aircraft_configuration("configs/aircraft_tabular.yaml")
    assert table_configuration.aerodynamic_backend == "table"
    assert table_configuration.aerodynamic_table_path is not None
    model = FixedWingFlightModel(table_configuration)
    state = aircraft_initial_state(table_configuration)
    loads = model.loads(0.0, state, AircraftControlCommand(throttle=0.28))
    assert loads.aerodynamic.drag_coefficient > 0.0
    comparison = compare_aircraft_aerodynamic_backends(
        analytic, table_configuration.aerodynamic_table_path
    )
    assert comparison.conditions_mach_alpha_beta.shape == (45, 3)
    assert comparison.table_sha256
    report = write_aircraft_aerodynamic_comparison(comparison, tmp_path / "aero.json")
    assert report.is_file()
    assert comparison.summary()["sample_count"] == 45
