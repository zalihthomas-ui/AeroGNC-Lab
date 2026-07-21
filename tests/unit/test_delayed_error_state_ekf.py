import numpy as np
import pytest

from aerognc.environment.rotating_planet import RotatingOblatePlanet
from aerognc.gnc.delayed_error_state_ekf import (
    DelayedRotatingNavigationESKF,
    InnovationGateConfiguration,
)
from aerognc.gnc.error_state_ekf import ErrorStateFilterTuning
from aerognc.gnc.strapdown_ins import ImuIncrement, RotatingNavigationState, gravity_ned_mps2
from aerognc.mathematics.geodesy import (
    GeodeticPosition,
    ReferenceEllipsoid,
    body_rotation_rate_ned,
    ecef_position_to_ned,
    geodetic_to_ecef,
)


def _planet() -> RotatingOblatePlanet:
    return RotatingOblatePlanet(
        "Orbis-A",
        ReferenceEllipsoid(6_400_000.0, 1.0 / 300.0),
        4.05e14,
        7.5e-5,
        1.1e-3,
    )


def _make_filter(
    *,
    north_error_m: float = 0.0,
    gate: InnovationGateConfiguration | None = None,
    fixed_lag_s: float = 1.5,
) -> tuple[DelayedRotatingNavigationESKF, GeodeticPosition]:
    planet = _planet()
    origin = GeodeticPosition(np.deg2rad(38.0), np.deg2rad(29.0), 850.0)
    displaced = GeodeticPosition(
        origin.latitude_rad + north_error_m / 6_400_000.0,
        origin.longitude_rad,
        origin.altitude_m,
    )
    covariance = np.diag(
        [
            *([100.0] * 3),
            *([4.0] * 3),
            *([np.deg2rad(2.0) ** 2] * 3),
            *([1.0e-6] * 3),
            *([1.0e-3] * 3),
        ]
    )
    navigation_filter = DelayedRotatingNavigationESKF(
        RotatingNavigationState(displaced, np.zeros(3), [1.0, 0.0, 0.0, 0.0]),
        covariance,
        ErrorStateFilterTuning(8.0e-4, 0.02, 2.0e-5, 2.0e-4),
        planet,
        origin,
        fixed_lag_s=fixed_lag_s,
        gate_configuration=gate
        or InnovationGateConfiguration(
            gnss_nis_threshold=1.0e6,
            barometer_nis_threshold=1.0e6,
        ),
    )
    return navigation_filter, origin


def _propagate_stationary(
    navigation_filter: DelayedRotatingNavigationESKF,
    origin: GeodeticPosition,
    duration_s: float,
    step_s: float = 0.02,
) -> None:
    planet = navigation_filter.planet
    angular_rate = body_rotation_rate_ned(origin.latitude_rad, planet.rotation_rate_radps)
    specific_force = -gravity_ned_mps2(origin, planet)
    step_count = round(duration_s / step_s)
    for _ in range(step_count):
        start_time_s = navigation_filter.current_time_s
        navigation_filter.predict(
            ImuIncrement(
                start_time_s,
                start_time_s + step_s,
                angular_rate * step_s,
                specific_force * step_s,
            )
        )


def _local_position(
    navigation_filter: DelayedRotatingNavigationESKF,
    origin: GeodeticPosition,
) -> np.ndarray:
    return ecef_position_to_ned(
        geodetic_to_ecef(
            navigation_filter.state.geodetic,
            navigation_filter.planet.ellipsoid,
        ),
        origin,
        navigation_filter.planet.ellipsoid,
    )


def test_delayed_gnss_update_replays_to_current_epoch_and_reduces_error() -> None:
    navigation_filter, origin = _make_filter(north_error_m=80.0)
    _propagate_stationary(navigation_filter, origin, 2.0)
    error_before_m = np.linalg.norm(_local_position(navigation_filter, origin))

    update = navigation_filter.update_gnss(
        np.zeros(3),
        np.zeros(3),
        np.diag([1.0, 1.0, 1.0, 0.04, 0.04, 0.04]),
        sample_time_s=1.5,
    )

    assert update.accepted
    assert update.replayed_step_count == 25
    assert navigation_filter.current_time_s == pytest.approx(2.0)
    assert np.linalg.norm(_local_position(navigation_filter, origin)) < 0.02 * error_before_m
    assert np.min(np.linalg.eigvalsh(navigation_filter.covariance)) >= -1.0e-10


def test_out_of_sequence_replay_retains_later_accepted_measurements() -> None:
    navigation_filter, origin = _make_filter(north_error_m=60.0)
    _propagate_stationary(navigation_filter, origin, 2.0)
    first = navigation_filter.update_gnss(
        np.zeros(3),
        np.zeros(3),
        np.diag([0.5, 0.5, 0.5, 0.02, 0.02, 0.02]),
        sample_time_s=1.5,
    )
    north_after_gnss_m = abs(_local_position(navigation_filter, origin)[0])
    second = navigation_filter.update_barometric_altitude(
        origin.altitude_m,
        0.25,
        sample_time_s=1.0,
    )

    assert first.accepted and second.accepted
    assert second.replayed_step_count == 50
    assert abs(_local_position(navigation_filter, origin)[0]) <= north_after_gnss_m + 0.05
    assert navigation_filter.sensor_integrity("gnss").accepted_count == 1
    assert navigation_filter.sensor_integrity("barometer").accepted_count == 1


def test_nis_gate_drives_degraded_failed_and_recovered_health_states() -> None:
    gate = InnovationGateConfiguration(
        gnss_nis_threshold=1.0,
        barometer_nis_threshold=1.0,
        degraded_after_rejections=2,
        failed_after_rejections=3,
    )
    navigation_filter, origin = _make_filter(gate=gate)
    _propagate_stationary(navigation_filter, origin, 0.2)
    covariance = np.eye(6)

    results = [
        navigation_filter.update_gnss(
            [1000.0, -1000.0, 800.0],
            [100.0, 100.0, -100.0],
            covariance,
            sample_time_s=0.2,
        )
        for _ in range(3)
    ]
    recovered = navigation_filter.update_gnss(
        np.zeros(3),
        np.zeros(3),
        covariance,
        sample_time_s=0.2,
    )

    assert [result.health for result in results] == ["healthy", "degraded", "failed"]
    assert all(not result.accepted and result.nis > result.threshold for result in results)
    assert recovered.accepted and recovered.health == "healthy"
    summary = navigation_filter.sensor_integrity("gnss")
    assert summary.accepted_count == 1
    assert summary.rejected_count == 3
    assert summary.consecutive_rejections == 0


def test_measurement_older_than_fixed_lag_is_rejected_without_health_penalty() -> None:
    navigation_filter, origin = _make_filter(fixed_lag_s=0.5)
    _propagate_stationary(navigation_filter, origin, 1.0)
    result = navigation_filter.update_barometric_altitude(
        origin.altitude_m,
        1.0,
        sample_time_s=0.1,
    )
    assert not result.accepted
    assert np.isnan(result.nis)
    assert result.reason == "outside fixed-lag history"
    assert navigation_filter.sensor_integrity("barometer").rejected_count == 0
