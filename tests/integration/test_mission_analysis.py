"""Integrated public-safe orbit-design, geometry, ephemeris, and burn workflow."""

import numpy as np
import pytest

from aerognc.astrodynamics.ephemeris_provider import TabulatedEphemerisProvider
from aerognc.astrodynamics.finite_burn import execute_two_body_finite_burn
from aerognc.astrodynamics.geometry import SphericalGroundStation, ground_station_access
from aerognc.astrodynamics.kepler import propagate_universal
from aerognc.astrodynamics.maneuvers import FiniteBurn
from aerognc.astrodynamics.multirevolution import search_lambert_transfers
from aerognc.mathematics.geodesy import dcm_inertial_to_ecef

MU_EARTH_M3_S2 = 3.986004418e14


def test_verified_transfer_visibility_ephemeris_and_finite_burn_workflow() -> None:
    radius_m = 7.0e6
    circular_speed_mps = np.sqrt(MU_EARTH_M3_S2 / radius_m)
    period_s = 2.0 * np.pi * np.sqrt(radius_m**3 / MU_EARTH_M3_S2)
    transfer_time_s = 1.25 * period_s
    departure_position_m = np.array([radius_m, 0.0, 0.0])
    arrival_position_m = np.array([0.0, radius_m, 0.0])

    search = search_lambert_transfers(
        departure_position_m,
        [0.0, circular_speed_mps, 0.0],
        arrival_position_m,
        [-circular_speed_mps, 0.0, 0.0],
        transfer_time_s,
        MU_EARTH_M3_S2,
        revolutions=(0, 1),
        directions=("prograde", "retrograde"),
        endpoint_tolerance_m=0.1,
    )
    selected = search.best.solution
    endpoint = propagate_universal(
        departure_position_m,
        selected.departure_velocity_mps,
        transfer_time_s,
        MU_EARTH_M3_S2,
    )

    assert search.attempted_geometry_count == 4
    assert selected.revolutions == 1
    assert selected.endpoint_position_error_m < 0.1
    np.testing.assert_allclose(endpoint.position_m, arrival_position_m, atol=0.1)

    time_s = np.linspace(0.0, transfer_time_s, 241)
    inertial_state = np.vstack(
        [
            np.concatenate(
                (
                    propagated.position_m,
                    propagated.velocity_mps,
                )
            )
            for propagated in (
                propagate_universal(
                    departure_position_m,
                    selected.departure_velocity_mps,
                    float(epoch_s),
                    MU_EARTH_M3_S2,
                )
                for epoch_s in time_s
            )
        ]
    )
    ephemeris = TabulatedEphemerisProvider(
        time_s,
        {"ResearchOrbiter": inertial_state},
        frame="TEST_ECI",
        center="FictionalEarth",
        time_system="TT",
        source="integrated deterministic mission fixture",
    )
    middle = ephemeris.state("ResearchOrbiter", float(time_s[120]))
    np.testing.assert_allclose(middle.position_m, inertial_state[120, :3])

    rotation_rate_radps = 7.292115e-5
    body_fixed_position_m = np.vstack(
        [
            dcm_inertial_to_ecef(rotation_rate_radps * epoch_s) @ position
            for epoch_s, position in zip(time_s, inertial_state[:, :3], strict=True)
        ]
    )
    access = ground_station_access(
        time_s,
        body_fixed_position_m,
        SphericalGroundStation(
            "Fictional equatorial station",
            0.0,
            0.0,
            0.0,
            6.371e6,
            minimum_elevation_rad=np.deg2rad(5.0),
            frame="TEST_ECEF",
        ),
        frame="TEST_ECEF",
    )
    assert access.access_intervals_s
    assert len(access.crossings) >= 2

    burn = FiniteBurn(
        "departure trim",
        start_time_s=10.0,
        duration_s=5.0,
        thrust_n=100.0,
        direction=(0.0, 1.0, 0.0),
        frame="inertial",
        specific_impulse_s=300.0,
    )
    execution = execute_two_body_finite_burn(
        np.concatenate((departure_position_m, [0.0, circular_speed_mps, 0.0], [100.0])),
        burn,
        gravitational_parameter_m3_s2=MU_EARTH_M3_S2,
        dry_mass_kg=50.0,
        end_time_s=30.0,
    )
    assert [event.name for event in execution.events] == ["burn_start", "burn_end"]
    assert execution.mass_balance_error_kg == pytest.approx(0.0, abs=1.0e-10)
    assert execution.state[-1, 6] >= 50.0
