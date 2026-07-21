from pathlib import Path

import numpy as np
import pytest

from aerognc.astrodynamics.patched_conics import hyperbolic_patch, plan_orbit_assisted_tour
from aerognc.configuration.planetary_catalog import load_planetary_catalog

CATALOG_PATH = Path("configs/fictional_planetary_system.yaml")


def test_hyperbolic_soi_patch_matches_vis_viva_and_time_equations() -> None:
    mu_m3_s2 = 3.9860044e14
    excess_speed_mps = 4_200.0
    periapsis_radius_m = 6_700_000.0
    soi_radius_m = 9.25e8

    patch = hyperbolic_patch(excess_speed_mps, mu_m3_s2, periapsis_radius_m, soi_radius_m)

    assert patch.semi_major_axis_m == pytest.approx(-mu_m3_s2 / excess_speed_mps**2)
    assert patch.eccentricity == pytest.approx(
        1.0 + periapsis_radius_m * excess_speed_mps**2 / mu_m3_s2
    )
    assert patch.periapsis_speed_mps**2 == pytest.approx(
        excess_speed_mps**2 + 2.0 * mu_m3_s2 / periapsis_radius_m
    )
    assert patch.sphere_of_influence_speed_mps**2 == pytest.approx(
        excess_speed_mps**2 + 2.0 * mu_m3_s2 / soi_radius_m
    )
    assert 0.0 < patch.true_anomaly_at_soi_rad < patch.asymptote_true_anomaly_rad < np.pi
    assert patch.time_periapsis_to_soi_s > 0.0


def test_orbit_assisted_tour_charges_capture_alignment_and_departure() -> None:
    catalog = load_planetary_catalog(CATALOG_PATH)
    tour = plan_orbit_assisted_tour(
        catalog.body("Asteria"),
        catalog.body("Neria"),
        catalog.body("Caelus"),
        catalog.primary.gravitational_parameter_m3_s2,
        catalog.primary.gravitational_parameter_m3_s2 / 6.67430e-11,
        0.0,
        240.0 * 86_400.0,
        2035.4 * 86_400.0,
        departure_parking_altitude_m=300_000.0,
        assist_parking_altitude_m=300_000.0,
        destination_parking_altitude_m=300_000.0,
        dwell_revolutions=2,
        initial_mass_kg=130_000.0,
        dry_mass_kg=8_000.0,
        specific_impulse_s=1_200.0,
    )

    assert tour.feasible
    assert len(tour.burns) == 5
    assert [burn.name for burn in tour.burns] == [
        "departure_parking_orbit_injection",
        "assist_orbit_capture",
        "assist_orbit_alignment",
        "assist_periapsis_departure",
        "destination_orbit_capture",
    ]
    assert all(
        later.mass_after_kg < earlier.mass_after_kg
        for earlier, later in zip(tour.burns[:-1], tour.burns[1:], strict=True)
    )
    assert tour.assist_departure_time_s - tour.assist_arrival_time_s == pytest.approx(
        2.0 * tour.assist_orbit_period_s
    )
    assert tour.total_delta_v_mps == pytest.approx(sum(burn.delta_v_mps for burn in tour.burns))
    assert tour.alignment_delta_v_mps > tour.assist_departure_patch.excess_speed_mps
    assert tour.departure_oberth_energy_gain_jpkg > 0.0


def test_patched_conic_validation_rejects_nonphysical_soi() -> None:
    with pytest.raises(ValueError, match="outside"):
        hyperbolic_patch(2_000.0, 1.0e14, 7.0e6, 6.0e6)
