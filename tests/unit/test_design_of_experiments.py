import numpy as np
import pytest

from aerognc.verification.design_of_experiments import (
    Factor,
    bootstrap_confidence_interval,
    latin_hypercube_design,
    morris_design,
    morris_elementary_effects,
    sensitivity_correlations,
    sobol_design,
    validate_samples_in_domain,
)


def _factors() -> tuple[Factor, Factor]:
    return Factor("mass", 10.0, 20.0, "kg"), Factor("wind", -1.0, 1.0, "m/s")


def test_seeded_latin_hypercube_occupies_every_stratum_once() -> None:
    first = latin_hypercube_design(_factors(), 16, seed=72)
    second = latin_hypercube_design(_factors(), 16, seed=72)

    np.testing.assert_array_equal(first.samples, second.samples)
    for column in range(2):
        strata = np.floor(first.unit_samples[:, column] * 16).astype(int)
        np.testing.assert_array_equal(np.sort(strata), np.arange(16))
    assert np.all((first.samples[:, 0] >= 10.0) & (first.samples[:, 0] <= 20.0))


def test_direct_sobol_first_points_match_checkable_gray_code_sequence() -> None:
    design = sobol_design(_factors(), 4)
    np.testing.assert_allclose(
        design.unit_samples,
        [[0.0, 0.0], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75]],
        atol=0.0,
    )
    skipped = sobol_design(_factors(), 2, skip=2)
    np.testing.assert_array_equal(skipped.unit_samples, design.unit_samples[2:])


def test_morris_recovers_linear_elementary_effects() -> None:
    design = morris_design(_factors(), 8, levels=4, seed=81)
    response = 2.0 * design.design.unit_samples[:, 0] - 3.0 * design.design.unit_samples[:, 1]
    effects = morris_elementary_effects(design, response)

    assert effects[0].mean == pytest.approx(2.0)
    assert effects[0].mean_absolute == pytest.approx(2.0)
    assert effects[1].mean == pytest.approx(-3.0)
    assert effects[1].mean_absolute == pytest.approx(3.0)
    assert effects[0].standard_deviation == pytest.approx(0.0, abs=1.0e-14)


def test_correlations_and_seeded_bootstrap_cover_small_analytical_cases() -> None:
    samples = np.array([[10.0, -1.0], [12.0, 0.5], [15.0, -0.5], [20.0, 1.0]])
    response = samples[:, 0]
    correlations = sensitivity_correlations(samples, response, _factors())
    assert correlations[0].linear == pytest.approx(1.0)
    assert correlations[0].rank == pytest.approx(1.0)

    first = bootstrap_confidence_interval([1.0, 2.0, 3.0, 4.0], seed=9, resamples=500)
    second = bootstrap_confidence_interval([1.0, 2.0, 3.0, 4.0], seed=9, resamples=500)
    assert first == second
    assert first.lower <= first.estimate <= first.upper
    assert first.estimate == pytest.approx(2.5)


def test_design_domain_and_configuration_validation() -> None:
    with pytest.raises(ValueError, match="outside"):
        validate_samples_in_domain([[9.0, 0.0]], _factors())
    with pytest.raises(ValueError, match="even"):
        morris_design(_factors(), 2, levels=5, seed=1)
    with pytest.raises(ValueError, match="16"):
        sobol_design(tuple(Factor(str(index), 0.0, 1.0) for index in range(17)), 2)
