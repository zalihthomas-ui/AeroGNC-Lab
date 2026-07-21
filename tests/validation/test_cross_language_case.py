from pathlib import Path

import numpy as np
import pytest

from aerognc.verification.cross_language import (
    analytical_constant_acceleration,
    compare_cross_language_results,
    load_constant_acceleration_case,
    read_state_csv,
    simulate_constant_acceleration,
    write_state_csv,
)


@pytest.mark.validation
def test_python_cross_language_case_matches_analytical_solution(tmp_path: Path) -> None:
    case = load_constant_acceleration_case(
        Path("matlab_validation/constant_acceleration_case.json")
    )
    result = simulate_constant_acceleration(case)
    exact = analytical_constant_acceleration(case, result.time_s)

    assert np.max(np.abs(result.state - exact)) < case.absolute_tolerance
    comparison = compare_cross_language_results(case, result)
    assert comparison.passed
    assert comparison.matlab_analytic_max_abs_error is None

    output = write_state_csv(tmp_path / "state.csv", result.time_s, result.state)
    restored_time, restored_state = read_state_csv(output)
    np.testing.assert_array_equal(restored_time, result.time_s)
    np.testing.assert_array_equal(restored_state, result.state)
