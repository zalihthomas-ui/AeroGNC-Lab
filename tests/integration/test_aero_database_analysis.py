import json
from pathlib import Path

import pytest

from aerognc.configuration import load_three_dof_configuration
from aerognc.vehicle.aero_database import AerodynamicCondition, TabulatedAerodynamicDatabase
from aerognc.verification.aero_database import (
    analyze_aerodynamic_database,
    write_aerodynamic_database_analysis,
)
from aerognc.visualisation.aero_database import plot_aerodynamic_database


def test_aerodynamic_database_analysis_writes_derivatives_and_figure(tmp_path: Path) -> None:
    configuration = load_three_dof_configuration("configs/three_dof_aero_database.yaml")
    condition = AerodynamicCondition(mach=0.8, alpha_rad=0.0, beta_rad=0.0)
    analysis = analyze_aerodynamic_database(configuration, condition)
    assert analysis.nominal_coefficients["drag"] == pytest.approx(0.628)
    assert analysis.coefficient_jacobian["normal"]["alpha_rad"] == pytest.approx(-2.4)
    assert analysis.coefficient_jacobian["yaw"]["beta_rad"] == pytest.approx(1.5)
    assert analysis.source_sha256 is not None

    report = write_aerodynamic_database_analysis(analysis, tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["interpolation_policy"] == "clamp"
    provider = configuration.vehicle.aerodynamics.coefficient_provider
    assert isinstance(provider, TabulatedAerodynamicDatabase)
    figure = plot_aerodynamic_database(provider, tmp_path, condition)
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert figure.stat().st_size > 10_000


def test_aerodynamic_database_analysis_rejects_legacy_vehicle() -> None:
    configuration = load_three_dof_configuration("configs/three_dof_nominal.yaml")
    with pytest.raises(ValueError, match="does not use"):
        analyze_aerodynamic_database(configuration)
