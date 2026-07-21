import csv
from pathlib import Path

import numpy as np
import pytest

from aerognc.vehicle.aero_database import (
    COEFFICIENT_NAMES,
    AerodynamicCondition,
    TabulatedAerodynamicDatabase,
)
from aerognc.vehicle.aerodynamics import AerodynamicModel


def _coefficient_values(
    mach_axis: np.ndarray,
    alpha_axis: np.ndarray,
    beta_axis: np.ndarray,
) -> dict[str, np.ndarray]:
    mach, alpha, beta = np.meshgrid(mach_axis, alpha_axis, beta_axis, indexing="ij")
    return {
        "drag": 0.4 + 0.1 * mach + 0.2 * alpha,
        "side": -1.5 * beta + 0.0 * mach,
        "normal": -2.4 * alpha + 0.0 * beta,
        "roll": 0.01 * mach - 0.05 * beta,
        "pitch": -1.8 * alpha + 0.0 * mach,
        "yaw": 1.5 * beta + 0.0 * alpha,
    }


def _database(out_of_range: str = "error") -> TabulatedAerodynamicDatabase:
    axes = (
        np.array([0.0, 1.0, 2.0]),
        np.array([-0.2, 0.0, 0.2]),
        np.array([-0.15, 0.0, 0.15]),
    )
    return TabulatedAerodynamicDatabase(
        axis_names=("mach", "alpha_rad", "beta_rad"),
        axes=axes,
        coefficient_values=_coefficient_values(*axes),
        out_of_range=out_of_range,  # type: ignore[arg-type]
    )


def test_database_interpolation_and_exact_jacobian() -> None:
    database = _database()
    condition = AerodynamicCondition(mach=0.7, alpha_rad=0.08, beta_rad=-0.04)
    coefficients = database.evaluate(condition)
    assert coefficients.drag == pytest.approx(0.4 + 0.07 + 0.016)
    assert coefficients.side == pytest.approx(0.06)
    assert coefficients.normal == pytest.approx(-0.192)
    assert coefficients.roll == pytest.approx(0.009)
    assert coefficients.pitch == pytest.approx(-0.144)
    assert coefficients.yaw == pytest.approx(-0.06)
    axis_names, jacobian = database.coefficient_jacobian(condition)
    assert axis_names == ("mach", "alpha_rad", "beta_rad")
    assert jacobian == pytest.approx(
        np.array(
            [
                [0.1, 0.2, 0.0],
                [0.0, 0.0, -1.5],
                [0.0, -2.4, 0.0],
                [0.01, 0.0, -0.05],
                [0.0, -1.8, 0.0],
                [0.0, 0.0, 1.5],
            ]
        ),
        abs=1.0e-14,
    )


def test_database_domain_diagnostics_and_aerodynamic_load_adapter() -> None:
    database = _database("clamp")
    outside = AerodynamicCondition(mach=3.0, alpha_rad=0.0, beta_rad=-0.3)
    diagnostics = database.diagnostics(outside)
    assert diagnostics.outside_axes == ("mach", "beta_rad")
    assert not diagnostics.inside_domain
    model = AerodynamicModel(
        reference_area_m2=0.03,
        reference_length_m=2.5,
        coefficient_provider=database,
    )
    loads = model.loads(
        [120.0, -4.0, 8.0],
        density_kgpm3=1.1,
        speed_of_sound_mps=340.0,
    )
    assert loads.coefficients.drag > 0.0
    assert np.dot(loads.force_body_n, np.array([120.0, -4.0, 8.0])) < 0.0
    assert loads.coefficients.pitch < 0.0


def _write_csv(path: Path, *, omit_last: bool = False) -> None:
    axes = (
        np.array([0.0, 1.0]),
        np.array([-0.1, 0.1]),
        np.array([-0.2, 0.2]),
    )
    values = _coefficient_values(*axes)
    rows: list[dict[str, float]] = []
    for i, mach in enumerate(axes[0]):
        for j, alpha in enumerate(axes[1]):
            for k, beta in enumerate(axes[2]):
                row = {"mach": mach, "alpha_rad": alpha, "beta_rad": beta}
                row.update({name: values[name][i, j, k] for name in COEFFICIENT_NAMES})
                rows.append(row)
    if omit_last:
        rows.pop()
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["mach", "alpha_rad", "beta_rad", *COEFFICIENT_NAMES]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_csv_import_records_provenance_and_rejects_incomplete_grid(tmp_path: Path) -> None:
    complete = tmp_path / "aero.csv"
    _write_csv(complete)
    database = TabulatedAerodynamicDatabase.from_csv(complete)
    assert database.source_path == complete.resolve()
    assert database.source_sha256 is not None and len(database.source_sha256) == 64
    assert database.evaluate(
        AerodynamicCondition(mach=0.5, alpha_rad=0.0, beta_rad=0.0)
    ).drag == pytest.approx(0.45)

    incomplete = tmp_path / "incomplete.csv"
    _write_csv(incomplete, omit_last=True)
    with pytest.raises(ValueError, match="tensor-product"):
        TabulatedAerodynamicDatabase.from_csv(incomplete)


def test_database_rejects_ambiguous_axes_and_model_sources() -> None:
    values = {name: np.zeros((2, 2)) for name in COEFFICIENT_NAMES}
    with pytest.raises(ValueError, match="unique"):
        TabulatedAerodynamicDatabase(
            axis_names=("mach", "mach"),
            axes=(np.array([0.0, 1.0]), np.array([0.0, 1.0])),
            coefficient_values=values,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        AerodynamicModel(
            reference_area_m2=0.03,
            reference_length_m=2.5,
            mach_points=[0.0, 1.0],
            drag_coefficients=[0.4, 0.5],
            coefficient_provider=_database(),
        )
