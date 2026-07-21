from pathlib import Path

import numpy as np
import pytest

from aerognc.vehicle.engineering_data import (
    import_aerodynamic_csv,
    import_mass_properties_csv,
    import_thrust_csv,
)


def test_thrust_import_requires_units_monotonicity_and_records_hash(tmp_path: Path) -> None:
    path = tmp_path / "thrust.csv"
    path.write_text("time_s,thrust_n\n0,0\n0.5,100\n1,0\n", encoding="utf-8")
    imported = import_thrust_csv(path, propellant_mass_kg=0.5)

    assert imported.curve.thrust_at_time_n(0.5) == pytest.approx(100.0)
    assert imported.curve.thrust_at_time_n(2.0) == 0.0
    assert len(imported.provenance.source_sha256) == 64
    assert imported.provenance.extrapolation == "zero"

    bad = tmp_path / "ambiguous.csv"
    bad.write_text("time,thrust\n0,0\n1,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unit-bearing"):
        import_thrust_csv(bad, propellant_mass_kg=1.0)


def test_mass_property_import_interpolates_tensor_and_enforces_policy(tmp_path: Path) -> None:
    path = tmp_path / "mass.csv"
    path.write_text(
        "time_s,mass_kg,cg_from_nose_m,inertia_xx_kgm2,inertia_yy_kgm2,"
        "inertia_zz_kgm2,inertia_xy_kgm2,inertia_xz_kgm2,inertia_yz_kgm2\n"
        "0,10,1.2,2,3,4,0,0,0\n"
        "2,8,1.0,1,2,3,0,0,0\n",
        encoding="utf-8",
    )
    table = import_mass_properties_csv(path)
    sample = table.at_time(1.0)

    assert sample.mass_kg == pytest.approx(9.0)
    assert sample.centre_of_gravity_from_nose_m == pytest.approx(1.1)
    np.testing.assert_allclose(np.diag(sample.inertia_body_kgm2), [1.5, 2.5, 3.5])
    with pytest.raises(ValueError, match="outside"):
        table.at_time(3.0)
    clamped = import_mass_properties_csv(path, out_of_range="clamp")
    assert clamped.at_time(3.0).mass_kg == pytest.approx(8.0)


def test_aerodynamic_import_requires_canonical_axes_and_coefficients(tmp_path: Path) -> None:
    path = tmp_path / "aero.csv"
    path.write_text(
        "mach,drag,side,normal,roll,pitch,yaw\n0,0.4,0,0,0,0,0\n1,0.5,0,0,0,0,0\n",
        encoding="utf-8",
    )
    imported = import_aerodynamic_csv(path, out_of_range="clamp")
    coefficients = imported.database.coefficients(0.5, 0.0, 0.0)
    assert coefficients.drag == pytest.approx(0.45)
    assert imported.provenance.interpolation == "multilinear"

    ambiguous = tmp_path / "bad_aero.csv"
    ambiguous.write_text(
        "alpha_deg,drag,side,normal,roll,pitch,yaw\n0,0.4,0,0,0,0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unit-ambiguous"):
        import_aerodynamic_csv(ambiguous)
