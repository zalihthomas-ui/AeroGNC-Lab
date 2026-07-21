"""Independent analytic-versus-static-table comparison for the fictional aircraft."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerognc.configuration.aircraft_loader import AircraftSandboxConfiguration
from aerognc.mathematics.vectors import FloatArray
from aerognc.vehicle.aero_database import TabulatedAerodynamicDatabase
from aerognc.vehicle.fixed_wing import aerodynamic_state

AIRCRAFT_COEFFICIENT_NAMES = ("CL", "CD", "CY", "Cl", "Cm", "Cn")


@dataclass(frozen=True, slots=True)
class AircraftAerodynamicComparison:
    """Matched conditions and coefficient differences for two model backends."""

    conditions_mach_alpha_beta: FloatArray
    analytic_coefficients: FloatArray
    table_coefficients: FloatArray
    table_source: Path
    table_sha256: str

    @property
    def differences(self) -> FloatArray:
        """Return table minus analytic coefficients at each matched condition."""
        return np.asarray(self.table_coefficients - self.analytic_coefficients)

    def summary(self) -> dict[str, object]:
        """Return coefficient-wise RMS and maximum absolute differences."""
        difference = self.differences
        return {
            "schema_version": "1.0",
            "scope": "synthetic fictional civilian aircraft static aerodynamics",
            "sample_count": int(difference.shape[0]),
            "table_source": str(self.table_source),
            "table_sha256": self.table_sha256,
            "metrics": {
                name: {
                    "rms_difference": float(np.sqrt(np.mean(difference[:, index] ** 2))),
                    "maximum_absolute_difference": float(np.max(np.abs(difference[:, index]))),
                }
                for index, name in enumerate(AIRCRAFT_COEFFICIENT_NAMES)
            },
            "interpretation": (
                "The table is an optional synthetic static baseline. Configured rate and "
                "control derivatives are added identically by both flight-model backends."
            ),
        }


def compare_aircraft_aerodynamic_backends(
    configuration: AircraftSandboxConfiguration,
    table_path: str | Path,
) -> AircraftAerodynamicComparison:
    """Compare matched static points using direct coefficient calculations."""
    database = TabulatedAerodynamicDatabase.from_csv(table_path, out_of_range="clamp")
    if set(database.axis_names) != {"mach", "alpha_rad", "beta_rad"}:
        raise ValueError("aircraft static comparison table must use Mach, alpha, and beta axes")
    mach_values = np.array([0.2, 0.5, 0.8])
    alpha_values = np.deg2rad(np.array([-15.0, -7.5, 0.0, 7.5, 15.0]))
    beta_values = np.deg2rad(np.array([-5.0, 0.0, 5.0]))
    conditions: list[list[float]] = []
    analytic: list[list[float]] = []
    tabular: list[list[float]] = []
    for mach in mach_values:
        for alpha in alpha_values:
            for beta in beta_values:
                airspeed = 340.0 * mach
                velocity = airspeed * np.array(
                    [np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)]
                )
                common = (
                    velocity,
                    np.zeros(3),
                    np.zeros(3),
                    1.0,
                    340.0,
                    configuration,
                )
                analytic_state = aerodynamic_state(*common)
                table_state = aerodynamic_state(*common, coefficient_database=database)
                conditions.append([mach, alpha, beta])
                analytic.append(
                    [
                        analytic_state.lift_coefficient,
                        analytic_state.drag_coefficient,
                        analytic_state.side_force_coefficient,
                        analytic_state.roll_moment_coefficient,
                        analytic_state.pitch_moment_coefficient,
                        analytic_state.yaw_moment_coefficient,
                    ]
                )
                tabular.append(
                    [
                        table_state.lift_coefficient,
                        table_state.drag_coefficient,
                        table_state.side_force_coefficient,
                        table_state.roll_moment_coefficient,
                        table_state.pitch_moment_coefficient,
                        table_state.yaw_moment_coefficient,
                    ]
                )
    if database.source_path is None or database.source_sha256 is None:  # pragma: no cover
        raise RuntimeError("CSV aerodynamic database did not retain provenance")
    return AircraftAerodynamicComparison(
        np.asarray(conditions, dtype=np.float64),
        np.asarray(analytic, dtype=np.float64),
        np.asarray(tabular, dtype=np.float64),
        database.source_path,
        database.source_sha256,
    )


def write_aircraft_aerodynamic_comparison(
    comparison: AircraftAerodynamicComparison,
    path: str | Path,
) -> Path:
    """Write the complete provenance-tagged comparison summary as JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(comparison.summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
