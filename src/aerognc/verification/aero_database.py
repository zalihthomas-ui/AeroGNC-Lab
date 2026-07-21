"""Aerodynamic database domain, coefficient, and derivative evidence."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from aerognc.configuration.models import ThreeDofConfiguration
from aerognc.vehicle.aero_database import (
    COEFFICIENT_NAMES,
    AerodynamicCondition,
    TabulatedAerodynamicDatabase,
)


@dataclass(frozen=True, slots=True)
class AerodynamicDatabaseAnalysis:
    """Serializable nominal database and local stability-derivative evidence."""

    source_path: str | None
    source_sha256: str | None
    interpolation_policy: str
    axis_bounds: dict[str, tuple[float, float]]
    nominal_condition: dict[str, float]
    nominal_coefficients: dict[str, float]
    coefficient_jacobian: dict[str, dict[str, float]]

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible analysis mapping."""
        return asdict(self)


def _database(configuration: ThreeDofConfiguration) -> TabulatedAerodynamicDatabase:
    provider = configuration.vehicle.aerodynamics.coefficient_provider
    if not isinstance(provider, TabulatedAerodynamicDatabase):
        raise ValueError("scenario vehicle does not use a tabulated aerodynamic database")
    return provider


def analyze_aerodynamic_database(
    configuration: ThreeDofConfiguration,
    condition: AerodynamicCondition | None = None,
) -> AerodynamicDatabaseAnalysis:
    """Evaluate database provenance, domain, nominal coefficients, and Jacobian."""
    database = _database(configuration)
    nominal = condition or AerodynamicCondition(mach=0.8)
    diagnostics = database.diagnostics(nominal)
    if not diagnostics.inside_domain:
        raise ValueError(
            "nominal aerodynamic condition is outside axes: " + ", ".join(diagnostics.outside_axes)
        )
    coefficients = database.evaluate(nominal)
    axis_names, jacobian = database.coefficient_jacobian(nominal)
    coefficient_mapping = {name: float(getattr(coefficients, name)) for name in COEFFICIENT_NAMES}
    jacobian_mapping = {
        coefficient_name: {
            axis_name: float(jacobian[row_index, column_index])
            for column_index, axis_name in enumerate(axis_names)
        }
        for row_index, coefficient_name in enumerate(COEFFICIENT_NAMES)
    }
    axis_bounds = {
        name: (float(axis[0]), float(axis[-1]))
        for name, axis in zip(database.axis_names, database.axes, strict=True)
    }
    nominal_mapping = nominal.as_mapping()
    return AerodynamicDatabaseAnalysis(
        source_path=str(database.source_path) if database.source_path is not None else None,
        source_sha256=database.source_sha256,
        interpolation_policy=database.out_of_range,
        axis_bounds=axis_bounds,
        nominal_condition={name: nominal_mapping[name] for name in database.axis_names},
        nominal_coefficients=coefficient_mapping,
        coefficient_jacobian=jacobian_mapping,
    )


def write_aerodynamic_database_analysis(
    analysis: AerodynamicDatabaseAnalysis,
    output_directory: str | Path,
) -> Path:
    """Write deterministic JSON analysis evidence."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "aerodynamic_database_analysis.json"
    path.write_text(
        json.dumps(analysis.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
