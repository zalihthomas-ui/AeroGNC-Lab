"""Honest optional GMAT/SPICE discovery and independent-result comparison hooks."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class ExternalToolStatus:
    """Availability/execution state without implying a result was produced."""

    name: str
    available: bool
    executable_or_module: str | None
    executed: bool
    message: str


def detect_external_astrodynamics_tools() -> tuple[ExternalToolStatus, ExternalToolStatus]:
    """Detect GMAT executable and spiceypy; never execute either tool implicitly."""
    gmat_path = next(
        (
            path
            for candidate in ("GMAT.exe", "GMAT", "GMAT-R2022a.exe")
            if (path := shutil.which(candidate)) is not None
        ),
        None,
    )
    spice_available = importlib.util.find_spec("spiceypy") is not None
    return (
        ExternalToolStatus(
            "GMAT",
            gmat_path is not None,
            gmat_path,
            False,
            (
                "GMAT detected; generated scripts still require explicit user execution."
                if gmat_path
                else "GMAT executable not detected; interface generated but validation not run."
            ),
        ),
        ExternalToolStatus(
            "SPICE/spiceypy",
            spice_available,
            "spiceypy" if spice_available else None,
            False,
            (
                "spiceypy detected; user-supplied kernels are still required."
                if spice_available
                else "spiceypy not installed; no SPICE comparison was executed."
            ),
        ),
    )


def write_gmat_two_body_script(
    path: str | Path,
    *,
    report_filename: str = "gmat_two_body_report.txt",
    duration_s: float = 5_400.0,
    report_step_s: float = 60.0,
) -> Path:
    """Write a standalone Earth point-mass GMAT comparison script."""
    if not np.isfinite([duration_s, report_step_s]).all() or duration_s <= 0.0:
        raise ValueError("GMAT validation duration and step must be positive and finite")
    if report_step_s <= 0.0 or duration_s < report_step_s:
        raise ValueError("GMAT report step must be positive and no longer than duration")
    step_count = round(duration_s / report_step_s)
    if not np.isclose(step_count * report_step_s, duration_s, atol=1.0e-9):
        raise ValueError("GMAT duration must be an integer number of report steps")
    report_parameters = " ".join(
        (
            "AeroGNCValidationSat.ElapsedSecs",
            "AeroGNCValidationSat.EarthMJ2000Eq.X",
            "AeroGNCValidationSat.EarthMJ2000Eq.Y",
            "AeroGNCValidationSat.EarthMJ2000Eq.Z",
            "AeroGNCValidationSat.EarthMJ2000Eq.VX",
            "AeroGNCValidationSat.EarthMJ2000Eq.VY",
            "AeroGNCValidationSat.EarthMJ2000Eq.VZ",
        )
    )
    report_command = f"Report AeroGNCReport {report_parameters};"
    propagate_command = (
        "Propagate AeroGNCProp(AeroGNCValidationSat) "
        f"{{AeroGNCValidationSat.ElapsedSecs = {report_step_s:.10g}}};"
    )
    script = f"""% AeroGNC-Lab independent two-body validation interface
% Fictional civilian verification spacecraft; no operational mission data.
Create Spacecraft AeroGNCValidationSat;
GMAT AeroGNCValidationSat.DateFormat = UTCGregorian;
GMAT AeroGNCValidationSat.Epoch = '01 Jan 2026 00:00:00.000';
GMAT AeroGNCValidationSat.CoordinateSystem = EarthMJ2000Eq;
GMAT AeroGNCValidationSat.DisplayStateType = Cartesian;
GMAT AeroGNCValidationSat.X = 7000.0;
GMAT AeroGNCValidationSat.Y = 0.0;
GMAT AeroGNCValidationSat.Z = 0.0;
GMAT AeroGNCValidationSat.VX = 0.0;
GMAT AeroGNCValidationSat.VY = 7.5;
GMAT AeroGNCValidationSat.VZ = 1.0;

Create ForceModel EarthPointMass;
GMAT EarthPointMass.CentralBody = Earth;
GMAT EarthPointMass.PrimaryBodies = {{Earth}};
GMAT EarthPointMass.Drag = None;
GMAT EarthPointMass.SRP = Off;
GMAT EarthPointMass.RelativisticCorrection = Off;

Create Propagator AeroGNCProp;
GMAT AeroGNCProp.FM = EarthPointMass;
GMAT AeroGNCProp.Type = PrinceDormand78;
GMAT AeroGNCProp.InitialStepSize = 10.0;
GMAT AeroGNCProp.Accuracy = 1.0e-12;

Create ReportFile AeroGNCReport;
GMAT AeroGNCReport.Filename = '{report_filename}';
GMAT AeroGNCReport.Precision = 16;
GMAT AeroGNCReport.FixedWidth = false;
GMAT AeroGNCReport.Delimiter = ' ';

Create Variable index;

BeginMissionSequence;
{report_command}
For index = 1:{step_count};
   {propagate_command}
   {report_command}
EndFor;
"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script, encoding="utf-8")
    return output


def compare_gmat_report(
    report_path: str | Path,
    reference_time_s: npt.ArrayLike,
    reference_states_si: npt.ArrayLike,
) -> dict[str, float]:
    """Compare a user-executed GMAT report against matching SI reference states."""
    reference_time = np.asarray(reference_time_s, dtype=np.float64)
    reference_states = np.asarray(reference_states_si, dtype=np.float64)
    if reference_states.shape != (reference_time.size, 6):
        raise ValueError("GMAT reference arrays must have shapes (N,) and (N,6)")
    try:
        report = np.loadtxt(report_path, dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot parse GMAT report: {error}") from error
    if report.shape != (reference_time.size, 7):
        raise ValueError("GMAT report must contain time plus six state columns")
    if not np.allclose(report[:, 0], reference_time, atol=1.0e-8, rtol=0.0):
        raise ValueError("GMAT report epochs do not match the reference samples")
    gmat_states_si = report[:, 1:] * 1_000.0
    position_error_m = np.linalg.norm(gmat_states_si[:, :3] - reference_states[:, :3], axis=1)
    velocity_error_mps = np.linalg.norm(gmat_states_si[:, 3:] - reference_states[:, 3:], axis=1)
    return {
        "maximum_position_error_m": float(np.max(position_error_m)),
        "rms_position_error_m": float(np.sqrt(np.mean(position_error_m**2))),
        "maximum_velocity_error_mps": float(np.max(velocity_error_mps)),
        "rms_velocity_error_mps": float(np.sqrt(np.mean(velocity_error_mps**2))),
    }
