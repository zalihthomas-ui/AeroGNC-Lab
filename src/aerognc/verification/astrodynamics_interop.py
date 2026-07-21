"""Standards export and truthful external-tool status for an orbit-tour result."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from aerognc.astrodynamics.reference_frames import transform_inertial_state
from aerognc.astrodynamics.time_systems import (
    IERS_BULLETIN_C_DATE,
    IERS_BULLETIN_C_NUMBER,
    IERS_BULLETIN_C_URL,
    LEAP_SECOND_TABLE_VALID_UNTIL,
    utc_to_time_scales,
)
from aerognc.interoperability.ccsds_oem import OemMetadata, write_oem_kvn
from aerognc.interoperability.external_tools import (
    detect_external_astrodynamics_tools,
    write_gmat_two_body_script,
)
from aerognc.simulation.orbit_assisted_tour import OrbitTourSimulation

CCSDS_ODM_SOURCE_URL = "https://ccsds.org/publications/allpubs/"
NAIF_TIME_SOURCE_URL = "https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html"


def write_astrodynamics_interoperability(
    simulation: OrbitTourSimulation,
    output_directory: str | Path,
) -> tuple[Path, Path, Path]:
    """Write OEM, unexecuted GMAT script, and external-tool/provenance status."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    columns = simulation.result.columns
    ecliptic_states = np.column_stack(
        (
            columns["position_x_m"],
            columns["position_y_m"],
            columns["position_z_m"],
            columns["velocity_x_mps"],
            columns["velocity_y_mps"],
            columns["velocity_z_mps"],
        )
    )
    j2000_states = np.vstack(
        [transform_inertial_state(state, "HELIOS_ECLIPJ2000", "J2000") for state in ecliptic_states]
    )
    metadata = OemMetadata(
        creation_date=datetime(2026, 7, 19),
        originator="AEROGNC-LAB",
        message_id="AEROGNC-SYNTHETIC-ORBIT-TOUR-001",
        object_name=simulation.configuration.spacecraft_name,
        object_id="FICTIONAL-SELENE-001",
        center_name="HELIOS",
        reference_frame="J2000",
        time_system="TDB",
        start_epoch=datetime(2035, 1, 1),
    )
    oem_path = write_oem_kvn(
        simulation.result.time_s,
        j2000_states,
        metadata,
        output / "orbit_assisted_tour.oem",
    )
    gmat_script_path = write_gmat_two_body_script(output / "gmat_two_body_validation.script")
    tool_status = detect_external_astrodynamics_tools()
    release_time = utc_to_time_scales(datetime(2026, 7, 19, tzinfo=UTC))
    status_payload = {
        "standards": {
            "ccsds_oem": {
                "standard": "CCSDS 502.0-B-3 Orbit Data Messages, OEM/KVN 3.0",
                "source_url": CCSDS_ODM_SOURCE_URL,
                "file": oem_path.name,
                "internal_units": "SI m and m/s",
                "file_units": "standard-mandated km and km/s",
                "frame_conversion": "HELIOS_ECLIPJ2000 to fixed J2000 mean-equator rotation",
                "scope": "fictional center/object identifiers; engineering interoperability",
            },
            "time": {
                "iers_bulletin": IERS_BULLETIN_C_NUMBER,
                "bulletin_date": IERS_BULLETIN_C_DATE,
                "source_url": IERS_BULLETIN_C_URL,
                "table_valid_until": LEAP_SECOND_TABLE_VALID_UNTIL,
                "tai_minus_utc_s": release_time.tai_minus_utc_s,
                "tt_minus_tai_s": 32.184,
                "tdb_approximation_source_url": NAIF_TIME_SOURCE_URL,
            },
        },
        "external_tools": [
            {
                "name": status.name,
                "available": status.available,
                "executable_or_module": status.executable_or_module,
                "executed": status.executed,
                "message": status.message,
            }
            for status in tool_status
        ],
        "claims": {
            "ccsds_file_written": True,
            "gmat_script_written": True,
            "gmat_validation_executed": False,
            "spice_validation_executed": False,
        },
        "next_steps": [
            "Run the generated GMAT script explicitly, retain its version/log/report, "
            "then compare.",
            "Install spiceypy and select public kernels explicitly before any SPICE comparison.",
            "Do not treat the fictional analytical catalog as an operational ephemeris.",
        ],
    }
    status_path = output / "external_astrodynamics_status.json"
    status_path.write_text(
        json.dumps(status_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return oem_path, gmat_script_path, status_path
