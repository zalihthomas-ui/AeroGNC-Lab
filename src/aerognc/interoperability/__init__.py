"""Optional standards/tool boundaries kept outside the numerical core."""

from aerognc.interoperability.ccsds_aem import (
    AemMetadata,
    parse_aem_kvn,
    write_aem_kvn,
)
from aerognc.interoperability.ccsds_oem import OemMetadata, parse_oem_kvn, write_oem_kvn
from aerognc.interoperability.ccsds_opm import OpmMetadata, OpmState, parse_opm_kvn, write_opm_kvn
from aerognc.interoperability.ccsds_tdm import (
    TdmMetadata,
    TdmObservation,
    parse_tdm_kvn,
    write_tdm_kvn,
)
from aerognc.interoperability.external_tools import (
    ExternalToolStatus,
    compare_gmat_report,
    detect_external_astrodynamics_tools,
    write_gmat_two_body_script,
)
from aerognc.interoperability.fmi_interface import (
    FmiFloat64Variable,
    attitude_controller_variables,
    build_fmi_controller_model_description,
    validate_fmi_controller_model_description,
    write_fmi_controller_interface,
)

__all__ = [
    "AemMetadata",
    "ExternalToolStatus",
    "FmiFloat64Variable",
    "OemMetadata",
    "OpmMetadata",
    "OpmState",
    "TdmMetadata",
    "TdmObservation",
    "attitude_controller_variables",
    "build_fmi_controller_model_description",
    "compare_gmat_report",
    "detect_external_astrodynamics_tools",
    "parse_aem_kvn",
    "parse_oem_kvn",
    "parse_opm_kvn",
    "parse_tdm_kvn",
    "validate_fmi_controller_model_description",
    "write_aem_kvn",
    "write_fmi_controller_interface",
    "write_gmat_two_body_script",
    "write_oem_kvn",
    "write_opm_kvn",
    "write_tdm_kvn",
]
