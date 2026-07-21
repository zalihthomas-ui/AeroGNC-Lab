import json
import xml.etree.ElementTree as ET

import pytest

from aerognc.interoperability.fmi_interface import (
    attitude_controller_variables,
    build_fmi_controller_model_description,
    validate_fmi_controller_model_description,
    write_fmi_controller_interface,
)


def test_fmi_controller_contract_has_unique_typed_inputs_parameters_and_outputs() -> None:
    variables = attitude_controller_variables()

    assert len(variables) == 24
    assert len({variable.name for variable in variables}) == len(variables)
    assert len({variable.value_reference for variable in variables}) == len(variables)
    assert sum(variable.causality == "input" for variable in variables) == 11
    assert sum(variable.causality == "parameter" for variable in variables) == 9
    assert sum(variable.causality == "output" for variable in variables) == 3


def test_fmi_controller_model_description_round_trip_and_status_are_honest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    xml_text = build_fmi_controller_model_description()
    validate_fmi_controller_model_description(xml_text)
    root = ET.fromstring(xml_text)
    assert root.attrib["fmiVersion"] == "3.0"
    assert root.attrib["version"] == "0.8.0"
    assert root.find("CoSimulation") is not None
    assert len(root.findall("./ModelStructure/Output")) == 3

    description, status = write_fmi_controller_interface(tmp_path)
    assert description.read_text("utf-8") == xml_text
    payload = json.loads(status.read_text("utf-8"))
    assert payload["fmu_built"] is False
    assert payload["fmu_executed"] is False
    assert payload["official_xsd_validation_executed"] is False


def test_fmi_contract_validator_rejects_modified_version() -> None:
    modified = build_fmi_controller_model_description().replace(
        'fmiVersion="3.0"',
        'fmiVersion="2.0"',
        1,
    )
    with pytest.raises(ValueError, match=r"FMI 3\.0"):
        validate_fmi_controller_model_description(modified)
