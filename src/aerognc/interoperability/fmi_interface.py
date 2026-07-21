"""Deterministic FMI 3.0 controller interface contract.

The generated XML describes a future Co-Simulation boundary.  It is deliberately
not packaged as an FMU because this Python project does not provide the mandatory
FMI C API binary/source implementation.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FMI_SPECIFICATION_URL = "https://fmi-standard.org/docs/3.0.2/"
MODEL_IDENTIFIER = "aerognc_attitude_controller"
INSTANTIATION_TOKEN = "{703c160d-f564-4adc-9c7f-d7eddb09521d}"

FmiCausality = Literal["independent", "input", "output", "parameter"]
FmiVariability = Literal["continuous", "discrete", "fixed"]


@dataclass(frozen=True, slots=True)
class FmiFloat64Variable:
    """One scalar Float64 interface declaration."""

    name: str
    value_reference: int
    causality: FmiCausality
    variability: FmiVariability
    unit: str | None
    description: str
    start: float | None = None


def attitude_controller_variables() -> tuple[FmiFloat64Variable, ...]:
    """Return the stable scalar contract for quaternion PD attitude control."""
    variables: list[FmiFloat64Variable] = [
        FmiFloat64Variable(
            "time",
            0,
            "independent",
            "continuous",
            "s",
            "Importer communication time",
        )
    ]
    reference_names = ("q0", "q1", "q2", "q3")
    for component, name in enumerate(reference_names, start=1):
        variables.append(
            FmiFloat64Variable(
                f"referenceQuaternionNb.{name}",
                component,
                "input",
                "discrete",
                None,
                "Scalar-first body-to-NED reference quaternion component",
                1.0 if name == "q0" else 0.0,
            )
        )
    for component, name in enumerate(reference_names, start=5):
        variables.append(
            FmiFloat64Variable(
                f"measuredQuaternionNb.{name}",
                component,
                "input",
                "discrete",
                None,
                "Scalar-first body-to-NED measured quaternion component",
                1.0 if name == "q0" else 0.0,
            )
        )
    for component, axis in enumerate("xyz", start=9):
        variables.append(
            FmiFloat64Variable(
                f"angularRateBody.{axis}",
                component,
                "input",
                "discrete",
                "rad/s",
                f"FRD body angular rate about {axis}",
                0.0,
            )
        )
    parameter_groups = (
        ("proportionalGain", 12, "N.m", (8.0, 45.0, 45.0)),
        ("rateDamping", 15, "N.m.s", (2.5, 14.0, 14.0)),
        ("momentLimit", 18, "N.m", (15.0, 50.0, 50.0)),
    )
    for group, first_reference, unit, starts in parameter_groups:
        for offset, (axis, start) in enumerate(zip("xyz", starts, strict=True)):
            variables.append(
                FmiFloat64Variable(
                    f"{group}.{axis}",
                    first_reference + offset,
                    "parameter",
                    "fixed",
                    unit,
                    f"Synthetic controller {group} about body {axis}",
                    start,
                )
            )
    for component, axis in enumerate("xyz", start=21):
        variables.append(
            FmiFloat64Variable(
                f"momentCommandBody.{axis}",
                component,
                "output",
                "discrete",
                "N.m",
                f"Limited commanded moment about FRD body {axis}",
            )
        )
    return tuple(variables)


def build_fmi_controller_model_description() -> str:
    """Build a deterministic FMI 3.0 modelDescription interface XML string."""
    root = ET.Element(
        "fmiModelDescription",
        {
            "fmiVersion": "3.0",
            "modelName": "AeroGNC.FictionalAttitudeController",
            "instantiationToken": INSTANTIATION_TOKEN,
            "description": (
                "Public-safe fictional research-rocket quaternion attitude controller interface"
            ),
            "version": "0.8.0",
            "generationTool": "AeroGNC-Lab",
            "variableNamingConvention": "structured",
        },
    )
    ET.SubElement(
        root,
        "CoSimulation",
        {
            "modelIdentifier": MODEL_IDENTIFIER,
            "canHandleVariableCommunicationStepSize": "false",
            "fixedInternalStepSize": "0.01",
        },
    )
    units = ET.SubElement(root, "UnitDefinitions")
    for name, exponents in (
        ("s", {"s": "1"}),
        ("rad/s", {"s": "-1", "rad": "1"}),
        ("N.m", {"kg": "1", "m": "2", "s": "-2"}),
        ("N.m.s", {"kg": "1", "m": "2", "s": "-1"}),
    ):
        unit = ET.SubElement(units, "Unit", {"name": name})
        ET.SubElement(unit, "BaseUnit", exponents)
    ET.SubElement(
        root,
        "DefaultExperiment",
        {"startTime": "0", "stopTime": "8", "stepSize": "0.01"},
    )
    model_variables = ET.SubElement(root, "ModelVariables")
    for variable in attitude_controller_variables():
        attributes = {
            "name": variable.name,
            "valueReference": str(variable.value_reference),
            "causality": variable.causality,
            "variability": variable.variability,
            "description": variable.description,
        }
        if variable.unit is not None:
            attributes["unit"] = variable.unit
        if variable.start is not None:
            attributes["start"] = f"{variable.start:.12g}"
        ET.SubElement(model_variables, "Float64", attributes)
    structure = ET.SubElement(root, "ModelStructure")
    dependencies = " ".join(str(value) for value in range(1, 21))
    for value_reference in range(21, 24):
        ET.SubElement(
            structure,
            "Output",
            {"valueReference": str(value_reference), "dependencies": dependencies},
        )
        ET.SubElement(
            structure,
            "InitialUnknown",
            {"valueReference": str(value_reference), "dependencies": dependencies},
        )
    ET.indent(root, space="  ")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        + ET.tostring(
            root,
            encoding="unicode",
            short_empty_elements=True,
        )
        + "\n"
    )


def validate_fmi_controller_model_description(xml_text: str) -> None:
    """Check the project contract without claiming official XSD validation."""
    root = ET.fromstring(xml_text)
    if root.tag != "fmiModelDescription" or root.get("fmiVersion") != "3.0":
        raise ValueError("controller interface must be an FMI 3.0 model description")
    co_simulation = root.find("CoSimulation")
    if co_simulation is None or co_simulation.get("modelIdentifier") != MODEL_IDENTIFIER:
        raise ValueError("controller interface CoSimulation identifier is missing or incorrect")
    declared = root.findall("./ModelVariables/Float64")
    expected = attitude_controller_variables()
    if len(declared) != len(expected):
        raise ValueError("controller interface variable count does not match the contract")
    references = [int(variable.attrib["valueReference"]) for variable in declared]
    names = [variable.attrib["name"] for variable in declared]
    if len(set(references)) != len(references) or len(set(names)) != len(names):
        raise ValueError("FMI variable names and value references must be unique")
    if tuple(names) != tuple(variable.name for variable in expected):
        raise ValueError("FMI controller variable order/names do not match the contract")
    output_references = {
        int(variable.attrib["valueReference"])
        for variable in declared
        if variable.attrib.get("causality") == "output"
    }
    structured_outputs = {
        int(output.attrib["valueReference"]) for output in root.findall("./ModelStructure/Output")
    }
    if output_references != structured_outputs:
        raise ValueError("every FMI output must appear in ModelStructure")


def write_fmi_controller_interface(output_directory: str | Path) -> tuple[Path, Path]:
    """Write the XML contract and an honest non-execution/non-FMU status record."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    xml_text = build_fmi_controller_model_description()
    validate_fmi_controller_model_description(xml_text)
    description_path = directory / "modelDescription.xml"
    description_path.write_text(xml_text, encoding="utf-8")
    status = {
        "artifact_type": "FMI 3.0 Co-Simulation interface contract only",
        "fmi_specification": FMI_SPECIFICATION_URL,
        "fmu_built": False,
        "fmu_executed": False,
        "official_xsd_validation_executed": False,
        "model_identifier": MODEL_IDENTIFIER,
        "reason": (
            "No mandatory FMI C API binary or source wrapper has been implemented; "
            "therefore this directory must not be distributed or described as an FMU."
        ),
        "variable_count": len(attitude_controller_variables()),
    }
    status_path = directory / "STATUS.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return description_path, status_path
