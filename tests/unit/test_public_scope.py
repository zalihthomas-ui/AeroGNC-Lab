import ast
from pathlib import Path
from typing import Any

import yaml

BANNED_IDENTIFIER_PARTS = ("target", "intercept", "homing", "engagement", "warhead")


def _configuration_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [
            *(str(key) for key in value),
            *(item for child in value.values() for item in _configuration_keys(child)),
        ]
    if isinstance(value, list):
        return [item for child in value for item in _configuration_keys(child)]
    return []


def test_public_apis_and_configuration_keys_stay_inside_civilian_scope() -> None:
    identifiers: list[str] = []
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        identifiers.extend(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
    configuration_keys: list[str] = []
    for path in Path("configs").glob("*.yaml"):
        configuration_keys.extend(
            _configuration_keys(yaml.safe_load(path.read_text(encoding="utf-8")))
        )
    for identifier in (*identifiers, *configuration_keys):
        lowered = identifier.lower()
        # B-plane targeting is the standard, non-operational name for mapping a
        # civilian planetary flyby asymptote into its encounter plane.
        if "bplane" in lowered or "b_plane" in lowered:
            continue
        assert not any(part in lowered for part in BANNED_IDENTIFIER_PARTS), identifier

    safety = Path("docs/public_safety.md").read_text(encoding="utf-8").lower()
    assert "fictional" in safety and "synthetic" in safety
    assert "no classified" in safety and "no target-state input" in safety
