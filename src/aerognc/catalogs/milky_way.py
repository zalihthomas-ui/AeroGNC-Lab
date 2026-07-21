"""Validated descriptive metadata for the Milky Way context layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml


@dataclass(frozen=True, slots=True)
class MilkyWayMetadata:
    """Approximate published galaxy context, never a navigation ephemeris."""

    name: str
    morphology: str
    disk_diameter_light_year_approx: float
    star_count_lower_estimate: int
    star_count_upper_estimate: int
    solar_arm: str
    solar_galactic_orbit_period_year_approx: float
    source_urls: tuple[str, ...]
    scope_note: str


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, object], value)


def _text(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Milky Way metadata field {key} must be text")
    return value.strip()


def _positive_float(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0.0:
        raise ValueError(f"Milky Way metadata field {key} must be positive")
    return float(value)


def load_milky_way_metadata(path: Path) -> MilkyWayMetadata:
    """Load provenance-tagged approximate galaxy metadata from YAML."""
    root = _mapping(cast(object, yaml.safe_load(path.read_text(encoding="utf-8"))), "root")
    stars = _mapping(root.get("star_count_estimate"), "star_count_estimate")
    lower = int(_positive_float(stars, "lower"))
    upper = int(_positive_float(stars, "upper"))
    if upper < lower:
        raise ValueError("Milky Way star-count upper estimate must exceed lower estimate")
    sources_object = root.get("source_urls")
    if not isinstance(sources_object, list) or not sources_object:
        raise ValueError("Milky Way metadata requires at least one source URL")
    sources = tuple(str(item).strip() for item in sources_object)
    if any(not item.startswith("https://") for item in sources):
        raise ValueError("Milky Way source URLs must use HTTPS")
    return MilkyWayMetadata(
        name=_text(root, "name"),
        morphology=_text(root, "morphology"),
        disk_diameter_light_year_approx=_positive_float(root, "disk_diameter_light_year_approx"),
        star_count_lower_estimate=lower,
        star_count_upper_estimate=upper,
        solar_arm=_text(root, "solar_arm"),
        solar_galactic_orbit_period_year_approx=_positive_float(
            root, "solar_galactic_orbit_period_year_approx"
        ),
        source_urls=sources,
        scope_note=_text(root, "scope_note"),
    )
