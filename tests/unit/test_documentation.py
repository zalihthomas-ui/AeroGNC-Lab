import re
from pathlib import Path


def test_required_engineering_documents_are_present_and_substantive() -> None:
    required = {
        "architecture.md",
        "mathematical_model.md",
        "coordinate_systems.md",
        "vehicle_model.md",
        "control_architecture.md",
        "navigation_filter.md",
        "verification_method.md",
        "validation_report.md",
        "monte_carlo_analysis.md",
        "synthetic_flight_test.md",
        "matlab_validation.md",
        "simulink_validation.md",
        "future_hil.md",
        "interplanetary_mission.md",
        "advanced_astrodynamics.md",
        "mission_designer.md",
        "flight_control_analysis.md",
        "error_state_navigation.md",
        "public_safety.md",
        "playback.md",
        "playback_3d.md",
        "geodesy_rotating_planet.md",
        "aerodynamic_database.md",
        "flight_envelope.md",
        "constrained_ascent_guidance.md",
        "advanced_navigation.md",
        "flight_data_identification.md",
        "orbit_assisted_tour.md",
        "launch_window_optimization.md",
        "astrodynamics_interoperability.md",
        "galaxy_catalog.md",
        "simulation_workbench.md",
        "fmi_interoperability.md",
        "ui_architecture_decision.md",
        "orbit_sandbox.md",
        "aircraft_simulation.md",
        "release_process.md",
    }
    docs = Path("docs")
    for name in required:
        document = docs / name
        assert document.is_file(), name
        assert len(document.read_text(encoding="utf-8")) >= 500, name

    readme = Path("README.md").read_text(encoding="utf-8").lower()
    for phrase in ("fictional", "synthetic", "quick start", "verification", "limitations"):
        assert phrase in readme


def test_ui_architecture_decision_is_scoped_and_complete() -> None:
    decision = Path("docs/ui_architecture_decision.md").read_text(encoding="utf-8")
    prototype = Path("docs/prototypes/workbench_local_web.html").read_text(encoding="utf-8")

    for criterion in (
        "Accessibility",
        "Deployment",
        "3D capability",
        "Licensing",
        "Startup and memory",
        "Maintenance",
    ):
        assert criterion in decision
    assert "retains its native Tk/ttk" in decision
    assert "PySide6 was unavailable and was not executed" in decision
    assert "not connected to the engineering solver" in decision
    assert "NON-PRODUCTION ARCHITECTURE PROTOTYPE" in prototype
    assert 'aria-live="polite"' in prototype


def test_relative_markdown_links_resolve_inside_the_repository() -> None:
    markdown_files = [Path("README.md"), Path("CONTRIBUTING.md")]
    markdown_files.extend(Path("docs").glob("*.md"))
    markdown_files.extend(Path("requirements").glob("*.md"))
    markdown_files.extend(Path("examples").glob("*.md"))
    markdown_files.extend(Path("notebooks").glob("*.md"))
    link_pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    for document in markdown_files:
        for raw_target in link_pattern.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip("<>").split("#", maxsplit=1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / target).resolve()
            assert resolved.exists(), f"broken link in {document}: {raw_target}"
