from pathlib import Path

import pytest
import yaml

from aerognc.project import (
    EngineeringProject,
    ProjectConfigurationError,
    ProjectScenario,
    ProjectSettings,
    create_empty_project,
    load_project,
    save_project,
)


def _minimal_project(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "case.yaml").write_text("case: true\n", encoding="utf-8")
    path = tmp_path / "project.aerognc.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "workspace_root": ".",
                "name": "Test-Project",
                "description": "A deterministic project fixture.",
                "safety_scope": "Fictional civilian vehicle with synthetic parameters.",
                "settings": {
                    "result_directory": "results",
                    "default_seed": 7,
                    "max_workers": 2,
                },
                "scenarios": [
                    {
                        "name": "case-a",
                        "workflow": "three-dof",
                        "configuration": "configs/case.yaml",
                        "description": "fixture",
                        "enabled": True,
                        "tags": ["baseline"],
                        "parameters": {"scale": 1.0},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_bundled_portfolio_project_resolves_every_scenario() -> None:
    project = load_project("projects/portfolio_demo.aerognc.yaml")

    assert project.name == "AeroGNC-Portfolio"
    assert [scenario.workflow for scenario in project.scenarios] == [
        "three-dof",
        "six-dof",
        "rotating-six-dof",
        "multistage-recovery",
        "orbit-tour",
    ]
    assert all(project.configuration_path(item).is_file() for item in project.scenarios)
    assert project.result_root == (Path.cwd() / "results/projects/portfolio_demo").resolve()


def test_project_round_trip_is_strict_and_portable(tmp_path: Path) -> None:
    source = _minimal_project(tmp_path)
    project = load_project(source)
    destination = tmp_path / "copies" / "copied.aerognc.yaml"

    save_project(project, destination)
    copied = load_project(destination)

    assert copied.name == project.name
    assert copied.workspace_root == project.workspace_root
    assert copied.scenario("case-a").parameters["scale"] == 1.0
    assert copied.configuration_path(copied.scenario("case-a")).is_file()


def test_project_rejects_unknown_key_and_unsafe_input_path(tmp_path: Path) -> None:
    source = _minimal_project(tmp_path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["surprise"] = True
    source.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ProjectConfigurationError, match="unknown keys"):
        load_project(source)

    source = _minimal_project(tmp_path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["scenarios"][0]["configuration"] = "../outside.yaml"
    source.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ProjectConfigurationError, match="normalized relative path"):
        load_project(source)


def test_project_rejects_duplicate_scenarios_and_unsupported_schema(tmp_path: Path) -> None:
    source = _minimal_project(tmp_path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["scenarios"].append(dict(data["scenarios"][0]))
    source.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ProjectConfigurationError, match="unique"):
        load_project(source)

    data["scenarios"] = data["scenarios"][:1]
    data["schema_version"] = "99"
    source.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ProjectConfigurationError, match="unsupported project schema"):
        load_project(source)


def test_create_empty_project_and_dataclass_guards(tmp_path: Path) -> None:
    project = create_empty_project(tmp_path / "new-project", "New-Project")
    assert project.scenarios == ()
    assert project.result_root.name == "results"

    with pytest.raises(ValueError, match="unsigned 32-bit"):
        ProjectSettings(default_seed=-1)
    with pytest.raises(ValueError, match="normalized"):
        ProjectScenario("case", "three-dof", Path("../escape.yaml"))
    with pytest.raises(ValueError, match="fictional"):
        EngineeringProject(
            name="unsafe",
            description="fixture",
            safety_scope="not declared",
            scenarios=(),
            settings=ProjectSettings(),
            source_path=project.source_path,
            workspace_root=project.workspace_root,
        )
