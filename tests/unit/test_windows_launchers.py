from pathlib import Path


def test_primary_windows_launcher_opens_unified_workbench() -> None:
    launcher = Path("run_aerognc.bat").read_text("utf-8").lower()

    assert ".venv\\scripts\\python.exe" in launcher
    assert "-m aerognc.cli workbench" in launcher
    assert "pause" in launcher
    assert "mission-designer" not in launcher


def test_named_simulation_and_advanced_launchers_delegate_correctly() -> None:
    simulation = Path("run_simulation.bat").read_text("utf-8").lower()
    solver = Path("run_solver.bat").read_text("utf-8").lower()
    advanced = Path("run_interplanetary.bat").read_text("utf-8").lower()
    diagnostic = Path("diagnose_aerognc.bat").read_text("utf-8").lower()

    assert "run_aerognc.bat" in simulation
    assert "run_aerognc.bat" in solver
    assert "-m aerognc.cli mission-designer" in advanced
    assert '"%python_exe%" scripts\\diagnose_environment.py' in diagnostic
    assert "py -3 scripts\\diagnose_environment.py" in diagnostic
    assert "pause" in diagnostic
