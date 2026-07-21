"""Measure lightweight UI startup capabilities without choosing a new framework."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import json
import threading
import time
import tracemalloc
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    """One explicitly scoped framework/prototype observation."""

    name: str
    available: bool
    live_probe_executed: bool
    startup_time_s: float | None
    python_allocation_peak_mb: float | None
    detail: str


class _PrototypeHandler(BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, _format: str, *_arguments: object) -> None:
        return


def _probe_local_web(prototype_path: Path) -> CapabilityResult:
    payload = prototype_path.read_bytes()
    _PrototypeHandler.payload = payload
    tracemalloc.start()
    started = time.perf_counter()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PrototypeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=2.0) as response:
            received = response.read()
            status = response.status
        elapsed = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        tracemalloc.stop()
    valid = status == 200 and received == payload
    return CapabilityResult(
        "stdlib local-web prototype",
        valid,
        True,
        elapsed,
        peak_bytes / 1.0e6,
        (
            f"Served {len(payload)} offline HTML bytes over ephemeral localhost HTTP; "
            "the page is not connected to an engineering solver."
        ),
    )


def _probe_tk(*, live: bool) -> CapabilityResult:
    available = importlib.util.find_spec("tkinter") is not None
    if not available:
        return CapabilityResult("Tk/ttk", False, False, None, None, "tkinter is unavailable")
    if not live:
        return CapabilityResult(
            "Tk/ttk", True, False, None, None, "Module detected; live window probe skipped"
        )
    tracemalloc.start()
    started = time.perf_counter()
    try:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.withdraw()
        notebook = ttk.Notebook(root)
        for label in ("Projects", "Rocket", "Planetary", "Astronomy", "Results", "Help"):
            page = ttk.Frame(notebook)
            notebook.add(page, text=label)
        ttk.Entry(notebook.nametowidget(notebook.tabs()[0])).pack()
        ttk.Button(notebook.nametowidget(notebook.tabs()[0]), text="Run").pack()
        root.update_idletasks()
        elapsed = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        root.destroy()
    except Exception as error:  # pragma: no cover - depends on desktop session
        tracemalloc.stop()
        return CapabilityResult("Tk/ttk", True, True, None, None, f"Live probe failed: {error}")
    tracemalloc.stop()
    return CapabilityResult(
        "Tk/ttk",
        True,
        True,
        elapsed,
        peak_bytes / 1.0e6,
        "Hidden eight-page widget skeleton constructed and destroyed; no solver was run.",
    )


def _pyside_status() -> CapabilityResult:
    available = importlib.util.find_spec("PySide6") is not None
    version: str | None = None
    if available:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            version = importlib.metadata.version("PySide6")
    detail = (
        f"PySide6 {version or 'module'} detected but not executed"
        if available
        else ("PySide6 is not installed; no Qt startup or memory result is claimed")
    )
    return CapabilityResult("PySide6/Qt", available, False, None, None, detail)


def _probe_full_workbench(*, live: bool) -> CapabilityResult:
    if not live:
        return CapabilityResult(
            "AeroGNC full workbench",
            True,
            False,
            None,
            None,
            "Production workbench construction skipped",
        )
    tracemalloc.start()
    started = time.perf_counter()
    root = None
    try:
        import tkinter as tk

        from aerognc.catalogs import (
            load_exoplanet_catalog,
            load_milky_way_metadata,
            load_solar_system_planets,
        )
        from aerognc.configuration import load_planetary_catalog
        from aerognc.visualisation.workbench import AeroGNCWorkbenchApp, WorkbenchPaths

        paths = WorkbenchPaths(
            *(
                Path(path).resolve()
                for path in (
                    "configs/six_dof_nominal.yaml",
                    "configs/orbit_assisted_tour.yaml",
                    "configs/fictional_planetary_system.yaml",
                    "configs/interplanetary_gravity_assist.yaml",
                    "data/catalogs/nasa_confirmed_exoplanets.csv",
                    "data/catalogs/nasa_confirmed_exoplanets.metadata.json",
                    "data/catalogs/milky_way_metadata.yaml",
                    "data/catalogs/solar_system_planets.csv",
                    "projects/portfolio_demo.aerognc.yaml",
                )
            )
        )
        paths.validate()
        root = tk.Tk()
        root.withdraw()
        app = AeroGNCWorkbenchApp(
            root,
            paths,
            load_planetary_catalog(paths.planetary_catalog),
            load_exoplanet_catalog(paths.exoplanet_csv, paths.exoplanet_metadata),
            load_milky_way_metadata(paths.milky_way_metadata),
            load_solar_system_planets(paths.solar_system_planets),
        )
        root.update_idletasks()
        labels = tuple(str(app.notebook.tab(tab, "text")) for tab in app.notebook.tabs())
        elapsed = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        if labels != (
            "Start",
            "Rocket",
            "Satellite Orbit",
            "Aircraft Flight",
            "Planet Trip",
            "Saved Runs",
            "Astronomy Data",
            "Checks",
        ):
            raise RuntimeError(f"unexpected production workbench pages: {labels}")
        if app.project_snapshot is None or len(app.project_snapshot.scenarios) != 5:
            raise RuntimeError("bundled five-scenario project did not load in the workbench")
        if (
            app.rocket_advanced_visible
            or app.orbit_advanced_visible
            or app.aircraft_advanced_visible
            or app.tour_advanced_visible
        ):
            raise RuntimeError("specialist solver inputs were not hidden on first construction")
    except Exception as error:  # pragma: no cover - depends on desktop session
        return CapabilityResult(
            "AeroGNC full workbench",
            True,
            True,
            None,
            None,
            f"Production construction failed: {error}",
        )
    finally:
        if root is not None:
            root.destroy()
        tracemalloc.stop()
    return CapabilityResult(
        "AeroGNC full workbench",
        True,
        True,
        elapsed,
        peak_bytes / 1.0e6,
        (
            "All eight purpose-labelled production pages and the bundled five-scenario project "
            "constructed with specialist solver inputs hidden by default."
        ),
    )


def probe_ui_capabilities(
    prototype_path: Path,
    *,
    live_tk: bool,
    full_workbench: bool,
) -> dict[str, object]:
    """Probe the retained UI and dependency-free local-web comparison shell."""
    results = (
        _probe_tk(live=live_tk),
        _probe_full_workbench(live=full_workbench),
        _pyside_status(),
        _probe_local_web(prototype_path),
    )
    return {
        "schema_version": "1.0",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "measurement_scope": (
            "Startup is construction-to-ready time. Memory is tracemalloc Python allocation "
            "peak, not process RSS. Values are machine-specific and are not acceptance limits."
        ),
        "results": [asdict(result) for result in results],
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prototype",
        type=Path,
        default=PROJECT_ROOT / "docs" / "prototypes" / "workbench_local_web.html",
    )
    parser.add_argument("--skip-tk-live", action="store_true")
    parser.add_argument(
        "--full-workbench",
        action="store_true",
        help="construct and immediately destroy the hidden production workbench",
    )
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(arguments)
    payload = probe_ui_capabilities(
        options.prototype,
        live_tk=not options.skip_tk_live,
        full_workbench=options.full_workbench,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
