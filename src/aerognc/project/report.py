"""Self-contained print-ready HTML reports for stored engineering runs."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np

from aerognc.project.comparison import RunComparison
from aerognc.project.manifest import RunManifest
from aerognc.project.result_store import ResultDataset, StoredRun


def _number(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.8g}"
    if value is None:
        return "—"
    return escape(str(value))


def _table(headers: tuple[str, ...], rows: Sequence[Sequence[object]]) -> str:
    heading = "".join(f"<th>{escape(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_number(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<div class=table-wrap><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>"


def _sparkline_svg(
    time_s: np.ndarray,
    values: np.ndarray,
    channel: str,
    unit: str,
) -> str:
    width, height = 860.0, 210.0
    left, right, top, bottom = 74.0, 18.0, 28.0, 42.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    t_min, t_max = float(time_s[0]), float(time_s[-1])
    y_min, y_max = float(np.min(values)), float(np.max(values))
    if y_max == y_min:
        padding = max(1.0, abs(y_min)) * 0.05
        y_min -= padding
        y_max += padding
    indices = np.linspace(0, time_s.size - 1, min(time_s.size, 700), dtype=int)
    selected_t = time_s[indices]
    selected_y = values[indices]
    x = left + (selected_t - t_min) / (t_max - t_min) * plot_width
    y = top + (y_max - selected_y) / (y_max - y_min) * plot_height
    points = " ".join(f"{x_value:.2f},{y_value:.2f}" for x_value, y_value in zip(x, y, strict=True))
    title = escape(channel.replace("_", " "))
    escaped_unit = escape(unit)
    return f"""
    <figure class=plot>
      <figcaption>{title} [{escaped_unit}]</figcaption>
      <svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{title} versus time">
        <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" class="plot-bg"/>
        <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>
        <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>
        <polyline points="{points}" class="trace"/>
        <text x="{left}" y="{height - 12}" class="tick">{t_min:.5g} s</text>
        <text x="{left + plot_width}" y="{height - 12}" text-anchor="end" class="tick">{t_max:.5g} s</text>
        <text x="{left - 8}" y="{top + 5}" text-anchor="end" class="tick">{y_max:.5g}</text>
        <text x="{left - 8}" y="{top + plot_height}" text-anchor="end" class="tick">{y_min:.5g}</text>
      </svg>
    </figure>
    """


def _provenance(manifest: RunManifest) -> str:
    rows = [
        ("Run ID", manifest.run_id),
        ("Input fingerprint", manifest.input_fingerprint),
        ("Created UTC", manifest.created_utc),
        ("Status", manifest.status),
        ("Workflow", manifest.workflow),
        ("Configuration", manifest.configuration_path),
        ("Configuration SHA-256", manifest.configuration_sha256),
        ("Seed", manifest.seed),
        ("AeroGNC-Lab", manifest.software_version),
        ("Python", manifest.python_version),
        ("Platform", manifest.platform),
        ("Execution time", f"{manifest.execution_time_s:.6g} s"),
    ]
    return _table(("Field", "Value"), rows)


def _requirements(manifest: RunManifest) -> str:
    if not manifest.requirements:
        return "<p class=muted>No run-specific requirement outcomes were supplied.</p>"
    rows = [
        (
            item.identifier,
            "PASS" if item.passed else "FAIL",
            item.value,
            item.limit,
            item.margin,
            item.unit,
            item.detail,
        )
        for item in manifest.requirements
    ]
    return _table(("Requirement", "Result", "Value", "Limit", "Margin", "Unit", "Detail"), rows)


def _events(dataset: ResultDataset | None) -> str:
    if dataset is None or not dataset.events:
        return "<p class=muted>No trajectory events are available.</p>"
    keys = ["name", "time_s"]
    extra = sorted({key for event in dataset.events for key in event if key not in keys})
    headers = tuple(keys + extra)
    rows = [tuple(event.get(key) for key in headers) for event in dataset.events]
    return _table(headers, rows)


def _maxima(dataset: ResultDataset | None) -> str:
    if dataset is None or not dataset.maxima:
        return "<p class=muted>No maximum-value summary is available.</p>"
    rows = [
        (
            name,
            record.get("value"),
            record.get("unit"),
            record.get("time_s"),
        )
        for name, record in dataset.maxima.items()
    ]
    return _table(("Quantity", "Value", "Unit", "Time [s]"), rows)


def _comparison(comparison: RunComparison | None) -> str:
    if comparison is None:
        return ""
    rows = [
        (
            item.channel,
            item.unit,
            item.bias,
            item.rms_difference,
            item.maximum_absolute_difference,
            item.final_difference,
            item.correlation,
        )
        for item in comparison.channels
    ]
    return (
        "<section><h2>Run comparison</h2>"
        f"<p>{escape(comparison.baseline_scenario)} versus "
        f"{escape(comparison.candidate_scenario)}, common domain "
        f"{comparison.start_time_s:.8g}&ndash;{comparison.end_time_s:.8g} s.</p>"
        + _table(
            ("Channel", "Unit", "Bias", "RMS", "Max abs", "Final", "Correlation"),
            rows,
        )
        + "</section>"
    )


def report_html(stored: StoredRun, comparison: RunComparison | None = None) -> str:
    """Render a complete offline report with escaped project-controlled text."""
    manifest = stored.manifest
    dataset = stored.dataset
    warnings = (
        "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in manifest.warnings) + "</ul>"
        if manifest.warnings
        else "<p class=muted>No warnings were recorded.</p>"
    )
    failure = (
        ""
        if manifest.failure_reason is None
        else f"<p class=failure><strong>Terminal reason:</strong> {escape(manifest.failure_reason)}</p>"
    )
    artifacts = (
        "<ul>"
        + "".join(
            f'<li><a href="{quote(item.relative_path)}">{escape(item.role)}</a> — '
            f"{item.size_bytes} bytes; SHA-256 <code>{item.sha256}</code></li>"
            for item in manifest.artifacts
        )
        + "</ul>"
        if manifest.artifacts
        else "<p class=muted>No derived artefacts were committed.</p>"
    )
    plots = ""
    if dataset is not None:
        preferred = (
            "altitude_m",
            "total_velocity_mps",
            "dynamic_pressure_pa",
            "mass_kg",
            "attitude_error_deg",
            "mach",
        )
        selected = [name for name in preferred if name in dataset.channels]
        if not selected:
            selected = list(dataset.channels)[:4]
        plots = "".join(
            _sparkline_svg(dataset.time_s, dataset.channels[name], name, dataset.units[name])
            for name in selected
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AeroGNC-Lab report — {escape(manifest.scenario_name)}</title>
<style>
:root {{ color-scheme: light; --ink:#172b43; --muted:#60758a; --line:#cbd6df; --blue:#2579b8; --green:#147d64; --red:#b33b45; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#eef3f7; color:var(--ink); font:15px/1.45 "Segoe UI",Arial,sans-serif; }}
main {{ max-width:1080px; margin:28px auto; background:white; padding:38px 46px; box-shadow:0 3px 18px #25384d22; }}
h1 {{ margin:0 0 4px; font-size:30px; }} h2 {{ margin-top:32px; border-bottom:2px solid var(--line); padding-bottom:7px; }}
.eyebrow {{ color:var(--green); font-weight:700; letter-spacing:.06em; text-transform:uppercase; }} .muted {{ color:var(--muted); }}
.safety {{ border-left:5px solid var(--green); background:#effaf6; padding:12px 16px; }} .failure {{ border-left:5px solid var(--red); background:#fff2f3; padding:12px 16px; }}
.table-wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; }} th,td {{ border:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }} th {{ background:#edf4f8; }} code {{ font-size:12px; overflow-wrap:anywhere; }}
.plot {{ margin:18px 0; }} figcaption {{ font-weight:650; margin-bottom:5px; }} svg {{ width:100%; height:auto; }} .plot-bg {{ fill:#f8fafc; stroke:var(--line); }} .axis {{ stroke:#70869a; stroke-width:1; }} .trace {{ fill:none; stroke:var(--blue); stroke-width:2.3; }} .tick {{ fill:#52697e; font-size:12px; }}
@media print {{ body {{ background:white; }} main {{ margin:0; max-width:none; box-shadow:none; padding:18mm; }} a {{ color:inherit; text-decoration:none; }} }}
</style>
</head>
<body><main>
<div class=eyebrow>AeroGNC-Lab engineering evidence</div>
<h1>{escape(manifest.scenario_name)}</h1>
<p>{escape(manifest.project_name)} · {escape(manifest.workflow)} · {escape(manifest.status.upper())}</p>
<p class=safety><strong>Public-safety scope:</strong> {escape(manifest.safety_scope)}</p>
{failure}
<section><h2>Provenance</h2>{_provenance(manifest)}</section>
<section><h2>Warnings</h2>{warnings}</section>
<section><h2>Requirement assessment</h2>{_requirements(manifest)}</section>
<section><h2>Events</h2>{_events(dataset)}</section>
<section><h2>Maximum-value summary</h2>{_maxima(dataset)}</section>
<section><h2>Selected histories</h2>{plots or "<p class=muted>No trajectory is available.</p>"}</section>
{_comparison(comparison)}
<section><h2>Committed artefacts</h2>{artifacts}</section>
<p class=muted>Generated entirely from the stored local run. No network data or hidden in-memory simulation state is required.</p>
</main></body></html>
"""


def write_engineering_report(
    stored: StoredRun,
    path: str | Path | None = None,
    *,
    comparison: RunComparison | None = None,
) -> Path:
    """Write a self-contained report atomically next to or outside the run."""
    destination = stored.directory / "report.html" if path is None else Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        report_html(stored, comparison),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)
    return destination
