"""Shared restrained engineering-plot style."""

from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib as mpl

NAVY = "#16324F"
BLUE = "#2878B5"
ORANGE = "#E07A1F"
GREEN = "#3A8D5D"
RED = "#B33A3A"
GREY = "#66717E"


@contextmanager
def engineering_style() -> Iterator[None]:
    """Apply a local, reproducible Matplotlib style without global side effects."""
    settings: dict[str, object] = {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "axes.edgecolor": "#424A52",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#D9DEE3",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.75,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    }
    # Matplotlib's stub enumerates every valid rc key; this runtime-validated map is
    # intentionally kept readable and local.
    with mpl.rc_context(settings):  # type: ignore[arg-type]
        yield
