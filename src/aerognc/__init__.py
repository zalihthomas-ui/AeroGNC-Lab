"""AeroGNC-Lab public package.

The package models only a fictional civilian research rocket with synthetic data.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

try:
    __version__ = version("aerognc-lab")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0+local"

if TYPE_CHECKING:
    from aerognc.api import fly_mission

__all__ = ["__version__", "fly_mission"]


def __getattr__(name: str) -> Any:
    """Lazily expose the high-level API without a heavy top-level import."""
    if name == "fly_mission":
        from aerognc.api import fly_mission

        return fly_mission
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
