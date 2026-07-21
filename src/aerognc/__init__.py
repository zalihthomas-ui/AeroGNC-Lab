"""AeroGNC-Lab public package.

The package models only a fictional civilian research rocket with synthetic data.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aerognc-lab")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
