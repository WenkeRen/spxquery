"""
SPXQuery: A package for SPHEREx spectral image data query and time-domain analysis.
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # For Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("spxquery")
except PackageNotFoundError:
    # Package is not installed, use fallback (for development)
    __version__ = "0.1.2"  # Sync with pyproject.toml manually for development

__author__ = "SPXQuery Team"

from .core.config import Source, QueryConfig
from .core.pipeline import SPXQueryPipeline

__all__ = ["Source", "QueryConfig", "SPXQueryPipeline"]