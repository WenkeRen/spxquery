"""
SPXQuery: A package for SPHEREx spectral image data query and time-domain analysis.
"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # For Python < 3.8
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("spxquery")
except PackageNotFoundError:
    # Package is not installed, use fallback (for development)
    __version__ = "0.2.2"  # Sync with pyproject.toml manually for development

__author__ = "SPXQuery Team"

from .core.config import AdvancedConfig, QueryConfig, Source
from .core.pipeline import SPXQueryPipeline

__all__ = ["Source", "AdvancedConfig", "QueryConfig", "SPXQueryPipeline"]
