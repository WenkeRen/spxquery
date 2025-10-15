"""
SPXQuery: A package for SPHEREx spectral image data query and time-domain analysis.
"""

__version__ = "0.1.0"
__author__ = "SPXQuery Team"

from .core.config import Source, QueryConfig
from .core.pipeline import SPXQueryPipeline

__all__ = ["Source", "QueryConfig", "SPXQueryPipeline"]