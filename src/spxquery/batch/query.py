"""Region-based query for SPHEREx full-frame images.

Delegates to :func:`spxquery.core.query.query_spherex_region`.
"""

import logging

from ..core.config import QueryResults
from ..core.query import query_spherex_region
from .config import BatchConfig

logger = logging.getLogger(__name__)


def query_region_observations(config: BatchConfig) -> QueryResults:
    """Query SPHEREx archive for full-frame images covering a sky region.

    Thin wrapper that translates :class:`BatchConfig` fields into
    :func:`~spxquery.core.query.query_spherex_region` parameters.

    Parameters
    ----------
    config : BatchConfig
        Batch configuration with region definition and query parameters.

    Returns
    -------
    QueryResults
        Matching observations with download URLs.
    """
    return query_spherex_region(
        center_ra=config.center_ra,
        center_dec=config.center_dec,
        radius=config.radius,
        coverage_mode=config.coverage_mode,
        bands=config.bands,
        mjd_range=config.mjd_range,
        max_images=config.max_images,
    )
