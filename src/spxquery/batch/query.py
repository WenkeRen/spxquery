"""Region-based query for SPHEREx full-frame images."""

import logging
import re
from datetime import datetime
from typing import List, Optional

import pyvo

from ..core.config import ObservationInfo, QueryResults, Source
from .config import BatchConfig

logger = logging.getLogger(__name__)

TAP_URL = "https://irsa.ipac.caltech.edu/TAP"

# obs_publisher_did format: "ivo://irsa.ipac/spherex_qr?2025W23_1C_0051_3/D4"
_OBS_ID_PATTERN = re.compile(r"\?([^/]+)")


def query_region_observations(config: BatchConfig) -> QueryResults:
    """Query SPHEREx archive for full-frame images covering a sky region.

    Parameters
    ----------
    config : BatchConfig
        Batch configuration with region definition and query parameters.

    Returns
    -------
    QueryResults
        Matching observations with download URLs.

    Raises
    ------
    RuntimeError
        If number of results exceeds ``config.max_images``.
    """
    region_source = Source(ra=config.center_ra, dec=config.center_dec, name="batch_region")

    # Build ADQL spatial predicate
    if config.coverage_mode == "any":
        spatial = f"INTERSECTS(p.poly, CIRCLE('ICRS', {config.center_ra}, {config.center_dec}, {config.radius})) = 1"
    else:
        spatial = f"CONTAINS(p.poly, CIRCLE('ICRS', {config.center_ra}, {config.center_dec}, {config.radius})) = 1"

    query = f"""
    SELECT
        'https://irsa.ipac.caltech.edu/' || a.uri AS download_url,
        p.obs_publisher_did,
        p.time_bounds_lower,
        p.time_bounds_upper,
        p.energy_bandpassname,
        p.energy_bounds_lower,
        p.energy_bounds_upper
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE {spatial}
    """

    if config.bands:
        band_conditions = " OR ".join(
            f"p.energy_bandpassname = 'SPHEREx-{band}'" for band in config.bands
        )
        query += f" AND ({band_conditions})"

    query += " ORDER BY p.time_bounds_lower"

    logger.info(f"Querying region: RA={config.center_ra}, Dec={config.center_dec}, radius={config.radius} deg")
    logger.info(f"Coverage mode: {config.coverage_mode}, Bands: {config.bands or 'all'}")

    service = pyvo.dal.TAPService(TAP_URL)
    results = service.search(query)

    # Parse results
    observations: List[ObservationInfo] = []
    for row in results:
        obs_publisher_did = row["obs_publisher_did"]
        match = _OBS_ID_PATTERN.search(obs_publisher_did)
        if not match:
            logger.warning(f"Could not extract obs_id from: {obs_publisher_did}")
            continue

        obs_id = match.group(1)
        band_name = row["energy_bandpassname"]
        band = band_name.split("-")[-1] if "-" in band_name else band_name
        mjd = (row["time_bounds_lower"] + row["time_bounds_upper"]) / 2.0
        wavelength_min = row["energy_bounds_lower"] * 1e6  # m -> um
        wavelength_max = row["energy_bounds_upper"] * 1e6

        observations.append(
            ObservationInfo(
                obs_id=obs_id,
                band=band,
                mjd=mjd,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
                download_url=row["download_url"],
                t_min=row["time_bounds_lower"],
                t_max=row["time_bounds_upper"],
            )
        )

    # Size gate
    if len(observations) > config.max_images:
        raise RuntimeError(
            f"Query returned {len(observations)} images, exceeding max_images={config.max_images}. "
            f"Increase max_images to proceed, or reduce your search region."
        )

    # Summary statistics
    band_counts: dict[str, int] = {}
    for band in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        count = sum(1 for obs in observations if obs.band == band)
        if count > 0:
            band_counts[band] = count

    time_span = (
        max(obs.mjd for obs in observations) - min(obs.mjd for obs in observations)
        if observations
        else 0.0
    )

    query_results = QueryResults(
        observations=observations,
        query_time=datetime.now(),
        source=region_source,
        total_size_gb=0.0,
        time_span_days=time_span,
        band_counts=band_counts,
    )

    logger.info(
        f"Found {len(observations)} observations "
        f"({len(band_counts)} bands, {time_span:.0f} days span)"
    )

    return query_results
