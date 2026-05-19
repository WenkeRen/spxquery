"""
TAP query functionality for SPHEREx data from IRSA.
"""

import logging
import re
from datetime import datetime
from typing import List, Optional

import pyvo

from .config import ObservationInfo, QueryResults, Source

logger = logging.getLogger(__name__)

# SPHEREx TAP service configuration
TAP_URL = "https://irsa.ipac.caltech.edu/TAP"

# Band wavelength ranges (microns) - for reference and summary display
BAND_WAVELENGTHS = {
    "D1": (0.75, 1.09),
    "D2": (1.10, 1.62),
    "D3": (1.63, 2.41),
    "D4": (2.42, 3.82),
    "D5": (3.83, 4.41),
    "D6": (4.42, 5.00),
}

# obs_publisher_did format: "ivo://irsa.ipac/spherex_qr?2025W23_1C_0051_3/D4"
_OBS_ID_PATTERN = re.compile(r"\?([^/]+)")

# ADQL SELECT + JOIN shared by all query modes
_ADQL_SELECT = """
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
"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_observation_row(row) -> Optional[ObservationInfo]:
    """Parse one TAP result row into an ObservationInfo.

    Returns None if obs_id cannot be extracted from obs_publisher_did.
    """
    obs_publisher_did = row["obs_publisher_did"]
    match = _OBS_ID_PATTERN.search(obs_publisher_did)
    if not match:
        logger.warning(f"Could not extract obs_id from: {obs_publisher_did}")
        return None

    obs_id = match.group(1)
    band_name = row["energy_bandpassname"]
    band = band_name.split("-")[-1] if "-" in band_name else band_name
    mjd = (row["time_bounds_lower"] + row["time_bounds_upper"]) / 2.0
    wavelength_min = row["energy_bounds_lower"] * 1e6  # m -> um
    wavelength_max = row["energy_bounds_upper"] * 1e6

    return ObservationInfo(
        obs_id=obs_id,
        band=band,
        mjd=mjd,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        download_url=row["download_url"],
        t_min=row["time_bounds_lower"],
        t_max=row["time_bounds_upper"],
    )


def _build_band_counts(observations: List[ObservationInfo]) -> dict[str, int]:
    """Count observations per detector band (D1-D6)."""
    counts: dict[str, int] = {}
    for band in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        count = sum(1 for obs in observations if obs.band == band)
        if count > 0:
            counts[band] = count
    return counts


def _execute_tap_query(service: pyvo.dal.TAPService, query: str):
    """Execute a TAP query: try synchronous first, fall back to async job.

    Parameters
    ----------
    service : TAPService
        Connected TAP service.
    query : str
        ADQL query string.

    Returns
    -------
    TAP results iterable.
    """
    try:
        logger.debug("Attempting synchronous TAP query")
        results = service.search(query)
        logger.debug("Synchronous query succeeded")
        return results
    except Exception as exc:
        logger.info(f"Sync query failed ({exc}), falling back to async job")
        job = service.submit_job(query)
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=300)

        if job.phase == "ERROR":
            raise RuntimeError(f"TAP async job failed: {job.error}") from exc

        return job.fetch_result()


def _build_adql_band_filter(bands: Optional[List[str]]) -> str:
    """Build ADQL AND-clause for band filtering, or empty string."""
    if not bands:
        return ""
    conditions = " OR ".join(f"p.energy_bandpassname = 'SPHEREx-{band}'" for band in bands)
    return f" AND ({conditions})"


def _build_adql_mjd_filter(mjd_range: Optional[tuple]) -> str:
    """Build ADQL AND-clause for MJD midpoint filtering, or empty string."""
    if mjd_range is None:
        return ""
    mjd_min, mjd_max = mjd_range
    return (
        f" AND (p.time_bounds_lower + p.time_bounds_upper)/2.0 >= {mjd_min}"
        f" AND (p.time_bounds_lower + p.time_bounds_upper)/2.0 <= {mjd_max}"
    )


def _parse_tap_results(results) -> List[ObservationInfo]:
    """Parse all rows from TAP results into ObservationInfo list."""
    observations: List[ObservationInfo] = []
    for row in results:
        obs = _parse_observation_row(row)
        if obs is not None:
            observations.append(obs)
    return observations


def _assemble_query_results(observations: List[ObservationInfo], source: Source) -> QueryResults:
    """Build QueryResults from parsed observations and source."""
    band_counts = _build_band_counts(observations)
    time_span = max(obs.mjd for obs in observations) - min(obs.mjd for obs in observations) if observations else 0.0
    return QueryResults(
        observations=observations,
        query_time=datetime.now(),
        source=source,
        total_size_gb=0.0,
        time_span_days=time_span,
        band_counts=band_counts,
    )


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------


def query_spherex_observations(
    source: Source,
    bands: Optional[List[str]] = None,
    cutout_size: Optional[str] = None,
    mjd_range: Optional[tuple] = None,
    max_images: int = 0,
) -> QueryResults:
    """Query SPHEREx observations for a given source position.

    Uses CONTAINS(POINT, polygon) to find images that cover the source.

    Parameters
    ----------
    source : Source
        Target source with RA/Dec coordinates.
    bands : list of str, optional
        Bands to query (e.g., ``['D1', 'D2']``).  ``None`` = all.
    cutout_size : str, optional
        Ignored (kept for backward compatibility).  Cutout parameters
        are appended during the download phase.
    mjd_range : tuple, optional
        ``(mjd_min, mjd_max)`` to restrict by observation time.
        Applied server-side in ADQL.
    max_images : int, optional
        Safety cap on result count.  0 = no limit (default).

    Returns
    -------
    QueryResults
    """
    logger.info(f"Querying SPHEREx observations for source at RA={source.ra}, Dec={source.dec}")

    spatial = f"CONTAINS(POINT('ICRS', {source.ra}, {source.dec}), p.poly) = 1"

    query = (
        _ADQL_SELECT
        + f"WHERE {spatial}"
        + _build_adql_band_filter(bands)
        + _build_adql_mjd_filter(mjd_range)
        + " ORDER BY p.time_bounds_lower"
    )

    logger.debug(f"ADQL query: {query}")

    service = pyvo.dal.TAPService(TAP_URL)
    results = _execute_tap_query(service, query)

    observations = _parse_tap_results(results)

    if max_images > 0 and len(observations) > max_images:
        raise RuntimeError(
            f"Query returned {len(observations)} observations, "
            f"exceeding max_images={max_images}. "
            f"Raise max_images or narrow the search."
        )

    query_results = _assemble_query_results(observations, source)

    logger.info(
        f"Found {len(observations)} observations "
        f"({len(query_results.band_counts)} bands, {query_results.time_span_days:.0f} days span)"
    )

    return query_results


def query_spherex_region(
    center_ra: float,
    center_dec: float,
    radius: float,
    coverage_mode: str = "any",
    bands: Optional[List[str]] = None,
    mjd_range: Optional[tuple] = None,
    max_images: int = 500,
) -> QueryResults:
    """Query SPHEREx observations covering a circular sky region.

    Parameters
    ----------
    center_ra, center_dec : float
        Region center in degrees (ICRS).
    radius : float
        Search radius in degrees.
    coverage_mode : str
        ``"any"`` — image overlaps with search circle (INTERSECTS).
        ``"full"`` — image fully contains the search circle (CONTAINS).
    bands : list of str, optional
        Bands to query (e.g., ``['D1', 'D3']``).  ``None`` = all.
    mjd_range : tuple, optional
        ``(mjd_min, mjd_max)`` to restrict by observation time.
        Applied server-side in ADQL.
    max_images : int
        Safety cap.  Raise if exceeded (default 500).

    Returns
    -------
    QueryResults
    """
    if coverage_mode == "any":
        spatial = f"INTERSECTS(p.poly, CIRCLE('ICRS', {center_ra}, {center_dec}, {radius})) = 1"
    else:
        spatial = f"CONTAINS(CIRCLE('ICRS', {center_ra}, {center_dec}, {radius}), p.poly) = 1"

    query = (
        _ADQL_SELECT
        + f"WHERE {spatial}"
        + _build_adql_band_filter(bands)
        + _build_adql_mjd_filter(mjd_range)
        + " ORDER BY p.time_bounds_lower"
    )

    logger.info(f"Querying region: RA={center_ra}, Dec={center_dec}, radius={radius} deg, mode={coverage_mode}")
    logger.debug(f"ADQL query: {query}")

    service = pyvo.dal.TAPService(TAP_URL)
    results = _execute_tap_query(service, query)

    observations = _parse_tap_results(results)

    if max_images > 0 and len(observations) > max_images:
        raise RuntimeError(
            f"Query returned {len(observations)} images, exceeding max_images={max_images}. "
            f"Increase max_images to proceed, or reduce your search region."
        )

    source = Source(ra=center_ra, dec=center_dec, name="region_query")
    query_results = _assemble_query_results(observations, source)

    logger.info(
        f"Found {len(observations)} observations "
        f"({len(query_results.band_counts)} bands, {query_results.time_span_days:.0f} days span)"
    )

    return query_results


def print_query_summary(query_results: QueryResults) -> None:
    """Print a summary of query results."""
    print(f"\n{'=' * 60}")
    print("SPHEREx Archive Search Results")
    print(f"{'=' * 60}")
    print(f"Source: RA={query_results.source.ra:.6f}, Dec={query_results.source.dec:.6f}")
    if query_results.source.name:
        print(f"        Name: {query_results.source.name}")
    print(f"Query time: {query_results.query_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTotal observations found: {len(query_results)}")

    print("\nObservations by band:")
    for band in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        count = query_results.band_counts.get(band, 0)
        if count > 0:
            wl_range = BAND_WAVELENGTHS[band]
            print(f"  {band} ({wl_range[0]:.2f}-{wl_range[1]:.2f} μm): {count:3d} observations")

    print(f"\nTime span: {query_results.time_span_days:.1f} days")
    print(f"Total data volume: {query_results.total_size_gb:.2f} GB")
    print(f"{'=' * 60}\n")
