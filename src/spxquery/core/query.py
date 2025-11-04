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

# Regex pattern to extract obs_id from obs_publisher_did
# Format: "ivo://irsa.ipac/spherex_qr?2025W23_1C_0051_3/D4"
# Extract: "2025W23_1C_0051_3"
OBS_ID_PATTERN = re.compile(r"\?([^/]+)")


def query_spherex_observations(
    source: Source, bands: Optional[List[str]] = None, cutout_size: Optional[str] = None
) -> QueryResults:
    """
    Query SPHEREx observations for a given source position.

    Uses spherex.artifact and spherex.plane tables to get direct download URLs.
    Cutout parameters are NOT included in the query - they will be appended
    during the download phase.

    Parameters
    ----------
    source : Source
        Target source with RA/Dec coordinates
    bands : List[str], optional
        List of bands to query (e.g., ['D1', 'D2']). If None, query all bands.
    cutout_size : str, optional
        NOT USED IN QUERY. Kept for backward compatibility.
        Cutout parameters are appended during download phase.

    Returns
    -------
    QueryResults
        Query results containing observation information with base download URLs
    """
    logger.info(f"Querying SPHEREx observations for source at RA={source.ra}, Dec={source.dec}")

    # Connect to TAP service
    service = pyvo.dal.TAPService(TAP_URL)

    # Build ADQL query using artifact + plane JOIN
    # This gives us direct download URLs without needing datalink resolution
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
    WHERE CONTAINS(POINT('ICRS', {source.ra}, {source.dec}), p.poly) = 1
    """

    # Add band filter if specified
    if bands:
        band_conditions = " OR ".join([f"p.energy_bandpassname = 'SPHEREx-{band}'" for band in bands])
        query += f" AND ({band_conditions})"

    query += " ORDER BY p.time_bounds_lower"

    logger.debug(f"ADQL query: {query}")

    # Submit and run query
    job = service.submit_job(query)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=300)

    if job.phase == "ERROR":
        raise RuntimeError(f"TAP query failed: {job.error}")

    results = job.fetch_result()

    # Process results
    observations = []
    for row in results:
        # Extract obs_id from obs_publisher_did
        # Format: "ivo://irsa.ipac/spherex_qr?2025W23_1C_0051_3/D4"
        obs_publisher_did = row["obs_publisher_did"]
        match = OBS_ID_PATTERN.search(obs_publisher_did)
        if not match:
            logger.warning(f"Could not extract obs_id from: {obs_publisher_did}")
            continue
        obs_id = match.group(1)

        # Extract band from energy_bandpassname (format: 'SPHEREx-D1')
        band_name = row["energy_bandpassname"]
        band = band_name.split("-")[-1] if "-" in band_name else band_name

        # Calculate MJD from time_bounds_lower and time_bounds_upper
        mjd = (row["time_bounds_lower"] + row["time_bounds_upper"]) / 2.0

        # Convert wavelength from meters to microns
        wavelength_min = row["energy_bounds_lower"] * 1e6  # m to μm
        wavelength_max = row["energy_bounds_upper"] * 1e6  # m to μm

        obs = ObservationInfo(
            obs_id=obs_id,
            band=band,
            mjd=mjd,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            download_url=row["download_url"],  # Base URL without cutout params
            t_min=row["time_bounds_lower"],
            t_max=row["time_bounds_upper"],
        )
        observations.append(obs)

    # Calculate summary statistics
    band_counts = {}
    for band in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        count = sum(1 for obs in observations if obs.band == band)
        if count > 0:
            band_counts[band] = count

    # Total size unknown until download completes
    total_size_gb = 0.0

    # Calculate time span
    if observations:
        time_span_days = max(obs.mjd for obs in observations) - min(obs.mjd for obs in observations)
    else:
        time_span_days = 0.0

    query_results = QueryResults(
        observations=observations,
        query_time=datetime.now(),
        source=source,
        total_size_gb=total_size_gb,  # Will be updated after download
        time_span_days=time_span_days,
        band_counts=band_counts,
    )

    logger.info(f"Found {len(observations)} observations spanning {time_span_days:.1f} days")

    return query_results


def print_query_summary(query_results: QueryResults) -> None:
    """
    Print a summary of query results.

    Parameters
    ----------
    query_results : QueryResults
        Query results to summarize
    """
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
