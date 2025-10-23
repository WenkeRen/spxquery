"""
TAP query functionality for SPHEREx data from IRSA.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pyvo
from pyvo.dal.adhoc import DatalinkResults
from tqdm import tqdm

from .config import Source, ObservationInfo, QueryResults

logger = logging.getLogger(__name__)

# SPHEREx TAP service configuration
TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
SPHEREX_TABLE = "spherex.obscore"

# Band wavelength ranges (microns)
BAND_WAVELENGTHS = {
    'D1': (0.75, 1.09),
    'D2': (1.10, 1.62),
    'D3': (1.63, 2.41),
    'D4': (2.42, 3.82),
    'D5': (3.83, 4.41),
    'D6': (4.42, 5.00)
}

# Full image file size (MB) for SPHEREx 2040x2040 FITS files
FULL_IMAGE_SIZE_MB = 71.6


def query_spherex_observations(
    source: Source,
    bands: Optional[List[str]] = None,
    cutout_size: Optional[str] = None
) -> QueryResults:
    """
    Query SPHEREx observations for a given source position.

    Parameters
    ----------
    source : Source
        Target source with RA/Dec coordinates
    bands : List[str], optional
        List of bands to query (e.g., ['D1', 'D2']). If None, query all bands.
    cutout_size : str, optional
        Cutout size parameter (e.g., "200px", "3arcmin") used for file size estimation.
        If None, estimates full image size (~71.6 MB).

    Returns
    -------
    QueryResults
        Query results containing observation information with estimated file sizes
    """
    logger.info(f"Querying SPHEREx observations for source at RA={source.ra}, Dec={source.dec}")
    
    # Connect to TAP service
    service = pyvo.dal.TAPService(TAP_URL)
    
    # Build ADQL query
    query = f"""
    SELECT * FROM {SPHEREX_TABLE}
    WHERE CONTAINS(POINT('ICRS', {source.ra}, {source.dec}), s_region) = 1
    """
    
    # Add band filter if specified
    if bands:
        band_conditions = " OR ".join([f"energy_bandpassname = 'SPHEREx-{band}'" for band in bands])
        query += f" AND ({band_conditions})"
    
    query += " ORDER BY t_min"
    
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
        # Extract band from energy_bandpassname (format: 'SPHEREx-D1')
        band_name = row['energy_bandpassname']
        band = band_name.split('-')[-1] if '-' in band_name else band_name
        
        # Calculate MJD from t_min and t_max
        mjd = (row['t_min'] + row['t_max']) / 2.0
        
        # Convert wavelength from meters to microns
        wavelength_min = row['em_min'] * 1e6  # m to μm
        wavelength_max = row['em_max'] * 1e6  # m to μm
        
        # Estimate file size based on cutout parameters
        # TAP service doesn't provide accurate sizes for cutouts, so we estimate
        from ..utils.helpers import estimate_cutout_size_mb
        file_size_mb = estimate_cutout_size_mb(cutout_size, full_size_mb=FULL_IMAGE_SIZE_MB)
        
        obs = ObservationInfo(
            obs_id=row['obs_id'],
            band=band,
            mjd=mjd,
            ra=row['s_ra'],
            dec=row['s_dec'],
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            access_url=row['access_url'],
            file_size_mb=file_size_mb,
            t_min=row['t_min'],
            t_max=row['t_max']
        )
        observations.append(obs)
    
    # Calculate summary statistics
    band_counts = {}
    for band in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        count = sum(1 for obs in observations if obs.band == band)
        if count > 0:
            band_counts[band] = count
    
    total_size_gb = sum(obs.file_size_mb for obs in observations) / 1024.0
    
    # Calculate time span
    if observations:
        time_span_days = max(obs.mjd for obs in observations) - min(obs.mjd for obs in observations)
    else:
        time_span_days = 0.0
    
    query_results = QueryResults(
        observations=observations,
        query_time=datetime.now(),
        source=source,
        total_size_gb=total_size_gb,
        time_span_days=time_span_days,
        band_counts=band_counts
    )
    
    logger.info(f"Found {len(observations)} observations spanning {time_span_days:.1f} days")
    
    return query_results


def _resolve_single_url(
    obs: ObservationInfo,
    cutout_size: Optional[str] = None,
    cutout_center: Optional[str] = None,
    source_ra: Optional[float] = None,
    source_dec: Optional[float] = None
) -> Tuple[ObservationInfo, Optional[str]]:
    """
    Resolve a single datalink URL to actual download URL with optional cutout parameters.

    Parameters
    ----------
    obs : ObservationInfo
        Observation with datalink URL
    cutout_size : str, optional
        Cutout size parameter (e.g., "200px", "3arcmin")
    cutout_center : str, optional
        Cutout center parameter (e.g., "70,20") or None to use source position
    source_ra : float, optional
        Source RA in degrees (used if cutout_center is None and cutout_size is specified)
    source_dec : float, optional
        Source Dec in degrees (used if cutout_center is None and cutout_size is specified)

    Returns
    -------
    Tuple[ObservationInfo, Optional[str]]
        Observation and resolved URL (None if failed)
    """
    try:
        # Get datalink content
        datalink_content = DatalinkResults.from_result_url(obs.access_url)

        # Find the primary product (#this)
        primary_product = next(datalink_content.bysemantics("#this"))
        download_url = primary_product.access_url

        # Append cutout parameters if specified
        if cutout_size:
            from ..utils.helpers import format_cutout_url_params
            # Use source position if available
            ra = source_ra if source_ra is not None else obs.ra
            dec = source_dec if source_dec is not None else obs.dec
            cutout_params = format_cutout_url_params(cutout_size, cutout_center, ra, dec)
            download_url = download_url + cutout_params
            logger.debug(f"Added cutout parameters to {obs.obs_id}: {cutout_params}")

        return (obs, download_url)

    except Exception as e:
        logger.error(f"Failed to get download URL for {obs.obs_id}: {e}")
        return (obs, None)


def get_download_urls(
    query_results: QueryResults,
    max_workers: int = 4,
    show_progress: bool = True,
    cache_file: Optional[Path] = None,
    cutout_size: Optional[str] = None,
    cutout_center: Optional[str] = None,
    max_retries: int = 3
) -> List[Tuple[ObservationInfo, str]]:
    """
    Process datalink URLs to get actual FITS file download URLs in parallel.

    Parameters
    ----------
    query_results : QueryResults
        Query results containing observations with datalink URLs
    max_workers : int
        Maximum number of parallel workers (default: 4, matches download workers)
    show_progress : bool
        Whether to show progress bar
    cache_file : Path, optional
        Path to cache file for storing/loading URLs
    cutout_size : str, optional
        Cutout size parameter (e.g., "200px", "3arcmin")
    cutout_center : str, optional
        Cutout center parameter (e.g., "70,20") or None to use source position
    max_retries : int
        Maximum number of retry attempts if cache is incomplete (default: 3)

    Returns
    -------
    List[Tuple[ObservationInfo, str]]
        List of (observation, download_url) tuples
    """
    logger.info(f"Resolving download URLs for {len(query_results.observations)} observations")

    # Check retry counter to prevent infinite loops
    retry_counter_file = cache_file.parent / '.url_retry_count' if cache_file else None
    if retry_counter_file and retry_counter_file.exists():
        try:
            import json
            with open(retry_counter_file, 'r') as f:
                retry_data = json.load(f)
                retry_count = retry_data.get('count', 0)
                if retry_count >= max_retries:
                    logger.warning(
                        f"Maximum retry attempts ({max_retries}) reached for URL resolution. "
                        f"Proceeding with available cached URLs even if incomplete."
                    )
                    # Load whatever we have in cache and return
                    if cache_file and cache_file.exists():
                        try:
                            cached_urls = load_url_cache(cache_file)
                            obs_by_composite_key = {
                                f"{obs.obs_id}_{obs.band}": obs for obs in query_results.observations
                            }
                            matched_urls = [
                                (obs_by_composite_key[key], url)
                                for key, url in cached_urls.items()
                                if key in obs_by_composite_key
                            ]
                            logger.info(f"Loaded {len(matched_urls)} URLs from cache (incomplete)")
                            # Reset retry counter
                            retry_counter_file.unlink()
                            return matched_urls
                        except Exception as e:
                            logger.error(f"Failed to load cached URLs after max retries: {e}")
                            retry_counter_file.unlink()
                            return []
        except Exception as e:
            logger.warning(f"Failed to read retry counter: {e}")

    # Try to load from cache if available
    if cache_file and cache_file.exists():
        try:
            cached_urls = load_url_cache(cache_file)
            # Match cached URLs with current observations using composite key (obs_id, band)
            obs_by_composite_key = {
                f"{obs.obs_id}_{obs.band}": obs for obs in query_results.observations
            }
            matched_urls = []

            for composite_key, url in cached_urls.items():
                if composite_key in obs_by_composite_key:
                    matched_urls.append((obs_by_composite_key[composite_key], url))

            if len(matched_urls) == len(query_results.observations):
                logger.info(f"Loaded {len(matched_urls)} URLs from cache")
                return matched_urls
            else:
                logger.info(f"Cache incomplete ({len(matched_urls)}/{len(query_results.observations)}), resolving missing URLs")
        except Exception as e:
            logger.warning(f"Failed to load URL cache: {e}")
    
    download_urls = []
    failed_count = 0
    
    # Set up progress bar
    if show_progress:
        pbar = tqdm(total=len(query_results.observations), desc="Resolving URLs", unit="urls")
    
    # Get source coordinates for cutout centering
    source_ra = query_results.source.ra
    source_dec = query_results.source.dec

    # Process URLs in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with cutout parameters
        future_to_obs = {
            executor.submit(
                _resolve_single_url,
                obs,
                cutout_size,
                cutout_center,
                source_ra,
                source_dec
            ): obs
            for obs in query_results.observations
        }

        # Process completed tasks
        for future in as_completed(future_to_obs):
            obs, url = future.result()

            if url:
                download_urls.append((obs, url))
            else:
                failed_count += 1

            if show_progress:
                pbar.update(1)
    
    if show_progress:
        pbar.close()
    
    logger.info(f"Resolved {len(download_urls)} download URLs, {failed_count} failed")
    
    # Save to cache if requested
    if cache_file and download_urls:
        try:
            save_url_cache(download_urls, cache_file)
            logger.info(f"Saved URLs to cache: {cache_file}")

            # Update retry counter if cache is still incomplete
            if len(download_urls) < len(query_results.observations):
                if retry_counter_file:
                    import json
                    current_count = 0
                    if retry_counter_file.exists():
                        try:
                            with open(retry_counter_file, 'r') as f:
                                retry_data = json.load(f)
                                current_count = retry_data.get('count', 0)
                        except Exception:
                            pass
                    current_count += 1
                    retry_counter_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(retry_counter_file, 'w') as f:
                        json.dump({'count': current_count}, f)
                    logger.info(f"URL resolution incomplete ({len(download_urls)}/{len(query_results.observations)}). Retry count: {current_count}/{max_retries}")
            else:
                # Cache is complete, reset retry counter
                if retry_counter_file and retry_counter_file.exists():
                    retry_counter_file.unlink()
                    logger.info("URL resolution complete, reset retry counter")

        except Exception as e:
            logger.warning(f"Failed to save URL cache: {e}")

    return download_urls


def save_url_cache(download_urls: List[Tuple[ObservationInfo, str]], cache_file: Path) -> None:
    """
    Save download URLs to cache file.

    Parameters
    ----------
    download_urls : List[Tuple[ObservationInfo, str]]
        List of (observation, url) tuples
    cache_file : Path
        Cache file path
    """
    import json

    # Use composite key (obs_id, band) to handle multiple bands per obs_id
    cache_data = {
        f"{obs.obs_id}_{obs.band}": url for obs, url in download_urls
    }

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)


def load_url_cache(cache_file: Path) -> dict:
    """
    Load download URLs from cache file.
    
    Parameters
    ----------
    cache_file : Path
        Cache file path
    
    Returns
    -------
    dict
        Dictionary mapping obs_id to download URL
    """
    import json
    
    with open(cache_file, 'r') as f:
        return json.load(f)


def has_complete_url_cache(query_results: QueryResults, cache_file: Path) -> bool:
    """
    Check if cache contains URLs for all observations.

    Parameters
    ----------
    query_results : QueryResults
        Query results to check against
    cache_file : Path
        Cache file path

    Returns
    -------
    bool
        True if cache is complete
    """
    if not cache_file.exists():
        return False

    try:
        cached_urls = load_url_cache(cache_file)
        # Use composite keys (obs_id, band) to check completeness
        obs_composite_keys = {f"{obs.obs_id}_{obs.band}" for obs in query_results.observations}
        cached_keys = set(cached_urls.keys())
        return obs_composite_keys.issubset(cached_keys)
    except Exception:
        return False


def print_query_summary(query_results: QueryResults) -> None:
    """
    Print a summary of query results.
    
    Parameters
    ----------
    query_results : QueryResults
        Query results to summarize
    """
    print(f"\n{'='*60}")
    print("SPHEREx Archive Search Results")
    print(f"{'='*60}")
    print(f"Source: RA={query_results.source.ra:.6f}, Dec={query_results.source.dec:.6f}")
    if query_results.source.name:
        print(f"        Name: {query_results.source.name}")
    print(f"Query time: {query_results.query_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTotal observations found: {len(query_results)}")
    
    print("\nObservations by band:")
    for band in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        count = query_results.band_counts.get(band, 0)
        if count > 0:
            wl_range = BAND_WAVELENGTHS[band]
            print(f"  {band} ({wl_range[0]:.2f}-{wl_range[1]:.2f} μm): {count:3d} observations")
    
    print(f"\nTime span: {query_results.time_span_days:.1f} days")
    print(f"Total data volume: {query_results.total_size_gb:.2f} GB")
    print(f"{'='*60}\n")