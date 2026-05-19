"""
IRSA TAP query for SPHEREx observations intersecting a sky region.

Delegates to spxquery's shared TAP infrastructure in core.query, then
groups results by detector for drizzle processing.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..core.query import query_spherex_observations
from .config import Drizzle3DConfig

logger = logging.getLogger(__name__)

# IRSA IBE data prefix stripped when resolving local mirror paths.
# download_url = "https://irsa.ipac.caltech.edu/ibe/data/spherex/qr2/..."
# local mirror  = data_mirror / "qr2/..."
_IRSA_DATA_PREFIX = "https://irsa.ipac.caltech.edu/ibe/data/spherex"


@dataclass
class DrizzleObservation:
    """Minimal observation descriptor for drizzle downloads."""

    obs_id: str
    band: str  # e.g. "D3"
    detector: int  # 1–6
    mjd: float
    download_url: str


def query_observations(config: Drizzle3DConfig) -> Dict[int, List[DrizzleObservation]]:
    """Query IRSA TAP for observations INTERSECTING the target sky region.

    Delegates to :func:`spxquery.core.query.query_spherex_observations`
    and converts the results into detector-grouped ``DrizzleObservation`` objects.

    Parameters
    ----------
    config : Drizzle3DConfig
        Drizzle configuration with center, size, detector, and mjd_range.

    Returns
    -------
    dict
        {detector_id: [DrizzleObservation, ...]} grouped by detector.
        Only detectors with results are included.
    """
    from ..core.config import Source

    source = Source(ra=config.center_ra, dec=config.center_dec)

    bands = None
    if config.detector != 0:
        bands = [f"D{config.detector}"]

    logger.info(
        f"Querying IRSA TAP: center=({config.center_ra:.4f}, {config.center_dec:.4f}), "
        f"detector={config.detector or 'all'}"
    )

    query_results = query_spherex_observations(
        source,
        bands=bands,
        mjd_range=config.mjd_range,
        max_images=config.max_images,
    )

    # Convert ObservationInfo → DrizzleObservation, grouped by detector
    by_detector: Dict[int, List[DrizzleObservation]] = {}
    for obs in query_results.observations:
        detector = int(obs.band[1]) if obs.band.startswith("D") else 0
        drizzle_obs = DrizzleObservation(
            obs_id=obs.obs_id,
            band=obs.band,
            detector=detector,
            mjd=obs.mjd,
            download_url=obs.download_url,
        )
        by_detector.setdefault(detector, []).append(drizzle_obs)

    for det, obs_list in sorted(by_detector.items()):
        logger.info(f"  D{det}: {len(obs_list)} observations")

    return by_detector


def download_observations(
    observations: List[DrizzleObservation],
    output_dir,
    max_workers: int = 4,
    skip_existing: bool = True,
    data_mirror: Optional[Path] = None,
) -> List:
    """Resolve FITS file paths for a list of observations.

    Two modes:

    - **Mirror mode** (``data_mirror`` is set): constructs local paths from the
      IRSA download URL by stripping ``/ibe/data/spherex`` and prepending
      ``data_mirror``. Verifies each file exists. No network access.
    - **Download mode** (default): downloads FITS files from IRSA via HTTP,
      delegating to :func:`spxquery.core.download.download_file`.

    Parameters
    ----------
    observations : list of DrizzleObservation
        Observations to resolve.
    output_dir : Path
        Base directory; files go into ``output_dir/images/D{N}/`` (download mode only).
    max_workers : int
        Parallel download threads (download mode only).
    skip_existing : bool
        Skip already-downloaded files (download mode only).
    data_mirror : Path, optional
        Local mirror root directory. When set, resolves paths from the mirror
        instead of downloading.

    Returns
    -------
    list of Path
        Paths to resolved FITS files.
    """
    output_dir = Path(output_dir)
    det = observations[0].detector if observations else 0

    # ── Mirror mode: resolve local paths, verify existence ────────────────
    if data_mirror is not None:
        data_mirror = Path(data_mirror)
        resolved = []
        missing = 0
        for obs in observations:
            relative = obs.download_url.removeprefix(_IRSA_DATA_PREFIX).lstrip("/")
            local_path = data_mirror / relative
            if local_path.exists():
                resolved.append(local_path)
            else:
                logger.warning(f"Mirror file not found: {local_path}")
                missing += 1
        logger.info(
            f"D{det}: {len(resolved)}/{len(observations)} files resolved from mirror"
            + (f" ({missing} missing)" if missing else "")
        )
        return resolved

    # ── Download mode: delegate to core.download ─────────────────────────
    from ..core.download import download_file

    det_dir = output_dir / "images" / f"D{det}"
    det_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for obs in observations:
        filename = f"{obs.obs_id}.fits"
        filepath = det_dir / filename

        if skip_existing and filepath.exists():
            downloaded.append(filepath)
            continue

        result = download_file(obs.download_url, filepath)
        if result.success:
            downloaded.append(filepath)
        else:
            logger.error(f"Failed to download {obs.obs_id}: {result.error}")

    logger.info(f"Downloaded {len(downloaded)}/{len(observations)} files for D{det}")
    return downloaded
