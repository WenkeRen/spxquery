"""Batch photometry pipeline — orchestrates query, download, extract, aggregate."""

import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional

import yaml

from ..core.config import DownloadConfig, DownloadResult, PhotometryConfig, QueryResults
from ..core.download import parallel_download, print_download_summary
from .aggregate import aggregate_lightcurves
from .config import BatchConfig, load_catalog
from .extract import run_extraction
from .query import query_region_observations

logger = logging.getLogger(__name__)

_QUERY_SUMMARY_FILE = "query_summary.yaml"


def _py(obj):
    """Convert numpy scalars to plain Python types for YAML serialization."""
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _save_query_summary(results: QueryResults, config: BatchConfig) -> Path:
    """Serialize query results to a YAML summary file."""
    summary = {
        "query_time": results.query_time.isoformat(),
        "region": {
            "center_ra": _py(config.center_ra),
            "center_dec": _py(config.center_dec),
            "radius_deg": _py(config.radius),
            "coverage_mode": config.coverage_mode,
        },
        "filters": {
            "bands": config.bands,
            "mjd_range": [float(x) for x in config.mjd_range] if config.mjd_range else None,
        },
        "n_observations": len(results),
        "band_counts": {k: int(v) for k, v in results.band_counts.items()},
        "time_span_days": float(results.time_span_days),
        "observations": [
            {
                "obs_id": obs.obs_id,
                "band": obs.band,
                "mjd": round(float(obs.mjd), 6),
                "wavelength_um": round(float(obs.wavelength_center), 4),
                "download_url": obs.download_url,
            }
            for obs in results.observations
        ],
    }

    path = config.output_dir / _QUERY_SUMMARY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved query summary to {path}")
    return path


def load_query_summary(output_dir: Path) -> dict:
    """Load a previously saved query_summary.yaml.

    Parameters
    ----------
    output_dir : Path
        Root batch output directory containing the YAML file.
    """
    path = Path(output_dir) / _QUERY_SUMMARY_FILE
    if not path.exists():
        raise FileNotFoundError(f"No query summary found at {path}")
    with open(path) as f:
        return yaml.safe_load(f)


class BatchPipeline:
    """Multi-source batch photometry pipeline.

    Four stages: query -> download -> extract -> aggregate.
    Each stage can be run independently for resumable execution.

    Parameters
    ----------
    config : BatchConfig
        Region, catalog, and processing configuration.
    """

    def __init__(self, config: BatchConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.image_dir.mkdir(parents=True, exist_ok=True)
        self.config.per_image_dir.mkdir(parents=True, exist_ok=True)

        self._query_results = None

        # Ensure spawn method for clean worker environments
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    def run_query(self):
        """Stage 1: Query IRSA for full-frame images covering the region."""
        logger.info("Stage 1/4: Querying region observations")
        self._query_results = query_region_observations(self.config)
        _save_query_summary(self._query_results, self.config)
        logger.info(f"Found {len(self._query_results)} observations")
        return self._query_results

    def run_download(self, skip_existing: bool = True) -> List[DownloadResult]:
        """Stage 2: Download full-frame FITS images (no cutouts)."""
        if self._query_results is None:
            raise RuntimeError("Run run_query() first")

        logger.info("Stage 2/4: Downloading full-frame images")

        # No cutout parameters — download full images
        download_info = [(obs, obs.download_url) for obs in self._query_results.observations]

        if not download_info:
            logger.warning("No observations to download")
            return []

        total_files = len(download_info)
        logger.info(f"Downloading {total_files} full-frame images (~{total_files * 0.07:.0f} GB)...")

        download_config = DownloadConfig(max_download_workers=self.config.max_download_workers)

        results = parallel_download(
            download_info,
            self.config.image_dir,
            max_workers=self.config.max_download_workers,
            skip_existing=skip_existing,
            download_config=download_config,
        )

        print_download_summary(results)
        n_success = sum(1 for r in results if r.success)
        logger.info(f"Downloaded {n_success}/{total_files} images")
        return results

    def run_extract(self, skip_existing: bool = True) -> int:
        """Stage 3: Extract multi-source photometry from each image."""
        logger.info("Stage 3/4: Extracting photometry")

        sources = load_catalog(self.config.catalog_path)
        logger.info(f"Loaded {len(sources)} sources from catalog")

        image_files = list(self.config.image_dir.rglob("*.fits"))
        if not image_files:
            raise FileNotFoundError(f"No FITS files found in {self.config.image_dir}")

        logger.info(f"Found {len(image_files)} FITS images")

        n_new = run_extraction(
            image_dir=self.config.image_dir,
            sources=sources,
            config=self.config.photometry,
            output_dir=self.config.per_image_dir,
            n_workers=self.config.max_extract_workers,
            skip_existing=skip_existing,
        )

        logger.info(f"Stage 3 complete: {n_new} new per-image CSVs")
        return n_new

    def run_aggregate(self, clean: bool = False) -> int:
        """Stage 4: Aggregate per-image CSVs into per-source light curves."""
        logger.info("Stage 4/4: Aggregating light curves")

        n_sources = aggregate_lightcurves(
            image_csv_dir=self.config.per_image_dir,
            lightcurve_dir=self.config.lightcurve_dir,
            bucket_dir=self.config.bucket_dir,
            num_buckets=self.config.num_buckets,
            clean=clean,
            keep_bucket_files=self.config.keep_bucket_files,
        )

        logger.info(f"Created {n_sources} light curve files")
        return n_sources

    def run_all(self, skip_existing: bool = True):
        """Run all four stages sequentially."""
        logger.info("=" * 60)
        logger.info("Batch Photometry Pipeline")
        logger.info(f"Region: RA={self.config.center_ra}, Dec={self.config.center_dec}, "
                     f"radius={self.config.radius} deg")
        logger.info("=" * 60)

        self.run_query()
        self.run_download(skip_existing=skip_existing)
        self.run_extract(skip_existing=skip_existing)
        self.run_aggregate()

        logger.info("Pipeline complete.")


def run_batch(
    catalog: str,
    center_ra: float,
    center_dec: float,
    radius: float,
    output_dir: str = "batch_output",
    bands: Optional[List[str]] = None,
    coverage_mode: str = "any",
    max_images: int = 500,
    max_download_workers: int = 4,
    max_extract_workers: int = 12,
    skip_existing: bool = True,
    photometry_config: Optional[PhotometryConfig] = None,
) -> BatchPipeline:
    """Run the full batch photometry pipeline with one function call.

    Parameters
    ----------
    catalog : str
        Path to CSV with columns targetid, ra, dec.
    center_ra, center_dec : float
        Region center in degrees.
    radius : float
        Region radius in degrees.
    output_dir : str
        Root output directory.
    bands : list of str or None
        Bands to query. None = all.
    coverage_mode : str
        ``"any"`` (INTERSECTS) or ``"full"`` (CONTAINS).
    max_images : int
        Safety gate — raise if exceeded.
    max_download_workers : int
        Parallel download threads.
    max_extract_workers : int
        Parallel extraction processes.
    skip_existing : bool
        Resume mode — skip already-processed images.
    photometry_config : PhotometryConfig or None
        Override default photometry parameters.

    Returns
    -------
    BatchPipeline
        The pipeline instance (for inspecting results).
    """
    config = BatchConfig(
        catalog_path=Path(catalog),
        center_ra=center_ra,
        center_dec=center_dec,
        radius=radius,
        output_dir=Path(output_dir),
        bands=bands,
        coverage_mode=coverage_mode,
        max_images=max_images,
        max_download_workers=max_download_workers,
        max_extract_workers=max_extract_workers,
        photometry=photometry_config or PhotometryConfig(),
    )

    pipeline = BatchPipeline(config)
    pipeline.run_all(skip_existing=skip_existing)
    return pipeline
