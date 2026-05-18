"""Configuration and utilities for batch photometry."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ..core.config import PhotometryConfig, Source

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for multi-source batch photometry over a sky region.

    Parameters
    ----------
    center_ra, center_dec : float
        Sky region center in degrees (ICRS).
    radius : float
        Search radius in degrees.
    catalog_path : Path
        CSV file with columns ``targetid``, ``ra``, ``dec``.
    coverage_mode : str
        ``"any"`` (INTERSECTS) or ``"full"`` (CONTAINS).
    bands : list of str or None
        Bands to query, e.g. ``["D1", "D3"]``.  ``None`` = all.
    mjd_range : tuple of (float, float) or None
        ``(mjd_min, mjd_max)`` to filter observations by time.
        ``None`` = no time filter (all epochs).
    max_images : int
        Safety gate — raise if query returns more images than this.
    output_dir : Path
        Root directory for all batch outputs.
    max_download_workers : int
        Parallel download threads.
    max_extract_workers : int
        Parallel extraction processes (spawn-based).
    photometry : PhotometryConfig
        Photometry parameters forwarded to extraction.
    num_buckets : int
        Hash-partition buckets for aggregation.
    keep_bucket_files : bool
        Keep temporary bucket CSVs after aggregation.
    """

    center_ra: float
    center_dec: float
    radius: float
    catalog_path: Path
    coverage_mode: str = "any"
    bands: Optional[List[str]] = None
    mjd_range: Optional[Tuple[float, float]] = None
    max_images: int = 500
    output_dir: Path = field(default_factory=lambda: Path("batch_output"))
    max_download_workers: int = 4
    max_extract_workers: int = 12
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    num_buckets: int = 64
    keep_bucket_files: bool = False

    def __post_init__(self):
        if not 0 <= self.center_ra <= 360:
            raise ValueError(f"center_ra must be 0-360 deg, got {self.center_ra}")
        if not -90 <= self.center_dec <= 90:
            raise ValueError(f"center_dec must be -90 to 90 deg, got {self.center_dec}")
        if self.radius <= 0:
            raise ValueError(f"radius must be > 0, got {self.radius}")
        if self.coverage_mode not in ("any", "full"):
            raise ValueError(f"coverage_mode must be 'any' or 'full', got '{self.coverage_mode}'")
        if self.max_images <= 0:
            raise ValueError(f"max_images must be > 0, got {self.max_images}")
        if self.mjd_range is not None:
            mjd_min, mjd_max = self.mjd_range
            if mjd_min >= mjd_max:
                raise ValueError(f"mjd_range must be (min, max) with min < max, got {self.mjd_range}")
        self.catalog_path = Path(self.catalog_path)
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {self.catalog_path}")
        self.output_dir = Path(self.output_dir)

    @property
    def image_dir(self) -> Path:
        return self.output_dir / "images"

    @property
    def per_image_dir(self) -> Path:
        return self.output_dir / "per_image"

    @property
    def lightcurve_dir(self) -> Path:
        return self.output_dir / "lightcurves"

    @property
    def bucket_dir(self) -> Path:
        return self.output_dir / "_aggregate_temp"


def load_catalog(catalog_path: Path) -> List[Source]:
    """Load a source catalog CSV into a list of Source objects.

    Expected columns: ``targetid``, ``ra``, ``dec``.
    """
    df = pd.read_csv(catalog_path, dtype={"targetid": str})
    required = {"targetid", "ra", "dec"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Catalog missing required columns: {missing}")

    sources = [
        Source(ra=row["ra"], dec=row["dec"], name=row["targetid"])
        for _, row in df.iterrows()
    ]
    logger.info(f"Loaded {len(sources)} sources from {catalog_path.name}")
    return sources
