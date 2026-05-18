"""Aggregate per-image photometry CSVs into per-source light curves."""

import hashlib
import logging
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

AGGREGATE_COLUMNS = [
    "target_id",
    "obs_id",
    "band",
    "mjd",
    "x",
    "y",
    "flux",
    "flux_error",
    "mag_ab",
    "mag_ab_error",
    "wavelength",
    "bandwidth",
    "flag",
    "bg_level",
    "bg_error",
    "aperture_radius",
]

AGGREGATE_DTYPES = {
    "target_id": "string",
    "obs_id": "string",
    "band": "string",
}


def _bucket_file_path(bucket_dir: Path, bucket_id: int) -> Path:
    return bucket_dir / f"bucket_{bucket_id:03d}.csv"


def _stable_bucket_id(target_id: str, num_buckets: int) -> int:
    digest = hashlib.blake2b(str(target_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") % num_buckets


def _read_aggregate_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        usecols=AGGREGATE_COLUMNS,
        dtype=AGGREGATE_DTYPES,
        low_memory=False,
    )


def _clear_matching_csvs(directory: Path, pattern: str) -> int:
    removed = 0
    for csv_file in directory.glob(pattern):
        csv_file.unlink()
        removed += 1
    return removed


def _bucket_per_image_photometry(
    image_csv_dir: Path,
    bucket_dir: Path,
    num_buckets: int,
) -> tuple[list[Path], int, int]:
    """Partition per-image CSV rows into deterministic bucket files."""
    csv_files = sorted(image_csv_dir.glob("*_photometry.csv"))
    if not csv_files:
        logger.warning("No per-image CSV files found")
        return [], 0, 0

    bucket_paths = [_bucket_file_path(bucket_dir, i) for i in range(num_buckets)]
    bucket_has_header = [False] * num_buckets
    total_rows = 0

    logger.info(f"Phase 1/2: Bucketing {len(csv_files)} CSV files into {num_buckets} buckets...")
    for csv_file in tqdm(csv_files, desc="Bucketing CSVs", unit="file"):
        try:
            df = _read_aggregate_csv(csv_file)
        except Exception as e:
            logger.warning(f"Failed to read {csv_file.name}: {e}")
            continue

        if df.empty:
            continue

        total_rows += len(df)
        df["_bucket_id"] = df["target_id"].map(lambda tid: _stable_bucket_id(tid, num_buckets))

        for bucket_id, bucket_df in df.groupby("_bucket_id", sort=False):
            idx = int(bucket_id)
            bucket_df.drop(columns="_bucket_id").to_csv(
                bucket_paths[idx],
                mode="a",
                header=not bucket_has_header[idx],
                index=False,
            )
            bucket_has_header[idx] = True

    nonempty = [p for p in bucket_paths if p.exists()]
    logger.info(f"Phase 1/2 done: {len(csv_files)} files -> {len(nonempty)} buckets, {total_rows} rows")
    return nonempty, len(csv_files), total_rows


def _tid_to_filename(tid) -> str:
    """Format target_id as integer string for filenames (no scientific notation)."""
    s = str(tid)
    try:
        return f"{int(float(s)):d}"
    except (ValueError, OverflowError):
        return s


def _materialize_lightcurves_from_buckets(bucket_paths: list[Path], output_dir: Path) -> int:
    """Sort one bucket at a time and write per-source light curve CSVs."""
    if not bucket_paths:
        return 0

    total_sources = 0
    logger.info(f"Phase 2/2: Writing light curves from {len(bucket_paths)} buckets...")
    for bucket_path in tqdm(bucket_paths, desc="Writing lightcurves", unit="bucket"):
        try:
            df = _read_aggregate_csv(bucket_path)
        except Exception as e:
            logger.warning(f"Failed to read {bucket_path.name}: {e}")
            continue

        if df.empty:
            continue

        df.sort_values(["target_id", "mjd"], kind="mergesort", inplace=True)

        for target_id, group in df.groupby("target_id", sort=False):
            output_file = output_dir / f"{_tid_to_filename(target_id)}.csv"
            group.drop(columns="target_id").to_csv(output_file, index=False)
            total_sources += 1

    logger.info(f"Phase 2/2 done: wrote {total_sources} light curves")
    return total_sources


def aggregate_lightcurves(
    image_csv_dir: Path,
    lightcurve_dir: Path,
    bucket_dir: Path,
    num_buckets: int = 64,
    clean: bool = False,
    keep_bucket_files: bool = False,
) -> int:
    """Aggregate per-image CSVs into individual source light curves.

    Two-phase bucket design keeps memory bounded:
      1. Stream per-image CSVs into hash-partitioned bucket files.
      2. Process one bucket at a time, sort, write per-source CSVs.
    """
    bucket_dir.mkdir(parents=True, exist_ok=True)
    lightcurve_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        nb = _clear_matching_csvs(bucket_dir, "bucket_*.csv")
        nl = _clear_matching_csvs(lightcurve_dir, "*.csv")
        if nb > 0 or nl > 0:
            logger.info(f"Cleaned: {nb} bucket files, {nl} light curves")
    else:
        if list(bucket_dir.glob("bucket_*.csv")):
            raise FileExistsError(
                f"Bucket files exist in {bucket_dir}. Use clean=True to remove."
            )
        if list(lightcurve_dir.glob("*.csv")):
            raise FileExistsError(
                f"Light curves exist in {lightcurve_dir}. Use clean=True to remove."
            )

    bucket_paths, _, _ = _bucket_per_image_photometry(image_csv_dir, bucket_dir, num_buckets)
    n_sources = _materialize_lightcurves_from_buckets(bucket_paths, lightcurve_dir)

    if not keep_bucket_files:
        removed = _clear_matching_csvs(bucket_dir, "bucket_*.csv")
        logger.info(f"Removed {removed} temporary bucket files")

    logger.info(f"Aggregation complete: {n_sources} light curves in {lightcurve_dir}")
    return n_sources
