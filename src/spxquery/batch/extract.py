"""Multi-source aperture photometry extraction from SPHEREx images."""

import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import List, Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from tqdm.auto import tqdm

from ..core.config import PhotometryConfig, Source
from ..processing.magnitudes import calculate_ab_magnitude_from_jy
from ..processing.photometry import (
    extract_aperture_photometry_with_background,
    process_flags_in_aperture,
    repair_variance_for_flagged_pixels,
)
from ..utils.spherex_mef import (
    get_pixel_scale_at_position,
    get_wavelength_at_position,
    read_spherex_mef,
    subtract_zodiacal_background,
)

logger = logging.getLogger(__name__)


def _init_worker():
    """Limit per-worker threads and suppress noisy logs."""
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    for name in ["spxquery", "spxquery.processing", "spxquery.utils"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)


def process_single_image(
    image_path: Path,
    sources: List[Source],
    config: PhotometryConfig,
    output_dir: Path,
    skip_existing: bool = True,
) -> Optional[Path]:
    """Extract aperture photometry for all catalog sources in one image.

    Reads the MEF once, projects all sources via WCS, filters to those
    in the field of view, then loops over in-FOV sources.

    Parameters
    ----------
    image_path : Path
        Path to a SPHEREx MEF FITS file.
    sources : list of Source
        All catalog sources to check.
    config : PhotometryConfig
        Photometry extraction parameters.
    output_dir : Path
        Directory for per-image CSV output.
    skip_existing : bool
        Skip images that already have an output CSV.

    Returns
    -------
    Path or None
        Path to the output CSV, or None if skipped / no results.
    """
    if skip_existing:
        output_csv = output_dir / f"{image_path.stem}_photometry.csv"
        if output_csv.exists():
            return None

    try:
        mef = read_spherex_mef(image_path, target_unit="uJy/arcsec2")
        mef.variance = repair_variance_for_flagged_pixels(mef.variance, mef.flags)

        image, zodi_scale = subtract_zodiacal_background(
            mef.image,
            mef.zodi,
            mef.flags,
            mef.variance,
            config.zodi_scale_min,
            config.zodi_scale_max,
        )

        ny, nx = image.shape
        obs_id = mef.header.get("OBSID", image_path.stem)
        detector_num = mef.detector
        band = f"D{detector_num}" if 1 <= detector_num <= 6 else "Unknown"
        mjd = mef.mjd

    except Exception as e:
        logger.error(f"Failed to load {image_path.name}: {e}")
        return None

    # Batch WCS projection — project all sources at once
    try:
        required_margin = max(config.aperture_diameter / 2.0, config.max_outer_radius)
        src_coords = SkyCoord(
            ra=[s.ra for s in sources] * u.deg,
            dec=[s.dec for s in sources] * u.deg,
        )
        px, py = mef.spatial_wcs.world_to_pixel(src_coords)
        in_bounds = (
            (px >= required_margin)
            & (px < nx - required_margin)
            & (py >= required_margin)
            & (py < ny - required_margin)
        )
        candidates = [
            (s, float(px[i]), float(py[i])) for i, s in enumerate(sources) if in_bounds[i]
        ]
    except Exception:
        candidates = []

    # Compute aperture radius once per image
    if config.aperture_method == "fwhm":
        fwhm_arcsec = mef.psf_fwhm
        pixel_scale_arcsec = mef.get_pixel_scale(
            nx / 2.0, ny / 2.0, fallback=config.pixel_scale_fallback
        )
        fwhm_pixels = fwhm_arcsec / pixel_scale_arcsec
        aperture_diameter = fwhm_pixels * config.fwhm_multiplier
        final_aperture_radius = aperture_diameter / 2.0
    else:
        final_aperture_radius = config.aperture_diameter / 2.0

    # Extract photometry for each in-FOV source
    results = []
    for source, x, y in candidates:
        try:
            wavelength, bandwidth = get_wavelength_at_position(mef, x, y)

            (
                flux_sum_uJy_per_arcsec2,
                flux_error_sum_uJy_per_arcsec2,
                bg_level,
                bg_error,
                n_bg_pixels,
            ) = extract_aperture_photometry_with_background(
                image,
                mef.variance,
                mef.flags,
                x,
                y,
                final_aperture_radius,
                config.background_method,
                config.window_size,
                config.min_usable_pixels,
                config.max_outer_radius,
                config.bg_sigma_clip_sigma,
                config.bg_sigma_clip_maxiters,
                config.max_annulus_attempts,
                config.annulus_expansion_step,
                config.annulus_inner_offset,
            )

            if n_bg_pixels == 0:
                continue

            pixel_scale_arcsec = get_pixel_scale_at_position(
                mef.spatial_wcs, x, y, config.pixel_scale_fallback
            )
            pixel_area_arcsec2 = pixel_scale_arcsec**2
            flux_ujy = flux_sum_uJy_per_arcsec2 * pixel_area_arcsec2
            flux_error_ujy = flux_error_sum_uJy_per_arcsec2 * pixel_area_arcsec2
            flux_jy = flux_ujy / 1e6
            flux_error_jy = flux_error_ujy / 1e6

            combined_flag = process_flags_in_aperture(mef.flags, x, y, final_aperture_radius)
            mag_ab, mag_ab_error = calculate_ab_magnitude_from_jy(flux_jy, flux_error_jy, wavelength)

            results.append({
                "target_id": source.name,
                "ra": source.ra,
                "dec": source.dec,
                "obs_id": obs_id,
                "band": band,
                "mjd": mjd,
                "x": x,
                "y": y,
                "flux": flux_ujy,
                "flux_error": flux_error_ujy,
                "mag_ab": mag_ab,
                "mag_ab_error": mag_ab_error,
                "wavelength": wavelength,
                "bandwidth": bandwidth,
                "flag": combined_flag,
                "bg_level": bg_level,
                "bg_error": bg_error,
                "aperture_radius": final_aperture_radius,
                "filename": image_path.name,
            })

        except Exception as e:
            logger.debug(f"Error processing {source.name} in {image_path.name}: {e}")
            continue

    if results:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = output_dir / f"{image_path.stem}_photometry.csv"
        try:
            pd.DataFrame(results).to_csv(output_filename, index=False)
            return output_filename
        except Exception as e:
            logger.error(f"Failed to save {image_path.name}: {e}")
            return None
    else:
        return None


def run_extraction(
    image_dir: Path,
    sources: List[Source],
    config: PhotometryConfig,
    output_dir: Path,
    n_workers: int = 12,
    skip_existing: bool = True,
) -> int:
    """Run multi-source extraction across all images in a directory.

    Parameters
    ----------
    image_dir : Path
        Directory containing SPHEREx FITS files (searched recursively).
    sources : list of Source
        Catalog sources to extract photometry for.
    config : PhotometryConfig
        Photometry parameters.
    output_dir : Path
        Per-image CSV output directory.
    n_workers : int
        Number of parallel workers.
    skip_existing : bool
        Skip images with existing output CSVs.

    Returns
    -------
    int
        Number of newly processed images.
    """
    image_files = sorted(image_dir.rglob("*.fits"))
    if not image_files:
        logger.warning(f"No FITS files found in {image_dir}")
        return 0

    n_existing = len(list(output_dir.glob("*_photometry.csv"))) if output_dir.exists() else 0
    if n_existing > 0 and skip_existing:
        logger.info(f"Incremental mode: {n_existing} CSVs already exist")

    output_dir.mkdir(parents=True, exist_ok=True)

    process_func = partial(
        process_single_image,
        sources=sources,
        config=config,
        output_dir=output_dir,
        skip_existing=skip_existing,
    )

    logger.info(f"Processing {len(image_files)} images with {n_workers} workers...")
    valid_count = 0

    with mp.Pool(processes=n_workers, initializer=_init_worker) as pool:
        progress = tqdm(
            pool.imap_unordered(process_func, image_files),
            total=len(image_files),
            desc="Extracting photometry",
            unit="image",
        )
        for result in progress:
            if result is not None:
                valid_count += 1

    logger.info(f"Extraction complete: {valid_count} new CSVs")
    return valid_count
