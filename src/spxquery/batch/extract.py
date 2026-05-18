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
from photutils.aperture import CircularAperture, aperture_photometry
from tqdm.auto import tqdm

from ..core.config import PhotometryConfig, Source
from ..processing.magnitudes import calculate_ab_magnitude_from_jy
from ..processing.photometry import (
    process_flags_in_aperture,
    repair_variance_for_flagged_pixels,
)
from ..utils.spherex_mef import (
    _BAD_FLAG_BITS,
    _SOURCE_BIT,
    read_spherex_mef,
    subtract_zodiacal_background,
)

logger = logging.getLogger(__name__)

# Derived masks: strict (all bad bits including SOURCE), relaxed (exclude SOURCE)
_BAD_BITS_STRICT = _BAD_FLAG_BITS | _SOURCE_BIT
_BAD_BITS_RELAXED = _BAD_FLAG_BITS


def _fast_sigma_clip(data: np.ndarray, sigma: float = 3.0, maxiters: int = 3):
    """Sigma-clipped stats using numpy directly (avoids astropy object overhead)."""
    clipped = data.ravel()
    for _ in range(maxiters):
        m = np.mean(clipped)
        s = np.std(clipped)
        if s == 0:
            break
        mask = np.abs(clipped - m) <= sigma * s
        if mask.all():
            break
        clipped = clipped[mask]
    return float(np.mean(clipped)), float(np.median(clipped)), float(np.std(clipped))


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

    Optimized for batch processing: pre-computes shared arrays (background mask,
    error map, pixel scale) once per image, then uses local cutouts for per-source
    photometry instead of operating on the full image.

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

    # --- Pre-compute per-image shared data ---
    error_array = np.sqrt(mef.variance)
    bg_mask_strict = (mef.flags & _BAD_BITS_STRICT) == 0
    bg_mask_relaxed = (mef.flags & _BAD_BITS_RELAXED) == 0

    pixel_scale_arcsec = mef.get_pixel_scale(
        nx / 2.0, ny / 2.0, fallback=config.pixel_scale_fallback
    )
    pixel_area_arcsec2 = pixel_scale_arcsec**2

    if config.aperture_method == "fwhm":
        fwhm_arcsec = mef.psf_fwhm
        fwhm_pixels = fwhm_arcsec / pixel_scale_arcsec
        aperture_diameter = fwhm_pixels * config.fwhm_multiplier
        final_aperture_radius = aperture_diameter / 2.0
    else:
        final_aperture_radius = config.aperture_diameter / 2.0

    # Aperture cutout half-size
    ri = int(np.ceil(final_aperture_radius)) + 1

    # Geometric aperture area for background subtraction (matches original photutils path)
    aperture_area = np.pi * final_aperture_radius**2

    # Window background parameters
    wh = ww = config.window_size

    # Required margin from image edge (ceil(wh/2)+1 accounts for floor/ceil window boundaries)
    required_margin = max(config.aperture_diameter / 2.0, config.max_outer_radius, wh // 2 + 1)

    # Batch WCS projection
    try:
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
            (i, float(px[i]), float(py[i]))
            for i in range(len(sources)) if in_bounds[i]
        ]
    except Exception:
        candidates = []

    if not candidates:
        return None

    # Batch spectral WCS
    cpx = np.array([c[1] for c in candidates])
    cpy = np.array([c[2] for c in candidates])
    try:
        spectral_coords = mef.spectral_wcs.pixel_to_world(cpx, cpy)
        wavelengths = spectral_coords[0].to(u.micron).value
        bandwidths = spectral_coords[1].to(u.micron).value
    except Exception:
        wavelengths = None
        bandwidths = None

    # Per-source extraction
    sigma = config.bg_sigma_clip_sigma
    maxiters = config.bg_sigma_clip_maxiters
    min_usable = config.min_usable_pixels

    results = []
    for idx, (src_idx, x, y) in enumerate(candidates):
        source = sources[src_idx]
        try:
            if wavelengths is not None:
                wavelength = float(wavelengths[idx])
                bandwidth = float(bandwidths[idx])
            else:
                wavelength, bandwidth = mef.pixel_to_wavelength(x, y)

            ix, iy = int(round(x)), int(round(y))

            # --- Window background on local cutout ---
            # Match original estimate_window_boundary: floor/ceil gives ±1 pixel
            # variation depending on fractional position of the source.
            wy0 = int(np.floor(y - wh / 2.0))
            wy1 = int(np.ceil(y + wh / 2.0))
            wx0 = int(np.floor(x - ww / 2.0))
            wx1 = int(np.ceil(x + ww / 2.0))

            img_win = image[wy0:wy1, wx0:wx1]
            bg_qual = bg_mask_strict[wy0:wy1, wx0:wx1]

            # Aperture exclusion: distance > aperture_radius + sqrt(2)/2
            yy_w, xx_w = np.ogrid[wy0:wy1, wx0:wx1]
            dist_sq = (xx_w - x) ** 2 + (yy_w - y) ** 2
            excl_r = final_aperture_radius + 0.707
            usable = bg_qual & (dist_sq > excl_r * excl_r)

            bg_pixels = img_win[usable]
            n_bg = len(bg_pixels)

            if n_bg < min_usable:
                bg_qual_relax = bg_mask_relaxed[wy0:wy1, wx0:wx1]
                usable = bg_qual_relax & (dist_sq > excl_r * excl_r)
                bg_pixels = img_win[usable]
                n_bg = len(bg_pixels)
                if n_bg < min_usable:
                    continue

            _, bg_level, bg_std = _fast_sigma_clip(bg_pixels, sigma, maxiters)
            bg_error = bg_std / np.sqrt(n_bg)

            # --- Aperture photometry on local cutout (uses photutils for exact overlap) ---
            cutout_pad = ri + 1
            ay0 = iy - cutout_pad
            ay1 = iy + cutout_pad + 1
            ax0 = ix - cutout_pad
            ax1 = ix + cutout_pad + 1

            img_cut = image[ay0:ay1, ax0:ax1]
            err_cut = error_array[ay0:ay1, ax0:ax1]

            x_c = x - ax0
            y_c = y - ay0
            aperture = CircularAperture((x_c, y_c), r=final_aperture_radius)
            phot = aperture_photometry(img_cut, aperture, error=err_cut)

            raw_flux = float(phot["aperture_sum"][0])
            raw_flux_error = float(phot["aperture_sum_err"][0])

            # Background subtraction using geometric area (matches original)
            bg_total = bg_level * aperture_area
            bg_error_total = bg_error * aperture_area

            net_flux = raw_flux - bg_total
            net_error = np.sqrt(raw_flux_error**2 + bg_error_total**2)

            # Unit conversion
            flux_ujy = net_flux * pixel_area_arcsec2
            flux_error_ujy = net_error * pixel_area_arcsec2

            combined_flag = process_flags_in_aperture(mef.flags, x, y, final_aperture_radius)

            flux_jy = flux_ujy / 1e6
            flux_error_jy = flux_error_ujy / 1e6
            mag_ab, mag_ab_error = calculate_ab_magnitude_from_jy(flux_jy, flux_error_jy, wavelength)

            results.append({
                "target_id": str(source.name),
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
