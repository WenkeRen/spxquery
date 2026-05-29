"""
High-level Drizzle3D pipeline.

Orchestrates: query → download → per-detector drizzle → save.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from astropy.wcs import WCS

from .accumulate import DrizzleCube, drizzle_image
from .config import Drizzle3DConfig
from .grid import build_output_wcs
from .io import save_cube
from .query import download_observations, query_observations
from .spatial import compute_spatial_mapping
from .spectral import build_z_grid

logger = logging.getLogger(__name__)


def drizzle_detector(
    fits_paths: List[Path],
    config: Drizzle3DConfig,
    detector: int,
    output_wcs: WCS,
) -> Optional[Path]:
    """Run the drizzle pipeline for one detector.

    Parameters
    ----------
    fits_paths : list of Path
        Downloaded FITS files for this detector.
    config : Drizzle3DConfig
        Drizzle configuration.
    detector : int
        Detector number (1–6).
    output_wcs : WCS
        Output spatial WCS.

    Returns
    -------
    Path or None
        Path to the output FITS file, or None if no valid inputs.
    """
    # Build Z grid
    zgrid = build_z_grid(detector, config.z_oversample, config.z_lambda_edges)
    pixscale = config.effective_pixscale()

    # Allocate accumulation cube
    cube = DrizzleCube.create(output_wcs, pixscale, zgrid, config, detector)
    output_shape = (config.output_ny(), config.output_nx())

    # Build exclude mask from config.exclude_flags
    from ..utils.helpers import create_flag_mask

    exclude_bits = create_flag_mask(config.exclude_flags)

    n_rejected = 0

    for fpath in fits_paths:
        try:
            img_data, var_data, flag_data, zodi_data, spatial_wcs, spectral_wcs = _read_input_fits(
                fpath, config.subtract_zodi, static_zodi=config.static_zodi
            )
        except Exception as e:
            logger.warning(f"Skipping {fpath.name}: {e}")
            n_rejected += 1
            continue

        if spatial_wcs is None:
            logger.warning(f"Skipping {fpath.name}: no spatial WCS")
            n_rejected += 1
            continue

        # Compute spatial mapping
        valid_mask, pix_idx, out_y, out_x, f_xy = compute_spatial_mapping(
            spatial_wcs,
            img_data.shape,
            output_wcs,
            output_shape,
            config.xy_shrink,
        )

        if len(out_y) == 0:
            logger.debug(f"Skipping {fpath.name}: no spatial overlap")
            n_rejected += 1
            continue

        # Get per-pixel wavelength from spectral WCS
        lambda_c_map, delta_lambda_map = _extract_wavelength_maps(spectral_wcs, img_data.shape)

        # Build pixel exclusion mask from flags
        exclude_mask = None
        if exclude_bits != 0:
            exclude_mask = (flag_data & exclude_bits) != 0

        # Accumulate
        drizzle_image(
            cube=cube,
            image=img_data,
            variance=var_data,
            flags=flag_data,
            lambda_c_map=lambda_c_map,
            delta_lambda_map=delta_lambda_map,
            pixel_idx=pix_idx,
            out_y=out_y,
            out_x=out_x,
            f_xy=f_xy,
            exclude_mask=exclude_mask,
        )

    cube.n_rejected = n_rejected
    cube.finalize_masks()

    # Save
    output_path = Path(config.output_dir) / f"drizzle_D{detector}.fits"
    save_cube(cube, output_path, overwrite=config.overwrite)
    return output_path


def drizzle(config: Drizzle3DConfig) -> Dict[int, Path]:
    """Top-level entry point: query → download → drizzle → save.

    Parameters
    ----------
    config : Drizzle3DConfig
        Complete drizzle configuration.

    Returns
    -------
    dict
        {detector_id: output_path} for each successfully processed detector.

    Examples
    --------
    >>> from spxquery.drizzle3d import Drizzle3DConfig, drizzle
    >>> config = Drizzle3DConfig(
    ...     center_ra=186.4536,
    ...     center_dec=33.5468,
    ...     width=30.0,
    ...     height=30.0,
    ...     detector=3,
    ... )
    >>> results = drizzle(config)
    """
    logger.info(f"Starting Drizzle3D pipeline: center=({config.center_ra}, {config.center_dec})")
    logger.info(
        f"  Region: {config.width}'×{config.height}', "
        f"detector={'all' if config.detector == 0 else f'D{config.detector}'}"
    )

    # 1. Build output spatial WCS
    output_wcs = build_output_wcs(config)

    # 2. Query IRSA
    obs_by_det = query_observations(config)

    if not obs_by_det:
        logger.warning("No observations found for the target region")
        return {}

    results: Dict[int, Path] = {}

    for det, observations in sorted(obs_by_det.items()):
        logger.info(f"Processing D{det}: {len(observations)} observations")

        # 3. Download / resolve from mirror
        fits_paths = download_observations(
            observations,
            output_dir=config.output_dir,
            max_workers=config.download_workers,
            skip_existing=config.skip_existing,
            data_mirror=config.data_mirror,
        )

        if not fits_paths:
            logger.warning(f"D{det}: no files downloaded, skipping")
            continue

        # 4. Drizzle
        output_path = drizzle_detector(fits_paths, config, det, output_wcs)
        if output_path is not None:
            results[det] = output_path

    logger.info(f"Drizzle3D complete: {len(results)} detector cubes produced")
    for det, path in sorted(results.items()):
        logger.info(f"  D{det}: {path}")

    return results


def _read_input_fits(filepath: Path, subtract_zodi: bool, static_zodi: bool = False):
    """Read a SPHEREx input FITS file using the shared MEF reader.

    Returns
    -------
    tuple
        (image, variance, flags, zodi, spatial_wcs, spectral_wcs)
    """
    from ..utils.spherex_mef import read_spherex_mef, subtract_zodiacal_background

    mef = read_spherex_mef(filepath)
    image = mef.image
    variance = mef.variance
    flags = mef.flags.astype(np.uint32)
    zodi = mef.zodi
    spatial_wcs = mef.spatial_wcs
    spectral_wcs = mef.spectral_wcs

    if subtract_zodi:
        image, _ = subtract_zodiacal_background(image, zodi, flags, variance, static_zodi=static_zodi)

    return image, variance, flags, zodi, spatial_wcs, spectral_wcs


def _extract_wavelength_maps(spectral_wcs: WCS, shape) -> tuple:
    """Extract per-pixel (λ_c, Δλ) maps from the spectral WCS.

    Uses low-level wcs_pix2world to avoid Quantity wrapping overhead
    for millions of pixels.

    Parameters
    ----------
    spectral_wcs : WCS
        Spectral WCS (alternative 'W' key) from the input FITS.
    shape : tuple
        (ny, nx) image shape.

    Returns
    -------
    lambda_c_map : np.ndarray
        (ny, nx) central wavelength [μm].
    delta_lambda_map : np.ndarray
        (ny, nx) bandwidth [μm].
    """
    import astropy.units as u

    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    xx_flat = xx.ravel().astype(np.float64)
    yy_flat = yy.ravel().astype(np.float64)

    try:
        world = spectral_wcs.wcs_pix2world(xx_flat, yy_flat, 0)
        # One-time scalar conversion from native WCS units to micrometers
        cunit = spectral_wcs.wcs.cunit[0]
        um_factor = cunit.to(u.micron) if cunit is not None else 1.0
        lambda_c_flat = world[0] * um_factor
        delta_lambda_flat = world[1] * um_factor
    except Exception as e:
        logger.warning(f"Failed to extract wavelength maps: {e}")
        lambda_c_flat = np.full(ny * nx, np.nan)
        delta_lambda_flat = np.full(ny * nx, np.nan)

    lambda_c_map = lambda_c_flat.reshape(ny, nx).astype(np.float64)
    delta_lambda_map = delta_lambda_flat.reshape(ny, nx).astype(np.float64)

    return lambda_c_map, delta_lambda_map
