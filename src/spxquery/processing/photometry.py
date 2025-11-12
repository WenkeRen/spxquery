"""
Aperture photometry extraction for SPHEREx data.
"""

import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from photutils.aperture import CircularAperture, aperture_photometry
from tqdm import tqdm

from ..core.config import PhotometryResult, Source
from ..utils.spherex_mef import (
    get_pixel_coordinates,
    get_pixel_scale_at_position,
    get_wavelength_at_position,
    read_spherex_mef,
    subtract_zodiacal_background,
)
from ..core.config import PhotometryConfig
from .magnitudes import calculate_ab_magnitude_from_jy
from .background import estimate_local_background

logger = logging.getLogger(__name__)


def repair_variance_for_flagged_pixels(
    variance: np.ndarray, flags: np.ndarray
) -> np.ndarray:
    """
    Repair NaN variance values for pixels with non-zero flags.

    This function validates that NaN variance correlates with flagged pixels
    (expected behavior) and provides a reasonable variance estimate using
    the median of valid variance values across the full image.

    Parameters
    ----------
    variance : np.ndarray
        Variance array that may contain NaN values
    flags : np.ndarray
        Flag array (bitmap) indicating pixel quality issues

    Returns
    -------
    np.ndarray
        Repaired variance array with NaN replaced for flagged pixels only

    Raises
    ------
    ValueError
        If NaN variance is found in pixels with zero flags (unexpected condition)

    Notes
    -----
    - Only repairs NaN variance if the pixel has non-zero flags
    - Uses median of valid variance values as replacement
    - Keeps NaN for unflagged pixels to trigger errors (data quality issue)
    """
    # Create a copy to avoid modifying the original
    repaired_variance = variance.copy()

    # Find pixels with NaN variance
    nan_mask = np.isnan(variance)

    if not nan_mask.any():
        # No NaN values, return as-is
        return repaired_variance

    # Check if NaN pixels have non-zero flags
    nan_with_zero_flags = nan_mask & (flags == 0)

    if nan_with_zero_flags.any():
        # Unexpected: NaN variance without flags
        n_bad = nan_with_zero_flags.sum()
        logger.error(
            f"Found {n_bad} pixels with NaN variance but zero flags. "
            f"This indicates unexpected data quality issues."
        )
        raise ValueError(
            f"Found {n_bad} pixels with NaN variance but zero flags. "
            f"Expected flagged pixels to have NaN variance, not unflagged pixels."
        )

    # All NaN pixels have non-zero flags (expected condition)
    # Calculate median of valid variance values
    valid_variance_mask = ~nan_mask & (variance > 0)

    if not valid_variance_mask.any():
        # No valid variance values to estimate from
        logger.error("No valid variance values found in image for median estimation")
        raise ValueError("Cannot repair variance: no valid variance values in image")

    median_variance = np.median(variance[valid_variance_mask])

    # Replace NaN values with median variance
    n_repaired = nan_mask.sum()
    repaired_variance[nan_mask] = median_variance

    logger.info(
        f"Repaired {n_repaired} pixels with NaN variance using median variance "
        f"({median_variance:.6e}). All repaired pixels have non-zero flags."
    )

    return repaired_variance


def extract_aperture_photometry(
    image: np.ndarray, error: np.ndarray, x: float, y: float, radius: float
) -> Tuple[float, float]:
    """
    Perform circular aperture photometry.

    Parameters
    ----------
    image : np.ndarray
        Image data (surface brightness units)
    error : np.ndarray
        Error array (same units as image)
    x, y : float
        Pixel coordinates (0-based)
    radius : float
        Aperture radius in pixels

    Returns
    -------
    flux : float
        Integrated flux (same units as image, summed over aperture)
    flux_error : float
        Flux uncertainty (same units as image)
    """
    # Create aperture
    aperture = CircularAperture((x, y), r=radius)

    # Perform photometry
    phot_table = aperture_photometry(image, aperture, error=error)

    flux = float(phot_table["aperture_sum"][0])
    flux_error = float(phot_table["aperture_sum_err"][0])

    return flux, flux_error


def process_flags_in_aperture(flags: np.ndarray, x: float, y: float, radius: float) -> int:
    """
    Combine flags within aperture using bitwise OR.

    Parameters
    ----------
    flags : np.ndarray
        Flag array (bitmap)
    x, y : float
        Pixel coordinates (0-based)
    radius : float
        Aperture radius in pixels

    Returns
    -------
    int
        Combined flag bitmap
    """
    # Create coordinate grids
    yy, xx = np.ogrid[: flags.shape[0], : flags.shape[1]]

    # Create circular mask
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2

    # Get flags within aperture
    aperture_flags = flags[mask]

    # Combine with bitwise OR
    if len(aperture_flags) > 0:
        combined_flag = np.bitwise_or.reduce(aperture_flags)
    else:
        combined_flag = 0

    return int(combined_flag)


def extract_aperture_photometry_with_background(
    image: np.ndarray,
    variance: np.ndarray,
    flags: np.ndarray,
    x: float,
    y: float,
    aperture_radius: float,
    background_method: str = "annulus",
    window_size: Union[int, Tuple[int, int]] = 50,
    min_usable_pixels: int = 10,
    max_outer_radius: float = 5.0,
    bg_sigma_clip_sigma: float = 3.0,
    bg_sigma_clip_maxiters: int = 3,
    max_annulus_attempts: int = 5,
    annulus_expansion_step: float = 0.5,
    annulus_inner_offset: float = 1.414,
) -> Tuple[float, float, float, float, int]:
    """
    Perform aperture photometry with local background subtraction.

    Parameters
    ----------
    image : np.ndarray
        Image data (surface brightness units)
    variance : np.ndarray
        Variance array (surface brightness units squared)
    flags : np.ndarray
        Flag array
    x, y : float
        Source coordinates
    aperture_radius : float
        Aperture radius in pixels. Used for photometry and for excluding
        aperture pixels from window background estimation.
    background_method : str
        Background estimation method ('annulus' or 'window')
    window_size : int or tuple of (height, width)
        Background window size in pixels (for window method).
        Aperture pixels are automatically excluded.
    min_usable_pixels : int
        Minimum number of usable background pixels
    max_outer_radius : float
        Maximum outer radius for background annulus (for annulus method)
    bg_sigma_clip_sigma : float
        Sigma threshold for sigma clipping
    bg_sigma_clip_maxiters : int
        Maximum iterations for sigma clipping
    max_annulus_attempts : int
        Maximum attempts to expand annulus (for annulus method)
    annulus_expansion_step : float
        Step size for annulus expansion (for annulus method)
    annulus_inner_offset : float
        Offset from aperture edge to inner annulus (for annulus method)

    Returns
    -------
    flux : float
        Background-subtracted flux (same units as image, summed over aperture)
    flux_error : float
        Flux error (same units as image)
    background : float
        Background level per pixel (same units as image)
    background_error : float
        Background error per pixel (same units as image)
    n_bg_pixels : int
        Number of background pixels used

    Notes
    -----
    Inner and outer annulus radii are calculated automatically based on aperture_radius
    and annulus_inner_offset. Not exposed as parameters to keep interface simple.
    """
    # Estimate local background using selected method
    if background_method == "annulus":
        bg_level, bg_error, n_bg_pixels = estimate_local_background(
            image,
            variance,
            flags,
            x,
            y,
            aperture_radius,
            min_usable_pixels,
            max_outer_radius,
            bg_sigma_clip_sigma,
            bg_sigma_clip_maxiters,
            max_annulus_attempts,
            annulus_expansion_step,
            annulus_inner_offset,
        )
    elif background_method == "window":
        from .background import estimate_window_background

        bg_level, bg_error, n_bg_pixels = estimate_window_background(
            image,
            variance,
            flags,
            x,
            y,
            window_size,
            aperture_radius,
            min_usable_pixels,
            bg_sigma_clip_sigma,
            bg_sigma_clip_maxiters,
        )
    else:
        raise ValueError(f"Invalid background_method: {background_method}")

    if n_bg_pixels == 0:
        # Return zero flux with high error if no background estimate
        return 0.0, 1e10, 0.0, 1e10, 0

    # Create aperture
    aperture = CircularAperture((x, y), r=aperture_radius)

    # Calculate aperture area
    aperture_area = np.pi * aperture_radius**2

    # Perform aperture photometry on original image
    error_array = np.sqrt(variance)
    phot_table = aperture_photometry(image, aperture, error=error_array)

    raw_flux = float(phot_table["aperture_sum"][0])
    raw_flux_error = float(phot_table["aperture_sum_err"][0])

    # Subtract background
    background_total = bg_level * aperture_area
    background_error_total = bg_error * aperture_area

    flux = raw_flux - background_total
    flux_error = np.sqrt(raw_flux_error**2 + background_error_total**2)

    logger.debug(
        f"Raw flux: {raw_flux:.6f} ± {raw_flux_error:.6f}, "
        f"Background: {background_total:.6f} ± {background_error_total:.6f}, "
        f"Final: {flux:.6f} ± {flux_error:.6f}"
    )

    return flux, flux_error, bg_level, bg_error, n_bg_pixels


def extract_source_photometry(
    mef_file: Path,
    source: Source,
    photometry_config: Optional[PhotometryConfig] = None,
) -> Optional[PhotometryResult]:
    """
    Extract photometry for a source from a SPHEREx MEF file with local background subtraction.

    This function reads SPHEREx data in uJy/arcsec2 units, performs aperture photometry,
    and converts to integrated flux density (microJansky) using WCS-derived pixel scales.

    All photometry parameters (aperture sizing, background method, zodiacal subtraction, etc.)
    are controlled via PhotometryConfig.

    Parameters
    ----------
    mef_file : Path
        Path to SPHEREx MEF file
    source : Source
        Source with RA/Dec coordinates
    photometry_config : PhotometryConfig, optional
        Photometry configuration. If None, uses defaults.

    Returns
    -------
    PhotometryResult or None
        Photometry result with flux in microJansky (μJy) and AB magnitude, or None if extraction failed

    Notes
    -----
    - Uses read_spherex_mef() with target_unit='uJy/arcsec2' for automatic unit conversion
    - All parameters come from photometry_config (aperture_method, subtract_zodi, background_method, etc.)
    """

    # Use default config if none provided
    if photometry_config is None:
        photometry_config = PhotometryConfig()

    try:
        # Read MEF with unit conversion to uJy/arcsec2 for simpler flux calculations
        mef = read_spherex_mef(mef_file, target_unit='uJy/arcsec2')

        # Repair variance for flagged pixels with NaN values
        # This validates that NaN variance correlates with flags and provides median estimate
        mef.variance = repair_variance_for_flagged_pixels(mef.variance, mef.flags)

        # Get pixel coordinates
        x, y = get_pixel_coordinates(mef, source.ra, source.dec)

        # Determine aperture radius based on aperture_method
        # Priority: aperture_method setting > explicit aperture_radius parameter

        if photometry_config.aperture_method == "fwhm":
            # FWHM-based aperture: calculate from PSF
            try:
                fwhm_arcsec = mef.get_psf_fwhm_estimate(x, y)

                # Convert FWHM from arcsec to pixels
                pixel_scale_arcsec = mef.get_pixel_scale(x, y, fallback=photometry_config.pixel_scale_fallback)
                fwhm_pixels = fwhm_arcsec / pixel_scale_arcsec

                # Aperture diameter = FWHM × multiplier, then convert to radius
                aperture_diameter = fwhm_pixels * photometry_config.fwhm_multiplier
                final_aperture_radius = aperture_diameter / 2.0

                logger.info(
                    f"FWHM-based aperture: FWHM={fwhm_arcsec:.3f}\" ({fwhm_pixels:.2f}px) "
                    f"→ diameter={aperture_diameter:.2f}px (radius={final_aperture_radius:.2f}px)"
                )

            except Exception as e:
                # FWHM estimation failed, use fallback
                final_aperture_radius = photometry_config.aperture_diameter / 2.0
                logger.warning(
                    f"FWHM estimation failed at ({x:.1f}, {y:.1f}): {e}. "
                    f"Using config aperture_diameter={photometry_config.aperture_diameter}px as fallback"
                )

        elif photometry_config.aperture_method == "fixed":
            # Fixed aperture: use explicit parameter or config value
            final_aperture_radius = photometry_config.aperture_diameter / 2.0
            logger.debug(
                f"Fixed aperture: using config diameter={photometry_config.aperture_diameter}px "
                f"(radius={final_aperture_radius:.2f}px)"
            )

        else:
            raise ValueError(f"Invalid aperture_method: {photometry_config.aperture_method}")

        # Check if coordinates are within image with extra margin for background annulus
        ny, nx = mef.image.shape
        max_outer_radius = photometry_config.max_outer_radius  # Maximum annulus outer radius
        required_margin = max(final_aperture_radius, max_outer_radius)

        if not (required_margin <= x < nx - required_margin and required_margin <= y < ny - required_margin):
            logger.warning(f"Source at ({x:.1f}, {y:.1f}) too close to edge for background annulus in {mef_file.name}")
            return None

        # Get wavelength info
        wavelength, bandwidth = get_wavelength_at_position(mef, x, y)

        # Prepare image (optionally subtract zodiacal light)
        if photometry_config.subtract_zodi:
            image, zodi_scale = subtract_zodiacal_background(
                mef.image,
                mef.zodi,
                mef.flags,
                mef.variance,
                photometry_config.zodi_scale_min,
                photometry_config.zodi_scale_max,
            )
            logger.debug(f"Applied zodiacal scaling factor: {zodi_scale:.4f}")
        else:
            image = mef.image

        # Extract photometry with local background subtraction
        # Image is in uJy/arcsec2, so flux_sum will be in uJy/arcsec2 (summed over pixels)
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
            photometry_config.background_method,
            photometry_config.window_size,
            photometry_config.min_usable_pixels,
            photometry_config.max_outer_radius,
            photometry_config.bg_sigma_clip_sigma,
            photometry_config.bg_sigma_clip_maxiters,
            photometry_config.max_annulus_attempts,
            photometry_config.annulus_expansion_step,
            photometry_config.annulus_inner_offset,
        )

        # Check if background estimation failed
        if n_bg_pixels == 0:
            logger.error(f"Background estimation failed for {mef_file.name} - dropping observation")
            return None

        logger.debug(f"Local background: {bg_level:.6f} ± {bg_error:.6f} uJy/arcsec2 from {n_bg_pixels} pixels")

        # Convert from uJy/arcsec2 to microJansky (μJy) for integrated flux
        # The aperture photometry returns a sum: Σ(surface_brightness_i) across pixels
        # To convert to flux: multiply by pixel area in arcsec2
        pixel_scale_arcsec = get_pixel_scale_at_position(mef.spatial_wcs, x, y, photometry_config.pixel_scale_fallback)
        pixel_area_arcsec2 = pixel_scale_arcsec ** 2  # Area of one pixel in arcsec2

        # Convert: (uJy/arcsec2 × pixels) × (arcsec2/pixel) = uJy
        flux_ujy = flux_sum_uJy_per_arcsec2 * pixel_area_arcsec2
        flux_error_ujy = flux_error_sum_uJy_per_arcsec2 * pixel_area_arcsec2

        # For magnitude calculation, convert to Jansky
        flux_jy = flux_ujy / 1e6
        flux_error_jy = flux_error_ujy / 1e6

        logger.debug(
            f"Unit conversion: {flux_sum_uJy_per_arcsec2:.6f} uJy/arcsec2·pix → {flux_ujy:.3f} μJy "
            f"(pixel area = {pixel_area_arcsec2:.4f} arcsec2/pix)"
        )

        # Process flags
        combined_flag = process_flags_in_aperture(mef.flags, x, y, final_aperture_radius)

        # Extract obs_id from MEF
        obs_id = mef.header.get("OBSID", mef_file.stem)

        # Extract band from DETECTOR header (1-6 maps to D1-D6)
        detector_num = mef.detector
        if 1 <= detector_num <= 6:
            band = f"D{detector_num}"
        else:
            band = "Unknown"
            logger.warning(f"Invalid detector number {detector_num} in {mef_file.name}, expected 1-6")

        # Calculate AB magnitude using flux in Jansky
        mag_ab, mag_ab_error = calculate_ab_magnitude_from_jy(flux_jy, flux_error_jy, wavelength)

        result = PhotometryResult(
            obs_id=obs_id,
            mjd=mef.mjd,
            flux=flux_ujy,
            flux_error=flux_error_ujy,
            wavelength=wavelength,
            bandwidth=bandwidth,
            flag=combined_flag,
            pix_x=x,
            pix_y=y,
            band=band,
            mag_ab=mag_ab,
            mag_ab_error=mag_ab_error,
        )

        logger.info(
            f"Extracted photometry from {mef_file.name}: "
            f"flux={flux_ujy:.3f}±{flux_error_ujy:.3f} μJy "
            f"({flux_jy:.6f}±{flux_error_jy:.6f} Jy) at λ={wavelength:.3f} μm, "
            f"mag_AB={mag_ab:.3f}±{mag_ab_error:.3f}"
        )

        return result

    except Exception as e:
        logger.error(f"Failed to extract photometry from {mef_file}: {e}")
        return None


def _process_single_file(args):
    """
    Process a single file - helper function for multiprocessing.

    Parameters
    ----------
    args : tuple
        (filepath, source, photometry_config)

    Returns
    -------
    PhotometryResult or None
    """
    filepath, source, photometry_config = args
    return extract_source_photometry(filepath, source, photometry_config)


def process_all_observations(
    file_paths: list[Path],
    source: Source,
    photometry_config: Optional[PhotometryConfig] = None,
) -> list[PhotometryResult]:
    """
    Process photometry for all observation files with local background subtraction.

    Reads SPHEREx data in uJy/arcsec2 units and converts to integrated flux using
    WCS-derived pixel scales. Supports both sequential and parallel processing.

    All photometry parameters (aperture sizing, background method, zodiacal subtraction,
    parallel processing workers, etc.) are controlled via PhotometryConfig.

    Parameters
    ----------
    file_paths : list[Path]
        List of MEF file paths
    source : Source
        Target source
    photometry_config : PhotometryConfig, optional
        Photometry configuration. If None, uses defaults.

    Returns
    -------
    list[PhotometryResult]
        List of photometry results with flux in microJansky (μJy)
    """

    # Use default config if none provided
    if photometry_config is None:
        photometry_config = PhotometryConfig()

    logger.info(f"Processing photometry for {len(file_paths)} observations")

    # Prepare arguments for processing
    args_list = [
        (filepath, source, photometry_config)
        for filepath in file_paths
    ]

    results = []

    # Determine processing mode based on config
    max_workers = photometry_config.max_processing_workers
    use_multiprocessing = max_workers > 1 and len(file_paths) > 1

    if use_multiprocessing:
        logger.info(f"Using multiprocessing with {max_workers} workers")

        try:
            # Use multiprocessing with progress bar
            with Pool(processes=max_workers) as pool:
                # Use imap for better progress tracking
                with tqdm(total=len(args_list), desc="Processing observations", unit="files") as pbar:
                    for result in pool.imap(_process_single_file, args_list):
                        if result:
                            results.append(result)
                        pbar.update(1)
        except RuntimeError as e:
            if "freeze_support" in str(e) or "bootstrapping" in str(e):
                logger.error(
                    "Multiprocessing failed. On macOS/Windows, you must protect your script with:\n"
                    "    if __name__ == '__main__':\n"
                    "        run_pipeline(...)\n"
                    "Falling back to sequential processing."
                )
                # Fall back to sequential processing
                logger.info("Falling back to sequential processing")
                for args in tqdm(args_list, desc="Processing observations", unit="files"):
                    result = _process_single_file(args)
                    if result:
                        results.append(result)
            else:
                raise
    else:
        if max_workers == 1:
            logger.info("Using sequential processing (max_workers=1)")
        else:
            logger.info("Using sequential processing (single file or invalid max_workers)")

        # Sequential processing with progress bar
        for args in tqdm(args_list, desc="Processing observations", unit="files"):
            result = _process_single_file(args)
            if result:
                results.append(result)

    logger.info(f"Successfully processed {len(results)} observations")

    return results
