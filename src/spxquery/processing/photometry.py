"""
Aperture photometry extraction for SPHEREx data.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
from multiprocessing import Pool
from functools import partial

import numpy as np
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.utils import calc_total_error
from tqdm import tqdm

from ..core.config import Source, PhotometryResult
from .fits_handler import (
    read_spherex_mef, get_pixel_coordinates,
    get_wavelength_at_position, subtract_zodiacal_background,
    create_background_mask, get_pixel_scale_at_position
)
from .magnitudes import calculate_ab_magnitude_from_jy

logger = logging.getLogger(__name__)


def extract_aperture_photometry(
    image: np.ndarray,
    error: np.ndarray,
    x: float,
    y: float,
    radius: float
) -> Tuple[float, float]:
    """
    Perform circular aperture photometry.
    
    Parameters
    ----------
    image : np.ndarray
        Image data in MJy/sr
    error : np.ndarray
        Error array in MJy/sr
    x, y : float
        Pixel coordinates (0-based)
    radius : float
        Aperture radius in pixels
    
    Returns
    -------
    flux : float
        Integrated flux in MJy/sr
    flux_error : float
        Flux uncertainty in MJy/sr
    """
    # Create aperture
    aperture = CircularAperture((x, y), r=radius)
    
    # Perform photometry
    phot_table = aperture_photometry(image, aperture, error=error)
    
    flux = float(phot_table['aperture_sum'][0])
    flux_error = float(phot_table['aperture_sum_err'][0])
    
    return flux, flux_error


def process_flags_in_aperture(
    flags: np.ndarray,
    x: float,
    y: float,
    radius: float
) -> int:
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
    yy, xx = np.ogrid[:flags.shape[0], :flags.shape[1]]
    
    # Create circular mask
    mask = (xx - x)**2 + (yy - y)**2 <= radius**2
    
    # Get flags within aperture
    aperture_flags = flags[mask]
    
    # Combine with bitwise OR
    if len(aperture_flags) > 0:
        combined_flag = np.bitwise_or.reduce(aperture_flags)
    else:
        combined_flag = 0
    
    return int(combined_flag)


def determine_annulus_radii(
    aperture_radius: float,
    inner_radius: Optional[float] = None,
    outer_radius: Optional[float] = None,
    min_annulus_area_pixels: int = 10,
    max_outer_radius: float = 5.0
) -> Tuple[float, float]:
    """
    Determine inner and outer radii for background annulus.
    
    Parameters
    ----------
    aperture_radius : float
        Source aperture radius in pixels
    inner_radius : float, optional
        Fixed inner radius. If None, calculated automatically
    outer_radius : float, optional  
        Fixed outer radius. If None, calculated automatically
    min_annulus_area_pixels : int
        Minimum annulus area in pixels
    max_outer_radius : float
        Maximum allowed outer radius
    
    Returns
    -------
    inner_radius, outer_radius : float, float
        Annulus inner and outer radii in pixels
    """
    # Calculate inner radius (1.414 pixels larger than aperture radius)
    if inner_radius is None:
        inner_radius = aperture_radius + 1.414
    
    # Calculate outer radius to achieve minimum annulus area
    if outer_radius is None:
        # Area of annulus = π(r_out² - r_in²)
        # Solve for r_out: r_out = sqrt(area/π + r_in²)
        target_outer_radius = np.sqrt(min_annulus_area_pixels / np.pi + inner_radius**2)
        outer_radius = min(target_outer_radius, max_outer_radius)
    
    logger.debug(f"Annulus radii: inner={inner_radius:.2f}, outer={outer_radius:.2f}")
    
    return inner_radius, outer_radius


def create_annulus_mask(
    image_shape: Tuple[int, int],
    x: float,
    y: float,
    inner_radius: float,
    outer_radius: float
) -> np.ndarray:
    """
    Create boolean mask for annular region.
    
    Parameters
    ----------
    image_shape : tuple
        Shape of image (ny, nx)
    x, y : float
        Center coordinates
    inner_radius, outer_radius : float
        Annulus radii in pixels
    
    Returns
    -------
    np.ndarray
        Boolean mask (True = within annulus)
    """
    ny, nx = image_shape
    yy, xx = np.ogrid[:ny, :nx]
    
    # Distance from center
    distances = np.sqrt((xx - x)**2 + (yy - y)**2)
    
    # Annulus mask (between inner and outer radii)
    mask = (distances >= inner_radius) & (distances <= outer_radius)
    
    return mask


def estimate_local_background(
    image: np.ndarray,
    variance: np.ndarray,
    flags: np.ndarray,
    x: float,
    y: float,
    aperture_radius: float,
    inner_radius: Optional[float] = None,
    outer_radius: Optional[float] = None,
    min_usable_pixels: int = 10,
    max_outer_radius: float = 5.0
) -> Tuple[float, float, int]:
    """
    Estimate local background using annular region around source.
    
    Parameters
    ----------
    image : np.ndarray
        Image data  
    variance : np.ndarray
        Variance array
    flags : np.ndarray
        Flag array
    x, y : float
        Source coordinates
    aperture_radius : float
        Source aperture radius
    inner_radius, outer_radius : float, optional
        Annulus radii (calculated if None)
    min_usable_pixels : int
        Minimum number of usable pixels required
    max_outer_radius : float
        Maximum outer radius
    
    Returns
    -------
    background_level : float
        Background level per pixel (MJy/sr)
    background_error : float  
        Background error per pixel (MJy/sr)
    n_usable : int
        Number of usable pixels in annulus
    """
    # Determine annulus radii
    inner_r, outer_r = determine_annulus_radii(
        aperture_radius, inner_radius, outer_radius, min_usable_pixels, max_outer_radius
    )
    
    # Check if annulus fits within image
    ny, nx = image.shape
    max_distance = min(x, y, nx - x, ny - y)
    
    if outer_r > max_distance:
        logger.warning(f"Outer radius {outer_r:.2f} exceeds image boundary {max_distance:.2f}")
        outer_r = max_distance
        
        # If outer radius was reduced, inner radius might need adjustment too
        if inner_r >= outer_r:
            inner_r = outer_r * 0.7  # Make inner radius 70% of outer radius
    
    # Try progressively larger outer radii until we get enough usable pixels
    attempt = 0
    max_attempts = 5
    
    while attempt < max_attempts:
        # Create annulus mask
        annulus_mask = create_annulus_mask(image.shape, x, y, inner_r, outer_r)
        
        # Create clean background mask (no flagged pixels)
        clean_mask = create_background_mask(flags)
        
        # Combine masks
        usable_mask = annulus_mask & clean_mask
        n_usable = np.sum(usable_mask)
        
        logger.debug(f"Attempt {attempt + 1}: annulus area={np.sum(annulus_mask)}, usable={n_usable}")
        
        # Check if we have enough usable pixels
        if n_usable >= min_usable_pixels:
            break
        
        # Expand outer radius if possible
        if outer_r < max_outer_radius and outer_r < max_distance:
            new_outer_r = min(outer_r + 0.5, max_outer_radius, max_distance)
            if new_outer_r > outer_r:
                outer_r = new_outer_r
                attempt += 1
                continue
        
        # Cannot expand further
        break
    
    # Extract background pixels
    if n_usable == 0:
        logger.error("No usable pixels in background annulus")
        return 0.0, 0.0, 0
    
    bg_pixels = image[usable_mask]
    bg_variance = variance[usable_mask]
    
    # Calculate background statistics using sigma-clipped mean
    from astropy.stats import sigma_clipped_stats
    bg_mean, bg_median, bg_std = sigma_clipped_stats(bg_pixels, sigma=3.0, maxiters=3)
    
    # Error on the mean background
    # Use variance if available, otherwise use std from sigma clipping
    if np.all(bg_variance > 0):
        bg_error = np.sqrt(np.mean(bg_variance[bg_variance > 0]))
    else:
        bg_error = bg_std / np.sqrt(n_usable)
    
    logger.debug(
        f"Background estimate: {bg_mean:.6f} ± {bg_error:.6f} MJy/sr "
        f"from {n_usable} pixels (r={inner_r:.1f}-{outer_r:.1f})"
    )
    
    return float(bg_mean), float(bg_error), n_usable


def extract_aperture_photometry_with_background(
    image: np.ndarray,
    variance: np.ndarray,
    flags: np.ndarray,
    x: float,
    y: float,
    aperture_radius: float,
    inner_radius: Optional[float] = None,
    outer_radius: Optional[float] = None
) -> Tuple[float, float, float, float, int]:
    """
    Perform aperture photometry with local background subtraction.
    
    Parameters
    ----------
    image : np.ndarray
        Image data in MJy/sr
    variance : np.ndarray  
        Variance array in (MJy/sr)²
    flags : np.ndarray
        Flag array
    x, y : float
        Source coordinates
    aperture_radius : float
        Aperture radius in pixels
    inner_radius, outer_radius : float, optional
        Background annulus radii
    
    Returns
    -------
    flux : float
        Background-subtracted flux (MJy/sr)
    flux_error : float
        Flux error (MJy/sr)
    background : float
        Background level per pixel (MJy/sr)
    background_error : float
        Background error per pixel (MJy/sr)
    n_bg_pixels : int
        Number of background pixels used
    """
    # Estimate local background
    bg_level, bg_error, n_bg_pixels = estimate_local_background(
        image, variance, flags, x, y, aperture_radius, inner_radius, outer_radius
    )
    
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
    
    raw_flux = float(phot_table['aperture_sum'][0])
    raw_flux_error = float(phot_table['aperture_sum_err'][0])
    
    # Subtract background
    background_total = bg_level * aperture_area
    background_error_total = bg_error * np.sqrt(aperture_area)
    
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
    aperture_radius: float = 1.5,  # 3 pixel diameter = 1.5 pixel radius
    subtract_zodi: bool = True,
    inner_radius: Optional[float] = None,
    outer_radius: Optional[float] = None
) -> Optional[PhotometryResult]:
    """
    Extract photometry for a source from a SPHEREx MEF file with local background subtraction.
    
    This function performs aperture photometry and properly converts surface brightness
    (MJy/sr) to flux density (Jansky) for magnitude calculation using the WCS-derived
    pixel scale at the source position.
    
    Parameters
    ----------
    mef_file : Path
        Path to SPHEREx MEF file
    source : Source
        Source with RA/Dec coordinates
    aperture_radius : float
        Aperture radius in pixels (default 1.5 for 3 pixel diameter)
    subtract_zodi : bool
        Whether to subtract zodiacal background
    inner_radius : float, optional
        Inner radius for background annulus. If None, calculated automatically.
    outer_radius : float, optional
        Outer radius for background annulus. If None, calculated automatically.
    
    Returns
    -------
    PhotometryResult or None
        Photometry result with flux in MJy/sr and AB magnitude, or None if extraction failed
    """
    try:
        # Read MEF
        mef = read_spherex_mef(mef_file)
        
        # Get pixel coordinates
        x, y = get_pixel_coordinates(mef, source.ra, source.dec)
        
        # Check if coordinates are within image with extra margin for background annulus
        ny, nx = mef.image.shape
        max_outer_radius = 5.0  # Maximum annulus outer radius
        required_margin = max(aperture_radius, max_outer_radius)
        
        if not (required_margin <= x < nx - required_margin and 
                required_margin <= y < ny - required_margin):
            logger.warning(
                f"Source at ({x:.1f}, {y:.1f}) too close to edge for background annulus in {mef_file.name}"
            )
            return None
        
        # Get wavelength info
        wavelength, bandwidth = get_wavelength_at_position(mef, x, y)
        
        # Prepare image (optionally subtract zodiacal light)
        if subtract_zodi:
            image, zodi_scale = subtract_zodiacal_background(
                mef.image, mef.zodi, mef.flags, mef.variance
            )
            logger.debug(f"Applied zodiacal scaling factor: {zodi_scale:.4f}")
        else:
            image = mef.image
        
        # Extract photometry with local background subtraction
        flux_mjy_sr, flux_error_mjy_sr, bg_level, bg_error, n_bg_pixels = extract_aperture_photometry_with_background(
            image, mef.variance, mef.flags, x, y, aperture_radius, inner_radius, outer_radius
        )
        
        # Check if background estimation failed
        if n_bg_pixels == 0:
            logger.error(f"Background estimation failed for {mef_file.name} - dropping observation")
            return None
        
        logger.debug(f"Local background: {bg_level:.6f} ± {bg_error:.6f} MJy/sr from {n_bg_pixels} pixels")

        # Convert from MJy/sr to Jansky for magnitude calculation
        # The aperture photometry returns a sum: Σ(surface_brightness_i) across pixels
        # To convert to flux: multiply by solid angle PER PIXEL, not total aperture
        pixel_scale_arcsec = get_pixel_scale_at_position(mef.spatial_wcs, x, y)
        pixel_solid_angle_sr = (pixel_scale_arcsec / 206265.0) ** 2  # Convert arcsec to radians, then square

        # Convert: (MJy/sr × pixels) × (sr/pixel) = MJy → Jy
        flux_mjy = flux_mjy_sr * pixel_solid_angle_sr
        flux_error_mjy = flux_error_mjy_sr * pixel_solid_angle_sr

        flux_jy = flux_mjy * 1e6  # Jansky
        flux_error_jy = flux_error_mjy * 1e6  # Jansky

        logger.debug(
            f"Unit conversion: {flux_mjy_sr:.6f} MJy/sr·pix → {flux_jy:.6f} Jy "
            f"(pixel solid angle = {pixel_solid_angle_sr:.2e} sr/pix)"
        )
        
        # Process flags
        combined_flag = process_flags_in_aperture(
            mef.flags, x, y, aperture_radius
        )
        
        # Extract obs_id from MEF
        obs_id = mef.header.get('OBSID', mef_file.stem)
        
        # Extract band from DETECTOR header (1-6 maps to D1-D6)
        detector_num = mef.detector
        if 1 <= detector_num <= 6:
            band = f'D{detector_num}'
        else:
            band = 'Unknown'
            logger.warning(f"Invalid detector number {detector_num} in {mef_file.name}, expected 1-6")
        
        # Calculate AB magnitude using flux in Jansky
        mag_ab, mag_ab_error = calculate_ab_magnitude_from_jy(
            flux_jy, flux_error_jy, wavelength
        )
        
        result = PhotometryResult(
            obs_id=obs_id,
            mjd=mef.mjd,
            flux=flux_mjy_sr,
            flux_error=flux_error_mjy_sr,
            wavelength=wavelength,
            bandwidth=bandwidth,
            flag=combined_flag,
            pix_x=x,
            pix_y=y,
            band=band,
            mag_ab=mag_ab,
            mag_ab_error=mag_ab_error
        )
        
        logger.info(
            f"Extracted photometry from {mef_file.name}: "
            f"flux={flux_mjy_sr:.3f}±{flux_error_mjy_sr:.3f} MJy/sr "
            f"({flux_jy:.3f}±{flux_error_jy:.3f} Jy) at λ={wavelength:.3f} μm, "
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
        (filepath, source, aperture_radius, subtract_zodi, inner_radius, outer_radius)
        
    Returns
    -------
    PhotometryResult or None
    """
    filepath, source, aperture_radius, subtract_zodi, inner_radius, outer_radius = args
    return extract_source_photometry(
        filepath, source, aperture_radius, subtract_zodi, inner_radius, outer_radius
    )


def process_all_observations(
    file_paths: list[Path],
    source: Source,
    aperture_radius: float = 1.5,
    subtract_zodi: bool = True,
    inner_radius: Optional[float] = None,
    outer_radius: Optional[float] = None,
    max_workers: int = 10
) -> list[PhotometryResult]:
    """
    Process photometry for all observation files with local background subtraction.
    
    Uses WCS-derived pixel scales for accurate unit conversion from MJy/sr to Jansky.
    Supports both sequential and parallel processing with progress bars.
    
    Parameters
    ----------
    file_paths : list[Path]
        List of MEF file paths
    source : Source
        Target source
    aperture_radius : float
        Aperture radius in pixels
    subtract_zodi : bool
        Whether to subtract zodiacal background
    inner_radius : float, optional
        Inner radius for background annulus
    outer_radius : float, optional
        Outer radius for background annulus
    max_workers : int
        Number of worker processes (default: 10). If 1 or invalid, runs sequentially.
    
    Returns
    -------
    list[PhotometryResult]
        List of photometry results with proper unit conversion
    """
    logger.info(f"Processing photometry for {len(file_paths)} observations")
    
    # Prepare arguments for processing
    args_list = [
        (filepath, source, aperture_radius, subtract_zodi, inner_radius, outer_radius)
        for filepath in file_paths
    ]
    
    results = []
    
    # Determine processing mode
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