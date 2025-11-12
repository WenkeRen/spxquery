"""
Background estimation for SPHEREx photometry.

This module provides both window-based and annulus-based background estimation
methods for local background subtraction in aperture photometry.
"""

import logging
from typing import Tuple, Union

import numpy as np
from astropy.stats import sigma_clipped_stats

logger = logging.getLogger(__name__)


def estimate_window_background(
    image: np.ndarray,
    variance: np.ndarray,
    flags: np.ndarray,
    x: float,
    y: float,
    window_size: Union[int, Tuple[int, int]],
    aperture_radius: float,
    min_usable_pixels: int = 10,
    bg_sigma_clip_sigma: float = 3.0,
    bg_sigma_clip_maxiters: int = 3,
) -> Tuple[float, float, int]:
    """
    Estimate local background using a rectangular window around the source.

    This method uses a rectangular region around the source with automatic
    exclusion of pixels intersecting the aperture. It is simpler than
    annulus-based methods and can be more robust in crowded fields.

    Parameters
    ----------
    image : np.ndarray
        Image data
    variance : np.ndarray
        Variance array
    flags : np.ndarray
        Flag bitmap array
    x, y : float
        Source coordinates (center of window)
    window_size : int or tuple of (height, width)
        Size of the background window in pixels.
        If int: creates square window of size window_size × window_size
        If tuple: (height, width) for rectangular window
    aperture_radius : float
        Aperture radius in pixels. All pixels intersecting this aperture
        are excluded from background estimation.
    min_usable_pixels : int
        Minimum number of unflagged pixels required
    bg_sigma_clip_sigma : float
        Sigma threshold for sigma clipping
    bg_sigma_clip_maxiters : int
        Maximum iterations for sigma clipping

    Returns
    -------
    background_level : float
        Background level per pixel
    background_error : float
        Background error per pixel
    n_usable : int
        Number of usable pixels in window

    Notes
    -----
    - If window extends beyond image boundaries, it is clipped to image edges
      with a warning (not an error)
    - Pixels with flags set are excluded from background estimation
    - Pixels intersecting the aperture are excluded (using conservative estimate)
    - Sigma clipping is used to reject outliers before computing statistics
    """
    # Parse window size
    if isinstance(window_size, int):
        win_height = win_width = window_size
    else:
        win_height, win_width = window_size

    # Validate window size
    if win_height <= 0 or win_width <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")

    # Calculate window boundaries (half-widths from center)
    half_height = win_height / 2.0
    half_width = win_width / 2.0

    # Window boundaries in pixel coordinates
    y_min_float = y - half_height
    y_max_float = y + half_height
    x_min_float = x - half_width
    x_max_float = x + half_width

    # Convert to integer indices (0-based)
    y_min = int(np.floor(y_min_float))
    y_max = int(np.ceil(y_max_float))
    x_min = int(np.floor(x_min_float))
    x_max = int(np.ceil(x_max_float))

    # Get image dimensions
    ny, nx = image.shape

    # Clip window to image boundaries
    x_min_clipped = max(0, x_min)
    x_max_clipped = min(nx, x_max)
    y_min_clipped = max(0, y_min)
    y_max_clipped = min(ny, y_max)

    # Warn if window was clipped
    if x_min_clipped != x_min or x_max_clipped != x_max or y_min_clipped != y_min or y_max_clipped != y_max:
        logger.warning(
            f"Background window [{x_min}:{x_max}, {y_min}:{y_max}] "
            f"exceeds image boundaries, clipped to [{x_min_clipped}:{x_max_clipped}, {y_min_clipped}:{y_max_clipped}]"
        )

    # Extract window region
    window_slice = (slice(y_min_clipped, y_max_clipped), slice(x_min_clipped, x_max_clipped))
    image_window = image[window_slice]
    variance_window = variance[window_slice]
    flags_window = flags[window_slice]

    # Create mask for usable pixels (unflagged)
    from ..utils.spherex_mef import create_background_mask

    clean_mask = create_background_mask(flags_window)

    # Exclude pixels intersecting with aperture
    # A pixel at (i,j) represents a square from [i-0.5, i+0.5] × [j-0.5, j+0.5]
    # To ensure we exclude ALL pixels with any intersection, use conservative estimate:
    # distance from aperture center to pixel center < aperture_radius + sqrt(2)/2
    # where sqrt(2)/2 ≈ 0.707 is half the diagonal of a unit square pixel
    yy, xx = np.ogrid[y_min_clipped:y_max_clipped, x_min_clipped:x_max_clipped]
    distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    aperture_exclusion_mask = distances > (aperture_radius + 0.707)

    # Combine masks: clean pixels AND outside aperture
    usable_mask = clean_mask & aperture_exclusion_mask

    n_usable = np.sum(usable_mask)

    # Check if we have enough pixels
    if n_usable < min_usable_pixels:
        logger.warning(
            f"Insufficient usable pixels with strict masking: {n_usable}/{min_usable_pixels} required. "
            f"Trying with relaxed masking (exclude_bad_only=False)."
        )
        # Try with relaxed masking
        clean_mask = create_background_mask(flags_window, False)
        usable_mask = clean_mask & aperture_exclusion_mask
        n_usable = np.sum(usable_mask)
        
        if n_usable < min_usable_pixels:
            logger.error(
                f"Insufficient usable pixels in background window even with relaxed masking: "
                f"{n_usable}/{min_usable_pixels} required "
                f"(window: {win_height}×{win_width}, aperture_radius: {aperture_radius:.1f})"
            )
            return 0.0, 0.0, 0

    # Extract background pixels
    bg_pixels = image_window[usable_mask]
    bg_variance = variance_window[usable_mask]

    # Calculate background statistics using sigma-clipped mean
    bg_mean, bg_median, bg_std = sigma_clipped_stats(
        bg_pixels, sigma=bg_sigma_clip_sigma, maxiters=bg_sigma_clip_maxiters
    )

    # Error on the mean background
    bg_error = bg_std / np.sqrt(n_usable)

    logger.debug(
        f"Window background estimate: {bg_median:.6f} ± {bg_error:.6f} from {n_usable} pixels "
        f"(window: {win_height}×{win_width}, aperture_radius: {aperture_radius:.1f})"
    )

    return float(bg_median), float(bg_error), n_usable


def determine_annulus_radii(
    aperture_radius: float,
    min_annulus_area_pixels: int = 10,
    max_outer_radius: float = 5.0,
    annulus_inner_offset: float = 1.414,
) -> Tuple[float, float]:
    """
    Determine inner and outer radii for background annulus.

    Inner and outer radii are calculated automatically based on aperture size
    and configuration parameters.

    Parameters
    ----------
    aperture_radius : float
        Source aperture radius in pixels
    min_annulus_area_pixels : int
        Minimum annulus area in pixels
    max_outer_radius : float
        Maximum allowed outer radius
    annulus_inner_offset : float
        Offset from aperture edge to inner annulus radius

    Returns
    -------
    inner_radius, outer_radius : float, float
        Annulus inner and outer radii in pixels

    Notes
    -----
    Inner radius is calculated as: aperture_radius + annulus_inner_offset
    Outer radius is calculated to achieve minimum annulus area, capped at max_outer_radius.
    """
    # Calculate inner radius (offset pixels larger than aperture radius)
    inner_radius = aperture_radius + annulus_inner_offset

    # Calculate outer radius to achieve minimum annulus area
    # Area of annulus = π(r_out² - r_in²)
    # Solve for r_out: r_out = sqrt(area/π + r_in²)
    target_outer_radius = np.sqrt(min_annulus_area_pixels / np.pi + inner_radius**2)
    outer_radius = min(target_outer_radius, max_outer_radius)

    logger.debug(f"Annulus radii: inner={inner_radius:.2f}, outer={outer_radius:.2f}")

    return inner_radius, outer_radius


def create_annulus_mask(
    image_shape: Tuple[int, int], x: float, y: float, inner_radius: float, outer_radius: float
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
    distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

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
    min_usable_pixels: int = 10,
    max_outer_radius: float = 5.0,
    bg_sigma_clip_sigma: float = 3.0,
    bg_sigma_clip_maxiters: int = 3,
    max_annulus_attempts: int = 5,
    annulus_expansion_step: float = 0.5,
    annulus_inner_offset: float = 1.414,
) -> Tuple[float, float, int]:
    """
    Estimate local background using annular region around source.

    Annulus radii are calculated automatically based on aperture size and
    configuration parameters.

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
    min_usable_pixels : int
        Minimum number of usable pixels required
    max_outer_radius : float
        Maximum outer radius
    bg_sigma_clip_sigma : float
        Sigma threshold for sigma clipping
    bg_sigma_clip_maxiters : int
        Maximum iterations for sigma clipping
    max_annulus_attempts : int
        Maximum attempts to expand annulus
    annulus_expansion_step : float
        Step size for annulus expansion
    annulus_inner_offset : float
        Offset from aperture edge to inner annulus

    Returns
    -------
    background_level : float
        Background level per pixel (MJy/sr)
    background_error : float
        Background error per pixel (MJy/sr)
    n_usable : int
        Number of usable pixels in annulus

    Notes
    -----
    Inner radius = aperture_radius + annulus_inner_offset
    Outer radius is calculated to achieve minimum annulus area, capped at max_outer_radius.
    """
    from ..utils.spherex_mef import create_background_mask

    # Determine annulus radii automatically
    inner_r, outer_r = determine_annulus_radii(
        aperture_radius, min_usable_pixels, max_outer_radius, annulus_inner_offset
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

    while attempt < max_annulus_attempts:
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

        # If this is the last attempt and we still don't have enough pixels, try relaxed masking
        if attempt == max_annulus_attempts - 1:
            logger.warning(
                f"Insufficient usable pixels with strict masking: {n_usable}/{min_usable_pixels} required. "
                f"Trying with relaxed masking (exclude_bad_only=False)."
            )
            clean_mask = create_background_mask(flags, False)
            usable_mask = annulus_mask & clean_mask
            n_usable = np.sum(usable_mask)
            
            if n_usable >= min_usable_pixels:
                break

        # Expand outer radius if possible
        if outer_r < max_outer_radius and outer_r < max_distance:
            new_outer_r = min(outer_r + annulus_expansion_step, max_outer_radius, max_distance)
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
    bg_mean, bg_median, bg_std = sigma_clipped_stats(
        bg_pixels, sigma=bg_sigma_clip_sigma, maxiters=bg_sigma_clip_maxiters
    )

    # Error on the mean background
    bg_error = bg_std / np.sqrt(n_usable)

    logger.debug(
        f"Background estimate: {bg_median:.6f} ± {bg_error:.6f} MJy/sr "
        f"from {n_usable} pixels (r={inner_r:.1f}-{outer_r:.1f})"
    )

    return float(bg_median), float(bg_error), n_usable
