"""
Utilities for extracting and processing SPHEREx PSF data.

SPHEREx PSF cubes contain 121 PSF planes arranged in an 11×11 grid across the
detector. Each PSF is 101×101 pixels with 10× oversampling.
"""

import logging
import re
from typing import Dict, Tuple

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)


def get_psf_zone_centers(psf_header: fits.Header) -> np.ndarray:
    """
    Extract PSF zone center coordinates from PSF header.

    SPHEREx PSF cubes have 121 zones (11×11 grid) with center coordinates
    stored in XCTR_i and YCTR_i keywords (i=1...121).

    Parameters
    ----------
    psf_header : fits.Header
        FITS header from PSF extension.

    Returns
    -------
    zone_centers : np.ndarray
        Array of shape (121, 3) with columns [zone_id, x_center, y_center].
        Coordinates are 1-indexed pixel positions on the parent image.

    Notes
    -----
    Zone centers are in 1-indexed pixel coordinates (FITS convention).
    """
    xctr = {}
    yctr = {}

    # Extract XCTR_* and YCTR_* keywords
    for key, val in psf_header.items():
        # Match XCTR_* keys
        xm = re.match(r"XCTR", key)
        if xm:
            zone_id = int(key.split("_")[1])
            xctr[zone_id] = val

        # Match YCTR_* keys
        ym = re.match(r"YCTR", key)
        if ym:
            zone_id = int(key.split("_")[1])
            yctr[zone_id] = val

    # Validate we got all 121 zones
    if len(xctr) != 121 or len(yctr) != 121:
        raise ValueError(
            f"Expected 121 PSF zones, got {len(xctr)} XCTR and {len(yctr)} YCTR keywords"
        )

    # Create array: [zone_id, x_center, y_center]
    zone_centers = np.zeros((121, 3))
    for i, zone_id in enumerate(sorted(xctr.keys())):
        zone_centers[i, 0] = zone_id
        zone_centers[i, 1] = xctr[zone_id]
        zone_centers[i, 2] = yctr[zone_id]

    logger.debug(f"Extracted {len(zone_centers)} PSF zone centers")
    return zone_centers


def map_cutout_to_parent_coords(
    cutout_x: float, cutout_y: float, cutout_header: fits.Header
) -> Tuple[float, float]:
    """
    Convert cutout pixel coordinates to parent image coordinates.

    SPHEREx cutouts store the cutout center position on the parent image in
    CRPIX1A and CRPIX2A keywords.

    Parameters
    ----------
    cutout_x : float
        X pixel coordinate on cutout (0-indexed).
    cutout_y : float
        Y pixel coordinate on cutout (0-indexed).
    cutout_header : fits.Header
        FITS header from cutout IMAGE extension.

    Returns
    -------
    parent_x : float
        X pixel coordinate on parent image (0-indexed).
    parent_y : float
        Y pixel coordinate on parent image (0-indexed).

    Notes
    -----
    Cutout pixel coordinates are 0-indexed (Python convention).
    Parent image coordinates are converted to 0-indexed for consistency.
    CRPIX1A/CRPIX2A are in FITS 1-indexed convention.

    The conversion formula is:
        parent_coord = 1 + cutout_coord - CRPIX_A
    where the result is 0-indexed.
    """
    crpix1a = cutout_header["CRPIX1A"]
    crpix2a = cutout_header["CRPIX2A"]

    # Convert: add 1 for FITS indexing, subtract CRPIX offset
    # Result is 0-indexed parent coordinate
    parent_x = 1 + cutout_x - crpix1a
    parent_y = 1 + cutout_y - crpix2a

    logger.debug(
        f"Mapped cutout coords ({cutout_x:.2f}, {cutout_y:.2f}) to "
        f"parent coords ({parent_x:.2f}, {parent_y:.2f})"
    )

    return parent_x, parent_y


def find_psf_zone(
    zone_centers: np.ndarray, source_x: float, source_y: float
) -> int:
    """
    Find the PSF zone closest to source position.

    Parameters
    ----------
    zone_centers : np.ndarray
        Array of shape (121, 3) from get_psf_zone_centers().
        Columns: [zone_id, x_center, y_center].
    source_x : float
        Source X coordinate on parent image (0-indexed).
    source_y : float
        Source Y coordinate on parent image (0-indexed).

    Returns
    -------
    zone_id : int
        Zone ID (1-121) of closest PSF zone.

    Notes
    -----
    Zone centers are 1-indexed, so we subtract 1 before distance calculation
    to match 0-indexed source coordinates.
    """
    # Zone centers are 1-indexed, convert to 0-indexed for distance calc
    x_centers = zone_centers[:, 1] - 1
    y_centers = zone_centers[:, 2] - 1

    # Calculate Euclidean distance to each zone center
    distances = np.sqrt((x_centers - source_x) ** 2 + (y_centers - source_y) ** 2)

    # Find closest zone
    min_idx = np.argmin(distances)
    zone_id = int(zone_centers[min_idx, 0])
    min_distance = distances[min_idx]

    logger.debug(
        f"Closest PSF zone is {zone_id} at distance {min_distance:.2f} pixels"
    )

    return zone_id


def extract_psf_zone(psf_cube: np.ndarray, zone_id: int) -> np.ndarray:
    """
    Extract PSF from cube using zone ID.

    Parameters
    ----------
    psf_cube : np.ndarray
        PSF data cube of shape (121, 101, 101).
    zone_id : int
        Zone ID (1-121, FITS convention).

    Returns
    -------
    psf : np.ndarray
        PSF array of shape (101, 101).

    Raises
    ------
    ValueError
        If zone_id is not in range 1-121.
    """
    if not 1 <= zone_id <= 121:
        raise ValueError(f"zone_id must be 1-121, got {zone_id}")

    # Convert 1-indexed zone_id to 0-indexed array index
    psf = psf_cube[zone_id - 1]

    logger.debug(f"Extracted PSF zone {zone_id}, shape: {psf.shape}")

    return psf


def normalize_psf(psf_array: np.ndarray) -> np.ndarray:
    """
    Normalize PSF to integrate to 1.0.

    This ensures flux conservation when convolving with a point source.

    Parameters
    ----------
    psf_array : np.ndarray
        PSF array of shape (101, 101).

    Returns
    -------
    psf_normalized : np.ndarray
        Normalized PSF with sum = 1.0.

    Raises
    ------
    ValueError
        If PSF sum is zero or negative.
    """
    psf_sum = np.sum(psf_array)

    if psf_sum <= 0:
        raise ValueError(f"PSF sum must be positive, got {psf_sum}")

    psf_normalized = psf_array / psf_sum

    logger.debug(
        f"Normalized PSF: original sum = {psf_sum:.6e}, "
        f"normalized sum = {np.sum(psf_normalized):.6e}"
    )

    return psf_normalized


def get_psf_for_source(
    psf_cube: np.ndarray,
    psf_header: fits.Header,
    cutout_header: fits.Header,
    source_x_cutout: float,
    source_y_cutout: float,
) -> Tuple[np.ndarray, int]:
    """
    Complete workflow to extract and normalize PSF for a source.

    This is a convenience function that combines all PSF extraction steps.

    Parameters
    ----------
    psf_cube : np.ndarray
        PSF data cube from MEF file, shape (121, 101, 101).
    psf_header : fits.Header
        PSF extension header with XCTR_*/YCTR_* keywords.
    cutout_header : fits.Header
        IMAGE extension header with CRPIX1A/CRPIX2A keywords.
    source_x_cutout : float
        Source X coordinate on cutout (0-indexed).
    source_y_cutout : float
        Source Y coordinate on cutout (0-indexed).

    Returns
    -------
    psf_normalized : np.ndarray
        Normalized PSF array, shape (101, 101), sum = 1.0.
    zone_id : int
        Zone ID (1-121) used for this PSF.

    Examples
    --------
    >>> from astropy.io import fits
    >>> with fits.open("spherex_cutout.fits") as hdul:
    ...     psf_cube = hdul['PSF'].data
    ...     psf_header = hdul['PSF'].header
    ...     image_header = hdul['IMAGE'].header
    >>> psf, zone = get_psf_for_source(
    ...     psf_cube, psf_header, image_header,
    ...     source_x_cutout=50.5, source_y_cutout=50.5
    ... )
    """
    # Step 1: Get zone centers from PSF header
    zone_centers = get_psf_zone_centers(psf_header)

    # Step 2: Map cutout coords to parent image coords
    source_x_parent, source_y_parent = map_cutout_to_parent_coords(
        source_x_cutout, source_y_cutout, cutout_header
    )

    # Step 3: Find closest PSF zone
    zone_id = find_psf_zone(zone_centers, source_x_parent, source_y_parent)

    # Step 4: Extract PSF from cube
    psf = extract_psf_zone(psf_cube, zone_id)

    # Step 5: Normalize PSF
    psf_normalized = normalize_psf(psf)

    logger.info(
        f"Extracted and normalized PSF zone {zone_id} for source at "
        f"cutout coords ({source_x_cutout:.2f}, {source_y_cutout:.2f})"
    )

    return psf_normalized, zone_id
