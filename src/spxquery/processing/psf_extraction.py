"""
PSF extraction for SPHEREx spectral images.

This module implements PSF extraction from SPHEREx cutouts following the IRSA tutorial.
SPHEREx PSFs are provided as a 121-plane cube with 11x11 zones, each containing a
101x101 pixel oversampled PSF (10x oversampling factor).

References:
    IRSA SPHEREx PSF Tutorial: https://irsa.ipac.caltech.edu/data/SPHEREx/docs/
"""

import re
from typing import Tuple

import numpy as np
from astropy.io import fits
from astropy.nddata import block_reduce


def extract_spherex_psf(
    psf_cube: np.ndarray,
    psf_header: fits.Header,
    x_parent: float,
    y_parent: float,
    downsample: bool = True,
    oversample_factor: int = 10,
) -> Tuple[np.ndarray, int, float]:
    """
    Extract PSF from SPHEREx PSF cube using nearest zone method.

    This function follows the IRSA tutorial algorithm:
    1. Extract zone center coordinates from XCTR_*/YCTR_* header keywords
    2. Find nearest zone to source position using Euclidean distance
    3. Extract corresponding PSF from cube
    4. Optionally downsample from oversampled to native resolution

    Parameters
    ----------
    psf_cube : np.ndarray
        PSF data cube with shape (121, 101, 101) - 121 zones, each 101x101 pixels
    psf_header : fits.Header
        PSF extension header containing XCTR_*/YCTR_* keywords
    x_parent : float
        X pixel coordinate on parent SPHEREx image (1-indexed FITS convention)
    y_parent : float
        Y pixel coordinate on parent SPHEREx image (1-indexed FITS convention)
    downsample : bool, optional
        Whether to downsample PSF to native resolution (default: True)
    oversample_factor : int, optional
        PSF oversampling factor from header OVERSAMP keyword (default: 10)

    Returns
    -------
    psf : np.ndarray
        Extracted PSF, either oversampled (101x101) or native (~10x10) resolution
    zone_id : int
        Zone ID (1-121) of the extracted PSF
    distance : float
        Euclidean distance in pixels from zone center to source position

    Raises
    ------
    ValueError
        If no XCTR_*/YCTR_* keywords found in header
        If PSF cube shape is invalid
        If oversample_factor is invalid

    Notes
    -----
    - SPHEREx PSFs are distributed in an 11x11 grid (121 zones total)
    - Each PSF is 101x101 pixels at 10x oversampling (0.615 arcsec/pixel)
    - Native SPHEREx resolution is 6.2 arcsec/pixel
    - Zone centers are in 1-indexed FITS convention
    - Downsampling uses astropy.nddata.block_reduce with sum aggregation

    Examples
    --------
    >>> with fits.open('spherex_cutout.fits') as hdul:
    ...     psf_cube = hdul['PSF'].data
    ...     psf_header = hdul['PSF'].header
    ...     psf, zone_id, dist = extract_spherex_psf(
    ...         psf_cube, psf_header, x_parent=1020.5, y_parent=1020.5
    ...     )
    >>> psf.shape  # Native resolution after downsampling
    (10, 10)
    """
    # Validate inputs
    if psf_cube.ndim != 3:
        raise ValueError(f"PSF cube must be 3D, got shape {psf_cube.shape}")

    n_zones, psf_height, psf_width = psf_cube.shape
    if n_zones != 121:
        raise ValueError(f"Expected 121 PSF zones, got {n_zones}")

    if oversample_factor <= 0:
        raise ValueError(f"oversample_factor must be > 0, got {oversample_factor}")

    # Extract zone center coordinates from header
    xctr = {}
    yctr = {}

    for key, val in psf_header.items():
        # Look for keys like XCTR_1, XCTR_2, ..., XCTR_121
        xm = re.match(r"XCTR_(\d+)", key)
        if xm:
            zone_id = int(xm.group(1))
            xctr[zone_id] = val

        # Look for keys like YCTR_1, YCTR_2, ..., YCTR_121
        ym = re.match(r"YCTR_(\d+)", key)
        if ym:
            zone_id = int(ym.group(1))
            yctr[zone_id] = val

    if not xctr or not yctr:
        raise ValueError("No XCTR_*/YCTR_* keywords found in PSF header")

    if len(xctr) != len(yctr):
        raise ValueError(f"Mismatch in XCTR ({len(xctr)}) and YCTR ({len(yctr)}) counts")

    # Find nearest zone using Euclidean distance
    # Note: Zone centers are in 1-indexed FITS convention
    min_distance = float("inf")
    nearest_zone_id = None

    for zone_id in xctr.keys():
        # Convert zone center from 1-indexed to match x_parent, y_parent
        # (x_parent, y_parent are already in 1-indexed convention from calling code)
        dx = xctr[zone_id] - x_parent
        dy = yctr[zone_id] - y_parent
        distance = np.sqrt(dx**2 + dy**2)

        if distance < min_distance:
            min_distance = distance
            nearest_zone_id = zone_id

    if nearest_zone_id is None:
        raise ValueError("Failed to find nearest PSF zone")

    # Extract PSF from cube (convert zone_id from 1-indexed to 0-indexed)
    psf_oversampled = psf_cube[nearest_zone_id - 1]

    # Ensure PSF is properly normalized (sum = 1.0) for model fitting
    psf_sum = np.sum(psf_oversampled)
    if abs(psf_sum - 1.0) > 1e-6:  # Only normalize if needed
        psf_oversampled = psf_oversampled / psf_sum

    # Optionally downsample to native resolution
    if downsample:
        # Use block_reduce with sum to preserve flux
        # Each oversampled pixel becomes 1/oversample_factor^2 of a native pixel
        psf_native = block_reduce(psf_oversampled, oversample_factor, func=np.sum)
        # The block_reduce with sum already preserves the total flux due to PSF normalization above
        # No additional scaling needed since PSF is already normalized to sum=1.0
        return psf_native, nearest_zone_id, min_distance
    else:
        return psf_oversampled, nearest_zone_id, min_distance


def extract_psf_from_cutout(
    cutout_header: fits.Header,
    psf_cube: np.ndarray,
    psf_header: fits.Header,
    ra: float,
    dec: float,
    downsample: bool = True,
) -> Tuple[np.ndarray, int, float]:
    """
    Extract PSF from cutout by first converting sky coordinates to parent image pixels.

    This is a convenience wrapper that handles the coordinate transformation from
    (RA, Dec) to parent image pixel coordinates before calling extract_spherex_psf.

    Parameters
    ----------
    cutout_header : fits.Header
        IMAGE extension header from cutout (contains WCS and CRPIX1A/CRPIX2A)
    psf_cube : np.ndarray
        PSF data cube with shape (121, 101, 101)
    psf_header : fits.Header
        PSF extension header containing XCTR_*/YCTR_* keywords
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    downsample : bool, optional
        Whether to downsample PSF to native resolution (default: True)

    Returns
    -------
    psf : np.ndarray
        Extracted PSF, either oversampled (101x101) or native (~10x10) resolution
    zone_id : int
        Zone ID (1-121) of the extracted PSF
    distance : float
        Euclidean distance in pixels from zone center to source position

    Raises
    ------
    ValueError
        If CRPIX1A or CRPIX2A keywords not found in cutout header
        If WCS transformation fails

    Notes
    -----
    This function performs the coordinate transformation described in the IRSA tutorial:
    1. Convert (RA, Dec) to cutout pixel coordinates using WCS
    2. Use CRPIX1A/CRPIX2A to shift to parent image coordinates
    3. Extract PSF using parent coordinates

    Examples
    --------
    >>> with fits.open('spherex_cutout.fits') as hdul:
    ...     cutout_hdr = hdul['IMAGE'].header
    ...     psf_cube = hdul['PSF'].data
    ...     psf_hdr = hdul['PSF'].header
    ...     psf, zone_id, dist = extract_psf_from_cutout(
    ...         cutout_hdr, psf_cube, psf_hdr, ra=305.59875, dec=41.14889
    ...     )
    """
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS

    # Convert sky coordinates to cutout pixel coordinates
    wcs = WCS(cutout_header)
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    xpix_cutout, ypix_cutout = wcs.world_to_pixel(coord)

    # Get cutout center on parent image
    if "CRPIX1A" not in cutout_header or "CRPIX2A" not in cutout_header:
        raise ValueError("CRPIX1A/CRPIX2A keywords not found in cutout header")

    crpix1a = cutout_header["CRPIX1A"]
    crpix2a = cutout_header["CRPIX2A"]

    # Convert cutout pixel coordinates to parent image coordinates
    # Formula from IRSA tutorial:
    # x_parent = 1 + x_cutout - CRPIX1A
    # y_parent = 1 + y_cutout - CRPIX2A
    x_parent = 1 + xpix_cutout - crpix1a
    y_parent = 1 + ypix_cutout - crpix2a

    # Extract PSF using parent coordinates
    return extract_spherex_psf(
        psf_cube=psf_cube,
        psf_header=psf_header,
        x_parent=x_parent,
        y_parent=y_parent,
        downsample=downsample,
    )


def extract_psf_from_mef(
    mef,
    ra: float,
    dec: float,
    downsample: bool = True,
) -> Tuple[np.ndarray, int, float]:
    """
    Extract PSF from SPHERExMEF object for given sky coordinates.

    This is a convenience wrapper for working with SPHERExMEF objects from
    fits_handler.read_spherex_mef(). It uses the existing WCS and PSF data.

    Parameters
    ----------
    mef : SPHERExMEF
        SPHEREx Multi-Extension FITS data container
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    downsample : bool, optional
        Whether to downsample PSF to native resolution (default: True)

    Returns
    -------
    psf : np.ndarray
        Extracted PSF, either oversampled (101×101) or native (~10×10) resolution
    zone_id : int
        Zone ID (1-121) of the extracted PSF
    distance : float
        Euclidean distance in pixels from zone center to source position

    Examples
    --------
    >>> from spxquery.processing.fits_handler import read_spherex_mef
    >>> mef = read_spherex_mef('spherex_cutout.fits')
    >>> psf, zone_id, dist = extract_psf_from_mef(mef, ra=305.59875, dec=41.14889)
    """
    return extract_psf_from_cutout(
        cutout_header=mef.header,
        psf_cube=mef.psf,
        psf_header=mef.psf_header,  # PSF header contains zone coordinate keywords
        ra=ra,
        dec=dec,
        downsample=downsample,
    )
