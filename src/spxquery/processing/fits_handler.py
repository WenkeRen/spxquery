"""
FITS Multi-Extension File handling for SPHEREx data.
"""

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import astropy.units as u
import numpy as np
from astropy import log as astropy_log
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


@contextmanager
def suppress_astropy_info():
    """
    Context manager to temporarily suppress astropy INFO messages and FITS warnings.

    This is needed because SPHEREx provides non-standard WCS headers that trigger
    harmless but annoying INFO messages about SIP distortion coefficients and
    warnings about redundant SCAMP distortion parameters.
    """
    original_level = astropy_log.level
    astropy_log.setLevel("WARNING")

    try:
        # Suppress specific FITSFixedWarning about redundant SCAMP distortion parameters
        with warnings.catch_warnings():
            # Catch the warning by message pattern, regardless of category
            warnings.filterwarnings("ignore", message=".*Removed redundant SCAMP distortion parameters.*")
            warnings.filterwarnings("ignore", message=".*because SIP parameters are also present.*")
            yield
    finally:
        astropy_log.setLevel(original_level)


@dataclass
class SPHERExMEF:
    """Container for SPHEREx Multi-Extension FITS data."""

    filepath: Path
    image: np.ndarray  # Calibrated flux in MJy/sr
    flags: np.ndarray  # Bitmap flags
    variance: np.ndarray  # Variance in (MJy/sr)^2
    zodi: np.ndarray  # Zodiacal light model in MJy/sr
    psf: np.ndarray  # PSF cube (101x101x121)
    spatial_wcs: WCS  # Primary astrometric WCS
    spectral_wcs: WCS  # Alternative spectral WCS
    header: fits.Header  # Primary image header
    obs_id: str
    detector: int
    mjd: float

    @property
    def image_zodi_subtracted(self) -> np.ndarray:
        """Return zodiacal light subtracted image with amplitude scaling."""
        corrected_image, _ = subtract_zodiacal_background(self.image, self.zodi, self.flags, self.variance)
        return corrected_image

    @property
    def error(self) -> np.ndarray:
        """Return error array (sqrt of variance)."""
        return np.sqrt(self.variance)


def read_spherex_mef(filepath: Path) -> SPHERExMEF:
    """
    Read SPHEREx Multi-Extension FITS file.

    Parameters
    ----------
    filepath : Path
        Path to SPHEREx MEF file

    Returns
    -------
    SPHERExMEF
        Container with all MEF data
    """
    logger.info(f"Reading SPHEREx MEF: {filepath}")

    with fits.open(filepath) as hdulist:
        # Verify expected structure
        if len(hdulist) < 7:
            raise ValueError(f"Expected at least 7 extensions, got {len(hdulist)}")

        # Read IMAGE extension
        image_hdu = hdulist["IMAGE"]
        image_data = image_hdu.data.astype(np.float32)
        image_header = image_hdu.header

        # Verify units are as expected (MJy/sr)
        bunit = image_header.get("BUNIT", "").strip().upper()
        if bunit and bunit not in ["MJY/SR", "MJY / SR", "MJY SR-1", "MJY/STERADIAN"]:
            logger.warning(f"Unexpected BUNIT '{bunit}' in {filepath}. Expected 'MJy/sr'")
        elif bunit:
            logger.debug(f"Verified BUNIT: {bunit}")
        else:
            logger.warning(f"Missing BUNIT header in {filepath}. Assuming MJy/sr")

        # Read other extensions
        flags_data = hdulist["FLAGS"].data.astype(np.int32)
        variance_data = hdulist["VARIANCE"].data.astype(np.float32)
        zodi_data = hdulist["ZODI"].data.astype(np.float32)
        psf_data = hdulist["PSF"].data.astype(np.float32)

        # Load WCS with suppressed warnings about SCAMP/SIP distortion parameters
        with suppress_astropy_info():
            # Load spatial WCS (primary)
            spatial_wcs = WCS(image_header)

            # Load spectral WCS (alternative 'W')
            # Need to pass HDUList for lookup table access
            spectral_wcs = WCS(header=image_header, fobj=hdulist, key="W")
            # Disable SIP distortion for spectral WCS
            spectral_wcs.sip = None

        # Extract metadata
        obs_id = image_header.get("OBSID", filepath.stem)
        detector = image_header.get("DETECTOR", 0)

        # Calculate MJD
        t_min = image_header.get("MJD-BEG", 0)
        t_max = image_header.get("MJD-END", 0)
        mjd = (t_min + t_max) / 2.0

        mef = SPHERExMEF(
            filepath=filepath,
            image=image_data,
            flags=flags_data,
            variance=variance_data,
            zodi=zodi_data,
            psf=psf_data,
            spatial_wcs=spatial_wcs,
            spectral_wcs=spectral_wcs,
            header=image_header,
            obs_id=obs_id,
            detector=detector,
            mjd=mjd,
        )

        logger.info(f"Loaded {obs_id}: detector {detector}, shape {image_data.shape}")

        return mef


def get_pixel_coordinates(mef: SPHERExMEF, ra: float, dec: float) -> Tuple[float, float]:
    """
    Convert RA/Dec to pixel coordinates.

    Parameters
    ----------
    mef : SPHERExMEF
        SPHEREx MEF data
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees

    Returns
    -------
    x, y : float
        Pixel coordinates (0-based)
    """
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    x, y = mef.spatial_wcs.world_to_pixel(coord)

    # Check if coordinates are within image
    ny, nx = mef.image.shape
    if not (0 <= x < nx and 0 <= y < ny):
        logger.warning(f"Coordinates ({x:.1f}, {y:.1f}) outside image bounds ({nx}, {ny})")

    return float(x), float(y)


def get_wavelength_at_position(mef: SPHERExMEF, x: float, y: float) -> Tuple[float, float]:
    """
    Get wavelength and bandwidth at pixel position.

    Parameters
    ----------
    mef : SPHERExMEF
        SPHEREx MEF data
    x, y : float
        Pixel coordinates (0-based)

    Returns
    -------
    wavelength : float
        Central wavelength in microns
    bandwidth : float
        Bandwidth in microns
    """
    # Use spectral WCS to get wavelength info
    spectral_coords = mef.spectral_wcs.pixel_to_world(x, y)

    # spectral_coords is a tuple of (wavelength, bandpass)
    wavelength = spectral_coords[0].to(u.micron).value
    bandwidth = spectral_coords[1].to(u.micron).value

    return wavelength, bandwidth


def get_pixel_scale_at_position(wcs: WCS, x: float, y: float, pixel_scale_fallback: float = 6.2) -> float:
    """
    Calculate the pixel scale in arcsec/pixel at a given position using WCS.

    This accounts for any distortions and provides the actual pixel scale
    at the specified position rather than assuming a fixed value.

    Parameters
    ----------
    wcs : WCS
        World Coordinate System object
    x, y : float
        Pixel coordinates (0-based)
    pixel_scale_fallback : float
        Fallback pixel scale in arcsec/pixel if WCS fails (default: 6.2 for SPHEREx)

    Returns
    -------
    float
        Pixel scale in arcseconds per pixel
    """
    # Calculate pixel scale using the determinant of the CD matrix
    # This gives the area of a pixel in degrees²
    try:
        # Get the pixel scale from the WCS at the specified position
        pixel_scales = wcs.proj_plane_pixel_scales()  # Returns scales in degrees/pixel

        # For SPHEREx, pixels should be roughly square, so take the geometric mean
        # of the two pixel scales to get a representative scale
        pixel_scale_deg = float(np.sqrt(pixel_scales[0] * pixel_scales[1]).value)

        # Convert from degrees to arcseconds
        pixel_scale_arcsec = pixel_scale_deg * 3600.0

        logger.debug(f"WCS pixel scale at ({x:.1f}, {y:.1f}): {pixel_scale_arcsec:.3f} arcsec/pixel")

        return pixel_scale_arcsec

    except Exception as e:
        logger.warning(
            f"Failed to calculate pixel scale from WCS: {e}. Using fallback {pixel_scale_fallback} arcsec/pixel"
        )
        return pixel_scale_fallback  # Fallback to configured pixel scale


def get_flag_info(flag_value: int) -> Dict[str, bool]:
    """
    Decode flag bitmap into individual flags.

    Parameters
    ----------
    flag_value : int
        Combined flag bitmap value

    Returns
    -------
    Dict[str, bool]
        Dictionary of flag names and their states
    """
    # Flag definitions from SPHEREx
    flags = {
        "TRANSIENT": 0,
        "OVERFLOW": 1,
        "SUR_ERROR": 2,
        "NONFUNC": 6,
        "DICHROIC": 7,
        "MISSING_DATA": 9,
        "HOT": 10,
        "COLD": 11,
        "FULLSAMPLE": 12,
        "PHANMISS": 14,
        "NONLINEAR": 15,
        "PERSIST": 17,
        "OUTLIER": 19,
        "SOURCE": 21,
    }

    flag_states = {}
    for name, bit in flags.items():
        flag_states[name] = bool(flag_value & (1 << bit))

    return flag_states


def format_flag_binary(flag_value: int, num_bits: int = 22) -> str:
    """
    Format flag value as binary string.

    Parameters
    ----------
    flag_value : int
        Flag bitmap value
    num_bits : int
        Number of bits to display

    Returns
    -------
    str
        Binary string representation
    """
    return format(flag_value, f"0{num_bits}b")


def create_background_mask(flags: np.ndarray) -> np.ndarray:
    """
    Create mask for background pixels (good for zodiacal matching).

    Masks out pixels with problematic flags including non-functional pixels, outliers, etc., but keeps SOURCE-flagged
    pixels as valid background pixels for local background estimation.

    Parameters
    ----------
    flags : np.ndarray
        Flag bitmap array

    Returns
    -------
    np.ndarray
        Boolean mask (True = good background pixel)
    """
    # Define flags that should be masked out for background estimation
    # Based on SPHEREx flag definitions
    # NOTE: SOURCE (bit 21) is intentionally EXCLUDED from this list
    # to allow source pixels to be used in local background annuli
    bad_flags = {
        "TRANSIENT": 0,  # Transient detections
        "OVERFLOW": 1,  # Overflow pixels
        "SUR_ERROR": 2,  # Processing errors
        "PHANTOM": 4,  # Phantom pixels
        "NONFUNC": 6,  # Non-functional pixels
        "MISSING_DATA": 9,  # Missing data
        "HOT": 10,  # Hot pixels
        "COLD": 11,  # Anomalously low signal
        "PHANMISS": 14,  # Phantom correction missing
        "NONLINEAR": 15,  # Nonlinearity issues
        "PERSIST": 17,  # Persistent charge
        "OUTLIER": 19,  # Statistical outliers
    }

    # Create combined mask
    mask = np.ones(flags.shape, dtype=bool)  # Start with all good

    for flag_name, bit in bad_flags.items():
        flag_mask = (flags & (1 << bit)) != 0
        mask &= ~flag_mask  # Remove flagged pixels
        logger.debug(f"Masked {np.sum(flag_mask)} pixels for {flag_name}")

    n_good = np.sum(mask)
    n_total = mask.size
    logger.info(f"Background mask: {n_good}/{n_total} ({n_good / n_total * 100:.1f}%) pixels available")

    return mask


def estimate_zodiacal_scaling(
    image: np.ndarray, zodi: np.ndarray, mask: np.ndarray, variance: Optional[np.ndarray] = None
) -> float:
    """
    Estimate scaling factor to match zodiacal model to observed background.

    Uses least-squares fitting on uncontaminated pixels to find the
    multiplicative factor that best matches the zodi model to the data.

    Parameters
    ----------
    image : np.ndarray
        Observed image data
    zodi : np.ndarray
        Zodiacal model
    mask : np.ndarray
        Boolean mask (True = good background pixels)
    variance : np.ndarray, optional
        Variance array for weighted fitting

    Returns
    -------
    float
        Scaling factor for zodiacal model
    """
    # Extract background pixels
    image_bg = image[mask]
    zodi_bg = zodi[mask]

    if len(image_bg) == 0:
        logger.warning("No uncontaminated pixels for zodiacal scaling - using factor 1.0")
        return 1.0

    # Remove pixels where zodi model is zero to avoid division issues
    nonzero_mask = zodi_bg != 0
    if np.sum(nonzero_mask) == 0:
        logger.warning("Zodiacal model is zero everywhere - using factor 1.0")
        return 1.0

    image_bg = image_bg[nonzero_mask]
    zodi_bg = zodi_bg[nonzero_mask]

    # Weighted least squares if variance is provided
    if variance is not None:
        var_bg = variance[mask][nonzero_mask]
        # Avoid zero/negative variance
        valid_var = var_bg > 0
        if np.sum(valid_var) > 0:
            weights = 1.0 / var_bg[valid_var]
            image_bg = image_bg[valid_var]
            zodi_bg = zodi_bg[valid_var]

            # Weighted least squares: scale = sum(w*img*zodi) / sum(w*zodi^2)
            scale_factor = np.sum(weights * image_bg * zodi_bg) / np.sum(weights * zodi_bg**2)
        else:
            # Fall back to unweighted
            scale_factor = np.sum(image_bg * zodi_bg) / np.sum(zodi_bg**2)
    else:
        # Unweighted least squares: scale = sum(img*zodi) / sum(zodi^2)
        scale_factor = np.sum(image_bg * zodi_bg) / np.sum(zodi_bg**2)

    logger.info(f"Zodiacal scaling factor: {scale_factor:.4f}")

    return scale_factor


def subtract_zodiacal_background(
    image: np.ndarray,
    zodi: np.ndarray,
    flags: np.ndarray,
    variance: Optional[np.ndarray] = None,
    zodi_scale_min: float = 0.0,
    zodi_scale_max: float = 10.0,
) -> Tuple[np.ndarray, float]:
    """
    Subtract zodiacal light background from image with amplitude scaling.

    Uses uncontaminated background pixels to determine the optimal
    scaling factor for the zodiacal model before subtraction.

    Parameters
    ----------
    image : np.ndarray
        Original image in MJy/sr
    zodi : np.ndarray
        Zodiacal light model in MJy/sr
    flags : np.ndarray
        Flag bitmap array
    variance : np.ndarray, optional
        Variance array for weighted fitting
    zodi_scale_min : float
        Minimum allowed zodiacal scaling factor
    zodi_scale_max : float
        Maximum allowed zodiacal scaling factor

    Returns
    -------
    corrected_image : np.ndarray
        Background-subtracted image
    scale_factor : float
        Applied scaling factor for the zodiacal model
    """
    # Create mask for background estimation
    bg_mask = create_background_mask(flags)

    # Estimate zodiacal scaling factor
    scale_factor = estimate_zodiacal_scaling(image, zodi, bg_mask, variance)

    # Validate scale factor
    if scale_factor <= zodi_scale_min or scale_factor > zodi_scale_max:
        logger.warning(
            f"Unusual scaling factor {scale_factor:.4f} (outside [{zodi_scale_min}, {zodi_scale_max}]) - using 1.0"
        )
        scale_factor = 1.0

    # Apply scaled subtraction
    corrected_image = image - (scale_factor * zodi)

    logger.info(f"Subtracted zodiacal background with scaling factor {scale_factor:.4f}")

    return corrected_image, scale_factor
