"""
Output spatial WCS construction for Drizzle3D.

Builds a tangent-plane (TAN) projection centered on the user-specified
(ra, dec) with pixel scale determined by xy_oversample.
"""

import logging

from astropy.wcs import WCS

from .config import Drizzle3DConfig

logger = logging.getLogger(__name__)


def build_output_wcs(config: Drizzle3DConfig) -> WCS:
    """Build output spatial WCS from Drizzle3DConfig.

    Creates a 2-D gnomonic (TAN) projection centered on (center_ra, center_dec)
    with pixel scale = 6.15" / xy_oversample.  The image extent matches
    (output_ny, output_nx) from the config.

    Parameters
    ----------
    config : Drizzle3DConfig
        Drizzle configuration with center, size, and oversampling parameters.

    Returns
    -------
    WCS
        2-D celestial WCS for the output spatial grid.

    Notes
    -----
    CRPIX is placed at the geometric center of the output image so that
    pixel (nx//2, ny//2) maps to (center_ra, center_dec).
    """
    nx = config.output_nx()
    ny = config.output_ny()
    pixscale_deg = config.effective_pixscale() / 3600.0  # arcsec → deg

    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2.0 + 0.5, ny / 2.0 + 0.5]  # FITS 1-based center
    w.wcs.crval = [config.center_ra, config.center_dec]
    w.wcs.cdelt = [-pixscale_deg, pixscale_deg]  # RA decreases with x
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]

    logger.info(
        f"Output WCS: {nx}×{ny} pixels, "
        f'pixscale={config.effective_pixscale():.3f}"/pix, '
        f"center=({config.center_ra:.4f}, {config.center_dec:.4f})"
    )
    return w
