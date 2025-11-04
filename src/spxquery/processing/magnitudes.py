"""
Magnitude conversion functions for SPHEREx photometry.

Converts flux density measurements in Jansky to AB magnitude system using astropy units.
"""

import logging
from typing import Optional, Tuple

import astropy.units as u
import numpy as np

logger = logging.getLogger(__name__)


def flux_jy_to_ab_magnitude(flux_jy: float, flux_error_jy: float, wavelength_micron: float) -> Tuple[float, float]:
    """
    Convert flux density in Jansky to AB magnitude using astropy units.

    AB magnitude is defined as:
    m_AB = -2.5 * log10(f_nu) - 48.6
    where f_nu is flux density in erg/s/cm²/Hz

    Parameters
    ----------
    flux_jy : float
        Flux density in Jansky
    flux_error_jy : float
        Flux density error in Jansky
    wavelength_micron : float
        Central wavelength in microns

    Returns
    -------
    mag_ab : float
        AB magnitude
    mag_ab_error : float
        AB magnitude error
    """
    if flux_jy <= 0:
        logger.warning(f"Non-positive flux {flux_jy} Jy - returning NaN magnitude")
        return np.nan, np.nan

    # Create astropy Quantity objects with proper units
    flux_density = flux_jy * u.Jy
    flux_density_error = flux_error_jy * u.Jy
    wavelength = wavelength_micron * u.micron

    # Convert flux density to CGS units for AB magnitude calculation
    # AB magnitude reference: f_nu in erg/s/cm²/Hz with zero point 48.6
    flux_density_cgs = flux_density.to(u.erg / u.s / u.cm**2 / u.Hz)
    flux_density_error_cgs = flux_density_error.to(u.erg / u.s / u.cm**2 / u.Hz)

    # AB magnitude calculation
    mag_ab = -2.5 * np.log10(flux_density_cgs.value) - 48.6

    # Error propagation for magnitude: d(mag)/d(flux) = -2.5 / (ln(10) * flux)
    mag_ab_error = 2.5 / (np.log(10) * flux_density_cgs.value) * flux_density_error_cgs.value

    return float(mag_ab), float(mag_ab_error)


def calculate_ab_magnitude_from_jy(
    flux_jy: float, flux_error_jy: float, wavelength_micron: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate AB magnitude from flux density in Jansky.

    Parameters
    ----------
    flux_jy : float
        Flux density in Jansky
    flux_error_jy : float
        Flux density error in Jansky
    wavelength_micron : float
        Central wavelength in microns

    Returns
    -------
    mag_ab : float or None
        AB magnitude
    mag_ab_error : float or None
        AB magnitude error
    """
    try:
        # Calculate AB magnitude
        mag_ab, mag_ab_error = flux_jy_to_ab_magnitude(flux_jy, flux_error_jy, wavelength_micron)

        # Return None for NaN values
        mag_ab = None if np.isnan(mag_ab) else mag_ab
        mag_ab_error = None if np.isnan(mag_ab_error) else mag_ab_error

        logger.debug(
            f"AB magnitude for {flux_jy:.6f} Jy at {wavelength_micron:.3f} μm: {mag_ab:.3f}±{mag_ab_error:.3f}"
        )

        return mag_ab, mag_ab_error

    except Exception as e:
        logger.error(f"Failed to calculate AB magnitude: {e}")
        return None, None


def magnitude_to_flux_jy(mag_ab: float, mag_ab_error: float, wavelength_micron: float) -> Tuple[float, float]:
    """
    Convert AB magnitude back to flux density in Jansky.

    Useful for validation and upper limit calculations.

    Parameters
    ----------
    mag_ab : float
        AB magnitude
    mag_ab_error : float
        AB magnitude error
    wavelength_micron : float
        Central wavelength in microns

    Returns
    -------
    flux_jy : float
        Flux density in Jansky
    flux_error_jy : float
        Flux density error in Jansky
    """
    # Convert magnitude to flux density (erg/s/cm²/Hz)
    flux_density_cgs_value = 10 ** (-0.4 * (mag_ab + 48.6))

    # Error propagation: df/dm = -0.4 * ln(10) * f
    flux_density_error_cgs_value = 0.4 * np.log(10) * flux_density_cgs_value * mag_ab_error

    # Create astropy quantities with proper units
    flux_density_cgs = flux_density_cgs_value * u.erg / u.s / u.cm**2 / u.Hz
    flux_density_error_cgs = flux_density_error_cgs_value * u.erg / u.s / u.cm**2 / u.Hz

    # Convert to Jansky
    flux_jy_quantity = flux_density_cgs.to(u.Jy)
    flux_error_jy_quantity = flux_density_error_cgs.to(u.Jy)

    return float(flux_jy_quantity.value), float(flux_error_jy_quantity.value)
