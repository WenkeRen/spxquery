"""
Magnitude conversion functions for SPHEREx photometry.

Converts flux measurements in MJy/sr to AB magnitude system using astropy units.
"""

import logging
import numpy as np
from typing import Tuple, Optional

import astropy.units as u
from astropy.constants import c

logger = logging.getLogger(__name__)


def flux_jy_to_ab_magnitude(
    flux_jy: float, 
    flux_error_jy: float,
    wavelength_micron: float
) -> Tuple[float, float]:
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


def flux_to_ab_magnitude(
    flux_mjy_sr: float, 
    flux_error_mjy_sr: float,
    wavelength_micron: float
) -> Tuple[float, float]:
    """
    DEPRECATED: Convert flux in MJy/sr to AB magnitude.
    
    This function is deprecated because it incorrectly handles units.
    Use flux_jy_to_ab_magnitude() after proper unit conversion instead.
    
    AB magnitude requires flux density (Jy), not surface brightness (MJy/sr).
    The conversion requires knowledge of the aperture solid angle.
    
    Parameters
    ----------
    flux_mjy_sr : float
        Flux in MJy/sr
    flux_error_mjy_sr : float
        Flux error in MJy/sr  
    wavelength_micron : float
        Central wavelength in microns
    
    Returns
    -------
    mag_ab : float
        AB magnitude (INCORRECT - do not use)
    mag_ab_error : float
        AB magnitude error (INCORRECT - do not use)
    """
    logger.warning(
        "flux_to_ab_magnitude() is deprecated and produces incorrect results. "
        "Use flux_jy_to_ab_magnitude() after converting MJy/sr to Jy."
    )
    
    if flux_mjy_sr <= 0:
        logger.warning(f"Non-positive flux {flux_mjy_sr} MJy/sr - returning NaN magnitude")
        return np.nan, np.nan
    
    # This is the old incorrect approach - kept for compatibility but deprecated
    flux = flux_mjy_sr * u.MJy / u.sr
    flux_error = flux_error_mjy_sr * u.MJy / u.sr
    wavelength = wavelength_micron * u.micron
    
    # Convert flux to flux density (INCORRECT - missing proper solid angle conversion)
    flux_density = flux.to(u.Jy / u.sr)  # Convert to Jy/sr
    flux_density_error = flux_error.to(u.Jy / u.sr)
    
    # For AB magnitude calculation, we need flux density in standard units
    flux_density_cgs = flux_density.to(u.erg / u.s / u.cm**2 / u.Hz / u.sr)
    flux_density_error_cgs = flux_density_error.to(u.erg / u.s / u.cm**2 / u.Hz / u.sr)
    
    # INCORRECT: Remove steradian unit without proper conversion
    flux_for_magnitude = flux_density_cgs.value  # erg/s/cm²/Hz
    flux_error_for_magnitude = flux_density_error_cgs.value
    
    # AB magnitude calculation
    mag_ab = -2.5 * np.log10(flux_for_magnitude) - 48.6
    
    # Error propagation for magnitude: d(mag)/d(flux) = -2.5 / (ln(10) * flux)
    mag_ab_error = 2.5 / (np.log(10) * flux_for_magnitude) * flux_error_for_magnitude
    
    return float(mag_ab), float(mag_ab_error)


def calculate_ab_magnitude_from_jy(
    flux_jy: float,
    flux_error_jy: float,
    wavelength_micron: float
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
        mag_ab, mag_ab_error = flux_jy_to_ab_magnitude(
            flux_jy, flux_error_jy, wavelength_micron
        )
        
        # Return None for NaN values
        mag_ab = None if np.isnan(mag_ab) else mag_ab
        mag_ab_error = None if np.isnan(mag_ab_error) else mag_ab_error
        
        logger.debug(
            f"AB magnitude for {flux_jy:.6f} Jy at {wavelength_micron:.3f} μm: "
            f"{mag_ab:.3f}±{mag_ab_error:.3f}"
        )
        
        return mag_ab, mag_ab_error
        
    except Exception as e:
        logger.error(f"Failed to calculate AB magnitude: {e}")
        return None, None


def calculate_ab_magnitude(
    flux_mjy_sr: float,
    flux_error_mjy_sr: float,
    wavelength_micron: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    DEPRECATED: Calculate AB magnitude from flux measurement in MJy/sr.
    
    This function is deprecated because it incorrectly handles units.
    Use calculate_ab_magnitude_from_jy() after proper unit conversion instead.
    
    Parameters
    ----------
    flux_mjy_sr : float
        Flux in MJy/sr
    flux_error_mjy_sr : float
        Flux error in MJy/sr
    wavelength_micron : float
        Central wavelength in microns
        
    Returns
    -------
    mag_ab : float or None
        AB magnitude (INCORRECT - do not use)
    mag_ab_error : float or None  
        AB magnitude error (INCORRECT - do not use)
    """
    logger.warning(
        "calculate_ab_magnitude() is deprecated and produces incorrect results. "
        "Use calculate_ab_magnitude_from_jy() after converting MJy/sr to Jy."
    )
    
    try:
        # Calculate AB magnitude using deprecated function
        mag_ab, mag_ab_error = flux_to_ab_magnitude(
            flux_mjy_sr, flux_error_mjy_sr, wavelength_micron
        )
        
        # Return None for NaN values
        mag_ab = None if np.isnan(mag_ab) else mag_ab
        mag_ab_error = None if np.isnan(mag_ab_error) else mag_ab_error
        
        logger.debug(
            f"AB magnitude (DEPRECATED) for {flux_mjy_sr:.3f} MJy/sr at {wavelength_micron:.3f} μm: "
            f"{mag_ab:.3f}±{mag_ab_error:.3f}"
        )
        
        return mag_ab, mag_ab_error
        
    except Exception as e:
        logger.error(f"Failed to calculate AB magnitude: {e}")
        return None, None


def magnitude_to_flux_ab(
    mag_ab: float,
    mag_ab_error: float,
    wavelength_micron: float
) -> Tuple[float, float]:
    """
    Convert AB magnitude back to flux density using astropy units.
    
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
    flux_mjy_sr : float
        Flux in MJy/sr
    flux_error_mjy_sr : float
        Flux error in MJy/sr
    """
    # Convert magnitude to flux density (erg/s/cm²/Hz)
    flux_density_cgs_value = 10**(-0.4 * (mag_ab + 48.6))
    
    # Error propagation: df/dm = -0.4 * ln(10) * f
    flux_density_error_cgs_value = 0.4 * np.log(10) * flux_density_cgs_value * mag_ab_error
    
    # Create astropy quantities with proper units
    flux_density_cgs = flux_density_cgs_value * u.erg / u.s / u.cm**2 / u.Hz
    flux_density_error_cgs = flux_density_error_cgs_value * u.erg / u.s / u.cm**2 / u.Hz
    
    # Convert to MJy/sr (assuming point source, so adding steradian unit back)
    flux_mjy_sr_quantity = flux_density_cgs.to(u.MJy / u.sr)
    flux_error_mjy_sr_quantity = flux_density_error_cgs.to(u.MJy / u.sr)
    
    return float(flux_mjy_sr_quantity.value), float(flux_error_mjy_sr_quantity.value)