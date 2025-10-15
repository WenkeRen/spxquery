"""
Module for quasar variability analysis using SPHEREx multi-epoch spectral data.

This module provides functions to:
1. Build empirical SED models from multi-epoch observations
2. Estimate magnitudes at specific wavelengths using SED interpolation
3. Generate corrected lightcurves accounting for wavelength differences
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from astropy.stats import sigma_clip
import warnings
from typing import Dict, Tuple, Optional, Union, List


def generate_empirical_sed(wavelengths: np.ndarray, 
                          magnitudes: np.ndarray, 
                          mag_errors: np.ndarray,
                          bandwidths: np.ndarray,
                          n_bins: int = 100,
                          smooth_scale: float = 0.1) -> Tuple[np.ndarray, np.ndarray, callable]:
    """
    Generate an empirical SED model from multi-epoch observations.
    
    Parameters
    ----------
    wavelengths : np.ndarray
        Central wavelengths of observations in microns
    magnitudes : np.ndarray
        AB magnitudes
    mag_errors : np.ndarray
        Magnitude uncertainties
    bandwidths : np.ndarray
        Bandwidths in microns
    n_bins : int, optional
        Number of wavelength bins for oversampling (default: 100)
    smooth_scale : float, optional
        Smoothing scale for interpolation in microns (default: 0.1)
        
    Returns
    -------
    wave_grid : np.ndarray
        Wavelength grid for the SED model
    sed_mag : np.ndarray
        SED magnitudes on the grid (masked where no data)
    sed_func : callable
        Interpolation function for the SED
    """
    # Sort data by wavelength
    sort_idx = np.argsort(wavelengths)
    wave_sorted = wavelengths[sort_idx]
    mag_sorted = magnitudes[sort_idx]
    err_sorted = mag_errors[sort_idx]
    bw_sorted = bandwidths[sort_idx]
    
    # Create oversampled wavelength grid
    wave_min = wave_sorted.min() - bw_sorted[0]
    wave_max = wave_sorted.max() + bw_sorted[-1]
    wave_grid = np.linspace(wave_min, wave_max, n_bins)
    
    # Initialize arrays for binned SED
    sed_mag = np.full(n_bins, np.nan)
    sed_err = np.full(n_bins, np.nan)
    sed_count = np.zeros(n_bins)
    
    # Bin the data
    for i in range(len(wave_sorted)):
        # Find bins covered by this observation
        wave_lo = wave_sorted[i] - bw_sorted[i]/2
        wave_hi = wave_sorted[i] + bw_sorted[i]/2
        bin_mask = (wave_grid >= wave_lo) & (wave_grid <= wave_hi)
        
        # Weight by inverse variance
        weight = 1.0 / err_sorted[i]**2
        
        # Add to bins
        for j in np.where(bin_mask)[0]:
            if np.isnan(sed_mag[j]):
                sed_mag[j] = mag_sorted[i] * weight
                sed_err[j] = weight
                sed_count[j] = 1
            else:
                sed_mag[j] += mag_sorted[i] * weight
                sed_err[j] += weight
                sed_count[j] += 1
    
    # Compute weighted average
    valid_mask = ~np.isnan(sed_mag)
    sed_mag[valid_mask] /= sed_err[valid_mask]
    sed_err[valid_mask] = 1.0 / np.sqrt(sed_err[valid_mask])
    
    # Interpolate to create smooth SED
    valid_waves = wave_grid[valid_mask]
    valid_mags = sed_mag[valid_mask]
    
    if len(valid_waves) < 4:
        raise ValueError("Not enough valid wavelength bins for interpolation")
    
    # Use cubic spline interpolation with smoothing
    sed_func = interpolate.UnivariateSpline(valid_waves, valid_mags, 
                                           s=smooth_scale*len(valid_waves), k=3)
    
    # Evaluate on full grid
    sed_smooth = sed_func(wave_grid)
    
    # Mask regions outside data coverage
    mask = np.zeros_like(wave_grid, dtype=bool)
    for i in range(len(wave_grid)):
        if wave_grid[i] < valid_waves.min() or wave_grid[i] > valid_waves.max():
            sed_smooth[i] = np.nan
            mask[i] = True
    
    return wave_grid, np.ma.array(sed_smooth, mask=mask), sed_func


def estimate_magnitude_at_wavelength(obs_wavelength: Union[float, np.ndarray],
                                   obs_magnitude: Union[float, np.ndarray],
                                   obs_error: Union[float, np.ndarray],
                                   target_wavelength: float,
                                   sed_function: callable) -> Tuple[float, float]:
    """
    Estimate magnitude at a specific wavelength using SED model.
    
    Parameters
    ----------
    obs_wavelength : float or array
        Observed wavelength(s) in microns
    obs_magnitude : float or array
        Observed magnitude(s)
    obs_error : float or array
        Magnitude error(s)
    target_wavelength : float
        Target wavelength for estimation in microns
    sed_function : callable
        SED interpolation function
        
    Returns
    -------
    est_magnitude : float
        Estimated magnitude at target wavelength
    est_error : float
        Estimated error (scaled from input error)
    """
    obs_wavelength = np.atleast_1d(obs_wavelength)
    obs_magnitude = np.atleast_1d(obs_magnitude)
    obs_error = np.atleast_1d(obs_error)
    
    # Get SED values at observed and target wavelengths
    sed_obs = sed_function(obs_wavelength)
    sed_target = sed_function(target_wavelength)
    
    # Calculate magnitude offset from SED
    mag_offset = obs_magnitude - sed_obs
    
    # Apply offset to target wavelength
    est_magnitude = sed_target + np.mean(mag_offset)
    
    # Scale error based on flux ratio (magnitude difference)
    # For small magnitude differences: Δf/f ≈ 0.4 * Δm
    flux_ratio = 10**(-0.4 * (sed_target - np.mean(sed_obs)))
    est_error = np.sqrt(np.mean(obs_error**2)) * flux_ratio
    
    return float(est_magnitude), float(est_error)


def process_lightcurve_variability(data: pd.DataFrame,
                                 target_wavelength: Optional[float] = None,
                                 min_points: int = 50) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Process lightcurve data to analyze variability.
    
    Parameters
    ----------
    data : pd.DataFrame
        Lightcurve data with columns: wavelength, mag_ab, mag_ab_error, mjd, band
    target_wavelength : float, optional
        Target wavelength for correction. If None, uses median wavelength
    min_points : int, optional
        Minimum number of points required for processing (default: 50)
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'sed_model': dict with wave_grid, sed_mag, sed_func
        - 'corrected_lightcurve': DataFrame with corrected magnitudes
        - 'daily_stats': DataFrame with daily median magnitudes
        - 'outliers': DataFrame with identified outliers
    """
    if len(data) < min_points:
        raise ValueError(f"Insufficient data points ({len(data)} < {min_points})")
    
    # Generate SED model
    wave_grid, sed_mag, sed_func = generate_empirical_sed(
        data['wavelength'].values,
        data['mag_ab'].values,
        data['mag_ab_error'].values,
        data['bandwidth'].values
    )
    
    # Set target wavelength if not specified
    if target_wavelength is None:
        target_wavelength = np.median(data['wavelength'])
    
    # Estimate magnitudes at target wavelength
    corrected_mags = []
    corrected_errors = []
    
    for idx, row in data.iterrows():
        est_mag, est_err = estimate_magnitude_at_wavelength(
            row['wavelength'],
            row['mag_ab'],
            row['mag_ab_error'],
            target_wavelength,
            sed_func
        )
        corrected_mags.append(est_mag)
        corrected_errors.append(est_err)
    
    # Create corrected lightcurve
    corrected_lc = data.copy()
    corrected_lc['mag_corrected'] = corrected_mags
    corrected_lc['mag_corrected_error'] = corrected_errors
    corrected_lc['target_wavelength'] = target_wavelength
    
    # Calculate daily statistics
    corrected_lc['mjd_day'] = np.floor(corrected_lc['mjd'])
    daily_groups = corrected_lc.groupby('mjd_day')
    
    daily_stats = []
    outlier_list = []
    
    for day, group in daily_groups:
        if len(group) > 1:
            # Use sigma clipping to identify outliers
            clipped = sigma_clip(group['mag_corrected'].values, sigma=3, maxiters=5)
            
            # Store outliers
            outlier_mask = clipped.mask
            if np.any(outlier_mask):
                outliers = group[outlier_mask].copy()
                outliers['outlier_reason'] = '3-sigma clipping'
                outlier_list.append(outliers)
            
            # Calculate statistics on good data
            good_mags = group['mag_corrected'].values[~outlier_mask]
            if len(good_mags) > 0:
                daily_stats.append({
                    'mjd_day': day,
                    'mjd_mean': group['mjd'].mean(),
                    'mag_median': np.median(good_mags),
                    'mag_std': np.std(good_mags),
                    'n_obs': len(good_mags),
                    'n_outliers': np.sum(outlier_mask)
                })
        else:
            # Single observation per day
            daily_stats.append({
                'mjd_day': day,
                'mjd_mean': group['mjd'].iloc[0],
                'mag_median': group['mag_corrected'].iloc[0],
                'mag_std': group['mag_corrected_error'].iloc[0],
                'n_obs': 1,
                'n_outliers': 0
            })
    
    daily_df = pd.DataFrame(daily_stats)
    outlier_df = pd.concat(outlier_list) if outlier_list else pd.DataFrame()
    
    return {
        'sed_model': {
            'wave_grid': wave_grid,
            'sed_mag': sed_mag,
            'sed_func': sed_func,
            'target_wavelength': target_wavelength
        },
        'corrected_lightcurve': corrected_lc,
        'daily_stats': daily_df,
        'outliers': outlier_df
    }