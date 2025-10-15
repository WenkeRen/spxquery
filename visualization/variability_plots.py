"""
Plotting functions for quasar variability analysis QA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, Optional, Tuple
import os


def plot_variability_qa(results: Dict,
                       source_name: str = "Source",
                       save_path: Optional[str] = None,
                       figsize: Tuple[float, float] = (16, 14),
                       dpi: int = 150) -> plt.Figure:
    """
    Generate comprehensive QA plots for variability analysis.
    
    Parameters
    ----------
    results : dict
        Results from process_lightcurve_variability function
    source_name : str, optional
        Source name for plot title (default: "Source")
    save_path : str, optional
        Path to save the figure. If None, only displays
    figsize : tuple, optional
        Figure size in inches (default: (16, 14))
    dpi : int, optional
        Resolution for saved figure (default: 150)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    sed_model = results['sed_model']
    corrected_lc = results['corrected_lightcurve']
    daily_stats = results['daily_stats']
    outliers = results['outliers']
    
    # Store current rcParams and temporarily disable constrained_layout
    old_constrained = plt.rcParams.get('figure.constrained_layout.use', False)
    plt.rcParams['figure.constrained_layout.use'] = False
    
    try:
        # Create figure with manual layout control
        fig = plt.figure(figsize=figsize)
        
        # Create subplot grid manually to have full control
        # Top panel for SED (full width)
        ax1 = plt.subplot2grid((6, 2), (0, 0), rowspan=2, colspan=2)
        plot_sed(ax1, sed_model, corrected_lc, source_name)
        
        # Middle panel for raw lightcurve (full width)
        ax2 = plt.subplot2grid((6, 2), (2, 0), rowspan=2, colspan=2)
        plot_raw_lightcurve(ax2, corrected_lc)
        
        # Bottom left for corrected lightcurve
        ax3 = plt.subplot2grid((6, 2), (4, 0), rowspan=2, colspan=1)
        plot_corrected_lightcurve(ax3, corrected_lc, daily_stats, outliers, 
                                 sed_model['target_wavelength'])
        
        # Bottom right for statistics
        ax4 = plt.subplot2grid((6, 2), (4, 1), rowspan=2, colspan=1)
        plot_variability_stats(ax4, daily_stats, corrected_lc)
        
        # Add title with explicit position
        fig.suptitle(f"Variability Analysis QA: {source_name}", fontsize=16, y=0.98)
        
        # Use tight_layout with padding
        plt.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=3.0, w_pad=3.0)
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.3)
            print(f"QA plot saved to: {save_path}")
        
        return fig
        
    finally:
        # Restore original rcParams
        plt.rcParams['figure.constrained_layout.use'] = old_constrained


def plot_sed(ax: plt.Axes, sed_model: Dict, data: pd.DataFrame, source_name: str):
    """Plot the empirical SED with observed data points."""
    wave_grid = sed_model['wave_grid']
    sed_mag = sed_model['sed_mag']
    
    # Plot observed data with transparency
    bands = data['band'].unique()
    colors = cm.rainbow(np.linspace(0, 1, len(bands)))
    
    for band, color in zip(bands, colors):
        band_data = data[data['band'] == band]
        ax.errorbar(band_data['wavelength'], band_data['mag_ab'], 
                   yerr=band_data['mag_ab_error'],
                   fmt='o', color=color, alpha=0.3, markersize=4,
                   label=f'{band} (observed)', zorder=1)
    
    # Plot SED model
    valid_mask = ~sed_mag.mask if hasattr(sed_mag, 'mask') else ~np.isnan(sed_mag)
    ax.plot(wave_grid[valid_mask], sed_mag[valid_mask], 
           'k-', linewidth=2, label='Empirical SED', zorder=2)
    
    # Mark target wavelength
    target_wave = sed_model['target_wavelength']
    ax.axvline(target_wave, color='red', linestyle='--', alpha=0.7,
              label=f'Target λ = {target_wave:.3f} μm')
    
    ax.set_xlabel('Wavelength (μm)', fontsize=11)
    ax.set_ylabel('AB Magnitude', fontsize=11)
    ax.set_title(f'Empirical SED Model', fontsize=13, pad=10)
    ax.invert_yaxis()
    ax.legend(loc='best', ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_raw_lightcurve(ax: plt.Axes, data: pd.DataFrame):
    """Plot raw lightcurve colored by wavelength."""
    # Create colormap based on wavelength
    wavelengths = data['wavelength'].values
    norm = plt.Normalize(vmin=wavelengths.min(), vmax=wavelengths.max())
    cmap = cm.viridis
    
    # Plot each point colored by wavelength
    scatter = ax.scatter(data['mjd'], data['mag_ab'], 
                        c=data['wavelength'], s=20, 
                        cmap=cmap, norm=norm, alpha=0.6)
    
    # Add errorbars
    ax.errorbar(data['mjd'], data['mag_ab'], 
               yerr=data['mag_ab_error'],
               fmt='none', ecolor='gray', alpha=0.3, zorder=0)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Wavelength (μm)', fontsize=10)
    
    ax.set_xlabel('MJD', fontsize=11)
    ax.set_ylabel('AB Magnitude', fontsize=11)
    ax.set_title('Raw Multi-wavelength Lightcurve', fontsize=13, pad=10)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)


def plot_corrected_lightcurve(ax: plt.Axes, data: pd.DataFrame, 
                             daily_stats: pd.DataFrame,
                             outliers: pd.DataFrame,
                             target_wavelength: float):
    """Plot corrected lightcurve with daily medians and outliers."""
    # Plot all corrected points
    ax.errorbar(data['mjd'], data['mag_corrected'], 
               yerr=data['mag_corrected_error'],
               fmt='o', color='gray', alpha=0.3, markersize=3,
               label='Individual obs.')
    
    # Plot daily medians with error bars
    ax.errorbar(daily_stats['mjd_mean'], daily_stats['mag_median'],
               yerr=daily_stats['mag_std'],
               fmt='o', color='blue', markersize=7,
               label='Daily median', zorder=3)
    
    # Highlight outliers if any
    if len(outliers) > 0:
        ax.scatter(outliers['mjd'], outliers['mag_corrected'],
                  color='red', s=40, marker='x', 
                  label='3σ outliers', zorder=4)
    
    ax.set_xlabel('MJD', fontsize=10)
    ax.set_ylabel('AB Magnitude', fontsize=10)
    ax.set_title(f'Wavelength-Corrected Lightcurve\n(λ = {target_wavelength:.3f} μm)', 
                fontsize=12, pad=8)
    ax.invert_yaxis()
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_variability_stats(ax: plt.Axes, daily_stats: pd.DataFrame, 
                          data: pd.DataFrame):
    """Plot variability statistics and histogram."""
    # Set background color
    ax.set_facecolor('#f8f8f8')
    
    # Calculate statistics
    all_mags = data['mag_corrected'].values
    daily_mags = daily_stats['mag_median'].values
    
    # Calculate outlier fraction correctly
    outlier_count = daily_stats['n_outliers'].sum()
    outlier_frac = 100 * outlier_count / len(data) if len(data) > 0 else 0
    
    # Create two subplots within this axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Statistics text
    stats_text = f"""Variability Statistics:

All Observations:
  Mean magnitude: {np.mean(all_mags):.3f}
  Std deviation: {np.std(all_mags):.3f}
  Range: {np.ptp(all_mags):.3f} mag
  N observations: {len(all_mags)}

Daily Medians:
  Mean magnitude: {np.mean(daily_mags):.3f}
  Std deviation: {np.std(daily_mags):.3f}
  Range: {np.ptp(daily_mags):.3f} mag
  N epochs: {len(daily_mags)}
  
Outlier fraction: {outlier_frac:.1f}%"""
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                    edgecolor='gray', alpha=0.8))
    
    # Add histogram at bottom
    ax_hist = ax.inset_axes([0.1, 0.05, 0.8, 0.35])
    n, bins, patches = ax_hist.hist(daily_mags, bins=12, alpha=0.7, 
                                    color='steelblue', edgecolor='black')
    ax_hist.set_xlabel('Daily Median Magnitude', fontsize=9)
    ax_hist.set_ylabel('Count', fontsize=9)
    ax_hist.tick_params(labelsize=8)
    ax_hist.grid(True, alpha=0.3, axis='y')
    
    # Add mean line
    mean_mag = np.mean(daily_mags)
    ax_hist.axvline(mean_mag, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Mean: {mean_mag:.3f}')
    ax_hist.legend(fontsize=8)