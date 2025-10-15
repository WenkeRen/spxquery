"""
Visualization functions for SPHEREx time-domain data.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from astropy.stats import sigma_clip

from ..core.config import PhotometryResult

logger = logging.getLogger(__name__)

# Plotting configuration
WAVELENGTH_CMAP = "rainbow"  # Colormap for wavelength coding
WAVELENGTH_RANGE = (0.75, 5.0)  # SPHEREx wavelength range in microns


def apply_sigma_clipping(
    photometry_results: List[PhotometryResult], sigma: float = 3.0, maxiters: int = 10
) -> List[PhotometryResult]:
    """
    Apply sigma clipping to remove outliers based on flux values.

    Parameters
    ----------
    photometry_results : List[PhotometryResult]
        Input photometry measurements
    sigma : float
        Number of standard deviations to use for clipping
    maxiters : int
        Maximum number of clipping iterations

    Returns
    -------
    List[PhotometryResult]
        Filtered photometry results with outliers removed
    """
    if not photometry_results:
        return photometry_results

    # Only clip regular measurements (not upper limits)
    regular_measurements = [p for p in photometry_results if not p.is_upper_limit]
    upper_limits = [p for p in photometry_results if p.is_upper_limit]

    if not regular_measurements:
        return photometry_results

    # Extract flux values for clipping
    fluxes = np.array([p.flux for p in regular_measurements])

    # Apply sigma clipping
    clipped_data = sigma_clip(fluxes, sigma=sigma, maxiters=maxiters)

    # Keep only non-clipped measurements
    if ma.is_masked(clipped_data):
        good_indices = ~clipped_data.mask
    else:
        # If no points were clipped, all points are good
        good_indices = np.ones(len(fluxes), dtype=bool)

    # Filter regular measurements
    filtered_regular = [regular_measurements[i] for i in range(len(regular_measurements)) if good_indices[i]]

    # Combine filtered regular measurements with upper limits
    filtered_results = filtered_regular + upper_limits

    logger.info(
        f"Sigma clipping: {len(photometry_results)} -> {len(filtered_results)} measurements "
        f"({len(photometry_results) - len(filtered_results)} outliers removed)"
    )

    return filtered_results


def create_spectrum_plot(
    photometry_results: List[PhotometryResult],
    ax: Optional[Axes] = None,
    apply_clipping: bool = True,
    sigma: float = 3.0,
) -> Axes:
    """
    Create spectrum plot (wavelength vs flux).

    Parameters
    ----------
    photometry_results : List[PhotometryResult]
        Photometry measurements
    ax : plt.Axes, optional
        Axes to plot on. If None, current axes are used.
    apply_clipping : bool
        Whether to apply sigma clipping to remove outliers
    sigma : float
        Number of standard deviations for sigma clipping

    Returns
    -------
    plt.Axes
        Axes with spectrum plot
    """
    if ax is None:
        ax = plt.gca()

    # Apply sigma clipping if requested
    if apply_clipping:
        photometry_results = apply_sigma_clipping(photometry_results, sigma=sigma)

    # Separate regular measurements and upper limits
    regular = [p for p in photometry_results if not p.is_upper_limit]
    upper_limits = [p for p in photometry_results if p.is_upper_limit]

    # Plot regular measurements with error bars
    if regular:
        wavelengths = [p.wavelength for p in regular]
        fluxes = [p.flux for p in regular]
        flux_errors = [p.flux_error for p in regular]
        bandwidths = [p.bandwidth for p in regular]

        ax.errorbar(
            wavelengths,
            fluxes,
            xerr=bandwidths,
            yerr=flux_errors,
            fmt="o",
            markersize=6,
            capsize=3,
            label="Measurements",
            alpha=0.8,
        )

    # Plot upper limits
    if upper_limits:
        ul_wavelengths = [p.wavelength for p in upper_limits]
        ul_fluxes = [p.flux + p.flux_error for p in upper_limits]  # Upper limit value
        ul_bandwidths = [p.bandwidth for p in upper_limits]

        ax.errorbar(
            ul_wavelengths,
            ul_fluxes,
            xerr=ul_bandwidths,
            yerr=None,
            fmt="v",
            markersize=8,
            capsize=0,
            label="Upper limits",
            alpha=0.8,
            color="red",
        )

    # Formatting
    ax.set_xlabel("Wavelength (μm)", fontsize=12)
    ax.set_ylabel("Flux (MJy/sr)", fontsize=12)
    ax.set_title("SPHEREx Spectrum", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set x-axis limits to SPHEREx range
    ax.set_xlim(0.7, 5.1)

    return ax


def create_lightcurve_plot(
    photometry_results: List[PhotometryResult],
    ax: Optional[Axes] = None,
    apply_clipping: bool = True,
    sigma: float = 3.0,
) -> Axes:
    """
    Create light curve plot (time vs flux) color-coded by wavelength.

    Parameters
    ----------
    photometry_results : List[PhotometryResult]
        Photometry measurements
    ax : plt.Axes, optional
        Axes to plot on. If None, current axes are used.
    apply_clipping : bool
        Whether to apply sigma clipping to remove outliers
    sigma : float
        Number of standard deviations for sigma clipping

    Returns
    -------
    plt.Axes
        Axes with light curve plot
    """
    if ax is None:
        ax = plt.gca()

    if not photometry_results:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return ax

    # Apply sigma clipping if requested
    if apply_clipping:
        photometry_results = apply_sigma_clipping(photometry_results, sigma=sigma)

    # Get colormap for wavelength coding
    cmap = cm.get_cmap(WAVELENGTH_CMAP)
    norm = Normalize(vmin=WAVELENGTH_RANGE[0], vmax=WAVELENGTH_RANGE[1])

    # Sort by MJD for proper time ordering
    sorted_results = sorted(photometry_results, key=lambda x: x.mjd)

    # Plot each point with color based on wavelength
    for result in sorted_results:
        color = cmap(norm(result.wavelength))

        if result.is_upper_limit:
            # Plot upper limit
            ax.errorbar(
                result.mjd, result.flux + result.flux_error, yerr=None, fmt="v", color=color, markersize=8, alpha=0.8
            )
        else:
            # Plot regular measurement
            ax.errorbar(
                result.mjd,
                result.flux,
                yerr=result.flux_error,
                fmt="o",
                color=color,
                markersize=6,
                capsize=3,
                alpha=0.8,
            )

    # Add colorbar for wavelength
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Wavelength (μm)", fontsize=10)

    # Formatting
    ax.set_xlabel("MJD", fontsize=12)
    ax.set_ylabel("Flux (MJy/sr)", fontsize=12)
    ax.set_title("SPHEREx Light Curve", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add some padding to x-axis
    mjds = [p.mjd for p in sorted_results]
    mjd_range = max(mjds) - min(mjds)
    if mjd_range > 0:
        ax.set_xlim(min(mjds) - 0.05 * mjd_range, max(mjds) + 0.05 * mjd_range)

    return ax


def create_combined_plot(
    photometry_results: List[PhotometryResult],
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 8),
    apply_clipping: bool = True,
    sigma: float = 3.0,
) -> Figure:
    """
    Create combined plot with spectrum and light curve.

    Parameters
    ----------
    photometry_results : List[PhotometryResult]
        Photometry measurements
    output_path : Path, optional
        Path to save figure. If None, figure is not saved.
    figsize : Tuple[float, float]
        Figure size in inches
    apply_clipping : bool
        Whether to apply sigma clipping to remove outliers
    sigma : float
        Number of standard deviations for sigma clipping

    Returns
    -------
    Figure
        Matplotlib figure with both plots
    """
    # Create figure with two subplots using constrained layout to handle colorbars
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 1]}, constrained_layout=True
    )

    # Create spectrum plot (top)
    create_spectrum_plot(photometry_results, ax1, apply_clipping=apply_clipping, sigma=sigma)

    # Create light curve plot (bottom)
    create_lightcurve_plot(photometry_results, ax2, apply_clipping=apply_clipping, sigma=sigma)

    # Save if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved combined plot to {output_path}")

    return fig


def plot_summary_statistics(photometry_results: List[PhotometryResult], output_path: Optional[Path] = None) -> Figure:
    """
    Create summary statistics plots.

    Parameters
    ----------
    photometry_results : List[PhotometryResult]
        Photometry measurements
    output_path : Path, optional
        Path to save figure

    Returns
    -------
    Figure
        Figure with summary plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Histogram of wavelengths
    ax = axes[0, 0]
    wavelengths = [p.wavelength for p in photometry_results]
    ax.hist(wavelengths, bins=20, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("Count")
    ax.set_title("Wavelength Distribution")
    ax.grid(True, alpha=0.3)

    # 2. Band distribution
    ax = axes[0, 1]
    bands = [p.band for p in photometry_results]
    unique_bands, counts = np.unique(bands, return_counts=True)
    ax.bar(unique_bands, counts, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Band")
    ax.set_ylabel("Count")
    ax.set_title("Observations per Band")
    ax.grid(True, alpha=0.3, axis="y")

    # 3. SNR distribution
    ax = axes[1, 0]
    snrs = [p.flux / p.flux_error for p in photometry_results if p.flux_error > 0]
    ax.hist(snrs, bins=20, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Signal-to-Noise Ratio")
    ax.set_ylabel("Count")
    ax.set_title("SNR Distribution")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, np.percentile(snrs, 95) if snrs else 10)

    # 4. Time coverage
    ax = axes[1, 1]
    bands_unique = sorted(set(bands))
    band_colors = cm.get_cmap("rainbow")(np.linspace(0, 1, len(bands_unique)))

    for band, color in zip(bands_unique, band_colors):
        band_mjds = [p.mjd for p in photometry_results if p.band == band]
        ax.scatter([band] * len(band_mjds), band_mjds, alpha=0.6, s=20, color=color)

    ax.set_xlabel("Band")
    ax.set_ylabel("MJD")
    ax.set_title("Temporal Coverage by Band")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved summary statistics to {output_path}")

    return fig
