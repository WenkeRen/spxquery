"""
Visualization functions for SPHEREx time-domain data.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from astropy.stats import sigma_clip
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from ..core.config import PhotometryResult

if TYPE_CHECKING:
    from ..core.config import VisualizationConfig

logger = logging.getLogger(__name__)

# Plotting configuration
WAVELENGTH_CMAP = "rainbow"  # Colormap for wavelength coding
WAVELENGTH_RANGE = (0.75, 5.0)  # SPHEREx wavelength range in microns


def calculate_smart_ylimits(
    y_values: List[float],
    percentile_range: Tuple[float, float] = (1.0, 99.0),
    padding_fraction: float = 0.1,
) -> Tuple[float, float]:
    """
    Calculate smart y-axis limits based on percentiles to exclude extreme outliers.

    Parameters
    ----------
    y_values : List[float]
        Y-axis values to analyze
    percentile_range : Tuple[float, float]
        Lower and upper percentiles to use for limits (default: 1st to 99th)
    padding_fraction : float
        Fraction of range to add as padding (default: 0.1 = 10%)

    Returns
    -------
    Tuple[float, float]
        (ymin, ymax) limits for y-axis
    """
    # Filter out NaN and infinite values
    valid_values = [v for v in y_values if np.isfinite(v)]

    if not valid_values:
        return (0, 1)  # Default range if no valid data

    # Calculate percentile-based limits
    y_min = np.percentile(valid_values, percentile_range[0])
    y_max = np.percentile(valid_values, percentile_range[1])

    # Add padding
    y_range = y_max - y_min
    if y_range > 0:
        padding = y_range * padding_fraction
        y_min -= padding
        y_max += padding
    else:
        # All values are the same, add symmetric padding
        if y_min != 0:
            y_min *= 0.9
            y_max *= 1.1
        else:
            y_min = -0.1
            y_max = 0.1

    return (y_min, y_max)


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
    apply_quality_filters: bool = True,
    sigma_threshold: float = 5.0,
    bad_flags_mask: Optional[int] = None,
    use_magnitude: bool = False,
    show_errorbars: bool = True,
) -> Axes:
    """
    Create spectrum plot (wavelength vs flux), color-coded by observation date.

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
    apply_quality_filters : bool
        Whether to classify points as good/rejected based on QC filters
    sigma_threshold : float
        Minimum SNR (flux/flux_err) for quality control
    bad_flags_mask : int, optional
        Integer mask with bad flag bits set (created by create_flag_mask)
    use_magnitude : bool
        If True, plot AB magnitude instead of flux (default: False)
    show_errorbars : bool
        If True, show errorbars (default: True)

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

    # Classify points by quality
    if apply_quality_filters and bad_flags_mask is not None:
        from ..utils.helpers import classify_photometry_by_quality

        classified = classify_photometry_by_quality(
            photometry_results,
            sigma_threshold=sigma_threshold,
            bad_flags_mask=bad_flags_mask,
            separate_upper_limits=True,
        )
        good_regular = classified.good_regular
        rejected_regular = classified.rejected_regular
        good_upper_limits = classified.good_upper_limits
        rejected_upper_limits = classified.rejected_upper_limits
    else:
        # No quality filtering - all points are "good"
        good_regular = [p for p in photometry_results if not p.is_upper_limit]
        good_upper_limits = [p for p in photometry_results if p.is_upper_limit]
        rejected_regular = []
        rejected_upper_limits = []

    # Plot good regular measurements with error bars, color-coded by date
    if good_regular:
        wavelengths = [p.wavelength for p in good_regular]
        bandwidths = [p.bandwidth for p in good_regular]
        mjds = [p.mjd for p in good_regular]

        # Get y values depending on magnitude or flux mode
        if use_magnitude:
            y_values = [p.mag_ab if p.mag_ab is not None else np.nan for p in good_regular]
            y_errors = [p.mag_ab_error if p.mag_ab_error is not None else np.nan for p in good_regular]
        else:
            y_values = [p.flux for p in good_regular]
            y_errors = [p.flux_error for p in good_regular]

        # Convert MJD to days since first observation
        mjd_min = min(mjds)
        days_since_first = [mjd - mjd_min for mjd in mjds]

        # Create colormap for date coding
        cmap = cm.get_cmap("viridis")
        norm = Normalize(vmin=0, vmax=max(days_since_first) if days_since_first else 1)

        # Two-pass plotting: errorbars first (transparent), then markers (solid)

        # Pass 1: Plot errorbars only (if enabled)
        if show_errorbars:
            for wl, y_val, y_err, bw, days in zip(wavelengths, y_values, y_errors, bandwidths, days_since_first):
                if np.isnan(y_val):
                    continue
                color = cmap(norm(days))
                ax.errorbar(
                    wl,
                    y_val,
                    xerr=bw,
                    yerr=y_err,
                    fmt="none",  # No marker
                    capsize=0,
                    linewidth=0.5,
                    elinewidth=0.5,
                    color=color,
                    alpha=0.2,  # Transparent errorbars
                    zorder=1,  # Behind markers
                )

        # Pass 2: Plot markers only (solid)
        for wl, y_val, days in zip(wavelengths, y_values, days_since_first):
            if np.isnan(y_val):
                continue
            color = cmap(norm(days))
            ax.plot(
                wl,
                y_val,
                "o",
                color=color,
                markersize=1.5,
                alpha=0.9,  # Solid markers
                zorder=2,  # On top of errorbars
            )

        # Add colorbar for date
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Days since first obs", fontsize=10)

    # Plot rejected regular measurements as crosses
    if rejected_regular:
        wavelengths = [p.wavelength for p in rejected_regular]
        bandwidths = [p.bandwidth for p in rejected_regular]

        if use_magnitude:
            y_values = [p.mag_ab if p.mag_ab is not None else np.nan for p in rejected_regular]
        else:
            y_values = [p.flux for p in rejected_regular]

        # Filter out NaN values
        valid_data = [(wl, y, bw) for wl, y, bw in zip(wavelengths, y_values, bandwidths) if not np.isnan(y)]
        if valid_data:
            wavelengths, y_values, bandwidths = zip(*valid_data)
            ax.errorbar(
                wavelengths,
                y_values,
                xerr=bandwidths,
                yerr=None,
                fmt="x",
                markersize=2,
                capsize=0,
                linewidth=0.5,
                elinewidth=0.5,
                label="Rejected",
                alpha=0.5,
                color="gray",
            )

    # Plot good upper limits
    if good_upper_limits:
        ul_wavelengths = [p.wavelength for p in good_upper_limits]
        ul_bandwidths = [p.bandwidth for p in good_upper_limits]

        if use_magnitude:
            # For magnitude, upper limit on flux becomes lower limit on magnitude
            # Use the stored mag_ab value (should represent the limit)
            ul_y_values = [p.mag_ab if p.mag_ab is not None else np.nan for p in good_upper_limits]
        else:
            ul_y_values = [p.flux + p.flux_error for p in good_upper_limits]  # Upper limit value

        # Filter out NaN values
        valid_data = [(wl, y, bw) for wl, y, bw in zip(ul_wavelengths, ul_y_values, ul_bandwidths) if not np.isnan(y)]
        if valid_data:
            ul_wavelengths, ul_y_values, ul_bandwidths = zip(*valid_data)
            ax.errorbar(
                ul_wavelengths,
                ul_y_values,
                xerr=ul_bandwidths,
                yerr=None,
                fmt="v" if not use_magnitude else "^",  # Flip arrow for magnitude
                markersize=3,
                capsize=0,
                linewidth=0.5,
                elinewidth=0.5,
                label="Upper limits" if not use_magnitude else "Lower limits (mag)",
                alpha=0.8,
                color="red",
            )

    # Plot rejected upper limits as small crosses
    if rejected_upper_limits:
        ul_wavelengths = [p.wavelength for p in rejected_upper_limits]
        ul_bandwidths = [p.bandwidth for p in rejected_upper_limits]

        if use_magnitude:
            ul_y_values = [p.mag_ab if p.mag_ab is not None else np.nan for p in rejected_upper_limits]
        else:
            ul_y_values = [p.flux + p.flux_error for p in rejected_upper_limits]

        # Filter out NaN values
        valid_data = [(wl, y, bw) for wl, y, bw in zip(ul_wavelengths, ul_y_values, ul_bandwidths) if not np.isnan(y)]
        if valid_data:
            ul_wavelengths, ul_y_values, ul_bandwidths = zip(*valid_data)
            ax.errorbar(
                ul_wavelengths,
                ul_y_values,
                xerr=ul_bandwidths,
                yerr=None,
                fmt="x",
                markersize=2,
                capsize=0,
                linewidth=0.5,
                elinewidth=0.5,
                label="Rejected (UL)",
                alpha=0.5,
                color="lightcoral",
            )

    # Formatting
    ax.set_xlabel("Wavelength (μm)", fontsize=12)
    if use_magnitude:
        ax.set_ylabel("AB Magnitude", fontsize=12)
        ax.invert_yaxis()  # Fainter sources have higher magnitudes
    else:
        ax.set_ylabel("Flux Density (μJy)", fontsize=12)
    ax.set_title("SPHEREx Spectrum", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set x-axis limits to SPHEREx range
    ax.set_xlim(0.7, 5.1)

    # Set smart y-axis limits based on percentiles to handle outliers
    all_y_values = []

    # Collect all y-values for limit calculation
    if use_magnitude:
        all_y_values.extend([p.mag_ab for p in good_regular if p.mag_ab is not None])
        all_y_values.extend([p.mag_ab for p in rejected_regular if p.mag_ab is not None])
        all_y_values.extend([p.mag_ab for p in good_upper_limits if p.mag_ab is not None])
        all_y_values.extend([p.mag_ab for p in rejected_upper_limits if p.mag_ab is not None])
    else:
        all_y_values.extend([p.flux for p in good_regular])
        all_y_values.extend([p.flux for p in rejected_regular])
        all_y_values.extend([p.flux + p.flux_error for p in good_upper_limits])
        all_y_values.extend([p.flux + p.flux_error for p in rejected_upper_limits])

    if all_y_values:
        y_min, y_max = calculate_smart_ylimits(all_y_values, percentile_range=(0.1, 99.9))
        # For magnitude plots, axis is inverted so we need to reverse the order
        if use_magnitude:
            ax.set_ylim(y_max, y_min)  # Reversed for inverted axis
        else:
            ax.set_ylim(y_min, y_max)

    return ax


def create_lightcurve_plot(
    photometry_results: List[PhotometryResult],
    ax: Optional[Axes] = None,
    apply_clipping: bool = True,
    sigma: float = 3.0,
    apply_quality_filters: bool = True,
    sigma_threshold: float = 5.0,
    bad_flags_mask: Optional[int] = None,
    use_magnitude: bool = False,
    show_errorbars: bool = True,
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
    apply_quality_filters : bool
        Whether to classify points as good/rejected based on QC filters
    sigma_threshold : float
        Minimum SNR (flux/flux_err) for quality control
    bad_flags_mask : int, optional
        Integer mask with bad flag bits set (created by create_flag_mask)
    use_magnitude : bool
        If True, plot AB magnitude instead of flux (default: False)
    show_errorbars : bool
        If True, show errorbars (default: True)

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

    # Classify points by quality
    if apply_quality_filters and bad_flags_mask is not None:
        from ..utils.helpers import classify_photometry_by_quality

        classified = classify_photometry_by_quality(
            photometry_results,
            sigma_threshold=sigma_threshold,
            bad_flags_mask=bad_flags_mask,
            separate_upper_limits=False,  # Light curve doesn't separate upper limits
        )
        # Combine all good points (regular + upper limits if any)
        good_points = classified.good_regular
        rejected_points = classified.rejected_regular
    else:
        # No quality filtering - all points are "good"
        good_points = photometry_results
        rejected_points = []

    # Get colormap for wavelength coding
    cmap = cm.get_cmap(WAVELENGTH_CMAP)
    norm = Normalize(vmin=WAVELENGTH_RANGE[0], vmax=WAVELENGTH_RANGE[1])

    # Sort by MJD for proper time ordering
    good_sorted = sorted(good_points, key=lambda x: x.mjd)
    rejected_sorted = sorted(rejected_points, key=lambda x: x.mjd)

    # Two-pass plotting: errorbars first (transparent), then markers (solid)

    # Pass 1: Plot errorbars only (if enabled)
    if show_errorbars:
        for result in good_sorted:
            color = cmap(norm(result.wavelength))

            if result.is_upper_limit:
                # Skip upper limits for errorbars
                continue
            else:
                # Plot regular measurement errorbars
                if use_magnitude:
                    y_val = result.mag_ab if result.mag_ab is not None else np.nan
                    y_err = result.mag_ab_error if result.mag_ab_error is not None else np.nan
                else:
                    y_val = result.flux
                    y_err = result.flux_error

                if np.isnan(y_val):
                    continue

                ax.errorbar(
                    result.mjd,
                    y_val,
                    yerr=y_err,
                    fmt="none",  # No marker
                    capsize=0,
                    linewidth=0.5,
                    elinewidth=0.5,
                    color=color,
                    alpha=0.2,  # Transparent errorbars
                    zorder=1,  # Behind markers
                )

    # Pass 2: Plot markers only (solid)
    for result in good_sorted:
        color = cmap(norm(result.wavelength))

        if result.is_upper_limit:
            # Plot upper limit
            if use_magnitude:
                y_val = result.mag_ab if result.mag_ab is not None else np.nan
                marker = "^"  # Flip for magnitude (lower limit)
            else:
                y_val = result.flux + result.flux_error
                marker = "v"

            if np.isnan(y_val):
                continue

            ax.plot(
                result.mjd,
                y_val,
                marker,
                color=color,
                markersize=3,
                alpha=0.9,  # Solid markers
                zorder=2,  # On top of errorbars
            )
        else:
            # Plot regular measurement
            if use_magnitude:
                y_val = result.mag_ab if result.mag_ab is not None else np.nan
            else:
                y_val = result.flux

            if np.isnan(y_val):
                continue

            ax.plot(
                result.mjd,
                y_val,
                "o",
                color=color,
                markersize=1.5,
                alpha=0.9,  # Solid markers
                zorder=2,  # On top of errorbars
            )

    # Plot rejected points as small gray crosses
    for result in rejected_sorted:
        if use_magnitude:
            y_val = result.mag_ab if result.mag_ab is not None else np.nan
        else:
            y_val = result.flux

        # Skip if invalid magnitude
        if np.isnan(y_val):
            continue

        ax.plot(
            result.mjd,
            y_val,
            "x",
            color="gray",
            markersize=2,
            alpha=0.5,
        )

    # Add colorbar for wavelength
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Wavelength (μm)", fontsize=10)

    # Formatting
    ax.set_xlabel("MJD", fontsize=12)
    if use_magnitude:
        ax.set_ylabel("AB Magnitude", fontsize=12)
        ax.invert_yaxis()  # Fainter sources have higher magnitudes
    else:
        ax.set_ylabel("Flux Density (μJy)", fontsize=12)
    ax.set_title("SPHEREx Light Curve", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add some padding to x-axis (use all points for range)
    all_sorted = sorted(photometry_results, key=lambda x: x.mjd)
    mjds = [p.mjd for p in all_sorted]
    mjd_range = max(mjds) - min(mjds)
    if mjd_range > 0:
        ax.set_xlim(min(mjds) - 0.05 * mjd_range, max(mjds) + 0.05 * mjd_range)

    # Set smart y-axis limits based on percentiles to handle outliers
    all_y_values = []

    # Collect all y-values for limit calculation
    for result in good_points:
        if use_magnitude:
            if result.mag_ab is not None:
                all_y_values.append(result.mag_ab)
        else:
            if result.is_upper_limit:
                all_y_values.append(result.flux + result.flux_error)
            else:
                all_y_values.append(result.flux)

    for result in rejected_points:
        if use_magnitude:
            if result.mag_ab is not None:
                all_y_values.append(result.mag_ab)
        else:
            all_y_values.append(result.flux)

    if all_y_values:
        y_min, y_max = calculate_smart_ylimits(all_y_values, percentile_range=(0.1, 99.9))
        # For magnitude plots, axis is inverted so we need to reverse the order
        if use_magnitude:
            ax.set_ylim(y_max, y_min)  # Reversed for inverted axis
        else:
            ax.set_ylim(y_min, y_max)

    return ax


def create_combined_plot(
    photometry_results: List[PhotometryResult],
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[float, float]] = None,
    apply_clipping: bool = True,
    sigma: Optional[float] = None,
    apply_quality_filters: bool = True,
    sigma_threshold: float = 5.0,
    bad_flags: Optional[List[int]] = None,
    use_magnitude: bool = False,
    show_errorbars: bool = True,
    visualization_config: Optional["VisualizationConfig"] = None,
) -> Figure:
    """
    Create combined plot with spectrum and light curve.

    Quality control filters classify points as good or rejected:
    - Good points: plotted normally (filled circles)
    - Rejected points: plotted as small gray crosses
    - All points appear in the plot and are saved in CSV output

    Parameters
    ----------
    photometry_results : List[PhotometryResult]
        Photometry measurements (all points included)
    output_path : Path, optional
        Path to save figure. If None, figure is not saved.
    figsize : Tuple[float, float], optional
        Figure size in inches (overrides visualization_config if provided)
    apply_clipping : bool
        Whether to apply sigma clipping to remove outliers
    sigma : float, optional
        Number of standard deviations for sigma clipping (overrides visualization_config if provided)
    apply_quality_filters : bool
        Whether to apply quality control filters (SNR and flags)
    sigma_threshold : float
        Minimum SNR (flux/flux_err) for quality control (default: 5.0)
    bad_flags : List[int], optional
        List of bad flag bit positions to reject
        Default: [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19]
    use_magnitude : bool
        If True, plot AB magnitude instead of flux (default: False)
    show_errorbars : bool
        If True, show errorbars (default: True)
    visualization_config : VisualizationConfig, optional
        Advanced visualization configuration. If None, uses defaults.

    Returns
    -------
    Figure
        Matplotlib figure with both plots

    Notes
    -----
    Priority: explicit parameters > visualization_config > defaults
    """
    from ..core.config import VisualizationConfig

    # Use default config if none provided
    if visualization_config is None:
        visualization_config = VisualizationConfig()

    # Apply parameter priority: explicit > config > defaults
    if figsize is None:
        figsize = visualization_config.figsize
    if sigma is None:
        sigma = visualization_config.sigma_clip_sigma
    # Create flag mask if quality filtering is requested
    bad_flags_mask = None
    if apply_quality_filters:
        from ..utils.helpers import check_flag_bits, create_flag_mask

        if bad_flags is None:
            bad_flags = [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19]

        bad_flags_mask = create_flag_mask(bad_flags)

        # Log filtering statistics
        good_count = 0
        rejected_count = 0
        for p in photometry_results:
            snr = p.flux / p.flux_error if p.flux_error > 0 else 0.0
            fails_snr = snr < sigma_threshold
            fails_flag = check_flag_bits(p.flag, bad_flags_mask)
            if fails_snr or fails_flag:
                rejected_count += 1
            else:
                good_count += 1

        logger.info(
            f"Quality filtering: {len(photometry_results)} total points "
            f"({good_count} good, {rejected_count} rejected - shown as crosses)"
        )

    # Create figure with two subplots using constrained layout to handle colorbars
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 1]}, constrained_layout=True
    )

    # Create spectrum plot (top) with QC classification
    create_spectrum_plot(
        photometry_results,
        ax1,
        apply_clipping=apply_clipping,
        sigma=sigma,
        apply_quality_filters=apply_quality_filters,
        sigma_threshold=sigma_threshold,
        bad_flags_mask=bad_flags_mask,
        use_magnitude=use_magnitude,
        show_errorbars=show_errorbars,
    )

    # Create light curve plot (bottom) with QC classification
    create_lightcurve_plot(
        photometry_results,
        ax2,
        apply_clipping=apply_clipping,
        sigma=sigma,
        apply_quality_filters=apply_quality_filters,
        sigma_threshold=sigma_threshold,
        bad_flags_mask=bad_flags_mask,
        use_magnitude=use_magnitude,
        show_errorbars=show_errorbars,
    )

    # Save if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=visualization_config.dpi, bbox_inches="tight")
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
