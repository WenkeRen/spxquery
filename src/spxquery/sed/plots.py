"""
Modular visualization functions for SED reconstruction diagnostics.

This module provides composable, publication-quality plotting functions for
reconstructed spectra, residuals, and quality metrics. Each function operates
on provided matplotlib Axes objects without creating figures or applying layouts.

Design Philosophy:
- Composable functions: Each function handles one specific plotting task
- No figure management: Functions work on provided Axes objects only
- ASCII text + LaTeX: All text uses ASCII; special characters use LaTeX
- User control: Users create their own figures and pass Axes to functions
- Publication quality: Maintain journal-quality plotting standards

Key Features:
- Smart errorbar sampling to avoid visual overlap
- Ensemble reconstruction support with confidence regions
- SPHEREx band-aware color coding
- Comprehensive statistical overlays
- Performance optimized for large datasets
"""

import logging
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
from scipy import stats

from .config import SEDConfig
from .data_loader import BandData
from .matrices import build_global_observation_data

logger = logging.getLogger(__name__)

# SPHEREx band color scheme constants
SPHEREX_BAND_COLORS = {
    "D1": "#8B4789",  # Purple
    "D2": "#1f77b4",  # Blue
    "D3": "#2ca02c",  # Green
    "D4": "#ff7f0e",  # Orange
    "D5": "#d62728",  # Red
    "D6": "#8B0000",  # Dark red
}

# SPHEREx band wavelength ranges (microns)
SPHEREX_BAND_RANGES = {
    "D1": (0.75, 1.12),
    "D2": (1.10, 1.65),
    "D3": (1.63, 2.44),
    "D4": (2.40, 3.85),
    "D5": (3.81, 4.43),
    "D6": (4.41, 5.01),
}


def get_spex_band_colors() -> Dict[str, str]:
    """
    Return standardized SPHEREx band colors.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping band names to color strings.
    """
    return SPHEREX_BAND_COLORS.copy()


def get_spex_band_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Return standardized SPHEREx band wavelength ranges.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Dictionary mapping band names to (min_wavelength, max_wavelength) tuples.
    """
    return SPHEREX_BAND_RANGES.copy()


def add_statistics_text(ax, stats_dict, position="top-right"):
    """
    Add formatted statistics text box to axes.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes object to add text to.
    stats_dict : dict
        Dictionary of statistic name -> value pairs.
    position : str
        Position for text box: 'top-right', 'top-left', 'bottom-right', 'bottom-left'.
    """
    # Format statistics text with proper alignment
    lines = []
    for name, value in stats_dict.items():
        if isinstance(value, float):
            if abs(value) < 0.01 or abs(value) > 100:
                lines.append(f"{name}: {value:.2e}")
            else:
                lines.append(f"{name}: {value:.3f}")
        else:
            lines.append(f"{name}: {value}")

    text_str = "\n".join(lines)

    # Determine position coordinates
    if position == "top-right":
        x, y, ha, va = 0.98, 0.98, "right", "top"
    elif position == "top-left":
        x, y, ha, va = 0.02, 0.98, "left", "top"
    elif position == "bottom-right":
        x, y, ha, va = 0.98, 0.02, "right", "bottom"
    elif position == "bottom-left":
        x, y, ha, va = 0.02, 0.02, "left", "bottom"
    else:
        raise ValueError(f"Unknown position: {position}")

    # Add text box with semi-transparent background
    ax.text(
        x,
        y,
        text_str,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        ha=ha,
        va=va,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        zorder=9,
    )


def sample_observations_for_errorbars(band_data_dict, n_samples=15, coverage_strategy="continuous"):
    """
    Select optimal subset of observations for displaying x-axis errorbars.

    Uses a greedy algorithm to maximize wavelength coverage while minimizing overlap.

    Parameters
    ----------
    band_data_dict : dict
        Dictionary mapping band names to BandData objects.
    n_samples : int, optional
        Number of samples to select (default: 15). Only used for 'uniform' strategy.
        For 'continuous' strategy, the algorithm determines the optimal number.
    coverage_strategy : str
        Strategy for sampling: 'continuous' (default) or 'uniform'.

    Returns
    -------
    List[Tuple[str, int]]
        List of (band_name, observation_index) tuples for selected observations.
    """
    selected = []
    all_observations = []

    # Collect all observations with their wavelength ranges
    for band_name, band_data in band_data_dict.items():
        for i in range(len(band_data.wavelength_center)):
            wl = band_data.wavelength_center[i]
            bw = band_data.bandwidth[i] if hasattr(band_data, "bandwidth") and len(band_data.bandwidth) > i else 0.01
            all_observations.append((wl - bw / 2, wl + bw / 2, band_name, i))

    # Sort by wavelength
    all_observations.sort()

    if coverage_strategy == "continuous":
        # Greedy algorithm for continuous coverage
        # Select observations that maximize wavelength coverage with minimal overlap
        # The algorithm naturally determines the optimal number based on data structure
        last_end = -np.inf

        for start, end, band_name, obs_idx in all_observations:
            # Add observation if it extends coverage beyond previous selection
            # This ensures we get representative coverage across the wavelength range
            if start > last_end:
                selected.append((band_name, obs_idx))
                last_end = end
            elif len(selected) == 0:  # Always add first observation
                selected.append((band_name, obs_idx))
                last_end = end

    else:  # uniform strategy
        # Distribute samples evenly across wavelength range using n_samples
        if len(all_observations) <= n_samples:
            selected = [(band, idx) for _, _, band, idx in all_observations]
        else:
            indices = np.linspace(0, len(all_observations) - 1, n_samples, dtype=int)
            selected = [(all_observations[i][2], all_observations[i][3]) for i in indices]

    return selected


def calculate_model_predictions(band_data_dict: Dict[str, BandData], config: SEDConfig, flux: np.ndarray) -> np.ndarray:
    """
    Calculate model predictions using H @ x forward model.

    Parameters
    ----------
    band_data_dict : Dict[str, BandData]
        Dictionary of band measurement data
    config : SEDConfig
        SED reconstruction configuration
    flux : np.ndarray
        Reconstructed spectrum flux values (N,)

    Returns
    -------
    np.ndarray
        Predicted observations y = H @ x (M,)
    """
    # Build global observation data to get H matrix
    global_dataset = build_global_observation_data(band_data_dict, config)

    # Extract H matrix from GlobalSpectralData
    H = sp.csr_matrix(
        (global_dataset.H_values.cpu().numpy(), global_dataset.H_indices.cpu().numpy()), shape=global_dataset.H_shape
    )

    # Calculate forward model: y = H @ x
    return H @ flux


def plot_reconstructed_spectrum_with_data(
    ax, wavelength, flux, band_data_dict, config=None, flux_std=None, validation_metrics=None
):
    """
    Plot reconstructed spectrum with observational data colored by SPHEREx bands.

    Features:
    - Reconstructed spectrum color-coded by SPHEREx bands (D1-D6 with defined colors)
    - Layered visualization with proper z-ordering:
        zorder=0: Background shading for detector coverage
        zorder=1: Observational data (highly transparent, alpha=0.25)
        zorder=2: Optional confidence bands around flux (colored by detector)
        zorder=3: Error bars (smart sampling to avoid overlap)
        zorder=4: Reconstructed spectrum segments (detector-colored, bold)
    - Smart errorbar sampling: Select subset to show x-axis errorbars without overlap
    - Optional statistics overlay: observations count, spectral points, chi^2 per pixel, negative flux fraction
    - Confidence interval support: 1-sigma confidence regions around flux with detector-specific colors
    - Single legend entry: "Rec. Spec"

    Based on: Cell 11 in sed_reconstruction_demo_dsp.ipynb

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes object to plot on.
    wavelength : array_like
        Array of wavelength values (microns).
    flux : array_like
        Array of reconstructed flux values (microJy).
    band_data_dict : dict
        Dictionary mapping band names to BandData objects.
    config : SEDConfig, optional
        Configuration object for metadata extraction.
    flux_std : array_like, optional
        Array of flux standard deviations (microJy) for 1-sigma confidence bands.
    validation_metrics : ValidationMetrics, optional
        Pre-computed validation metrics. If provided, statistics box will display these values.
        If None, no statistics box will be shown.

    Notes
    -----
    This function modifies the provided Axes object in-place.
    The spectrum is automatically segmented by SPHEREx detector coverage regions
    (D1: 0.75-1.12 μm, D2: 1.10-1.65 μm, D3: 1.63-2.44 μm, D4: 2.40-3.85 μm,
     D5: 3.81-4.43 μm, D6: 4.41-5.01 μm) and colored accordingly.
    """
    # Convert to numpy arrays if needed
    wavelength = np.asarray(wavelength)
    flux = np.asarray(flux)

    colors = get_spex_band_colors()
    band_ranges = get_spex_band_ranges()

    # Plot all observations as transparent gray scatter points (background layer)
    all_wavelengths = []
    all_fluxes = []

    for band_name, band_data in band_data_dict.items():
        all_wavelengths.extend(band_data.wavelength_center)
        all_fluxes.extend(band_data.flux)

    if all_wavelengths:
        ax.scatter(
            all_wavelengths,
            all_fluxes,
            c="gray",
            s=5,
            alpha=0.25,
            edgecolors="black",
            linewidth=0.3,
            zorder=1,
            label="Observed",
        )

    # Add confidence bands if provided (background layer, above observations)
    if flux_std is not None:
        flux_std = np.asarray(flux_std)

        for band_name, (wl_min, wl_max) in band_ranges.items():
            if band_name in band_data_dict:
                # Find wavelength range for this band
                band_mask = (wavelength >= wl_min) & (wavelength <= wl_max)
                if np.any(band_mask):
                    color = colors.get(band_name, "gray")
                    # Plot confidence band for this detector segment
                    ax.fill_between(
                        wavelength[band_mask],
                        flux[band_mask] - flux_std[band_mask],
                        flux[band_mask] + flux_std[band_mask],
                        alpha=0.4,
                        color=color,
                        zorder=2,
                    )

    # Plot reconstructed spectrum color-coded by SPHEREx bands (foreground layer)
    for band_name, (wl_min, wl_max) in band_ranges.items():
        if band_name in band_data_dict:
            # Find wavelength range for this band
            band_mask = (wavelength >= wl_min) & (wavelength <= wl_max)
            if np.any(band_mask):
                color = colors.get(band_name, "gray")
                ax.plot(wavelength[band_mask], flux[band_mask], "-", color=color, linewidth=2.0, zorder=4)

    # Add single legend entry for the entire reconstructed spectrum
    from matplotlib.lines import Line2D

    proxy_line = Line2D([0], [0], color="black", linewidth=2.0, label="Rec. Spec")
    ax.add_line(proxy_line)

    # Add smart errorbar sampling (middle layer)
    selected_obs = sample_observations_for_errorbars(band_data_dict, coverage_strategy="continuous")

    for band_name, obs_idx in selected_obs:
        band_data = band_data_dict[band_name]
        wl = band_data.wavelength_center[obs_idx]
        flux_val = band_data.flux[obs_idx]
        flux_err = band_data.flux_error[obs_idx]

        # Bandwidth for horizontal error bars
        if hasattr(band_data, "bandwidth") and len(band_data.bandwidth) > obs_idx:
            bw = band_data.bandwidth[obs_idx]
        else:
            # Default bandwidth estimate
            if band_name in band_ranges:
                bw = (band_ranges[band_name][1] - band_ranges[band_name][0]) / 10
            else:
                bw = 0.1

        # Plot error bars (middle layer, above observations but below spectrum)
        ax.errorbar(
            wl, flux_val, yerr=flux_err, xerr=bw / 2, fmt="none", ecolor="black", alpha=0.8, capsize=2, zorder=3
        )

    # Add statistics text box if validation metrics provided
    if validation_metrics is not None:
        stats_dict = {
            "N_obs": validation_metrics.n_obs,
            "N_spec": len(wavelength),
            "$\\chi^2/M$": f"{validation_metrics.chi_squared_per_obs:.3f}",
            "Neg flux": f"{validation_metrics.negative_flux_fraction * 100:.1f}%",
        }
        add_statistics_text(ax, stats_dict, position="top-right")

    # Set appropriate axis limits
    # X-axis: wavelength range with padding
    wl_min, wl_max = wavelength.min(), wavelength.max()
    wl_range = wl_max - wl_min
    ax.set_xlim(wl_min - 0.01 * wl_range, wl_max + 0.01 * wl_range)

    # Y-axis: flux range with padding, but ensure we include observed data
    all_flux_values = list(flux)
    if all_wavelengths:
        all_flux_values.extend(all_fluxes)

    flux_min, flux_max = min(all_flux_values), max(all_flux_values)
    flux_range = flux_max - flux_min
    if flux_range > 0:
        ax.set_ylim(flux_min - 0.05 * flux_range, flux_max + 0.05 * flux_range)

    # Labels and formatting
    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"$F_\lambda$ ($\mu$Jy)", fontsize=12)

    # Set title with configuration info if available
    if config:
        title_parts = ["SED Reconstruction"]
        if hasattr(config, "dip_filters"):
            title_parts.append(f"DIP: {config.dip_filters} filters")
        if hasattr(config, "dip_depth"):
            title_parts.append(f"{config.dip_depth} layers")
        if hasattr(config, "ensemble_size") and config.ensemble_size > 1:
            title_parts.append(f"Ensemble: {config.ensemble_size:d}")
        title = " | ".join(title_parts)

        ax.set_title(title, fontsize=11)
    else:
        ax.set_title("SED Reconstruction", fontsize=11)

    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)


def plot_weighted_residuals(ax, wavelength, flux, band_data_dict, validation_metrics):
    """
    Plot weighted residuals (observed - model) / sigma with statistics.

    Features:
    - Weighted residuals calculated as (observed - expected) / flux_error
    - Scatter points colored by SPHEREx bands
    - Reference lines: y=0 (red dashed), y=+/-1 (orange dotted), y=+/-3 (red dotted)
    - Statistics overlay: weighted residual mean, standard deviation, normality p-value
    - Residual oscillation metric
    - Proper axis labels with LaTeX formatting

    Based on: Fourth subplot in Cell 20 of spectral_reconstruction_demo_with_wandb.ipynb

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes object to plot on.
    wavelength : array_like
        Array of wavelength values (microns).
    flux : array_like
        Array of reconstructed flux values (microJy).
    band_data_dict : dict
        Dictionary mapping band names to BandData objects.
    validation_metrics : ValidationMetrics
        Validation metrics object with residual statistics.

    Notes
    -----
    This function modifies the provided Axes object in-place.
    """
    # Convert to numpy arrays if needed
    wavelength = np.asarray(wavelength)
    flux = np.asarray(flux)

    colors = get_spex_band_colors()

    # Calculate weighted residuals
    all_wavelengths = []
    all_weighted_residuals = []
    all_colors = []

    for band_name, band_data in band_data_dict.items():
        color = colors.get(band_name, "gray")

        # Interpolate reconstructed spectrum at measurement wavelengths
        model_flux = np.interp(band_data.wavelength_center, wavelength, flux)
        residuals = band_data.flux - model_flux
        weighted_residuals = residuals / band_data.flux_error

        all_wavelengths.extend(band_data.wavelength_center)
        all_weighted_residuals.extend(weighted_residuals)
        all_colors.extend([color] * len(band_data.wavelength_center))

    if not all_weighted_residuals:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center", va="center", fontsize=12)
        return

    # Plot weighted residuals
    ax.scatter(
        all_wavelengths,
        all_weighted_residuals,
        c=all_colors,
        s=10,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        zorder=3,
    )

    # Add reference lines
    ax.axhline(0, color="#17becf", linestyle="--", linewidth=2.5, label="Zero residual", zorder=4)
    ax.axhline(1, color="#FF1493", linestyle=":", linewidth=2.5, alpha=0.8, label="+/-1$\\sigma$", zorder=4)
    ax.axhline(-1, color="#FF1493", linestyle=":", linewidth=2.5, alpha=0.8, zorder=4)
    ax.axhline(3, color="#008080", linestyle=":", linewidth=1.5, alpha=0.8, label="+/-3$\\sigma$", zorder=4)
    ax.axhline(-3, color="#008080", linestyle=":", linewidth=1.5, alpha=0.8, zorder=4)

    # Calculate statistics
    all_weighted_residuals = np.array(all_weighted_residuals)

    # Normality test (Shapiro-Wilk) - only if we have validation_metrics with proper p-value
    if hasattr(validation_metrics, "normality_pvalue") and not np.isnan(validation_metrics.normality_pvalue):
        normality_text = f"p={validation_metrics.normality_pvalue:.3f}"
    else:
        normality_text = "N/A"

    # Statistics dictionary
    stats_dict = {
        "mu": f"{validation_metrics.weighted_residual_mean:.3f}",
        "sigma": f"{validation_metrics.weighted_residual_std:.3f}",
        "Normality": normality_text,
    }

    # Add residual oscillation metric if available
    if hasattr(validation_metrics, "residual_oscillation") and not np.isnan(validation_metrics.residual_oscillation):
        stats_dict["Oscillation"] = f"{validation_metrics.residual_oscillation:.3f}"

    add_statistics_text(ax, stats_dict, position="top-right")

    # Set appropriate axis limits
    # X-axis: wavelength range with padding, match spectrum data if available
    if len(all_wavelengths) > 0:
        wl_min_obs, wl_max_obs = min(all_wavelengths), max(all_wavelengths)
        wl_range_obs = wl_max_obs - wl_min_obs
        ax.set_xlim(wl_min_obs - 0.01 * wl_range_obs, wl_max_obs + 0.01 * wl_range_obs)
    else:
        # Fall back to spectrum wavelength range
        wl_min, wl_max = wavelength.min(), wavelength.max()
        wl_range = wl_max - wl_min
        ax.set_xlim(wl_min - 0.01 * wl_range, wl_max + 0.01 * wl_range)

    # Y-axis: weighted residual limits with minimum range
    y_max = max(4, np.abs(all_weighted_residuals).max() * 1.1)
    ax.set_ylim(-y_max, y_max)

    # Labels and formatting
    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"Weighted Residual ($\sigma$)", fontsize=12)
    ax.set_title("Weighted Residuals", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)


def plot_photometry_comparison(
    ax,
    band_data_dict,
    config,
    flux,
    wavelength=None,
    observed_flux=None,
    predicted_flux=None,
    band_indices=None,
):
    """
    Direct wavelength-based comparison of observed vs predicted photometry points with minimal spectrum context.

    Features:
    - Minimal spectrum line (very thin black, linewidth=0.5) for wavelength context
    - Observed flux plotted at their respective wavelengths as circle markers
    - Predicted flux (H @ x) plotted at same wavelengths as square markers
    - Points colored by SPHEREx bands (D1-D6) for easy identification
    - Smart errorbar sampling for observed data to avoid visual overlap
    - Clean background without band shading for clear data visibility
    - Statistics overlay: correlation coefficient, RMS difference, observation count
    - Automatic H @ x forward model calculation when predicted_flux not provided

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes object to plot on.
    band_data_dict : dict
        Dictionary mapping band names to BandData objects.
    config : SEDConfig
        Configuration for H matrix construction.
    flux : np.ndarray
        Reconstructed spectrum flux values at wavelength grid.
    wavelength : array_like, optional
        Array of wavelength values (microns) corresponding to flux.
        If provided, plots minimal spectrum line using flux values.
    observed_flux : array_like, optional
        Array of observed flux values. If None, extracted from band_data_dict.
    predicted_flux : array_like, optional
        Array of predicted flux values. If None, calculated as H @ x.
    band_indices : array_like, optional
        Array indicating band index for each observation (0-5 for D1-D6).
        If None, generated from band_data_dict.

    Notes
    -----
    This function can work in two modes:
    1. Automatic: Calculate predicted_flux as H @ x from band_data_dict, config, flux
    2. Manual: Use provided observed_flux, predicted_flux, band_indices arrays

    The spectrum line (if wavelength provided) serves as a subtle wavelength reference
    and should be very thin to not obscure the data points.
    """
    colors = get_spex_band_colors()
    band_names = ["D1", "D2", "D3", "D4", "D5", "D6"]

    # Automatic mode: calculate from H @ x
    if predicted_flux is None and band_data_dict is not None and config is not None and flux is not None:
        try:
            # Calculate predicted flux using H @ x
            predicted_flux = calculate_model_predictions(band_data_dict, config, flux)

            # Extract observed flux from band_data_dict if not provided
            if observed_flux is None:
                observed_flux = np.concatenate([band_data.flux for band_data in band_data_dict.values()])

            # Generate band indices if not provided
            if band_indices is None:
                band_indices = []
                for i, band_name in enumerate(band_names):
                    if band_name in band_data_dict:
                        n_measurements = len(band_data_dict[band_name].flux)
                        band_indices.extend([i] * n_measurements)
                band_indices = np.array(band_indices)

        except Exception as e:
            logger.error(f"Failed to calculate photometry comparison with H @ x: {e}")
            return

    # Manual mode: use provided arrays
    elif observed_flux is None or predicted_flux is None or band_indices is None:
        raise ValueError("In manual mode, you must provide observed_flux, predicted_flux, and band_indices")

    # Convert to numpy arrays
    observed_flux = np.asarray(observed_flux)
    predicted_flux = np.asarray(predicted_flux)
    band_indices = np.asarray(band_indices)

    # Extract observation wavelengths from band_data_dict
    observed_wavelengths = []
    for band_name in band_names:
        if band_name in band_data_dict:
            observed_wavelengths.extend(band_data_dict[band_name].wavelength_center)
    observed_wavelengths = np.array(observed_wavelengths)

    # Plot minimal spectrum line if wavelength provided (use flux as spectrum)
    if wavelength is not None:
        wavelength = np.asarray(wavelength)
        flux_array = np.asarray(flux)
        ax.plot(wavelength, flux_array, "k-", linewidth=1, alpha=0.7, label="Spectrum", zorder=5)

    # Plot observed data points as circles
    ax.scatter(
        observed_wavelengths,
        observed_flux,
        c="gray",
        s=5,
        alpha=0.25,
        edgecolors="black",
        linewidth=0.3,
        zorder=1,
        label="Observed",
    )

    # Plot predicted data points as squares
    for i, band_name in enumerate(band_names):
        mask = band_indices == i
        if np.any(mask):
            color = colors.get(band_name, "gray")
            ax.scatter(
                observed_wavelengths[mask],
                predicted_flux[mask],
                marker="s",
                c=color,
                s=10,
                alpha=0.8,
                # edgecolors="",
                linewidth=0.5,
                label="Predicted" if i == 0 else None,
                zorder=3,
            )

    # Add smart errorbar sampling for observed data
    selected_obs = sample_observations_for_errorbars(band_data_dict, coverage_strategy="continuous")

    for band_name, obs_idx in selected_obs:
        band_data = band_data_dict[band_name]
        wl = band_data.wavelength_center[obs_idx]
        flux_val = band_data.flux[obs_idx]
        flux_err = band_data.flux_error[obs_idx]

        # Bandwidth for horizontal error bars
        if hasattr(band_data, "bandwidth") and len(band_data.bandwidth) > obs_idx:
            bw = band_data.bandwidth[obs_idx]
        else:
            # Default bandwidth estimate
            band_ranges = get_spex_band_ranges()
            if band_name in band_ranges:
                bw = (band_ranges[band_name][1] - band_ranges[band_name][0]) / 10
            else:
                bw = 0.1

        # Plot error bars for observed data
        ax.errorbar(
            wl, flux_val, yerr=flux_err, xerr=bw / 2, fmt="none", ecolor="black", alpha=0.8, capsize=2, zorder=4
        )

    # Set appropriate axis limits
    # X-axis: wavelength range with padding
    wl_min_obs, wl_max_obs = observed_wavelengths.min(), observed_wavelengths.max()
    wl_range_obs = wl_max_obs - wl_min_obs
    ax.set_xlim(wl_min_obs - 0.05 * wl_range_obs, wl_max_obs + 0.05 * wl_range_obs)

    # Y-axis: flux range covering spectrum, observed and predicted values
    all_flux_values = list(observed_flux) + list(predicted_flux)
    if wavelength is not None:
        all_flux_values.extend(list(np.asarray(flux)))

    flux_min, flux_max = min(all_flux_values), max(all_flux_values)
    flux_range = flux_max - flux_min
    if flux_range > 0:
        ax.set_ylim(flux_min - 0.1 * flux_range, flux_max + 0.1 * flux_range)

    # Calculate statistics
    correlation = np.corrcoef(observed_flux, predicted_flux)[0, 1]
    rms_diff = np.sqrt(np.mean((observed_flux - predicted_flux) ** 2))
    n_obs = len(observed_flux)

    # Statistics dictionary
    stats_dict = {
        "N_obs": n_obs,
        "r": f"{correlation:.3f}",
        "RMS": f"{rms_diff:.3f}",
    }

    add_statistics_text(ax, stats_dict, position="top-right")

    # Labels and formatting
    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"Flux ($\mu$Jy)", fontsize=12)
    ax.set_title("Photometry Comparison: Observed vs Predicted", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)


def plot_residual_distribution(ax, weighted_residuals):
    """
    Analyze statistical distribution of residuals.

    Features:
    - Histogram of weighted residuals with overlaid standard Gaussian
    - Q-Q plot (Quantile-Quantile) against theoretical normal distribution
    - Annotation of Skewness and Kurtosis metrics

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes object to plot on.
    weighted_residuals : array_like
        Array of weighted residuals.
    """
    weighted_residuals = np.asarray(weighted_residuals)

    if len(weighted_residuals) == 0:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center", va="center", fontsize=12)
        return

    # Set appropriate axis limits based on data
    residual_min, residual_max = weighted_residuals.min(), weighted_residuals.max()
    residual_range = residual_max - residual_min

    # Extend limits slightly for better visualization
    x_min = residual_min - 0.1 * residual_range
    x_max = residual_max + 0.1 * residual_range

    # Ensure we include standard normal range (-3, +3) as minimum
    x_min = min(x_min, -4)
    x_max = max(x_max, 4)

    # Create histogram
    n_bins = min(50, max(10, int(len(weighted_residuals) / 20)))
    counts, bins, patches = ax.hist(
        weighted_residuals, bins=n_bins, alpha=0.7, color="steelblue", edgecolor="black", density=True, zorder=2
    )

    # Overlay theoretical normal distribution (extend range for smooth curve)
    x_range = np.linspace(x_min, x_max, 200)
    theoretical_normal = stats.norm.pdf(x_range, 0, 1)
    ax.plot(x_range, theoretical_normal, "r-", linewidth=2, label="Standard Normal", zorder=3)

    # Add vertical line at mean
    mean_residual = np.mean(weighted_residuals)
    ax.axvline(
        mean_residual, color="orange", linestyle="-", linewidth=1.5, label=f"Mean = {mean_residual:.3f}", zorder=1
    )

    # Calculate statistics
    skewness = stats.skew(weighted_residuals)
    kurtosis = stats.kurtosis(weighted_residuals, fisher=False)  # Pearson's kurtosis

    # Statistics text
    stats_text = f"Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}"
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )

    # Set axis limits
    ax.set_xlim(x_min, x_max)

    # Y-axis: use histogram height plus some padding
    y_max = max(counts.max(), theoretical_normal.max()) * 1.1
    ax.set_ylim(0, y_max)

    # Labels and formatting
    ax.set_xlabel("Weighted Residual ($\\sigma$)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Residual Distribution", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)


def plot_cumulative_chi_squared(ax, band_data_dict, weighted_residuals):
    """
    Identify spectral regions contributing most to the error.

    Features:
    - Cumulative sum of squared weighted residuals vs observed wavelength
    - Steep jumps indicate problematic spectral features or bad data points

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes object to plot on.
    band_data_dict : dict
        Dictionary mapping band names to BandData objects.
    weighted_residuals : array_like
        Array of weighted residuals (one per observation).
    """
    weighted_residuals = np.asarray(weighted_residuals)

    # Build observed wavelengths from band_data_dict
    band_names = ["D1", "D2", "D3", "D4", "D5", "D6"]
    observed_wavelengths = []
    for band_name in band_names:
        if band_name in band_data_dict:
            observed_wavelengths.extend(band_data_dict[band_name].wavelength_center)
    observed_wavelengths = np.array(observed_wavelengths)

    if len(observed_wavelengths) != len(weighted_residuals) or len(weighted_residuals) == 0:
        ax.text(0.5, 0.5, "Invalid data", transform=ax.transAxes, ha="center", va="center", fontsize=12)
        return

    # Sort by wavelength for proper cumulative plotting
    sort_indices = np.argsort(observed_wavelengths)
    wavelength_sorted = observed_wavelengths[sort_indices]
    residuals_sorted = weighted_residuals[sort_indices]

    # Calculate cumulative chi-squared
    cumulative_chi2 = np.cumsum(residuals_sorted**2)

    # Set appropriate axis limits
    # X-axis: wavelength range with padding
    wl_min, wl_max = wavelength_sorted.min(), wavelength_sorted.max()
    wl_range = wl_max - wl_min
    ax.set_xlim(wl_min - 0.02 * wl_range, wl_max + 0.02 * wl_range)

    # Y-axis: cumulative chi-squared range with padding
    chi2_min, chi2_max = 0, cumulative_chi2[-1]
    chi2_range = chi2_max - chi2_min
    if chi2_range > 0:
        ax.set_ylim(chi2_min - 0.02 * chi2_range, chi2_max + 0.05 * chi2_range)
    else:
        ax.set_ylim(0, 1)  # Default range if no variation

    # Plot cumulative chi-squared
    ax.plot(wavelength_sorted, cumulative_chi2, "b-", linewidth=2, zorder=2)
    ax.fill_between(wavelength_sorted, 0, cumulative_chi2, alpha=0.3, color="blue", zorder=1)

    # Add reference line for expected uniform distribution
    expected_chi2 = np.linspace(0, cumulative_chi2[-1], len(wavelength_sorted))
    ax.plot(wavelength_sorted, expected_chi2, "r--", linewidth=1.5, label="Uniform distribution", zorder=3)

    # Use horizontal line at the bottom to indicate the detector coverage regions
    band_ranges = get_spex_band_ranges()
    for band_name, (wl_min_band, wl_max_band) in band_ranges.items():
        if band_name in band_data_dict:
            ax.axvspan(
                wl_min_band,
                wl_max_band,
                ymin=0.01,
                ymax=0.02,
                color=get_spex_band_colors().get(band_name, "gray"),
                zorder=3,
            )

    # Labels and formatting
    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=12)
    ax.set_ylabel("Cumulative $\\chi^2$", fontsize=12)
    ax.set_title("Cumulative $\\chi^2$ Distribution", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)


def plot_residuals_vs_flux(ax, band_data_dict, config, flux, predicted_flux=None, weighted_residuals=None):
    """
    Check for heteroscedasticity (errors scaling with brightness).

    Features:
    - Weighted residuals vs Predicted Flux
    - Reference lines at 0, +/-1$\\sigma$, +/-3$\\sigma$
    - "Fan" shape detection
    - Calculates predicted_flux as H @ x internally when not provided

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes object to plot on.
    band_data_dict : dict
        Dictionary mapping band names to BandData objects.
    config : SEDConfig
        Configuration for H matrix construction.
    flux : np.ndarray
        Reconstructed spectrum flux values.
    predicted_flux : array_like, optional
        Array of predicted flux values. If None, calculated as H @ x.
    weighted_residuals : array_like, optional
        Array of weighted residuals. If None, calculated from data.

    Notes
    -----
    This function can work in two modes:
    1. Automatic: Calculate predicted_flux as H @ x and weighted_residuals from data
    2. Manual: Use provided predicted_flux and weighted_residuals arrays
    """
    # Automatic mode: calculate from H @ x
    if predicted_flux is None and band_data_dict is not None and config is not None and flux is not None:
        try:
            # Calculate predicted flux using H @ x
            predicted_flux = calculate_model_predictions(band_data_dict, config, flux)

            # Calculate weighted residuals from data
            observed_flux = np.concatenate([band_data.flux for band_data in band_data_dict.values()])
            flux_errors = np.concatenate([band_data.flux_error for band_data in band_data_dict.values()])
            weighted_residuals = (observed_flux - predicted_flux) / flux_errors

        except Exception as e:
            logger.error(f"Failed to calculate residuals vs flux with H @ x: {e}")
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes, ha="center", va="center", fontsize=12)
            return

    # Manual mode: use provided arrays
    elif predicted_flux is None or weighted_residuals is None:
        raise ValueError("In manual mode, you must provide both predicted_flux and weighted_residuals")

    # Convert to numpy arrays
    predicted_flux = np.asarray(predicted_flux)
    weighted_residuals = np.asarray(weighted_residuals)

    if len(predicted_flux) != len(weighted_residuals) or len(weighted_residuals) == 0:
        ax.text(0.5, 0.5, "Invalid data", transform=ax.transAxes, ha="center", va="center", fontsize=12)
        return

    # Set appropriate axis limits
    # X-axis: predicted flux range with padding
    flux_min, flux_max = predicted_flux.min(), predicted_flux.max()
    flux_range = flux_max - flux_min
    if flux_range > 0:
        ax.set_xlim(flux_min - 0.05 * flux_range, flux_max + 0.05 * flux_range)

    # Y-axis: weighted residuals with minimum range including reference lines
    residual_max = max(4, np.abs(weighted_residuals).max() * 1.1)
    ax.set_ylim(-residual_max, residual_max)

    # Plot weighted residuals vs predicted flux
    ax.scatter(
        predicted_flux,
        weighted_residuals,
        s=10,
        alpha=0.3,
        color="gray",
        edgecolors="black",
        linewidth=0.5,
        zorder=3,
    )

    # Add reference lines
    ax.axhline(0, color="#17becf", linestyle="--", linewidth=2.5, label="Zero residual", zorder=4)
    ax.axhline(1, color="#FF1493", linestyle=":", linewidth=2.5, alpha=0.8, label="+/-1$\\sigma$", zorder=4)
    ax.axhline(-1, color="#FF1493", linestyle=":", linewidth=2.5, alpha=0.8, zorder=4)
    ax.axhline(3, color="#008080", linestyle=":", linewidth=1.5, alpha=0.8, label="+/-3$\\sigma$", zorder=4)
    ax.axhline(-3, color="#008080", linestyle=":", linewidth=1.5, alpha=0.8, zorder=4)

    # Check for fan shape (heteroscedasticity)
    flux_bins = np.percentile(predicted_flux, [0, 25, 50, 75, 100])
    bin_stds = []
    bin_centers = []

    for i in range(len(flux_bins) - 1):
        mask = (predicted_flux >= flux_bins[i]) & (predicted_flux <= flux_bins[i + 1])
        if np.sum(mask) > 2:  # Need at least 3 points
            bin_stds.append(np.std(weighted_residuals[mask], ddof=1))
            bin_centers.append((flux_bins[i] + flux_bins[i + 1]) / 2)

    # Add heteroscedasticity metric
    if len(bin_stds) >= 2:
        heteroscedasticity = np.std(bin_stds) / np.mean(bin_stds)
        ax.text(
            0.98,
            0.02,
            f"Heteroscedasticity: {heteroscedasticity:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    x_range = np.percentile(predicted_flux, [1, 99])
    ax.set_xlim(x_range[0] - 0.05 * (x_range[1] - x_range[0]), x_range[1] + 0.05 * (x_range[1] - x_range[0]))
    y_range = np.percentile(weighted_residuals, [1, 99])
    y_extend = max(5, np.abs(y_range).max())
    ax.set_ylim(-y_extend, y_extend)

    # Labels and formatting
    ax.set_xlabel(r"Predicted Flux ($\mu$Jy)", fontsize=12)
    ax.set_ylabel(r"Weighted Residual ($\sigma$)", fontsize=12)
    ax.set_title("Residuals vs Flux", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)


def plot_sed_reconstruction_dashboard(
    result,
    output_path,
    dpi=300,
    title="SED Reconstruction Quality Assessment",
):
    """
    Create and save a comprehensive 4-row diagnostic dashboard for SED reconstruction.

    This function creates a single figure combining all standard diagnostic plots:
    - Row 1: Reconstructed spectrum with observational data
    - Row 2: Photometry comparison (observed vs predicted)
    - Row 3: Weighted residuals vs wavelength
    - Row 4: Three columns - Residual distribution, Cumulative chi-squared, Residuals vs flux

    Automatically handles both single and ensemble reconstruction results.
    Extracts config and band_data directly from the result object.

    Parameters
    ----------
    result : SEDReconstructionResult or EnsembleResult
        Reconstruction result object. Can be either:
        - Single result (SEDReconstructionResult) with attributes:
          wavelength, flux, config, band_data, validation_metrics
        - Ensemble result (EnsembleResult) with attributes:
          wavelength, mean_flux, std_flux, config, band_data, validation_metrics
    output_path : str or Path
        Path where the figure will be saved. File extension determines format
        (e.g., '.pdf', '.png', '.svg'). For PDF format, dpi parameter is ignored.
    dpi : int, optional
        Dots per inch for raster formats (PNG, JPG). Default: 300.
        Ignored for vector formats (PDF, SVG).
    title : str, optional
        Overall figure title. Default: "SED Reconstruction Quality Assessment".

    Returns
    -------
    None
        Figure is saved directly to output_path.

    Examples
    --------
    >>> # Single reconstruction
    >>> plot_sed_reconstruction_dashboard(
    ...     result=single_result,
    ...     output_path="sed_diagnostic.png",
    ...     dpi=300
    ... )

    >>> # Ensemble reconstruction
    >>> plot_sed_reconstruction_dashboard(
    ...     result=ensemble_result,
    ...     output_path="sed_diagnostic.pdf",
    ...     title="Ensemble SED Reconstruction (N=100)"
    ... )
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    # Extract config and band_data from result object
    config = result.config
    band_data_dict = result.band_data

    # Detect result type and extract data
    if hasattr(result, "mean_flux"):
        # Ensemble result detected
        logger.info("Ensemble reconstruction detected - using mean flux and confidence bands")
        wavelength = result.wavelength
        flux = result.mean_flux
        ensemble_std = result.std_flux
        validation_metrics = result.validation_metrics
    else:
        # Single result detected
        logger.info("Single reconstruction detected")
        wavelength = result.wavelength
        flux = result.flux
        ensemble_std = None
        validation_metrics = result.validation_metrics

    # Create figure with 4 rows: first 3 rows have 1 column, 4th row has 3 columns
    fig = plt.figure(figsize=(16, 14), constrained_layout=False)

    # Create grid specification: 4 rows, with row 4 having 3 columns
    # Height ratios: taller rows for spectrum plots, shorter for bottom diagnostics
    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=[1.2, 1.0, 0.8, 0.9],
        hspace=0.35,
        wspace=0.30,
        top=0.95,  # Controls the top of the subplot area
        bottom=0.05,
        left=0.08,
        right=0.95,
    )

    # Row 1: Reconstructed spectrum with data (spans all 3 columns)
    ax1 = fig.add_subplot(gs[0, :])
    plot_reconstructed_spectrum_with_data(
        ax1, wavelength, flux, band_data_dict, config, ensemble_std, validation_metrics
    )
    ax1.set_title(f"{title} - Spectrum & Data", fontsize=13, fontweight="bold")

    # Row 2: Photometry comparison (spans all 3 columns)
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1, sharey=ax1)
    plot_photometry_comparison(ax2, band_data_dict, config, flux, wavelength=wavelength)
    ax2.set_title("Photometry Comparison: Observed vs Predicted", fontsize=13, fontweight="bold")

    # Row 3: Weighted residuals (spans all 3 columns)
    ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
    plot_weighted_residuals(ax3, wavelength, flux, band_data_dict, validation_metrics)
    ax3.set_title("Weighted Residuals vs Wavelength", fontsize=13, fontweight="bold")

    # Row 4: Three diagnostic plots (each in separate column)
    # Column 1: Residual distribution
    ax4 = fig.add_subplot(gs[3, 0])
    plot_residual_distribution(ax4, validation_metrics.weighted_residuals)
    ax4.set_title("Residual Distribution", fontsize=12, fontweight="bold")

    # Column 2: Cumulative chi-squared
    ax5 = fig.add_subplot(gs[3, 1])
    plot_cumulative_chi_squared(ax5, band_data_dict, validation_metrics.weighted_residuals)
    ax5.set_title(r"Cumulative $\chi^2$ by Wavelength", fontsize=12, fontweight="bold")

    # Column 3: Residuals vs flux
    ax6 = fig.add_subplot(gs[3, 2])
    plot_residuals_vs_flux(ax6, band_data_dict, config, flux)
    ax6.set_title("Residuals vs Predicted Flux", fontsize=12, fontweight="bold")

    # Add overall figure title
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine format from extension
    file_format = output_path.suffix.lower().lstrip(".")

    # Save with appropriate parameters
    if file_format in ["pdf", "svg", "eps", "ps"]:
        # Vector formats: dpi doesn't apply
        logger.info(f"Saving vector format dashboard to {output_path}")
        fig.savefig(output_path, format=file_format, bbox_inches="tight")
    else:
        # Raster formats: use dpi
        logger.info(f"Saving raster format dashboard to {output_path} at {dpi} dpi")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    logger.info(f"Dashboard saved successfully to {output_path}")
