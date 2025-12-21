"""
Visualization functions for SED reconstruction diagnostics.

This module provides publication-quality plotting for reconstructed spectra,
residuals, and quality metrics for PyTorch-based Deep Image Prior reconstruction.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .reconstruction import SEDReconstructionResult
from .data_loader import BandData

logger = logging.getLogger(__name__)


# SPHEREx band wavelength ranges (approximate, for color coding)
BAND_WAVELENGTH_RANGES = {
    "D1": (0.75, 1.09),
    "D2": (1.10, 1.62),
    "D3": (1.63, 2.41),
    "D4": (2.42, 3.82),
    "D5": (3.83, 4.41),
    "D6": (4.42, 5.00),
}
# Band colors for plotting
BAND_COLORS = {
    "D1": "#8B4789",  # Purple
    "D2": "#1f77b4",  # Blue
    "D3": "#2ca02c",  # Green
    "D4": "#ff7f0e",  # Orange
    "D5": "#d62728",  # Red
    "D6": "#8B0000",  # Darkred
}


def plot_reconstructed_spectrum(
    result: SEDReconstructionResult,
    ax: Optional[plt.Axes] = None,
    show_measurements: bool = True,
    figsize: tuple = (12, 6),
) -> plt.Axes:
    """
    Plot reconstructed spectrum with overlaid measurements.

    Parameters
    ----------
    result : SEDReconstructionResult
        Reconstruction result with global spectrum and band data.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_measurements : bool
        Whether to overlay original measurements as scatter points.
    figsize : tuple
        Figure size if creating new figure.

    Returns
    -------
    plt.Axes
        Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot reconstructed spectrum
    ax.plot(
        result.wavelength,
        result.flux,
        "k-",
        linewidth=1.5,
        label="Reconstructed spectrum",
        zorder=2,
    )

    # Overlay measurements if requested
    if show_measurements and result.band_data:
        for band_name, band_data in result.band_data.items():
            color = BAND_COLORS.get(band_name, "gray")
            ax.scatter(
                band_data.wavelength_center,
                band_data.flux,
                c=color,
                s=30,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
                label=f"{band_name} measurements",
                zorder=3,
            )

    # Add wavelength ranges for bands
    for band_name, (lambda_min, lambda_max) in BAND_WAVELENGTH_RANGES.items():
        if band_name in result.band_data:
            color = BAND_COLORS.get(band_name, "gray")
            ax.axvspan(
                lambda_min,
                lambda_max,
                alpha=0.1,
                color=color,
                zorder=1,
            )

    # Labels and formatting
    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"$F_\lambda$ ($\mu$Jy)", fontsize=12)
    ax.set_title(
        f"Global SED Reconstruction\n"
        f"DIP: {result.config.dip_filters} filters, {result.config.dip_depth} layers, "
        f"$\\chi^2_\\nu$={result.validation_metrics.chi2_nu:.3f}",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(result.config.wavelength_range)

    return ax


def plot_residuals(
    result: SEDReconstructionResult,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 6),
) -> plt.Axes:
    """
    Plot residuals (observed - model) for all measurements.

    Parameters
    ----------
    result : SEDReconstructionResult
        Reconstruction result with validation metrics.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple
        Figure size if creating new figure.

    Returns
    -------
    plt.Axes
        Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if not result.band_data:
        logger.warning("No band data available for residual plotting")
        return ax

    # Collect residuals for all bands
    all_wavelengths = []
    all_residuals = []
    all_colors = []

    for band_name, band_data in result.band_data.items():
        color = BAND_COLORS.get(band_name, "gray")

        # Interpolate reconstructed spectrum at measurement wavelengths
        model_flux = np.interp(band_data.wavelength_center, result.wavelength, result.flux)
        residuals = band_data.flux - model_flux

        all_wavelengths.extend(band_data.wavelength_center)
        all_residuals.extend(residuals)
        all_colors.extend([color] * len(band_data.wavelength_center))

    # Plot residuals
    for i, (wl, res, color) in enumerate(zip(all_wavelengths, all_residuals, all_colors)):
        ax.scatter(wl, res, c=color, s=20, alpha=0.7, edgecolors="black", linewidth=0.5)

    # Add zero line
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero residual", zorder=1)

    # Labels and formatting
    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"Residual ($\mu$Jy)", fontsize=12)
    ax.set_title(
        f"Measurement Residuals\n"
        f"Mean = {np.mean(all_residuals):.2e} $\\mu$Jy, "
        f"Std = {np.std(all_residuals):.2e} $\\mu$Jy",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(result.config.wavelength_range)

    return ax


def plot_weighted_residuals(
    result: SEDReconstructionResult,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 6),
) -> plt.Axes:
    """
    Plot weighted residuals in units of measurement uncertainties.

    Parameters
    ----------
    result : SEDReconstructionResult
        Reconstruction result with validation metrics.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple
        Figure size if creating new figure.

    Returns
    -------
    plt.Axes
        Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if not result.band_data:
        logger.warning("No band data available for weighted residual plotting")
        return ax

    # Collect weighted residuals for all bands
    all_wavelengths = []
    all_weighted_residuals = []
    all_colors = []

    for band_name, band_data in result.band_data.items():
        color = BAND_COLORS.get(band_name, "gray")

        # Interpolate reconstructed spectrum at measurement wavelengths
        model_flux = np.interp(band_data.wavelength_center, result.wavelength, result.flux)
        residuals = band_data.flux - model_flux
        weighted_residuals = residuals / band_data.flux_error

        all_wavelengths.extend(band_data.wavelength_center)
        all_weighted_residuals.extend(weighted_residuals)
        all_colors.extend([color] * len(band_data.wavelength_center))

    # Plot weighted residuals
    for i, (wl, w_res, color) in enumerate(zip(all_wavelengths, all_weighted_residuals, all_colors)):
        ax.scatter(wl, w_res, c=color, s=20, alpha=0.7, edgecolors="black", linewidth=0.5)

    # Add zero line and ±1, ±2 sigma lines
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero", zorder=1)
    ax.axhline(1, color="orange", linestyle=":", linewidth=1, alpha=0.7, label="±1sig")
    ax.axhline(-1, color="orange", linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(2, color="yellow", linestyle=":", linewidth=1, alpha=0.5, label="±2sig")
    ax.axhline(-2, color="yellow", linestyle=":", linewidth=1, alpha=0.5)

    # Labels and formatting
    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"Weighted Residual ($\sigma$)", fontsize=12)
    ax.set_title(
        f"Weighted Residuals\nMean = {np.mean(all_weighted_residuals):.3f}, Std = {np.std(all_weighted_residuals):.3f}",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(result.config.wavelength_range)

    return ax


def plot_diagnostic_summary(
    result: SEDReconstructionResult,
    figsize: tuple = (16, 12),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create comprehensive diagnostic plot with multiple panels.

    Parameters
    ----------
    result : SEDReconstructionResult
        Reconstruction result with all diagnostics.
    figsize : tuple
        Figure size (width, height) in inches.
    save_path : Path, optional
        Path to save the figure. If None, doesn't save.

    Returns
    -------
    plt.Figure
        Figure object with diagnostic panels.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Reconstructed spectrum
    ax1 = fig.add_subplot(gs[0, :])
    plot_reconstructed_spectrum(result, ax=ax1, show_measurements=True)

    # Panel 2: Raw residuals
    ax2 = fig.add_subplot(gs[1, 0])
    plot_residuals(result, ax=ax2)

    # Panel 3: Weighted residuals
    ax3 = fig.add_subplot(gs[1, 1])
    plot_weighted_residuals(result, ax=ax3)

    # Panel 4: Residual histogram
    ax4 = fig.add_subplot(gs[2, 0])
    all_weighted_residuals = []
    for band_data in result.band_data.values():
        model_flux = np.interp(band_data.wavelength_center, result.wavelength, result.flux)
        residuals = (band_data.flux - model_flux) / band_data.flux_error
        all_weighted_residuals.extend(residuals)

    ax4.hist(all_weighted_residuals, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax4.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    ax4.axvline(
        np.mean(all_weighted_residuals),
        color="orange",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean = {np.mean(all_weighted_residuals):.3f}",
    )
    ax4.set_xlabel("Weighted Residual ($\\sigma$)")
    ax4.set_ylabel("Count")
    ax4.set_title("Weighted Residual Distribution")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Panel 5: Quality metrics summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    metrics_text = f"""Quality Metrics:

$\\chi^2_\\nu$ = {result.validation_metrics.chi2_nu:.3f}
Solver: {result.solver_status}
Time: {result.solver_time:.2f} s

Configuration:
DIP Filters: {result.config.dip_filters}
DIP Depth: {result.config.dip_depth}
Regularization: {result.config.regularization_weight}
Learning Rate: {result.config.learning_rate}
Epochs: {result.config.epochs}
Device: {result.config.device}

Data:
Bands: {len(result.band_data)}
Total Observations: {sum(band.n_measurements for band in result.band_data.values())}
Resolution: {result.config.global_resolution}
"""

    ax5.text(
        0.1,
        0.9,
        metrics_text,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5),
    )

    # Overall title
    fig.suptitle("SED Reconstruction Diagnostic Summary", fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved diagnostic plot to {save_path}")

    return fig


def save_all_plots(
    result: SEDReconstructionResult,
    output_dir: Path,
    formats: List[str] = ["png", "pdf"],
) -> None:
    """
    Save all standard plots to files.

    Parameters
    ----------
    result : SEDReconstructionResult
        Reconstruction result to plot.
    output_dir : Path
        Directory to save plots.
    formats : List[str]
        File formats to save (e.g., ['png', 'pdf', 'svg']).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save diagnostic summary
    diag_fig = plot_diagnostic_summary(result)
    for fmt in formats:
        diag_path = output_dir / f"sed_diagnostic.{fmt}"
        diag_fig.savefig(diag_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved diagnostic plot to {diag_path}")

    # Save individual spectrum plot
    spectrum_fig, ax = plt.subplots(figsize=(12, 6))
    plot_reconstructed_spectrum(result, ax=ax)
    for fmt in formats:
        spec_path = output_dir / f"sed_spectrum.{fmt}"
        spectrum_fig.savefig(spec_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved spectrum plot to {spec_path}")

    plt.close("all")
