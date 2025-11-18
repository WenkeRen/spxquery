"""
Visualization functions for SED reconstruction diagnostics.

This module provides publication-quality plotting for reconstructed spectra,
residuals, and quality metrics.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.sparse as sp

from .reconstruction import BandReconstructionResult, SEDReconstructionResult
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


def plot_reconstructed_spectrum(
    result: BandReconstructionResult,
    original_data: Optional[BandData] = None,
    ax: Optional[plt.Axes] = None,
    show_original: bool = True,
) -> plt.Axes:
    """
    Plot reconstructed spectrum for a single band.

    Parameters
    ----------
    result : BandReconstructionResult
        Reconstruction result with wavelength and flux.
    original_data : BandData, optional
        Original narrow-band measurements to overlay as scatter points.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_original : bool
        Whether to show original measurements as scatter points.

    Returns
    -------
    plt.Axes
        Axes object with plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot reconstructed spectrum
    ax.plot(
        result.wavelength,
        result.flux,
        'k-',
        linewidth=1.5,
        label=f"Reconstructed ({result.band})",
        zorder=2,
    )

    # Overlay original measurements if provided
    if show_original and original_data is not None:
        ax.errorbar(
            original_data.wavelength_center,
            original_data.flux,
            yerr=original_data.flux_error,
            fmt='o',
            markersize=3,
            alpha=0.5,
            label="Original measurements",
            zorder=1,
        )

    # Labels
    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"$F_\lambda$ ($\mu$Jy)", fontsize=12)
    ax.set_title(
        f"Band {result.band} Reconstruction\n"
        f"continuum={result.lambda_vector[0]:.2e}, noise={result.lambda_vector[-1]:.2e}, "
        f"$\\chi^2_\\nu$={result.validation_metrics.chi_squared_reduced:.2f}",
        fontsize=11,
    )
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    return ax


def plot_residuals(
    result: BandReconstructionResult,
    ax_raw: Optional[plt.Axes] = None,
    ax_weighted: Optional[plt.Axes] = None,
) -> tuple:
    """
    Plot residual histograms (raw and weighted).

    Parameters
    ----------
    result : BandReconstructionResult
        Reconstruction result with validation metrics.
    ax_raw : plt.Axes, optional
        Axes for raw residuals histogram.
    ax_weighted : plt.Axes, optional
        Axes for weighted residuals histogram.

    Returns
    -------
    ax_raw : plt.Axes
        Raw residuals axes.
    ax_weighted : plt.Axes
        Weighted residuals axes.
    """
    metrics = result.validation_metrics

    # Create axes if not provided
    if ax_raw is None or ax_weighted is None:
        fig, (ax_raw, ax_weighted) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw residuals histogram
    ax_raw.hist(metrics.residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax_raw.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero')
    ax_raw.set_xlabel(r"Residual ($\mu$Jy)", fontsize=12)
    ax_raw.set_ylabel("Count", fontsize=12)
    ax_raw.set_title(
        f"Raw Residuals ({result.band})\n"
        f"Mean={metrics.residual_mean:.2f}, Std={metrics.residual_std:.2f}",
        fontsize=11,
    )
    ax_raw.legend(fontsize=10)
    ax_raw.grid(alpha=0.3)

    # Weighted residuals histogram
    ax_weighted.hist(
        metrics.weighted_residuals, bins=50, alpha=0.7, color='coral', edgecolor='black'
    )
    ax_weighted.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero')
    ax_weighted.set_xlabel(r"Weighted Residual ($\sigma$ units)", fontsize=12)
    ax_weighted.set_ylabel("Count", fontsize=12)
    ax_weighted.set_title(
        f"Weighted Residuals ({result.band})\n"
        f"Mean={metrics.weighted_residual_mean:.2f}, Std={metrics.weighted_residual_std:.2f}",
        fontsize=11,
    )
    ax_weighted.legend(fontsize=10)
    ax_weighted.grid(alpha=0.3)

    return ax_raw, ax_weighted




def plot_band_comparison(
    result: SEDReconstructionResult,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """
    Create multi-panel figure comparing all band reconstructions.

    Parameters
    ----------
    result : SEDReconstructionResult
        Full reconstruction result.
    figsize : tuple
        Figure size (width, height) in inches.

    Returns
    -------
    plt.Figure
        Figure object with 6 subplots (one per band).
    """
    n_bands = len(result.band_results)
    ncols = 3
    nrows = int(np.ceil(n_bands / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, (band, band_result) in enumerate(sorted(result.band_results.items())):
        ax = axes[idx]

        # Plot reconstructed spectrum
        ax.plot(
            band_result.wavelength,
            band_result.flux,
            'k-',
            linewidth=1.5,
        )

        ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=10)
        ax.set_ylabel(r"$F_\lambda$ ($\mu$Jy)", fontsize=10)
        ax.set_title(
            f"{band}: $\\chi^2_\\nu$={band_result.validation_metrics.chi_squared_reduced:.2f}",
            fontsize=10,
        )
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n_bands, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(
        f"Band-by-Band Reconstructions: {result.source_name}",
        fontsize=14,
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    return fig


def plot_diagnostic_summary(
    result: SEDReconstructionResult,
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """
    Create comprehensive diagnostic figure with multiple panels.

    Layout:
    - Top row: Individual band spectra (3 bands)
    - Middle row: Individual band spectra (remaining bands)
    - Bottom row: Residuals for first band (example)

    Parameters
    ----------
    result : SEDReconstructionResult
        Full reconstruction result.
    figsize : tuple
        Figure size (width, height) in inches.

    Returns
    -------
    plt.Figure
        Figure object with diagnostic plots.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Top row: Band comparison (first 3 bands)
    band_list = sorted(result.band_results.keys())
    for idx, band in enumerate(band_list[:3]):
        ax = fig.add_subplot(gs[0, idx])
        band_result = result.band_results[band]
        ax.plot(band_result.wavelength, band_result.flux, 'k-', linewidth=1)
        ax.set_title(f"{band}", fontsize=10)
        ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=9)
        ax.set_ylabel(r"$F_\lambda$ ($\mu$Jy)", fontsize=9)
        ax.grid(alpha=0.3)

    # Middle row: Remaining bands
    for idx, band in enumerate(band_list[3:6]):
        ax = fig.add_subplot(gs[1, idx])
        band_result = result.band_results[band]
        ax.plot(band_result.wavelength, band_result.flux, 'k-', linewidth=1)
        ax.set_title(f"{band}", fontsize=10)
        ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=9)
        ax.set_ylabel(r"$F_\lambda$ ($\mu$Jy)", fontsize=9)
        ax.grid(alpha=0.3)

    # Bottom row: Residuals for first band
    if band_list:
        first_band = band_list[0]
        first_result = result.band_results[first_band]

        ax_raw = fig.add_subplot(gs[2, 0])
        ax_weighted = fig.add_subplot(gs[2, 1])
        plot_residuals(first_result, ax_raw=ax_raw, ax_weighted=ax_weighted)

        # Chi-squared summary text
        ax_text = fig.add_subplot(gs[2, 2])
        ax_text.axis('off')

        summary_text = "Reconstruction Quality\n" + "="*25 + "\n\n"
        for band in band_list:
            metrics = result.band_results[band].validation_metrics
            summary_text += (
                f"{band}:\n"
                f"  $\\chi^2_\\nu$ = {metrics.chi_squared_reduced:.3f}\n"
                f"  dof = {metrics.degrees_of_freedom}\n\n"
            )

        ax_text.text(
            0.1, 0.9, summary_text,
            transform=ax_text.transAxes,
            fontsize=9,
            verticalalignment='top',
            family='monospace',
        )

    fig.suptitle(
        f"SED Reconstruction Diagnostics: {result.source_name}",
        fontsize=14,
        fontweight='bold',
    )

    return fig


def save_all_plots(
    result: SEDReconstructionResult,
    output_dir: Path,
    prefix: str = "sed",
) -> List[Path]:
    """
    Generate and save all diagnostic plots.

    Parameters
    ----------
    result : SEDReconstructionResult
        Full reconstruction result.
    output_dir : Path
        Output directory for plot files.
    prefix : str
        Filename prefix.

    Returns
    -------
    List[Path]
        List of saved plot file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Diagnostic summary
    fig_summary = plot_diagnostic_summary(result)
    path_summary = output_dir / f"{prefix}_diagnostic_summary.png"
    fig_summary.savefig(path_summary, dpi=150, bbox_inches='tight')
    plt.close(fig_summary)
    saved_files.append(path_summary)
    logger.info(f"Saved diagnostic summary to {path_summary}")

    # Band comparison
    fig_comparison = plot_band_comparison(result)
    path_comparison = output_dir / f"{prefix}_band_comparison.png"
    fig_comparison.savefig(path_comparison, dpi=150, bbox_inches='tight')
    plt.close(fig_comparison)
    saved_files.append(path_comparison)
    logger.info(f"Saved band comparison to {path_comparison}")

    logger.info(f"Saved {len(saved_files)} plot files to {output_dir}")

    return saved_files
