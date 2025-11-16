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
        f"$\\lambda_1$={result.lambda1:.2e}, $\\lambda_2$={result.lambda2:.2e}, "
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


def plot_stitched_spectrum(
    result: SEDReconstructionResult,
    ax: Optional[plt.Axes] = None,
    colormap: str = "rainbow",
) -> plt.Axes:
    """
    Plot stitched multi-band spectrum.

    Parameters
    ----------
    result : SEDReconstructionResult
        Full reconstruction result with stitched spectrum.
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    colormap : str
        Matplotlib colormap name for coloring bands.

    Returns
    -------
    plt.Axes
        Axes object with plot.
    """
    if result.stitched_spectrum is None:
        raise ValueError("No stitched spectrum available in result")

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    stitched = result.stitched_spectrum

    # Color code by band
    cmap = plt.get_cmap(colormap)
    band_colors = {
        "D1": cmap(0.0),
        "D2": cmap(0.2),
        "D3": cmap(0.4),
        "D4": cmap(0.6),
        "D5": cmap(0.8),
        "D6": cmap(1.0),
    }

    # Plot each band segment
    for band in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        if band not in result.band_results:
            continue

        mask = stitched.band_labels == band
        if not np.any(mask):
            continue

        ax.plot(
            stitched.wavelength[mask],
            stitched.flux[mask],
            '-',
            color=band_colors.get(band, 'black'),
            linewidth=1.5,
            label=f"{band} ($\\eta$={stitched.normalization_factors[band]:.3f})",
        )

    # Labels
    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"$F_\lambda$ ($\mu$Jy)", fontsize=12)
    ax.set_title(
        f"Stitched SED: {result.source_name}\n"
        f"{len(result.band_results)} bands, "
        f"{len(stitched.wavelength)} wavelength points",
        fontsize=11,
    )
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    return ax


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
    - Top row: Stitched spectrum (full width)
    - Middle row: Individual band spectra (2x3 grid)
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

    # Top row: Stitched spectrum (if available)
    if result.stitched_spectrum is not None:
        ax_stitched = fig.add_subplot(gs[0, :])
        plot_stitched_spectrum(result, ax=ax_stitched)

    # Middle row: Band comparison (first 3 bands)
    band_list = sorted(result.band_results.keys())
    for idx, band in enumerate(band_list[:3]):
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

    # Stitched spectrum (if available)
    if result.stitched_spectrum is not None:
        fig_stitched, ax_stitched = plt.subplots(figsize=(14, 6))
        plot_stitched_spectrum(result, ax=ax_stitched)
        path_stitched = output_dir / f"{prefix}_stitched_spectrum.png"
        fig_stitched.savefig(path_stitched, dpi=150, bbox_inches='tight')
        plt.close(fig_stitched)
        saved_files.append(path_stitched)
        logger.info(f"Saved stitched spectrum to {path_stitched}")

    logger.info(f"Saved {len(saved_files)} plot files to {output_dir}")

    return saved_files
