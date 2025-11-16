"""
Main orchestrator for SED reconstruction from SPHEREx narrow-band photometry.

This module provides the high-level SEDReconstructor class that coordinates
data loading, matrix construction, optimization, validation, and multi-band stitching.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import yaml

from .config import SEDConfig
from .data_loader import load_all_bands, BandData
from .matrices import build_all_matrices
from .solver import ReconstructionResult, reconstruct_single_band
from .validation import assess_reconstruction_quality, ValidationMetrics
from .tuning import tune_and_reconstruct, TuningResult
from .stitching import stitch_all_bands, StitchedSpectrum

logger = logging.getLogger(__name__)


@dataclass
class BandReconstructionResult:
    """
    Complete reconstruction result for a single detector band using wavelet regularization.

    Attributes
    ----------
    band : str
        Band identifier (e.g., 'D1').
    wavelength : np.ndarray
        Wavelength grid in microns, shape (N,).
    flux : np.ndarray
        Reconstructed flux density in microJansky, shape (N,).
    lambda_low : float
        Approximation coefficient regularization weight used (continuum).
    lambda_detail : float
        Detail coefficient regularization weight used (noise suppression).
    wavelet_info : dict
        Wavelet decomposition information (level, coefficient counts).
    auto_tuned : bool
        Whether hyperparameters were automatically tuned.
    solver_status : str
        CVXPY solver status.
    solver_time : float
        Solver wall-clock time in seconds.
    validation_metrics : ValidationMetrics
        Quality assessment metrics.
    tuning_result : Optional[TuningResult]
        Grid search results if auto_tuned=True, else None.
    """

    band: str
    wavelength: np.ndarray
    flux: np.ndarray
    lambda_low: float
    lambda_detail: float
    wavelet_info: Dict[str, int]
    auto_tuned: bool
    solver_status: str
    solver_time: float
    validation_metrics: ValidationMetrics
    tuning_result: Optional[TuningResult] = None


@dataclass
class SEDReconstructionResult:
    """
    Complete reconstruction result for all bands and stitched spectrum.

    Attributes
    ----------
    source_name : str
        Source identifier.
    source_ra : float
        Source right ascension in degrees.
    source_dec : float
        Source declination in degrees.
    reconstruction_date : str
        ISO format timestamp of reconstruction.
    config : SEDConfig
        Configuration used for reconstruction.
    band_results : Dict[str, BandReconstructionResult]
        Individual band reconstruction results.
    stitched_spectrum : Optional[StitchedSpectrum]
        Stitched multi-band spectrum, or None if stitching disabled.
    """

    source_name: str
    source_ra: Optional[float]
    source_dec: Optional[float]
    reconstruction_date: str
    config: SEDConfig
    band_results: Dict[str, BandReconstructionResult]
    stitched_spectrum: Optional[StitchedSpectrum] = None

    def to_csv(self, output_dir: Path, filename: str = "sed_reconstruction.csv") -> Path:
        """
        Save reconstructed spectrum to CSV file.

        If stitching is enabled, saves the stitched spectrum.
        Otherwise, saves individual band spectra concatenated.

        Parameters
        ----------
        output_dir : Path
            Output directory.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to written CSV file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename

        if self.stitched_spectrum is not None:
            # Save stitched spectrum
            df = pd.DataFrame({
                "wavelength": self.stitched_spectrum.wavelength,
                "flux": self.stitched_spectrum.flux,
                "band": self.stitched_spectrum.band_labels,
            })
        else:
            # Concatenate individual band spectra
            data_list = []
            for band, result in self.band_results.items():
                band_df = pd.DataFrame({
                    "wavelength": result.wavelength,
                    "flux": result.flux,
                    "band": band,
                })
                data_list.append(band_df)
            df = pd.concat(data_list, ignore_index=True)
            df = df.sort_values("wavelength").reset_index(drop=True)

        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Saved reconstructed spectrum to {filepath}")

        return filepath

    def to_yaml(self, output_dir: Path, filename: str = "sed_metadata.yaml") -> Path:
        """
        Save reconstruction metadata to YAML file.

        Includes:
        - Source information
        - Hyperparameters (lambda_low, lambda_detail)
        - Wavelet decomposition information
        - Auto-tuning status
        - Normalization factors (if stitched)
        - Quality metrics (chi-squared)

        Parameters
        ----------
        output_dir : Path
            Output directory.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to written YAML file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename

        # Build metadata dictionary
        metadata = {
            "source_name": self.source_name,
            "source_ra": self.source_ra,
            "source_dec": self.source_dec,
            "reconstruction_date": self.reconstruction_date,
            "auto_tuned": self.config.auto_tune,
            "bands_reconstructed": list(self.band_results.keys()),
            "wavelet_family": self.config.wavelet_family,
        }

        # Add per-band hyperparameters and quality metrics
        for band, result in self.band_results.items():
            metadata[f"{band}_lambda_low"] = result.lambda_low
            metadata[f"{band}_lambda_detail"] = result.lambda_detail
            metadata[f"{band}_wavelet_level"] = result.wavelet_info.get("level", None)
            metadata[f"{band}_chi_squared_reduced"] = result.validation_metrics.chi_squared_reduced
            metadata[f"{band}_solver_time"] = result.solver_time

        # Add stitching information
        if self.stitched_spectrum is not None:
            metadata["stitched"] = True
            metadata["normalization_factors"] = self.stitched_spectrum.normalization_factors
            metadata["wavelength_ranges"] = {
                band: {"min": wrange[0], "max": wrange[1]}
                for band, wrange in self.stitched_spectrum.wavelength_ranges.items()
            }
        else:
            metadata["stitched"] = False

        # Write YAML
        with open(filepath, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved reconstruction metadata to {filepath}")

        return filepath

    def save_all(self, output_dir: Path) -> Tuple[Path, Path]:
        """
        Save both spectrum CSV and metadata YAML.

        Parameters
        ----------
        output_dir : Path
            Output directory.

        Returns
        -------
        csv_path : Path
            Path to spectrum CSV file.
        yaml_path : Path
            Path to metadata YAML file.
        """
        csv_path = self.to_csv(output_dir)
        yaml_path = self.to_yaml(output_dir)
        return csv_path, yaml_path


class SEDReconstructor:
    """
    Main class for orchestrating SED reconstruction from lightcurve data.

    This class coordinates all steps of the reconstruction pipeline:
    1. Load and validate lightcurve CSV
    2. For each band:
       a. Build measurement matrix H and smoothness operator D2
       b. Either use manual hyperparameters or run auto-tuning
       c. Solve optimization problem
       d. Assess reconstruction quality
    3. Optionally stitch bands into continuous spectrum

    Parameters
    ----------
    config : SEDConfig
        Configuration for reconstruction.

    Examples
    --------
    >>> config = SEDConfig(auto_tune=True, stitch_bands=True)
    >>> reconstructor = SEDReconstructor(config)
    >>> result = reconstructor.reconstruct_from_csv("lightcurve.csv")
    >>> result.save_all("output/")
    """

    def __init__(self, config: SEDConfig):
        """
        Initialize reconstructor with configuration.

        Parameters
        ----------
        config : SEDConfig
            Reconstruction configuration.
        """
        self.config = config
        logger.info("Initialized SEDReconstructor")
        logger.info(f"  Auto-tuning: {config.auto_tune}")
        logger.info(f"  Stitch bands: {config.stitch_bands}")
        logger.info(f"  Resolution: {config.resolution_samples} samples")

    def reconstruct_from_csv(
        self, csv_path: Path, metadata: Optional[Dict] = None
    ) -> SEDReconstructionResult:
        """
        Reconstruct SED from lightcurve CSV file.

        This is the main entry point for the reconstruction pipeline.

        Parameters
        ----------
        csv_path : Path
            Path to lightcurve.csv file from SPXQuery processing.
        metadata : Dict, optional
            Additional metadata to include in result.

        Returns
        -------
        SEDReconstructionResult
            Complete reconstruction result.
        """
        csv_path = Path(csv_path)
        logger.info(f"Starting SED reconstruction from {csv_path}")

        # Load all bands
        band_data_dict, file_metadata = load_all_bands(csv_path, self.config)

        # Merge metadata
        if metadata is not None:
            file_metadata.update(metadata)

        # Reconstruct each band
        band_results = {}
        for band, band_data in band_data_dict.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Reconstructing {band}")
            logger.info(f"{'='*60}")

            band_result = self._reconstruct_single_band(band_data)
            band_results[band] = band_result

        # Stitch bands if requested
        stitched_spectrum = None
        if self.config.stitch_bands and len(band_results) > 1:
            logger.info(f"\n{'='*60}")
            logger.info("Stitching bands")
            logger.info(f"{'='*60}")

            band_spectra = {
                band: (result.wavelength, result.flux)
                for band, result in band_results.items()
            }
            stitched_spectrum = stitch_all_bands(band_spectra)

        # Package results
        result = SEDReconstructionResult(
            source_name=file_metadata.get("source_name", "unknown"),
            source_ra=file_metadata.get("source_ra"),
            source_dec=file_metadata.get("source_dec"),
            reconstruction_date=datetime.utcnow().isoformat(),
            config=self.config,
            band_results=band_results,
            stitched_spectrum=stitched_spectrum,
        )

        logger.info("\n" + "="*60)
        logger.info("Reconstruction complete!")
        logger.info(f"  Bands reconstructed: {list(band_results.keys())}")
        logger.info(f"  Stitched: {stitched_spectrum is not None}")
        logger.info("="*60)

        return result

    def _reconstruct_single_band(self, band_data: BandData) -> BandReconstructionResult:
        """
        Reconstruct spectrum for a single band.

        Parameters
        ----------
        band_data : BandData
            Prepared photometry data for one band.

        Returns
        -------
        BandReconstructionResult
            Complete reconstruction result for this band.
        """
        # Build matrices (now includes wavelet matrices and edge padding info)
        H, Psi_approx, Psi_detail, weights, wavelength_grid, level_info, edge_info = build_all_matrices(
            band_data, self.config
        )

        # Extract measurement data
        y = band_data.flux

        # Decide whether to auto-tune or use manual hyperparameters
        if self.config.auto_tune:
            logger.info("Auto-tuning enabled: running grid search")
            tuning_result, spectrum = tune_and_reconstruct(
                y, H, Psi_approx, Psi_detail, weights, wavelength_grid, level_info, self.config, edge_info
            )
            lambda_low_final = tuning_result.optimal_lambda_low
            lambda_detail_final = tuning_result.optimal_lambda_detail
            auto_tuned = True

            # Reconstruct again on full data for final result object
            # (tune_and_reconstruct already did this, but we need the full result object)
            final_result = reconstruct_single_band(
                y, H, Psi_approx, Psi_detail, weights, wavelength_grid, level_info, self.config,
                lambda_low=lambda_low_final, lambda_detail=lambda_detail_final, edge_info=edge_info
            )

        else:
            logger.info(
                f"Using manual hyperparameters: "
                f"lambda_low={self.config.lambda_low:.2e}, lambda_detail={self.config.lambda_detail:.2e}"
            )
            final_result = reconstruct_single_band(
                y, H, Psi_approx, Psi_detail, weights, wavelength_grid, level_info, self.config,
                edge_info=edge_info
            )
            lambda_low_final = final_result.lambda_low
            lambda_detail_final = final_result.lambda_detail
            tuning_result = None
            auto_tuned = False

        if not final_result.success:
            raise RuntimeError(
                f"Reconstruction failed for {band_data.band}: {final_result.solver_status}"
            )

        # Assess quality
        validation_metrics = assess_reconstruction_quality(
            y, H, final_result.spectrum, weights
        )

        # Trim spectrum to detector range if edge padding was used
        if edge_info is not None and "trim_start" in edge_info:
            trim_start = edge_info["trim_start"]
            trim_end = edge_info["trim_end"]
            wavelength_trimmed = final_result.wavelength_grid[trim_start:trim_end]
            flux_trimmed = final_result.spectrum[trim_start:trim_end]

            logger.info(
                f"Trimmed spectrum from {len(final_result.spectrum)} to {len(flux_trimmed)} points "
                f"(removed {trim_start} edge pixels on each side)"
            )
        else:
            # No edge padding, use full spectrum
            wavelength_trimmed = final_result.wavelength_grid
            flux_trimmed = final_result.spectrum

        # Package band result
        band_result = BandReconstructionResult(
            band=band_data.band,
            wavelength=wavelength_trimmed,
            flux=flux_trimmed,
            lambda_low=lambda_low_final,
            lambda_detail=lambda_detail_final,
            wavelet_info=final_result.wavelet_info,
            auto_tuned=auto_tuned,
            solver_status=final_result.solver_status,
            solver_time=final_result.solver_time,
            validation_metrics=validation_metrics,
            tuning_result=tuning_result,
        )

        return band_result


def reconstruct_sed_from_csv(
    csv_path: Path,
    config: Optional[SEDConfig] = None,
    output_dir: Optional[Path] = None,
) -> SEDReconstructionResult:
    """
    Convenience function to reconstruct SED from CSV with optional auto-save.

    Parameters
    ----------
    csv_path : Path
        Path to lightcurve.csv file.
    config : SEDConfig, optional
        Configuration. If None, uses default SEDConfig.
    output_dir : Path, optional
        If provided, saves results to this directory.

    Returns
    -------
    SEDReconstructionResult
        Reconstruction result.
    """
    if config is None:
        config = SEDConfig()

    reconstructor = SEDReconstructor(config)
    result = reconstructor.reconstruct_from_csv(csv_path)

    if output_dir is not None:
        result.save_all(output_dir)

    return result
