"""
Main orchestrator for SED reconstruction from SPHEREx narrow-band photometry.

This module provides the high-level SEDReconstructor class that coordinates
data loading, matrix construction, optimization, validation, and multi-band stitching.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import yaml

from .config import SEDConfig, DETECTOR_WAVELENGTH_RANGES
from .data_loader import load_all_bands, BandData
from .matrices import build_all_matrices, build_pixel_observation_dataset, build_measurement_matrix, build_weight_vector
from .solver import reconstruct_single_band
from .solver_torch import solve_global_reconstruction
from .validation import assess_reconstruction_quality, ValidationMetrics
from .tuning import tune_and_reconstruct, TuningResult

logger = logging.getLogger(__name__)


@dataclass
class BandReconstructionResult:
    """
    Complete reconstruction result for a single detector band.
    
    Attributes
    ----------
    band : str
        Band identifier (e.g., 'D1').
    wavelength : np.ndarray
        Wavelength grid in microns.
    flux : np.ndarray
        Reconstructed flux density in microJansky.
    lambda_vector : np.ndarray
        Regularization weights used.
    wavelet_info : dict
        Wavelet decomposition info (if applicable).
    auto_tuned : bool
        Whether hyperparameters were tuned.
    solver_status : str
        Solver status.
    solver_time : float
        Solver time in seconds.
    validation_metrics : ValidationMetrics
        Quality assessment metrics.
    tuning_result : Optional[TuningResult]
        Tuning details.
    per_scale_penalties : np.ndarray
        Regularization penalties.
    total_penalty : float
        Total penalty.
    """

    band: str
    wavelength: np.ndarray
    flux: np.ndarray
    lambda_vector: np.ndarray
    wavelet_info: Dict[str, int]
    auto_tuned: bool
    solver_status: str
    solver_time: float
    validation_metrics: ValidationMetrics
    tuning_result: Optional[TuningResult] = None
    per_scale_penalties: np.ndarray = None
    total_penalty: float = None

    def __post_init__(self):
        """Set default values for optional fields."""
        if self.per_scale_penalties is None and self.lambda_vector is not None:
            self.per_scale_penalties = np.zeros(len(self.lambda_vector))
        if self.total_penalty is None:
            self.total_penalty = 0.0


@dataclass
class SEDReconstructionResult:
    """
    Complete reconstruction result for all bands.

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
    """

    source_name: str
    source_ra: Optional[float]
    source_dec: Optional[float]
    reconstruction_date: str
    config: SEDConfig
    band_results: Dict[str, BandReconstructionResult]

    def to_csv(self, output_dir: Path, filename: str = "sed_reconstruction.csv") -> Path:
        """
        Save reconstructed spectrum to CSV file.

        Saves individual band spectra concatenated by wavelength.

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

        # Concatenate individual band spectra
        data_list = []
        # Sort bands for consistent output
        sorted_bands = sorted(self.band_results.keys())
        
        for band in sorted_bands:
            result = self.band_results[band]
            band_df = pd.DataFrame({
                "wavelength": result.wavelength,
                "flux": result.flux,
                "band": band,
            })
            data_list.append(band_df)
            
        if data_list:
            df = pd.concat(data_list, ignore_index=True)
            df = df.sort_values("wavelength").reset_index(drop=True)
            # Save to CSV
            df.to_csv(filepath, index=False)
            logger.info(f"Saved reconstructed spectrum to {filepath}")
        else:
            logger.warning("No band results to save to CSV")

        return filepath

    def to_yaml(self, output_dir: Path, filename: str = "sed_metadata.yaml") -> Path:
        """
        Save reconstruction metadata to YAML file.

        Includes:
        - Source information
        - Hyperparameters (lambda_low, lambda_detail)
        - Wavelet decomposition information
        - Auto-tuning status
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
            "solver_type": self.config.solver_type,
            "bands_reconstructed": list(self.band_results.keys()),
            "wavelet_family": self.config.wavelet_family,
        }

        # Add per-band hyperparameters and quality metrics
        for band, result in self.band_results.items():
            # Old-style DWT parameters (backward compatibility)
            if hasattr(result, 'lambda_vector') and result.lambda_vector is not None and len(result.lambda_vector) > 0:
                metadata[f"{band}_lambda_low"] = float(result.lambda_vector[0])
                metadata[f"{band}_lambda_detail"] = float(np.mean(result.lambda_vector[1:]))
                metadata[f"{band}_lambda_vector"] = result.lambda_vector.tolist()
                metadata[f"{band}_num_swt_operators"] = len(result.lambda_vector)
            else:
                # Defaults if vector missing (e.g. torch solver might not use lambda_vector same way)
                metadata[f"{band}_lambda_vector"] = []

            # SWT penalty information
            if hasattr(result, 'total_penalty') and result.total_penalty is not None:
                metadata[f"{band}_total_penalty"] = float(result.total_penalty)
            if hasattr(result, 'per_scale_penalties') and result.per_scale_penalties is not None:
                metadata[f"{band}_per_scale_penalties"] = result.per_scale_penalties.tolist()

            if result.wavelet_info:
                metadata[f"{band}_wavelet_level"] = result.wavelet_info.get("level", None)
                
            if result.validation_metrics:
                metadata[f"{band}_chi_squared_reduced"] = float(result.validation_metrics.chi_squared_reduced)
                
            metadata[f"{band}_solver_time"] = float(result.solver_time)

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
       a. Build measurement matrix H and SWT operators
       b. Either use manual hyperparameters or run auto-tuning
       c. Solve optimization problem
       d. Assess reconstruction quality

    Parameters
    ----------
    config : SEDConfig
        Configuration for reconstruction.

    Examples
    --------
    >>> config = SEDConfig(auto_tune=True)
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
        logger.info(f"  Solver: {config.solver_type}")
        if config.solver_type == "cvxpy":
            logger.info(f"  Auto-tuning: {config.auto_tune}")
            logger.info(f"  Resolution: {config.resolution_samples} samples")
        else:
            logger.info(f"  Global Resolution: {config.global_resolution} samples")
            logger.info(f"  Device: {config.device}")

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

        band_results = {}

        # Branch based on solver type
        if self.config.solver_type == "torch":
            logger.info("Using PyTorch solver for global reconstruction")
            
            # 1. Build global pixel observation dataset
            start_time = datetime.now()
            dataset = build_pixel_observation_dataset(band_data_dict, self.config)
            
            # 2. Solve global reconstruction
            global_spectrum_tensor = solve_global_reconstruction(dataset, self.config)
            global_spectrum = global_spectrum_tensor.numpy()
            
            solver_time = (datetime.now() - start_time).total_seconds()
            
            # 3. Calculate GLOBAL validation metrics (since fitting was global)
            # Reconstruct global H and y/weights to compute a single global chi-squared
            # This avoids re-computing per-band chi-squared which is less meaningful for global fit
            
            global_wavelength_grid = dataset.global_wavelength_grid.numpy() # Should match config.global_resolution
            
            # Collect all per-band matrices to form global system for validation
            all_H_bands = []
            all_fluxes = []
            all_weights = []
            
            # Sort bands to ensure deterministic order (though build_pixel_observation_dataset does its own thing,
            # we need to be consistent here for validation)
            sorted_band_keys = sorted(band_data_dict.keys())
            
            for band in sorted_band_keys:
                band_data = band_data_dict[band]
                H_band = build_measurement_matrix(band_data, global_wavelength_grid, self.config)
                weights_band = build_weight_vector(band_data, self.config)
                
                all_H_bands.append(H_band)
                all_fluxes.append(band_data.flux)
                all_weights.append(weights_band)
                
            # Stack into global system
            if all_H_bands:
                import scipy.sparse as sp
                H_global = sp.vstack(all_H_bands)
                y_global = np.concatenate(all_fluxes)
                w_global = np.concatenate(all_weights)
                
                # Assess global quality
                global_metrics = assess_reconstruction_quality(
                    y_global, H_global, global_spectrum, w_global
                )
                logger.info(f"Global reconstruction quality: chi2_red={global_metrics.chi_squared_reduced:.3f}")
            else:
                # Should not happen if band_data_dict is not empty
                global_metrics = None

            # 4. Package results per band
            for band, band_data in band_data_dict.items():
                # Get detector range
                lambda_min, lambda_max = self.config.wavelength_range
                l_min, l_max = DETECTOR_WAVELENGTH_RANGES.get(band, (lambda_min, lambda_max))
                
                # Find indices in global grid corresponding to this band
                idx_start = np.searchsorted(global_wavelength_grid, l_min)
                idx_end = np.searchsorted(global_wavelength_grid, l_max)
                
                # Slice global spectrum
                wavelength_band = global_wavelength_grid[idx_start:idx_end]
                flux_band = global_spectrum[idx_start:idx_end]
                
                # Use the GLOBAL metrics for every band result, as they share the same model
                band_result = BandReconstructionResult(
                    band=band,
                    wavelength=wavelength_band,
                    flux=flux_band,
                    lambda_vector=np.array([self.config.regularization_weight]), # Placeholder
                    wavelet_info={},
                    auto_tuned=False,
                    solver_status="optimal" if global_metrics and np.isfinite(global_metrics.chi_squared) else "error",
                    solver_time=solver_time,
                    validation_metrics=global_metrics # Shared global metrics
                )
                band_results[band] = band_result
                
        else:
            # Existing CVXPY per-band workflow
            for band, band_data in band_data_dict.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"Reconstructing {band}")
                logger.info(f"{'='*60}")

                band_result = self._reconstruct_single_band(band_data)
                band_results[band] = band_result

        # Package results
        result = SEDReconstructionResult(
            source_name=file_metadata.get("source_name", "unknown"),
            source_ra=file_metadata.get("source_ra"),
            source_dec=file_metadata.get("source_dec"),
            reconstruction_date=datetime.now().isoformat(),
            config=self.config,
            band_results=band_results,
        )

        logger.info("\n" + "="*60)
        logger.info("Reconstruction complete!")
        logger.info(f"  Bands reconstructed: {list(band_results.keys())}")
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
        # Build matrices (now includes SWT matrices, edge padding info, and spatial weights)
        H, Psi_operators, weights, wavelength_grid, level_info, edge_info, spatial_weights = build_all_matrices(
            band_data, self.config
        )

        # Extract measurement data
        y = band_data.flux

        # Decide whether to auto-tune or use manual hyperparameters
        if self.config.auto_tune:
            logger.info("Auto-tuning enabled: running grid search")
            tuning_result, _ = tune_and_reconstruct(
                y, H, Psi_operators, weights, wavelength_grid, level_info, self.config, edge_info
            )
            lambda_vector_final = tuning_result.optimal_lambda_vector
            auto_tuned = True

            # Reconstruct again on full data for final result object
            # (tune_and_reconstruct already did this, but we need the full result object)
            final_result = reconstruct_single_band(
                y, H, Psi_operators, weights, wavelength_grid, level_info, self.config,
                lambda_vector=lambda_vector_final, edge_info=edge_info, spatial_weights=spatial_weights
            )

        else:
            logger.info(
                f"Using manual SWT hyperparameters: "
                f"continuum={self.config.lambda_continuum:.2e}, noise={self.config.lambda_noise:.2e}"
            )
            final_result = reconstruct_single_band(
                y, H, Psi_operators, weights, wavelength_grid, level_info, self.config,
                edge_info=edge_info, spatial_weights=spatial_weights
            )
            lambda_vector_final = final_result.lambda_vector
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
            lambda_vector=lambda_vector_final,
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
