"""
Main orchestrator for SED reconstruction from SPHEREx narrow-band photometry.

This module provides the high-level SEDReconstructor class that coordinates
data loading, global dataset construction, PyTorch-based Deep Image Prior optimization,
and validation for unified spectral reconstruction across all SPHEREx detector bands.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from .config import SEDConfig
from .data_loader import BandData, load_all_bands
from .matrices import build_global_observation_data
from .solver_torch import solve_global_reconstruction
from .validation import ValidationMetrics, assess_reconstruction_quality

logger = logging.getLogger(__name__)


@dataclass
class SEDReconstructionResult:
    """
    Complete reconstruction result for global SED reconstruction.

    Attributes
    ----------
    wavelength : np.ndarray
        Global wavelength grid in microns.
    flux : np.ndarray
        Reconstructed flux density in microJansky.
    config : SEDConfig
        Configuration used for reconstruction.
    solver_status : str
        PyTorch solver status.
    solver_time : float
        Solver time in seconds.
    validation_metrics : ValidationMetrics
        Quality assessment metrics.
    metadata : dict
        Reconstruction metadata including timestamps and parameters.
    band_data : Dict[str, BandData]
        Input photometry data per band.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    config: SEDConfig
    solver_status: str
    solver_time: float
    validation_metrics: ValidationMetrics
    metadata: Dict[str, any]
    band_data: Dict[str, BandData]

    def save_all(self, output_dir: Path) -> None:
        """
        Save reconstruction results to files.

        Parameters
        ----------
        output_dir : Path
            Directory to save results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save reconstructed spectrum
        spectrum_df = pd.DataFrame(
            {
                "wavelength_microns": self.wavelength,
                "flux_microjansky": self.flux,
            }
        )
        spectrum_path = output_dir / "sed_reconstruction.csv"
        spectrum_df.to_csv(spectrum_path, index=False)
        logger.info(f"Saved reconstructed spectrum to {spectrum_path}")

        # Save metadata
        metadata_path = output_dir / "sed_metadata.yaml"
        with open(metadata_path, "w") as f:
            yaml.dump(self.metadata, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved reconstruction metadata to {metadata_path}")

    def to_dict(self) -> Dict[str, any]:
        """Convert result to dictionary for serialization."""
        return {
            "wavelength": self.wavelength.tolist(),
            "flux": self.flux.tolist(),
            "config": self.config.to_dict(),
            "solver_status": self.solver_status,
            "solver_time": self.solver_time,
            "metadata": self.metadata,
        }


class SEDReconstructor:
    """
    Main orchestrator for SED reconstruction using PyTorch Deep Image Prior.

    This class provides a high-level interface for reconstructing high-resolution
    spectra from SPHEREx narrow-band photometry using global optimization
    with Continuous Wavelet Transform regularization.
    """

    def __init__(self, config: SEDConfig):
        """
        Initialize the reconstructor.

        Parameters
        ----------
        config : SEDConfig
            Configuration for reconstruction.
        """
        self.config = config
        logger.info(f"Initialized SEDReconstructor with device='{config.device}'")

    def reconstruct_from_csv(
        self,
        csv_path: Path,
        metadata: Optional[Dict[str, any]] = None,
    ) -> SEDReconstructionResult:
        """
        Reconstruct SED from CSV file containing SPHEREx photometry.

        Parameters
        ----------
        csv_path : Path
            Path to CSV file with photometry data.
        metadata : Optional[Dict[str, any]]
            Additional metadata to include in results.

        Returns
        -------
        SEDReconstructionResult
            Complete reconstruction result.
        """
        logger.info(f"Starting SED reconstruction from {csv_path}")

        # Load photometry data
        all_band_data, _ = load_all_bands(csv_path, self.config)
        if not all_band_data:
            raise ValueError("No valid photometry data found in CSV file")

        logger.info(f"Loaded data for {len(all_band_data)} bands: {list(all_band_data.keys())}")

        # Build global dataset
        global_dataset = build_global_observation_data(all_band_data, self.config)

        # Solve using PyTorch Deep Image Prior
        logger.info("Starting PyTorch Deep Image Prior optimization...")
        result_spectrum, solver_status, solver_time = solve_global_reconstruction(global_dataset, self.config)

        # Assess reconstruction quality
        # Convert sparse matrix to scipy csr format for validation
        H_sparse = sp.csr_matrix(
            (global_dataset.H_values.cpu().numpy(), global_dataset.H_indices.cpu().numpy()),
            shape=global_dataset.H_shape,
        )
        validation_metrics = assess_reconstruction_quality(
            global_dataset.observations.cpu().numpy(),
            H_sparse,
            result_spectrum.cpu().numpy(),
            global_dataset.weights.cpu().numpy(),
        )

        # Create reconstruction metadata
        reconstruction_metadata = {
            "timestamp": datetime.now().isoformat(),
            "csv_path": str(csv_path),
            "solver_type": "torch",
            "solver_status": solver_status,
            "solver_time_seconds": solver_time,
            "global_resolution": self.config.global_resolution,
            "wavelength_range": self.config.wavelength_range,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "regularization_weight": self.config.regularization_weight,
            "cwt_scales": self.config.cwt_scales,
            "n_bands": len(all_band_data),
            "bands": list(all_band_data.keys()),
            "total_observations": sum(band.n_measurements for band in all_band_data.values()),
        }

        # Add user-provided metadata
        if metadata:
            reconstruction_metadata.update(metadata)

        # Convert results to numpy arrays
        wavelength_grid = global_dataset.global_wavelength_grid.cpu().numpy()
        flux_spectrum = result_spectrum.cpu().numpy()

        # Create result object
        result = SEDReconstructionResult(
            wavelength=wavelength_grid,
            flux=flux_spectrum,
            config=self.config,
            solver_status=solver_status,
            solver_time=solver_time,
            validation_metrics=validation_metrics,
            metadata=reconstruction_metadata,
            band_data=all_band_data,
        )

        logger.info(
            f"Reconstruction complete: {solver_status} in {solver_time:.2f}s, "
            f"χ²_ν = {validation_metrics.chi_squared_reduced:.3f}"
        )

        return result


def reconstruct_sed_from_csv(
    csv_path: Path,
    config: Optional[SEDConfig] = None,
    metadata: Optional[Dict[str, any]] = None,
) -> SEDReconstructionResult:
    """
    Convenience function for one-line SED reconstruction.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file with SPHEREx photometry.
    config : Optional[SEDConfig]
        Configuration for reconstruction. If None, uses defaults.
    metadata : Optional[Dict[str, any]]
        Additional metadata to include in results.

    Returns
    -------
    SEDReconstructionResult
        Complete reconstruction result.
    """
    if config is None:
        config = SEDConfig()

    reconstructor = SEDReconstructor(config)
    return reconstructor.reconstruct_from_csv(csv_path, metadata)
