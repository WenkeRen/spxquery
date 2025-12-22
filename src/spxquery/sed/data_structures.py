"""
Data structures for SED reconstruction.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from .config import SEDConfig
from .data_loader import BandData
from .validation import ValidationMetrics


@dataclass
class GlobalSpectralData:
    """
    Container for global spectral reconstruction data.

    Stores the sparse measurement matrix H, observed fluxes y, and weights w.
    H maps the global spectral grid x to observations y: y = H @ x.

    Attributes
    ----------
    H_indices : torch.Tensor
        Indices for sparse H matrix (2, nnz).
    H_values : torch.Tensor
        Values for sparse H matrix (nnz).
    H_shape : Tuple[int, int]
        Shape of H matrix (M_observations, N_spectral_bins).
    observations : torch.Tensor
        Observed flux densities y (M_observations).
    weights : torch.Tensor
        Observation weights w (M_observations).
    global_wavelength_grid : torch.Tensor
        The global wavelength grid (N_spectral_bins).
    """

    H_indices: torch.Tensor
    H_values: torch.Tensor
    H_shape: Tuple[int, int]
    observations: torch.Tensor
    weights: torch.Tensor
    global_wavelength_grid: torch.Tensor


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

    def save_all(self, output_dir) -> None:
        """
        Save reconstruction results to files.

        Parameters
        ----------
        output_dir : Path or str
            Directory to save results.
        """
        import json
        import pandas as pd
        import yaml
        from pathlib import Path

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

        # Save metadata
        metadata_path = output_dir / "sed_metadata.yaml"
        with open(metadata_path, "w") as f:
            yaml.dump(self.metadata, f, default_flow_style=False, sort_keys=False)

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


@dataclass
class EnsembleResult:
    """
    Complete ensemble reconstruction result for SED reconstruction.

    This class stores results from multiple ensemble members along with
    aggregated statistics and uncertainty quantification.

    Attributes
    ----------
    wavelength : np.ndarray
        Global wavelength grid in microns.
    ensemble_fluxes : np.ndarray
        Flux density for each ensemble member in microJansky.
        Shape: (n_ensemble, n_wavelength).
    config : SEDConfig
        Configuration used for ensemble reconstruction.
    ensemble_size : int
        Number of ensemble members.
    aggregation_method : str
        Method used to aggregate ensemble results ('mean', 'median', 'weighted_mean').
    mean_flux : np.ndarray
        Mean ensemble flux density in microJansky.
    std_flux : np.ndarray
        Standard deviation of ensemble fluxes (uncertainty estimate) in microJansky.
    median_flux : np.ndarray
        Median ensemble flux density in microJansky.
    member_results : List[SEDReconstructionResult]
        Individual reconstruction results for each ensemble member.
    ensemble_metadata : Dict[str, Any]
        Ensemble-specific metadata including strategy and aggregation details.
    band_data : Dict[str, BandData]
        Input photometry data per band (shared across ensemble).
    """

    wavelength: np.ndarray
    ensemble_fluxes: np.ndarray
    config: SEDConfig
    member_results: List[SEDReconstructionResult]
    ensemble_metadata: Dict[str, any]
    band_data: Dict[str, any]

    # Computed properties
    ensemble_size: int
    mean_flux: np.ndarray
    std_flux: np.ndarray
    median_flux: np.ndarray

    def __post_init__(self):
        """Compute derived statistics after initialization."""
        self.ensemble_size = len(self.member_results)
        self.mean_flux = np.mean(self.ensemble_fluxes, axis=0)
        self.std_flux = np.std(self.ensemble_fluxes, axis=0, ddof=1)
        self.median_flux = np.median(self.ensemble_fluxes, axis=0)

    def save_all(self, output_dir) -> None:
        """
        Save ensemble reconstruction results to files.

        Parameters
        ----------
        output_dir : Path or str
            Directory to save ensemble results.
        """
        import json
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save mean spectrum
        import pandas as pd

        mean_spectrum_df = pd.DataFrame(
            {
                "wavelength_microns": self.wavelength.tolist(),
                "flux_microjansky": self.mean_flux.tolist(),
                "flux_uncertainty_microjansky": self.std_flux.tolist(),
                "median_flux_microjansky": self.median_flux.tolist(),
            }
        )

        mean_path = output_dir / "ensemble_mean_spectrum.csv"

        mean_spectrum_df.to_csv(mean_path, index=False)

        # Save individual member spectra
        for i, member_result in enumerate(self.member_results):
            member_path = output_dir / f"member_{i:02d}_spectrum.csv"
            member_df = {
                "wavelength_microns": member_result.wavelength.tolist(),
                "flux_microjansky": member_result.flux.tolist(),
            }
            pd.DataFrame(member_df).to_csv(member_path, index=False)

        # Save ensemble metadata
        metadata = {
            "ensemble_size": self.ensemble_size,
            "ensemble_metadata": self.ensemble_metadata,
            "config": self.config.to_dict(),
            "statistics": {
                "mean_flux_mean": float(np.mean(self.mean_flux)),
                "mean_flux_std": float(np.std(self.mean_flux)),
                "uncertainty_mean": float(np.mean(self.std_flux)),
                "uncertainty_max": float(np.max(self.std_flux)),
                "uncertainty_min": float(np.min(self.std_flux)),
            },
        }

        metadata_path = output_dir / "ensemble_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def to_dict(self) -> Dict[str, any]:
        """Convert ensemble result to dictionary for serialization."""
        return {
            "wavelength": self.wavelength.tolist(),
            "ensemble_fluxes": self.ensemble_fluxes.tolist(),
            "config": self.config.to_dict(),
            "ensemble_size": self.ensemble_size,
            "mean_flux": self.mean_flux.tolist(),
            "std_flux": self.std_flux.tolist(),
            "median_flux": self.median_flux.tolist(),
            "ensemble_metadata": self.ensemble_metadata,
        }
