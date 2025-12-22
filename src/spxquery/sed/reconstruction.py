"""
Main orchestrator for SED reconstruction from SPHEREx narrow-band photometry.

This module provides the high-level SEDReconstructor class that coordinates
data loading, global dataset construction, PyTorch-based Deep Image Prior optimization,
and validation for unified spectral reconstruction across all SPHEREx detector bands.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import scipy.sparse as sp

from .config import SEDConfig
from .data_loader import BandData, load_all_bands
from .data_structures import EnsembleResult, SEDReconstructionResult
from .matrices import build_global_observation_data
from .solver_torch import solve_global_reconstruction
from .validation import SpectralEvaluator

logger = logging.getLogger(__name__)


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

    def _prepare_data_from_csv(
        self,
        csv_path: Path,
    ) -> Dict[str, BandData]:
        """
        Load and prepare photometry data from CSV file.

        Parameters
        ----------
        csv_path : Path
            Path to CSV file with photometry data.

        Returns
        -------
        Dict[str, BandData]
            Dictionary mapping band names to BandData objects.

        Raises
        ------
        ValueError
            If no valid photometry data is found in the CSV file.
        """
        logger.info(f"Loading photometry data from {csv_path}")

        # Load photometry data
        all_band_data, _ = load_all_bands(csv_path, self.config)
        if not all_band_data:
            raise ValueError("No valid photometry data found in CSV file")

        logger.info(f"Loaded data for {len(all_band_data)} bands: {list(all_band_data.keys())}")
        return all_band_data

    def _run_single_reconstruction(
        self,
        band_data_dict: Dict[str, BandData],
        metadata: Optional[Dict[str, any]] = None,
    ) -> SEDReconstructionResult:
        """
        Internal method to perform single SED reconstruction from BandData objects.

        Parameters
        ----------
        band_data_dict : Dict[str, BandData]
            Dictionary mapping band names to BandData objects.
        metadata : Optional[Dict[str, any]]
            Additional metadata to include in results.

        Returns
        -------
        SEDReconstructionResult
            Complete reconstruction result.
        """
        logger.info(f"Starting SED reconstruction from {len(band_data_dict)} bands")

        # Build global dataset
        global_dataset = build_global_observation_data(band_data_dict, self.config)

        # Solve using PyTorch Deep Image Prior
        logger.info("Starting PyTorch Deep Image Prior optimization...")
        result_spectrum, solver_status, solver_time = solve_global_reconstruction(global_dataset, self.config)

        # Assess reconstruction quality
        # Convert sparse matrix to scipy csr format for validation
        H_sparse = sp.csr_matrix(
            (global_dataset.H_values.cpu().numpy(), global_dataset.H_indices.cpu().numpy()),
            shape=global_dataset.H_shape,
        )
        evaluator = SpectralEvaluator()
        validation_metrics = evaluator.assess_reconstruction_quality(
            global_dataset.observations.cpu().numpy(),
            H_sparse,
            result_spectrum.cpu().numpy(),
            global_dataset.weights.cpu().numpy(),
        )

        # Create reconstruction metadata
        reconstruction_metadata = {
            "timestamp": datetime.now().isoformat(),
            "solver_type": "torch",
            "solver_status": solver_status,
            "solver_time_seconds": solver_time,
            "global_resolution": self.config.global_resolution,
            "wavelength_range": self.config.wavelength_range,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "regularization_weight": self.config.regularization_weight,
            "cwt_scales": self.config.cwt_scales,
            "n_bands": len(band_data_dict),
            "bands": list(band_data_dict.keys()),
            "total_observations": sum(band.n_measurements for band in band_data_dict.values()),
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
            band_data=band_data_dict,
        )

        logger.info(
            f"Reconstruction complete: {solver_status} in {solver_time:.2f}s, "
            f"chi^2/M = {validation_metrics.chi_squared_per_obs:.3f}"
        )

        return result

    def _run_ensemble_reconstruction(
        self,
        band_data_dict: Dict[str, BandData],
        metadata: Optional[Dict[str, any]],
        csv_path: Optional[Path] = None,
    ) -> EnsembleResult:
        """
        Internal method to run ensemble reconstruction with multiple independent runs.

        Parameters
        ----------
        band_data_dict : Dict[str, BandData]
            Pre-loaded photometry data.
        metadata : Optional[Dict[str, any]]
            Additional metadata to include in results.
        csv_path : Optional[Path]
            Original CSV path for metadata (if available).

        Returns
        -------
        EnsembleResult
            Complete ensemble reconstruction result with aggregated statistics.
        """
        logger.info(f"Starting ensemble reconstruction with {self.config.ensemble_size} members")

        # Create ensemble member configurations with different random seeds
        ensemble_configs = self._create_ensemble_configs()

        # Run ensemble members
        member_results = []
        ensemble_fluxes = []

        for i, member_config in enumerate(ensemble_configs):
            logger.info(f"Running ensemble member {i + 1}/{self.config.ensemble_size}")

            # Create member metadata
            member_metadata = {
                "ensemble_member": i,
                "ensemble_size": self.config.ensemble_size,
                "random_seed": member_config.ensemble_random_seed,
            }
            if csv_path is not None:
                member_metadata["csv_path"] = str(csv_path)
            if metadata:
                member_metadata.update(metadata)

            # Temporarily replace config and run reconstruction
            original_config = self.config
            self.config = member_config
            try:
                member_result = self._run_single_reconstruction(band_data_dict, member_metadata)
                member_results.append(member_result)
                ensemble_fluxes.append(member_result.flux)
            finally:
                self.config = original_config

        # Convert to numpy array
        ensemble_fluxes = np.array(ensemble_fluxes)

        # Create ensemble metadata
        ensemble_metadata = {
            "strategy": self.config.ensemble_strategy,
            "ensemble_size": self.config.ensemble_size,
            "random_seed_base": self.config.ensemble_random_seed,
            "timestamp": datetime.now().isoformat(),
        }
        if csv_path is not None:
            ensemble_metadata["csv_path"] = str(csv_path)

        # Create ensemble result
        ensemble_result = EnsembleResult(
            wavelength=member_results[0].wavelength,
            ensemble_fluxes=ensemble_fluxes,
            config=self.config,
            member_results=member_results,
            ensemble_metadata=ensemble_metadata,
            band_data=band_data_dict,
            ensemble_size=self.config.ensemble_size,
            mean_flux=np.mean(ensemble_fluxes, axis=0),
            std_flux=np.std(ensemble_fluxes, axis=0, ddof=1),
            median_flux=np.median(ensemble_fluxes, axis=0),
        )

        logger.info(
            f"Ensemble reconstruction complete: {self.config.ensemble_size} members, "
            f"mean chi^2/M = {np.mean([r.validation_metrics.chi_squared_per_obs for r in member_results]):.3f} "
            f"+- {np.std([r.validation_metrics.chi_squared_per_obs for r in member_results]):.3f}"
        )

        return ensemble_result

    def _create_ensemble_configs(self) -> list[SEDConfig]:
        """
        Create configuration objects for each ensemble member.

        Returns
        -------
        list[SEDConfig]
            List of configuration objects, one for each ensemble member.
        """
        configs = []

        for i in range(self.config.ensemble_size):
            # Create a copy of the current config
            member_config = self.config.copy_with_overrides(
                ensemble_size=1,  # Each member runs as single reconstruction
            )

            # Set random seed for reproducible ensembles
            if self.config.ensemble_random_seed is not None:
                member_seed = self.config.ensemble_random_seed + i
                member_config = member_config.copy_with_overrides(
                    ensemble_random_seed=member_seed,
                )

            # Disable wandb for subsequent members to prevent conflicts (only first member logged)
            if i > 0:
                member_config = member_config.copy_with_overrides(wandb_run=None)

            configs.append(member_config)

        return configs

    def reconstruct_from_csv(
        self,
        csv_path: Path,
        metadata: Optional[Dict[str, any]] = None,
    ) -> Union[SEDReconstructionResult, EnsembleResult]:
        """
        Reconstruct SED from CSV file containing SPHEREx photometry.

        Automatically determines whether to run ensemble or single reconstruction
        based on the config.ensemble_size parameter.

        Parameters
        ----------
        csv_path : Path
            Path to CSV file with photometry data.
        metadata : Optional[Dict[str, any]]
            Additional metadata to include in results.

        Returns
        -------
        SEDReconstructionResult | EnsembleResult
            Complete reconstruction result. Returns EnsembleResult if config.ensemble_size > 1,
            otherwise returns SEDReconstructionResult.
        """
        # Load data from CSV
        band_data_dict = self._prepare_data_from_csv(csv_path)

        # Add CSV path to metadata
        csv_metadata = {"csv_path": str(csv_path)}
        if metadata:
            csv_metadata.update(metadata)

        # Reconstruct from loaded data (pass CSV path for ensemble metadata)
        return self._run_reconstruction_with_path(band_data_dict, csv_metadata, csv_path)

    def _run_reconstruction_with_path(
        self,
        band_data_dict: Dict[str, BandData],
        metadata: Optional[Dict[str, any]],
        csv_path: Optional[Path] = None,
    ) -> Union[SEDReconstructionResult, EnsembleResult]:
        """
        Internal method that decides between ensemble/single reconstruction with path support.
        """
        # Check if ensemble reconstruction is needed
        if self.config.ensemble_size > 1:
            logger.info(f"Ensemble reconstruction requested with {self.config.ensemble_size} members")
            return self._run_ensemble_reconstruction(band_data_dict, metadata, csv_path)
        else:
            logger.info("Single reconstruction requested")
            return self._run_single_reconstruction(band_data_dict, metadata)

    def reconstruct_from_data(
        self,
        band_data_dict: Dict[str, BandData],
        metadata: Optional[Dict[str, any]] = None,
    ) -> Union[SEDReconstructionResult, EnsembleResult]:
        """
        Reconstruct SED from pre-loaded BandData objects.

        Automatically determines whether to run ensemble or single reconstruction
        based on the config.ensemble_size parameter.

        Parameters
        ----------
        band_data_dict : Dict[str, BandData]
            Dictionary mapping band names to BandData objects.
        metadata : Optional[Dict[str, any]]
            Additional metadata to include in results.

        Returns
        -------
        SEDReconstructionResult | EnsembleResult
            Complete reconstruction result. Returns EnsembleResult if config.ensemble_size > 1,
            otherwise returns SEDReconstructionResult.
        """
        # Use the internal decision method
        return self._run_reconstruction_with_path(band_data_dict, metadata)


def reconstruct_sed_from_csv(
    csv_path: Path,
    config: Optional[SEDConfig] = None,
    metadata: Optional[Dict[str, any]] = None,
) -> Union[SEDReconstructionResult, EnsembleResult]:
    """
    Convenience function for one-line SED reconstruction.

    Automatically determines whether to run ensemble or single reconstruction
    based on the config.ensemble_size parameter.

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
    SEDReconstructionResult | EnsembleResult
        Complete reconstruction result. Returns EnsembleResult if config.ensemble_size > 1,
        otherwise returns SEDReconstructionResult.
    """
    if config is None:
        config = SEDConfig()

    reconstructor = SEDReconstructor(config)
    return reconstructor.reconstruct_from_csv(csv_path, metadata)
