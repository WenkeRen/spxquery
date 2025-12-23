"""
Data structures for SED reconstruction.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml

from .config import SEDConfig
from .data_loader import BandData
from .validation import ValidationMetrics

logger = logging.getLogger(__name__)


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

    def save_all(self, output_dir: Union[str, Path]) -> None:
        """
        Save complete reconstruction results with proper folder structure.

        Creates the following structure:
        output_dir/
        ├── config/sed_config.yaml
        ├── banddata/{band}.csv
        ├── results/sed.csv, validation.json, residual.csv
        ├── plots/sed_qa_plot.png
        └── logs/reconstruction.yaml

        Parameters
        ----------
        output_dir : Path or str
            Directory to save results.
        """
        output_dir = Path(output_dir)

        # Create folder structure
        config_dir = output_dir / "config"
        banddata_dir = output_dir / "banddata"
        results_dir = output_dir / "results"
        plots_dir = output_dir / "plots"
        logs_dir = output_dir / "logs"

        for dir_path in [config_dir, banddata_dir, results_dir, plots_dir, logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 1. Save config
        self.config.to_yaml_file(config_dir / "sed_config.yaml")
        logger.info(f"Saved config to {config_dir / 'sed_config.yaml'}")

        # 2. Save band data
        for band_name, band_data in self.band_data.items():
            band_data.save_to_csv(banddata_dir)

        # 3. Save SED result
        sed_df = pd.DataFrame(
            {
                "wavelength": self.wavelength,
                "flux": self.flux,
            }
        )
        sed_path = results_dir / "sed.csv"
        sed_df.to_csv(sed_path, index=False)
        logger.info(f"Saved SED result to {sed_path}")

        # 4. Save validation metrics
        self.validation_metrics.save_to_files(results_dir)

        # 5. Generate QA plot
        try:
            from .plots import plot_sed_reconstruction_dashboard

            plot_path = plots_dir / "sed_qa_plot.png"
            plot_sed_reconstruction_dashboard(self, plot_path)
            logger.info(f"Saved QA plot to {plot_path}")
        except ImportError as e:
            logger.warning(f"Could not generate QA plot: {e}")

        # 6. Save metadata to reconstruction.yaml
        log_data = {
            "solver_status": self.solver_status,
            "solver_time": self.solver_time,
            "metadata": self.metadata,
        }
        log_path = logs_dir / "reconstruction.yaml"
        with open(log_path, "w") as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved reconstruction log to {log_path}")

        logger.info(f"Complete SED reconstruction saved to {output_dir}")

    @classmethod
    def load_all(cls, output_dir: Union[str, Path]) -> "SEDReconstructionResult":
        """
        Load SEDReconstructionResult from saved directory.

        Parameters
        ----------
        output_dir : Path or str
            Directory containing saved reconstruction results.

        Returns
        -------
        SEDReconstructionResult
            Loaded reconstruction result.

        Raises
        ------
        FileNotFoundError
            If required files or directories are missing.
        """
        output_dir = Path(output_dir)

        # Define paths
        config_path = output_dir / "config" / "sed_config.yaml"
        banddata_dir = output_dir / "banddata"
        results_dir = output_dir / "results"
        logs_dir = output_dir / "logs"

        # Validate required paths
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not banddata_dir.exists():
            raise FileNotFoundError(f"Banddata directory not found: {banddata_dir}")
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        # 1. Load config
        config = SEDConfig.from_yaml_file(config_path)
        logger.info(f"Loaded config from {config_path}")

        # 2. Load band data
        band_data = {}
        for csv_file in sorted(banddata_dir.glob("*.csv")):
            band_obj = BandData.from_csv(csv_file)
            band_data[band_obj.band] = band_obj
        logger.info(f"Loaded {len(band_data)} band data files")

        # 3. Load SED result
        sed_path = results_dir / "sed.csv"
        if not sed_path.exists():
            raise FileNotFoundError(f"SED result file not found: {sed_path}")
        sed_df = pd.read_csv(sed_path)
        wavelength = sed_df["wavelength"].values
        flux = sed_df["flux"].values
        logger.info(f"Loaded SED result from {sed_path}")

        # 4. Load validation metrics
        validation_path = results_dir / "validation.json"
        residual_path = results_dir / "residual.csv"
        if not validation_path.exists() or not residual_path.exists():
            raise FileNotFoundError(f"Validation files not found in {results_dir}")
        validation_metrics = ValidationMetrics.from_files(validation_path, residual_path)
        logger.info("Loaded validation metrics")

        # 5. Load metadata from reconstruction.yaml
        solver_status = "unknown"
        solver_time = 0.0
        metadata = {}

        log_path = logs_dir / "reconstruction.yaml"
        if log_path.exists():
            with open(log_path, "r") as f:
                log_data = yaml.safe_load(f)
                solver_status = log_data.get("solver_status", "unknown")
                solver_time = log_data.get("solver_time", 0.0)
                metadata = log_data.get("metadata", {})
            logger.info(f"Loaded reconstruction log from {log_path}")
        else:
            logger.warning(f"Reconstruction log not found: {log_path}")

        # Create SEDReconstructionResult object
        result = cls(
            wavelength=wavelength,
            flux=flux,
            config=config,
            solver_status=solver_status,
            solver_time=solver_time,
            validation_metrics=validation_metrics,
            metadata=metadata,
            band_data=band_data,
        )

        logger.info(f"Loaded complete SED reconstruction from {output_dir}")

        return result

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
    validation_metrics : ValidationMetrics
        Quality assessment metrics for the ensemble mean spectrum.
    """

    wavelength: np.ndarray
    ensemble_fluxes: np.ndarray
    config: SEDConfig
    member_results: List[SEDReconstructionResult]
    ensemble_metadata: Dict[str, any]
    band_data: Dict[str, any]
    validation_metrics: ValidationMetrics

    # Computed properties
    ensemble_size: int
    mean_flux: np.ndarray
    std_flux: np.ndarray
    median_flux: np.ndarray

    def save_all(self, output_dir: Union[str, Path], save_members: bool = False) -> None:
        """
        Save complete ensemble reconstruction results with proper folder structure.

        Creates the following structure:
        output_dir/
        ├── config/sed_config.yaml
        ├── banddata/{band}.csv
        ├── results/sed.csv, validation.json, residual.csv
        ├── results/members/ (optional, if save_members=True)
        │   ├── member_1/sed.csv, validation.json, residual.csv
        │   ├── member_2/...
        │   └── ...
        ├── plots/sed_qa_plot.png
        └── logs/reconstruction.yaml

        Parameters
        ----------
        output_dir : Path or str
            Directory to save results.
        save_members : bool
            If True, save individual ensemble member results to results/members/ folder.
        """
        output_dir = Path(output_dir)

        # Create folder structure
        config_dir = output_dir / "config"
        banddata_dir = output_dir / "banddata"
        results_dir = output_dir / "results"
        plots_dir = output_dir / "plots"
        logs_dir = output_dir / "logs"

        for dir_path in [config_dir, banddata_dir, results_dir, plots_dir, logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 1. Save config
        self.config.to_yaml_file(config_dir / "sed_config.yaml")
        logger.info(f"Saved config to {config_dir / 'sed_config.yaml'}")

        # 2. Save band data
        for band_name, band_data in self.band_data.items():
            band_data.save_to_csv(banddata_dir)

        # 3. Save ensemble SED result with member columns
        # Columns: wavelength, flux (mean), median_flux, flux_err (std), flux_member_1, flux_member_2, ...
        sed_data = {
            "wavelength": self.wavelength,
            "flux": self.mean_flux,
            "median_flux": self.median_flux,
            "flux_err": self.std_flux,
        }

        # Add individual member fluxes
        for i in range(self.ensemble_size):
            sed_data[f"flux_member_{i + 1}"] = self.ensemble_fluxes[i]

        sed_df = pd.DataFrame(sed_data)
        sed_path = results_dir / "sed.csv"
        sed_df.to_csv(sed_path, index=False)
        logger.info(f"Saved ensemble SED result to {sed_path}")

        # 4. Save validation metrics
        self.validation_metrics.save_to_files(results_dir)

        # 5. Optionally save individual member results
        if save_members and self.member_results:
            members_dir = results_dir / "members"
            members_dir.mkdir(exist_ok=True)

            for i, member_result in enumerate(self.member_results, start=1):
                member_dir = members_dir / f"member_{i}"
                member_dir.mkdir(exist_ok=True)

                # Save member SED
                member_sed_df = pd.DataFrame(
                    {
                        "wavelength": member_result.wavelength,
                        "flux": member_result.flux,
                    }
                )
                member_sed_path = member_dir / "sed.csv"
                member_sed_df.to_csv(member_sed_path, index=False)

                # Save member validation metrics
                member_result.validation_metrics.save_to_files(member_dir)

            logger.info(f"Saved {len(self.member_results)} ensemble member results to {members_dir}")

        # 6. Generate QA plot
        try:
            from .plots import plot_sed_reconstruction_dashboard

            plot_path = plots_dir / "sed_qa_plot.png"
            plot_sed_reconstruction_dashboard(self, plot_path)
            logger.info(f"Saved QA plot to {plot_path}")
        except ImportError as e:
            logger.warning(f"Could not generate QA plot: {e}")

        # 7. Save ensemble metadata to reconstruction.yaml
        # Collect member solver info
        member_solver_info = []
        for i, member_result in enumerate(self.member_results, start=1):
            member_solver_info.append(
                {
                    "member": i,
                    "solver_status": member_result.solver_status,
                    "solver_time": member_result.solver_time,
                }
            )

        log_data = {
            "ensemble_size": self.ensemble_size,
            "ensemble_metadata": self.ensemble_metadata,
            "member_solver_info": member_solver_info,
        }
        log_path = logs_dir / "reconstruction.yaml"
        with open(log_path, "w") as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved ensemble reconstruction log to {log_path}")

        logger.info(f"Complete ensemble reconstruction saved to {output_dir}")

    @classmethod
    def load_all(cls, output_dir: Union[str, Path]) -> "EnsembleResult":
        """
        Load EnsembleResult from saved directory.

        Parameters
        ----------
        output_dir : Path or str
            Directory containing saved ensemble reconstruction results.

        Returns
        -------
        EnsembleResult
            Loaded ensemble result.

        Raises
        ------
        FileNotFoundError
            If required files or directories are missing.
        """
        output_dir = Path(output_dir)

        # Define paths
        config_path = output_dir / "config" / "sed_config.yaml"
        banddata_dir = output_dir / "banddata"
        results_dir = output_dir / "results"
        logs_dir = output_dir / "logs"
        members_dir = results_dir / "members"

        # Validate required paths
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not banddata_dir.exists():
            raise FileNotFoundError(f"Banddata directory not found: {banddata_dir}")
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        # 1. Load config
        config = SEDConfig.from_yaml_file(config_path)
        logger.info(f"Loaded config from {config_path}")

        # Check ensemble_size from config
        ensemble_size = getattr(config, "ensemble_size", 1)
        logger.info(f"Detected ensemble_size={ensemble_size} from config")

        # 2. Load band data
        band_data = {}
        for csv_file in sorted(banddata_dir.glob("*.csv")):
            band_obj = BandData.from_csv(csv_file)
            band_data[band_obj.band] = band_obj
        logger.info(f"Loaded {len(band_data)} band data files")

        # 3. Load ensemble SED result
        sed_path = results_dir / "sed.csv"
        if not sed_path.exists():
            raise FileNotFoundError(f"SED result file not found: {sed_path}")
        sed_df = pd.read_csv(sed_path)

        wavelength = sed_df["wavelength"].values
        mean_flux = sed_df["flux"].values
        median_flux = sed_df["median_flux"].values
        std_flux = sed_df["flux_err"].values

        # Load individual member fluxes from columns flux_member_1, flux_member_2, ...
        member_fluxes_list = []
        i = 1
        while f"flux_member_{i}" in sed_df.columns:
            member_fluxes_list.append(sed_df[f"flux_member_{i}"].values)
            i += 1

        ensemble_fluxes = np.array(member_fluxes_list)
        logger.info(f"Loaded ensemble SED result with {len(member_fluxes_list)} members")

        # 4. Load validation metrics
        validation_path = results_dir / "validation.json"
        residual_path = results_dir / "residual.csv"
        if not validation_path.exists() or not residual_path.exists():
            raise FileNotFoundError(f"Validation files not found in {results_dir}")
        validation_metrics = ValidationMetrics.from_files(validation_path, residual_path)
        logger.info("Loaded validation metrics")

        # 5. Load member results if members folder exists
        member_results = []
        ensemble_metadata = {}

        if members_dir.exists():
            # Load individual member results
            for member_folder in sorted(members_dir.glob("member_*")):
                if not member_folder.is_dir():
                    continue

                # Extract member number
                member_num = int(member_folder.name.split("_")[1])

                # Load member SED
                member_sed_path = member_folder / "sed.csv"
                if not member_sed_path.exists():
                    logger.warning(f"Member {member_num}: sed.csv not found, skipping")
                    continue

                member_sed_df = pd.read_csv(member_sed_path)
                member_wavelength = member_sed_df["wavelength"].values
                member_flux = member_sed_df["flux"].values

                # Load member validation metrics
                member_validation_path = member_folder / "validation.json"
                member_residual_path = member_folder / "residual.csv"
                if not member_validation_path.exists() or not member_residual_path.exists():
                    logger.warning(f"Member {member_num}: validation files not found, skipping")
                    continue

                member_validation = ValidationMetrics.from_files(member_validation_path, member_residual_path)

                # Create member SEDReconstructionResult
                # Use parent config and band_data (as per user requirement)
                member_result = SEDReconstructionResult(
                    wavelength=member_wavelength,
                    flux=member_flux,
                    config=config,  # Use parent config
                    solver_status="loaded_from_save",
                    solver_time=0.0,
                    validation_metrics=member_validation,
                    metadata={"member_number": member_num},
                    band_data=band_data,  # Use parent band_data
                )
                member_results.append(member_result)

            logger.info(f"Loaded {len(member_results)} ensemble member results from {members_dir}")

        # 6. Load ensemble metadata from reconstruction.yaml
        log_path = logs_dir / "reconstruction.yaml"
        if log_path.exists():
            with open(log_path, "r") as f:
                log_data = yaml.safe_load(f)
                ensemble_metadata = log_data.get("ensemble_metadata", {})

                # If no member_results were loaded from folder, add solver info to metadata
                if not member_results and "member_solver_info" in log_data:
                    ensemble_metadata["member_solver_info"] = log_data["member_solver_info"]

            logger.info(f"Loaded ensemble reconstruction log from {log_path}")
        else:
            logger.warning(f"Ensemble reconstruction log not found: {log_path}")

        # Create EnsembleResult object
        result = cls(
            wavelength=wavelength,
            ensemble_fluxes=ensemble_fluxes,
            config=config,
            member_results=member_results,
            ensemble_metadata=ensemble_metadata,
            band_data=band_data,
            validation_metrics=validation_metrics,
            ensemble_size=len(member_fluxes_list),
            mean_flux=mean_flux,
            std_flux=std_flux,
            median_flux=median_flux,
        )

        logger.info(f"Loaded complete ensemble reconstruction from {output_dir}")

        return result

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
            "validation_metrics": {
                "chi_squared": self.validation_metrics.chi_squared,
                "chi_squared_per_obs": self.validation_metrics.chi_squared_per_obs,
                "n_obs": self.validation_metrics.n_obs,
                "n_sample": self.validation_metrics.n_sample,
                "residual_mean": self.validation_metrics.residual_mean,
                "residual_std": self.validation_metrics.residual_std,
                "weighted_residual_mean": self.validation_metrics.weighted_residual_mean,
                "weighted_residual_std": self.validation_metrics.weighted_residual_std,
                "max_residual": self.validation_metrics.max_residual,
                "normality_pvalue": self.validation_metrics.normality_pvalue,
                "negative_flux_fraction": self.validation_metrics.negative_flux_fraction,
                "smoothness_tv": self.validation_metrics.smoothness_tv,
                "residual_oscillation": self.validation_metrics.residual_oscillation,
                "residual_rms": self.validation_metrics.residual_rms,
            },
        }
