"""
Main orchestrator for SED reconstruction from SPHEREx narrow-band photometry.

This module provides the high-level SEDReconstructor class that coordinates
data loading, global dataset construction, PyTorch-based Deep Image Prior optimization,
and validation for unified spectral reconstruction across all SPHEREx detector bands.
"""

import concurrent.futures
import logging
import multiprocessing as mp
import time
import traceback
from collections import deque
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


def _ensemble_worker_entry(
    result_queue,
    member_index: int,
    member_config: SEDConfig,
    band_data_dict: Dict[str, BandData],
    metadata: Optional[Dict[str, any]] = None,
    csv_path: Optional[Path] = None,
):
    """Run a single ensemble member in a dedicated subprocess.

    This wrapper allows the parent process to enforce a wall-time timeout and
    terminate/retry a worker that becomes unresponsive.
    """
    try:
        idx, result = _run_ensemble_member(
            member_index=member_index,
            member_config=member_config,
            band_data_dict=band_data_dict,
            metadata=metadata,
            csv_path=csv_path,
        )
        result_queue.put(("ok", idx, result))
    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put(("err", member_index, repr(e), tb))


def _run_ensemble_member(
    member_index: int,
    member_config: SEDConfig,
    band_data_dict: Dict[str, BandData],
    metadata: Optional[Dict[str, any]] = None,
    csv_path: Optional[Path] = None,
) -> tuple[int, SEDReconstructionResult]:
    """
    Standalone function to run a single ensemble member reconstruction.

    This function is designed to be used with multiprocessing for parallel ensemble execution.

    Parameters
    ----------
    member_index : int
        Index of this ensemble member.
    member_config : SEDConfig
        Configuration for this ensemble member (with specific random seed).
    band_data_dict : Dict[str, BandData]
        Band data for all SPHEREx detectors.
    metadata : Optional[Dict[str, any]]
        Additional metadata to include in results.
    csv_path : Optional[Path]
        Original CSV path for metadata (if available).

    Returns
    -------
    tuple[int, SEDReconstructionResult]
        Tuple of (member_index, reconstruction_result) for ordered results.
    """
    logger.info(f"Running ensemble member {member_index + 1}")

    # Apply perturbation if enabled
    if member_config.ensemble_perturb_observations:
        logger.debug(f"Applying Gaussian perturbation to observations for ensemble member {member_index}")
        # Create perturbed band data dictionary
        perturbed_band_data_dict = {}
        for band, band_data in band_data_dict.items():
            perturbed_band_data_dict[band] = band_data.perturb_flux()
        band_data_dict = perturbed_band_data_dict

    # Create member metadata
    member_metadata = {
        "ensemble_member": member_index,
        "ensemble_size": member_config.ensemble_size,  # TODO: always be 1 due to member configs, inconsistent to real value.
        "random_seed": member_config.ensemble_random_seed,
    }
    if csv_path is not None:
        member_metadata["csv_path"] = str(csv_path)
    if metadata:
        member_metadata.update(metadata)

    # Create reconstructor with member config and run reconstruction
    reconstructor = SEDReconstructor(member_config)
    # Get ensemble size from config for progress bar display
    ensemble_total = member_config.ensemble_size  # TODO: inconsistent to real value.
    member_result = reconstructor._run_single_reconstruction(
        band_data_dict, member_metadata, ensemble_member=member_index, ensemble_total=ensemble_total
    )

    return (member_index, member_result)


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
        ensemble_member: Optional[int] = None,
        ensemble_total: Optional[int] = None,
    ) -> SEDReconstructionResult:
        """
        Internal method to perform single SED reconstruction from BandData objects.

        Parameters
        ----------
        band_data_dict : Dict[str, BandData]
            Dictionary mapping band names to BandData objects.
        metadata : Optional[Dict[str, any]]
            Additional metadata to include in results.
        ensemble_member : Optional[int]
            Ensemble member index (0-based) if running as part of ensemble.
        ensemble_total : Optional[int]
            Total number of ensemble members.

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
        result_spectrum, solver_status, solver_time = solve_global_reconstruction(
            global_dataset, self.config, ensemble_member=ensemble_member, ensemble_total=ensemble_total
        )

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

        # Determine if parallel processing should be used
        use_parallel = self.config.ensemble_n_workers is not None and self.config.ensemble_n_workers > 1

        if use_parallel:
            n_workers = min(self.config.ensemble_n_workers, self.config.ensemble_size)
            logger.info(f"Using parallel ensemble processing with {n_workers} workers")

            enable_watchdog = (
                self.config.ensemble_member_timeout_seconds is not None or self.config.ensemble_max_retries > 0
            )

            if not enable_watchdog:
                # Default behavior: ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = []
                    for i, member_config in enumerate(ensemble_configs):
                        future = executor.submit(
                            _run_ensemble_member,
                            member_index=i,
                            member_config=member_config,
                            band_data_dict=band_data_dict,
                            metadata=metadata,
                            csv_path=csv_path,
                        )
                        futures.append(future)

                    # Collect results in order as they complete
                    member_results_unordered = []
                    for future in concurrent.futures.as_completed(futures):
                        member_index, member_result = future.result()
                        member_results_unordered.append((member_index, member_result))

                    # Sort results by member_index to ensure correct order
                    member_results_unordered.sort(key=lambda x: x[0])
                    member_results = [result for _, result in member_results_unordered]

            else:
                # Watchdog behavior: dedicated subprocess per member with timeout + retry.
                timeout_s = self.config.ensemble_member_timeout_seconds
                max_retries = self.config.ensemble_max_retries
                backoff_s = self.config.ensemble_retry_backoff_seconds

                logger.info(
                    f"Ensemble watchdog enabled: timeout={timeout_s}s, max_retries={max_retries}, backoff={backoff_s}s"
                )

                ctx = mp.get_context("spawn")
                result_queue = ctx.Queue()

                pending = deque(
                    {
                        "member_index": i,
                        "member_config": cfg,
                        "attempt": 0,
                    }
                    for i, cfg in enumerate(ensemble_configs)
                )

                running = {}  # member_index -> dict(process=..., start=..., attempt=...)
                completed = {}

                def _start_job(job):
                    proc = ctx.Process(
                        target=_ensemble_worker_entry,
                        args=(
                            result_queue,
                            job["member_index"],
                            job["member_config"],
                            band_data_dict,
                            metadata,
                            csv_path,
                        ),
                    )
                    proc.start()
                    running[job["member_index"]] = {
                        "process": proc,
                        "start": time.monotonic(),
                        "attempt": job["attempt"],
                        "member_config": job["member_config"],
                    }

                def _resubmit_or_raise(member_index: int, reason: str, attempt: int):
                    if attempt < max_retries:
                        if backoff_s > 0:
                            time.sleep(backoff_s)
                        next_job = {
                            "member_index": member_index,
                            "member_config": ensemble_configs[member_index],
                            "attempt": attempt + 1,
                        }
                        logger.warning(
                            f"Retrying ensemble member {member_index + 1}/{self.config.ensemble_size} "
                            f"(attempt {attempt + 1}/{max_retries}) after {reason}"
                        )
                        pending.appendleft(next_job)
                    else:
                        raise TimeoutError(
                            f"Ensemble member {member_index + 1}/{self.config.ensemble_size} failed: {reason}; "
                            f"retries exhausted (max_retries={max_retries})."
                        )

                # Main loop
                while len(completed) < self.config.ensemble_size:
                    # Launch up to n_workers
                    while pending and len(running) < n_workers:
                        job = pending.popleft()
                        _start_job(job)

                    # Check for timeout
                    if timeout_s is not None:
                        now = time.monotonic()
                        for member_index, info in list(running.items()):
                            if now - info["start"] > timeout_s:
                                proc = info["process"]
                                attempt = info["attempt"]
                                logger.error(
                                    f"Timeout: ensemble member {member_index + 1}/{self.config.ensemble_size} "
                                    f"exceeded {timeout_s}s; terminating process (pid={proc.pid})"
                                )
                                if proc.is_alive():
                                    proc.terminate()
                                proc.join(timeout=5)
                                running.pop(member_index, None)
                                _resubmit_or_raise(member_index, reason=f"timeout>{timeout_s}s", attempt=attempt)

                    # Collect results (non-blocking-ish)
                    try:
                        msg = result_queue.get(timeout=0.5)
                    except Exception:
                        msg = None

                    if msg is not None:
                        kind = msg[0]
                        if kind == "ok":
                            _, member_index, member_result = msg
                            info = running.pop(member_index, None)
                            if info is not None:
                                info["process"].join(timeout=5)
                            completed[member_index] = member_result
                        else:
                            _, member_index, err_repr, tb = msg
                            info = running.pop(member_index, None)
                            attempt = info["attempt"] if info is not None else 0
                            if info is not None:
                                info["process"].join(timeout=5)
                            logger.error(
                                f"Ensemble member {member_index + 1}/{self.config.ensemble_size} "
                                f"crashed: {err_repr}\n{tb}"
                            )
                            _resubmit_or_raise(member_index, reason=f"exception: {err_repr}", attempt=attempt)

                    # Also handle unexpectedly-dead processes without queue output
                    for member_index, info in list(running.items()):
                        proc = info["process"]
                        if not proc.is_alive() and proc.exitcode not in (0, None):
                            attempt = info["attempt"]
                            exitcode = proc.exitcode
                            proc.join(timeout=1)
                            running.pop(member_index, None)
                            logger.error(
                                f"Ensemble member {member_index + 1}/{self.config.ensemble_size} "
                                f"exited with code {exitcode}"
                            )
                            _resubmit_or_raise(member_index, reason=f"exitcode={exitcode}", attempt=attempt)

                # Order results
                member_results = [completed[i] for i in range(self.config.ensemble_size)]

            # Extract fluxes from results
            ensemble_fluxes = [result.flux for result in member_results]

        else:
            # Sequential processing (original behavior)
            logger.info("Using sequential ensemble processing")
            member_results = []
            ensemble_fluxes = []

            for i, member_config in enumerate(ensemble_configs):
                logger.info(f"Running ensemble member {i + 1}/{self.config.ensemble_size}")

                # Apply perturbation if enabled (use original band_data_dict for each member)
                member_band_data = band_data_dict
                if member_config.ensemble_perturb_observations:
                    logger.debug(f"Applying Gaussian perturbation to observations for ensemble member {i}")
                    # Create perturbed band data dictionary
                    perturbed_band_data_dict = {}
                    for band, band_data in band_data_dict.items():
                        perturbed_band_data_dict[band] = band_data.perturb_flux()
                    member_band_data = perturbed_band_data_dict

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
                    member_result = self._run_single_reconstruction(
                        member_band_data, member_metadata, ensemble_member=i, ensemble_total=self.config.ensemble_size
                    )
                    member_results.append(member_result)
                    ensemble_fluxes.append(member_result.flux)
                finally:
                    self.config = original_config

        # Convert to numpy array
        ensemble_fluxes = np.array(ensemble_fluxes)

        # Compute ensemble statistics
        mean_flux = np.mean(ensemble_fluxes, axis=0)
        std_flux = np.std(ensemble_fluxes, axis=0, ddof=1)
        median_flux = np.median(ensemble_fluxes, axis=0)

        # Build global dataset once for validation (minimize computation time)
        logger.info("Building global dataset for ensemble validation")
        global_dataset = build_global_observation_data(band_data_dict, self.config)

        # Convert PyTorch tensors to numpy/SciPy format for validation
        logger.info("Computing validation metrics for ensemble mean spectrum")
        H_sparse = sp.csr_matrix(
            (global_dataset.H_values.cpu().numpy(), global_dataset.H_indices.cpu().numpy()),
            shape=global_dataset.H_shape,
        )

        # Compute validation metrics for ensemble mean flux
        evaluator = SpectralEvaluator()
        validation_metrics = evaluator.assess_reconstruction_quality(
            y=global_dataset.observations.cpu().numpy(),
            H=H_sparse,
            spectrum=mean_flux,
            weights=global_dataset.weights.cpu().numpy(),
        )

        # Create ensemble metadata
        ensemble_metadata = {
            "strategy": self.config.ensemble_strategy,
            "ensemble_size": self.config.ensemble_size,
            "random_seed_base": self.config.ensemble_random_seed,
            "timestamp": datetime.now().isoformat(),
        }
        if csv_path is not None:
            ensemble_metadata["csv_path"] = str(csv_path)

        # Create ensemble result with validation metrics
        ensemble_result = EnsembleResult(
            wavelength=member_results[0].wavelength,
            ensemble_fluxes=ensemble_fluxes,
            config=self.config,
            member_results=member_results,
            ensemble_metadata=ensemble_metadata,
            band_data=band_data_dict,
            validation_metrics=validation_metrics,
            ensemble_size=self.config.ensemble_size,
            mean_flux=mean_flux,
            std_flux=std_flux,
            median_flux=median_flux,
        )

        logger.info(
            f"Ensemble reconstruction complete: {self.config.ensemble_size} members, "
            f"mean chi^2/M = {np.mean([r.validation_metrics.chi_squared_per_obs for r in member_results]):.3f} "
            f"+- {np.std([r.validation_metrics.chi_squared_per_obs for r in member_results]):.3f}, "
            f"ensemble mean chi^2/M = {validation_metrics.chi_squared_per_obs:.3f}"
        )

        return ensemble_result

    def _create_ensemble_configs(self) -> list[SEDConfig]:
        """
        Create configuration objects for each ensemble member.

        This method creates specialized configurations for each ensemble member by:
        - Setting ensemble_size=1 (each member is a single reconstruction)
        - Setting unique random seeds for reproducible ensembles
        - Disabling nested parallelism (ensemble_n_workers=None)
        - Preserving ensemble_perturb_observations flag
        - Disabling wandb for members 1+ to avoid logging conflicts

        Returns
        -------
        list[SEDConfig]
            List of configuration objects, one for each ensemble member.
        """
        configs = []

        for i in range(self.config.ensemble_size):
            # Create a copy of the current config with ensemble-specific overrides
            member_config = self.config.copy_with_overrides(
                ensemble_size=1,  # Each member runs as single reconstruction
                ensemble_n_workers=None,  # Disable nested parallelism for ensemble members
                # ensemble_perturb_observations is preserved from parent config
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
