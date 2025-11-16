"""
Hyperparameter tuning for SED reconstruction with wavelet regularization.

This module implements grid search over wavelet regularization weights
(lambda_low, lambda_detail) to find optimal values that minimize validation error.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import scipy.sparse as sp

from .config import SEDConfig
from .solver import solve_reconstruction
from .validation import split_train_validation, compute_validation_error

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """
    Result container for hyperparameter tuning with wavelet regularization.

    Attributes
    ----------
    optimal_lambda_low : float
        Best approximation coefficient regularization weight (continuum).
    optimal_lambda_detail : float
        Best detail coefficient regularization weight (noise suppression).
    optimal_validation_error : float
        Validation error at optimal hyperparameters.
    grid_results : List[Dict]
        List of dictionaries with keys:
        - 'lambda_low': Continuum regularization weight
        - 'lambda_detail': Noise suppression weight
        - 'validation_error': Validation error
        - 'success': Whether solver converged
    n_evaluated : int
        Number of hyperparameter combinations evaluated.
    n_failed : int
        Number of combinations where solver failed.
    """

    optimal_lambda_low: float
    optimal_lambda_detail: float
    optimal_validation_error: float
    grid_results: List[Dict]
    n_evaluated: int
    n_failed: int


def grid_search(
    y: np.ndarray,
    H: sp.csr_matrix,
    Psi_approx: sp.csr_matrix,
    Psi_detail: sp.csr_matrix,
    weights: np.ndarray,
    wavelength_grid: np.ndarray,
    level_info: Dict[str, int],
    config: SEDConfig,
    edge_info: Optional[Dict[str, any]] = None,
) -> TuningResult:
    """
    Perform grid search to find optimal wavelet regularization hyperparameters.

    The algorithm:
    1. Split data into training and validation sets
    2. For each (lambda_low, lambda_detail) pair in grid:
       a. Solve reconstruction on training set
       b. Evaluate validation error on held-out validation set
    3. Select hyperparameters with lowest validation error

    Parameters
    ----------
    y : np.ndarray
        Observed flux measurements, shape (M,).
    H : sp.csr_matrix
        Measurement matrix, shape (M, N).
    Psi_approx : sp.csr_matrix
        Approximation coefficient extraction matrix, shape (n_approx, N).
    Psi_detail : sp.csr_matrix
        Detail coefficient extraction matrix, shape (n_detail, N).
    weights : np.ndarray
        Measurement weights, shape (M,).
    wavelength_grid : np.ndarray
        Wavelength grid, shape (N,).
    level_info : dict
        Wavelet decomposition information.
    config : SEDConfig
        Configuration with lambda grids and validation fraction.
    edge_info : dict, optional
        Edge padding information from wavelet boundary extension.

    Returns
    -------
    TuningResult
        Container with optimal hyperparameters and grid search results.

    Raises
    ------
    ValueError
        If all solver attempts fail.
    """
    logger.info("Starting grid search for optimal wavelet hyperparameters")
    logger.info(f"  λ_low grid (continuum): {config.lambda_low_grid}")
    logger.info(f"  λ_detail grid (noise): {config.lambda_detail_grid}")

    # Split data into train/validation
    train_data, val_data = split_train_validation(
        y, H, weights, config.validation_fraction
    )

    # Extract training and validation components
    y_train = train_data["y"]
    H_train = train_data["H"]
    weights_train = train_data["weights"]

    y_val = val_data["y"]
    H_val = val_data["H"]
    weights_val = val_data["weights"]

    # Grid search
    grid_results = []
    best_error = np.inf
    best_lambda_low = None
    best_lambda_detail = None
    n_failed = 0

    total_combinations = len(config.lambda_low_grid) * len(config.lambda_detail_grid)
    logger.info(f"Evaluating {total_combinations} hyperparameter combinations...")

    for i, lambda_low in enumerate(config.lambda_low_grid):
        for j, lambda_detail in enumerate(config.lambda_detail_grid):
            combo_idx = i * len(config.lambda_detail_grid) + j + 1
            logger.info(
                f"  [{combo_idx}/{total_combinations}] "
                f"λ_low={lambda_low:.2e}, λ_detail={lambda_detail:.2e}"
            )

            # Solve on training set
            result = solve_reconstruction(
                y=y_train,
                H=H_train,
                Psi_approx=Psi_approx,
                Psi_detail=Psi_detail,
                weights=weights_train,
                wavelength_grid=wavelength_grid,
                lambda_low=lambda_low,
                lambda_detail=lambda_detail,
                wavelet_info=level_info,
                config=config,
                edge_info=edge_info,
            )

            if not result.success:
                logger.warning(f"    Solver failed: {result.solver_status}")
                n_failed += 1
                grid_results.append({
                    "lambda_low": lambda_low,
                    "lambda_detail": lambda_detail,
                    "validation_error": np.inf,
                    "success": False,
                })
                continue

            # Evaluate on validation set
            val_error = compute_validation_error(
                y_val, H_val, weights_val, result.spectrum
            )

            logger.info(f"    Validation error: {val_error:.4f}")

            # Record result
            grid_results.append({
                "lambda_low": lambda_low,
                "lambda_detail": lambda_detail,
                "validation_error": val_error,
                "success": True,
            })

            # Update best
            if val_error < best_error:
                best_error = val_error
                best_lambda_low = lambda_low
                best_lambda_detail = lambda_detail
                logger.info(
                    f"    ** New best: λ_low={lambda_low:.2e}, λ_detail={lambda_detail:.2e}, "
                    f"error={val_error:.4f}"
                )

    # Check if any combinations succeeded
    if best_lambda_low is None:
        raise ValueError(
            f"All {total_combinations} hyperparameter combinations failed. "
            "Check solver settings or data quality."
        )

    logger.info("Grid search complete:")
    logger.info(f"  Optimal λ_low: {best_lambda_low:.2e}")
    logger.info(f"  Optimal λ_detail: {best_lambda_detail:.2e}")
    logger.info(f"  Validation error: {best_error:.4f}")
    logger.info(f"  Success rate: {total_combinations - n_failed}/{total_combinations}")

    # Package results
    tuning_result = TuningResult(
        optimal_lambda_low=best_lambda_low,
        optimal_lambda_detail=best_lambda_detail,
        optimal_validation_error=best_error,
        grid_results=grid_results,
        n_evaluated=total_combinations,
        n_failed=n_failed,
    )

    return tuning_result


def tune_and_reconstruct(
    y: np.ndarray,
    H: sp.csr_matrix,
    Psi_approx: sp.csr_matrix,
    Psi_detail: sp.csr_matrix,
    weights: np.ndarray,
    wavelength_grid: np.ndarray,
    level_info: Dict[str, int],
    config: SEDConfig,
    edge_info: Optional[Dict[str, Any]] = None,
) -> Tuple[TuningResult, np.ndarray]:
    """
    Perform grid search tuning and reconstruct spectrum with optimal hyperparameters.

    This is a convenience function that combines:
    1. Grid search to find optimal (lambda_low, lambda_detail)
    2. Final reconstruction on full dataset with optimal hyperparameters

    Parameters
    ----------
    y : np.ndarray
        Observed flux measurements, shape (M,).
    H : sp.csr_matrix
        Measurement matrix, shape (M, N).
    Psi_approx : sp.csr_matrix
        Approximation coefficient extraction matrix, shape (n_approx, N).
    Psi_detail : sp.csr_matrix
        Detail coefficient extraction matrix, shape (n_detail, N).
    weights : np.ndarray
        Measurement weights, shape (M,).
    wavelength_grid : np.ndarray
        Wavelength grid, shape (N,).
    level_info : dict
        Wavelet decomposition information.
    config : SEDConfig
        Configuration with tuning parameters.
    edge_info : dict, optional
        Edge padding information from wavelet boundary extension.

    Returns
    -------
    tuning_result : TuningResult
        Grid search results with optimal hyperparameters.
    spectrum : np.ndarray
        Reconstructed spectrum using optimal hyperparameters, shape (N,).

    Raises
    ------
    ValueError
        If tuning fails or final reconstruction fails.
    """
    # Run grid search
    tuning_result = grid_search(
        y, H, Psi_approx, Psi_detail, weights, wavelength_grid, level_info, config, edge_info
    )

    # Reconstruct on full dataset with optimal hyperparameters
    logger.info("Reconstructing on full dataset with optimal hyperparameters...")
    final_result = solve_reconstruction(
        y=y,
        H=H,
        Psi_approx=Psi_approx,
        Psi_detail=Psi_detail,
        weights=weights,
        wavelength_grid=wavelength_grid,
        lambda_low=tuning_result.optimal_lambda_low,
        lambda_detail=tuning_result.optimal_lambda_detail,
        wavelet_info=level_info,
        config=config,
        edge_info=edge_info,
    )

    if not final_result.success:
        raise ValueError(
            f"Final reconstruction failed with status: {final_result.solver_status}"
        )

    logger.info("Final reconstruction successful")

    return tuning_result, final_result.spectrum
