"""
Hyperparameter tuning for SED reconstruction with SWT regularization.

This module implements grid search over grouped SWT regularization weights
(lambda_continuum, lambda_low_features, lambda_main_features, lambda_noise)
to find optimal values that minimize validation error using a 4-group system.

The 4 groups correspond to physically meaningful spectral scales:
- Group A: Approximation coefficients (low-frequency continuum)
- Group B: Coarse detail coefficients (large-scale features)
- Group C: Medium detail coefficients (emission lines, main features)
- Group D: Fine detail coefficients (high-frequency noise)
"""

import logging
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import scipy.sparse as sp

from .config import SEDConfig
from .solver import solve_reconstruction
from .validation import split_train_validation, compute_validation_error
from .hyperparameter_groups import generate_grouped_parameter_grid, get_lambda_vector_description

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """
    Result container for hyperparameter tuning with SWT regularization.

    Attributes
    ----------
    optimal_lambda_vector : np.ndarray
        Best SWT regularization weight vector of shape (J+1,).
    optimal_group_params : Tuple[float, float, float, float]
        Best grouped hyperparameters (lambda_continuum, lambda_low_features,
        lambda_main_features, lambda_noise).
    optimal_validation_error : float
        Validation error at optimal hyperparameters.
    grid_results : List[Dict]
        List of dictionaries with keys:
        - 'lambda_vector': SWT regularization weight vector
        - 'group_params': (lambda_A, lambda_B, lambda_C, lambda_D) tuple
        - 'validation_error': Validation error
        - 'success': Whether solver converged
    n_evaluated : int
        Number of hyperparameter combinations evaluated.
    n_failed : int
        Number of combinations where solver failed.
    """

    optimal_lambda_vector: np.ndarray
    optimal_group_params: Tuple[float, float, float, float]
    optimal_validation_error: float
    grid_results: List[Dict]
    n_evaluated: int
    n_failed: int


def grid_search(
    y: np.ndarray,
    H: sp.csr_matrix,
    Psi_operators: List[sp.csr_matrix],
    weights: np.ndarray,
    wavelength_grid: np.ndarray,
    level_info: Dict[str, int],
    config: SEDConfig,
    edge_info: Optional[Dict[str, any]] = None,
) -> TuningResult:
    """
    Perform grid search to find optimal SWT regularization hyperparameters.

    The algorithm:
    1. Split data into training and validation sets
    2. Generate grouped hyperparameter combinations using 4-group system
    3. For each (lambda_A, lambda_B, lambda_C, lambda_D) combination:
       a. Map to full lambda vector using SWT operator grouping
       b. Solve reconstruction on training set
       c. Evaluate validation error on held-out validation set
    4. Select hyperparameters with lowest validation error

    Parameters
    ----------
    y : np.ndarray
        Observed flux measurements, shape (M,).
    H : sp.csr_matrix
        Measurement matrix, shape (M, N).
    Psi_operators : List[sp.csr_matrix]
        List of SWT coefficient extraction matrices [Ψ_0, Ψ_1, ..., Ψ_J].
    weights : np.ndarray
        Measurement weights, shape (M,).
    wavelength_grid : np.ndarray
        Wavelength grid, shape (N,).
    level_info : dict
        SWT decomposition information.
    config : SEDConfig
        Configuration with grouped lambda grids and validation fraction.
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
    logger.info("Starting grouped grid search for optimal SWT hyperparameters")
    logger.info(f"  λ_continuum grid: {config.lambda_continuum_grid}")
    logger.info(f"  λ_low_features grid: {config.lambda_low_features_grid}")
    logger.info(f"  λ_main_features grid: {config.lambda_main_features_grid}")
    logger.info(f"  λ_noise grid: {config.lambda_noise_grid}")

    # Generate grouped parameter combinations
    num_operators = len(Psi_operators)
    lambda_vectors, param_combinations = generate_grouped_parameter_grid(
        num_operators,
        config.lambda_continuum_grid,
        config.lambda_low_features_grid,
        config.lambda_main_features_grid,
        config.lambda_noise_grid
    )

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
    best_lambda_vector = None
    best_group_params = None
    n_failed = 0

    total_combinations = len(lambda_vectors)
    logger.info(f"Evaluating {total_combinations} grouped hyperparameter combinations...")

    for i, (lambda_vector, group_params) in enumerate(zip(lambda_vectors, param_combinations)):
        logger.info(
            f"  [{i+1}/{total_combinations}] "
            f"Group params: {get_lambda_vector_description(lambda_vector, num_operators)}"
        )

        # Solve on training set
        result = solve_reconstruction(
            y=y_train,
            H=H_train,
            Psi_operators=Psi_operators,
            weights=weights_train,
            wavelength_grid=wavelength_grid,
            lambda_vector=lambda_vector,
            wavelet_info=level_info,
            config=config,
            edge_info=edge_info,
        )

        if not result.success:
            logger.warning(f"    Solver failed: {result.solver_status}")
            n_failed += 1
            grid_results.append({
                "lambda_vector": lambda_vector,
                "group_params": group_params,
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
            "lambda_vector": lambda_vector,
            "group_params": group_params,
            "validation_error": val_error,
            "success": True,
        })

        # Update best
        if val_error < best_error:
            best_error = val_error
            best_lambda_vector = lambda_vector
            best_group_params = group_params
            logger.info(
                f"    ** New best: {get_lambda_vector_description(lambda_vector, num_operators)}, "
                f"error={val_error:.4f}"
            )

    # Check if any combinations succeeded
    if best_lambda_vector is None:
        raise ValueError(
            f"All {total_combinations} hyperparameter combinations failed. "
            "Check solver settings or data quality."
        )

    logger.info("Grid search complete:")
    logger.info(f"  Optimal group params: {get_lambda_vector_description(best_lambda_vector, num_operators)}")
    logger.info(f"  Validation error: {best_error:.4f}")
    logger.info(f"  Success rate: {total_combinations - n_failed}/{total_combinations}")

    # Package results
    tuning_result = TuningResult(
        optimal_lambda_vector=best_lambda_vector,
        optimal_group_params=best_group_params,
        optimal_validation_error=best_error,
        grid_results=grid_results,
        n_evaluated=total_combinations,
        n_failed=n_failed,
    )

    return tuning_result


def tune_and_reconstruct(
    y: np.ndarray,
    H: sp.csr_matrix,
    Psi_operators: List[sp.csr_matrix],
    weights: np.ndarray,
    wavelength_grid: np.ndarray,
    level_info: Dict[str, int],
    config: SEDConfig,
    edge_info: Optional[Dict[str, Any]] = None,
) -> Tuple[TuningResult, np.ndarray]:
    """
    Perform grid search tuning and reconstruct spectrum with optimal hyperparameters.

    This is a convenience function that combines:
    1. Grid search to find optimal grouped SWT hyperparameters
    2. Final reconstruction on full dataset with optimal hyperparameters

    Parameters
    ----------
    y : np.ndarray
        Observed flux measurements, shape (M,).
    H : sp.csr_matrix
        Measurement matrix, shape (M, N).
    Psi_operators : List[sp.csr_matrix]
        List of SWT coefficient extraction matrices [Ψ_0, Ψ_1, ..., Ψ_J].
    weights : np.ndarray
        Measurement weights, shape (M,).
    wavelength_grid : np.ndarray
        Wavelength grid, shape (N,).
    level_info : dict
        SWT decomposition information.
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
        y, H, Psi_operators, weights, wavelength_grid, level_info, config, edge_info
    )

    # Reconstruct on full dataset with optimal hyperparameters
    logger.info("Reconstructing on full dataset with optimal hyperparameters...")
    final_result = solve_reconstruction(
        y=y,
        H=H,
        Psi_operators=Psi_operators,
        weights=weights,
        wavelength_grid=wavelength_grid,
        lambda_vector=tuning_result.optimal_lambda_vector,
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
