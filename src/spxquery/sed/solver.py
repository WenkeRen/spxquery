"""
CVXPY-based convex optimization solver for SED reconstruction.

This module implements Stationary Wavelet Transform (SWT) based multi-scale regularization:
    min_x ( ||w * (y - H @ x)||_2^2 + Σ_{j=0}^{J} λ_j * ||Ψ_j @ x||_1 )

where:
    - Data fidelity term: weighted L2 norm (chi-squared)
    - Per-scale regularization: L1 penalty on SWT coefficients at each frequency scale
    - Ψ_0: Approximation coefficients (low-frequency continuum)
    - Ψ_1..J: Detail coefficients at different frequency scales (coarse to fine)
    - λ_0..λ_J: Vector hyperparameters controlling regularization per scale
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from .config import SEDConfig

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionResult:
    """
    Result container for single-band spectral reconstruction using SWT regularization.

    Attributes
    ----------
    spectrum : np.ndarray
        Reconstructed flux density, shape (N,), in microJansky.
    wavelength_grid : np.ndarray
        Wavelength grid for spectrum, shape (N,), in microns.
    lambda_vector : np.ndarray
        Vector of SWT regularization weights used, shape (J+1,).
        lambda_vector[0]: approximation coefficient regularization (continuum)
        lambda_vector[1:]: detail coefficient regularization for each scale
    solver_status : str
        CVXPY solver status ('optimal', 'optimal_inaccurate', 'infeasible', etc.).
    solver_time : float
        Wall-clock time for solver in seconds.
    objective_value : float
        Final objective function value.
    data_fidelity : float
        Data fidelity term value (weighted residual sum of squares).
    total_penalty : float
        Total SWT regularization term value (sum over all scales).
    per_scale_penalties : np.ndarray
        Regularization term values for each SWT scale, shape (J+1,).
    wavelet_info : dict
        SWT decomposition information (level, coefficient counts).
    success : bool
        True if solver converged to optimal solution.
    edge_info : dict, optional
        Edge padding information with keys:
        - 'edge_padding_pixels': Number of pixels padded on each side
        - 'trim_start': Index where valid detector range starts
        - 'trim_end': Index where valid detector range ends
        - 'wavelength_grid_trimmed': Trimmed grid matching detector range
        - 'detector_range': Hardcoded detector wavelength range
    """

    spectrum: np.ndarray
    wavelength_grid: np.ndarray
    lambda_vector: np.ndarray
    solver_status: str
    solver_time: float
    objective_value: float
    data_fidelity: float
    total_penalty: float
    per_scale_penalties: np.ndarray
    wavelet_info: Dict[str, int]
    success: bool
    edge_info: Optional[Dict[str, any]] = None


def solve_reconstruction(
    y: np.ndarray,
    H: sp.csr_matrix,
    Psi_operators: list,
    weights: np.ndarray,
    wavelength_grid: np.ndarray,
    lambda_vector: np.ndarray,
    wavelet_info: Dict[str, int],
    config: SEDConfig,
    edge_info: Optional[Dict[str, any]] = None,
    spatial_weights: Optional[np.ndarray] = None,
) -> ReconstructionResult:
    """
    Solve SWT-regularized reconstruction problem using CVXPY.

    Minimizes:
        ||w * (y - H @ x)||_2^2 + Σ_{j=0}^{J} λ_j * ||Ψ_j @ x * w_spatial||_1

    where Ψ_j are the SWT coefficient extraction matrices:
    - Ψ_0: Approximation coefficients (continuum)
    - Ψ_1..J: Detail coefficients at different frequency scales
    - w_spatial: Spatial weight vector to reduce Gibbs Phenomenon at edges

    Parameters
    ----------
    y : np.ndarray
        Observed flux measurements, shape (M,), in microJansky.
    H : sp.csr_matrix
        Measurement matrix, shape (M, N).
    Psi_operators : list of sp.csr_matrix
        List of SWT coefficient extraction matrices [Ψ_0, Ψ_1, ..., Ψ_J],
        where each matrix has shape (n_coeffs_j, N).
    weights : np.ndarray
        Measurement weights (1/sigma), shape (M,).
    wavelength_grid : np.ndarray
        Wavelength grid for reconstructed spectrum, shape (N,), in microns.
    lambda_vector : np.ndarray
        Vector of regularization weights, shape (J+1,).
        lambda_vector[0]: approximation coefficients (continuum)
        lambda_vector[1:]: detail coefficients at each frequency scale
    wavelet_info : dict
        SWT decomposition information (level, coefficient counts).
    config : SEDConfig
        Configuration with solver settings.
    edge_info : dict, optional
        Edge padding information from wavelet boundary extension.
    spatial_weights : np.ndarray, optional
        Spatial weight vector for L1 regularization, shape (N,).
        Applied element-wise to SWT coefficients to reduce Gibbs Phenomenon.

    Returns
    -------
    ReconstructionResult
        Container with reconstructed spectrum and solver diagnostics.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent.
    """
    # Validate dimensions
    M, N = H.shape
    num_operators = len(Psi_operators)

    if y.shape != (M,):
        raise ValueError(f"y shape {y.shape} inconsistent with H shape {H.shape}")
    if weights.shape != (M,):
        raise ValueError(f"weights shape {weights.shape} inconsistent with H shape {H.shape}")
    if lambda_vector.shape != (num_operators,):
        raise ValueError(f"lambda_vector shape {lambda_vector.shape} inconsistent with num_operators={num_operators}")
    if wavelength_grid.shape != (N,):
        raise ValueError(f"wavelength_grid shape {wavelength_grid.shape} inconsistent with N={N}")

    # Validate each SWT operator
    for i, Psi in enumerate(Psi_operators):
        if Psi.shape[1] != N:
            raise ValueError(f"Psi_operators[{i}] shape {Psi.shape} inconsistent with N={N}")

    logger.info(
        f"Setting up SWT CVXPY problem: M={M} measurements, N={N} wavelength bins"
    )
    logger.info(
        f"  SWT operators: {num_operators} matrices (level {wavelet_info['level']})"
    )
    logger.info(f"  Regularization: lambda_vector shape {lambda_vector.shape}, values {lambda_vector}")

    # Define optimization variable
    x = cp.Variable(N, name="spectrum")

    # Build objective function terms
    # 1. Data fidelity (weighted chi-squared)
    residual = y - H @ x
    weighted_residual = cp.multiply(weights, residual)
    data_fidelity = cp.sum_squares(weighted_residual)

    # 2. Per-scale SWT regularization with spatial weighting
    regularization_terms = []
    for i, Psi in enumerate(Psi_operators):
        coeffs_i = Psi @ x
        lambda_i = lambda_vector[i]

        # Apply spatial weights if provided to reduce Gibbs Phenomenon at edges
        if spatial_weights is not None:
            weighted_coeffs_i = cp.multiply(spatial_weights, coeffs_i)
            reg_term_i = lambda_i * cp.norm1(weighted_coeffs_i)
        else:
            reg_term_i = lambda_i * cp.norm1(coeffs_i)

        regularization_terms.append(reg_term_i)

    # Total regularization is sum over all scales
    total_regularization = cp.sum(regularization_terms)

    # Total objective
    objective = cp.Minimize(data_fidelity + total_regularization)

    # Define problem
    problem = cp.Problem(objective)

    # Solve
    logger.info(f"Solving with {config.solver} solver...")
    start_time = time.time()

    try:
        problem.solve(
            solver=config.solver,
            verbose=config.solver_verbose,
        )
        solver_time = time.time() - start_time

    except Exception as e:
        logger.error(f"Solver failed with exception: {e}")
        solver_time = time.time() - start_time

        # Return failure result
        return ReconstructionResult(
            spectrum=np.full(N, np.nan),
            wavelength_grid=wavelength_grid,
            lambda_vector=lambda_vector,
            solver_status="error",
            solver_time=solver_time,
            objective_value=np.nan,
            data_fidelity=np.nan,
            total_penalty=np.nan,
            per_scale_penalties=np.full(num_operators, np.nan),
            wavelet_info=wavelet_info,
            success=False,
            edge_info=edge_info,
        )

    # Extract solution
    status = problem.status
    success = status in ["optimal", "optimal_inaccurate"]

    if success:
        spectrum = x.value
        obj_value = problem.value

        # Compute individual term values for diagnostics
        residual_val = y - H @ spectrum
        weighted_residual_val = weights * residual_val
        data_fidelity_val = np.sum(weighted_residual_val**2)

        # Compute per-scale regularization penalties
        per_scale_penalties_val = []
        for i, Psi in enumerate(Psi_operators):
            coeffs_val = Psi @ spectrum

            # Apply spatial weights if provided
            if spatial_weights is not None:
                weighted_coeffs_val = spatial_weights * coeffs_val
                penalty_val = lambda_vector[i] * np.sum(np.abs(weighted_coeffs_val))
            else:
                penalty_val = lambda_vector[i] * np.sum(np.abs(coeffs_val))

            per_scale_penalties_val.append(penalty_val)

        per_scale_penalties_val = np.array(per_scale_penalties_val)
        total_penalty_val = np.sum(per_scale_penalties_val)

        logger.info(f"Solver converged in {solver_time:.2f}s: status={status}, objective={obj_value:.2e}")
        logger.info(
            f"  Data fidelity: {data_fidelity_val:.2e}, "
            f"Total penalty: {total_penalty_val:.2e}"
        )
        logger.info(f"  Per-scale penalties: {per_scale_penalties_val}")

    else:
        logger.warning(f"Solver did not converge: status={status}")
        spectrum = np.full(N, np.nan) if x.value is None else x.value
        obj_value = problem.value if problem.value is not None else np.nan
        data_fidelity_val = np.nan
        per_scale_penalties_val = np.full(num_operators, np.nan)
        total_penalty_val = np.nan

    # Package results
    result = ReconstructionResult(
        spectrum=spectrum,
        wavelength_grid=wavelength_grid,
        lambda_vector=lambda_vector,
        solver_status=status,
        solver_time=solver_time,
        objective_value=obj_value,
        data_fidelity=data_fidelity_val,
        total_penalty=total_penalty_val,
        per_scale_penalties=per_scale_penalties_val,
        wavelet_info=wavelet_info,
        success=success,
        edge_info=edge_info,
    )

    return result


def reconstruct_single_band(
    y: np.ndarray,
    H: sp.csr_matrix,
    Psi_operators: list,
    weights: np.ndarray,
    wavelength_grid: np.ndarray,
    wavelet_info: Dict[str, int],
    config: SEDConfig,
    lambda_vector: Optional[np.ndarray] = None,
    edge_info: Optional[Dict[str, any]] = None,
    spatial_weights: Optional[np.ndarray] = None,
) -> ReconstructionResult:
    """
    Reconstruct spectrum for a single band using SWT regularization.

    This is a convenience wrapper around solve_reconstruction that handles
    hyperparameter selection from config if not explicitly provided.

    Parameters
    ----------
    y : np.ndarray
        Observed flux measurements, shape (M,).
    H : sp.csr_matrix
        Measurement matrix, shape (M, N).
    Psi_operators : list of sp.csr_matrix
        List of SWT coefficient extraction matrices [Ψ_0, Ψ_1, ..., Ψ_J].
    weights : np.ndarray
        Measurement weights, shape (M,).
    wavelength_grid : np.ndarray
        Wavelength grid, shape (N,).
    wavelet_info : dict
        SWT decomposition information.
    config : SEDConfig
        Configuration with default hyperparameters.
    lambda_vector : np.ndarray, optional
        Override regularization weight vector. If None, uses default grouping strategy.
    edge_info : dict, optional
        Edge padding information from wavelet boundary extension.
    spatial_weights : np.ndarray, optional
        Spatial weight vector for L1 regularization, shape (N,).
        Applied element-wise to SWT coefficients to reduce Gibbs Phenomenon.

    Returns
    -------
    ReconstructionResult
        Reconstruction result with spectrum and diagnostics.
    """
    # Use provided hyperparameters or create default using grouping strategy
    if lambda_vector is None:
        from .hyperparameter_groups import create_default_lambda_vector
        lambda_vector = create_default_lambda_vector(len(Psi_operators), config)

    result = solve_reconstruction(
        y=y,
        H=H,
        Psi_operators=Psi_operators,
        weights=weights,
        wavelength_grid=wavelength_grid,
        lambda_vector=lambda_vector,
        wavelet_info=wavelet_info,
        config=config,
        edge_info=edge_info,
        spatial_weights=spatial_weights,
    )

    return result
