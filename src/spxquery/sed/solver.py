"""
CVXPY-based convex optimization solver for SED reconstruction.

This module implements wavelet-based multi-scale regularization:
    min_x ( ||w * (y - H @ x)||_2^2 + lambda_low * ||Psi_approx @ x||_1 + lambda_detail * ||Psi_detail @ x||_1 )

where:
    - Data fidelity term: weighted L2 norm (chi-squared)
    - Approximation regularization: L1 penalty on low-frequency wavelet coefficients (continuum)
    - Detail regularization: L1 penalty on high-frequency wavelet coefficients (noise suppression)
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
    Result container for single-band spectral reconstruction using wavelet regularization.

    Attributes
    ----------
    spectrum : np.ndarray
        Reconstructed flux density, shape (N,), in microJansky.
    wavelength_grid : np.ndarray
        Wavelength grid for spectrum, shape (N,), in microns.
    lambda_low : float
        Approximation coefficient regularization weight used.
    lambda_detail : float
        Detail coefficient regularization weight used.
    solver_status : str
        CVXPY solver status ('optimal', 'optimal_inaccurate', 'infeasible', etc.).
    solver_time : float
        Wall-clock time for solver in seconds.
    objective_value : float
        Final objective function value.
    data_fidelity : float
        Data fidelity term value (weighted residual sum of squares).
    approx_penalty : float
        Approximation coefficient regularization term value.
    detail_penalty : float
        Detail coefficient regularization term value.
    wavelet_info : dict
        Wavelet decomposition information (level, coefficient counts).
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
    lambda_low: float
    lambda_detail: float
    solver_status: str
    solver_time: float
    objective_value: float
    data_fidelity: float
    approx_penalty: float
    detail_penalty: float
    wavelet_info: Dict[str, int]
    success: bool
    edge_info: Optional[Dict[str, any]] = None


def solve_reconstruction(
    y: np.ndarray,
    H: sp.csr_matrix,
    Psi_approx: sp.csr_matrix,
    Psi_detail: sp.csr_matrix,
    weights: np.ndarray,
    wavelength_grid: np.ndarray,
    lambda_low: float,
    lambda_detail: float,
    wavelet_info: Dict[str, int],
    config: SEDConfig,
    edge_info: Optional[Dict[str, any]] = None,
) -> ReconstructionResult:
    """
    Solve wavelet-regularized reconstruction problem using CVXPY.

    Minimizes:
        ||w * (y - H @ x)||_2^2 + lambda_low * ||Psi_approx @ x||_1 + lambda_detail * ||Psi_detail @ x||_1

    Parameters
    ----------
    y : np.ndarray
        Observed flux measurements, shape (M,), in microJansky.
    H : sp.csr_matrix
        Measurement matrix, shape (M, N).
    Psi_approx : sp.csr_matrix
        Approximation coefficient extraction matrix, shape (n_approx, N).
    Psi_detail : sp.csr_matrix
        Detail coefficient extraction matrix, shape (n_detail, N).
    weights : np.ndarray
        Measurement weights (1/sigma), shape (M,).
    wavelength_grid : np.ndarray
        Wavelength grid for reconstructed spectrum, shape (N,), in microns.
    lambda_low : float
        Approximation coefficient regularization weight (non-negative).
    lambda_detail : float
        Detail coefficient regularization weight (non-negative).
    wavelet_info : dict
        Wavelet decomposition information (level, coefficient counts).
    config : SEDConfig
        Configuration with solver settings.
    edge_info : dict, optional
        Edge padding information from wavelet boundary extension.

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
    n_approx, _ = Psi_approx.shape
    n_detail, _ = Psi_detail.shape

    if y.shape != (M,):
        raise ValueError(f"y shape {y.shape} inconsistent with H shape {H.shape}")
    if weights.shape != (M,):
        raise ValueError(f"weights shape {weights.shape} inconsistent with H shape {H.shape}")
    if Psi_approx.shape[1] != N:
        raise ValueError(f"Psi_approx shape {Psi_approx.shape} inconsistent with N={N}")
    if Psi_detail.shape[1] != N:
        raise ValueError(f"Psi_detail shape {Psi_detail.shape} inconsistent with N={N}")
    if wavelength_grid.shape != (N,):
        raise ValueError(f"wavelength_grid shape {wavelength_grid.shape} inconsistent with N={N}")

    logger.info(
        f"Setting up wavelet CVXPY problem: M={M} measurements, N={N} wavelength bins"
    )
    logger.info(
        f"  Wavelet coefficients: {n_approx} approx, {n_detail} detail (level {wavelet_info['level']})"
    )
    logger.info(f"  Regularization: lambda_low={lambda_low:.2e}, lambda_detail={lambda_detail:.2e}")

    # Define optimization variable
    x = cp.Variable(N, name="spectrum")

    # Build objective function terms
    # 1. Data fidelity (weighted chi-squared)
    residual = y - H @ x
    weighted_residual = cp.multiply(weights, residual)
    data_fidelity = cp.sum_squares(weighted_residual)

    # 2. Approximation coefficient regularization (continuum)
    approx_coeffs = Psi_approx @ x
    approx_regularization = lambda_low * cp.norm1(approx_coeffs)

    # 3. Detail coefficient regularization (noise suppression)
    detail_coeffs = Psi_detail @ x
    detail_regularization = lambda_detail * cp.norm1(detail_coeffs)

    # Total objective
    objective = cp.Minimize(data_fidelity + approx_regularization + detail_regularization)

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
            lambda_low=lambda_low,
            lambda_detail=lambda_detail,
            solver_status="error",
            solver_time=solver_time,
            objective_value=np.nan,
            data_fidelity=np.nan,
            approx_penalty=np.nan,
            detail_penalty=np.nan,
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

        approx_coeffs_val = Psi_approx @ spectrum
        approx_penalty_val = lambda_low * np.sum(np.abs(approx_coeffs_val))

        detail_coeffs_val = Psi_detail @ spectrum
        detail_penalty_val = lambda_detail * np.sum(np.abs(detail_coeffs_val))

        logger.info(f"Solver converged in {solver_time:.2f}s: status={status}, objective={obj_value:.2e}")
        logger.info(
            f"  Data fidelity: {data_fidelity_val:.2e}, "
            f"Approx penalty: {approx_penalty_val:.2e}, "
            f"Detail penalty: {detail_penalty_val:.2e}"
        )

    else:
        logger.warning(f"Solver did not converge: status={status}")
        spectrum = np.full(N, np.nan) if x.value is None else x.value
        obj_value = problem.value if problem.value is not None else np.nan
        data_fidelity_val = np.nan
        approx_penalty_val = np.nan
        detail_penalty_val = np.nan

    # Package results
    result = ReconstructionResult(
        spectrum=spectrum,
        wavelength_grid=wavelength_grid,
        lambda_low=lambda_low,
        lambda_detail=lambda_detail,
        solver_status=status,
        solver_time=solver_time,
        objective_value=obj_value,
        data_fidelity=data_fidelity_val,
        approx_penalty=approx_penalty_val,
        detail_penalty=detail_penalty_val,
        wavelet_info=wavelet_info,
        success=success,
        edge_info=edge_info,
    )

    return result


def reconstruct_single_band(
    y: np.ndarray,
    H: sp.csr_matrix,
    Psi_approx: sp.csr_matrix,
    Psi_detail: sp.csr_matrix,
    weights: np.ndarray,
    wavelength_grid: np.ndarray,
    wavelet_info: Dict[str, int],
    config: SEDConfig,
    lambda_low: Optional[float] = None,
    lambda_detail: Optional[float] = None,
    edge_info: Optional[Dict[str, any]] = None,
) -> ReconstructionResult:
    """
    Reconstruct spectrum for a single band using wavelet regularization.

    This is a convenience wrapper around solve_reconstruction that handles
    hyperparameter selection from config if not explicitly provided.

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
    wavelet_info : dict
        Wavelet decomposition information.
    config : SEDConfig
        Configuration with default hyperparameters.
    lambda_low : float, optional
        Override approximation regularization weight. If None, uses config.lambda_low.
    lambda_detail : float, optional
        Override detail regularization weight. If None, uses config.lambda_detail.
    edge_info : dict, optional
        Edge padding information from wavelet boundary extension.

    Returns
    -------
    ReconstructionResult
        Reconstruction result with spectrum and diagnostics.
    """
    # Use provided hyperparameters or defaults from config
    l_low = lambda_low if lambda_low is not None else config.lambda_low
    l_detail = lambda_detail if lambda_detail is not None else config.lambda_detail

    result = solve_reconstruction(
        y=y,
        H=H,
        Psi_approx=Psi_approx,
        Psi_detail=Psi_detail,
        weights=weights,
        wavelength_grid=wavelength_grid,
        lambda_low=l_low,
        lambda_detail=l_detail,
        wavelet_info=wavelet_info,
        config=config,
        edge_info=edge_info,
    )

    return result
