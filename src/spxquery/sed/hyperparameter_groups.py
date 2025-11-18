"""
4-Group Hyperparameter System for SWT-based Spectral Reconstruction.

This module implements automatic grouping of SWT hyperparameters into physically meaningful
categories based on frequency scale characteristics. This reduces the J+1 dimensional
parameter space to a manageable 4-dimensional space while preserving per-scale control.

The four groups correspond to physically distinct spectral features:
- Group A: Approximation coefficients (low-frequency continuum)
- Group B: Coarse detail coefficients (large-scale features)
- Group C: Medium detail coefficients (emission lines, main features)
- Group D: Fine detail coefficients (high-frequency noise)

This implements the grouping strategy described in DWT2SWT.md.
"""

import logging
from typing import List, Tuple

import numpy as np

from .config import SEDConfig

logger = logging.getLogger(__name__)


def create_default_lambda_vector(num_operators: int, config: SEDConfig) -> np.ndarray:
    """
    Create default lambda vector using 4-group automatic parameter mapping.

    Maps J+1 SWT operators to 4 physically meaningful groups:
    - Group A (index 0): Approximation coefficients - continuum
    - Group B (indices 1..n_B): Coarse details - large-scale features
    - Group C (indices n_B+1..n_B+n_C): Medium details - emission lines
    - Group D (remaining indices): Fine details - noise

    Parameters
    ----------
    num_operators : int
        Number of SWT operators (J+1 where J is decomposition level).
    config : SEDConfig
        Configuration with grouped hyperparameter values.

    Returns
    -------
    np.ndarray
        Lambda vector of shape (num_operators,) with grouped regularization weights.
    """
    if num_operators < 2:
        raise ValueError(f"Need at least 2 operators for grouping, got {num_operators}")

    # Map J+1 operators to 4 groups using strategy from DWT2SWT.md
    # For J=9 levels (10 operators): [A, D9, D8, D7, D6, D5, D4, D3, D2, D1]
    # Groups: A[0], B[1:5] (D9-D6), C[5:8] (D5-D3), D[8:10] (D2-D1)

    lambda_vec = np.zeros(num_operators)

    # Get grouping configuration (will be added to SEDConfig in next step)
    val_A = getattr(config, 'lambda_continuum', 0.1)
    val_B = getattr(config, 'lambda_low_features', 1.0)
    val_C = getattr(config, 'lambda_main_features', 5.0)
    val_D = getattr(config, 'lambda_noise', 100.0)

    if num_operators == 2:
        # Minimal case: J=1 (A, D1)
        lambda_vec[0] = val_A    # Approximation
        lambda_vec[1] = val_D    # Detail (treated as noise)
        logger.info(f"Minimal grouping (2 operators): A={val_A:.2e}, D={val_D:.2e}")

    elif num_operators <= 5:
        # Small case: split into A, B, D
        n_B = max(1, (num_operators - 1) // 2)
        n_D = num_operators - 1 - n_B

        lambda_vec[0] = val_A  # Approximation
        lambda_vec[1:1+n_B] = val_B  # Coarse details
        lambda_vec[1+n_B:] = val_D  # Fine details
        logger.info(f"Small grouping ({num_operators} operators): A={val_A:.2e}, B({n_B})={val_B:.2e}, D({n_D})={val_D:.2e}")

    else:
        # Full 4-group strategy
        n_B = max(2, (num_operators - 1) // 3)  # Coarse details
        n_C = max(2, (num_operators - 1 - n_B) // 2)  # Medium details
        n_D = num_operators - 1 - n_B - n_C  # Fine details

        lambda_vec[0] = val_A  # Approximation (Group A)
        lambda_vec[1:1+n_B] = val_B  # Coarse details (Group B)
        lambda_vec[1+n_B:1+n_B+n_C] = val_C  # Medium details (Group C)
        lambda_vec[1+n_B+n_C:] = val_D  # Fine details (Group D)

        logger.info(f"Full 4-group grouping ({num_operators} operators): A={val_A:.2e}, "
                   f"B({n_B})={val_B:.2e}, C({n_C})={val_C:.2e}, D({n_D})={val_D:.2e}")

    return lambda_vec


def create_grouped_lambda_vectors(
    num_operators: int,
    val_A: float,
    val_B: float,
    val_C: float,
    val_D: float
) -> List[np.ndarray]:
    """
    Create multiple lambda vectors for hyperparameter tuning using grouped values.

    Parameters
    ----------
    num_operators : int
        Number of SWT operators (J+1).
    val_A : float
        Regularization weight for approximation coefficients (continuum).
    val_B : float
        Regularization weight for coarse detail coefficients.
    val_C : float
        Regularization weight for medium detail coefficients.
    val_D : float
        Regularization weight for fine detail coefficients (noise).

    Returns
    -------
    List[np.ndarray]
        Single lambda vector with grouped values.
    """
    # Create a single lambda vector with specified grouped values
    lambda_vec = np.zeros(num_operators)

    if num_operators == 2:
        # Minimal case: A, D1
        lambda_vec[0] = val_A    # Approximation
        lambda_vec[1] = val_D    # Detail

    elif num_operators <= 5:
        # Small case: A, B, D
        n_B = max(1, (num_operators - 1) // 2)
        lambda_vec[0] = val_A  # Approximation
        lambda_vec[1:1+n_B] = val_B  # Coarse details
        lambda_vec[1+n_B:] = val_D  # Fine details

    else:
        # Full 4-group strategy
        n_B = max(2, (num_operators - 1) // 3)
        n_C = max(2, (num_operators - 1 - n_B) // 2)
        n_D = num_operators - 1 - n_B - n_C

        lambda_vec[0] = val_A  # Group A: Approximation
        lambda_vec[1:1+n_B] = val_B  # Group B: Coarse details
        lambda_vec[1+n_B:1+n_B+n_C] = val_C  # Group C: Medium details
        lambda_vec[1+n_B+n_C:] = val_D  # Group D: Fine details

    return [lambda_vec]


def generate_grouped_parameter_grid(
    num_operators: int,
    val_A_grid: List[float] = None,
    val_B_grid: List[float] = None,
    val_C_grid: List[float] = None,
    val_D_grid: List[float] = None
) -> Tuple[List[np.ndarray], List[Tuple[float, float, float, float]]]:
    """
    Generate hyperparameter grid for grouped parameter search.

    Creates all combinations of grouped hyperparameter values for grid search.
    This reduces the search from (J+1)-dimensional space to 4-dimensional space.

    Parameters
    ----------
    num_operators : int
        Number of SWT operators (J+1).
    val_A_grid : List[float], optional
        Grid values for approximation (continuum) regularization.
    val_B_grid : List[float], optional
        Grid values for coarse feature regularization.
    val_C_grid : List[float], optional
        Grid values for main feature regularization.
    val_D_grid : List[float], optional
        Grid values for noise suppression regularization.

    Returns
    -------
    lambda_vectors : List[np.ndarray]
        List of lambda vectors for grid search.
    param_combinations : List[Tuple[float, float, float, float]]
        Corresponding (val_A, val_B, val_C, val_D) combinations for logging.
    """
    # Default parameter grids
    if val_A_grid is None:
        val_A_grid = [0.01, 0.1, 1.0]  # Continuum: preserve structure
    if val_B_grid is None:
        val_B_grid = [0.1, 1.0, 10.0]  # Coarse features: moderate regularization
    if val_C_grid is None:
        val_C_grid = [1.0, 10.0, 100.0]  # Main features: stronger regularization
    if val_D_grid is None:
        val_D_grid = [10.0, 100.0, 1000.0]  # Noise: strong regularization

    lambda_vectors = []
    param_combinations = []

    # Generate all 4D combinations
    for val_A in val_A_grid:
        for val_B in val_B_grid:
            for val_C in val_C_grid:
                for val_D in val_D_grid:
                    lambda_vec = create_grouped_lambda_vectors(
                        num_operators, val_A, val_B, val_C, val_D
                    )[0]  # Take the first (and only) vector
                    lambda_vectors.append(lambda_vec)
                    param_combinations.append((val_A, val_B, val_C, val_D))

    logger.info(f"Generated grouped parameter grid: {len(lambda_vectors)} combinations "
               f"from {len(val_A_grid)}×{len(val_B_grid)}×{len(val_C_grid)}×{len(val_D_grid)}")

    return lambda_vectors, param_combinations


def get_lambda_vector_description(lambda_vector: np.ndarray, num_operators: int) -> str:
    """
    Generate human-readable description of lambda vector grouping.

    Parameters
    ----------
    lambda_vector : np.ndarray
        Lambda vector with grouped values.
    num_operators : int
        Number of SWT operators.

    Returns
    -------
    str
        Description of the grouping pattern.
    """
    if len(lambda_vector) != num_operators:
        raise ValueError("Lambda vector length doesn't match num_operators")

    if num_operators == 2:
        return f"A={lambda_vector[0]:.2e}, D={lambda_vector[1]:.2e}"
    elif num_operators <= 5:
        n_B = max(1, (num_operators - 1) // 2)
        val_A = lambda_vector[0]
        val_B = lambda_vector[1] if n_B > 0 else 0
        val_D = lambda_vector[-1] if n_B < num_operators - 1 else 0
        return f"A={val_A:.2e}, B={val_B:.2e}, D={val_D:.2e}"
    else:
        n_B = max(2, (num_operators - 1) // 3)
        n_C = max(2, (num_operators - 1 - n_B) // 2)

        val_A = lambda_vector[0]
        val_B = lambda_vector[1] if n_B > 0 else 0
        val_C = lambda_vector[1+n_B] if n_C > 0 else 0
        val_D = lambda_vector[-1] if n_B + n_C < num_operators - 1 else 0

        return f"A={val_A:.2e}, B={val_B:.2e}, C={val_C:.2e}, D={val_D:.2e}"


def validate_lambda_vector(lambda_vector: np.ndarray) -> bool:
    """
    Validate lambda vector for SWT regularization.

    Parameters
    ----------
    lambda_vector : np.ndarray
        Lambda vector to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    # Check basic properties
    if not isinstance(lambda_vector, np.ndarray):
        logger.error("Lambda vector must be numpy array")
        return False

    if len(lambda_vector) == 0:
        logger.error("Lambda vector cannot be empty")
        return False

    # Check for non-negative values
    if np.any(lambda_vector < 0):
        logger.error("All lambda values must be non-negative")
        return False

    # Check for infinite or NaN values
    if np.any(np.isinf(lambda_vector)) or np.any(np.isnan(lambda_vector)):
        logger.error("Lambda vector cannot contain inf or NaN values")
        return False

    return True


def explain_grouping_strategy(num_operators: int) -> str:
    """
    Explain the grouping strategy for given number of operators.

    Parameters
    ----------
    num_operators : int
        Number of SWT operators (J+1).

    Returns
    -------
    str
        Human-readable explanation of the grouping.
    """
    if num_operators < 2:
        return f"Too few operators ({num_operators}) for meaningful grouping"

    if num_operators == 2:
        return "2 operators: [A, D1] - Approximation + Detail (treated as noise)"
    elif num_operators <= 5:
        n_B = max(1, (num_operators - 1) // 2)
        n_D = num_operators - 1 - n_B
        return f"{num_operators} operators: [A, B×{n_B}, D×{n_D}] - Continuum + Coarse features + Noise"
    else:
        n_B = max(2, (num_operators - 1) // 3)
        n_C = max(2, (num_operators - 1 - n_B) // 2)
        n_D = num_operators - 1 - n_B - n_C
        return (f"{num_operators} operators: [A, B×{n_B}, C×{n_C}, D×{n_D}] - "
                "Continuum + Large features + Emission lines + Noise")