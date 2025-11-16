"""
Validation and quality assessment for SED reconstruction.

This module provides functions to evaluate reconstruction quality through
residual analysis, chi-squared statistics, and goodness-of-fit metrics.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import scipy.sparse as sp
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """
    Container for reconstruction quality metrics.

    Attributes
    ----------
    chi_squared : float
        Sum of squared weighted residuals.
    chi_squared_reduced : float
        Chi-squared divided by degrees of freedom.
    degrees_of_freedom : int
        Number of measurements minus number of parameters (M - N).
    residuals : np.ndarray
        Raw residuals (y - H @ x), shape (M,).
    weighted_residuals : np.ndarray
        Weighted residuals w * (y - H @ x), shape (M,).
    residual_mean : float
        Mean of raw residuals.
    residual_std : float
        Standard deviation of raw residuals.
    weighted_residual_mean : float
        Mean of weighted residuals (should be ~0).
    weighted_residual_std : float
        Standard deviation of weighted residuals (should be ~1).
    max_residual : float
        Maximum absolute raw residual.
    normality_pvalue : float
        P-value from Shapiro-Wilk test on weighted residuals.
        High p-value (>0.05) suggests Gaussian residuals.
    """

    chi_squared: float
    chi_squared_reduced: float
    degrees_of_freedom: int
    residuals: np.ndarray
    weighted_residuals: np.ndarray
    residual_mean: float
    residual_std: float
    weighted_residual_mean: float
    weighted_residual_std: float
    max_residual: float
    normality_pvalue: float


def compute_residuals(
    y: np.ndarray, H: sp.csr_matrix, spectrum: np.ndarray
) -> np.ndarray:
    """
    Compute raw residuals between observations and model.

    Residuals are: r = y - H @ x

    Parameters
    ----------
    y : np.ndarray
        Observed flux measurements, shape (M,).
    H : sp.csr_matrix
        Measurement matrix, shape (M, N).
    spectrum : np.ndarray
        Reconstructed spectrum, shape (N,).

    Returns
    -------
    np.ndarray
        Residuals, shape (M,).
    """
    y_model = H @ spectrum
    residuals = y - y_model
    return residuals


def compute_weighted_residuals(
    residuals: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Compute weighted residuals.

    Weighted residuals are: r_weighted = w * r = w * (y - H @ x)

    If weights are w = 1/sigma, then weighted residuals should be
    approximately standard normal distributed N(0, 1) if the model is correct.

    Parameters
    ----------
    residuals : np.ndarray
        Raw residuals, shape (M,).
    weights : np.ndarray
        Measurement weights (1/sigma), shape (M,).

    Returns
    -------
    np.ndarray
        Weighted residuals, shape (M,).
    """
    weighted_residuals = weights * residuals
    return weighted_residuals


def compute_chi_squared(weighted_residuals: np.ndarray) -> float:
    """
    Compute chi-squared statistic.

    Chi-squared is the sum of squared weighted residuals:
        chi^2 = sum((y - H @ x)^2 / sigma^2)

    Parameters
    ----------
    weighted_residuals : np.ndarray
        Weighted residuals, shape (M,).

    Returns
    -------
    float
        Chi-squared value.
    """
    chi_squared = np.sum(weighted_residuals**2)
    return chi_squared


def compute_reduced_chi_squared(
    chi_squared: float, n_measurements: int, n_parameters: int
) -> float:
    """
    Compute reduced chi-squared (chi^2 / dof).

    Degrees of freedom (dof) = M - N, where M is number of measurements
    and N is number of parameters (wavelength bins in reconstructed spectrum).

    A reduced chi-squared close to 1.0 indicates a good fit with appropriate
    error estimates. Values >> 1 suggest poor fit or underestimated errors.
    Values << 1 suggest overfitting or overestimated errors.

    Parameters
    ----------
    chi_squared : float
        Chi-squared value.
    n_measurements : int
        Number of measurements M.
    n_parameters : int
        Number of parameters N (wavelength bins).

    Returns
    -------
    float
        Reduced chi-squared.
    """
    dof = n_measurements - n_parameters
    if dof <= 0:
        logger.warning(
            f"Degrees of freedom <= 0 (M={n_measurements}, N={n_parameters}). "
            "Cannot compute reduced chi-squared."
        )
        return np.nan

    reduced_chi_squared = chi_squared / dof
    return reduced_chi_squared


def assess_reconstruction_quality(
    y: np.ndarray,
    H: sp.csr_matrix,
    spectrum: np.ndarray,
    weights: np.ndarray,
) -> ValidationMetrics:
    """
    Compute comprehensive quality metrics for reconstructed spectrum.

    This function computes:
    - Raw and weighted residuals
    - Chi-squared and reduced chi-squared
    - Residual statistics (mean, std, max)
    - Normality test on weighted residuals

    Parameters
    ----------
    y : np.ndarray
        Observed flux measurements, shape (M,).
    H : sp.csr_matrix
        Measurement matrix, shape (M, N).
    spectrum : np.ndarray
        Reconstructed spectrum, shape (N,).
    weights : np.ndarray
        Measurement weights, shape (M,).

    Returns
    -------
    ValidationMetrics
        Container with all quality metrics.
    """
    M, N = H.shape

    # Compute residuals
    residuals = compute_residuals(y, H, spectrum)
    weighted_residuals = compute_weighted_residuals(residuals, weights)

    # Chi-squared statistics
    chi_squared = compute_chi_squared(weighted_residuals)
    chi_squared_reduced = compute_reduced_chi_squared(chi_squared, M, N)
    dof = M - N

    # Residual statistics
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    weighted_residual_mean = np.mean(weighted_residuals)
    weighted_residual_std = np.std(weighted_residuals)
    max_residual = np.max(np.abs(residuals))

    # Normality test on weighted residuals
    # Shapiro-Wilk test: null hypothesis is that data is normally distributed
    # High p-value (>0.05) suggests residuals are consistent with Gaussian
    if len(weighted_residuals) >= 3:  # Minimum for Shapiro-Wilk
        try:
            _, normality_pvalue = stats.shapiro(weighted_residuals)
        except Exception as e:
            logger.warning(f"Normality test failed: {e}")
            normality_pvalue = np.nan
    else:
        normality_pvalue = np.nan

    # Package metrics
    metrics = ValidationMetrics(
        chi_squared=chi_squared,
        chi_squared_reduced=chi_squared_reduced,
        degrees_of_freedom=dof,
        residuals=residuals,
        weighted_residuals=weighted_residuals,
        residual_mean=residual_mean,
        residual_std=residual_std,
        weighted_residual_mean=weighted_residual_mean,
        weighted_residual_std=weighted_residual_std,
        max_residual=max_residual,
        normality_pvalue=normality_pvalue,
    )

    # Log summary
    logger.info("Reconstruction quality metrics:")
    logger.info(f"  Chi-squared: {chi_squared:.2f} (dof={dof})")
    logger.info(f"  Reduced chi-squared: {chi_squared_reduced:.3f}")
    logger.info(f"  Weighted residuals: mean={weighted_residual_mean:.3f}, std={weighted_residual_std:.3f}")
    logger.info(f"  Normality p-value: {normality_pvalue:.3f}" if not np.isnan(normality_pvalue) else "  Normality p-value: N/A")

    # Interpret reduced chi-squared
    if 0.5 <= chi_squared_reduced <= 2.0:
        logger.info("  -> Good fit (reduced chi-squared near 1)")
    elif chi_squared_reduced > 2.0:
        logger.warning("  -> Poor fit or underestimated errors (reduced chi-squared >> 1)")
    else:
        logger.warning("  -> Possible overfitting or overestimated errors (reduced chi-squared << 1)")

    return metrics


def split_train_validation(
    y: np.ndarray,
    H: sp.csr_matrix,
    weights: np.ndarray,
    validation_fraction: float,
    random_seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split data into training and validation sets for hyperparameter tuning.

    Parameters
    ----------
    y : np.ndarray
        Observed flux measurements, shape (M,).
    H : sp.csr_matrix
        Measurement matrix, shape (M, N).
    weights : np.ndarray
        Measurement weights, shape (M,).
    validation_fraction : float
        Fraction of data to reserve for validation (e.g., 0.2 for 80/20 split).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_data : Dict[str, np.ndarray]
        Dictionary with keys 'y', 'H', 'weights' for training set.
    val_data : Dict[str, np.ndarray]
        Dictionary with keys 'y', 'H', 'weights' for validation set.
    """
    M = len(y)
    n_val = int(M * validation_fraction)
    n_train = M - n_val

    # Generate random indices
    rng = np.random.RandomState(random_seed)
    indices = rng.permutation(M)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Split data
    train_data = {
        "y": y[train_indices],
        "H": H[train_indices, :],
        "weights": weights[train_indices],
    }

    val_data = {
        "y": y[val_indices],
        "H": H[val_indices, :],
        "weights": weights[val_indices],
    }

    logger.info(
        f"Split data: {n_train} training ({100 * (1 - validation_fraction):.0f}%), "
        f"{n_val} validation ({100 * validation_fraction:.0f}%)"
    )

    return train_data, val_data


def compute_validation_error(
    y_val: np.ndarray,
    H_val: sp.csr_matrix,
    weights_val: np.ndarray,
    spectrum: np.ndarray,
) -> float:
    """
    Compute validation error (weighted RMSE) for a reconstructed spectrum.

    Validation error is the root mean squared weighted residual:
        error = sqrt(mean((w * (y - H @ x))^2))

    Parameters
    ----------
    y_val : np.ndarray
        Validation flux measurements, shape (M_val,).
    H_val : sp.csr_matrix
        Validation measurement matrix, shape (M_val, N).
    weights_val : np.ndarray
        Validation weights, shape (M_val,).
    spectrum : np.ndarray
        Reconstructed spectrum, shape (N,).

    Returns
    -------
    float
        Validation error (weighted RMSE).
    """
    residuals = compute_residuals(y_val, H_val, spectrum)
    weighted_residuals = compute_weighted_residuals(residuals, weights_val)
    error = np.sqrt(np.mean(weighted_residuals**2))
    return error
