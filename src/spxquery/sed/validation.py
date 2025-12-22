"""
Validation and quality assessment for SED reconstruction.

This module provides functions to evaluate reconstruction quality through
residual analysis, chi-squared statistics, and goodness-of-fit metrics.
"""

import logging
from dataclasses import dataclass

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
    chi_squared_per_obs : float
        Chi-squared per observation (chi^2 / M). Average weighted residual squared per observation.
        Ideal: ~= 1.0, > 2.0: Poor fit or underestimated errors, < 0.5: Overfitting or overestimated errors.
    n_obs : int
        Number of observations (M).
    n_sample : int
        Number of spectral samples/parameters (N).
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
    negative_flux_fraction : float
        Fraction of spectral bins with negative flux.
        Ideal: 0.0 (physical spectra must be non-negative), Warning: > 5%.
    smoothness_tv : float
        Normalized Total Variation of the spectrum. Measures spectral roughness/oscillation.
        Lower values indicate smoother spectra (preferred for continuum).
    residual_oscillation : float
        von Neumann Ratio test p-value on weighted residuals for oscillation detection.
        p < 0.05 indicates significant oscillation exists.
    residual_rms : float
        Root Mean Square of weighted residuals. Expected to be close to 1.0.
    """

    chi_squared: float
    chi_squared_per_obs: float
    n_obs: int
    n_sample: int
    residuals: np.ndarray
    weighted_residuals: np.ndarray
    residual_mean: float
    residual_std: float
    weighted_residual_mean: float
    weighted_residual_std: float
    max_residual: float
    normality_pvalue: float
    negative_flux_fraction: float
    smoothness_tv: float
    residual_oscillation: float
    residual_rms: float


class SpectralEvaluator:
    """
    Comprehensive evaluator for SED reconstruction quality assessment.

    This class provides statistically valid and physically meaningful metrics
    for Deep Image Prior reconstruction, including chi-squared statistics,
    physicality checks, smoothness metrics, and oscillation detection.
    """

    def assess_reconstruction_quality(
        self,
        y: np.ndarray,
        H: sp.csr_matrix,
        spectrum: np.ndarray,
        weights: np.ndarray,
    ) -> ValidationMetrics:
        """
        Compute comprehensive quality metrics for reconstructed spectrum.

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
        n_obs, n_sample = H.shape

        # Compute residuals
        residuals = self._compute_residuals(y, H, spectrum)
        weighted_residuals = self._compute_weighted_residuals(residuals, weights)

        # Chi-squared statistics
        chi_squared, chi_squared_per_obs = self._compute_chi_squared_statistics(weighted_residuals, n_obs)

        # Physicality metrics
        negative_flux_fraction = self._compute_negative_flux_fraction(spectrum)

        # Smoothness metrics
        smoothness_tv = self._compute_smoothness_tv(spectrum)

        # Residual oscillation detection
        residual_oscillation = self._compute_residual_oscillation(weighted_residuals)

        # RMS of weighted residuals
        residual_rms = self._compute_residual_rms(weighted_residuals)

        # Basic residual statistics
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        weighted_residual_mean = np.mean(weighted_residuals)
        weighted_residual_std = np.std(weighted_residuals)
        max_residual = np.max(np.abs(residuals))

        # Normality test on weighted residuals
        normality_pvalue = self._compute_normality_test(weighted_residuals)

        # Create validation metrics
        metrics = ValidationMetrics(
            chi_squared=chi_squared,
            chi_squared_per_obs=chi_squared_per_obs,
            n_obs=n_obs,
            n_sample=n_sample,
            residuals=residuals,
            weighted_residuals=weighted_residuals,
            residual_mean=residual_mean,
            residual_std=residual_std,
            weighted_residual_mean=weighted_residual_mean,
            weighted_residual_std=weighted_residual_std,
            max_residual=max_residual,
            normality_pvalue=normality_pvalue,
            negative_flux_fraction=negative_flux_fraction,
            smoothness_tv=smoothness_tv,
            residual_oscillation=residual_oscillation,
            residual_rms=residual_rms,
        )

        # Log interpretation
        self._log_quality_interpretation(metrics)

        return metrics

    def _compute_residuals(self, y: np.ndarray, H: sp.csr_matrix, spectrum: np.ndarray) -> np.ndarray:
        """Compute raw residuals between observations and model."""
        y_model = H @ spectrum
        return y - y_model

    def _compute_weighted_residuals(self, residuals: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute weighted residuals."""
        return weights * residuals

    def _compute_chi_squared_statistics(self, weighted_residuals: np.ndarray, n_obs: int) -> tuple[float, float]:
        """
        Compute chi-squared and chi-squared per observation.

        chi_squared_per_obs = chi^2 / M (average weighted residual squared per observation)
        """
        chi_squared = np.sum(weighted_residuals**2)
        chi_squared_per_obs = chi_squared / n_obs
        return chi_squared, chi_squared_per_obs

    def _compute_negative_flux_fraction(self, spectrum: np.ndarray) -> float:
        """
        Compute fraction of spectral bins with negative flux.

        Physical spectra must be non-negative. Values > 0.05 (5%) indicate problematic reconstruction.
        """
        negative_count = np.sum(spectrum < 0)
        return negative_count / len(spectrum)

    def _compute_smoothness_tv(self, spectrum: np.ndarray) -> float:
        """
        Compute normalized Total Variation of the spectrum.

        TV measures spectral roughness/oscillation. Lower values indicate smoother spectra.
        TV = sum(|spectrum[i+1] - spectrum[i]|) for i = 0 to N-2
        """
        if len(spectrum) < 2:
            return 0.0

        tv = np.sum(np.abs(np.diff(spectrum)))
        # Normalize by the spectrum range to make it scale-independent
        spectrum_range = np.ptp(spectrum)  # peak-to-peak range
        if spectrum_range > 0:
            return tv / spectrum_range
        else:
            return 0.0

    def _compute_residual_oscillation(self, weighted_residuals: np.ndarray) -> float:
        """
        Compute von Neumann Ratio test p-value on weighted residuals.

        Tests for systematic oscillation in residuals using Mean Square Successive Difference.

        Statistic: M = sum((r_{i+1} - r_i)^2) / sum((r_i - mean(r))^2)
        Expected value: mu_M ~= 2, Standard deviation: sigma_M ~= sqrt(4/N)

        Returns p-value for oscillation detection. p < 0.05 indicates significant oscillation.
        """
        if len(weighted_residuals) < 3:
            return np.nan

        residuals = weighted_residuals
        n = len(residuals)
        r_mean = np.mean(residuals)

        # von Neumann Ratio statistic
        numerator = np.sum(np.diff(residuals) ** 2)  # sum((r_{i+1} - r_i)^2)
        denominator = np.sum((residuals - r_mean) ** 2)  # sum((r_i - mean(r))^2)

        if denominator == 0:
            return np.nan

        m_statistic = numerator / denominator

        # Expected value and standard deviation under null hypothesis
        expected_m = 2.0
        std_m = np.sqrt(4.0 / n)

        if std_m == 0:
            return np.nan

        # Z-score (negative if smooth oscillation exists)
        z_score = (m_statistic - expected_m) / std_m

        # Two-tailed p-value using standard normal CDF
        p_value = stats.norm.cdf(z_score)

        return p_value

    def _compute_residual_rms(self, weighted_residuals: np.ndarray) -> float:
        """
        Compute Root Mean Square of weighted residuals.

        Expected to be close to 1.0 if weights are properly calibrated.
        """
        return np.sqrt(np.mean(weighted_residuals**2))

    def _compute_normality_test(self, weighted_residuals: np.ndarray) -> float:
        """
        Compute normality test p-value on weighted residuals.

        Uses Shapiro-Wilk for N < 5000, and D'Agostino's K^2 test for N >= 5000.
        High p-value (>0.05) suggests residuals are consistent with Gaussian distribution.
        """
        n = len(weighted_residuals)
        if n < 3:
            return np.nan

        try:
            if n < 5000:
                # Shapiro-Wilk is powerful for small to medium samples
                _, normality_pvalue = stats.shapiro(weighted_residuals)
            else:
                # For large samples (N > 5000), Shapiro-Wilk p-value is inaccurate.
                # Use D'Agostino's K^2 test which is robust for large N.
                _, normality_pvalue = stats.normaltest(weighted_residuals)

            return float(normality_pvalue)
        except Exception as e:
            logger.warning(f"Normality test failed: {e}")
            return np.nan

    def _log_quality_interpretation(self, metrics: ValidationMetrics) -> None:
        """Log interpretation of quality metrics."""
        logger.info("SED Reconstruction Quality Assessment:")
        logger.info(f"  Observations: {metrics.n_obs}, Spectral samples: {metrics.n_sample}")
        logger.info(f"  chi^2 = {metrics.chi_squared:.2f}, chi^2/M = {metrics.chi_squared_per_obs:.3f}")
        logger.info(
            f"  Weighted residuals: mean = {metrics.weighted_residual_mean:.3f}, "
            f"std = {metrics.weighted_residual_std:.3f}"
        )
        logger.info(f"  RMS weighted residuals: {metrics.residual_rms:.3f}")

        # Physicality assessment
        logger.info(f"  Negative flux fraction: {metrics.negative_flux_fraction:.1%}")
        if metrics.negative_flux_fraction == 0.0:
            logger.info("  -> Physical spectrum: no negative flux values")
        elif metrics.negative_flux_fraction > 0.05:
            logger.warning(
                f"  -> Unphysical spectrum: {metrics.negative_flux_fraction:.1%} negative flux (>5% threshold)"
            )
        else:
            logger.info(f"  -> Minor unphysical values: {metrics.negative_flux_fraction:.1%} negative flux")

        # Oscillation detection
        if not np.isnan(metrics.residual_oscillation):
            logger.info(f"  Residual oscillation test: p = {metrics.residual_oscillation:.3f}")
            if metrics.residual_oscillation < 0.05:
                logger.warning("  -> Significant residual oscillation detected (p < 0.05)")
            else:
                logger.info("  -> No significant residual oscillation")
        else:
            logger.info("  Residual oscillation test: N/A")

        # Normality test
        if not np.isnan(metrics.normality_pvalue):
            logger.info(f"  Normality test: p = {metrics.normality_pvalue:.3f}")
            if metrics.normality_pvalue > 0.05:
                logger.info("  -> Residuals consistent with Gaussian distribution")
            else:
                logger.warning("  -> Residuals deviate from Gaussian distribution")
        else:
            logger.info("  Normality test: N/A")

        # Chi-squared interpretation
        chi2_per_obs = metrics.chi_squared_per_obs
        if 0.5 <= chi2_per_obs <= 2.0:
            logger.info("  -> Good fit (chi^2/M near 1.0)")
        elif chi2_per_obs > 2.0:
            logger.warning("  -> Poor fit or underestimated errors (chi^2/M >> 1.0)")
        else:
            logger.warning("  -> Possible overfitting or overestimated errors (chi^2/M << 1.0)")
