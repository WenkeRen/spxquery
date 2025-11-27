"""
Data loading and validation for SED reconstruction.

This module provides functions to load lightcurve CSV files from the
SPXQuery processing pipeline and prepare data for spectral reconstruction.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from astropy.stats import sigma_clip

from .config import SEDConfig

logger = logging.getLogger(__name__)


class BandData:
    """
    Container for single-band photometry data prepared for reconstruction.

    Attributes
    ----------
    band : str
        Band identifier (e.g., 'D1', 'D2', ..., 'D6').
    wavelength_range : Tuple[float, float]
        (min_wavelength, max_wavelength) in microns for this band.
    flux : np.ndarray
        Flux measurements in microJansky, shape (M,).
    flux_error : np.ndarray
        Flux uncertainties in microJansky, shape (M,).
    wavelength_center : np.ndarray
        Central wavelength of each narrow-band measurement in microns, shape (M,).
    bandwidth : np.ndarray
        Bandwidth of each narrow-band measurement in microns, shape (M,).
    n_measurements : int
        Number of measurements M.
    n_rejected : int
        Number of measurements rejected by quality filters.
    """

    def __init__(
        self,
        band: str,
        flux: np.ndarray,
        flux_error: np.ndarray,
        wavelength_center: np.ndarray,
        bandwidth: np.ndarray,
        n_rejected: int = 0,
    ):
        """
        Initialize BandData container.

        Parameters
        ----------
        band : str
            Band identifier.
        flux : np.ndarray
            Flux measurements (microJansky).
        flux_error : np.ndarray
            Flux uncertainties (microJansky).
        wavelength_center : np.ndarray
            Central wavelengths (microns).
        bandwidth : np.ndarray
            Bandwidths (microns).
        n_rejected : int
            Number of rejected measurements.
        """
        self.band = band
        self.flux = flux
        self.flux_error = flux_error
        self.wavelength_center = wavelength_center
        self.bandwidth = bandwidth
        self.n_measurements = len(flux)
        self.n_rejected = n_rejected

        # Compute wavelength range from data
        half_widths = bandwidth / 2.0
        self.wavelength_range = (
            float(np.min(wavelength_center - half_widths)),
            float(np.max(wavelength_center + half_widths)),
        )

    def __repr__(self) -> str:
        """String representation of BandData."""
        return (
            f"BandData(band={self.band}, "
            f"n_measurements={self.n_measurements}, "
            f"n_rejected={self.n_rejected}, "
            f"wavelength_range={self.wavelength_range[0]:.3f}-{self.wavelength_range[1]:.3f} um)"
        )


def load_lightcurve_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load lightcurve CSV file from SPXQuery processing pipeline.

    Parameters
    ----------
    csv_path : Path
        Path to lightcurve.csv file.

    Returns
    -------
    pd.DataFrame
        Lightcurve data with all columns and metadata attributes.

    Raises
    ------
    FileNotFoundError
        If CSV file does not exist.
    ValueError
        If required columns are missing.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Lightcurve CSV not found: {csv_path}")

    logger.info(f"Loading lightcurve data from {csv_path}")

    # Load CSV with metadata preservation
    df = pd.read_csv(csv_path, comment="#")

    # Validate required columns
    required_columns = [
        "flux",
        "flux_error",
        "wavelength",
        "bandwidth",
        "band",
        "flag",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing_columns}. Found columns: {list(df.columns)}"
        )

    # Calculate SNR if not present (SNR = flux / flux_error)
    if "snr" not in df.columns:
        # Avoid division by zero
        df["snr"] = np.where(df["flux_error"] > 0, df["flux"] / df["flux_error"], 0.0)
        logger.info("Calculated SNR from flux/flux_error")

    logger.info(f"Loaded {len(df)} measurements across {df['band'].nunique()} bands")

    return df


def apply_rolling_mad_sigma_clip_single_band(
    band_data: pd.DataFrame,
    sigma_clip_window: int,
    sigma_clip_sigma: float,
    band_name: str,
    max_iterations: int = 10,
) -> Tuple[np.ndarray, int]:
    """
    Apply iterative rolling MAD-based sigma clipping to a single band's data.

    This function implements robust outlier detection using rolling window
    Median Absolute Deviation (MAD) statistics with iterative refinement:
    1. Sort measurements by wavelength for meaningful rolling windows
    2. Calculate rolling median flux in local windows
    3. Calculate rolling MAD (robust dispersion estimate scaled to sigma)
    4. Identify outliers where |flux - median| > sigma_threshold * MAD
    5. Remove outliers and repeat until no more outliers or max_iterations reached

    Iterative approach catches outliers masked by more extreme values in early passes.

    Parameters
    ----------
    band_data : pd.DataFrame
        Data for a single band. Must contain 'wavelength' and 'flux' columns.
        Index should be the original DataFrame indices for mapping back.
    sigma_clip_window : int
        Window size for rolling MAD statistics.
    sigma_clip_sigma : float
        Sigma threshold for outlier detection.
    band_name : str
        Band identifier (e.g., 'D1') for logging messages.
    max_iterations : int
        Maximum number of iterative clipping passes. Default: 10.

    Returns
    -------
    outlier_indices : np.ndarray
        Array of original DataFrame indices that are outliers (all iterations combined).
    n_clipped : int
        Total number of outliers detected across all iterations.

    Notes
    -----
    - If band has fewer measurements than sigma_clip_window, applies simple MAD-based
      sigma clipping to all data as a single window using astropy.stats.sigma_clip.
    - MAD is scaled to equivalent standard deviation (scale='normal').
    - Rolling windows use min_periods = window_size // 4 for edge handling.
    - Iteration stops when no new outliers found or max_iterations reached.
    """
    # Check if enough data for rolling window
    if len(band_data) < sigma_clip_window:
        logger.warning(
            f"  Band {band_name}: only {len(band_data)} measurements "
            f"(< window size {sigma_clip_window}), applying simple sigma clipping to all data"
        )
        # Apply simple sigma clipping to all data as one window
        clipped_data = sigma_clip(
            np.asarray(band_data["flux"]),
            sigma=sigma_clip_sigma,
            maxiters=max_iterations,
            cenfunc="median",
            stdfunc="mad_std",
            masked=True
        )
        outlier_mask = clipped_data.mask
        outlier_indices = band_data.index[outlier_mask].values
        n_clipped = len(outlier_indices)

        if n_clipped == 0:
            logger.info(f"  Band {band_name}: no outliers detected")
        else:
            logger.info(f"  Band {band_name}: removed {n_clipped} outliers ({len(band_data) - n_clipped} remaining)")

        return outlier_indices, n_clipped

    # Initialize iteration tracking
    all_outlier_indices = []
    current_data = band_data.copy()
    iteration = 0

    # Iterative sigma clipping loop
    while iteration < max_iterations:
        iteration += 1

        # Store original DataFrame indices in a column BEFORE sorting
        # This maintains the mapping after reset_index(drop=True)
        current_data_copy = current_data.copy()
        current_data_copy["_original_idx"] = current_data_copy.index

        # CRITICAL: Sort by wavelength for rolling window statistics
        sorted_data = current_data_copy.sort_values("wavelength").reset_index(drop=True)

        # Calculate rolling median of flux
        # min_periods ensures estimates even at edges (use at least 1/4 of window)
        min_periods = max(1, sigma_clip_window // 4)
        sorted_data["flux_median"] = (
            sorted_data["flux"].rolling(window=sigma_clip_window, center=True, min_periods=min_periods).median()
        )

        # Calculate rolling MAD (Median Absolute Deviation)
        # scale='normal' makes MAD equivalent to standard deviation (multiply by 1.4826)
        def rolling_mad(flux_window):
            """Calculate MAD scaled to equivalent std for a rolling window."""
            return median_abs_deviation(flux_window, scale="normal", nan_policy="omit")

        sorted_data["flux_mad_std"] = (
            sorted_data["flux"]
            .rolling(window=sigma_clip_window, center=True, min_periods=min_periods)
            .apply(rolling_mad, raw=True)
        )

        # Calculate residuals from local median
        sorted_data["residual"] = sorted_data["flux"] - sorted_data["flux_median"]

        # Identify outliers: |residual| > sigma_threshold * MAD
        is_outlier = np.abs(sorted_data["residual"]) > (sigma_clip_sigma * sorted_data["flux_mad_std"])

        # Extract original indices for outliers using the stored column
        outlier_indices_this_iter = sorted_data.loc[is_outlier, "_original_idx"].values
        n_clipped_this_iter = len(outlier_indices_this_iter)

        # If no outliers found, stop iteration
        if n_clipped_this_iter == 0:
            if iteration == 1:
                logger.info(f"  Band {band_name}: no outliers detected")
            else:
                logger.info(f"  Band {band_name}: converged after {iteration} iterations")
            break

        # Add outliers to accumulated list
        all_outlier_indices.extend(outlier_indices_this_iter)

        logger.info(
            f"  Band {band_name}: iteration {iteration} removed {n_clipped_this_iter} outliers "
            f"({len(current_data) - n_clipped_this_iter} remaining)"
        )

        # Remove outliers from data for next iteration
        # Use .drop() to remove by index labels
        current_data = current_data.drop(outlier_indices_this_iter, errors='ignore')

        # Check if too few measurements remain for another iteration
        if len(current_data) < sigma_clip_window:
            logger.info(
                f"  Band {band_name}: stopping after {iteration} iterations "
                f"(< {sigma_clip_window} measurements remaining)"
            )
            break
    else:
        # Loop completed without break (reached max_iterations)
        logger.info(f"  Band {band_name}: stopped at max_iterations ({max_iterations})")

    # Return all accumulated outlier indices
    total_clipped = len(all_outlier_indices)
    return np.array(all_outlier_indices) if all_outlier_indices else np.array([]), total_clipped


def apply_quality_filters(df: pd.DataFrame, config: SEDConfig) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply quality control filters to lightcurve data.

    Filters are applied in the following order (logical progression):
    1. Bad pixel flags - remove known problematic data from pipeline
    2. NaN values in critical columns (flux, flux_error, wavelength, bandwidth)
    3. Non-positive flux_error values - invalid uncertainties (flux can be negative)
    4. SNR threshold - ensure sufficient signal quality (flux/flux_error >= sigma_threshold)
    5. Rolling MAD sigma clipping - remove statistical outliers per band on clean data

    Parameters
    ----------
    df : pd.DataFrame
        Lightcurve data from load_lightcurve_csv.
    config : SEDConfig
        Configuration with sigma_threshold and bad_flags.

    Returns
    -------
    df_filtered : pd.DataFrame
        Filtered data with only good measurements.
    rejection_stats : Dict[str, int]
        Statistics on rejected measurements by reason.
    """
    n_initial = len(df)
    rejection_stats = {
        "initial_count": n_initial,
        "bad_flags": 0,
        "nan_values": 0,
        "non_positive_error": 0,
        "low_snr": 0,
        "sigma_clipped": 0,
        "final_count": 0,
    }

    # Create a mask for good measurements
    good_mask = np.ones(len(df), dtype=bool)

    # Filter 1: Bad pixel flags (remove known problematic data first)
    # Check if any bad flag bit is set
    if config.bad_flags:
        bad_flag_mask = np.zeros(len(df), dtype=bool)
        for bad_flag in config.bad_flags:
            # Check if bit 'bad_flag' is set in the flag column
            bad_flag_mask |= (df["flag"].values & (1 << bad_flag)) != 0
        rejection_stats["bad_flags"] = bad_flag_mask.sum()
        good_mask &= ~bad_flag_mask

    # Filter 2: Remove NaN values in critical columns
    critical_columns = ["flux", "flux_error", "wavelength", "bandwidth"]
    nan_mask = df[critical_columns].isna().any(axis=1)
    rejection_stats["nan_values"] = nan_mask.sum()
    good_mask &= ~nan_mask

    # Filter 3: Remove non-positive flux_error (flux can be negative, but error must be positive)
    non_positive_error = df["flux_error"] <= 0
    rejection_stats["non_positive_error"] = non_positive_error.sum()
    good_mask &= ~non_positive_error

    # Filter 4: SNR threshold (ensure sufficient signal quality)
    low_snr = df["snr"] < config.sigma_threshold
    rejection_stats["low_snr"] = low_snr.sum()
    good_mask &= ~low_snr

    # Filter 5: Rolling window sigma clipping (remove statistical outliers last)
    # Applied per band using MAD on clean data
    if config.enable_sigma_clip:
        # Apply rolling MAD-based sigma clipping to each band separately
        clipped_full_mask = np.zeros(len(df), dtype=bool)
        current_good_data = df[good_mask].copy()

        if len(current_good_data) > 0:
            # Get unique bands
            bands = current_good_data["band"].unique()
            total_clipped = 0

            for band in sorted(bands):
                # Filter data for this band
                band_mask = current_good_data["band"] == band
                band_data = current_good_data[band_mask].copy()

                # Apply rolling MAD sigma clipping to this band
                outlier_indices, n_clipped = apply_rolling_mad_sigma_clip_single_band(
                    band_data=band_data,
                    sigma_clip_window=config.sigma_clip_window,
                    sigma_clip_sigma=config.sigma_clip_sigma,
                    band_name=band,
                    max_iterations=config.sigma_clip_max_iterations,
                )

                # Mark outliers in the full mask
                clipped_full_mask[outlier_indices] = True
                total_clipped += n_clipped

            rejection_stats["sigma_clipped"] = total_clipped
            good_mask &= ~clipped_full_mask

            logger.info(
                f"Rolling MAD sigma clipping ({config.sigma_clip_sigma}-sigma, "
                f"window={config.sigma_clip_window}): removed {total_clipped} outliers across all bands"
            )

    # Apply combined mask
    df_filtered = df[good_mask].copy()
    rejection_stats["final_count"] = len(df_filtered)

    n_rejected = n_initial - len(df_filtered)
    logger.info(
        f"Quality filtering: {n_initial} -> {len(df_filtered)} measurements "
        f"({n_rejected} rejected, {100 * n_rejected / n_initial:.1f}%)"
    )

    # Log detailed rejection breakdown (in application order)
    if n_rejected > 0:
        logger.info("  Rejection breakdown (in order applied):")
        if rejection_stats["bad_flags"] > 0:
            logger.info(f"    1. Bad pixel flags: {rejection_stats['bad_flags']}")
        if rejection_stats["nan_values"] > 0:
            logger.info(f"    2. NaN values: {rejection_stats['nan_values']}")
        if rejection_stats["non_positive_error"] > 0:
            logger.info(f"    3. Non-positive flux_error: {rejection_stats['non_positive_error']}")
        if rejection_stats["low_snr"] > 0:
            logger.info(f"    4. Low SNR (< {config.sigma_threshold}): {rejection_stats['low_snr']}")
        if rejection_stats["sigma_clipped"] > 0:
            logger.info(f"    5. Rolling MAD sigma clipping: {rejection_stats['sigma_clipped']}")

    if len(df_filtered) == 0:
        logger.warning("All measurements rejected by quality filters! Consider relaxing sigma_threshold or bad_flags.")

    return df_filtered, rejection_stats


def filter_by_band(df: pd.DataFrame, band: str) -> pd.DataFrame:
    """
    Extract measurements for a single detector band.

    Parameters
    ----------
    df : pd.DataFrame
        Lightcurve data (should be quality-filtered first).
    band : str
        Band identifier (e.g., 'D1', 'D2', ..., 'D6').

    Returns
    -------
    pd.DataFrame
        Subset of data for the specified band.

    Raises
    ------
    ValueError
        If band identifier is not found in data.
    """
    valid_bands = df["band"].unique()

    if band not in valid_bands:
        raise ValueError(f"Band '{band}' not found in data. Available bands: {sorted(valid_bands)}")

    band_df = df[df["band"] == band].copy()

    logger.info(f"Band {band}: {len(band_df)} measurements")

    return band_df


def prepare_band_data(df: pd.DataFrame, band: str, config: SEDConfig) -> Optional[BandData]:
    """
    Prepare single-band data for reconstruction.

    This function:
    1. Filters by band
    2. Extracts required arrays (flux, flux_error, wavelength, bandwidth)
    3. Validates data consistency
    4. Packages into BandData container

    Parameters
    ----------
    df : pd.DataFrame
        Quality-filtered lightcurve data.
    band : str
        Band identifier.
    config : SEDConfig
        Configuration (currently unused, reserved for future extensions).

    Returns
    -------
    BandData or None
        Prepared data for reconstruction, or None if insufficient data.
    """
    # Filter by band
    try:
        band_df = filter_by_band(df, band)
    except ValueError as e:
        logger.warning(f"Cannot prepare {band}: {e}")
        return None

    # Check for minimum measurements
    if len(band_df) < 10:
        logger.warning(
            f"Band {band} has only {len(band_df)} measurements (< 10). Skipping reconstruction for this band."
        )
        return None

    # Extract arrays
    flux = band_df["flux"].values
    flux_error = band_df["flux_error"].values
    wavelength_center = band_df["wavelength"].values
    bandwidth = band_df["bandwidth"].values

    # Validate array shapes
    assert flux.shape == flux_error.shape == wavelength_center.shape == bandwidth.shape

    # Create BandData container
    band_data = BandData(
        band=band,
        flux=flux,
        flux_error=flux_error,
        wavelength_center=wavelength_center,
        bandwidth=bandwidth,
        n_rejected=0,  # Already filtered
    )

    logger.info(f"Prepared {band_data}")

    return band_data


def load_all_bands(csv_path: Path, config: SEDConfig) -> Tuple[Dict[str, BandData], Dict[str, Any]]:
    """
    Load and prepare data for all detector bands.

    This is the main entry point for data loading in the reconstruction pipeline.

    Parameters
    ----------
    csv_path : Path
        Path to lightcurve.csv file.
    config : SEDConfig
        Configuration with quality filters.

    Returns
    -------
    band_data_dict : Dict[str, BandData]
        Dictionary mapping band names to BandData objects.
    metadata : Dict[str, Any]
        Metadata including source info and rejection statistics.

    Raises
    ------
    ValueError
        If no bands have sufficient data for reconstruction.
    """
    # Load CSV
    df = load_lightcurve_csv(csv_path)

    # Extract metadata (if available in DataFrame attributes)
    metadata = {
        "source_name": getattr(df, "attrs", {}).get("source_name", "unknown"),
        "source_ra": getattr(df, "attrs", {}).get("source_ra", None),
        "source_dec": getattr(df, "attrs", {}).get("source_dec", None),
        "csv_path": str(csv_path),
    }

    # Apply quality filters
    df_filtered, rejection_stats = apply_quality_filters(df, config)
    metadata["rejection_stats"] = rejection_stats

    if len(df_filtered) == 0:
        raise ValueError("No measurements passed quality filters. Consider relaxing sigma_threshold or bad_flags.")

    # Prepare data for each band
    all_bands = ["D1", "D2", "D3", "D4", "D5", "D6"]
    band_data_dict = {}

    for band in all_bands:
        band_data = prepare_band_data(df_filtered, band, config)
        if band_data is not None:
            band_data_dict[band] = band_data

    if not band_data_dict:
        raise ValueError("No bands have sufficient data for reconstruction (minimum 10 measurements per band).")

    logger.info(f"Successfully prepared data for {len(band_data_dict)} bands: {list(band_data_dict.keys())}")

    return band_data_dict, metadata
