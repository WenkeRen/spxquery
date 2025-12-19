"""
Matrix construction for SED reconstruction with unit consistency.

This module implements mathematical operators required for PyTorch-based
Deep Image Prior spectral reconstruction:
- H: Measurement matrix (M x N) relating observations to spectrum with frequency step normalization
- w: Weight vector (M,) for chi-squared data fidelity
- Frequency grid computation and step normalization

The measurement matrix H incorporates proper frequency step normalization
to handle non-uniform wavelength grids while maintaining energy conservation.
Input and output fluxes remain in microjansky (μJy).
"""

import logging
from typing import Callable, Dict

import numpy as np
import scipy.sparse as sp
import torch
from astropy.constants import c

from .config import SEDConfig
from .data_loader import BandData
from .data_structures import GlobalSpectralData

logger = logging.getLogger(__name__)


def boxcar_filter_response(wavelength: float, center: float, bandwidth: float) -> float:
    """
    Compute boxcar (rectangular) filter response at a given wavelength.

    The filter has unit response within [center - bandwidth/2, center + bandwidth/2]
    and zero response outside this range.

    Parameters
    ----------
    wavelength : float
        Wavelength to evaluate filter response at (microns).
    center : float
        Central wavelength of the narrow-band filter (microns).
    bandwidth : float
        Full width of the filter (microns).

    Returns
    -------
    float
        Filter response: 1.0 if within band, 0.0 otherwise.
    """
    half_width = bandwidth / 2.0
    lower_edge = center - half_width
    upper_edge = center + half_width

    if lower_edge <= wavelength <= upper_edge:
        return 1.0
    else:
        return 0.0


def get_filter_response_function(profile: str) -> Callable[[float, float, float], float]:
    """
    Get filter response function for specified profile type.

    Parameters
    ----------
    profile : str
        Filter profile name. Currently only 'boxcar' is supported.

    Returns
    -------
    Callable[[float, float, float], float]
        Filter response function with signature (wavelength, center, bandwidth) -> response.

    Raises
    ------
    ValueError
        If profile is not recognized.
    """
    if profile == "boxcar":
        return boxcar_filter_response
    else:
        raise ValueError(f"Unknown filter profile: '{profile}'. Only 'boxcar' is supported.")


def build_measurement_matrix(band_data: BandData, wavelength_grid: np.ndarray, config: SEDConfig) -> sp.csr_matrix:
    """
    Build measurement matrix H relating spectrum to observations with frequency step normalization.

    The matrix H is M x N where:
    - M = number of measurements (band_data.n_measurements)
    - N = number of wavelength bins (len(wavelength_grid))

    The forward model is: y = H @ x
    where y is observed flux in μJy (M,) and x is true spectrum in μJy (N,).

    H[i, j] represents the weight of wavelength bin j in measurement i, accounting for:
    1. Filter response at wavelength_grid[j] for the filter centered at band_data.wavelength_center[i]
    2. Frequency step normalization: weight = response × (Δν_j / ΣΔν_window)

    This ensures proper energy conservation for non-uniform wavelength grids.

    Parameters
    ----------
    band_data : BandData
        Measurement data (flux, wavelengths, bandwidths).
    wavelength_grid : np.ndarray
        Wavelength grid for reconstructed spectrum (microns), shape (N,).
    config : SEDConfig
        Configuration with filter_profile setting.

    Returns
    -------
    sp.csr_matrix
        Measurement matrix in CSR format, shape (M, N).

    Notes
    -----
    The matrix incorporates frequency step normalization to handle non-uniform
    wavelength grids while maintaining energy conservation. Each row sums to 1,
    representing the proper weighting of spectrum contributions to each measurement.

    The matrix is built in COO format (efficient for construction) then
    converted to CSR format (efficient for matrix-vector multiplication).
    """
    M = band_data.n_measurements
    N = len(wavelength_grid)

    logger.info(f"Building measurement matrix H: {M} measurements x {N} wavelength bins")

    # Get filter response function
    filter_func = get_filter_response_function(config.filter_profile)

    # Build frequency grid and compute frequency steps for normalization
    frequency_grid = build_frequency_grid(wavelength_grid)
    delta_nu = compute_frequency_steps(frequency_grid)

    # Lists for COO sparse matrix construction
    rows = []
    cols = []
    data = []

    # Build H row by row (one row per measurement)
    for i in range(M):
        center = band_data.wavelength_center[i]
        bandwidth = band_data.bandwidth[i]

        # Determine which wavelength bins are covered by this filter
        # For efficiency, only check wavelengths near the filter center
        half_width = bandwidth / 2.0
        lower_bound = center - half_width
        upper_bound = center + half_width

        # Find indices of wavelength bins within this range
        in_range = (wavelength_grid >= lower_bound) & (wavelength_grid <= upper_bound)
        j_indices = np.where(in_range)[0]

        # Compute frequency normalization for this measurement window
        if len(j_indices) > 0:
            # Sum of frequency steps within the window
            window_freq_sum = np.sum(delta_nu[j_indices])

            # Compute filter response for each wavelength bin in range
            for j in j_indices:
                wavelength = wavelength_grid[j]
                response = filter_func(wavelength, center, bandwidth)

                if response > 0:  # Only store non-zero entries
                    # Apply frequency step normalization: weight = response × (Δν_j / ΣΔν_window)
                    weight = response * (delta_nu[j] / window_freq_sum)
                    rows.append(i)
                    cols.append(j)
                    data.append(weight)

    # Convert lists to arrays
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.array(data, dtype=np.float64)

    # Create sparse matrix in COO format
    H_coo = sp.coo_matrix((data, (rows, cols)), shape=(M, N), dtype=np.float64)

    # Convert to CSR for efficient matrix-vector operations
    H_csr = H_coo.tocsr()

    # Log sparsity statistics
    n_nonzero = H_csr.nnz
    sparsity = 1.0 - (n_nonzero / (M * N))
    logger.info(
        f"H matrix: {n_nonzero:,} non-zero entries ({sparsity:.2%} sparse, {n_nonzero / M:.1f} entries/row avg)"
    )

    return H_csr


def build_frequency_grid(wavelength_grid: np.ndarray) -> np.ndarray:
    """
    Convert wavelength grid to frequency grid using astropy constants.

    Parameters
    ----------
    wavelength_grid : np.ndarray
        Wavelength grid in microns, shape (N,).

    Returns
    -------
    np.ndarray
        Frequency grid in Hz, shape (N,).

    Notes
    -----
    Uses the relationship ν = c/λ where c is the speed of light.
    """
    # Convert wavelength from microns to meters
    wavelength_m = wavelength_grid * 1e-6
    # Calculate frequency in Hz
    frequency_grid = c.value / wavelength_m
    return frequency_grid


def compute_frequency_steps(frequency_grid: np.ndarray) -> np.ndarray:
    """
    Compute frequency step sizes using centered differences.

    Parameters
    ----------
    frequency_grid : np.ndarray
        Frequency grid in Hz, shape (N,).

    Returns
    -------
    np.ndarray
        Frequency step sizes Δν, shape (N,).

    Notes
    -----
    Uses centered differences for interior points and forward/backward
    differences for boundary points to ensure accurate representation
    of non-uniform frequency grids.
    """
    N = len(frequency_grid)
    delta_nu = np.zeros(N, dtype=np.float64)

    if N < 2:
        return delta_nu

    # Interior points: centered differences
    delta_nu[1:-1] = (frequency_grid[2:] - frequency_grid[:-2]) / 2.0

    # Boundary points: forward/backward differences
    delta_nu[0] = frequency_grid[1] - frequency_grid[0]
    delta_nu[-1] = frequency_grid[-1] - frequency_grid[-2]

    return delta_nu


def build_weight_vector(band_data: BandData, config: SEDConfig) -> np.ndarray:
    """
    Build weight vector for chi-squared data fidelity term.

    Weights are inversely proportional to measurement uncertainties:
        w[i] = 1 / (flux_error[i] + epsilon)

    This implements proper chi-squared weighting in the objective function:
        data_fidelity = ||w * (y - H @ x)||_2^2

    Parameters
    ----------
    band_data : BandData
        Measurement data with flux_error array.
    config : SEDConfig
        Configuration with epsilon_weight parameter.

    Returns
    -------
    np.ndarray
        Weight vector, shape (M,).

    Notes
    -----
    The epsilon parameter prevents division by zero for measurements
    with very small reported uncertainties.
    """
    M = band_data.n_measurements

    # Compute weights with epsilon to avoid division by zero
    weights = 1.0 / (band_data.flux_error + config.epsilon_weight)

    # Check for invalid weights
    n_invalid = np.sum(~np.isfinite(weights))
    if n_invalid > 0:
        logger.warning(
            f"Found {n_invalid} invalid weights (inf/nan). Setting to zero (these measurements will be ignored)."
        )
        weights[~np.isfinite(weights)] = 0.0

    logger.info(
        f"Weight vector: mean={np.mean(weights):.2e}, "
        f"median={np.median(weights):.2e}, "
        f"range=[{np.min(weights):.2e}, {np.max(weights):.2e}]"
    )

    return weights


def build_global_observation_data(all_band_data: Dict[str, BandData], config: SEDConfig) -> GlobalSpectralData:
    """
    Build global dataset for reconstruction, preserving the integral constraint.

    Constructs the global measurement matrix H by stacking per-band matrices.

    Parameters
    ----------
    all_band_data : Dict[str, BandData]
        Dictionary mapping band names to BandData objects.
    config : SEDConfig
        Configuration with wavelength range and resolution.

    Returns
    -------
    GlobalSpectralData
        Dataset containing sparse H, observations y, and weights w.
    """
    logger.info("Building global spectral observation data...")

    # 1. Generate Global Wavelength Grid
    lambda_min, lambda_max = config.wavelength_range
    N = config.global_resolution
    global_wavelength_grid = np.linspace(lambda_min, lambda_max, N, dtype=np.float64)

    all_H = []
    all_y = []
    all_w = []

    # Sort bands for deterministic order
    sorted_bands = sorted(all_band_data.keys())

    for band in sorted_bands:
        band_data = all_band_data[band]
        logger.debug(f"Processing {band}...")

        # Build H for this band relative to global grid
        H_band = build_measurement_matrix(band_data, global_wavelength_grid, config)
        all_H.append(H_band)

        # Get observations
        all_y.append(band_data.flux)

        # Build weights
        w_band = build_weight_vector(band_data, config)
        all_w.append(w_band)

    # Stack everything
    if not all_H:
        raise ValueError("No data found to build dataset")

    H_global = sp.vstack(all_H)  # CSR matrix
    y_global = np.concatenate(all_y)
    w_global = np.concatenate(all_w)

    # Convert H to Torch Sparse COO format
    H_coo = H_global.tocoo()

    indices = np.vstack((H_coo.row, H_coo.col))
    t_indices = torch.from_numpy(indices).long()
    t_values = torch.from_numpy(H_coo.data).float()

    t_observations = torch.from_numpy(y_global).float()
    t_weights = torch.from_numpy(w_global).float()
    t_grid = torch.from_numpy(global_wavelength_grid).float()

    dataset = GlobalSpectralData(
        H_indices=t_indices,
        H_values=t_values,
        H_shape=H_global.shape,
        observations=t_observations,
        weights=t_weights,
        global_wavelength_grid=t_grid,
    )

    logger.info(
        f"Built global dataset: {H_global.shape[0]} observations, {H_global.shape[1]} spectral bins, {H_global.nnz} non-zeros"
    )

    return dataset
