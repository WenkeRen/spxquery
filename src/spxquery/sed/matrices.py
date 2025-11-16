"""
Sparse matrix construction for SED reconstruction.

This module implements the mathematical operators required for regularized
least-squares spectral reconstruction:
- H: Measurement matrix (M x N) relating observations to spectrum
- Psi_approx: Wavelet approximation coefficient extraction matrix
- Psi_detail: Wavelet detail coefficient extraction matrix
- w: Weight vector (M,) for chi-squared data fidelity
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pywt
import scipy.sparse as sp

from .config import DETECTOR_WAVELENGTH_RANGES, SEDConfig
from .data_loader import BandData

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
    Build measurement matrix H relating spectrum to observations.

    The matrix H is M x N where:
    - M = number of measurements (band_data.n_measurements)
    - N = number of wavelength bins (len(wavelength_grid))

    The forward model is: y = H @ x
    where y is observed flux (M,) and x is true spectrum (N,).

    H[i, j] represents the contribution of wavelength bin j to measurement i.
    This is determined by the narrow-band filter response at wavelength_grid[j]
    for the filter centered at band_data.wavelength_center[i].

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
    The matrix is built in COO format (efficient for construction) then
    converted to CSR format (efficient for matrix-vector multiplication).
    """
    M = band_data.n_measurements
    N = len(wavelength_grid)

    logger.info(f"Building measurement matrix H: {M} measurements x {N} wavelength bins")

    # Get filter response function
    filter_func = get_filter_response_function(config.filter_profile)

    # Lists for COO sparse matrix construction
    rows = []
    cols = []
    data = []

    # Wavelength grid spacing (assume uniform)
    delta_lambda = wavelength_grid[1] - wavelength_grid[0]

    # Build H row by row (one row per measurement)
    for i in range(M):
        center = band_data.wavelength_center[i]
        bandwidth = band_data.bandwidth[i]

        # Determine which wavelength bins are covered by this filter
        # For efficiency, only check wavelengths near the filter center
        half_width = bandwidth / 2.0
        lower_bound = center - half_width - delta_lambda  # Small margin
        upper_bound = center + half_width + delta_lambda

        # Find indices of wavelength bins within this range
        in_range = (wavelength_grid >= lower_bound) & (wavelength_grid <= upper_bound)
        j_indices = np.where(in_range)[0]

        # Compute filter response for each wavelength bin in range
        for j in j_indices:
            wavelength = wavelength_grid[j]
            response = filter_func(wavelength, center, bandwidth)

            if response > 0:  # Only store non-zero entries
                rows.append(i)
                cols.append(j)
                data.append(response)

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


def build_smoothness_operator(N: int) -> sp.csr_matrix:
    """
    Build second-order smoothness operator D2.

    The operator approximates the second derivative via finite differences:
        (D2 @ x)[i] = x[i] - 2*x[i+1] + x[i+2]

    This is used in the regularization term: lambda2 * ||D2 @ x||_2^2
    which penalizes curvature in the reconstructed spectrum.

    Parameters
    ----------
    N : int
        Number of wavelength bins in spectrum.

    Returns
    -------
    sp.csr_matrix
        Smoothness operator in CSR format, shape (N-2, N).

    Notes
    -----
    The matrix has a tridiagonal structure with pattern:
        [1, -2, 1, 0, 0, ...]
        [0, 1, -2, 1, 0, ...]
        ...
    """
    if N < 3:
        raise ValueError(f"N must be >= 3 for D2 operator, got N={N}")

    logger.info(f"Building smoothness operator D2: ({N - 2}, {N})")

    # Define diagonals: [1, -2, 1]
    # Offsets: 0, 1, 2 (relative to main diagonal)
    diagonals = [
        np.ones(N - 2, dtype=np.float64),  # First column: 1
        -2 * np.ones(N - 1, dtype=np.float64),  # Second column: -2
        np.ones(N, dtype=np.float64),  # Third column: 1
    ]
    offsets = [0, 1, 2]

    # Create sparse matrix using scipy.sparse.diags
    D2 = sp.diags(diagonals, offsets, shape=(N - 2, N), format="csr", dtype=np.float64)

    logger.info(f"D2 matrix: {D2.nnz:,} non-zero entries")

    return D2


def build_wavelet_matrices(
    N: int, wavelet: str = "sym6", level: Optional[int] = None, mode: str = "symmetric"
) -> Tuple[sp.csr_matrix, sp.csr_matrix, Dict[str, int]]:
    """
    Build wavelet transform matrices for multi-scale regularization.

    This function constructs two sparse matrices that extract wavelet coefficients:
    - Psi_approx: Extracts approximation coefficients (low-frequency continuum)
    - Psi_detail: Extracts all detail coefficients (high-frequency features/noise)

    The wavelet decomposition separates the spectrum into frequency scales:
        Level 0 (finest):   High-frequency noise
        Level 1-k:          Emission lines at various scales
        Level k (coarsest): Low-frequency continuum (approximation)

    Parameters
    ----------
    N : int
        Number of wavelength bins in spectrum.
    wavelet : str, optional
        Wavelet family name (default: 'sym6').
        Common choices: 'sym4', 'sym6', 'sym8', 'db4', 'db6', 'db8'.
    level : int, optional
        Number of decomposition levels. If None, auto-detect using
        pywt.dwt_max_level(N, wavelet).
    mode : str, optional
        Boundary mode for wavelet transform (default: 'symmetric').
        Options: 'symmetric', 'periodic', 'periodization', 'zero', 'constant',
        'reflect', 'antisymmetric', 'antireflect', 'smooth'.
        Symmetric mode is recommended for spectra to avoid edge artifacts.

    Returns
    -------
    Psi_approx : sp.csr_matrix
        Approximation coefficient extraction matrix, shape (n_approx, N).
    Psi_detail : sp.csr_matrix
        Detail coefficient extraction matrix, shape (n_detail, N).
    level_info : dict
        Dictionary with keys:
        - 'level': decomposition level used
        - 'n_approx': number of approximation coefficients
        - 'n_detail': number of detail coefficients
        - 'detail_lengths': list of detail coefficient counts per level

    Raises
    ------
    ValueError
        If N is too small for wavelet decomposition or wavelet is invalid.

    Notes
    -----
    The matrices are built by explicitly constructing the wavelet transform
    as a sparse linear operator. This allows CVXPY to use them directly in
    convex optimization.

    The detail coefficients are concatenated from all levels:
        c_detail = [cD_1, cD_2, ..., cD_level]
    where cD_1 are the finest details (highest frequency).
    """
    if N < 4:
        raise ValueError(f"N must be >= 4 for wavelet decomposition, got N={N}")

    # Auto-detect decomposition level if not specified
    if level is None:
        level = pywt.dwt_max_level(N, wavelet)
        logger.info(f"Auto-detected wavelet level: {level} (for N={N}, wavelet='{wavelet}')")
    else:
        max_level = pywt.dwt_max_level(N, wavelet)
        if level > max_level:
            logger.warning(f"Requested level {level} exceeds maximum {max_level}. Using level={max_level} instead.")
            level = max_level

    # Get wavelet object
    try:
        wav = pywt.Wavelet(wavelet)
    except ValueError as e:
        raise ValueError(f"Invalid wavelet '{wavelet}': {e}")

    logger.info(f"Building wavelet matrices: N={N}, wavelet='{wavelet}', level={level}, mode='{mode}'")

    # Perform test decomposition to get coefficient structure
    test_signal = np.zeros(N, dtype=np.float64)
    test_signal[N // 2] = 1.0  # Delta function at center
    coeffs = pywt.wavedec(test_signal, wavelet, level=level, mode=mode)

    # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    # where n = level
    cA_test = coeffs[0]  # Approximation
    cD_tests = coeffs[1:]  # Details (coarse to fine)

    n_approx = len(cA_test)
    detail_lengths = [len(cD) for cD in cD_tests]
    n_detail = sum(detail_lengths)

    logger.info(
        f"Wavelet coefficient structure: n_approx={n_approx}, n_detail={n_detail} (detail levels: {detail_lengths})"
    )

    # Build Psi_approx: Extract approximation coefficients
    # This is done by computing the full DWT and selecting cA
    rows_approx = []
    cols_approx = []
    data_approx = []

    for i in range(n_approx):
        # Create unit vector in approximation coefficient space
        unit_coeffs = [np.zeros_like(c) for c in coeffs]
        unit_coeffs[0][i] = 1.0

        # Reconstruct signal from this unit coefficient
        signal = pywt.waverec(unit_coeffs, wavelet, mode=mode)

        # Handle potential length mismatch due to padding
        if len(signal) > N:
            signal = signal[:N]
        elif len(signal) < N:
            signal = np.pad(signal, (0, N - len(signal)), mode="constant")

        # Store non-zero entries
        nonzero_indices = np.where(np.abs(signal) > 1e-12)[0]
        for j in nonzero_indices:
            rows_approx.append(i)
            cols_approx.append(j)
            data_approx.append(signal[j])

    Psi_approx = sp.coo_matrix((data_approx, (rows_approx, cols_approx)), shape=(n_approx, N), dtype=np.float64).tocsr()

    # Build Psi_detail: Extract all detail coefficients (concatenated)
    rows_detail = []
    cols_detail = []
    data_detail = []

    row_offset = 0
    for level_idx, cD_test in enumerate(cD_tests):
        n_coeff = len(cD_test)

        for i in range(n_coeff):
            # Create unit vector in detail coefficient space
            unit_coeffs = [np.zeros_like(c) for c in coeffs]
            unit_coeffs[level_idx + 1][i] = 1.0  # +1 because coeffs[0] is cA

            # Reconstruct signal from this unit coefficient
            signal = pywt.waverec(unit_coeffs, wavelet, mode=mode)

            # Handle length mismatch
            if len(signal) > N:
                signal = signal[:N]
            elif len(signal) < N:
                signal = np.pad(signal, (0, N - len(signal)), mode="constant")

            # Store non-zero entries
            nonzero_indices = np.where(np.abs(signal) > 1e-12)[0]
            for j in nonzero_indices:
                rows_detail.append(row_offset + i)
                cols_detail.append(j)
                data_detail.append(signal[j])

        row_offset += n_coeff

    Psi_detail = sp.coo_matrix((data_detail, (rows_detail, cols_detail)), shape=(n_detail, N), dtype=np.float64).tocsr()

    # Log sparsity statistics
    sparsity_approx = 1.0 - (Psi_approx.nnz / (n_approx * N))
    sparsity_detail = 1.0 - (Psi_detail.nnz / (n_detail * N))
    logger.info(
        f"Psi_approx: {Psi_approx.nnz:,} non-zero entries ({sparsity_approx:.2%} sparse), shape ({n_approx}, {N})"
    )
    logger.info(
        f"Psi_detail: {Psi_detail.nnz:,} non-zero entries ({sparsity_detail:.2%} sparse), shape ({n_detail}, {N})"
    )

    level_info = {
        "level": level,
        "n_approx": n_approx,
        "n_detail": n_detail,
        "detail_lengths": detail_lengths,
    }

    return Psi_approx, Psi_detail, level_info


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


def build_all_matrices(
    band_data: BandData, config: SEDConfig
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray, Dict[str, int], Dict[str, Any]]:
    """
    Build all matrices required for wavelet-based reconstruction of a single band.

    This is the main entry point for matrix construction. It:
    1. Uses hardcoded detector wavelength range (not data-derived)
    2. Adds edge padding based on wavelet filter length to mitigate boundary effects
    3. Builds measurement matrix H on extended grid
    4. Builds wavelet transform matrices (Psi_approx, Psi_detail) on extended grid
    5. Builds weight vector w

    Parameters
    ----------
    band_data : BandData
        Measurement data for one detector band.
    config : SEDConfig
        Configuration with resolution_samples, wavelet settings, etc.

    Returns
    -------
    H : sp.csr_matrix
        Measurement matrix on extended grid, shape (M, N_extended).
    Psi_approx : sp.csr_matrix
        Approximation coefficient extraction matrix, shape (n_approx, N_extended).
    Psi_detail : sp.csr_matrix
        Detail coefficient extraction matrix, shape (n_detail, N_extended).
    weights : np.ndarray
        Weight vector, shape (M,).
    wavelength_grid_extended : np.ndarray
        Extended wavelength grid including edge padding, shape (N_extended,).
    level_info : dict
        Wavelet decomposition information (level, coefficient counts).
    edge_info : dict
        Edge padding information with keys:
        - 'edge_padding_pixels': Number of pixels padded on each side
        - 'trim_start': Index where valid detector range starts
        - 'trim_end': Index where valid detector range ends
        - 'wavelength_grid_trimmed': Trimmed grid matching detector range (N,)
        - 'detector_range': Hardcoded detector wavelength range (lambda_min, lambda_max)

    Notes
    -----
    The wavelength grid is extended beyond the detector range by 2 * wavelet_filter_length
    on each side to push boundary effects into padding regions. After reconstruction,
    the padded regions should be trimmed to recover the detector-specified range.

    The wavelet matrices enable multi-scale regularization:
    - Psi_approx extracts low-frequency continuum
    - Psi_detail extracts high-frequency features (emission lines + noise)
    """
    N = config.resolution_samples

    # Get hardcoded detector range (not data-derived)
    lambda_min_detector, lambda_max_detector = DETECTOR_WAVELENGTH_RANGES[band_data.band]

    # Calculate wavelet filter length for edge padding
    wavelet_obj = pywt.Wavelet(config.wavelet_family)
    filter_len = wavelet_obj.dec_len  # Decomposition filter length
    edge_padding_pixels = 2 * filter_len

    logger.info(
        f"Constructing matrices for {band_data.band}: "
        f"detector range {lambda_min_detector:.3f}-{lambda_max_detector:.3f} um, "
        f"N={N} bins, edge_padding={edge_padding_pixels} pixels"
    )

    # Calculate pixel size from detector range
    delta_lambda = (lambda_max_detector - lambda_min_detector) / N

    # Extend wavelength range by edge padding
    edge_padding_wavelength = edge_padding_pixels * delta_lambda
    lambda_min_extended = lambda_min_detector - edge_padding_wavelength
    lambda_max_extended = lambda_max_detector + edge_padding_wavelength

    # Extended grid size
    N_extended = N + 2 * edge_padding_pixels

    # Define extended wavelength grid
    wavelength_grid_extended = np.linspace(lambda_min_extended, lambda_max_extended, N_extended, dtype=np.float64)

    logger.info(
        f"Extended grid: {lambda_min_extended:.3f}-{lambda_max_extended:.3f} um, "
        f"N_extended={N_extended} bins, resolution={delta_lambda:.6f} um ({1e4 * delta_lambda:.2f} Angstrom)"
    )

    # Build matrices on extended grid
    H = build_measurement_matrix(band_data, wavelength_grid_extended, config)
    Psi_approx, Psi_detail, level_info = build_wavelet_matrices(
        N_extended,
        wavelet=config.wavelet_family,
        level=config.wavelet_level,
        mode=config.wavelet_boundary_mode,
    )
    weights = build_weight_vector(band_data, config)

    # Prepare trimmed wavelength grid (detector range only)
    trim_start = edge_padding_pixels
    trim_end = N_extended - edge_padding_pixels
    wavelength_grid_trimmed = wavelength_grid_extended[trim_start:trim_end]

    # Package edge information
    edge_info = {
        "edge_padding_pixels": edge_padding_pixels,
        "trim_start": trim_start,
        "trim_end": trim_end,
        "wavelength_grid_trimmed": wavelength_grid_trimmed,
        "detector_range": (lambda_min_detector, lambda_max_detector),
    }

    logger.info(f"Edge padding applied: {edge_padding_pixels} pixels = {edge_padding_wavelength:.4f} um per side")

    return H, Psi_approx, Psi_detail, weights, wavelength_grid_extended, level_info, edge_info
