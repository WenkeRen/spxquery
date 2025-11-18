"""
Sparse matrix construction for SED reconstruction with unit consistency.

This module implements the mathematical operators required for regularized
least-squares spectral reconstruction:
- H: Measurement matrix (M x N) relating observations to spectrum with frequency step normalization
- Psi_approx: Wavelet approximation coefficient extraction matrix
- Psi_detail: Wavelet detail coefficient extraction matrix
- w: Weight vector (M,) for chi-squared data fidelity

The measurement matrix H now incorporates proper frequency step normalization
to handle non-uniform wavelength grids while maintaining energy conservation.
Input and output fluxes remain in microjansky (μJy).
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pywt
import scipy.sparse as sp
from astropy.constants import c

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


def build_swt_matrices(
    N: int, wavelet: str = "sym6", level: Optional[int] = None, mode: str = "symmetric"
) -> Tuple[List[sp.csr_matrix], Dict[str, int]]:
    """
    Build Stationary Wavelet Transform (SWT) matrices for multi-scale regularization.

    This function constructs J+1 sparse matrices that extract SWT coefficients:
    - List[0]: Psi_A - Extracts approximation coefficients cA_J (low-frequency continuum)
    - List[1..J]: Psi_D_j - Extracts detail coefficients cD_j at each scale
    where J is the decomposition level (coarsest level = J, finest = 1)

    SWT provides shift-invariant decomposition, eliminating pseudo-Gibbs artifacts
    that occur with standard DWT. All coefficient arrays have approximately the same
    length as the input signal (redundant representation).

    Parameters
    ----------
    N : int
        Number of wavelength bins in spectrum.
    wavelet : str, optional
        Wavelet family name (default: 'sym6').
        Common choices: 'sym4', 'sym6', 'sym8', 'db4', 'db6', 'db8'.
    level : int, optional
        Number of decomposition levels. If None, auto-detect using
        pywt.swt_max_level(N, wavelet).
    mode : str, optional
        Boundary mode for wavelet transform (default: 'symmetric').
        SWT supports: 'symmetric', 'periodic', 'periodization', 'smooth', 'constant',
        'reflect', 'antisymmetric', 'antireflect'.
        Symmetric mode is recommended for spectra to avoid edge artifacts.

    Returns
    -------
    Psi_operators : List[sp.csr_matrix]
        List of J+1 coefficient extraction matrices: [Psi_A, Psi_D_J, ..., Psi_D_1].
        Index 0 contains Psi_A (approximation), indices 1..J contain Psi_D_j (detail).
    level_info : dict
        Dictionary with keys:
        - 'level': decomposition level used
        - 'n_approx': number of approximation coefficients
        - 'num_operators': total number of operators (J+1)
        - 'detail_lengths': list of detail coefficient counts per level

    Raises
    ------
    ValueError
        If N is too small for SWT decomposition or wavelet is invalid.

    Notes
    -----
    SWT (Stationary Wavelet Transform) differs from DWT in key ways:
    - Shift-invariant: No pseudo-Gibbs artifacts from signal shifts
    - Redundant coefficients: All levels have ~N coefficients (vs N/2^j for DWT)
    - Per-scale regularization: Each frequency scale can be controlled independently

    The SWT coefficient structure is: pywt.swt(signal, level=J) returns
    [(cA_J, cD_J), (cA_{J-1}, cD_{J-1}), ..., (cA_1, cD_1)]
    which we rearrange to [cA_J, cD_J, cD_{J-1}, ..., cD_1] for matrix construction.
    """
    if N < 4:
        raise ValueError(f"N must be >= 4 for SWT decomposition, got N={N}")

    # Auto-detect decomposition level if not specified
    # For SWT, we use pywt.dwt_max_level (SWT uses same max level as DWT)
    if level is None:
        level = pywt.dwt_max_level(N, wavelet)
        logger.info(f"Auto-detected SWT level: {level} (for N={N}, wavelet='{wavelet}')")
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

    logger.info(f"Building SWT matrices: N={N}, wavelet='{wavelet}', level={level}, mode='{mode}'")

    # Perform test SWT decomposition to get coefficient structure
    test_signal = np.zeros(N, dtype=np.float64)
    test_signal[N // 2] = 1.0  # Delta function at center
    coeffs_swt = pywt.swt(test_signal, wavelet, level=level, start_level=0, trim_approx=False, norm=True)

    # coeffs_swt = [(cA_J, cD_J), (cA_{J-1}, cD_{J-1}), ..., (cA_1, cD_1)]
    # where J = level
    # We'll build J+1 matrices: Psi_A, Psi_D_J, Psi_D_{J-1}, ..., Psi_D_1

    # Extract coefficient structure
    cA_J = coeffs_swt[0][0]  # Coarsest approximation coefficients
    cD_tests = [coeffs_swt[i][1] for i in range(level)]  # Detail coefficients from coarsest to finest
    # cD_tests[0] = cD_J (coarsest), cD_tests[level-1] = cD_1 (finest)

    n_approx = len(cA_J)  # All SWT coefficients have same length (~N)
    detail_lengths = [len(cD) for cD in cD_tests]  # All should be ~N
    num_operators = level + 1  # J+1 operators total

    logger.info(
        f"SWT coefficient structure: n_approx={n_approx}, num_operators={num_operators} "
        f"(detail lengths: {detail_lengths})"
    )

    # Build J+1 SWT matrices: [Psi_A, Psi_D_J, Psi_D_{J-1}, ..., Psi_D_1]
    Psi_operators = []

    # 1. Build Psi_A: Extract approximation coefficients cA_J
    rows_A = []
    cols_A = []
    data_A = []

    logger.info("Building SWT approximation matrix (Psi_A)...")
    for i in range(n_approx):
        # Create SWT coefficient structure with unit at cA_J[i]
        unit_coeffs_swt = []
        for lev in range(level):
            # For each level, we need (cA_{J-lev}, cD_{J-lev}) pairs
            cA_level = np.zeros_like(coeffs_swt[lev][0])
            cD_level = np.zeros_like(coeffs_swt[lev][1])
            unit_coeffs_swt.append((cA_level, cD_level))

        # Set unit in coarsest approximation coefficients cA_J
        unit_coeffs_swt[0] = (np.zeros_like(coeffs_swt[0][0]), np.zeros_like(coeffs_swt[0][1]))
        cA_J_unit = np.zeros_like(cA_J)
        cA_J_unit[i] = 1.0
        unit_coeffs_swt[0] = (cA_J_unit, unit_coeffs_swt[0][1])

        # Reconstruct signal using inverse SWT
        try:
            signal = pywt.iswt(unit_coeffs_swt, wavelet)
        except Exception as e:
            logger.error(f"Error in ISWT reconstruction for Psi_A[{i}]: {e}")
            continue

        # Handle potential length mismatch due to padding
        if len(signal) > N:
            signal = signal[:N]
        elif len(signal) < N:
            signal = np.pad(signal, (0, N - len(signal)), mode="constant")

        # Store non-zero entries
        nonzero_indices = np.where(np.abs(signal) > 1e-12)[0]
        for j in nonzero_indices:
            rows_A.append(i)
            cols_A.append(j)
            data_A.append(signal[j])

    Psi_A = sp.coo_matrix((data_A, (rows_A, cols_A)), shape=(n_approx, N), dtype=np.float64).tocsr()
    Psi_operators.append(Psi_A)

    # 2. Build Psi_D matrices for each detail level
    for level_idx in range(level):
        # level_idx = 0 corresponds to cD_J (coarsest details)
        # level_idx = level-1 corresponds to cD_1 (finest details)

        rows_D = []
        cols_D = []
        data_D = []

        logger.info(f"Building SWT detail matrix (Psi_D_{level_idx + 1})...")

        for i in range(detail_lengths[level_idx]):
            # Create SWT coefficient structure with unit at cD_{J-level_idx}[i]
            unit_coeffs_swt = []
            for lev in range(level):
                cA_level = np.zeros_like(coeffs_swt[lev][0])
                cD_level = np.zeros_like(coeffs_swt[lev][1])
                unit_coeffs_swt.append((cA_level, cD_level))

            # Set unit in appropriate detail coefficients
            if level_idx == 0:
                # cD_J (coarsest details)
                cD_J_unit = np.zeros_like(coeffs_swt[0][1])
                cD_J_unit[i] = 1.0
                unit_coeffs_swt[0] = (unit_coeffs_swt[0][0], cD_J_unit)
            else:
                # For other levels, we need to set the detail coefficients at that level
                target_coeffs = coeffs_swt[level_idx]
                cD_target_unit = np.zeros_like(target_coeffs[1])
                cD_target_unit[i] = 1.0
                unit_coeffs_swt[level_idx] = (unit_coeffs_swt[level_idx][0], cD_target_unit)

            # Reconstruct signal using inverse SWT
            try:
                signal = pywt.iswt(unit_coeffs_swt, wavelet)
            except Exception as e:
                logger.error(f"Error in ISWT reconstruction for Psi_D_{level_idx + 1}[{i}]: {e}")
                continue

            # Handle potential length mismatch
            if len(signal) > N:
                signal = signal[:N]
            elif len(signal) < N:
                signal = np.pad(signal, (0, N - len(signal)), mode="constant")

            # Store non-zero entries
            nonzero_indices = np.where(np.abs(signal) > 1e-12)[0]
            for j in nonzero_indices:
                rows_D.append(i)
                cols_D.append(j)
                data_D.append(signal[j])

        Psi_D = sp.coo_matrix(
            (data_D, (rows_D, cols_D)), shape=(detail_lengths[level_idx], N), dtype=np.float64
        ).tocsr()
        Psi_operators.append(Psi_D)

    # Log sparsity statistics for all operators
    logger.info(f"Psi_A: {Psi_A.nnz:,} non-zero entries, shape ({n_approx}, {N})")
    for i, Psi_D in enumerate(Psi_operators[1:]):
        logger.info(f"Psi_D_{level - i}: {Psi_D.nnz:,} non-zero entries, shape ({detail_lengths[i]}, {N})")

    level_info = {
        "level": level,
        "n_approx": n_approx,
        "num_operators": num_operators,
        "detail_lengths": detail_lengths,
    }

    return Psi_operators, level_info


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


def compute_spatial_weights(N_extended: int, edge_info: Dict[str, Any], config: SEDConfig) -> np.ndarray:
    """
    Compute spatial weight vector for L1 regularization to reduce Gibbs Phenomenon.

    The spatial weighting reduces regularization constraints in edge padding regions
    while maintaining strong regularization in scientific regions. This allows the
    optimizer to freely adjust wavelet coefficients in padding areas where they
    primarily contain boundary artifacts rather than scientific signal.

    Parameters
    ----------
    N_extended : int
        Length of extended wavelength grid including padding.
    edge_info : Dict[str, Any]
        Edge information dictionary containing:
        - 'trim_start': Index where valid detector range starts
        - 'trim_end': Index where valid detector range ends
        - 'edge_padding_pixels': Number of pixels padded on each side
    config : SEDConfig
        Configuration with spatial weighting parameters.

    Returns
    -------
    np.ndarray
        Spatial weight vector, shape (N_extended,).
        - Science regions: science_region_weight (typically 1.0)
        - Padding regions: padding_region_weight (typically 0.0)
        - Transition regions: smooth interpolation between padding and science weights

    Notes
    -----
    The implementation uses smooth transitions between regions based on:
    - transition_width: Width of transition zone in pixels
    - transition_region_weight: Weight in transition zone (if transition_width = 0, this acts as hard cutoff)
    """
    if not config.spatial_weight_enabled:
        # Return uniform weights if spatial weighting is disabled
        return np.ones(N_extended, dtype=np.float64)

    trim_start = edge_info["trim_start"]
    trim_end = edge_info["trim_end"]

    # Initialize with science region weights
    W_spatial = np.full(N_extended, config.science_region_weight, dtype=np.float64)

    # Set padding regions with smaller weights
    W_spatial[:trim_start] = config.padding_region_weight
    W_spatial[trim_end:] = config.padding_region_weight

    # Apply smooth transitions if transition_width > 0
    if config.transition_width > 0:
        # Left transition zone (from padding to science)
        left_transition_end = min(trim_start + config.transition_width, trim_end)
        if left_transition_end > trim_start:
            # Create smooth transition from padding to science weights
            left_transition = np.linspace(
                config.padding_region_weight, config.science_region_weight, left_transition_end - trim_start
            )
            W_spatial[trim_start:left_transition_end] = left_transition

        # Right transition zone (from science to padding)
        right_transition_start = max(trim_end - config.transition_width, trim_start)
        if right_transition_start < trim_end:
            # Create smooth transition from science to padding weights
            right_transition = np.linspace(
                config.science_region_weight, config.padding_region_weight, trim_end - right_transition_start
            )
            W_spatial[right_transition_start:trim_end] = right_transition

    # Log spatial weight distribution
    n_science = trim_end - trim_start
    n_padding = N_extended - n_science
    transition_pixels = 2 * min(config.transition_width, n_science // 2) if config.transition_width > 0 else 0

    logger.info(
        f"Spatial weights: science={n_science} pixels (weight={config.science_region_weight}), "
        f"padding={n_padding} pixels (weight={config.padding_region_weight})"
    )
    if transition_pixels > 0:
        logger.info(
            f"  Transition: {transition_pixels} pixels with smooth interpolation "
            f"from {config.padding_region_weight} to {config.science_region_weight}"
        )
    logger.info(
        f"  Index range: science=[{trim_start}, {trim_end}), "
        f"left_padding=[0, {trim_start}), right_padding=[{trim_end}, {N_extended})"
    )

    return W_spatial


def build_all_matrices(
    band_data: BandData, config: SEDConfig
) -> Tuple[sp.csr_matrix, List[sp.csr_matrix], np.ndarray, np.ndarray, Dict[str, int], Dict[str, Any], np.ndarray]:
    """
    Build all matrices required for SWT-based reconstruction of a single band.

    This is the main entry point for matrix construction. It:
    1. Uses hardcoded detector wavelength range (not data-derived)
    2. Adds edge padding based on wavelet filter length to mitigate boundary effects
    3. Ensures signal length meets SWT requirements (multiple of 2**level)
    4. Builds measurement matrix H on extended grid
    5. Builds Stationary Wavelet Transform matrices (Psi_A, Psi_D_J, ..., Psi_D_1) on extended grid
    6. Builds weight vector w
    7. Builds spatial weight vector for L1 regularization (if enabled)

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
    Psi_operators : List[sp.csr_matrix]
        List of J+1 SWT coefficient extraction matrices: [Psi_A, Psi_D_J, ..., Psi_D_1].
        Index 0 contains Psi_A (approximation), indices 1..J contain Psi_D_j (detail).
    weights : np.ndarray
        Weight vector, shape (M,).
    wavelength_grid_extended : np.ndarray
        Extended wavelength grid including padding, shape (N_extended,).
    level_info : dict
        SWT decomposition information (level, coefficient counts).
    edge_info : dict
        Edge padding information with keys:
        - 'edge_padding_pixels': Number of pixels padded on each side
        - 'trim_start': Index where valid detector range starts
        - 'trim_end': Index where valid detector range ends
        - 'wavelength_grid_trimmed': Trimmed grid matching detector range (N,)
        - 'detector_range': Hardcoded detector wavelength range (lambda_min, lambda_max)
        - 'swt_padding_pixels': Additional padding for SWT requirements
        - 'total_padding_pixels': Total padding applied
    spatial_weights : np.ndarray
        Spatial weight vector for L1 regularization, shape (N_extended,).
        Applied element-wise to SWT coefficients to reduce Gibbs Phenomenon at edges.

    Notes
    -----
    The wavelength grid is extended beyond the detector range to:
    1. Push boundary effects into padding regions (edge padding)
    2. Meet SWT requirements that signal length must be multiple of 2**level

    SWT requires signal length to be a multiple of 2**level for the "algorithm a-trous"
    implementation. If this requirement is not met, additional padding is added.

    The SWT matrices enable shift-invariant multi-scale regularization:
    - Psi_A (Psi_operators[0]) extracts low-frequency continuum
    - Psi_D_j (Psi_operators[1..J]) extract detail coefficients at different frequency scales
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
        f"N={N} bins, initial edge_padding={edge_padding_pixels} pixels"
    )

    # Determine SWT level for validation
    if config.wavelet_level is None:
        swt_level = pywt.dwt_max_level(N + 2 * edge_padding_pixels, config.wavelet_family)
    else:
        swt_level = config.wavelet_level

    # Calculate SWT padding requirements
    # SWT requires signal length to be multiple of 2**level
    base_size = N + 2 * edge_padding_pixels
    required_multiple = 2**swt_level

    if base_size % required_multiple != 0:
        # Calculate total additional padding needed
        additional_padding_needed = required_multiple - (base_size % required_multiple)

        # Ensure additional padding is even (for symmetric padding on both sides)
        if additional_padding_needed % 2 != 0:
            additional_padding_needed += required_multiple

        # Split padding evenly on both sides
        swt_padding_each_side = additional_padding_needed // 2
    else:
        swt_padding_each_side = 0

    # Check total padding percentage
    total_padding_pixels = edge_padding_pixels + swt_padding_each_side
    padding_percentage = (2 * total_padding_pixels) / N * 100

    if padding_percentage > 15.0:
        logger.warning(
            f"Total padding ({2 * total_padding_pixels} pixels, {padding_percentage:.1f}%) "
            f"exceeds 15% of resolution_samples ({N}). "
            f"Consider reducing wavelet_level ({swt_level}) or increasing resolution_samples."
        )

    # Validate user-specified level vs auto-detected level
    if config.wavelet_level is not None:
        auto_level = pywt.dwt_max_level(N + 2 * edge_padding_pixels, config.wavelet_family)
        if config.wavelet_level < auto_level - 2:
            logger.warning(
                f"User-specified wavelet_level ({config.wavelet_level}) is much lower than "
                f"auto-detected maximum ({auto_level}). This may limit reconstruction quality. "
                f"Consider using a higher level or removing the wavelet_level constraint."
            )

    # Calculate final extended grid size
    N_extended = N + 2 * total_padding_pixels

    # Calculate pixel size from detector range
    delta_lambda = (lambda_max_detector - lambda_min_detector) / N

    # Extend wavelength range by total padding
    total_padding_wavelength = total_padding_pixels * delta_lambda
    lambda_min_extended = lambda_min_detector - total_padding_wavelength
    lambda_max_extended = lambda_max_detector + total_padding_wavelength

    # Define extended wavelength grid
    wavelength_grid_extended = np.linspace(lambda_min_extended, lambda_max_extended, N_extended, dtype=np.float64)

    logger.info(
        f"Final extended grid: {lambda_min_extended:.3f}-{lambda_max_extended:.3f} um, "
        f"N_extended={N_extended} bins (base={N}, edge_padding={edge_padding_pixels}, "
        f"swt_padding={swt_padding_each_side}), resolution={delta_lambda:.6f} um "
        f"({1e4 * delta_lambda:.2f} Angstrom), level={swt_level}, padding={padding_percentage:.1f}%"
    )

    # Build matrices on extended grid
    H = build_measurement_matrix(band_data, wavelength_grid_extended, config)
    Psi_operators, level_info = build_swt_matrices(
        N_extended,
        wavelet=config.wavelet_family,
        level=config.wavelet_level,
        mode=config.wavelet_boundary_mode,
    )
    weights = build_weight_vector(band_data, config)

    # Prepare trimmed wavelength grid (detector range only)
    trim_start = total_padding_pixels
    trim_end = N_extended - total_padding_pixels
    wavelength_grid_trimmed = wavelength_grid_extended[trim_start:trim_end]

    # Package edge information
    edge_info = {
        "edge_padding_pixels": edge_padding_pixels,
        "swt_padding_pixels": swt_padding_each_side,
        "total_padding_pixels": total_padding_pixels,
        "padding_percentage": padding_percentage,
        "trim_start": trim_start,
        "trim_end": trim_end,
        "wavelength_grid_trimmed": wavelength_grid_trimmed,
        "detector_range": (lambda_min_detector, lambda_max_detector),
        "swt_level": swt_level,
        "required_multiple": required_multiple,
    }

    logger.info(
        f"Padding applied: {edge_padding_pixels} edge + {swt_padding_each_side} SWT = "
        f"{total_padding_pixels} total pixels = {total_padding_wavelength:.4f} um per side "
        f"({padding_percentage:.1f}% of N)"
    )

    # 7. Build spatial weights for L1 regularization
    spatial_weights = compute_spatial_weights(N_extended, edge_info, config)

    return H, Psi_operators, weights, wavelength_grid_extended, level_info, edge_info, spatial_weights
