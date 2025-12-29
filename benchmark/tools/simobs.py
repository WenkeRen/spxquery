"""
SPHEREx Observation Simulator for Benchmark Tests

This module provides tools to simulate SPHEREx observations from synthetic models:
- Load calibration data (abs_gain, readnoise) from SPHEREx QR2 files
- Convert model spectra from physical units to observational units
- Simulate clean band photometry using filter response functions
- Add realistic SPHEREx noise (read noise + photon noise)
- Generate BandData objects compatible with reconstruction pipeline

Author: SPHEREx Spectral Reconstruction Team
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.io import fits

from spxquery.sed.config import SEDConfig
from spxquery.sed.data_loader import BandData
from spxquery.sed.matrices import build_measurement_matrix

# SPHEREx calibration data path
SPHEREX_QR2_PATH = Path("/Volumes/QuarkStar/SPHEREx_NEP/qr2")

# ============================================================================
# Calibration Data Loaders
# ============================================================================


def load_abs_gain_for_band(band_id: str, qr2_path: Optional[Path] = None) -> float:
    """
    Load absolute gain factor for a SPHEREx band from calibration FITS files.

    Parameters
    ----------
    band_id : str
        Band identifier (e.g., 'D1', 'D2', ..., 'D6').
    qr2_path : Path, optional
        Path to SPHEREx QR2 calibration directory.
        Defaults to SPHEREX_QR2_PATH.

    Returns
    -------
    abs_gain : float
        Median absolute gain factor in units of [MJy/sr] / [e-/s].

    Raises
    ------
    ValueError
        If band_id format is invalid or FITS file cannot be read.
    FileNotFoundError
        If calibration file is not found.

    Notes
    -----
    Loads from: qr2_path/abs_gain_matrix/cal-agm-v7-2025-218/{N}/
                    abs_gain_matrix_D{N}_spx_cal-agm-v7-2025-218.fits
    where N is extracted from band_id (e.g., 'D1' -> N=1).

    The median of extension [1].data is returned as the representative value.
    """
    if qr2_path is None:
        qr2_path = SPHEREX_QR2_PATH

    # Extract band number from band_id (e.g., 'D1' -> 1)
    if not band_id.startswith("D") or len(band_id) != 2:
        raise ValueError(f"Invalid band_id format: '{band_id}'. Expected format: 'D1', 'D2', etc.")

    try:
        band_num = int(band_id[1])
    except ValueError:
        raise ValueError(f"Cannot extract band number from band_id: '{band_id}'")

    # Construct file path
    abs_gain_file = (
        qr2_path
        / "abs_gain_matrix"
        / "cal-agm-v7-2025-218"
        / str(band_num)
        / f"abs_gain_matrix_D{band_num}_spx_cal-agm-v7-2025-218.fits"
    )

    if not abs_gain_file.exists():
        raise FileNotFoundError(f"Absolute gain file not found: {abs_gain_file}")

    # Load FITS file
    with fits.open(abs_gain_file) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"Unexpected FITS file structure in {abs_gain_file}")

        # Get median of extension [1].data
        abs_gain_data = hdul[1].data
        abs_gain = float(np.median(abs_gain_data))

    return abs_gain


def load_readnoise_for_band(band_id: str, qr2_path: Optional[Path] = None) -> float:
    """
    Load detector read noise for a SPHEREx band from calibration FITS files.

    Parameters
    ----------
    band_id : str
        Band identifier (e.g., 'D1', 'D2', ..., 'D6').
    qr2_path : Path, optional
        Path to SPHEREx QR2 calibration directory.
        Defaults to SPHEREX_QR2_PATH.

    Returns
    -------
    sigma_det : float
        Median read noise in electrons (e-).

    Raises
    ------
    ValueError
        If band_id format is invalid or FITS file cannot be read.
    FileNotFoundError
        If calibration file is not found.

    Notes
    -----
    Loads from: qr2_path/readnoise_pars/base-2025-158/{N}/
                    readnoise_pars_D{N}_spx_base-2025-158.fits
    where N is extracted from band_id (e.g., 'D1' -> N=1).

    The median of extension [2].data is returned as the representative value.
    """
    if qr2_path is None:
        qr2_path = SPHEREX_QR2_PATH

    # Extract band number from band_id (e.g., 'D1' -> 1)
    if not band_id.startswith("D") or len(band_id) != 2:
        raise ValueError(f"Invalid band_id format: '{band_id}'. Expected format: 'D1', 'D2', etc.")

    try:
        band_num = int(band_id[1])
    except ValueError:
        raise ValueError(f"Cannot extract band number from band_id: '{band_id}'")

    # Construct file path
    readnoise_file = (
        qr2_path
        / "readnoise_pars"
        / "base-2025-158"
        / str(band_num)
        / f"readnoise_pars_D{band_num}_spx_base-2025-158.fits"
    )

    if not readnoise_file.exists():
        raise FileNotFoundError(f"Readnoise file not found: {readnoise_file}")

    # Load FITS file
    with fits.open(readnoise_file) as hdul:
        if len(hdul) < 3:
            raise ValueError(f"Unexpected FITS file structure in {readnoise_file}")

        # Get median of extension [2].data
        readnoise_data = hdul[2].data
        sigma_det = float(np.median(readnoise_data))

    return sigma_det


# ============================================================================
# Unit Conversion Functions
# ============================================================================


def convert_model_flux_to_uJy(
    wavelength_microns: np.ndarray,
    flux_erg_s_cm2_A: np.ndarray,
) -> np.ndarray:
    """
    Convert model spectrum flux from erg/s/cm^2/A to microJansky.

    Parameters
    ----------
    wavelength_microns : np.ndarray
        Wavelength array in microns.
    flux_erg_s_cm2_A : np.ndarray
        Flux array in erg/s/cm^2/Angstrom.

    Returns
    -------
    flux_uJy : np.ndarray
        Flux array in microJansky (uJy).

    Notes
    -----
    Uses astropy.units for conversion with spectral density equivalency.

    Example:
    --------
    >>> wavelength = np.array([1.0, 2.0, 3.0])  # microns
    >>> flux_model = np.array([1.0e-15, 1.0e-15, 1.0e-15])  # erg/s/cm^2/A
    >>> flux_uJy = convert_model_flux_to_uJy(wavelength, flux_model)
    >>> print(f"Flux in uJy: {flux_uJy}")
    """
    # Create astropy quantities
    wav = wavelength_microns * u.micron
    flux = flux_erg_s_cm2_A * u.erg / u.s / u.cm**2 / u.AA

    # Convert to uJy using spectral density equivalency
    flux_uJy = flux.to(u.uJy, equivalencies=u.spectral_density(wav)).value

    return flux_uJy


# ============================================================================
# Observation Simulation Functions
# ============================================================================


def simulate_clean_observations_from_model(
    model_spectrum: np.ndarray,
    wavelength_centers: np.ndarray,
    bandwidths: np.ndarray,
    band_id: str,
    filter_profile: str = "boxcar",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate clean SPHEREx observations from a model spectrum.

    This function:
    1. Converts model spectrum from erg/s/cm^2/A to uJy
    2. Builds a measurement matrix H using filter response functions
    3. Applies H to convert high-resolution spectrum to band observations

    Parameters
    ----------
    model_spectrum : np.ndarray
        Model spectrum as 2-row array:
        - Row 0: wavelength in microns
        - Row 1: flux in erg/s/cm^2/Angstrom
    wavelength_centers : np.ndarray
        Central wavelengths of observations in microns, shape (M,).
    bandwidths : np.ndarray
        Bandwidths of observations in microns, shape (M,).
    band_id : str
        Band identifier (e.g., 'D1', 'D2', ..., 'D6').
    filter_profile : str, optional
        Filter response profile: 'boxcar' or 'gaussian'. Default: 'boxcar'.

    Returns
    -------
    flux_clean_uJy : np.ndarray
        Simulated flux measurements in microJansky, shape (M,).
    wavelength_grid : np.ndarray
        Wavelength grid used for the measurement matrix (microns), shape (N,).

    Raises
    ------
    ValueError
        If model_spectrum shape is incorrect or filter_profile is invalid.

    Notes
    -----
    The measurement matrix H is built using the same approach as in
    src/spxquery/sed/matrices.py, incorporating frequency step normalization
    for proper energy conservation with non-uniform wavelength grids.

    Forward model: y = H @ x
    where:
    - y: observed flux in uJy (M,)
    - H: measurement matrix (M x N)
    - x: model spectrum in uJy (N,)
    """
    # Validate model spectrum shape
    if model_spectrum.shape[0] != 2:
        raise ValueError(
            f"model_spectrum must have shape (2, N), got {model_spectrum.shape}. "
            "Row 0: wavelength (microns), Row 1: flux (erg/s/cm^2/A)"
        )

    # Extract wavelength and flux from model spectrum
    wavelength_grid = model_spectrum[0, :].copy()
    flux_model = model_spectrum[1, :].copy()

    # Convert model flux to uJy
    flux_model_uJy = convert_model_flux_to_uJy(wavelength_grid, flux_model)

    # Create a temporary BandData object for measurement matrix construction
    # (We only need wavelength_centers and bandwidths)
    temp_band_data = BandData(
        band=band_id,
        flux=np.zeros(len(wavelength_centers)),  # Dummy flux, will be overwritten
        flux_error=np.ones(len(wavelength_centers)),  # Dummy errors
        wavelength_center=wavelength_centers,
        bandwidth=bandwidths,
    )

    # Create SEDConfig with specified filter profile
    config = SEDConfig(filter_profile=filter_profile)

    # Build measurement matrix H
    H = build_measurement_matrix(temp_band_data, wavelength_grid, config)

    # Apply measurement matrix: y = H @ x
    # H is sparse CSR matrix, flux_model_uJy is numpy array
    flux_clean_uJy = H @ flux_model_uJy

    return flux_clean_uJy, wavelength_grid


def calculate_spherex_flux_error(
    source_flux,
    num_pixels,
    sigma_det,
    abs_gain,
    omega_pix_arcsec2=37.82,  # 6.15 arcsec/pixel
    bg_surface_brightness=0.1,  # MJy/sr, typical Zodi
    T_int=113.58,  # Default from SPHEREx Table 2
    N_reads=77,  # Default full ramp
):
    """
    Estimates the 1-sigma uncertainty of SPHEREx flux measurements (in uJy).

    Vectorized to accept both scalar and array inputs for source_flux.

    The variance of the measured slope (e-/s)^2 is calculated using the formula from
    Robberto (2007) / SPHEREx Eq 5, accounting for read noise and photon noise.
        Var = [ sigma_det(N)^2 + (6/5) * (N^2+1)/(N^2-1) * (F * T_int) ] / T_{int}^2
    where F is the pixel signal rate (source + background) in e-/s.
    The variance in [MJy/s]^2 can be then calculated as follow:
        Var_flux = K_conv^2 * [ readnoise_term + photon_term ]
    where K_conv is the conversion factor in unit of [MJy]/[e-/s], readnoise_term is the first term, and photon_term is
    the second term in the numerator above.
        K_conv [MJy]/[e-/s] = omega_pix [sr] * abs_gain [MJy/sr]/[e-/s]
        readnoise_term ([e-/s]^2) = num_pixels * sigma_det^2 / (T_int^2)
        photon_term [e-/s]^2 = (6/5) * (N^2+1)/(N^2-1) * (F_tot [e-/s] * T_int) / (T_int^2) (This is a experimental
        term which does not follow dimensional analysis)
        F_tot [e-/s] = F_source + F_bg = (flux_source [MJy] + num_pixels * flux_bg_per_pixel [MJy]) / K_conv

    Parameters:
    -----------
    source_flux : float or np.ndarray
        The total flux of the source in micro-Janskys (uJy).
        Can be a scalar or array.
    num_pixels : int
        Number of pixels in the aperture (N_pix).
    sigma_det : float
        Detector read noise in electrons (e-) per pixel (RMS of the ramp).
    abs_gain : float
        Absolute Gain factor in [MJy/sr] / [e-/s].
    omega_pix_arcsec2 : float
        Solid angle of one pixel in arcsec^2 (e.g., 6.15^2 ~ 37.8).
    bg_surface_brightness : float
        Background surface brightness (Zodi + others) in MJy/sr.
    T_int : float, optional
        Total integration time in seconds. Default is 113.58 s.
    N_reads : int, optional
        Total number of non-destructive reads. Default is 77.

    Returns:
    --------
    sigma_flux_uJy : float or np.ndarray
        The estimated 1-sigma error of the flux in uJy.
        Same shape as input source_flux.
    """
    import astropy.units as u

    # --- 1. Constants & Conversions ---
    # Convert Omega from arcsec^2 to steradians
    omega_pix_sr = omega_pix_arcsec2 * (u.arcsec**2).to(u.sr)

    # Conversion Factor K_conv
    # K_conv is the factor to convert from electrons/sec to MJy
    # Units: [MJy] / [e-/s] = [sr] * [MJy/sr] / [e-/s]
    K_conv = omega_pix_sr * abs_gain  # in [MJy/[e-/s]]

    # --- 2. Calculate Signals in Electrons/sec ---
    # Source signal rate
    # source_flux is in uJy, convert to MJy first (1 uJy = 1e-12 MJy)
    source_flux_MJy = np.asarray(source_flux) * 1.0e-12
    rate_source_e_s = source_flux_MJy / K_conv

    # Background signal rate
    # BG Flux per pixel (MJy) = Surface Brightness (MJy/sr) * Omega_pix (sr)
    flux_bg_per_pixel_MJy = bg_surface_brightness * omega_pix_sr
    # Total BG Flux (MJy) = flux_bg_per_pixel_MJy * num_pixels
    # Rate = Flux / K_conv
    rate_bg_e_s = (flux_bg_per_pixel_MJy * num_pixels) / K_conv

    # Total rate driving photon noise (Source + Background)
    # Note: Photon noise depends on the total accumulated charge
    total_rate_e_s = rate_source_e_s + rate_bg_e_s

    # --- 3. Calculate Variance of the Slope (Flux) ---
    # Formula from Robberto (2007) / SPHEREx Eq 5:
    # Var_slope = (1/T^2) * [ ReadNoise_term + PhotonNoise_term ]

    # A. Read Noise Term
    # Sigma_read^2 scales linearly with number of pixels in aperture
    # Note: This is the variance on the SLOPE fit due to read noise.
    var_read_slope = num_pixels * (sigma_det**2) / (T_int**2)

    # B. Photon Noise Term
    # Factor accounting for correlation in up-the-ramp sampling
    if N_reads > 1:
        corr_factor = (6 / 5) * (N_reads**2 + 1) / (N_reads**2 - 1)
    else:
        corr_factor = 1.0  # Fallback (should not happen for SUR)

    # Photon variance on the slope = Factor * (Total Rate / T_int)
    var_photon_slope = corr_factor * (total_rate_e_s / T_int)

    # Total Variance in (e-/s)^2
    total_var_slope_e2_s2 = var_read_slope + var_photon_slope

    # --- 4. Convert Error back to uJy ---
    sigma_slope_e_s = np.sqrt(total_var_slope_e2_s2)
    # Convert sigma from e-/s to MJy, then to uJy
    sigma_flux_MJy = sigma_slope_e_s * K_conv
    sigma_flux_uJy = sigma_flux_MJy * 1.0e12

    # Return scalar if input was scalar, otherwise array
    if np.isscalar(source_flux):
        return float(sigma_flux_uJy)
    return sigma_flux_uJy


def add_spherex_noise(
    flux_clean_uJy: np.ndarray,
    band_id: str,
    sigma_det: Optional[float] = None,
    abs_gain: Optional[float] = None,
    num_pixels: int = 5,
    bg_surface_brightness: float = 0.1,
    T_int: float = 113.58,
    N_reads: int = 77,
    qr2_path: Optional[Path] = None,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add realistic SPHEREx noise to clean flux measurements (vectorized).

    This function:
    1. Loads abs_gain and sigma_det from FITS files if not provided
    2. Calculates per-measurement uncertainties using SPHEREx noise model (vectorized)
    3. Adds Gaussian noise: flux_noisy = flux_clean + N(0, sigma) (vectorized)

    Parameters
    ----------
    flux_clean_uJy : np.ndarray
        Clean flux measurements in microJansky, shape (M,).
    band_id : str
        Band identifier (e.g., 'D1', 'D2', ..., 'D6').
    sigma_det : float, optional
        Detector read noise in electrons. If None, loads from calibration files.
    abs_gain : float, optional
        Absolute gain in [MJy/sr] / [e-/s]. If None, loads from calibration files.
    num_pixels : int, optional
        Aperture size in pixels. Default: 5 (typical for 2.5 x FWHM).
    bg_surface_brightness : float, optional
        Background surface brightness in MJy/sr. Default: 0.1.
    T_int : float, optional
        Integration time in seconds. Default: 113.58 s.
    N_reads : int, optional
        Number of reads. Default: 77.
    qr2_path : Path, optional
        Path to SPHEREx QR2 calibration directory.
    random_seed : int, optional
        Random seed for reproducibility. If None, uses random state.

    Returns
    -------
    flux_noisy_uJy : np.ndarray
        Noisy flux measurements in microJansky, shape (M,).
    flux_error_uJy : np.ndarray
        Per-measurement uncertainties in microJansky, shape (M,).

    Notes
    -----
    Uses the Robberto (2007) / SPHEREx Eq 5 noise model, which accounts for:
    - Read noise: detector-independent noise per read
    - Photon noise: Poisson noise from source and background

    This function is vectorized for efficiency and returns both noisy flux and errors
    to avoid redundant calculations in the calling code.
    """
    # Load calibration data if not provided
    if abs_gain is None:
        abs_gain = load_abs_gain_for_band(band_id, qr2_path)

    if sigma_det is None:
        sigma_det = load_readnoise_for_band(band_id, qr2_path)

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Calculate uncertainties for all measurements at once (vectorized)
    flux_error_uJy = calculate_spherex_flux_error(
        source_flux=flux_clean_uJy,
        num_pixels=num_pixels,
        sigma_det=sigma_det,
        abs_gain=abs_gain,
        bg_surface_brightness=bg_surface_brightness,
        T_int=T_int,
        N_reads=N_reads,
    )

    # Add Gaussian noise for all measurements at once (vectorized)
    flux_noisy_uJy = flux_clean_uJy + np.random.normal(0, 1, size=flux_clean_uJy.shape) * flux_error_uJy

    return flux_noisy_uJy, flux_error_uJy


def simulate_spherex_observations(
    model_spectrum: np.ndarray,
    wavelength_centers: np.ndarray,
    bandwidths: np.ndarray,
    band_id: str,
    filter_profile: str = "boxcar",
    sigma_det: Optional[float] = None,
    abs_gain: Optional[float] = None,
    num_pixels: int = 5,
    bg_surface_brightness: float = 0.1,
    T_int: float = 113.58,
    N_reads: int = 77,
    qr2_path: Optional[Path] = None,
    random_seed: Optional[int] = None,
) -> BandData:
    """
    Simulate complete SPHEREx observations from a model spectrum.

    This is the main orchestration function that:
    1. Simulates clean observations using filter response functions
    2. Adds realistic SPHEREx noise (read noise + photon noise)
    3. Returns a BandData object compatible with reconstruction pipeline

    Parameters
    ----------
    model_spectrum : np.ndarray
        Model spectrum as 2-row array:
        - Row 0: wavelength in microns
        - Row 1: flux in erg/s/cm^2/Angstrom
    wavelength_centers : np.ndarray
        Central wavelengths of observations in microns, shape (M,).
    bandwidths : np.ndarray
        Bandwidths of observations in microns, shape (M,).
    band_id : str
        Band identifier (e.g., 'D1', 'D2', ..., 'D6').
    filter_profile : str, optional
        Filter response profile: 'boxcar' or 'gaussian'. Default: 'boxcar'.
    sigma_det : float, optional
        Detector read noise in electrons. If None, loads from calibration files.
    abs_gain : float, optional
        Absolute gain in [MJy/sr] / [e-/s]. If None, loads from calibration files.
    num_pixels : int, optional
        Aperture size in pixels. Default: 5.
    bg_surface_brightness : float, optional
        Background surface brightness in MJy/sr. Default: 0.1.
    T_int : float, optional
        Integration time in seconds. Default: 113.58 s.
    N_reads : int, optional
        Number of reads. Default: 77.
    qr2_path : Path, optional
        Path to SPHEREx QR2 calibration directory.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    band_data : BandData
        Simulated observations with all required fields:
        - flux: noisy flux measurements in uJy
        - flux_error: per-measurement uncertainties in uJy
        - wavelength_center: central wavelengths in microns
        - bandwidth: bandwidths in microns
        - band: band identifier
        - n_measurements: number of observations
        - wavelength_range: (min_wavelength, max_wavelength) in microns

    Example
    -------
    >>> from benchmark.tools.models import generate_spectral_model, create_default_model
    >>> config = create_default_model()
    >>> wavelength, flux = generate_spectral_model(config)
    >>> model_spectrum = np.vstack([wavelength, flux])
    >>> wavelength_centers = np.array([1.8, 1.9, 2.0, 2.1, 2.2])
    >>> bandwidths = np.full(5, 0.1)  # 100 nm bandwidth
    >>> band_data = simulate_spherex_observations(
    ...     model_spectrum,
    ...     wavelength_centers,
    ...     bandwidths,
    ...     band_id="D3",
    ...     filter_profile="boxcar",
    ...     random_seed=42
    ... )
    >>> print(f"Simulated {band_data.n_measurements} observations for {band_data.band}")
    """
    # Step 1: Simulate clean observations
    flux_clean_uJy, wavelength_grid = simulate_clean_observations_from_model(
        model_spectrum=model_spectrum,
        wavelength_centers=wavelength_centers,
        bandwidths=bandwidths,
        band_id=band_id,
        filter_profile=filter_profile,
    )

    # Step 2: Add noise and get both noisy flux and errors (vectorized, single calculation)
    flux_noisy_uJy, flux_error_uJy = add_spherex_noise(
        flux_clean_uJy=flux_clean_uJy,
        band_id=band_id,
        sigma_det=sigma_det,
        abs_gain=abs_gain,
        num_pixels=num_pixels,
        bg_surface_brightness=bg_surface_brightness,
        T_int=T_int,
        N_reads=N_reads,
        qr2_path=qr2_path,
        random_seed=random_seed,
    )

    # Step 3: Create BandData object
    band_data = BandData(
        band=band_id,
        flux=flux_noisy_uJy,
        flux_error=flux_error_uJy,
        wavelength_center=wavelength_centers,
        bandwidth=bandwidths,
    )

    return band_data


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("SPHEREx Observation Simulator - Example Usage\n")

    # Example 1: Load calibration data
    print("=== Example 1: Loading Calibration Data ===")
    try:
        abs_gain_D1 = load_abs_gain_for_band("D1")
        print(f"Absolute gain for D1: {abs_gain_D1:.6f} [MJy/sr] / [e-/s]")

        readnoise_D1 = load_readnoise_for_band("D1")
        print(f"Read noise for D1: {readnoise_D1:.6f} e-")
    except FileNotFoundError as e:
        print(f"Note: Calibration files not found: {e}")
        print("Using placeholder values for demonstration...")
        abs_gain_D1 = 0.115
        readnoise_D1 = 0.0022

    # Example 2: Unit conversion
    print("\n=== Example 2: Unit Conversion ===")
    wavelength_test = np.array([1.0, 2.0, 3.0])  # microns
    flux_model_test = np.array([1.0e-15, 1.0e-15, 1.0e-15])  # erg/s/cm^2/A
    flux_uJy_test = convert_model_flux_to_uJy(wavelength_test, flux_model_test)
    print(f"Wavelength: {wavelength_test} microns")
    print(f"Flux (model): {flux_model_test} erg/s/cm^2/A")
    print(f"Flux (uJy): {flux_uJy_test}")

    # Example 3: Full simulation pipeline
    print("\n=== Example 3: Full Simulation Pipeline ===")
    print("Loading model spectrum from benchmark/tools/models.py...")

    try:
        from models import create_default_model, generate_spectral_model

        # Generate a model spectrum
        config = create_default_model()
        wavelength_model, flux_model = generate_spectral_model(config)
        model_spectrum = np.vstack([wavelength_model, flux_model])

        print(f"Model spectrum: {wavelength_model.min():.2f} - {wavelength_model.max():.2f} microns")
        print(f"Model flux range: {flux_model.min():.2e} - {flux_model.max():.2e} erg/s/cm^2/A")

        # Define observation parameters (simulating band D3)
        wavelength_centers = np.linspace(1.63, 2.41, 50)  # 50 observations across Band 3
        bandwidths = np.full(50, 0.05)  # 50 nm bandwidth each

        # Simulate observations
        print("\nSimulating SPHEREx observations...")
        band_data = simulate_spherex_observations(
            model_spectrum=model_spectrum,
            wavelength_centers=wavelength_centers,
            bandwidths=bandwidths,
            band_id="D3",
            filter_profile="boxcar",
            random_seed=42,
        )

        print("\nResults:")
        print(f"  Band: {band_data.band}")
        print(f"  Number of observations: {band_data.n_measurements}")
        print(f"  Wavelength range: {band_data.wavelength_range[0]:.3f} - {band_data.wavelength_range[1]:.3f} microns")
        print(f"  Flux range: {band_data.flux.min():.2f} - {band_data.flux.max():.2f} uJy")
        print(f"  Error range: {band_data.flux_error.min():.4f} - {band_data.flux_error.max():.4f} uJy")

        # Calculate SNR for cleaner output
        snr = band_data.flux / band_data.flux_error
        print(f"  SNR range: {snr.min():.1f} - {snr.max():.1f}")

        # Example 4: Visualization (if matplotlib available)
        print("\n=== Example 4: Visualization ===")
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex="all")

            # Plot 1: Model spectrum
            ax1.plot(wavelength_model, flux_model, "b-", linewidth=2, label="Model spectrum")
            ax1.set_xlabel(r"Wavelength ($\mu$m)", fontsize=11)
            ax1.set_ylabel(r"Flux ($erg~s^{-1}~cm^{-2}~\AA^{-1}$)", fontsize=11)
            ax1.set_title("Input Model Spectrum", fontsize=12, fontweight="bold")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Plot 2: Simulated observations
            ax2.errorbar(
                wavelength_centers,
                band_data.flux,
                yerr=band_data.flux_error,
                fmt="ro",
                capsize=3,
                label="Simulated observations",
            )
            ax2.set_xlabel(r"Wavelength ($\mu$m)", fontsize=11)
            ax2.set_ylabel(r"Flux ($\mu$Jy)", fontsize=11)
            ax2.set_title("Simulated SPHEREx Observations (D3)", fontsize=12, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            print("Displaying plots...")
            plt.show()

        except ImportError:
            print("Matplotlib not available - skipping visualization")

    except ImportError as e:
        print(f"Cannot run full example: {e}")
        print("Make sure benchmark/tools/models.py is available")

    print("\n[OK] Examples completed!")
