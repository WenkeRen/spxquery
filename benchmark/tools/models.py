"""
Spectral Model Generator for SPHEREx Benchmark Tests

This module provides tools to generate synthetic spectra consisting of:
- A power-law continuum: F_lambda ~ lambda^slope
- Two Gaussian emission lines with configurable parameters

The model is designed for benchmark testing of SPHEREx spectral reconstruction
algorithms, with parameters matching the experiment design in the Benchmark plan.

Author: SPHEREx Spectral Reconstruction Team
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import astropy.units as u
import numpy as np
from astropy import constants as const

# ============================================================================
# SPHEREx Band Definitions
# ============================================================================

SPHEREX_BANDS: Dict[str, Dict[str, float]] = {
    "D1": {"wave_min": 0.75, "wave_max": 1.09, "R": 41},
    "D2": {"wave_min": 1.10, "wave_max": 1.62, "R": 41},
    "D3": {"wave_min": 1.63, "wave_max": 2.41, "R": 41},
    "D4": {"wave_min": 2.42, "wave_max": 3.82, "R": 35},
    "D5": {"wave_min": 3.83, "wave_max": 4.41, "R": 110},
    "D6": {"wave_min": 4.42, "wave_max": 5.00, "R": 130},
}


# ============================================================================
# Configuration Dataclass
# ============================================================================


@dataclass
class SpectralModelConfig:
    """
    Configuration parameters for synthetic spectral model generation.

    The model consists of a power-law continuum (F_lambda ~ lambda^alpha) with two Gaussian
    emission lines. AB magnitude is defined in f_nu space (standard convention),
    but the spectrum is output in f_lambda (erg/s/cm^2/A) for line profile analysis.

    Attributes:
    -----------
    magnitude : float
        AB magnitude of the continuum at reference_wavelength.
        Typical range: 17-19 AB mag for benchmark tests.

    slope : float
        Spectral index alpha where F_lambda ~ lambda^alpha.
        Values: 0 (flat in f_lambda), -1, -2 for benchmark tests.
        Note: Negative slope means redder spectrum (more flux at longer lambda).

    reference_wavelength : float
        Wavelength in microns where AB magnitude is defined.
        Defaults to band center wavelength. Used for continuum normalization.

    line1_center : float
        Central wavelength of the primary emission line in microns.

    line1_ew : float
        Equivalent width of the primary line in nanometers.
        Typical benchmark values: 10, 20, 40, 80 nm.

    line1_fwhm_vel : float
        FWHM of the primary line in km/s (velocity width).
        Typical benchmark values: 500, 1000, 2000, 4000 km/s.

    line2_flux_ratio : float
        Flux ratio of Line 2 to Line 1.
        Set to 0 to disable Line 2. Typical values: 0, 1, 1.5, 3.

    line2_separation_fwhm : float
        Separation between lines in units of Line 1's FWHM.
        Positive value: Line 2 at longer wavelength.
        Typical values: 0, 1, 2, 4 FWHM.

    line2_same_fwhm : bool, optional
        If True, Line 2 has same FWHM as Line 1. If False, Line 2 FWHM
        will be scaled by flux_ratio (empirical approximation).

    wavelength_min : float, optional
        Minimum wavelength in microns for output spectrum.

    wavelength_max : float, optional
        Maximum wavelength in microns for output spectrum.

    spectral_resolution : float, optional
        Spectral resolution R = lambda/delta_lambda. Used to set pixel sampling.

    oversample_factor : float, optional
        Oversampling factor for wavelength grid. Values > 1 provide
        finer sampling than the nominal resolution. Default: 2.5.

    band_id : str, optional
        SPHEREx band ID to use as template for wavelength range and resolution.
        If specified, overrides wavelength_min/max and spectral_resolution.
        Default: "D3" (1.63-2.41 um, R=41).
    """

    # Continuum parameters
    magnitude: float
    slope: float
    reference_wavelength: Optional[float] = None

    # Line 1 parameters (primary)
    line1_center: float = 2.0  # Default: 2.0 microns (Band 4 region)
    line1_ew: float = 20.0  # Default: 20 nm
    line1_fwhm_vel: float = 1000.0  # Default: 1000 km/s

    # Line 2 parameters (secondary)
    line2_flux_ratio: float = 0.0  # Default: disabled (single line)
    line2_separation_fwhm: float = 1.0  # Default: 1 FWHM separation
    line2_same_fwhm: bool = True  # Default: same FWHM as Line 1

    # Wavelength grid parameters
    wavelength_min: Optional[float] = None
    wavelength_max: Optional[float] = None
    spectral_resolution: Optional[float] = None
    oversample_factor: float = 500

    # SPHEREx band preset
    band_id: str = "D3"  # Default: Band 3

    def __post_init__(self):
        """Validate and set default values from SPHEREx band definitions."""
        # Apply band preset if specified
        if self.band_id in SPHEREX_BANDS:
            band = SPHEREX_BANDS[self.band_id]
            if self.wavelength_min is None:
                self.wavelength_min = band["wave_min"]
            if self.wavelength_max is None:
                self.wavelength_max = band["wave_max"]
            if self.spectral_resolution is None:
                self.spectral_resolution = band["R"]

        # Set reference wavelength to band center if not specified
        if self.reference_wavelength is None:
            if self.wavelength_min is not None and self.wavelength_max is not None:
                self.reference_wavelength = (self.wavelength_min + self.wavelength_max) / 2.0
            else:
                self.reference_wavelength = 2.0  # Default fallback

        # Validate wavelength range
        if self.wavelength_min is None or self.wavelength_max is None:
            raise ValueError("Must specify wavelength_min/max or a valid band_id")

        if self.wavelength_min >= self.wavelength_max:
            raise ValueError(f"wavelength_min ({self.wavelength_min}) must be < wavelength_max ({self.wavelength_max})")

        # Validate spectral resolution
        if self.spectral_resolution is None or self.spectral_resolution <= 0:
            raise ValueError(f"spectral_resolution must be > 0, got {self.spectral_resolution}")

        # Type assertions: these should never be None after __post_init__
        assert self.wavelength_min is not None
        assert self.wavelength_max is not None
        assert self.spectral_resolution is not None
        assert self.reference_wavelength is not None


# ============================================================================
# Conversion Functions
# ============================================================================


def ab_mag_to_flux(magnitude: float) -> float:
    """
    Convert AB magnitude to flux density in f_nu space.

    AB magnitude system: m_AB = -2.5 * log10(f_nu) - 48.60
    where f_nu is in erg/s/cm^2/Hz.

    Parameters:
    -----------
    magnitude : float
        AB magnitude value.

    Returns:
    --------
    f_nu : float
        Flux density in erg/s/cm^2/Hz.
    """
    # f_nu in erg/s/cm^2/Hz
    f_nu = 10.0 ** (-0.4 * (magnitude + 48.60))
    return f_nu


def fnu_to_flambda_at_wavelength(f_nu: float, wavelength_microns: float, slope: float = 0.0) -> float:
    """
    Convert f_nu to f_lambda at a specific wavelength with power-law slope.

    The relationship is: f_lambda = f_nu * c / lambda^2
    For a power-law F_lambda ~ lambda^alpha, we calculate the normalization at reference wavelength.

    Parameters:
    -----------
    f_nu : float
        Flux density in erg/s/cm^2/Hz at reference wavelength.
    wavelength_microns : float
        Wavelength in microns at which to calculate f_lambda.
    slope : float, optional
        Spectral index alpha where F_lambda ~ lambda^alpha. Default: 0.

    Returns:
    --------
    f_lambda : float
        Flux density in erg/s/cm^2/A at the specified wavelength.
    """
    # Convert wavelength from microns to Angstroms
    wavelength_angstrom = wavelength_microns * 1e4

    # Speed of light in Angstroms/s
    c_angstrom_s = const.c.to(u.Angstrom / u.s).value

    # Base conversion: f_lambda = f_nu * c / lambda^2
    f_lambda_base = f_nu * c_angstrom_s / (wavelength_angstrom**2)

    # Apply power-law slope: F_lambda ~ lambda^alpha
    # This is already implicit in the conversion, so we just return the base value
    # The slope will be applied when generating the continuum array
    return f_lambda_base


def velocity_fwhm_to_wavelength_fwhm(wavelength_center: float, fwhm_vel: float) -> float:
    """
    Convert FWHM from velocity space to wavelength space.

    For non-relativistic velocities: delta_lambda/lambda = delta_v/c
    For relativistic velocities, the relativistic Doppler formula should be used,
    but for typical emission line widths (up to 10000 km/s), the non-relativistic
    approximation is sufficient (< 0.2% error at 10000 km/s).

    Parameters:
    -----------
    wavelength_center : float
        Central wavelength of the line in microns.
    fwhm_vel : float
        FWHM in km/s.

    Returns:
    --------
    fwhm_wavelength : float
        FWHM in microns.
    """
    # Speed of light in km/s
    c_km_s = const.c.to(u.km / u.s).value

    # Non-relativistic approximation
    delta_lambda_over_lambda = fwhm_vel / c_km_s
    fwhm_wavelength = wavelength_center * delta_lambda_over_lambda

    return fwhm_wavelength


# ============================================================================
# Spectral Component Functions
# ============================================================================


def generate_wavelength_grid(
    wavelength_min: float,
    wavelength_max: float,
    spectral_resolution: float,
    oversample_factor: float = 2.5,
) -> np.ndarray:
    """
    Generate a wavelength grid with uniform sampling in lambda.

    The pixel size is set by spectral_resolution and oversample_factor:
    delta_lambda = lambda / (R * oversample_factor)
    where R = lambda/delta_lambda is the spectral resolution.

    Parameters:
    -----------
    wavelength_min : float
        Minimum wavelength in microns.
    wavelength_max : float
        Maximum wavelength in microns.
    spectral_resolution : float
        Spectral resolution R = lambda/delta_lambda.
    oversample_factor : float, optional
        Oversampling factor. Default: 2.5.

    Returns:
    --------
    wavelength : np.ndarray
        Wavelength array in microns.
    """
    # Use central wavelength to determine pixel size
    wavelength_center = (wavelength_min + wavelength_max) / 2.0

    # Calculate pixel size: delta_lambda = lambda / (R * oversample)
    delta_lambda = wavelength_center / (spectral_resolution * oversample_factor)

    # Generate wavelength grid
    n_pixels = int((wavelength_max - wavelength_min) / delta_lambda)
    wavelength = np.linspace(wavelength_min, wavelength_max, n_pixels)

    return wavelength


def powerlaw_continuum(
    wavelength: np.ndarray,
    reference_wavelength: float,
    f_lambda_ref: float,
    slope: float,
) -> np.ndarray:
    """
    Generate a power-law continuum F_lambda ~ lambda^alpha.

    Parameters:
    -----------
    wavelength : np.ndarray
        Wavelength array in microns.
    reference_wavelength : float
        Reference wavelength in microns where f_lambda_ref is defined.
    f_lambda_ref : float
        Flux density at reference wavelength in erg/s/cm^2/A.
    slope : float
        Spectral index alpha where F_lambda ~ lambda^alpha.

    Returns:
    --------
    continuum : np.ndarray
        Continuum flux array in erg/s/cm^2/A.
    """
    # Power law: F_lambda = F_lambda,ref * (lambda / lambda_ref)^alpha
    continuum = f_lambda_ref * (wavelength / reference_wavelength) ** slope
    return continuum


def gaussian_emission_line(
    wavelength: np.ndarray,
    center: float,
    fwhm_wavelength: float,
    amplitude: float,
) -> np.ndarray:
    """
    Generate a Gaussian emission line profile.

    The Gaussian is defined as:
    F(lambda) = A * exp(-0.5 * ((lambda - lambda_0) / sigma)^2)
    where sigma = FWHM / (2 * sqrt(2 * ln(2)))

    Parameters:
    -----------
    wavelength : np.ndarray
        Wavelength array in microns.
    center : float
        Central wavelength of the line in microns.
    fwhm_wavelength : float
        FWHM of the line in microns.
    amplitude : float
        Peak amplitude of the line in erg/s/cm^2/A.

    Returns:
    --------
    line_flux : np.ndarray
        Line flux contribution array in erg/s/cm^2/A.
    """
    # Convert FWHM to standard deviation
    sigma = fwhm_wavelength / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Gaussian profile
    line_flux = amplitude * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)

    return line_flux


def calculate_line_amplitude_from_ew(
    continuum_level: float,
    equivalent_width: float,
    fwhm_wavelength: float,
) -> float:
    """
    Calculate the required amplitude of a Gaussian to achieve a specific equivalent width.

    For a Gaussian profile, the equivalent width is:
    EW = A * sqrt(2*pi) * sigma / C
    where A is amplitude, sigma is standard deviation, and C is continuum level.
    Therefore: A = C * EW / (sqrt(2*pi) * sigma)

    Parameters:
    -----------
    continuum_level : float
        Continuum flux level at the line center in erg/s/cm^2/A.
    equivalent_width : float
        Desired equivalent width in Angstroms.
    fwhm_wavelength : float
        FWHM of the line in Angstroms.

    Returns:
    --------
    amplitude : float
        Required peak amplitude in erg/s/cm^2/A.
    """
    # Convert FWHM to sigma
    sigma = fwhm_wavelength / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Calculate amplitude: A = C * EW / (sqrt(2*pi) * sigma)
    amplitude = continuum_level * equivalent_width / (np.sqrt(2.0 * np.pi) * sigma)

    return amplitude


# ============================================================================
# Main Model Generation Function
# ============================================================================


def generate_spectral_model(config: SpectralModelConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic spectrum with power-law continuum and emission lines.

    The model consists of:
    1. Power-law continuum: F_lambda ~ lambda^alpha
    2. Primary Gaussian emission line (Line 1)
    3. Secondary Gaussian emission line (Line 2, optional)

    Parameters:
    -----------
    config : SpectralModelConfig
        Configuration object with all model parameters.

    Returns:
    --------
    wavelength : np.ndarray
        Wavelength array in microns.
    flux : np.ndarray
        Flux array in erg/s/cm^2/A.

    Example:
    --------
    >>> # Create a simple model with single emission line
    >>> config = SpectralModelConfig(
    ...     magnitude=18.0,
    ...     slope=-1.0,
    ...     line1_center=2.0,
    ...     line1_ew=20.0,
    ...     line1_fwhm_vel=1000.0,
    ...     line2_flux_ratio=0.0,  # Single line
    ...     band_id="D3"
    ... )
    >>> wavelength, flux = generate_spectral_model(config)
    >>> print(f"Wavelength range: {wavelength.min():.2f} - {wavelength.max():.2f} microns")
    >>> print(f"Flux range: {flux.min():.2e} - {flux.max():.2e} erg/s/cm^2/A")
    """
    # 1. Generate wavelength grid
    wavelength = generate_wavelength_grid(
        wavelength_min=config.wavelength_min,
        wavelength_max=config.wavelength_max,
        spectral_resolution=config.spectral_resolution,
        oversample_factor=config.oversample_factor,
    )

    # 2. Generate continuum
    # Convert AB magnitude to f_nu
    f_nu_ref = ab_mag_to_flux(config.magnitude)

    # Convert f_nu to f_lambda at reference wavelength
    f_lambda_ref = fnu_to_flambda_at_wavelength(
        f_nu=f_nu_ref,
        wavelength_microns=config.reference_wavelength,
        slope=config.slope,
    )

    # Generate power-law continuum
    continuum = powerlaw_continuum(
        wavelength=wavelength,
        reference_wavelength=config.reference_wavelength,
        f_lambda_ref=f_lambda_ref,
        slope=config.slope,
    )

    # Initialize flux with continuum
    flux = continuum.copy()

    # 3. Add Line 1 (primary emission line)
    # Convert FWHM from velocity to wavelength
    line1_fwhm_wave = velocity_fwhm_to_wavelength_fwhm(
        wavelength_center=config.line1_center,
        fwhm_vel=config.line1_fwhm_vel,
    )

    # Get continuum level at Line 1 center
    continuum_at_line1 = np.interp(config.line1_center, wavelength, continuum)

    # Calculate amplitude to achieve desired EW
    # Note: config.line1_ew is in nm, convert to Angstroms
    line1_amplitude = calculate_line_amplitude_from_ew(
        continuum_level=continuum_at_line1,
        equivalent_width=config.line1_ew * 10.0,  # nm to Angstroms
        fwhm_wavelength=line1_fwhm_wave * 1e4,  # microns to Angstroms
    )

    # Add Line 1
    line1_contribution = gaussian_emission_line(
        wavelength=wavelength,
        center=config.line1_center,
        fwhm_wavelength=line1_fwhm_wave,
        amplitude=line1_amplitude,
    )
    flux += line1_contribution

    # 4. Add Line 2 (secondary emission line) if enabled
    if config.line2_flux_ratio > 0:
        # Calculate Line 2 center (separated by N * FWHM of Line 1)
        line2_center = config.line1_center + config.line2_separation_fwhm * line1_fwhm_wave

        # Determine Line 2 FWHM
        if config.line2_same_fwhm:
            line2_fwhm_wave = line1_fwhm_wave
        else:
            # Empirical approximation: scale FWHM with flux_ratio
            # This is a simplified model - in reality, line width depends on
            # physical conditions, not just flux
            line2_fwhm_wave = line1_fwhm_wave * np.sqrt(config.line2_flux_ratio)

        # Line 2 amplitude is scaled by flux_ratio
        # Note: This ensures the integrated flux scales correctly
        line2_amplitude = line1_amplitude * config.line2_flux_ratio

        # Add Line 2
        line2_contribution = gaussian_emission_line(
            wavelength=wavelength,
            center=line2_center,
            fwhm_wavelength=line2_fwhm_wave,
            amplitude=line2_amplitude,
        )
        flux += line2_contribution

    return wavelength, flux


# ============================================================================
# Convenience Functions for Benchmark Scenarios
# ============================================================================


def create_default_model() -> SpectralModelConfig:
    """
    Create a default model configuration for basic testing.

    Returns:
    --------
    config : SpectralModelConfig
        Default configuration with moderate parameters.
    """
    config = SpectralModelConfig(
        magnitude=18.0,
        slope=0.0,
        line1_center=2.0,
        line1_ew=20.0,
        line1_fwhm_vel=3000.0,
        line2_flux_ratio=0.0,
        band_id="D3",
    )
    return config


def create_double_line_model(
    flux_ratio: float = 1.5,
    separation_fwhm: float = 2.0,
) -> SpectralModelConfig:
    """
    Create a model with two emission lines for resolution testing.

    Parameters:
    -----------
    flux_ratio : float, optional
        Flux ratio of Line 2 to Line 1. Default: 1.5.
    separation_fwhm : float, optional
        Line separation in units of Line 1 FWHM. Default: 2.0.

    Returns:
    --------
    config : SpectralModelConfig
        Configuration with two emission lines.
    """
    config = SpectralModelConfig(
        magnitude=18.0,
        slope=0.0,
        line1_center=2.0,
        line1_ew=20.0,
        line1_fwhm_vel=1000.0,
        line2_flux_ratio=flux_ratio,
        line2_separation_fwhm=separation_fwhm,
        line2_same_fwhm=True,
        band_id="D3",
    )
    return config


def get_spectrum_as_2row_array(wavelength: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """
    Format spectrum as a 2-row array for compatibility with SPHEREx data.

    Parameters:
    -----------
    wavelength : np.ndarray
        Wavelength array in microns.
    flux : np.ndarray
        Flux array in erg/s/cm^2/A.

    Returns:
    --------
    spectrum : np.ndarray
        2D array with shape (2, n_pixels), where row 0 is wavelength and row 1 is flux.
    """
    return np.vstack([wavelength, flux])


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing spectral model generation...")

    # Try to import matplotlib for visualization
    try:
        import matplotlib.pyplot as plt

        HAS_MATPLOTLIB = True
        print("Matplotlib available - will generate visualizations")
    except ImportError:
        HAS_MATPLOTLIB = False
        print("Matplotlib not available - skipping visualizations")

    # Create figure for visualization
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("SPHEREx Spectral Model Generator - Test Results", fontsize=14, fontweight="bold")

    # Test 1: Single line model
    print("\n=== Test 1: Single Emission Line ===")
    config1 = create_default_model()
    print(
        f"Configuration: magnitude={config1.magnitude}, "
        f"slope={config1.slope}, "
        f"line1_ew={config1.line1_ew} nm, "
        f"line1_fwhm_vel={config1.line1_fwhm_vel} km/s"
    )

    wave1, flux1 = generate_spectral_model(config1)
    print(f"Generated spectrum: {len(wave1)} wavelength points")
    print(f"Wavelength range: {wave1.min():.3f} - {wave1.max():.3f} microns")
    print(f"Flux range: {flux1.min():.2e} - {flux1.max():.2e} erg/s/cm^2/A")

    if HAS_MATPLOTLIB:
        ax = axes[0, 0]
        ax.plot(wave1, flux1, "b-", linewidth=2, label="Single emission line")
        ax.set_xlabel("Wavelength ($\\mu$m)", fontsize=11)
        ax.set_ylabel("Flux (erg s$^{-1}$ cm$^{-2}$ $\\AA$$^{-1}$)", fontsize=11)
        ax.set_title("Test 1: Single Emission Line", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Test 2: Double line model
    print("\n=== Test 2: Double Emission Line ===")
    config2 = create_double_line_model(flux_ratio=1.5, separation_fwhm=2.0)
    print(f"Line 2: flux_ratio={config2.line2_flux_ratio}, separation={config2.line2_separation_fwhm} FWHM")

    wave2, flux2 = generate_spectral_model(config2)
    print(f"Generated spectrum: {len(wave2)} wavelength points")

    if HAS_MATPLOTLIB:
        ax = axes[0, 1]
        ax.plot(wave2, flux2, "r-", linewidth=2, label="Double emission line")
        ax.set_xlabel("Wavelength ($\\mu$m)", fontsize=11)
        ax.set_ylabel("Flux (erg s$^{-1}$ cm$^{-2}$ $\\AA$$^{-1}$)", fontsize=11)
        ax.set_title("Test 2: Double Line (Resolution Test)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Test 3: Different slopes
    print("\n=== Test 3: Different Continuum Slopes ===")
    if HAS_MATPLOTLIB:
        ax = axes[1, 0]

    for slope in [0, -1, -2]:
        config3 = SpectralModelConfig(
            magnitude=18.0,
            slope=slope,
            line1_center=2.0,
            line1_ew=20.0,
            line1_fwhm_vel=1000.0,
            line2_flux_ratio=0.0,
            band_id="D3",
        )
        wave3, flux3 = generate_spectral_model(config3)
        print(
            f"Slope {slope}: flux at 1.7um = {np.interp(1.7, wave3, flux3):.2e}, "
            f"flux at 2.3um = {np.interp(2.3, wave3, flux3):.2e}"
        )

        if HAS_MATPLOTLIB:
            ax.plot(wave3, flux3, linewidth=2, label=f"Slope $\\alpha$ = {slope}")

    if HAS_MATPLOTLIB:
        ax.set_xlabel("Wavelength ($\\mu$m)", fontsize=11)
        ax.set_ylabel("Flux (erg s$^{-1}$ cm$^{-2}$ $\\AA$$^{-1}$)", fontsize=11)
        ax.set_title("Test 3: Power-law Continuum Slopes", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Test 4: Different bands
    print("\n=== Test 4: Different SPHEREx Bands ===")
    if HAS_MATPLOTLIB:
        ax = axes[1, 1]

    for band_id in ["D1", "D3", "D6"]:
        config4 = SpectralModelConfig(
            magnitude=18.0,
            slope=0.0,
            line1_center=2.0,
            line1_ew=20.0,
            line1_fwhm_vel=1000.0,
            line2_flux_ratio=0.0,
            band_id=band_id,
        )
        wave4, flux4 = generate_spectral_model(config4)
        print(
            f"{band_id}: {wave4.min():.2f}-{wave4.max():.2f} um, {len(wave4)} pixels, R={config4.spectral_resolution}"
        )

        if HAS_MATPLOTLIB:
            # Normalize flux for better visualization
            flux_norm = flux4 / np.median(flux4)
            ax.plot(wave4, flux_norm, linewidth=2, label=f"{band_id} (R={config4.spectral_resolution})")

    if HAS_MATPLOTLIB:
        ax.set_xlabel("Wavelength ($\\mu$m)", fontsize=11)
        ax.set_ylabel("Normalized Flux", fontsize=11)
        ax.set_title("Test 4: SPHEREx Bands", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        print("\n[INFO] Displaying plot window...")
        plt.show()

    print("\n[OK] All tests passed!")
