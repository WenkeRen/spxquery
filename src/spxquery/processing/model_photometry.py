"""
Model-based photometry using AstroPhot.

This module implements model-based photometry (point source, Sersic, etc.) using
AstroPhot to address SPHEREx undersampling (6.2"/pixel vs 5" FWHM PSF).
Uses 10x oversampled PSF images for point source models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import astrophot as ap
import numpy as np
from astropy import units as u
from tqdm import tqdm

from ..core.config import ModelPhotometryConfig, PhotometryConfig, PhotometryResult, Source
from .fits_handler import (
    create_background_mask,
    create_psf_wcs,
    get_pixel_scale,
    get_wavelength_at_position,
    read_spherex_mef,
    validate_psf_pixel_scale,
)
from .magnitudes import calculate_ab_magnitude_from_jy

logger = logging.getLogger(__name__)


def setup_astrophot_psf(psf_array: np.ndarray, psf_pixel_scale: float) -> ap.image.PSF_Image:
    """
    Create AstroPhot PSF_Image from SPHEREx PSF array.

    Parameters
    ----------
    psf_array : np.ndarray
        Normalized PSF array, shape (101, 101).
    psf_pixel_scale : float
        PSF pixel scale in arcsec/pixel (absolute units, not relative).
        Should be image_pixel_scale / oversample_factor.

    Returns
    -------
    psf_image : ap.image.PSF_Image
        AstroPhot PSF_Image object.

    Notes
    -----
    The PSF array should be normalized (sum=1.0) before passing to this function.
    AstroPhot requires PSF pixelscale to be a multiple of image pixelscale.
    Pixelscale must be in absolute units (arcsec/pixel), not relative scale.
    """
    psf_image = ap.image.PSF_Image(
        data=psf_array.astype(np.float64),
        pixelscale=psf_pixel_scale,  # Absolute arcsec/pixel
    )

    logger.debug(
        f"Created AstroPhot PSF: shape={psf_array.shape}, "
        f"pixelscale={psf_pixel_scale:.4f} arcsec/pix, sum={np.sum(psf_array):.6e}"
    )

    return psf_image


def setup_astrophot_target(
    image_data: np.ndarray,
    variance_data: np.ndarray,
    # wcs: WCS,
    pixelscale: float,
    psf: ap.image.PSF_Image,
    mask: np.ndarray,
    zeropoint: float,
) -> ap.image.Target_Image:
    """
    Create AstroPhot Target_Image from SPHEREx data with WCS, PSF, and mask.

    Parameters
    ----------
    image_data : np.ndarray
        Image data in MJy/sr.
    variance_data : np.ndarray
        Variance data in (MJy/sr)^2.
    wcs : WCS
        Astropy WCS object for spatial coordinates.
    psf : ap.image.PSF_Image
        AstroPhot PSF_Image object.
    mask : np.ndarray
        Boolean mask (True = pixel to ignore).
    zeropoint : float
        AB magnitude zeropoint.

    Returns
    -------
    target : ap.image.Target_Image
        AstroPhot target image object.

    Notes
    -----
    AstroPhot uses variance for uncertainty weighting during fitting.
    Providing WCS object allows AstroPhot to handle coordinates automatically.
    PSF must be provided during Target_Image creation for proper modeling.
    Mask excludes problematic pixels from fitting.
    """
    target = ap.image.Target_Image(
        data=image_data.astype(np.float64),
        # wcs=wcs,
        pixelscale=pixelscale,
        variance=variance_data.astype(np.float64),
        psf=psf,
        mask=mask,
        zeropoint=zeropoint,
    )

    logger.debug(
        f"Created AstroPhot target: shape={image_data.shape}, "
        f"zeropoint={zeropoint}, with WCS, PSF, and mask ({np.sum(mask)} masked pixels)"
    )

    return target


def fit_source_model(
    target: ap.image.Target_Image,
    psf: ap.image.PSF_Image,
    initial_center: Tuple[float, float],
    initial_flux_log10: Optional[float],
    config: ModelPhotometryConfig,
) -> Dict:
    """
    Fit point source + flat background model using AstroPhot.

    Parameters
    ----------
    target : ap.image.Target_Image
        Target image to fit (in flux/arcsec^2 units).
    psf : ap.image.PSF_Image
        PSF model.
    initial_center : tuple of float
        Initial (x, y) pixel coordinates for source center.
    initial_flux_log10 : float or None
        Initial flux guess in log10(flux). If None, AstroPhot will estimate.
    config : ModelPhotometryConfig
        Model photometry configuration.

    Returns
    -------
    result : dict
        Fitting results with keys:
        - 'flux': Fitted flux in log10 units
        - 'flux_error': Flux uncertainty in log10 units
        - 'center_x': Fitted x coordinate
        - 'center_y': Fitted y coordinate
        - 'background': Fitted background level (flux/arcsec^2)
        - 'chi2': Reduced chi-squared
        - 'converged': bool, whether fit converged
        - 'message': Convergence message

    Notes
    -----
    AstroPhot flux parameter units: log10(flux)
    Target image units: flux/arcsec^2 (surface brightness)

    Fits a model combining:
    - Point source (or other model type) with log10(flux) parameter
    - Flat background (sky model) with constant surface brightness

    AstroPhot fitting methods:
    - "LM": Levenberg-Marquardt (recommended for PSF fitting)
    - "Nested": Nested sampling (slower but more robust)
    - "HMC": Hamiltonian Monte Carlo (for posteriors)
    """
    # Create source model based on model_type
    source_params = {
        "center": list(initial_center),  # [x, y] in pixels
    }

    if initial_flux_log10 is not None:
        # Flux parameter is log10(flux) - directly use the input
        source_params["flux"] = initial_flux_log10

    # Map config model_type to AstroPhot model_type
    model_type_mapping = {
        "point_source": "point model",
        "sersic": "sersic galaxy model",
        "exponential": "exponential galaxy model",
        "gaussian": "gaussian model",
    }

    astrophot_model_type = model_type_mapping.get(config.model_type, "point model")

    # Create source model
    source_model = ap.models.AstroPhot_Model(
        name="source",
        model_type=astrophot_model_type,
        target=target,
        psf=psf,
        psf_mode="full",
        parameters=source_params,
    )

    # Create flat background model
    background_model = ap.models.AstroPhot_Model(
        name="background",
        model_type="flat sky model",
        target=target,
    )

    # Create group model combining source + background
    model = ap.models.AstroPhot_Model(
        name="astrophot model",
        model_type="group model",
        target=target,
        models=[source_model, background_model],
    )

    # Initialize model
    model.initialize()

    logger.debug(
        f"Initialized {config.model_type} + background model at ({initial_center[0]:.2f}, {initial_center[1]:.2f})"
    )

    # Select fitting method
    if config.fitting_method == "LM":
        fitter = ap.fit.LM(
            model,
            verbose=0,
            # max_iter=config.max_iterations,
            # tol=config.convergence_tolerance,
        )
    elif config.fitting_method == "Nested":
        fitter = ap.fit.Nested(
            model,
            verbose=0,
            # max_iter=config.max_iterations,
        )
    elif config.fitting_method == "HMC":
        fitter = ap.fit.HMC(
            model,
            verbose=0,
            # max_iter=config.max_iterations,
        )
    else:
        raise ValueError(f"Unknown fitting method: {config.fitting_method}")

    # Perform fit
    try:
        fit_result = fitter.fit()
        converged = fit_result.success if hasattr(fit_result, "success") else True
        message = fit_result.message if hasattr(fit_result, "message") else "Fit completed"
    except Exception as e:
        logger.error(f"Model fitting failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "flux_log10": 0.0,
            "flux_error_log10": np.inf,
            "center_x": initial_center[0],
            "center_y": initial_center[1],
            "background": 0.0,
            "chi2": np.inf,
            "converged": False,
            "message": f"Fitting error: {str(e)}",
        }

    # Extract fitted parameters
    try:
        # Extract center from source model
        fitted_center = source_model["center"].value.detach().cpu().numpy()

        # Extract flux (in log10 units) from source model
        fitted_flux_log10 = source_model["flux"].value.detach().cpu().numpy()

        # Get flux uncertainty if available (in log10 units)
        if hasattr(source_model["flux"], "uncertainty"):
            flux_error_log10 = source_model["flux"].uncertainty.detach().cpu().numpy()
        else:
            # Estimate: 10% uncertainty in linear flux ≈ 0.04 in log10
            flux_error_log10 = 0.04

        # Extract background level from background model
        fitted_background = background_model["I0"].value.detach().cpu().numpy()

        # Calculate chi-squared
        residuals = (model.target.data - model().data) / np.sqrt(model.target.variance)
        chi2 = np.sum(residuals**2) / (residuals.size - len(model.parameters))

    except Exception as e:
        logger.error(f"Failed to extract fitted parameters: {e}")
        return {
            "flux_log10": 0.0,
            "flux_error_log10": np.inf,
            "center_x": initial_center[0],
            "center_y": initial_center[1],
            "background": 0.0,
            "chi2": np.inf,
            "converged": False,
            "message": f"Parameter extraction error: {str(e)}",
        }

    logger.debug(
        f"Fit completed: flux_log10={fitted_flux_log10:.3f}±{flux_error_log10:.3f}, "
        f"center=({fitted_center[0]:.2f}, {fitted_center[1]:.2f}), "
        f"background={fitted_background:.6e} MJy/sr, chi2={chi2:.3f}, converged={converged}"
    )

    return {
        "flux_log10": float(fitted_flux_log10),
        "flux_error_log10": float(flux_error_log10),
        "center_x": float(fitted_center[0]),
        "center_y": float(fitted_center[1]),
        "background": float(fitted_background),
        "chi2": float(chi2),
        "converged": bool(converged),
        "message": message,
    }


def convert_log10_flux_to_ujy(
    flux_log10: float,
    image_unit: str = "MJy/sr",
) -> float:
    """
    Convert log10(flux) from AstroPhot to microJansky.

    Parameters
    ----------
    flux_log10 : float
        Flux from AstroPhot fit in log10 scale.
        The flux is in image units (same as target image).
    image_unit : str
        Unit of target image. Default: "MJy/sr".

    Returns
    -------
    flux_ujy : float
        Flux in microJansky (μJy).

    Notes
    -----
    Conversion chain:
    1. Convert log10(flux) to linear: 10^flux_log10
    2. Flux is in image units (MJy/sr for SPHEREx)
    3. For total flux, this is already integrated (not per solid angle)
    4. Convert: MJy → Jy (×10^6) → μJy (×10^6)

    AstroPhot's flux parameter represents total integrated flux, not surface brightness.
    """
    # Convert log10 to linear
    flux_linear = 10**flux_log10

    # Convert MJy to Jy to μJy
    if image_unit == "MJy/sr":
        flux_jy = flux_linear * 1e6  # MJy → Jy
        flux_ujy = flux_jy * 1e6  # Jy → μJy
    else:
        raise ValueError(f"Unsupported image unit: {image_unit}")

    logger.debug(
        f"Flux conversion: log10(flux)={flux_log10:.3f} → linear={flux_linear:.6e} MJy/sr → {flux_ujy:.3f} μJy"
    )

    return flux_ujy


def process_single_observation_model(
    mef_file: Path,
    source: Source,
    config: ModelPhotometryConfig,
    photometry_config: PhotometryConfig,
    aperture_result: Optional[PhotometryResult] = None,
) -> Optional[PhotometryResult]:
    """
    Process single MEF file with model-based photometry.

    Parameters
    ----------
    mef_file : Path
        Path to SPHEREx MEF file.
    source : Source
        Source coordinates (RA, Dec).
    config : PSFModelingConfig
        PSF modeling configuration.
    photometry_config : PhotometryConfig
        Photometry configuration (for pixel scale fallback, etc.).
    aperture_result : PhotometryResult, optional
        Aperture photometry result for initial flux guess.

    Returns
    -------
    result : PhotometryResult or None
        PSF photometry result, or None if processing failed.

    Notes
    -----
    This function:
    1. Reads MEF file
    2. Extracts PSF for source position
    3. Optionally subtracts zodiacal background
    4. Sets up AstroPhot target and PSF
    5. Fits point source model
    6. Converts flux to μJy
    7. Returns PhotometryResult
    """
    try:
        # Read MEF file
        mef = read_spherex_mef(mef_file)

        # Get pixel coordinates
        x, y = mef.spatial_wcs.world_to_pixel(source.coord)

        # Get wavelength and bandwidth
        wavelength, bandwidth = get_wavelength_at_position(mef, x, y)

        logger.info(
            f"Processing {mef_file.name}: PSF photometry at pixel ({x:.2f}, {y:.2f}), wavelength={wavelength:.3f} μm"
        )

        # Extract PSF for source position using the MEF method
        psf_array, zone_id = mef.get_psf_at_position(x, y)

        # Get image pixel scale from WCS
        image_pixel_scale = get_pixel_scale(mef.spatial_wcs, photometry_config.pixel_scale_fallback)

        # Round to 2 decimals for consistency with AstroPhot
        image_pixel_scale_rounded = round(image_pixel_scale, 2)

        # Create PSF WCS and validate pixel scale
        psf_wcs = create_psf_wcs(mef.psf_header)
        psf_pixel_scale = validate_psf_pixel_scale(image_pixel_scale, psf_wcs, config.psf_oversample_factor)

        logger.debug(
            f"Pixel scales: image={image_pixel_scale:.4f} arcsec/pix, "
            f"PSF={psf_pixel_scale:.4f} arcsec/pix (1/{config.psf_oversample_factor}x)"
        )

        # Unit conversion factor: The image is in MJy/sr, convert to uJy/arcsec^2
        unit_conv_factor = (u.MJy / u.sr).to(u.uJy / u.arcsec**2)

        # Define zeropoint (in AB magnitudes with pixel value in unit of uJy/arcsec^2)
        zeropoint = 2.5 * np.log10(3631e6)

        # Create WCS-aware cutout around source (reduces memory and computation)
        # Use original image - AstroPhot will handle sky modeling during fitting
        image_cutout, variance_cutout, flags_cutout, cutout_wcs = mef.create_cutout(
            position=(x, y), size=config.fitting_box_size
        )

        # Apply unit conversion to cutout data
        image_cutout *= unit_conv_factor  # Now in uJy/arcsec^2
        variance_cutout *= unit_conv_factor**2  # Variance scales as square of conversion factor

        # Create mask from flags
        mask = np.logical_not(create_background_mask(flags_cutout))

        # Validate mask - ensure we have enough valid pixels
        valid_pixel_fraction = np.sum(~mask) / mask.size
        logger.debug(
            f"Mask statistics: {np.sum(mask)} masked pixels, {np.sum(~mask)} valid pixels ({valid_pixel_fraction:.1%})"
        )

        if valid_pixel_fraction < 0.1:
            logger.warning(
                f"Only {valid_pixel_fraction:.1%} of pixels are valid. "
                "Consider increasing fitting_box_size or relaxing quality flags."
            )

        # Source is now at the center of the cutout
        x_pix, y_pix = cutout_wcs.world_to_pixel(source.coord)
        cutout_center = (float(x_pix), float(y_pix))

        # Validate cutout size vs PSF size
        psf_size_pixels = psf_array.shape[0] / config.psf_oversample_factor
        min_required_size = psf_size_pixels * 2  # Need 2x PSF size for proper fitting

        if config.fitting_box_size < min_required_size:
            logger.warning(
                f"fitting_box_size={config.fitting_box_size} may be too small. "
                f"PSF is {psf_size_pixels:.1f} pixels (effective size), "
                f"recommend >= {min_required_size:.0f} pixels for robust fitting."
            )

        # Set up AstroPhot PSF first with absolute pixel scale
        psf = setup_astrophot_psf(psf_array, psf_pixel_scale)

        # Set up AstroPhot target with adjusted WCS, PSF, and mask
        target = setup_astrophot_target(image_cutout, variance_cutout, image_pixel_scale_rounded, psf, mask, zeropoint)

        # Get initial flux guess in log10 units
        if config.initial_flux_from_aperture and aperture_result is not None:
            flux_ujy = aperture_result.flux  # uJy
            initial_flux_log10 = np.log10(flux_ujy)
            logger.debug(f"Using aperture flux as initial guess: log10(flux)={initial_flux_log10:.3f}")
        else:
            initial_flux_log10 = None  # Let AstroPhot estimate

        # Fit source model (use cutout center as initial position)
        fit_result = fit_source_model(target, psf, cutout_center, initial_flux_log10, config)

        if not fit_result["converged"]:
            logger.warning(f"Model fitting did not converge for {mef_file.name}: {fit_result['message']}")

        # Convert flux from log10 units to μJy
        flux_ujy = convert_log10_flux_to_ujy(fit_result["flux_log10"])

        # Convert flux error from log10 units to linear μJy
        # Error propagation: σ(f) ≈ f × ln(10) × σ(log10(f))
        flux_linear = 10 ** fit_result["flux_log10"]
        flux_error_linear = flux_linear * np.log(10) * fit_result["flux_error_log10"]
        flux_error_ujy = flux_error_linear * 1e12  # MJy/sr → μJy

        # Calculate AB magnitude
        flux_jy = flux_ujy * 1e-6
        flux_error_jy = flux_error_ujy * 1e-6
        mag_ab, mag_ab_error = calculate_ab_magnitude_from_jy(flux_jy, flux_error_jy, wavelength)

        # Convert fitted center from cutout coordinates to full image coordinates
        # Use WCS transformation: cutout pixel -> sky -> full image pixel
        fitted_coord = cutout_wcs.pixel_to_world(fit_result["center_x"], fit_result["center_y"])
        fitted_x_full, fitted_y_full = mef.spatial_wcs.world_to_pixel(fitted_coord)

        logger.debug(
            f"Fitted position: cutout ({fit_result['center_x']:.2f}, {fit_result['center_y']:.2f}) -> "
            f"full image ({fitted_x_full:.2f}, {fitted_y_full:.2f})"
        )

        # Extract observation metadata
        obs_id = mef.header.get("OBSID", mef_file.stem)
        detector_num = mef.detector
        if 1 <= detector_num <= 6:
            band = f"D{detector_num}"
        else:
            band = "Unknown"

        # Create PhotometryResult with model-specific fields
        result = PhotometryResult(
            obs_id=obs_id,
            mjd=mef.mjd,
            flux=flux_ujy,
            flux_error=flux_error_ujy,
            wavelength=wavelength,
            bandwidth=bandwidth,
            flag=0,  # Model fitting doesn't use pixel flags directly
            pix_x=float(fitted_x_full),  # Fitted position in full image coordinates
            pix_y=float(fitted_y_full),
            band=band,
            mag_ab=mag_ab,
            mag_ab_error=mag_ab_error,
            photometry_method="model",
            model_type=config.model_type,
            model_chi2=fit_result["chi2"],
            psf_zone_id=zone_id if config.model_type == "point_source" else None,
        )

        logger.info(
            f"Model photometry ({config.model_type}) from {mef_file.name}: "
            f"flux={flux_ujy:.3f}±{flux_error_ujy:.3f} μJy, "
            f"mag_AB={mag_ab:.3f}±{mag_ab_error:.3f}, "
            f"background={fit_result['background']:.6e} MJy/sr, "
            f"chi2={fit_result['chi2']:.3f}" + (f", zone={zone_id}" if config.model_type == "point_source" else "")
        )

        return result

    except Exception as e:
        logger.error(f"Failed to process {mef_file} with PSF photometry: {e}")
        import traceback

        traceback.print_exc()
        return None


def process_all_observations_model(
    mef_files: List[Path],
    source: Source,
    config: ModelPhotometryConfig,
    photometry_config: PhotometryConfig,
    aperture_results: Optional[List[PhotometryResult]] = None,
) -> List[PhotometryResult]:
    """
    Process all MEF files with model-based photometry.

    Parameters
    ----------
    mef_files : list of Path
        Paths to SPHEREx MEF files.
    source : Source
        Source coordinates.
    config : ModelPhotometryConfig
        Model photometry configuration.
    photometry_config : PhotometryConfig
        Photometry configuration.
    aperture_results : list of PhotometryResult, optional
        Aperture photometry results for initial flux guesses.

    Returns
    -------
    results : list of PhotometryResult
        Model photometry results.

    Notes
    -----
    Uses sequential processing with progress bar.
    Progress is shown via tqdm progress bar.
    """
    logger.info(f"Starting model photometry ({config.model_type}) for {len(mef_files)} observations")

    # Create aperture results lookup if provided
    aperture_lookup = {}
    if aperture_results is not None:
        aperture_lookup = {r.obs_id: r for r in aperture_results}

    results = []

    # Process files with progress bar
    with tqdm(total=len(mef_files), desc=f"Model photometry ({config.model_type})") as pbar:
        for mef_file in mef_files:
            # Get corresponding aperture result if available
            obs_id = mef_file.stem
            aperture_result = aperture_lookup.get(obs_id)

            result = process_single_observation_model(mef_file, source, config, photometry_config, aperture_result)

            if result is not None:
                results.append(result)

            pbar.update(1)

    logger.info(f"Completed model photometry ({config.model_type}): {len(results)}/{len(mef_files)} successful")

    return results
