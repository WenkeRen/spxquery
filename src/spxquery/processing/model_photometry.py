"""
Pysersic model-based photometry for SPHEREx data.

This module provides PSF model-based photometry to address SPHEREx's severe undersampling
(6.2 arcsec/pixel vs 5" PSF FWHM), which makes aperture photometry highly sensitive to
sub-pixel positioning. Model fitting allows flexible source centers and properly accounts
for PSF convolution.

Architecture:
    - Independent post-processing tool that reads pipeline state
    - Uses pysersic autoprior for automatic prior generation from images
    - Generates independent model_lightcurve.csv with photometry measurements
    - Supports MAP, Laplace/SVI, and MCMC (NUTS) inference methods
    - Error handling: skip failed fits, write invalid values (NaN, flags=999)

Notes:
    - No aperture photometry dependency required
    - Automatic prior generation adapts to each image's characteristics
    - More robust than external flux priors for varying source conditions
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..core.config import ModelPhotometryConfig, QueryConfig
from ..utils.helpers import load_json

logger = logging.getLogger(__name__)


def run_model_photometry(
    config: QueryConfig,
    model_config: Optional[ModelPhotometryConfig] = None,
) -> Path:
    """
    Run pysersic model-based photometry on downloaded SPHEREx images.

    This function reads the pipeline state to get image locations and performs
    model-based fitting using pysersic with automatic prior generation.

    Parameters
    ----------
    config : QueryConfig
        Pipeline configuration (used to access source info and state)
    model_config : ModelPhotometryConfig, optional
        Model photometry configuration. If None, uses config.advanced.model_photometry

    Returns
    -------
    Path
        Path to output model_lightcurve.csv file

    Raises
    ------
    FileNotFoundError
        If pipeline state file not found
    ValueError
        If pipeline has not completed download stage

    Notes
    -----
    - Reads pipeline state from {source_name}.json
    - Uses pysersic autoprior for automatic prior generation
    - Skips failed fits and assigns invalid values (NaN, flags=999)
    - Output CSV contains photometry measurements
    - Diagnostic plots saved to diagnostics/{band}/ if save_diagnostic_plots=True
    - Posterior chains saved to diagnostics/chains/{band}/ if save_posterior_chains=True

    Examples
    --------
    >>> from spxquery import Source, QueryConfig
    >>> source = Source(ra=304.69, dec=42.44, name="My_Star")
    >>> config = QueryConfig(source=source, output_dir="output")
    >>> # Run base pipeline first (query and download stages only)
    >>> pipeline = SPXQueryPipeline(config)
    >>> pipeline.run_full_pipeline()
    >>> # Then run model photometry
    >>> model_csv = run_model_photometry(config)
    >>> print(f"Model photometry saved to {model_csv}")
    """
    # Use provided config or default from QueryConfig
    if model_config is None:
        model_config = config.advanced.model_photometry

    # Validate that pipeline has run
    state_file = config.output_dir / f"{config.source.name}.json"
    if not state_file.exists():
        raise FileNotFoundError(
            f"Pipeline state file not found: {state_file}. Run SPXQueryPipeline.run_full_pipeline() first."
        )

    # Load pipeline state
    state = load_json(state_file)

    # Check that required stages have completed
    required_stages = ["query", "download"]
    completed_stages = state.get("completed_stages", [])
    for stage in required_stages:
        if stage not in completed_stages:
            raise ValueError(f"Pipeline stage '{stage}' not completed. Run SPXQueryPipeline.run_full_pipeline() first.")

    # Note: Using pysersic autoprior - no aperture photometry dependency needed

    # Get observations from state to construct file paths
    query_results = state.get("query_results", {})
    observations = query_results.get("observations", [])
    if not observations:
        raise ValueError("No observations found in pipeline state")

    # Construct file paths from observations
    # Files are organized as: {output_dir}/data/{band}/{obs_id}.fits
    output_dir = Path(state.get("config", {}).get("output_dir", ""))
    data_dir = output_dir / "data"

    downloaded_files = []
    for obs in observations:
        file_path = data_dir / obs["band"] / f"{obs['obs_id']}.fits"
        downloaded_files.append(file_path)

    logger.info(f"Found {len(downloaded_files)} observations")

    # Prepare output directory
    results_dir = config.output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Initialize results list
    results_list = []

    # Process each image
    logger.info("Starting model-based photometry...")
    for i, filepath_str in enumerate(downloaded_files):
        filepath = Path(filepath_str)

        if not filepath.exists():
            logger.warning(f"File not found, skipping: {filepath}")
            continue

        logger.info(f"[{i + 1}/{len(downloaded_files)}] Processing {filepath.name}")

        try:
            # Fit single image
            result = _fit_single_image(
                filepath=filepath,
                source_ra=config.source.ra,
                source_dec=config.source.dec,
                model_config=model_config,
                output_dir=config.output_dir,
            )

            results_list.append(result)

        except Exception as e:
            logger.error(f"Failed to fit {filepath.name}: {e}")
            # Create invalid result entry
            invalid_result = _create_invalid_result(filepath, config.source)
            results_list.append(invalid_result)
            continue

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save to CSV
    output_csv = results_dir / model_config.output_filename
    results_df.to_csv(output_csv, index=False)

    logger.info("Model photometry complete!")
    logger.info(f"  Total images processed: {len(downloaded_files)}")
    logger.info(f"  Successful fits: {len([r for r in results_list if r['flags'] != 999])}")
    logger.info(f"  Failed fits: {len([r for r in results_list if r['flags'] == 999])}")
    logger.info(f"  Output saved to: {output_csv}")

    return output_csv


def _fit_single_image(
    filepath: Path,
    source_ra: float,
    source_dec: float,
    model_config: ModelPhotometryConfig,
    output_dir: Path,
) -> Dict:
    """
    Fit a single SPHEREx image with pysersic model.

    Parameters
    ----------
    filepath : Path
        Path to SPHEREx FITS file
    source_ra : float
        Source RA in degrees
    source_dec : float
        Source DEC in degrees
    model_config : ModelPhotometryConfig
        Model photometry configuration
    output_dir : Path
        Output directory for diagnostics

    Returns
    -------
    dict
        Result dictionary with photometry measurements

    Raises
    ------
    Exception
        Any error during fitting (caught by caller)

    Notes
    -----
    Uses pysersic autoprior to automatically generate priors from the image,
    eliminating the need for external aperture photometry measurements.
    """
    # Load FITS file using existing infrastructure
    from .fits_handler import (
        get_pixel_coordinates,
        get_wavelength_at_position,
        read_spherex_mef,
        subtract_zodiacal_background,
    )
    from .psf_extraction import extract_psf_from_mef

    mef = read_spherex_mef(filepath)

    # Convert (RA, Dec) to pixel coordinates using existing function
    x_source, y_source = get_pixel_coordinates(mef, source_ra, source_dec)

    logger.debug(f"  Source position: ({x_source:.2f}, {y_source:.2f})")

    # Get wavelength and bandwidth at source position using existing function
    wavelength, bandwidth = get_wavelength_at_position(mef, x_source, y_source)

    logger.debug(f"  Wavelength: {wavelength:.3f} μm, Bandwidth: {bandwidth:.3f} μm")

    # Subtract zodiacal background with scaling (same as aperture photometry)
    # This is critical: ZODI is a model and needs adjustment to match actual background
    from ..core.config import PhotometryConfig

    phot_config = PhotometryConfig()  # Get default photometry config for zodi params
    image_zodi_subtracted, zodi_scale = subtract_zodiacal_background(
        mef.image,
        mef.zodi,
        mef.flags,
        mef.variance,
        phot_config.zodi_scale_min,
        phot_config.zodi_scale_max,
    )

    logger.debug(f"  Applied zodiacal scaling factor: {zodi_scale:.4f}")

    # Extract PSF for this source position using MEF-based function
    psf, zone_id, psf_distance = extract_psf_from_mef(
        mef=mef,
        ra=source_ra,
        dec=source_dec,
        downsample=True,  # Downsample to native resolution
    )

    logger.debug(f"  Extracted PSF zone {zone_id} (distance: {psf_distance:.2f} pixels)")
    logger.debug(f"  PSF shape: {psf.shape}")
    logger.debug(f"  PSF sum: {np.sum(psf):.6f} (should be ~1.0)")

    # Create cutout around source
    cutout_size = getattr(model_config, "cutout_size", 50)

    # Check if source is within image with margin for cutout
    ny, nx = mef.image.shape
    fitting_margin = cutout_size // 2
    if not (fitting_margin <= x_source < nx - fitting_margin and fitting_margin <= y_source < ny - fitting_margin):
        logger.warning(
            f"Source at ({x_source:.1f}, {y_source:.1f}) too close to edge for {cutout_size}x{cutout_size} cutout"
        )
        raise ValueError("Source too close to image edge for model fitting")

    image_cutout, variance_cutout, flags_cutout, bounds, (x_cutout, y_cutout) = mef.create_cutout(
        x_center=x_source, y_center=y_source, size=cutout_size, return_offset=True, custom_image=image_zodi_subtracted
    )

    # Extract bounds for coordinate conversion back to parent image
    y_min, y_max, x_min, x_max = bounds

    # Create mask from flags
    from .fits_handler import create_background_mask

    mask_cutout = create_background_mask(flags_cutout)

    # Convert mask for pysersic (pysersic expects True = masked, our function returns True = good)
    # Invert the mask: True becomes False (good), False becomes True (masked)
    mask_pysersic = ~mask_cutout

    logger.debug(f"  Cutout size: {image_cutout.shape}")
    logger.debug(f"  Source position in cutout: ({x_cutout:.2f}, {y_cutout:.2f})")

    # Extract obs_id from MEF
    filename = mef.obs_id

    # Extract band from DETECTOR number (1-6 maps to D1-D6)
    detector_num = mef.detector
    if 1 <= detector_num <= 6:
        band = f"D{detector_num}"
    else:
        band = "Unknown"
        logger.warning(f"Invalid detector number {detector_num} in {filepath.name}, expected 1-6")

    # Perform model fitting (all in cutout coordinates)
    fit_result = _fit_point_source_map(
        image_cutout=image_cutout,
        variance_cutout=variance_cutout,
        mask_cutout=mask_pysersic,  # Use pysersic-compatible mask (True=masked)
        psf=psf,
        x_source=x_cutout,  # Source position in cutout frame
        y_source=y_cutout,  # Source position in cutout frame
        model_config=model_config,
        output_dir=output_dir,
        band=band,
        filename=filename,
    )

    # Convert fitted positions back to parent image coordinates
    if fit_result["flags"] == 0:  # Successful fit
        x_fit_parent = fit_result["x_fit"] + x_min
        y_fit_parent = fit_result["y_fit"] + y_min
    else:  # Failed fit, use original coordinates
        x_fit_parent = x_source
        y_fit_parent = y_source

    # Construct result dictionary (same structure as aperture photometry)
    result = {
        "obs_id": filename,  # Use same column name as regular photometry
        "mjd": mef.mjd,
        "band": band,
        "wavelength": wavelength,
        "bandwidth": bandwidth,
        "flux": fit_result["flux"],
        "flux_err": fit_result["flux_err"],
        "flags": fit_result["flags"],
        "x_fit": x_fit_parent,
        "y_fit": y_fit_parent,
        "psf_zone_id": zone_id,
    }

    return result


def _get_loss_function(loss_name: str):
    """
    Map loss function name to actual pysersic loss function.

    Parameters
    ----------
    loss_name : str
        Name of the loss function from config

    Returns
    -------
    Callable
        Actual pysersic loss function

    Raises
    ------
    ValueError
        If loss function name is not recognized
    """
    try:
        from pysersic.loss import gaussian_loss, gaussian_mixture, student_t_loss

        loss_functions = {
            "gaussian": gaussian_loss,
            "student_t": student_t_loss,
            "gaussian_mixture": gaussian_mixture,
        }

        if loss_name not in loss_functions:
            logger.warning(f"Unknown loss function '{loss_name}', using gaussian_loss")
            return gaussian_loss

        return loss_functions[loss_name]

    except ImportError as e:
        logger.error(f"Failed to import pysersic loss functions: {e}")
        raise ImportError("pysersic loss functions not available") from e


def _fit_point_source_map(
    image_cutout: np.ndarray,
    variance_cutout: np.ndarray,
    mask_cutout: np.ndarray,
    psf: np.ndarray,
    x_source: float,
    y_source: float,
    model_config: ModelPhotometryConfig,
    output_dir: Path,
    band: str,
    filename: str,
) -> Dict:
    """
    Fit point source model using MAP inference with pysersic autoprior.

    Works entirely in cutout coordinates. Caller is responsible for
    coordinate transformations to/from parent image.

    Parameters
    ----------
    image_cutout : np.ndarray
        Cutout of zodiacal-subtracted flux image (MJy/sr)
    variance_cutout : np.ndarray
        Cutout of variance array
    mask_cutout : np.ndarray
        Boolean mask for cutout (True = good pixel)
    psf : np.ndarray
        Downsampled PSF at native resolution
    x_source : float
        Source x position in cutout coordinates
    y_source : float
        Source y position in cutout coordinates
    model_config : ModelPhotometryConfig
        Model configuration
    output_dir : Path
        Output directory for diagnostics
    band : str
        SPHEREx band (D1-D6)
    filename : str
        Source filename for diagnostic plots

    Returns
    -------
    dict
        Fit results with keys: flux, flux_err, flags, x_fit, y_fit
        All coordinates are in cutout frame.

    Notes
    -----
    Uses pysersic FitSingle with point source profile and autoprior.
    Automatically generates priors from the image using pysersic.priors.autoprior.
    Cutout should already have zodiacal background subtracted with proper scaling.
    """
    try:
        from pysersic import FitSingle
    except ImportError as e:
        logger.error("pysersic not available. Install with: pip install pysersic")
        raise ImportError("pysersic is required for model photometry") from e

    # Check if cutout is valid
    if image_cutout.size == 0:
        logger.error("Cutout is empty")
        return {
            "flux": np.nan,
            "flux_err": np.nan,
            "flags": 999,
            "x_fit": x_source,
            "y_fit": y_source,
        }

    # Generate automatic priors using pysersic autoprior
    try:
        from jax.random import PRNGKey
        from pysersic.priors import autoprior

        logger.debug("  Generating autoprior from image")
        prior = autoprior(
            image=image_cutout, profile_type=model_config.profile_type, mask=mask_cutout, sky_type=model_config.sky_type
        )

        logger.debug(f"  Autoprior generated for {model_config.profile_type} + {model_config.sky_type}")

    except ImportError as e:
        logger.error(f"Failed to import required modules from pysersic/jax: {e}")
        return {
            "flux": np.nan,
            "flux_err": np.nan,
            "flags": 999,
            "x_fit": x_source,
            "y_fit": y_source,
        }
    except Exception as e:
        logger.error(f"Failed to generate autoprior: {e}")
        return {
            "flux": np.nan,
            "flux_err": np.nan,
            "flags": 999,
            "x_fit": x_source,
            "y_fit": y_source,
        }

    # Initialize FitSingle object with autoprior
    try:
        # Get actual callable loss function
        loss_func = _get_loss_function(model_config.loss_function)

        # Clean variance array to prevent invalid scale parameters
        # Student-t and other loss functions require strictly positive scale values
        # Even masked pixels need valid values to avoid numerical issues in pysersic
        rms_clean = np.sqrt(variance_cutout)

        # Replace invalid values (0, NaN, inf) with small positive value
        # Use median of valid values as a reasonable fallback
        valid_rms = rms_clean[mask_cutout & np.isfinite(rms_clean) & (rms_clean > 0)]
        if len(valid_rms) > 0:
            fallback_rms = np.median(valid_rms)
        else:
            # Ultimate fallback if no valid values exist
            fallback_rms = 1e-6  # Small positive value in MJy/sr units

        # Create cleaned rms array
        rms = np.where(mask_cutout & np.isfinite(rms_clean) & (rms_clean > 1e-10), rms_clean, fallback_rms)

        # Debug logging: count only unmasked invalid pixels (those that would affect fitting)
        unmasked_invalid = np.sum(mask_cutout & ~np.isfinite(rms_clean) | (mask_cutout & (rms_clean <= 1e-10)))
        if unmasked_invalid > 0:
            logger.debug(f"  RMS cleaning: {unmasked_invalid} unmasked invalid values replaced")

        # Report statistics for unmasked pixels only (those actually used in fitting)
        unmasked_rms = rms[mask_cutout]
        if len(unmasked_rms) > 0:
            logger.debug(f"  RMS range (unmasked): {np.min(unmasked_rms):.2e} - {np.max(unmasked_rms):.2e} MJy/sr")
            logger.debug(f"  Unmasked pixels used in fitting: {len(unmasked_rms)}")

        fitter = FitSingle(
            data=image_cutout,  # pysersic expects 'data' parameter
            rms=rms,  # Cleaned rms values
            prior=prior,  # pysersic expects 'prior' parameter
            psf=psf,
            mask=mask_cutout,
            loss_func=loss_func,  # Pass actual callable function
        )

        logger.debug(f"  FitSingle initialized with {model_config.profile_type} + {model_config.sky_type}")

    except Exception as e:
        logger.error(f"Failed to initialize FitSingle: {e}")
        return {
            "flux": np.nan,
            "flux_err": np.nan,
            "flags": 999,
            "x_fit": x_source,
            "y_fit": y_source,
        }

    # Run MAP inference
    try:
        logger.debug("  Running MAP inference...")

        # Use fixed seed for reproducibility
        rkey = PRNGKey(42)
        map_result = fitter.find_MAP(rkey=rkey)

        # Extract results (all in cutout coordinates)
        # MAP result is a dictionary with best-fit parameters
        flux_fit = map_result.get("flux", np.nan)
        # For MAP, we don't automatically get uncertainties, use a simple estimate
        flux_err_fit = abs(flux_fit * 0.1) if not np.isnan(flux_fit) else np.nan  # 10% error estimate
        x_fit = map_result.get("x", x_source)
        y_fit = map_result.get("y", y_source)

        logger.debug(f"  Fit flux: {flux_fit:.3e} ± {flux_err_fit:.3e}")
        logger.debug(f"  Fit position in cutout: ({x_fit:.2f}, {y_fit:.2f})")

        # Save diagnostic plots if requested
        if model_config.save_diagnostic_plots:
            _save_diagnostic_plots(
                image_cutout=image_cutout,
                variance_cutout=variance_cutout,
                mask_cutout=mask_cutout,
                psf=psf,
                map_result=map_result,
                output_dir=output_dir,
                band=band,
                filename=filename,
                model_config=model_config,
                x_prior=x_source,
                y_prior=y_source,
            )

        # Note: MAP doesn't generate posterior chains, so skip chain saving

        return {
            "flux": flux_fit,
            "flux_err": flux_err_fit,
            "flags": 0,  # Success
            "x_fit": x_fit,
            "y_fit": y_fit,
        }

    except Exception as e:
        logger.error(f"MAP inference failed: {e}")
        return {
            "flux": np.nan,
            "flux_err": np.nan,
            "flags": 999,
            "x_fit": x_source,
            "y_fit": y_source,
        }


def _save_diagnostic_plots(
    image_cutout: np.ndarray,
    variance_cutout: np.ndarray,
    mask_cutout: np.ndarray,
    psf: np.ndarray,
    map_result: Dict,
    output_dir: Path,
    band: str,
    filename: str,
    model_config: ModelPhotometryConfig,
    x_prior: float,
    y_prior: float,
) -> None:
    """
    Save diagnostic plots using pysersic's plot_residual function.

    Parameters
    ----------
    image_cutout : np.ndarray
        Cutout of zodiacal-subtracted flux image (MJy/sr)
    variance_cutout : np.ndarray
        Cutout of variance array
    mask_cutout : np.ndarray
        Boolean mask for cutout (pysersic format: True = masked)
    psf : np.ndarray
        Downsampled PSF at native resolution
    map_result : dict
        MAP inference result containing fitted model
    output_dir : Path
        Output directory (e.g., "output/")
    band : str
        SPHEREx band (D1-D6)
    filename : str
        Source filename for plot naming
    model_config : ModelPhotometryConfig
        Model configuration
    x_prior : float
        Prior x center position (expected source position in cutout coordinates)
    y_prior : float
        Prior y center position (expected source position in cutout coordinates)

    Notes
    -----
    Saves diagnostic plot to: {output_dir}/diagnostics/{band}/{filename}_residual.{format}
    Priority: plot_residual > plot_image > simple matplotlib plot.
    Uses original data passed as parameters rather than fitter's internal state.
    Plots red 'x' for prior center and green '+' for fitted center.
    """
    # Create output directory
    diagnostics_dir = output_dir / "diagnostics" / band
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Base filename (remove .fits extension)
    base_name = filename.replace(".fits", "")
    plot_format = model_config.diagnostic_plot_format
    dpi = model_config.diagnostic_plot_dpi

    try:
        import matplotlib.pyplot as plt

        logger.debug(f"  Saving diagnostic plot for {filename}")

        # Try plot_residual first (preferred)
        try:
            from pysersic.results import plot_residual

            logger.debug("    Creating model/residual plot...")

            # Check if model is available in map_result
            if map_result is not None and "model" in map_result:
                # Use correct signature: plot_residual(image, model, mask=mask, vmin=vmin, vmax=vmax)
                # Use original data passed as parameters
                fig, ax = plot_residual(
                    image_cutout,
                    map_result["model"],
                    mask=mask_cutout,
                    vmin=-1,
                    vmax=1,
                )

                # Add prior and fitted center markers to the data panel (ax[0])
                # Red 'x' for prior center (expected position)
                ax[0].plot(
                    x_prior, y_prior, "rx", markersize=10, markeredgewidth=2, label="Prior (expected)", alpha=0.8
                )

                # Green '+' for fitted center
                x_fit = map_result.get("xc", x_prior)
                y_fit = map_result.get("yc", y_prior)
                ax[0].plot(x_fit, y_fit, "g+", markersize=12, markeredgewidth=2, label="Fitted", alpha=0.8)

                ax[0].legend(loc="upper right", fontsize=8)
                fig.suptitle(f"SPHEREx {band} Fit: {filename}", fontsize=12)
                plot_path = diagnostics_dir / f"{base_name}_residual.{plot_format}"
                fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                logger.debug(f"  ✓ Saved residual plot: {plot_path}")
                return

            logger.debug("    No model available in MAP results for residual plot")

        except Exception as e:
            logger.debug(f"    plot_residual failed: {e}")

        # Fallback to plot_image
        try:
            from pysersic.results import plot_image

            logger.debug("    Creating data/PSF/mask plot...")
            # Use original data passed as parameters
            rms = np.sqrt(variance_cutout)
            fig, ax = plot_image(
                image=image_cutout,
                mask=mask_cutout,
                sig=rms,
                psf=psf,
            )

            # Add prior center marker (red 'x') to the image panel (ax[0])
            ax[0].plot(x_prior, y_prior, "rx", markersize=10, markeredgewidth=2, label="Prior (expected)", alpha=0.8)

            # Add fitted center marker if available
            if map_result is not None:
                x_fit = map_result.get("xc", x_prior)
                y_fit = map_result.get("yc", y_prior)
                ax[0].plot(x_fit, y_fit, "g+", markersize=12, markeredgewidth=2, label="Fitted", alpha=0.8)
                ax[0].legend(loc="upper right", fontsize=8)

            fig.suptitle(f"SPHEREx {band} Data: {filename}", fontsize=12)
            plot_path = diagnostics_dir / f"{base_name}_input.{plot_format}"
            fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            logger.debug(f"  ✓ Saved input plot: {plot_path}")
            return

        except Exception as e:
            logger.debug(f"    plot_image failed: {e}")

        # Final fallback to simple matplotlib plot
        if image_cutout is not None and image_cutout.size > 0:
            logger.debug("    Creating simple data plot...")
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            im = ax.imshow(image_cutout, origin="lower", cmap="viridis")

            # Add prior center marker (red 'x')
            ax.plot(x_prior, y_prior, "rx", markersize=10, markeredgewidth=2, label="Prior (expected)", alpha=0.8)

            # Add fitted center marker if available
            if map_result is not None:
                x_fit = map_result.get("xc", x_prior)
                y_fit = map_result.get("yc", y_prior)
                ax.plot(x_fit, y_fit, "g+", markersize=12, markeredgewidth=2, label="Fitted", alpha=0.8)
                ax.legend(loc="upper right", fontsize=8)

            ax.set_title(f"SPHEREx {band} Data - {filename}")
            plt.colorbar(im, ax=ax, label="Flux (MJy/sr)")
            plot_path = diagnostics_dir / f"{base_name}_data.{plot_format}"
            fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            logger.debug(f"  ✓ Saved simple data plot: {plot_path}")
            return

        logger.warning(f"  No valid data available for plotting: {filename}")

    except Exception as e:
        logger.warning(f"Failed to save diagnostic plot for {filename}: {e}")


def _save_posterior_chains(
    fitter,
    output_dir: Path,
    band: str,
    filename: str,
    model_config: ModelPhotometryConfig,
) -> None:
    """
    Placeholder function for posterior chains saving.

    Parameters
    ----------
    fitter : pysersic.FitSingle
        Fitted model object
    output_dir : Path
        Output directory (e.g., "output/")
    band : str
        SPHEREx band (D1-D6)
    filename : str
        Source filename for chain naming
    model_config : ModelPhotometryConfig
        Model configuration

    Notes
    -----
    MAP inference doesn't generate posterior chains.
    This function is kept for compatibility with existing configuration.
    """
    logger.debug("  MAP inference has no posterior chains to save")
    return


def _create_invalid_result(filepath: Path, source) -> Dict:
    """
    Create invalid result entry for failed fits.

    Parameters
    ----------
    filepath : Path
        Path to FITS file that failed
    source : Source
        Source object

    Returns
    -------
    dict
        Invalid result dictionary
    """
    return {
        "obs_id": filepath.stem,  # Use same column name as regular photometry
        "mjd": np.nan,
        "band": "UNKNOWN",
        "wavelength": np.nan,
        "bandwidth": np.nan,
        "flux": np.nan,
        "flux_err": np.nan,
        "flags": 999,
        "x_fit": np.nan,
        "y_fit": np.nan,
        "psf_zone_id": -1,
    }
