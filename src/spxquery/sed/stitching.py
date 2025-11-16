"""
Multi-band stitching and normalization for SED reconstruction.

This module combines reconstructed spectra from 6 SPHEREx detector bands
into a continuous spectrum spanning 0.75-5.0 microns, accounting for
flux calibration differences between detectors.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from .config import DETECTOR_WAVELENGTH_RANGES

logger = logging.getLogger(__name__)


@dataclass
class StitchedSpectrum:
    """
    Container for stitched multi-band spectrum.

    Attributes
    ----------
    wavelength : np.ndarray
        Continuous wavelength grid in microns, shape (N_total,).
    flux : np.ndarray
        Normalized and stitched flux density in microJansky, shape (N_total,).
    band_labels : np.ndarray
        Band identifier for each wavelength point, shape (N_total,).
        Values are strings 'D1', 'D2', ..., 'D6'.
    normalization_factors : Dict[str, float]
        Normalization factor (eta) for each band.
        D1 is reference with eta=1.0.
    wavelength_ranges : Dict[str, Tuple[float, float]]
        (min, max) wavelength in microns for each band.
    overlap_statistics : List[Dict]
        Statistics for each band overlap used in normalization.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    band_labels: np.ndarray
    normalization_factors: Dict[str, float]
    wavelength_ranges: Dict[str, Tuple[float, float]]
    overlap_statistics: List[Dict]


def find_overlap_region(
    wavelength1: np.ndarray,
    flux1: np.ndarray,
    wavelength2: np.ndarray,
    flux2: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find overlapping wavelength region between two spectra.

    Parameters
    ----------
    wavelength1 : np.ndarray
        Wavelength grid for first spectrum, shape (N1,).
    flux1 : np.ndarray
        Flux for first spectrum, shape (N1,).
    wavelength2 : np.ndarray
        Wavelength grid for second spectrum, shape (N2,).
    flux2 : np.ndarray
        Flux for second spectrum, shape (N2,).

    Returns
    -------
    flux1_overlap : np.ndarray or None
        Flux values from spectrum 1 in overlap region.
    flux2_interp : np.ndarray or None
        Flux values from spectrum 2 interpolated to wavelength1 grid.
        Returns None if no overlap exists.
    """
    # Find overlap range
    lambda_min = max(wavelength1.min(), wavelength2.min())
    lambda_max = min(wavelength1.max(), wavelength2.max())

    if lambda_min >= lambda_max:
        # No overlap
        return None, None

    # Extract flux1 in overlap region
    overlap_mask = (wavelength1 >= lambda_min) & (wavelength1 <= lambda_max)
    wavelength1_overlap = wavelength1[overlap_mask]
    flux1_overlap = flux1[overlap_mask]

    if len(flux1_overlap) == 0:
        return None, None

    # Interpolate flux2 to wavelength1 grid
    flux2_interp = np.interp(wavelength1_overlap, wavelength2, flux2)

    return flux1_overlap, flux2_interp


def compute_normalization_factor(
    flux_reference: np.ndarray,
    flux_target: np.ndarray,
    method: str = "median",
) -> float:
    """
    Compute normalization factor to scale target spectrum to reference.

    The normalization factor eta is computed such that:
        flux_target_normalized = eta * flux_target

    matches flux_reference in the overlap region.

    Parameters
    ----------
    flux_reference : np.ndarray
        Flux from reference spectrum in overlap region.
    flux_target : np.ndarray
        Flux from target spectrum in overlap region (same wavelength grid).
    method : str
        Method to compute normalization: 'median' (robust) or 'mean'.

    Returns
    -------
    float
        Normalization factor eta.
    """
    # Compute flux ratio
    # Avoid division by zero by excluding near-zero target flux
    valid = (flux_target > 0) & np.isfinite(flux_reference) & np.isfinite(flux_target)

    if not np.any(valid):
        logger.warning("No valid flux ratios in overlap region. Using eta=1.0")
        return 1.0

    ratios = flux_reference[valid] / flux_target[valid]

    # Compute normalization factor
    if method == "median":
        eta = np.median(ratios)
    elif method == "mean":
        eta = np.mean(ratios)
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")

    return eta


def stitch_band_pair(
    wavelength1: np.ndarray,
    flux1: np.ndarray,
    band1: str,
    wavelength2: np.ndarray,
    flux2: np.ndarray,
    band2: str,
    eta2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stitch two adjacent bands with normalization.

    Parameters
    ----------
    wavelength1 : np.ndarray
        Wavelength grid for band 1 (reference), shape (N1,).
    flux1 : np.ndarray
        Flux for band 1 (already normalized), shape (N1,).
    band1 : str
        Band 1 identifier (e.g., 'D1').
    wavelength2 : np.ndarray
        Wavelength grid for band 2, shape (N2,).
    flux2 : np.ndarray
        Flux for band 2 (to be normalized), shape (N2,).
    band2 : str
        Band 2 identifier (e.g., 'D2').
    eta2 : float
        Normalization factor for band 2.

    Returns
    -------
    wavelength_stitched : np.ndarray
        Combined wavelength grid, sorted.
    flux_stitched : np.ndarray
        Combined flux (band 2 normalized by eta2).
    band_labels : np.ndarray
        Band labels for each wavelength point.
    """
    # Normalize band 2
    flux2_normalized = eta2 * flux2

    # Concatenate wavelengths and fluxes
    wavelength_stitched = np.concatenate([wavelength1, wavelength2])
    flux_stitched = np.concatenate([flux1, flux2_normalized])
    band_labels = np.concatenate([
        np.full(len(wavelength1), band1, dtype=object),
        np.full(len(wavelength2), band2, dtype=object),
    ])

    # Sort by wavelength
    sort_idx = np.argsort(wavelength_stitched)
    wavelength_stitched = wavelength_stitched[sort_idx]
    flux_stitched = flux_stitched[sort_idx]
    band_labels = band_labels[sort_idx]

    return wavelength_stitched, flux_stitched, band_labels


def stitch_all_bands(
    band_spectra: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> StitchedSpectrum:
    """
    Stitch all reconstructed band spectra into a continuous spectrum.

    The stitching procedure:
    1. Use D1 as reference (eta_D1 = 1.0)
    2. For each subsequent band in order (D2, D3, ..., D6):
       a. Find overlap with previously stitched spectrum
       b. Compute normalization factor (eta) from flux ratio in overlap
       c. Normalize and append band to stitched spectrum

    Parameters
    ----------
    band_spectra : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary mapping band name to (wavelength, flux) arrays.
        Keys should be in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6'].

    Returns
    -------
    StitchedSpectrum
        Container with stitched spectrum and normalization metadata.

    Raises
    ------
    ValueError
        If no bands are provided or D1 is missing.
    """
    if not band_spectra:
        raise ValueError("No band spectra provided for stitching")

    # Define band order
    band_order = ["D1", "D2", "D3", "D4", "D5", "D6"]
    available_bands = [b for b in band_order if b in band_spectra]

    if not available_bands:
        raise ValueError("No valid bands found in band_spectra")

    logger.info(f"Stitching {len(available_bands)} bands: {available_bands}")

    # Use first available band as reference
    reference_band = available_bands[0]
    wavelength_stitched, flux_stitched = band_spectra[reference_band]
    band_labels = np.full(len(wavelength_stitched), reference_band, dtype=object)

    normalization_factors = {reference_band: 1.0}
    # Use hardcoded detector ranges instead of data-derived ranges
    wavelength_ranges = {
        reference_band: DETECTOR_WAVELENGTH_RANGES[reference_band]
    }
    overlap_statistics = []

    logger.info(f"Reference band: {reference_band} (eta=1.0)")

    # Iteratively stitch remaining bands
    for band in available_bands[1:]:
        wavelength_new, flux_new = band_spectra[band]

        # Find overlap with current stitched spectrum
        flux_ref_overlap, flux_new_overlap = find_overlap_region(
            wavelength_stitched, flux_stitched, wavelength_new, flux_new
        )

        if flux_ref_overlap is None or len(flux_ref_overlap) < 5:
            logger.warning(
                f"Insufficient overlap between stitched spectrum and {band} "
                f"(<5 points). Using eta=1.0 for {band}."
            )
            eta = 1.0
            overlap_info = {
                "band": band,
                "n_overlap_points": 0,
                "eta": eta,
                "overlap_wavelength_range": (np.nan, np.nan),
            }
        else:
            # Compute normalization factor
            eta = compute_normalization_factor(flux_ref_overlap, flux_new_overlap)

            # Record overlap statistics
            wavelength_overlap_min = wavelength_new.min()
            wavelength_overlap_max = wavelength_new.max()
            overlap_info = {
                "band": band,
                "n_overlap_points": len(flux_ref_overlap),
                "eta": eta,
                "overlap_wavelength_range": (
                    float(wavelength_overlap_min),
                    float(wavelength_overlap_max),
                ),
                "flux_ratio_median": eta,
                "flux_ratio_std": float(np.std(flux_ref_overlap / flux_new_overlap)),
            }

            logger.info(
                f"{band}: eta={eta:.4f} from {len(flux_ref_overlap)} overlap points"
            )

        overlap_statistics.append(overlap_info)
        normalization_factors[band] = eta
        # Use hardcoded detector range for this band
        wavelength_ranges[band] = DETECTOR_WAVELENGTH_RANGES[band]

        # Stitch this band to the growing spectrum
        # For simplicity, we'll just concatenate and sort
        # (overlap regions will have duplicate wavelength entries with potentially different fluxes)
        flux_new_normalized = eta * flux_new
        wavelength_stitched = np.concatenate([wavelength_stitched, wavelength_new])
        flux_stitched = np.concatenate([flux_stitched, flux_new_normalized])
        band_labels = np.concatenate([
            band_labels,
            np.full(len(wavelength_new), band, dtype=object),
        ])

        # Sort by wavelength
        sort_idx = np.argsort(wavelength_stitched)
        wavelength_stitched = wavelength_stitched[sort_idx]
        flux_stitched = flux_stitched[sort_idx]
        band_labels = band_labels[sort_idx]

    # Create result container
    result = StitchedSpectrum(
        wavelength=wavelength_stitched,
        flux=flux_stitched,
        band_labels=band_labels,
        normalization_factors=normalization_factors,
        wavelength_ranges=wavelength_ranges,
        overlap_statistics=overlap_statistics,
    )

    logger.info(
        f"Stitching complete: {len(wavelength_stitched)} wavelength points "
        f"spanning {wavelength_stitched.min():.3f}-{wavelength_stitched.max():.3f} um"
    )

    return result
