"""
Spectral axis (Z) grid construction for Drizzle3D.

Reads SPECTRAL_CHANNELS from the bundled aux FITS file and builds the
wavelength bin grid (native, oversampled, or user-defined custom).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits

logger = logging.getLogger(__name__)

# Physical constant: speed of light [m/s]
_C_SI = 299792458.0


@dataclass
class ZGrid:
    """Spectral axis grid descriptor for one detector.

    Attributes
    ----------
    edges : np.ndarray
        (n_z+1,) bin edges [μm].
    centers : np.ndarray
        (n_z,) bin centers [μm].
    widths : np.ndarray
        (n_z,) bin widths Δλ [μm].
    """

    edges: np.ndarray
    centers: np.ndarray
    widths: np.ndarray

    @property
    def n_z(self) -> int:
        return len(self.centers)

    def frequencies(self) -> np.ndarray:
        """Central frequencies [Hz] for each bin."""
        lam_m = self.centers * 1e-6
        return _C_SI / lam_m

    def delta_nu(self) -> np.ndarray:
        """Frequency bin widths [Hz] for each bin."""
        nu_max = _C_SI / (self.edges[:-1] * 1e-6)  # shorter λ → higher ν
        nu_min = _C_SI / (self.edges[1:] * 1e-6)
        return nu_max - nu_min

    def resolving_power(self) -> np.ndarray:
        """Effective resolving power λ/Δλ per bin."""
        return self.centers / self.widths


def _aux_fits_path() -> Path:
    """Return path to the bundled spectral channels FITS file."""
    return Path(__file__).resolve().parent.parent / "aux" / "spectral_channels_spx_cal-sch-v1-2026-106.fits"


def load_spectral_table() -> pd.DataFrame:
    """Load SPECTRAL_CHANNELS (HDU 2) from the bundled aux FITS file.

    Returns
    -------
    pd.DataFrame
        102 rows × (DETECTOR, SUBCHAN, WAVELENGTH, WL_MIN, WL_MAX, R, R_STD, BANDWIDTH).
    """
    path = _aux_fits_path()
    if not path.exists():
        raise FileNotFoundError(f"Spectral channels file not found: {path}")

    with fits.open(path) as hdul:
        table = hdul[2].data  # SPECTRAL_CHANNELS
        df = pd.DataFrame(
            {
                "DETECTOR": table["DETECTOR"].astype(int),
                "SUBCHAN": table["SUBCHAN"].astype(int),
                "WAVELENGTH": table["WAVELENGTH"].astype(np.float64),
                "WL_MIN": table["WL_MIN"].astype(np.float64),
                "WL_MAX": table["WL_MAX"].astype(np.float64),
                "R": table["R"].astype(np.float64),
                "R_STD": table["R_STD"].astype(np.float64),
                "BANDWIDTH": table["BANDWIDTH"].astype(np.float64),
            }
        )

    logger.debug(f"Loaded spectral table: {len(df)} rows, detectors D1–D6")
    return df


def detector_spectral_table(detector: int) -> pd.DataFrame:
    """Get the 17-row spectral table for a single detector.

    Parameters
    ----------
    detector : int
        Detector number 1–6.

    Returns
    -------
    pd.DataFrame
        17 rows sorted by WAVELENGTH.
    """
    full = load_spectral_table()
    det = full[full["DETECTOR"] == detector].sort_values("WAVELENGTH").reset_index(drop=True)
    if len(det) == 0:
        raise ValueError(f"No spectral channels for detector {detector}")
    return det


def build_z_grid_native(detector: int) -> ZGrid:
    """Build Z grid from native 17 subchannels of a detector.

    Uses WL_MIN / WL_MAX from SPECTRAL_CHANNELS as bin edges.
    Adjacent subchannels may overlap; edges are taken directly from the table.
    """
    det = detector_spectral_table(detector)
    edges = np.concatenate([det["WL_MIN"].values, [det["WL_MAX"].values[-1]]])
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    # Clamp zero-width bins (shouldn't happen but guard)
    widths = np.maximum(widths, 1e-6)
    return ZGrid(edges=edges, centers=centers, widths=widths)


def build_z_grid_oversampled(detector: int, z_oversample: float) -> ZGrid:
    """Build oversampled Z grid with constant effective resolving power.

    Algorithm
    ---------
    1. Compute mean R-bar across all 17 subchannels of the detector.
    2. R_eff = R-bar / z_oversample  (z_oversample=2 → double resolution).
    3. Generate edges using constant-R logarithmic spacing:
       λ_{k+1} = λ_k * (1 + 1/R_eff)
    4. Start from λ_min (first subchannel WL_MIN), stop at λ_max (last WL_MAX).
    """
    det = detector_spectral_table(detector)
    r_mean = det["R"].mean()
    r_eff = r_mean * z_oversample
    lam_min = det["WL_MIN"].values[0]
    lam_max = det["WL_MAX"].values[-1]

    edges = [lam_min]
    lam = lam_min
    while lam < lam_max:
        lam = lam * (1.0 + 1.0 / r_eff)
        edges.append(lam)
    edges = np.array(edges)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    return ZGrid(edges=edges, centers=centers, widths=widths)


def build_z_grid_custom(lambda_edges: np.ndarray) -> ZGrid:
    """Build Z grid from user-provided bin edges [μm].

    Parameters
    ----------
    lambda_edges : np.ndarray
        (n_z+1,) monotonically increasing bin edges in μm.
    """
    edges = np.asarray(lambda_edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("z_lambda_edges must be 1-D with ≥2 elements")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("z_lambda_edges must be monotonically increasing")

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    return ZGrid(edges=edges, centers=centers, widths=widths)


def build_z_grid(detector: int, z_oversample: float = 1.0, z_lambda_edges: Optional[np.ndarray] = None) -> ZGrid:
    """Dispatch to the appropriate Z-grid builder.

    Parameters
    ----------
    detector : int
        Detector number (1–6).
    z_oversample : float
        Spectral oversampling factor (default 1.0 = native).
    z_lambda_edges : np.ndarray, optional
        Custom bin edges [μm]; overrides z_oversample if provided.

    Returns
    -------
    ZGrid
    """
    if z_lambda_edges is not None:
        logger.info(f"D{detector}: using custom Z grid ({len(z_lambda_edges) - 1} bins)")
        return build_z_grid_custom(z_lambda_edges)
    if z_oversample != 1.0:
        zgrid = build_z_grid_oversampled(detector, z_oversample)
        logger.info(f"D{detector}: oversampled Z grid ({zgrid.n_z} bins, R_eff×{z_oversample:.1f})")
        return zgrid
    zgrid = build_z_grid_native(detector)
    logger.info(f"D{detector}: native Z grid ({zgrid.n_z} bins)")
    return zgrid
