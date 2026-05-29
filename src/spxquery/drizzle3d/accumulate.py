"""
Core accumulation for Drizzle3D.

Contains the DrizzleCube data container and the per-image accumulation loop
that combines spatial + spectral overlaps into the output 3D cube.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy.wcs import WCS

from .config import Drizzle3DConfig
from .spectral import ZGrid

logger = logging.getLogger(__name__)


@dataclass
class DrizzleCube:
    """In-memory accumulation buffers for one detector's 3D drizzle.

    All image arrays have shape (n_z, n_y, n_x) — FITS ordering with
    the spectral axis as the first dimension.
    """

    # Spatial grid metadata
    wcs: WCS
    pixscale: float

    # Spectral grid
    zgrid: ZGrid

    # Accumulation arrays — shape (n_z, n_y, n_x)
    flux_weighted: np.ndarray  # Σ w_i × f_xy × f_z × F_i    [float64]
    weight_total: np.ndarray  # Σ w_i × f_xy × f_z          [float64]
    var_accum: np.ndarray  # Σ (w_i × f_xy × f_z)² × σ²  [float64]
    count_map: np.ndarray  # Number of contributing pixels [uint16]
    and_mask: np.ndarray  # Bitwise AND of input FLAGS    [uint32]
    or_mask: np.ndarray  # Bitwise OR of input FLAGS     [uint32]

    # Config snapshot
    config: Drizzle3DConfig
    detector: int
    n_inputs: int = 0
    n_rejected: int = 0

    @property
    def nz(self) -> int:
        return self.zgrid.n_z

    @property
    def ny(self) -> int:
        return self.flux_weighted.shape[1]

    @property
    def nx(self) -> int:
        return self.flux_weighted.shape[2]

    @property
    def flux(self) -> np.ndarray:
        """Normalized flux: flux_weighted / weight_total."""
        out = np.full_like(self.flux_weighted, np.nan)
        mask = self.weight_total > 0
        out[mask] = self.flux_weighted[mask] / self.weight_total[mask]
        return out.astype(np.float32)

    @property
    def variance(self) -> np.ndarray:
        """Per-voxel variance: var_accum / weight_total²."""
        out = np.full_like(self.var_accum, np.nan)
        mask = self.weight_total > 0
        out[mask] = self.var_accum[mask] / self.weight_total[mask] ** 2
        return out.astype(np.float32)

    def finalize_masks(self) -> None:
        """Clean AND_MASK: set to 0 where no data was accumulated."""
        self.and_mask[self.count_map == 0] = 0

    @classmethod
    def create(cls, wcs: WCS, pixscale: float, zgrid: ZGrid, config: Drizzle3DConfig, detector: int) -> "DrizzleCube":
        """Allocate an empty DrizzleCube with zero-initialized arrays."""
        nz = zgrid.n_z
        ny = config.output_ny()
        nx = config.output_nx()

        logger.info(f"D{detector}: allocating DrizzleCube ({nz}×{ny}×{nx})")

        return cls(
            wcs=wcs,
            pixscale=pixscale,
            zgrid=zgrid,
            flux_weighted=np.zeros((nz, ny, nx), dtype=np.float64),
            weight_total=np.zeros((nz, ny, nx), dtype=np.float64),
            var_accum=np.zeros((nz, ny, nx), dtype=np.float64),
            count_map=np.zeros((nz, ny, nx), dtype=np.uint16),
            and_mask=np.full((nz, ny, nx), 0xFFFFFFFF, dtype=np.uint32),
            or_mask=np.zeros((nz, ny, nx), dtype=np.uint32),
            config=config,
            detector=detector,
        )


def drizzle_image(
    cube: DrizzleCube,
    image: np.ndarray,
    variance: np.ndarray,
    flags: np.ndarray,
    lambda_c_map: np.ndarray,
    delta_lambda_map: np.ndarray,
    pixel_idx: np.ndarray,
    out_y: np.ndarray,
    out_x: np.ndarray,
    f_xy: np.ndarray,
    exclude_mask: Optional[np.ndarray] = None,
) -> None:
    """Drizzle one input image into the cube (in-place accumulation).

    Parameters
    ----------
    cube : DrizzleCube
        Target accumulation cube (modified in place).
    image : np.ndarray
        (ny_in, nx_in) input flux in MJy/sr.
    variance : np.ndarray
        (ny_in, nx_in) input per-pixel variance.
    flags : np.ndarray
        (ny_in, nx_in) input pixel quality flags.
    lambda_c_map : np.ndarray
        (ny_in, nx_in) per-pixel central wavelength [μm].
    delta_lambda_map : np.ndarray
        (ny_in, nx_in) per-pixel bandwidth [μm].
    pixel_idx : np.ndarray (int64)
        Flat input-pixel indices from compute_spatial_mapping.
        Each entry identifies which input pixel this spatial contribution
        belongs to.
    out_y, out_x : np.ndarray (int32)
        Output pixel coordinates from compute_spatial_mapping.
    f_xy : np.ndarray (float64)
        Spatial overlap fractions from compute_spatial_mapping.
    exclude_mask : np.ndarray, optional
        (ny_in, nx_in) bool — True for pixels to exclude (flagged).

    Notes
    -----
    The spatial mapping arrays (pixel_idx, out_y, out_x, f_xy) may have
    multiple entries per input pixel (bilinear produces up to 4).
    ``pixel_idx`` maps each contribution back to its source input pixel.
    """
    z_shrink = cube.config.effective_z_shrink()
    ivar_max = cube.config.ivar_max
    min_overlap = cube.config.min_overlap
    z_edges = cube.zgrid.edges
    n_z = cube.zgrid.n_z
    ny_in, nx_in = image.shape

    # Flatten per-pixel input data for fast lookup via pixel_idx
    image_flat = image.ravel().astype(np.float64)
    var_flat = variance.ravel().astype(np.float64)
    flags_flat = flags.ravel().astype(np.uint32)
    lam_flat = lambda_c_map.ravel()
    dlam_flat = delta_lambda_map.ravel()

    # Build exclude bitmask for flat indexing
    if exclude_mask is not None:
        exclude_flat = exclude_mask.ravel()
    else:
        exclude_flat = np.zeros(ny_in * nx_in, dtype=bool)

    # Filter spatial contributions by valid per-pixel data
    src = pixel_idx  # flat index into input image
    valid = (
        np.isfinite(lam_flat[src])
        & (lam_flat[src] > 0)
        & np.isfinite(dlam_flat[src])
        & (dlam_flat[src] > 0)
        & (var_flat[src] > 0)
        & ~exclude_flat[src]
    )

    if not np.any(valid):
        logger.debug("No valid spatial contributions after per-pixel filtering")
        cube.n_inputs += 1
        return

    # Keep only valid contributions
    src = src[valid]
    oy = out_y[valid]
    ox = out_x[valid]
    fxy = f_xy[valid]

    n_contrib = len(src)
    logger.debug(f"Drizzling {n_contrib} spatial contributions from this image")

    # Group by unique input pixel — vectorized spectral overlap computation
    unique_pix, inverse = np.unique(src, return_inverse=True)
    n_unique = len(unique_pix)

    # --- Vectorized spectral overlaps for all unique pixels at once ---
    # Dense overlap matrix: (n_unique, n_z)
    u_lam = lam_flat[unique_pix]
    u_dlam = dlam_flat[unique_pix]
    half_w = 0.5 * u_dlam * z_shrink
    u_lo = u_lam - half_w  # (n_unique,)
    u_hi = u_lam + half_w  # (n_unique,)

    overlap_lo = np.maximum(u_lo[:, None], z_edges[:-1][None, :])  # (n_unique, n_z)
    overlap_hi = np.minimum(u_hi[:, None], z_edges[1:][None, :])  # (n_unique, n_z)
    overlap_len = np.maximum(overlap_hi - overlap_lo, 0.0)  # (n_unique, n_z)

    dlam_shrunk = u_dlam * z_shrink
    f_z_dense = np.zeros((n_unique, n_z), dtype=np.float64)
    nonzero = dlam_shrunk > 0
    f_z_dense[nonzero] = overlap_len[nonzero] / dlam_shrunk[nonzero, None]

    # Normalize rows to sum ≈ 1
    row_sums = f_z_dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    f_z_dense /= row_sums

    # Per-unique-pixel data
    u_ivar = np.minimum(1.0 / var_flat[unique_pix], ivar_max)
    u_flux = image_flat[unique_pix]
    u_var = var_flat[unique_pix]
    u_flag = flags_flat[unique_pix]

    # Filter: keep only pixels with positive ivar AND spectral overlap
    good_pix = (u_ivar > 0) & (overlap_len.sum(axis=1) > 0)
    good_contrib = good_pix[inverse]

    src = src[good_contrib]
    oy = oy[good_contrib]
    ox = ox[good_contrib]
    fxy = fxy[good_contrib]
    inv = inverse[good_contrib]

    n_contrib = len(src)
    if n_contrib == 0:
        cube.n_inputs += 1
        return

    # --- Build cross-product: spatial (n_contrib) × spectral (n_z) ---
    # Per-contribution pixel data via inverse mapping
    w_i = u_ivar[inv]  # (n_contrib,)
    flux_i = u_flux[inv]  # (n_contrib,)
    var_i = u_var[inv]  # (n_contrib,)
    flag_i = u_flag[inv]  # (n_contrib,)
    f_z_con = f_z_dense[inv]  # (n_contrib, n_z)

    # total_f = f_xy × f_z  and  wxf = w_i × total_f
    total_f = fxy[:, None] * f_z_con  # (n_contrib, n_z)
    wxf = w_i[:, None] * total_f  # (n_contrib, n_z)

    # Apply min_overlap filter
    valid_mask = total_f >= min_overlap

    # Broadcast z-index and spatial coordinates
    z_bcast = np.broadcast_to(np.arange(n_z, dtype=np.int32)[None, :], total_f.shape)
    y_bcast = np.broadcast_to(oy[:, None], total_f.shape)
    x_bcast = np.broadcast_to(ox[:, None], total_f.shape)

    # Flatten and filter to valid voxel updates
    z_flat = z_bcast[valid_mask]
    y_flat = y_bcast[valid_mask]
    x_flat = x_bcast[valid_mask]
    wxf_flat = wxf[valid_mask]
    flux_flat = np.broadcast_to(flux_i[:, None], total_f.shape)[valid_mask]
    var_flat_out = np.broadcast_to(var_i[:, None], total_f.shape)[valid_mask]
    flag_flat = np.broadcast_to(flag_i[:, None], total_f.shape)[valid_mask].astype(np.uint32)

    n_voxels = len(z_flat)
    logger.debug(f"Accumulating {n_contrib} spatial × {n_z} spectral → {n_voxels} voxel updates")

    # --- Accumulate into cube ---
    # Use np.bincount for float accumulations (5-10x faster than np.add.at
    # because bincount uses a single C-level histogram pass, while add.at
    # iterates element-by-element with Python-level index unpacking).
    ny_out = cube.ny
    nx_out = cube.nx
    flat_size = n_z * ny_out * nx_out

    # Flatten 3D indices to 1D for bincount
    flat_idx = z_flat.astype(np.int64) * (ny_out * nx_out) + y_flat.astype(np.int64) * nx_out + x_flat.astype(np.int64)

    cube.flux_weighted.ravel()[:] += np.bincount(flat_idx, weights=wxf_flat * flux_flat, minlength=flat_size)
    cube.weight_total.ravel()[:] += np.bincount(flat_idx, weights=wxf_flat, minlength=flat_size)
    cube.var_accum.ravel()[:] += np.bincount(flat_idx, weights=wxf_flat**2 * var_flat_out, minlength=flat_size)
    cube.count_map.ravel()[:] += np.bincount(flat_idx, minlength=flat_size).astype(np.uint16)

    # Bitwise accumulations still use add.at (bincount has no bitwise mode)
    np.bitwise_and.at(cube.and_mask, (z_flat, y_flat, x_flat), flag_flat)
    np.bitwise_or.at(cube.or_mask, (z_flat, y_flat, x_flat), flag_flat)

    cube.n_inputs += 1
    logger.debug(f"Accumulated {n_voxels} voxel contributions from this image")
