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


@dataclass
class VoxelContributions:
    """Flat voxel update arrays from one input image.

    Produced by workers, consumed by the main process for accumulation.
    All arrays are 1-D with matching length.
    """

    z_flat: np.ndarray  # (n_voxels,) int32 — Z bin index
    y_flat: np.ndarray  # (n_voxels,) int32 — Y pixel index
    x_flat: np.ndarray  # (n_voxels,) int32 — X pixel index
    wxf_flat: np.ndarray  # (n_voxels,) float64 — weight × f_xy × f_z
    flux_flat: np.ndarray  # (n_voxels,) float64 — input flux
    var_flat: np.ndarray  # (n_voxels,) float64 — input variance
    flag_flat: np.ndarray  # (n_voxels,) uint32 — input flags


def _compute_voxel_contributions(
    image: np.ndarray,
    variance: np.ndarray,
    flags: np.ndarray,
    lambda_c_map: np.ndarray,
    delta_lambda_map: np.ndarray,
    pixel_idx: np.ndarray,
    out_y: np.ndarray,
    out_x: np.ndarray,
    f_xy: np.ndarray,
    z_shrink: float,
    ivar_max: float,
    min_overlap: float,
    z_edges: np.ndarray,
    n_z: int,
    exclude_mask: Optional[np.ndarray] = None,
) -> Optional[VoxelContributions]:
    """Compute voxel contributions from one input image (no accumulation).

    This is the parallelizable half of drizzle_image: it produces the
    flat index/weight arrays but does not touch the shared DrizzleCube.
    """
    ny_in, nx_in = image.shape

    image_flat = image.ravel().astype(np.float64)
    var_flat = variance.ravel().astype(np.float64)
    flags_flat = flags.ravel().astype(np.uint32)
    lam_flat = lambda_c_map.ravel()
    dlam_flat = delta_lambda_map.ravel()

    if exclude_mask is not None:
        exclude_flat = exclude_mask.ravel()
    else:
        exclude_flat = np.zeros(ny_in * nx_in, dtype=bool)

    src = pixel_idx
    valid = (
        np.isfinite(lam_flat[src])
        & (lam_flat[src] > 0)
        & np.isfinite(dlam_flat[src])
        & (dlam_flat[src] > 0)
        & (var_flat[src] > 0)
        & ~exclude_flat[src]
    )

    if not np.any(valid):
        return None

    src = src[valid]
    oy = out_y[valid]
    ox = out_x[valid]
    fxy = f_xy[valid]

    n_contrib = len(src)
    unique_pix, inverse = np.unique(src, return_inverse=True)
    n_unique = len(unique_pix)

    # Vectorized spectral overlaps: dense (n_unique, n_z) matrix
    u_lam = lam_flat[unique_pix]
    u_dlam = dlam_flat[unique_pix]
    half_w = 0.5 * u_dlam * z_shrink
    u_lo = u_lam - half_w
    u_hi = u_lam + half_w

    overlap_lo = np.maximum(u_lo[:, None], z_edges[:-1][None, :])
    overlap_hi = np.minimum(u_hi[:, None], z_edges[1:][None, :])
    overlap_len = np.maximum(overlap_hi - overlap_lo, 0.0)

    dlam_shrunk = u_dlam * z_shrink
    f_z_dense = np.zeros((n_unique, n_z), dtype=np.float64)
    nonzero = dlam_shrunk > 0
    f_z_dense[nonzero] = overlap_len[nonzero] / dlam_shrunk[nonzero, None]

    row_sums = f_z_dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    f_z_dense /= row_sums

    u_ivar = np.minimum(1.0 / var_flat[unique_pix], ivar_max)
    u_flux = image_flat[unique_pix]
    u_var = var_flat[unique_pix]
    u_flag = flags_flat[unique_pix]

    good_pix = (u_ivar > 0) & (overlap_len.sum(axis=1) > 0)
    good_contrib = good_pix[inverse]

    src = src[good_contrib]
    oy = oy[good_contrib]
    ox = ox[good_contrib]
    fxy = fxy[good_contrib]
    inv = inverse[good_contrib]

    n_contrib = len(src)
    if n_contrib == 0:
        return None

    # Cross-product: spatial (n_contrib) × spectral (n_z)
    w_i = u_ivar[inv]
    flux_i = u_flux[inv]
    var_i = u_var[inv]
    flag_i = u_flag[inv]
    f_z_con = f_z_dense[inv]

    total_f = fxy[:, None] * f_z_con
    wxf = w_i[:, None] * total_f

    valid_mask = total_f >= min_overlap

    z_bcast = np.broadcast_to(np.arange(n_z, dtype=np.int32)[None, :], total_f.shape)
    y_bcast = np.broadcast_to(oy[:, None], total_f.shape)
    x_bcast = np.broadcast_to(ox[:, None], total_f.shape)

    z_flat = z_bcast[valid_mask]
    y_flat = y_bcast[valid_mask]
    x_flat = x_bcast[valid_mask]
    wxf_flat = wxf[valid_mask]
    flux_flat = np.broadcast_to(flux_i[:, None], total_f.shape)[valid_mask]
    var_flat_out = np.broadcast_to(var_i[:, None], total_f.shape)[valid_mask]
    flag_flat = np.broadcast_to(flag_i[:, None], total_f.shape)[valid_mask].astype(np.uint32)

    return VoxelContributions(
        z_flat=z_flat,
        y_flat=y_flat,
        x_flat=x_flat,
        wxf_flat=wxf_flat,
        flux_flat=flux_flat,
        var_flat=var_flat_out,
        flag_flat=flag_flat,
    )


def _accumulate_voxels(cube: DrizzleCube, contrib: VoxelContributions) -> None:
    """Accumulate pre-computed voxel contributions into a DrizzleCube.

    Must be called from the main process (serial access to the shared cube).
    """
    ny_out = cube.ny
    nx_out = cube.nx
    n_z = cube.nz
    flat_size = n_z * ny_out * nx_out

    flat_idx = (
        contrib.z_flat.astype(np.int64) * (ny_out * nx_out)
        + contrib.y_flat.astype(np.int64) * nx_out
        + contrib.x_flat.astype(np.int64)
    )

    cube.flux_weighted.ravel()[:] += np.bincount(flat_idx, weights=contrib.wxf_flat * contrib.flux_flat, minlength=flat_size)
    cube.weight_total.ravel()[:] += np.bincount(flat_idx, weights=contrib.wxf_flat, minlength=flat_size)
    cube.var_accum.ravel()[:] += np.bincount(flat_idx, weights=contrib.wxf_flat**2 * contrib.var_flat, minlength=flat_size)
    cube.count_map.ravel()[:] += np.bincount(flat_idx, minlength=flat_size).astype(np.uint16)

    np.bitwise_and.at(cube.and_mask, (contrib.z_flat, contrib.y_flat, contrib.x_flat), contrib.flag_flat)
    np.bitwise_or.at(cube.or_mask, (contrib.z_flat, contrib.y_flat, contrib.x_flat), contrib.flag_flat)


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

    Thin wrapper that computes voxel contributions then accumulates them.
    Backward-compatible with the pre-parallelization API.
    """
    contrib = _compute_voxel_contributions(
        image,
        variance,
        flags,
        lambda_c_map,
        delta_lambda_map,
        pixel_idx,
        out_y,
        out_x,
        f_xy,
        z_shrink=cube.config.effective_z_shrink(),
        ivar_max=cube.config.ivar_max,
        min_overlap=cube.config.min_overlap,
        z_edges=cube.zgrid.edges,
        n_z=cube.zgrid.n_z,
        exclude_mask=exclude_mask,
    )
    if contrib is not None:
        _accumulate_voxels(cube, contrib)
    cube.n_inputs += 1
