"""
Spatial (XY) drizzle kernel for Drizzle3D.

Computes the mapping from input image pixels to output grid pixels,
including pixfrac shrinkage and fractional overlap areas.

Uses vectorised numpy operations for efficiency on full-frame images.
"""

import logging
from typing import Tuple

import numpy as np
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


def compute_spatial_mapping(
    input_wcs: WCS,
    input_shape: Tuple[int, int],
    output_wcs: WCS,
    output_shape: Tuple[int, int],
    xy_shrink: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute spatial overlap mapping from input pixels to output grid.

    For each input pixel, computes the output pixel(s) it contributes to
    and the fractional overlap area using bilinear weight distribution.

    Parameters
    ----------
    input_wcs : WCS
        Input image spatial WCS.
    input_shape : tuple of int
        (ny_in, nx_in) shape of the input image.
    output_wcs : WCS
        Output grid spatial WCS (2-D tangent plane).
    output_shape : tuple of int
        (ny_out, nx_out) shape of the output grid.
    xy_shrink : float
        Droplet shrink factor (0, 1]. 1.0 = full pixel footprint.

    Returns
    -------
    valid_mask : np.ndarray (bool)
        (ny_in, nx_in) — True for pixels that land inside the output grid.
    pixel_idx : np.ndarray (int64)
        Flat index into the input image for each output contribution.
        Use ``rows, cols = np.unravel_index(pixel_idx, input_shape)`` to
        recover (y, x) coordinates.
    out_y : np.ndarray (int32)
        Output Y pixel indices.
    out_x : np.ndarray (int32)
        Output X pixel indices.
    f_xy : np.ndarray (float64)
        Spatial overlap fractions.
    """
    ny_in, nx_in = input_shape
    ny_out, nx_out = output_shape
    n_pix = ny_in * nx_in

    # Build pixel coordinate grids
    xx, yy = np.meshgrid(np.arange(nx_in, dtype=np.float64), np.arange(ny_in, dtype=np.float64))
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()

    # Transform all input pixel centers to sky coordinates
    sky = input_wcs.pixel_to_world(xx_flat, yy_flat)
    ra_flat = sky.ra.deg
    dec_flat = sky.dec.deg

    # Project onto the output tangent plane
    out_xf, out_yf = output_wcs.world_to_pixel_values(ra_flat, dec_flat)

    # Validity mask: pixels that land inside the output grid (with margin)
    margin = 2.0
    valid = (
        (out_xf >= -margin)
        & (out_xf < nx_out + margin)
        & (out_yf >= -margin)
        & (out_yf < ny_out + margin)
        & np.isfinite(out_xf)
        & np.isfinite(out_yf)
    )
    valid_mask = valid.reshape(ny_in, nx_in)

    if not np.any(valid):
        logger.warning("No input pixels overlap the output grid")
        empty = np.array([], dtype=np.int64)
        empty_i = np.array([], dtype=np.int32)
        empty_f = np.array([], dtype=np.float64)
        return valid_mask, empty, empty_i, empty_i, empty_f

    # Work with the valid subset
    valid_idx = np.where(valid)[0]  # flat indices into input image
    vx = out_xf[valid]
    vy = out_yf[valid]
    n_valid = len(valid_idx)

    # Bilinear weight distribution: each input pixel contributes to up to 4 output pixels.
    # For output pixel (ox, oy), weight = (1 - |vx - (ox+0.5)|) * (1 - |vy - (oy+0.5)|) * shrink²
    # We only need to check the 4 pixels surrounding the continuous position.

    # Lower-left corner of the 2×2 neighborhood
    ox0 = np.floor(vx).astype(np.int32)
    oy0 = np.floor(vy).astype(np.int32)

    pix_idx_list = []
    out_y_list = []
    out_x_list = []
    f_xy_list = []

    for dy in (0, 1):
        for dx in (0, 1):
            ox = ox0 + dx
            oy = oy0 + dy

            # Distance from input position to center of this output pixel
            dist_x = np.abs(vx - (ox + 0.5))
            dist_y = np.abs(vy - (oy + 0.5))

            # Bilinear weight: nonzero only when dist < 1
            weight = np.maximum(1.0 - dist_x, 0.0) * np.maximum(1.0 - dist_y, 0.0)
            weight *= xy_shrink**2  # shrink factor

            # Select contributions with nonzero weight and within bounds
            mask = (weight > 1e-10) & (ox >= 0) & (ox < nx_out) & (oy >= 0) & (oy < ny_out)
            if not np.any(mask):
                continue

            pix_idx_list.append(valid_idx[mask])
            out_y_list.append(oy[mask])
            out_x_list.append(ox[mask])
            f_xy_list.append(weight[mask])

    if not pix_idx_list:
        logger.warning("No valid spatial overlaps after bilinear distribution")
        empty = np.array([], dtype=np.int64)
        empty_i = np.array([], dtype=np.int32)
        empty_f = np.array([], dtype=np.float64)
        return valid_mask, empty, empty_i, empty_i, empty_f

    pixel_idx_arr = np.concatenate(pix_idx_list)
    out_y_arr = np.concatenate(out_y_list)
    out_x_arr = np.concatenate(out_x_list)
    f_xy_arr = np.concatenate(f_xy_list)

    logger.debug(f"Spatial mapping: {n_valid} valid input pixels → {len(out_y_arr)} output contributions")

    return valid_mask, pixel_idx_arr, out_y_arr, out_x_arr, f_xy_arr


def compute_spectral_overlaps(
    lambda_c: float,
    delta_lambda: float,
    z_shrink: float,
    z_edges: np.ndarray,
    z_centers: np.ndarray,
    z_widths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spectral overlap fractions for a single input pixel.

    Uses a top-hat (boxcar) kernel centered at lambda_c with width
    delta_lambda * z_shrink.

    Parameters
    ----------
    lambda_c : float
        Input pixel central wavelength [μm].
    delta_lambda : float
        Input pixel bandwidth [μm].
    z_shrink : float
        Spectral droplet shrink factor.
    z_edges : np.ndarray
        (n_z+1,) output Z bin edges [μm].
    z_centers : np.ndarray
        (n_z,) output Z bin centers [μm].
    z_widths : np.ndarray
        (n_z,) output Z bin widths [μm].

    Returns
    -------
    z_idx : np.ndarray (int32)
        Indices of overlapping Z bins.
    f_z : np.ndarray (float64)
        Fractional overlap for each Z bin, summing to ~1.
    """
    half_width = 0.5 * delta_lambda * z_shrink
    lo = lambda_c - half_width
    hi = lambda_c + half_width
    dlam_shrunk = delta_lambda * z_shrink

    if dlam_shrunk <= 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    # Find Z bins that overlap [lo, hi]
    overlap_lo = np.maximum(lo, z_edges[:-1])
    overlap_hi = np.minimum(hi, z_edges[1:])
    overlap_len = overlap_hi - overlap_lo

    mask = overlap_len > 0
    if not np.any(mask):
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    z_idx = np.where(mask)[0].astype(np.int32)
    f_z = overlap_len[mask] / dlam_shrunk

    # Normalize to ensure sum ≈ 1 (edge effects near grid boundaries)
    f_z_sum = f_z.sum()
    if f_z_sum > 0:
        f_z /= f_z_sum

    return z_idx, f_z
