"""Profile drizzle_image to identify per-step timing."""

import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Accumulated timing stats across calls
_stats = {}


def _t(name):
    """Return a context manager that records elapsed time for a named step."""
    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *args):
            dt = time.perf_counter() - self.t0
            _stats.setdefault(name, []).append(dt)
    return _Timer()


def print_stats():
    """Print accumulated timing statistics."""
    if not _stats:
        return
    print("\n" + "=" * 60)
    print("drizzle_image profiling summary (over {} calls)".format(
        len(next(iter(_stats.values())))))
    print("-" * 60)
    total = sum(sum(v) for v in _stats.values())
    for name, times in _stats.items():
        t = sum(times)
        pct = 100 * t / total if total > 0 else 0
        avg = t / len(times)
        print(f"  {name:40s}  {t:7.3f}s  ({pct:5.1f}%)  avg {avg:.4f}s")
    print(f"  {'TOTAL':40s}  {total:7.3f}s")
    print("=" * 60)


def drizzle_image_profiled(
    cube,
    image,
    variance,
    flags,
    lambda_c_map,
    delta_lambda_map,
    pixel_idx,
    out_y,
    out_x,
    f_xy,
    exclude_mask=None,
):
    """Drop-in replacement for drizzle_image with per-step timing."""
    from .accumulate import DrizzleCube

    z_shrink = cube.config.effective_z_shrink()
    ivar_max = cube.config.ivar_max
    min_overlap = cube.config.min_overlap
    z_edges = cube.zgrid.edges
    n_z = cube.zgrid.n_z
    ny_in, nx_in = image.shape

    # Step 1: Flatten input arrays
    with _t("1_flatten"):
        image_flat = image.ravel().astype(np.float64)
        var_flat = variance.ravel().astype(np.float64)
        flags_flat = flags.ravel().astype(np.uint32)
        lam_flat = lambda_c_map.ravel()
        dlam_flat = delta_lambda_map.ravel()

        if exclude_mask is not None:
            exclude_flat = exclude_mask.ravel()
        else:
            exclude_flat = np.zeros(ny_in * nx_in, dtype=bool)

    # Step 2: Filter valid spatial contributions
    with _t("2_filter"):
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
            cube.n_inputs += 1
            return

        src = src[valid]
        oy = out_y[valid]
        ox = out_x[valid]
        fxy = f_xy[valid]

    n_contrib = len(src)
    with _t("3_unique"):
        unique_pix, inverse = np.unique(src, return_inverse=True)
    n_unique = len(unique_pix)

    # Step 4: Dense spectral overlap
    with _t("4_spectral_overlap"):
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

    # Step 5: Per-pixel data
    with _t("5_per_pix_data"):
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
        cube.n_inputs += 1
        return

    # Step 6: Cross-product
    with _t("6_cross_product"):
        w_i = u_ivar[inv]
        flux_i = u_flux[inv]
        var_i = u_var[inv]
        flag_i = u_flag[inv]
        f_z_con = f_z_dense[inv]

        total_f = fxy[:, None] * f_z_con
        wxf = w_i[:, None] * total_f

    # Step 7: Filter + flatten
    with _t("7_flatten_filter"):
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

    n_voxels = len(z_flat)

    # Step 8: Accumulate (bincount for float, add.at for bitwise)
    with _t("8_accum"):
        ny_out = cube.ny
        nx_out = cube.nx
        flat_size = n_z * ny_out * nx_out
        flat_idx = z_flat.astype(np.int64) * (ny_out * nx_out) + y_flat.astype(np.int64) * nx_out + x_flat.astype(np.int64)

        cube.flux_weighted.ravel()[:] += np.bincount(flat_idx, weights=wxf_flat * flux_flat, minlength=flat_size)
        cube.weight_total.ravel()[:] += np.bincount(flat_idx, weights=wxf_flat, minlength=flat_size)
        cube.var_accum.ravel()[:] += np.bincount(flat_idx, weights=wxf_flat**2 * var_flat_out, minlength=flat_size)
        cube.count_map.ravel()[:] += np.bincount(flat_idx, minlength=flat_size).astype(np.uint16)

        np.bitwise_and.at(cube.and_mask, (z_flat, y_flat, x_flat), flag_flat)
        np.bitwise_or.at(cube.or_mask, (z_flat, y_flat, x_flat), flag_flat)

    cube.n_inputs += 1

    # Print per-call summary on first few calls
    n_calls = len(_stats.get("1_flatten", []))
    if n_calls <= 3:
        logger.info(
            f"Profile #{n_calls}: n_contrib={n_contrib}, n_unique={n_unique}, "
            f"n_z={n_z}, n_voxels={n_voxels}"
        )
