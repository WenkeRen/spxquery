"""
High-level Drizzle3D pipeline.

Orchestrates: query → download → per-detector drizzle → save.
Supports parallel preprocessing via multiprocessing when drizzle_workers > 1.
"""

import logging
import multiprocessing as mp
import os
import queue
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from astropy.wcs import WCS

from .accumulate import (
    DrizzleCube,
    _compute_voxel_contributions,
    drizzle_image,
)
from .config import Drizzle3DConfig
from .grid import build_output_wcs
from .io import save_cube
from .query import download_observations, query_observations
from .spatial import compute_spatial_mapping
from .spectral import ZGrid, build_z_grid
from ..utils.helpers import evict_file_pages

logger = logging.getLogger(__name__)

# ── Worker process state (set by _init_drizzle_worker) ────────────────────
_worker_config: Optional[Drizzle3DConfig] = None
_worker_output_wcs: Optional[WCS] = None
_worker_output_shape: Optional[Tuple[int, int]] = None
_worker_zgrid: Optional[ZGrid] = None
_worker_exclude_bits: int = 0


def _init_drizzle_worker(
    config: Drizzle3DConfig,
    output_wcs: WCS,
    output_shape: Tuple[int, int],
    zgrid: ZGrid,
    exclude_bits: int,
) -> None:
    """Initialize a worker process: limit threads and store shared read-only state."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # Suppress noisy per-file logging in workers
    for name in ("spxquery", "spxquery.drizzle3d", "spxquery.utils"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    global _worker_config, _worker_output_wcs, _worker_output_shape, _worker_zgrid, _worker_exclude_bits
    _worker_config = config
    _worker_output_wcs = output_wcs
    _worker_output_shape = output_shape
    _worker_zgrid = zgrid
    _worker_exclude_bits = exclude_bits


def _worker_compute(fpath: Path) -> Optional[str]:
    """Worker: read one FITS file, compute and pre-bin voxel contributions.

    Does the full compute + bincount in the worker process so the main
    process only needs fast array addition (no bincount). Writes pre-binned
    arrays to a temp .npz and returns the path (tiny IPC).
    """
    import tempfile

    try:
        img_data, var_data, flag_data, _, spatial_wcs, spectral_wcs = _read_input_fits(
            fpath, _worker_config.subtract_zodi, static_zodi=_worker_config.static_zodi,
            bg_fraction_reject_level=_worker_config.zodi_bg_fraction_min,
        )
    except Exception:
        return None

    if spatial_wcs is None:
        return None

    _, pix_idx, out_y, out_x, f_xy = compute_spatial_mapping(
        spatial_wcs,
        img_data.shape,
        _worker_output_wcs,
        _worker_output_shape,
        _worker_config.xy_shrink,
    )

    if len(out_y) == 0:
        return None

    lambda_c_map, delta_lambda_map = _extract_wavelength_maps(spectral_wcs, img_data.shape)

    exclude_mask = None
    if _worker_exclude_bits != 0:
        exclude_mask = (flag_data & _worker_exclude_bits) != 0

    contrib = _compute_voxel_contributions(
        image=img_data,
        variance=var_data,
        flags=flag_data,
        lambda_c_map=lambda_c_map,
        delta_lambda_map=delta_lambda_map,
        pixel_idx=pix_idx,
        out_y=out_y,
        out_x=out_x,
        f_xy=f_xy,
        z_shrink=_worker_config.effective_z_shrink(),
        ivar_max=_worker_config.ivar_max,
        min_overlap=_worker_config.min_overlap,
        z_edges=_worker_zgrid.edges,
        n_z=_worker_zgrid.n_z,
        exclude_mask=exclude_mask,
    )

    if contrib is None:
        return None

    # Pre-bin into cube-shaped arrays (the expensive bincount runs in worker)
    ny_out, nx_out = _worker_output_shape
    n_z = _worker_zgrid.n_z
    flat_size = n_z * ny_out * nx_out

    flat_idx = (
        contrib.z_flat.astype(np.int64) * (ny_out * nx_out)
        + contrib.y_flat.astype(np.int64) * nx_out
        + contrib.x_flat.astype(np.int64)
    )

    flux_acc = np.bincount(flat_idx, weights=contrib.wxf_flat * contrib.flux_flat, minlength=flat_size)
    weight_acc = np.bincount(flat_idx, weights=contrib.wxf_flat, minlength=flat_size)
    var_acc = np.bincount(flat_idx, weights=contrib.wxf_flat**2 * contrib.var_flat, minlength=flat_size)
    count_acc = np.bincount(flat_idx, minlength=flat_size)

    # Bitwise masks per worker: AND from all-ones, OR from zero
    and_mask = np.full(flat_size, 0xFFFFFFFF, dtype=np.uint32)
    np.bitwise_and.at(and_mask, flat_idx, contrib.flag_flat)
    or_mask = np.zeros(flat_size, dtype=np.uint32)
    np.bitwise_or.at(or_mask, flat_idx, contrib.flag_flat)

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False, dir=_worker_config.output_dir)
    np.savez(
        tmp.name,
        flux=flux_acc,
        weight=weight_acc,
        var=var_acc,
        count=count_acc,
        and_mask=and_mask,
        or_mask=or_mask,
    )
    tmp.close()

    # Release ALL large arrays first — this drops references to spatial_wcs /
    # spectral_wcs which may hold mmap handles from astropy, preventing
    # posix_fadvise from evicting the file's pages from the page cache.
    import gc

    del img_data, var_data, flag_data, spatial_wcs, spectral_wcs
    del lambda_c_map, delta_lambda_map, contrib
    del flux_acc, weight_acc, var_acc, count_acc, and_mask, or_mask, flat_idx
    gc.collect()

    # Evict the input FITS pages only after dropping WCS references that may
    # still hold mmap-backed astropy state.
    evict_file_pages(fpath)

    return tmp.name


def _iter_bounded_unordered(
    pool: mp.Pool,
    func: Callable[[Path], Optional[str]],
    items: Iterable[Path],
    max_pending: int,
) -> Iterator[Optional[str]]:
    """Yield unordered results while capping the number of in-flight tasks."""
    result_queue: "queue.Queue[tuple[bool, object]]" = queue.Queue()
    iterator = iter(items)
    pending = 0
    exhausted = False

    def _submit_one() -> bool:
        nonlocal pending, exhausted
        if exhausted:
            return False
        try:
            item = next(iterator)
        except StopIteration:
            exhausted = True
            return False

        pool.apply_async(
            func,
            (item,),
            callback=lambda result: result_queue.put((True, result)),
            error_callback=lambda exc: result_queue.put((False, exc)),
        )
        pending += 1
        return True

    max_pending = max(1, max_pending)
    while pending < max_pending and _submit_one():
        pass

    while pending > 0:
        ok, payload = result_queue.get()
        pending -= 1

        while pending < max_pending and _submit_one():
            pass

        if ok:
            yield payload  # type: ignore[misc]
        else:
            raise payload  # type: ignore[misc]


def drizzle_detector(
    fits_paths: List[Path],
    config: Drizzle3DConfig,
    detector: int,
    output_wcs: WCS,
) -> Optional[Path]:
    """Run the drizzle pipeline for one detector.

    When config.drizzle_workers > 1, file preprocessing runs in parallel
    across worker processes; accumulation into the shared cube is serial.

    Parameters
    ----------
    fits_paths : list of Path
        Downloaded FITS files for this detector.
    config : Drizzle3DConfig
        Drizzle configuration.
    detector : int
        Detector number (1–6).
    output_wcs : WCS
        Output spatial WCS.

    Returns
    -------
    Path or None
        Path to the output FITS file, or None if no valid inputs.
    """
    zgrid = build_z_grid(detector, config.z_oversample, config.z_lambda_edges)
    pixscale = config.effective_pixscale()
    cube = DrizzleCube.create(output_wcs, pixscale, zgrid, config, detector)
    output_shape = (config.output_ny(), config.output_nx())

    from ..utils.helpers import create_flag_mask

    exclude_bits = create_flag_mask(config.exclude_flags)
    n_rejected = 0

    if config.drizzle_workers <= 1:
        # ── Serial path (backward compatible) ────────────────────────────
        for fpath in fits_paths:
            try:
                img_data, var_data, flag_data, _, spatial_wcs, spectral_wcs = _read_input_fits(
                    fpath, config.subtract_zodi, static_zodi=config.static_zodi,
                    bg_fraction_reject_level=config.zodi_bg_fraction_min,
                )
            except Exception as e:
                logger.warning(f"Skipping {fpath.name}: {e}")
                n_rejected += 1
                continue

            if spatial_wcs is None:
                logger.warning(f"Skipping {fpath.name}: no spatial WCS")
                n_rejected += 1
                continue

            _, pix_idx, out_y, out_x, f_xy = compute_spatial_mapping(
                spatial_wcs, img_data.shape, output_wcs, output_shape, config.xy_shrink,
            )

            if len(out_y) == 0:
                logger.debug(f"Skipping {fpath.name}: no spatial overlap")
                n_rejected += 1
                continue

            lambda_c_map, delta_lambda_map = _extract_wavelength_maps(spectral_wcs, img_data.shape)

            exclude_mask = None
            if exclude_bits != 0:
                exclude_mask = (flag_data & exclude_bits) != 0

            drizzle_image(
                cube=cube,
                image=img_data,
                variance=var_data,
                flags=flag_data,
                lambda_c_map=lambda_c_map,
                delta_lambda_map=delta_lambda_map,
                pixel_idx=pix_idx,
                out_y=out_y,
                out_x=out_x,
                f_xy=f_xy,
                exclude_mask=exclude_mask,
            )

            # Release large arrays and evict input FITS from page cache.
            # Same rationale as the parallel path in _worker_compute():
            # astropy WCS objects may hold mmap handles that keep the kernel
            # from releasing the file's pages.  Explicit del + gc.collect()
            # drops those references before posix_fadvise(DONTNEED).
            del img_data, var_data, flag_data, spatial_wcs, spectral_wcs
            del lambda_c_map, delta_lambda_map, pix_idx, out_y, out_x, f_xy
            if exclude_mask is not None:
                del exclude_mask
            import gc
            gc.collect()
            evict_file_pages(fpath)
    else:
        # ── Parallel path ────────────────────────────────────────────────
        n_workers = min(config.drizzle_workers, len(fits_paths))
        logger.info(f"D{detector}: parallel drizzle with {n_workers} workers")

        from tqdm import tqdm

        max_pending_tmp = config.max_pending_tmp if config.max_pending_tmp is not None else n_workers * 2
        max_pending_tmp = max(max_pending_tmp, n_workers)
        logger.info(f"D{detector}: bounded tmp backlog = {max_pending_tmp}")

        with mp.Pool(
            processes=n_workers,
            initializer=_init_drizzle_worker,
            initargs=(config, output_wcs, output_shape, zgrid, exclude_bits),
        ) as pool:
            pbar = tqdm(fits_paths, desc=f"D{detector} ({n_workers}w)", unit="file")
            for tmp_path in _iter_bounded_unordered(pool, _worker_compute, fits_paths, max_pending_tmp):
                pbar.update(1)
                if tmp_path is None:
                    n_rejected += 1
                else:
                    tmp_path = Path(tmp_path)
                    with np.load(tmp_path) as data:
                        # Fast array addition (no bincount — that was done in worker)
                        cube.flux_weighted.ravel()[:] += data["flux"]
                        cube.weight_total.ravel()[:] += data["weight"]
                        cube.var_accum.ravel()[:] += data["var"]
                        cube.count_map.ravel()[:] += data["count"].astype(np.uint16)
                        # Bitwise merge: AND mask from all-ones identity, OR from zero identity
                        cube.and_mask.ravel()[:] &= data["and_mask"]
                        cube.or_mask.ravel()[:] |= data["or_mask"]
                        cube.n_inputs += 1

                    evict_file_pages(tmp_path)
                    os.unlink(tmp_path)
            pbar.close()

    cube.n_rejected = n_rejected
    cube.finalize_masks()

    output_path = Path(config.output_dir) / f"drizzle_D{detector}.fits"
    save_cube(cube, output_path, overwrite=config.overwrite)
    return output_path


def drizzle(config: Drizzle3DConfig) -> Dict[int, Path]:
    """Top-level entry point: query → download → drizzle → save.

    Parameters
    ----------
    config : Drizzle3DConfig
        Complete drizzle configuration.

    Returns
    -------
    dict
        {detector_id: output_path} for each successfully processed detector.

    Examples
    --------
    >>> from spxquery.drizzle3d import Drizzle3DConfig, drizzle
    >>> config = Drizzle3DConfig(
    ...     center_ra=186.4536,
    ...     center_dec=33.5468,
    ...     width=30.0,
    ...     height=30.0,
    ...     detector=3,
    ... )
    >>> results = drizzle(config)
    """
    logger.info(f"Starting Drizzle3D pipeline: center=({config.center_ra}, {config.center_dec})")
    logger.info(
        f"  Region: {config.width}'×{config.height}', "
        f"detector={'all' if config.detector == 0 else f'D{config.detector}'}"
    )

    # 1. Build output spatial WCS
    output_wcs = build_output_wcs(config)

    # 2. Query IRSA
    obs_by_det = query_observations(config)

    if not obs_by_det:
        logger.warning("No observations found for the target region")
        return {}

    results: Dict[int, Path] = {}

    for det, observations in sorted(obs_by_det.items()):
        logger.info(f"Processing D{det}: {len(observations)} observations")

        # 3. Download / resolve from mirror
        fits_paths = download_observations(
            observations,
            output_dir=config.output_dir,
            max_workers=config.download_workers,
            skip_existing=config.skip_existing,
            data_mirror=config.data_mirror,
        )

        if not fits_paths:
            logger.warning(f"D{det}: no files downloaded, skipping")
            continue

        # 4. Drizzle
        output_path = drizzle_detector(fits_paths, config, det, output_wcs)
        if output_path is not None:
            results[det] = output_path

    logger.info(f"Drizzle3D complete: {len(results)} detector cubes produced")
    for det, path in sorted(results.items()):
        logger.info(f"  D{det}: {path}")

    return results


def _read_input_fits(filepath: Path, subtract_zodi: bool, static_zodi: bool = False, bg_fraction_reject_level: float = 0.5):
    """Read a SPHEREx input FITS file using the shared MEF reader.

    Returns
    -------
    tuple
        (image, variance, flags, zodi, spatial_wcs, spectral_wcs)
    """
    from ..utils.spherex_mef import read_spherex_mef, subtract_zodiacal_background

    mef = read_spherex_mef(filepath)
    image = mef.image
    variance = mef.variance
    flags = mef.flags.astype(np.uint32)
    zodi = mef.zodi
    spatial_wcs = mef.spatial_wcs
    spectral_wcs = mef.spectral_wcs

    if subtract_zodi:
        image, _ = subtract_zodiacal_background(
            image, zodi, flags, variance, static_zodi=static_zodi,
            bg_fraction_reject_level=bg_fraction_reject_level,
        )

    return image, variance, flags, zodi, spatial_wcs, spectral_wcs


def _extract_wavelength_maps(spectral_wcs: WCS, shape) -> tuple:
    """Extract per-pixel (λ_c, Δλ) maps from the spectral WCS.

    Uses low-level wcs_pix2world to avoid Quantity wrapping overhead
    for millions of pixels.

    Parameters
    ----------
    spectral_wcs : WCS
        Spectral WCS (alternative 'W' key) from the input FITS.
    shape : tuple
        (ny, nx) image shape.

    Returns
    -------
    lambda_c_map : np.ndarray
        (ny, nx) central wavelength [μm].
    delta_lambda_map : np.ndarray
        (ny, nx) bandwidth [μm].
    """
    import astropy.units as u

    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    xx_flat = xx.ravel().astype(np.float64)
    yy_flat = yy.ravel().astype(np.float64)

    try:
        world = spectral_wcs.wcs_pix2world(xx_flat, yy_flat, 0)
        # One-time scalar conversion from native WCS units to micrometers
        cunit = spectral_wcs.wcs.cunit[0]
        um_factor = cunit.to(u.micron) if cunit is not None else 1.0
        lambda_c_flat = world[0] * um_factor
        delta_lambda_flat = world[1] * um_factor
    except Exception as e:
        logger.warning(f"Failed to extract wavelength maps: {e}")
        lambda_c_flat = np.full(ny * nx, np.nan)
        delta_lambda_flat = np.full(ny * nx, np.nan)

    lambda_c_map = lambda_c_flat.reshape(ny, nx).astype(np.float64)
    delta_lambda_map = delta_lambda_flat.reshape(ny, nx).astype(np.float64)

    return lambda_c_map, delta_lambda_map
