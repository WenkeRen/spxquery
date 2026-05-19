"""
FITS I/O for Drizzle3D output cubes.

Writes and reads the 7-HDU format:
  HDU 0 — PRIMARY (empty, with metadata headers)
  HDU 1 — SCI (float32, flux-weighted mean surface brightness)
  HDU 2 — VARIANCE (float32)
  HDU 3 — AND_MASK (uint32, conservative flag)
  HDU 4 — OR_MASK (uint32, inclusive flag)
  HDU 5 — COUNT (uint16, contribution count)
  HDU 6 — WAVELENGTH (BinTableHDU, spectral axis lookup)
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .accumulate import DrizzleCube

logger = logging.getLogger(__name__)

_MODULE_VERSION = "0.1.0"


def save_cube(cube: DrizzleCube, path: Path, overwrite: bool = False) -> None:
    """Write DrizzleCube to 7-HDU FITS.

    Parameters
    ----------
    cube : DrizzleCube
        Completed accumulation cube.
    path : Path
        Output FITS file path.
    overwrite : bool
        Whether to overwrite an existing file.

    Notes
    -----
    After saving, the FITS file can be opened in DS9, CARTA, or any
    standard FITS viewer.  The SCI HDU contains an approximate linear
    3-D WCS for display; the exact per-plane wavelengths are in the
    WAVELENGTH table (HDU 6).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")

    # Finalize cube before writing
    cube.finalize_masks()

    # Compute output arrays
    sci = cube.flux  # float32 (n_z, n_y, n_x)
    var = cube.variance  # float32
    and_mask = cube.and_mask  # uint32
    or_mask = cube.or_mask  # uint32
    count = cube.count_map  # uint16

    # ── HDU 0: PRIMARY ────────────────────────────────────────────────────
    primary = fits.PrimaryHDU()
    hdr = primary.header
    hdr["TELESCOP"] = "SPHEREx"
    hdr["INSTRUME"] = "SPHEREx"
    hdr["FILETYPE"] = "DRIZZLED_CUBE"
    hdr["DETECTOR"] = (cube.detector, "Detector number (1-6)")
    hdr["N_INPUTS"] = (cube.n_inputs, "Number of input images combined")
    hdr["N_REJECT"] = (cube.n_rejected, "Number of input images rejected")
    hdr["XYSHRINK"] = (cube.config.xy_shrink, "XY droplet shrink factor")
    hdr["ZSHRINK"] = (cube.config.effective_z_shrink(), "Z droplet shrink factor")
    hdr["XYOVRSM"] = (cube.config.xy_oversample, "XY output grid oversampling")
    hdr["ZOVRSM"] = (cube.config.z_oversample, "Z output grid oversampling")
    hdr["PIXSCALE"] = (cube.config.BASE_PIXSCALE, "Base pixel scale [arcsec]")
    hdr["EPIXSCLE"] = (cube.pixscale, "Effective pixel scale [arcsec]")
    hdr["BUNIT"] = "MJy/sr"
    hdr["DRIZVER"] = (_MODULE_VERSION, "Drizzle3D module version")
    hdr["DRIZDATE"] = (datetime.now(timezone.utc).isoformat(), "Processing timestamp")
    hdr["BAND_NM"] = (f"D{cube.detector}", "Band name")
    hdr["LAM_MIN"] = (float(cube.zgrid.edges[0]), "Min wavelength [um]")
    hdr["LAM_MAX"] = (float(cube.zgrid.edges[-1]), "Max wavelength [um]")
    hdr["NZ"] = (cube.nz, "Number of Z planes")
    hdr["NX"] = (cube.nx, "Spatial X dimension")
    hdr["NY"] = (cube.ny, "Spatial Y dimension")

    # ── Build 3-D WCS for SCI HDU ─────────────────────────────────────────
    sci_hdu = fits.ImageHDU(data=sci, name="SCI")
    _write_3d_wcs_header(sci_hdu.header, cube.wcs, cube.zgrid)
    sci_hdu.header["BUNIT"] = "MJy/sr"

    # ── HDU 2: VARIANCE ───────────────────────────────────────────────────
    var_hdu = fits.ImageHDU(data=var, name="VARIANCE")
    _write_3d_wcs_header(var_hdu.header, cube.wcs, cube.zgrid)
    var_hdu.header["BUNIT"] = "MJy^2/sr^2"

    # ── HDU 3: AND_MASK ───────────────────────────────────────────────────
    and_hdu = fits.ImageHDU(data=and_mask, name="AND_MASK")

    # ── HDU 4: OR_MASK ────────────────────────────────────────────────────
    or_hdu = fits.ImageHDU(data=or_mask, name="OR_MASK")

    # ── HDU 5: COUNT ──────────────────────────────────────────────────────
    cnt_hdu = fits.ImageHDU(data=count, name="COUNT")

    # ── HDU 6: WAVELENGTH table ───────────────────────────────────────────
    wl_table = _build_wavelength_table(cube.zgrid)
    wl_hdu = fits.BinTableHDU(wl_table, name="WAVELENGTH")

    # ── Write ─────────────────────────────────────────────────────────────
    hdul = fits.HDUList([primary, sci_hdu, var_hdu, and_hdu, or_hdu, cnt_hdu, wl_hdu])
    hdul.writeto(path, overwrite=overwrite)
    hdul.close()

    size_mb = path.stat().st_size / 1024 / 1024
    logger.info(f"Saved D{cube.detector} cube to {path} ({size_mb:.1f} MB)")


def _write_3d_wcs_header(header: fits.Header, wcs_2d: WCS, zgrid) -> None:
    """Inject 3-D WCS keywords into an image HDU header.

    Approximate linear WCS for the Z axis so standard viewers can display
    the cube.  The exact wavelengths are in the WAVELENGTH extension.
    """
    # Spatial WCS
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = wcs_2d.wcs.crval[0]
    header["CRVAL2"] = wcs_2d.wcs.crval[1]
    header["CRPIX1"] = wcs_2d.wcs.crpix[0]
    header["CRPIX2"] = wcs_2d.wcs.crpix[1]
    header["CDELT1"] = wcs_2d.wcs.cdelt[0]
    header["CDELT2"] = wcs_2d.wcs.cdelt[1]
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"

    # Spectral WCS (approximate linear)
    header["CTYPE3"] = "WAVE"
    header["CRVAL3"] = zgrid.centers[0] * 1e-6  # μm → m
    header["CRPIX3"] = 1.0
    mean_dlambda = np.mean(zgrid.widths) * 1e-6  # μm → m
    header["CDELT3"] = mean_dlambda
    header["CUNIT3"] = "m"


def _build_wavelength_table(zgrid) -> fits.FITS_rec:
    """Build the WAVELENGTH BinTableHDU data from ZGrid."""
    n_z = zgrid.n_z
    cols = [
        fits.Column(name="INDEX", format="I", array=np.arange(n_z, dtype=np.int16)),
        fits.Column(name="LAMBDA", format="D", array=zgrid.centers, unit="um"),
        fits.Column(name="LAMBDA_MIN", format="D", array=zgrid.edges[:-1], unit="um"),
        fits.Column(name="LAMBDA_MAX", format="D", array=zgrid.edges[1:], unit="um"),
        fits.Column(name="DLAMBDA", format="D", array=zgrid.widths, unit="um"),
        fits.Column(name="NU", format="D", array=zgrid.frequencies(), unit="Hz"),
        fits.Column(name="DNU", format="D", array=zgrid.delta_nu(), unit="Hz"),
        fits.Column(name="R_EFF", format="E", array=zgrid.resolving_power().astype(np.float32)),
        # N_INPUT_SC would require per-bin tracking during accumulation;
        # populate with placeholder for now.
        fits.Column(name="N_INPUT_SC", format="I", array=np.ones(n_z, dtype=np.int16)),
    ]
    return fits.FITS_rec.from_columns(fits.ColDefs(cols))


def load_cube(path) -> dict:
    """Load a drizzled cube from FITS for inspection.

    Returns a dict with 'sci', 'variance', 'and_mask', 'or_mask', 'count',
    'wavelength' (BinTable), and 'header' keys.

    Parameters
    ----------
    path : Path
        Path to the drizzled FITS file.

    Returns
    -------
    dict
        Dictionary of HDU data and header.
    """
    path = Path(path)
    with fits.open(path) as hdul:
        result = {
            "header": hdul[0].header,
            "sci": hdul["SCI"].data,
            "variance": hdul["VARIANCE"].data,
            "and_mask": hdul["AND_MASK"].data,
            "or_mask": hdul["OR_MASK"].data,
            "count": hdul["COUNT"].data,
            "wavelength": hdul["WAVELENGTH"].data,
        }

    logger.info(f"Loaded cube from {path}: SCI shape {result['sci'].shape}")
    return result
