"""
Enhanced SPHEREx Multi-Extension FITS (MEF) file handling with utility methods.

This module provides the SPHERExMEF class and related utilities for working with
SPHEREx spectral imaging data, including:
- Unit conversion (MJy/sr to μJy/arcsec² or other units)
- PSF extraction from 121-zone (11×11) oversampled grid
- WCS coordinate transformation wrappers
- Image cutout extraction
- Zodiacal background subtraction
"""

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import astropy.units as u
import numpy as np
from astropy import log as astropy_log
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


@contextmanager
def suppress_astropy_info():
    """
    Context manager to temporarily suppress astropy INFO messages and FITS warnings.

    This is needed because SPHEREx provides non-standard WCS headers that trigger
    harmless but annoying INFO messages about SIP distortion coefficients and
    warnings about redundant SCAMP distortion parameters.
    """
    original_level = astropy_log.level
    astropy_log.setLevel("WARNING")

    try:
        # Suppress specific FITSFixedWarning about redundant SCAMP distortion parameters
        with warnings.catch_warnings():
            # Catch the warning by message pattern, regardless of category
            warnings.filterwarnings("ignore", message=".*Removed redundant SCAMP distortion parameters.*")
            warnings.filterwarnings("ignore", message=".*because SIP parameters are also present.*")
            yield
    finally:
        astropy_log.setLevel(original_level)


@dataclass
class SPHERExMEF:
    """
    Container for SPHEREx Multi-Extension FITS data with enhanced methods.

    Attributes
    ----------
    filepath : Path
        Path to original FITS file
    image : np.ndarray
        Calibrated flux (units specified by image_unit)
    flags : np.ndarray
        Bitmap flags for pixel quality
    variance : np.ndarray
        Variance array (units: image_unit²)
    zodi : np.ndarray
        Zodiacal light model (same units as image)
    psf : np.ndarray
        PSF cube (101×101×121) - 11×11 zones, each oversampled 10× (101×101 pixels)
    spatial_wcs : WCS
        Primary astrometric WCS (RA/Dec)
    spectral_wcs : WCS
        Alternative spectral WCS (wavelength/bandwidth)
    header : fits.Header
        Primary IMAGE extension header
    psf_header : fits.Header
        PSF extension header (contains XCTR_*, YCTR_*, OVERSAMP)
    obs_id : str
        Observation ID
    detector : int
        Detector number
    mjd : float
        Modified Julian Date (average of MJD-BEG and MJD-END)
    image_unit : str
        Units of image data (e.g., 'MJy/sr', 'uJy/arcsec2')
    native_unit : str
        Original unit from FITS file (always 'MJy/sr')
    _psf_zone_centers : Optional[Dict[int, Tuple[float, float]]]
        Cached PSF zone center coordinates (zone_id -> (x, y))
    _psf_oversamp : Optional[int]
        Cached PSF oversampling factor from header
    """

    filepath: Path
    image: np.ndarray
    flags: np.ndarray
    variance: np.ndarray
    zodi: np.ndarray
    psf: np.ndarray
    spatial_wcs: WCS
    spectral_wcs: WCS
    header: fits.Header
    psf_header: fits.Header
    obs_id: str
    detector: int
    mjd: float
    image_unit: str = "MJy/sr"
    native_unit: str = "MJy/sr"
    _psf_zone_centers: Optional[Dict[int, Tuple[float, float]]] = None
    _psf_oversamp: Optional[int] = None

    @property
    def image_zodi_subtracted(self) -> np.ndarray:
        """Return zodiacal light subtracted image with amplitude scaling."""
        corrected_image, _ = subtract_zodiacal_background(self.image, self.zodi, self.flags, self.variance)
        return corrected_image

    @property
    def error(self) -> np.ndarray:
        """Return error array (sqrt of variance)."""
        return np.sqrt(self.variance)

    @property
    def psf_oversamp(self) -> int:
        """
        Get PSF oversampling factor from header.

        Returns
        -------
        int
            Oversampling factor (typically 10 for SPHEREx)
        """
        if self._psf_oversamp is None:
            self._psf_oversamp = self.psf_header.get("OVERSAMP", 10)
        return self._psf_oversamp

    @property
    def psf_pixel_scale(self) -> float:
        """
        Get PSF pixel scale in arcsec/pixel using oversampling factor from header.

        The PSF pixel scale is the detector pixel scale divided by the oversampling factor,
        which is read from the OVERSAMP keyword in the PSF header.

        Returns
        -------
        float
            PSF pixel scale in arcsec/pixel
        """
        # Get detector pixel scale from WCS at image center
        ny, nx = self.image.shape
        detector_pixel_scale = self.get_pixel_scale(nx / 2.0, ny / 2.0, fallback=6.2)

        # PSF is oversampled by factor from header (typically 10)
        oversamp = self.psf_oversamp
        psf_scale = detector_pixel_scale / oversamp

        logger.debug(
            f"PSF pixel scale: {psf_scale:.4f} arcsec/pixel (detector: {detector_pixel_scale:.4f}, oversamp: {oversamp})"
        )

        return psf_scale

    def _load_psf_zone_centers(self) -> Dict[int, Tuple[float, float]]:
        """
        Load PSF zone center coordinates from PSF header.

        Reads XCTR_i and YCTR_i keywords (i=1...121) to get the center
        coordinates of each PSF zone on the parent detector image.

        Returns
        -------
        Dict[int, Tuple[float, float]]
            Dictionary mapping zone_id (1-121) to (x, y) coordinates in FITS convention (1-based)
        """
        if self._psf_zone_centers is not None:
            return self._psf_zone_centers

        import re

        xctr = {}
        yctr = {}

        for key, val in self.psf_header.items():
            # Look for keys like XCTR_1, XCTR_2, ..., XCTR_121
            xm = re.match(r"XCTR_(\d+)", key)
            if xm:
                zone_id = int(xm.group(1))
                xctr[zone_id] = float(val)

            # Look for keys like YCTR_1, YCTR_2, ..., YCTR_121
            ym = re.match(r"YCTR_(\d+)", key)
            if ym:
                zone_id = int(ym.group(1))
                yctr[zone_id] = float(val)

        # Verify we got all zones
        if len(xctr) != len(yctr):
            logger.warning(f"Mismatch in PSF zone counts: {len(xctr)} X centers vs {len(yctr)} Y centers")

        # Build dictionary of zone centers
        zone_centers = {}
        for zone_id in xctr.keys():
            if zone_id in yctr:
                zone_centers[zone_id] = (xctr[zone_id], yctr[zone_id])

        logger.debug(f"Loaded {len(zone_centers)} PSF zone centers from header")

        # Cache for future use
        self._psf_zone_centers = zone_centers
        return zone_centers

    # ==================== WCS Wrapper Methods ====================

    def world_to_pixel(self, ra: float, dec: float) -> Tuple[float, float]:
        """
        Convert world coordinates (RA/Dec) to pixel coordinates.

        Parameters
        ----------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees

        Returns
        -------
        x, y : float
            Pixel coordinates (0-based)
        """
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        x, y = self.spatial_wcs.world_to_pixel(coord)

        # Check if coordinates are within image
        ny, nx = self.image.shape
        if not (0 <= x < nx and 0 <= y < ny):
            logger.warning(f"Coordinates ({x:.1f}, {y:.1f}) outside image bounds ({nx}, {ny})")

        return float(x), float(y)

    def pixel_to_world(self, x: float, y: float) -> SkyCoord:
        """
        Convert pixel coordinates to world coordinates (RA/Dec).

        Parameters
        ----------
        x, y : float
            Pixel coordinates (0-based)

        Returns
        -------
        SkyCoord
            Sky coordinates with RA/Dec
        """
        return self.spatial_wcs.pixel_to_world(x, y)

    def pixel_to_wavelength(self, x: float, y: float) -> Tuple[float, float]:
        """
        Get wavelength and bandwidth at pixel position.

        Uses the spectral WCS (alternative 'W') which provides wavelength
        information via lookup table for SPHEREx's 2D spectral mapping.

        Parameters
        ----------
        x, y : float
            Pixel coordinates (0-based)

        Returns
        -------
        wavelength : float
            Central wavelength in microns
        bandwidth : float
            Bandwidth in microns
        """
        # Use spectral WCS to get wavelength info
        spectral_coords = self.spectral_wcs.pixel_to_world(x, y)

        # spectral_coords is a tuple of (wavelength, bandpass)
        wavelength = spectral_coords[0].to(u.micron).value
        bandwidth = spectral_coords[1].to(u.micron).value

        return wavelength, bandwidth

    def get_pixel_scale(self, x: float, y: float, fallback: float = 6.2) -> float:
        """
        Calculate the pixel scale in arcsec/pixel at a given position.

        This accounts for any WCS distortions and provides the actual pixel scale
        at the specified position.

        Parameters
        ----------
        x, y : float
            Pixel coordinates (0-based)
        fallback : float
            Fallback pixel scale in arcsec/pixel if WCS fails (default: 6.2 for SPHEREx)

        Returns
        -------
        float
            Pixel scale in arcseconds per pixel
        """
        try:
            # Get the pixel scale from the WCS at the specified position
            pixel_scales = self.spatial_wcs.proj_plane_pixel_scales()  # Returns scales in degrees/pixel

            # For SPHEREx, pixels should be roughly square, so take the geometric mean
            pixel_scale_deg = float(np.sqrt(pixel_scales[0] * pixel_scales[1]).value)

            # Convert from degrees to arcseconds
            pixel_scale_arcsec = pixel_scale_deg * 3600.0

            logger.debug(f"WCS pixel scale at ({x:.1f}, {y:.1f}): {pixel_scale_arcsec:.3f} arcsec/pixel")

            return pixel_scale_arcsec

        except Exception as e:
            logger.warning(f"Failed to calculate pixel scale from WCS: {e}. Using fallback {fallback} arcsec/pixel")
            return fallback

    # ==================== PSF Extraction Methods ====================

    def extract_psf_at_position(self, x: float, y: float) -> np.ndarray:
        """
        Extract PSF at a specific pixel position by finding the closest PSF zone.

        SPHEREx PSF cube structure:
        - 121 spatial zones distributed across the detector
        - Zone centers defined by XCTR_i and YCTR_i keywords in PSF header
        - Each PSF is 101×101 pixels (oversampled by OVERSAMP factor, typically 10)

        This method:
        1. Converts pixel position to parent image coordinates (if cutout)
        2. Loads PSF zone centers from PSF header (XCTR_*, YCTR_*)
        3. Finds the zone with minimum distance to the position
        4. Returns the corresponding PSF from the cube

        Parameters
        ----------
        x, y : float
            Pixel coordinates (0-based) on current image where PSF is needed

        Returns
        -------
        np.ndarray
            PSF array (101×101 or other size based on header) at the specified position
        """
        # Check if this is a cutout image by looking for CRPIX1A/CRPIX2A
        # These keywords give the pixel position of the cutout center on the parent image
        if "CRPIX1A" in self.header and "CRPIX2A" in self.header:
            # This is a cutout - need to translate to parent image coordinates
            crpix1a = self.header["CRPIX1A"]
            crpix2a = self.header["CRPIX2A"]

            # Convert cutout pixel coordinates (0-based) to parent image coordinates (0-based)
            # FITS convention: CRPIX is 1-based, so we add 1, subtract CRPIX, then get 0-based
            x_parent = x + 1 - crpix1a  # Now in FITS 1-based system
            y_parent = y + 1 - crpix2a  # Now in FITS 1-based system

            logger.debug(
                f"Cutout detected: pixel ({x:.1f}, {y:.1f}) on cutout → "
                f"({x_parent:.1f}, {y_parent:.1f}) on parent image (1-based)"
            )

            # Use parent coordinates for zone matching (in FITS 1-based convention)
            x_for_zone = x_parent
            y_for_zone = y_parent
        else:
            # Full image - convert 0-based to 1-based for matching with zone centers
            x_for_zone = x + 1
            y_for_zone = y + 1
            logger.debug(f"Full image: pixel ({x:.1f}, {y:.1f}) → ({x_for_zone:.1f}, {y_for_zone:.1f}) (1-based)")

        # Load PSF zone centers from header
        zone_centers = self._load_psf_zone_centers()

        if not zone_centers:
            raise ValueError("No PSF zone centers found in PSF header (XCTR_*/YCTR_* keywords missing)")

        # Find the zone with minimum distance to the position
        # Note: Zone centers are in FITS 1-based convention, need to subtract 1 for distance calc
        min_distance = float("inf")
        best_zone_id = None

        for zone_id, (xctr, yctr) in zone_centers.items():
            # Calculate distance (both in 1-based system, so subtract 1 from zone centers)
            distance = np.sqrt((xctr - 1 - (x_for_zone - 1)) ** 2 + (yctr - 1 - (y_for_zone - 1)) ** 2)

            if distance < min_distance:
                min_distance = distance
                best_zone_id = zone_id

        if best_zone_id is None:
            raise ValueError("Failed to find closest PSF zone")

        # Extract PSF from cube
        # Zone IDs are 1-based, but array indexing is 0-based
        psf_array = self.psf[best_zone_id - 1]

        logger.debug(
            f"Extracted PSF at pixel ({x:.1f}, {y:.1f}) from zone {best_zone_id} "
            f"(center: {zone_centers[best_zone_id]}, distance: {min_distance:.1f} pixels)"
        )

        return psf_array

    def get_psf_fwhm_estimate(self, x: float, y: float) -> float:
        """
        Estimate PSF FWHM at a given position using radial profile analysis.

        Uses photutils.profiles.RadialProfile to convert the 2D PSF to a 1D
        radial profile, then interpolates to find the half-maximum radius.

        Parameters
        ----------
        x, y : float
            Pixel coordinates (0-based)

        Returns
        -------
        float
            Estimated FWHM in arcseconds

        Raises
        ------
        Exception
            If FWHM estimation fails. Caller should handle the error and
            decide on fallback behavior.

        Notes
        -----
        PSF is 101×101 pixels with 10× oversampling. The PSF center is determined
        from the peak pixel position (not geometric center), and radial sampling
        is adapted to the PSF size. Uses constant extrapolation at radial profile
        boundaries for robust interpolation.
        """
        from photutils.profiles import RadialProfile

        psf = self.extract_psf_at_position(x, y)

        # Find PSF center from peak pixel position (not geometric center)
        # np.unravel_index returns (row, col) = (y, x) for 2D array
        peak_idx = np.unravel_index(psf.argmax(), psf.shape)
        center_y, center_x = peak_idx
        center_xy = (float(center_x), float(center_y))  # (x, y) order for photutils

        logger.debug(f"PSF peak at pixel ({center_x}, {center_y}), value = {psf[center_y, center_x]:.6e}")

        # Find peak value for half-maximum calculation
        peak_value = psf.max()
        half_max = peak_value / 2.0

        # Get PSF dimensions for max radius calculation
        psf_height, psf_width = psf.shape

        # Determine max radius from PSF size (use half the smaller dimension)
        max_radius = min(psf_height, psf_width) / 2.0

        # Create radial bins using linspace for even sampling
        # Use 500 points for smooth interpolation
        n_radial_bins = 500
        radii = np.linspace(0, max_radius, n_radial_bins)

        # Create 1D radial profile by averaging flux in concentric rings
        rp = RadialProfile(psf, center_xy, radii)
        radial_profile = rp.profile
        radial_bins = rp.radius

        # Find radius where profile = half_max using root-finding
        # This is more robust than np.interp as it doesn't require monotonicity
        from scipy.interpolate import interp1d
        from scipy.optimize import fsolve

        # Create interpolation function: f(r) -> intensity
        # Use edge values for constant extrapolation beyond radial_bins range
        fill_value = (radial_profile[0], radial_profile[-1])
        f_interp = interp1d(radial_bins, radial_profile, kind="quadratic", fill_value=fill_value, bounds_error=False)

        # Find root of: f(r) - half_max = 0
        # Initial guess: 10 PSF pixels (reasonable for typical PSF core)
        fwhm_radius_psf_pixels = fsolve(lambda r: f_interp(r) - half_max, x0=5.0)[0]

        # Convert from PSF pixels to arcseconds
        # PSF pixel scale is 1/10 of detector pixel scale (10× oversampling)
        fwhm_diameter_arcsec = fwhm_radius_psf_pixels * 2.0 * self.psf_pixel_scale

        logger.debug(
            f"Estimated PSF FWHM at ({x:.1f}, {y:.1f}): "
            f"{fwhm_diameter_arcsec:.3f} arcsec "
            f"(radius: {fwhm_radius_psf_pixels:.1f} PSF pixels)"
        )

        return fwhm_diameter_arcsec

    # ==================== Cutout Methods ====================

    def get_cutout(
        self,
        position: Union[Tuple[float, float], SkyCoord],
        size: Union[int, Tuple[int, int]],
        include_extensions: Optional[list] = None,
        mode: str = "trim",
    ) -> Dict[str, np.ndarray]:
        """
        Extract a cutout from the MEF data using astropy.nddata.Cutout2D.

        Returns a dictionary with cutout arrays and updated WCS. Uses Cutout2D
        for proper WCS handling, which automatically updates WCS keywords for
        the cutout region.

        Parameters
        ----------
        position : tuple of float or SkyCoord
            Position of cutout center. Can be either:
            - Tuple (x, y) of pixel coordinates (0-based)
            - SkyCoord object with RA/Dec
        size : int or tuple of int
            Size of cutout in pixels. If int, creates square cutout.
            If tuple (height, width), creates rectangular cutout.
            Note: Order is (height, width) to match numpy array shape convention.
        include_extensions : list of str, optional
            List of extensions to include in cutout.
            Default: ['image', 'flags', 'variance', 'zodi']
            Available: ['image', 'flags', 'variance', 'zodi', 'psf']
        mode : str, optional
            Mode for handling cutouts that extend beyond image boundaries.
            Options: 'trim' (default), 'partial', 'strict'
            See astropy.nddata.Cutout2D documentation for details.

        Returns
        -------
        dict
            Dictionary containing:
            - Requested extension cutouts (np.ndarray)
            - 'wcs': Updated WCS for the cutout region
            - 'position_original': Original position in parent image (x, y)
            - 'position_cutout': Position in cutout coordinates (x, y)
            - 'bbox_original': Bounding box in original image (BoundingBox object)
            - 'shape': Shape of cutout (height, width)

        Examples
        --------
        >>> # Square cutout using pixel coordinates
        >>> cutout = mef.get_cutout(position=(1020, 1020), size=50)
        >>> cutout_image = cutout['image']
        >>> cutout_wcs = cutout['wcs']
        >>>
        >>> # Rectangular cutout using SkyCoord
        >>> from astropy.coordinates import SkyCoord
        >>> import astropy.units as u
        >>> coord = SkyCoord(ra=304.5*u.deg, dec=42.3*u.deg)
        >>> cutout = mef.get_cutout(position=coord, size=(100, 50),
        ...                         include_extensions=['image', 'flags'])
        >>>
        >>> # Access cutout center in cutout pixel coordinates
        >>> center_x_cutout, center_y_cutout = cutout['position_cutout']
        """
        from astropy.nddata import Cutout2D

        if include_extensions is None:
            include_extensions = ["image", "flags", "variance", "zodi"]

        # Create cutout from image using Cutout2D (handles WCS automatically)
        try:
            cutout_obj = Cutout2D(self.image, position=position, size=size, wcs=self.spatial_wcs, mode=mode, copy=True)
        except Exception as e:
            logger.error(f"Failed to create cutout: {e}")
            raise

        # Build result dictionary with metadata and image cutout
        cutout_dict = {
            "wcs": cutout_obj.wcs,  # Updated WCS for cutout
            "position_original": cutout_obj.position_original,  # (x, y) in original image
            "position_cutout": cutout_obj.position_cutout,  # (x, y) in cutout
            "bbox_original": cutout_obj.bbox_original,  # BoundingBox in original image
            "shape": cutout_obj.shape,  # (height, width)
        }

        # Add image cutout if requested (reuse cutout_obj to avoid recalculating)
        if "image" in include_extensions:
            cutout_dict["image"] = cutout_obj.data

        # Extract cutouts for other extensions using the same bounding box
        # Use slices from cutout_obj to avoid recalculating Cutout2D
        extension_map = {
            "flags": self.flags,
            "variance": self.variance,
            "zodi": self.zodi,
        }

        for ext_name in include_extensions:
            if ext_name == "image":
                continue  # Already handled above
            elif ext_name in extension_map:
                # Use the bounding box from cutout_obj to slice directly
                cutout_ext = Cutout2D(
                    extension_map[ext_name], position=position, size=size, wcs=self.spatial_wcs, mode=mode, copy=True
                )
                cutout_dict[ext_name] = cutout_ext.data
            elif ext_name == "psf":
                # PSF is 3D and position-dependent across the detector
                # Include full PSF cube for cutouts
                cutout_dict["psf"] = self.psf.copy()
            else:
                logger.warning(f"Unknown extension '{ext_name}' requested for cutout, skipping")

        # Log cutout info
        bbox = cutout_obj.bbox_original
        logger.info(
            f"Extracted cutout: position={position}, size={size}, "
            f"shape={cutout_obj.shape}, "
            f"bbox=[{bbox.ixmin}:{bbox.ixmax}, {bbox.iymin}:{bbox.iymax}]"
        )

        return cutout_dict


# ==================== File I/O Functions ====================


def read_spherex_mef(filepath: Path, target_unit: Optional[str] = None) -> SPHERExMEF:
    """
    Read SPHEREx Multi-Extension FITS file with optional unit conversion.

    Parameters
    ----------
    filepath : Path
        Path to SPHEREx MEF file
    target_unit : str, optional
        Target unit for image data. Options:
        - None: Keep native MJy/sr (default)
        - 'uJy/arcsec2' or 'microJy/arcsec2': Convert to μJy/arcsec²
        - 'Jy/arcsec2': Convert to Jy/arcsec²
        - 'MJy/sr': No conversion (native)

    Returns
    -------
    SPHERExMEF
        Container with all MEF data. If unit conversion was applied,
        image, variance, and zodi arrays are in target units.

    Notes
    -----
    Conversion from MJy/sr to μJy/arcsec²:
    1 MJy/sr = 1e6 Jy/sr = 1e6 Jy / (206265 arcsec)² = 0.02350443 μJy/arcsec²
    Conversion factor: 1 MJy/sr × 0.02350443 = μJy/arcsec²
    """
    logger.info(f"Reading SPHEREx MEF: {filepath}")

    with fits.open(filepath) as hdulist:
        # Verify expected structure
        if len(hdulist) < 7:
            raise ValueError(f"Expected at least 7 extensions, got {len(hdulist)}")

        # Read IMAGE extension
        image_hdu = hdulist["IMAGE"]
        image_data = image_hdu.data.astype(np.float32)
        image_header = image_hdu.header

        # Verify units are as expected (MJy/sr)
        bunit = image_header.get("BUNIT", "").strip().upper()
        if bunit and bunit not in ["MJY/SR", "MJY / SR", "MJY SR-1", "MJY/STERADIAN"]:
            logger.warning(f"Unexpected BUNIT '{bunit}' in {filepath}. Expected 'MJy/sr'")
        elif bunit:
            logger.debug(f"Verified BUNIT: {bunit}")
        else:
            logger.warning(f"Missing BUNIT header in {filepath}. Assuming MJy/sr")

        # Read other extensions
        flags_data = hdulist["FLAGS"].data.astype(np.int32)
        variance_data = hdulist["VARIANCE"].data.astype(np.float32)
        zodi_data = hdulist["ZODI"].data.astype(np.float32)

        # Read PSF extension (both data and header for zone information)
        psf_data = hdulist["PSF"].data.astype(np.float32)
        psf_header = hdulist["PSF"].header

        # Load WCS with suppressed warnings about SCAMP/SIP distortion parameters
        with suppress_astropy_info():
            # Load spatial WCS (primary)
            spatial_wcs = WCS(image_header)

            # Load spectral WCS (alternative 'W')
            # Need to pass HDUList for lookup table access
            spectral_wcs = WCS(header=image_header, fobj=hdulist, key="W")
            # Disable SIP distortion for spectral WCS
            spectral_wcs.sip = None

        # Extract metadata
        obs_id = image_header.get("OBSID", filepath.stem)
        detector = image_header.get("DETECTOR", 0)

        # Calculate MJD
        t_min = image_header.get("MJD-BEG", 0)
        t_max = image_header.get("MJD-END", 0)
        mjd = (t_min + t_max) / 2.0

        # Apply unit conversion if requested
        native_unit = "MJy/sr"
        if target_unit is not None and target_unit.lower() not in ["mjy/sr", "mjy / sr"]:
            conversion_factor = _get_unit_conversion_factor(native_unit, target_unit)
            image_data = image_data * conversion_factor
            variance_data = variance_data * (conversion_factor**2)  # Variance scales as square
            zodi_data = zodi_data * conversion_factor
            final_unit = _normalize_unit_string(target_unit)
            logger.info(f"Converted units: {native_unit} → {final_unit} (factor: {conversion_factor:.6e})")
        else:
            final_unit = native_unit

        mef = SPHERExMEF(
            filepath=filepath,
            image=image_data,
            flags=flags_data,
            variance=variance_data,
            zodi=zodi_data,
            psf=psf_data,
            psf_header=psf_header,
            spatial_wcs=spatial_wcs,
            spectral_wcs=spectral_wcs,
            header=image_header,
            obs_id=obs_id,
            detector=detector,
            mjd=mjd,
            image_unit=final_unit,
            native_unit=native_unit,
        )

        logger.info(f"Loaded {obs_id}: detector {detector}, shape {image_data.shape}, units {final_unit}")

        return mef


def _get_unit_conversion_factor(from_unit: str, to_unit: str) -> float:
    """
    Get conversion factor between surface brightness units using astropy.units.

    Parameters
    ----------
    from_unit : str
        Source unit (expected: 'MJy/sr')
    to_unit : str
        Target unit

    Returns
    -------
    float
        Multiplication factor to convert from_unit to to_unit

    Raises
    ------
    ValueError
        If unit conversion is not supported or units are incompatible
    """
    # Normalize unit strings to astropy-compatible format
    from_unit_norm = from_unit.lower().replace(" ", "").replace("_", "")
    to_unit_norm = to_unit.lower().replace(" ", "").replace("_", "")

    # Map common variations to astropy unit strings
    unit_map = {
        "mjy/sr": "MJy/sr",
        "mjy/steradian": "MJy/sr",
        "ujy/arcsec2": "uJy/arcsec2",
        "microjy/arcsec2": "uJy/arcsec2",
        "jy/arcsec2": "Jy/arcsec2",
        "mjy/arcsec2": "MJy/arcsec2",
    }

    # Get standardized unit strings
    from_unit_std = unit_map.get(from_unit_norm)
    to_unit_std = unit_map.get(to_unit_norm)

    if from_unit_std is None:
        raise ValueError(f"Unrecognized source unit: '{from_unit}'. Expected 'MJy/sr' or similar.")

    if to_unit_std is None:
        raise ValueError(
            f"Unrecognized target unit: '{to_unit}'. Supported: uJy/arcsec2, Jy/arcsec2, MJy/arcsec2, MJy/sr"
        )

    # Use astropy.units for conversion
    try:
        from_quantity = 1.0 * u.Unit(from_unit_std)
        to_quantity = from_quantity.to(u.Unit(to_unit_std))
        conversion_factor = to_quantity.value

        logger.debug(f"Unit conversion factor: {from_unit_std} → {to_unit_std} = {conversion_factor:.6e}")

        return conversion_factor

    except Exception as e:
        raise ValueError(
            f"Failed to convert {from_unit} → {to_unit}: {e}. "
            f"Units may be incompatible (not both surface brightness units)."
        )


def _normalize_unit_string(unit: str) -> str:
    """Normalize unit string for display."""
    unit_lower = unit.lower().replace(" ", "")
    if unit_lower in ["ujy/arcsec2", "microjy/arcsec2"]:
        return "μJy/arcsec²"
    elif unit_lower == "jy/arcsec2":
        return "Jy/arcsec²"
    elif unit_lower == "mjy/arcsec2":
        return "MJy/arcsec²"
    elif unit_lower in ["mjy/sr", "mjy/sr"]:
        return "MJy/sr"
    else:
        return unit


# ==================== Legacy Compatibility Functions ====================
# These functions are kept for backward compatibility with existing code


def get_pixel_coordinates(mef: SPHERExMEF, ra: float, dec: float) -> Tuple[float, float]:
    """
    Convert RA/Dec to pixel coordinates (legacy function).

    Use mef.world_to_pixel() instead for new code.
    """
    return mef.world_to_pixel(ra, dec)


def get_wavelength_at_position(mef: SPHERExMEF, x: float, y: float) -> Tuple[float, float]:
    """
    Get wavelength and bandwidth at pixel position (legacy function).

    Use mef.pixel_to_wavelength() instead for new code.
    """
    return mef.pixel_to_wavelength(x, y)


def get_pixel_scale_at_position(wcs: WCS, x: float, y: float, pixel_scale_fallback: float = 6.2) -> float:
    """
    Calculate pixel scale at position (legacy function).

    Use mef.get_pixel_scale() instead for new code.
    """
    # This function is used by photometry.py, so keep it functional
    try:
        pixel_scales = wcs.proj_plane_pixel_scales()
        pixel_scale_deg = float(np.sqrt(pixel_scales[0] * pixel_scales[1]).value)
        pixel_scale_arcsec = pixel_scale_deg * 3600.0
        return pixel_scale_arcsec
    except Exception as e:
        logger.warning(f"Failed to calculate pixel scale from WCS: {e}. Using fallback {pixel_scale_fallback}")
        return pixel_scale_fallback


# ==================== Flag Handling Functions ====================


def get_flag_info(flag_value: int) -> Dict[str, bool]:
    """
    Decode flag bitmap into individual flags.

    Parameters
    ----------
    flag_value : int
        Combined flag bitmap value

    Returns
    -------
    Dict[str, bool]
        Dictionary of flag names and their states
    """
    # Flag definitions from SPHEREx
    flags = {
        "TRANSIENT": 0,
        "OVERFLOW": 1,
        "SUR_ERROR": 2,
        "NONFUNC": 6,
        "DICHROIC": 7,
        "MISSING_DATA": 9,
        "HOT": 10,
        "COLD": 11,
        "FULLSAMPLE": 12,
        "PHANMISS": 14,
        "NONLINEAR": 15,
        "PERSIST": 17,
        "OUTLIER": 19,
        "SOURCE": 21,
    }

    flag_states = {}
    for name, bit in flags.items():
        flag_states[name] = bool(flag_value & (1 << bit))

    return flag_states


def format_flag_binary(flag_value: int, num_bits: int = 22) -> str:
    """
    Format flag value as binary string.

    Parameters
    ----------
    flag_value : int
        Flag bitmap value
    num_bits : int
        Number of bits to display

    Returns
    -------
    str
        Binary string representation
    """
    return format(flag_value, f"0{num_bits}b")


def create_background_mask(flags: np.ndarray, exclude_source: bool = True) -> np.ndarray:
    """
    Create mask for background pixels (good for zodiacal matching).

    Masks out pixels with problematic flags including non-functional pixels,
    outliers, etc. By default, SOURCE-flagged pixels are kept as valid background
    pixels for local background estimation, but this can be changed.

    Parameters
    ----------
    flags : np.ndarray
        Flag bitmap array
    exclude_source : bool, optional
        If True, also exclude SOURCE-flagged pixels (bit 21) from the mask.
        If False (default), SOURCE pixels are kept for local background estimation.

    Returns
    -------
    np.ndarray
        Boolean mask (True = good background pixel)
    """
    # Define flags that should be masked out for background estimation
    bad_flags = {
        "TRANSIENT": 0,  # Transient detections
        "OVERFLOW": 1,  # Overflow pixels
        "SUR_ERROR": 2,  # Processing errors
        "NONFUNC": 6,  # Non-functional pixels
        "DICHROIC": 7,  # Dichroic edge effects
        "MISSING_DATA": 9,  # Missing data
        "HOT": 10,  # Hot pixels
        "COLD": 11,  # Anomalously low signal
        "PHANMISS": 14,  # Phantom correction missing
        "NONLINEAR": 15,  # Nonlinearity issues
        "PERSIST": 17,  # Persistent charge
        "OUTLIER": 19,  # Statistical outliers
    }
    # Optionally include SOURCE flag (bit 21)
    if exclude_source:
        bad_flags["SOURCE"] = 21

    # Create combined mask
    mask = np.ones(flags.shape, dtype=bool)  # Start with all good

    for flag_name, bit in bad_flags.items():
        flag_mask = (flags & (1 << bit)) != 0
        mask &= ~flag_mask  # Remove flagged pixels
        logger.debug(f"Masked {np.sum(flag_mask)} pixels for {flag_name}")

    n_good = np.sum(mask)
    n_total = mask.size
    logger.info(f"Background mask: {n_good}/{n_total} ({n_good / n_total * 100:.1f}%) pixels available")

    return mask


# ==================== Zodiacal Background Subtraction ====================


def estimate_zodiacal_scaling(
    image: np.ndarray, zodi: np.ndarray, mask: np.ndarray, variance: Optional[np.ndarray] = None
) -> float:
    """
    Estimate scaling factor to match zodiacal model to observed background.

    Uses least-squares fitting on uncontaminated pixels to find the
    multiplicative factor that best matches the zodi model to the data.

    Parameters
    ----------
    image : np.ndarray
        Observed image data
    zodi : np.ndarray
        Zodiacal model
    mask : np.ndarray
        Boolean mask (True = good background pixels)
    variance : np.ndarray, optional
        Variance array for weighted fitting

    Returns
    -------
    float
        Scaling factor for zodiacal model
    """
    # Extract background pixels
    image_bg = image[mask]
    zodi_bg = zodi[mask]

    if len(image_bg) == 0:
        logger.warning("No uncontaminated pixels for zodiacal scaling - using factor 1.0")
        return 1.0

    # Remove pixels where zodi model is zero to avoid division issues
    nonzero_mask = zodi_bg != 0
    if np.sum(nonzero_mask) == 0:
        logger.warning("Zodiacal model is zero everywhere - using factor 1.0")
        return 1.0

    image_bg = image_bg[nonzero_mask]
    zodi_bg = zodi_bg[nonzero_mask]

    # Weighted least squares if variance is provided
    if variance is not None:
        var_bg = variance[mask][nonzero_mask]
        # Avoid zero/negative variance
        valid_var = var_bg > 0
        if np.sum(valid_var) > 0:
            weights = 1.0 / var_bg[valid_var]
            image_bg = image_bg[valid_var]
            zodi_bg = zodi_bg[valid_var]

            # Weighted least squares: scale = sum(w*img*zodi) / sum(w*zodi^2)
            scale_factor = np.sum(weights * image_bg * zodi_bg) / np.sum(weights * zodi_bg**2)
        else:
            # Fall back to unweighted
            scale_factor = np.sum(image_bg * zodi_bg) / np.sum(zodi_bg**2)
    else:
        # Unweighted least squares: scale = sum(img*zodi) / sum(zodi^2)
        scale_factor = np.sum(image_bg * zodi_bg) / np.sum(zodi_bg**2)

    logger.info(f"Zodiacal scaling factor: {scale_factor:.4f}")

    return scale_factor


def subtract_zodiacal_background(
    image: np.ndarray,
    zodi: np.ndarray,
    flags: np.ndarray,
    variance: Optional[np.ndarray] = None,
    zodi_scale_min: float = 0.0,
    zodi_scale_max: float = 10.0,
) -> Tuple[np.ndarray, float]:
    """
    Subtract zodiacal light background from image with amplitude scaling.

    Uses uncontaminated background pixels to determine the optimal
    scaling factor for the zodiacal model before subtraction.

    Parameters
    ----------
    image : np.ndarray
        Original image
    zodi : np.ndarray
        Zodiacal light model
    flags : np.ndarray
        Flag bitmap array
    variance : np.ndarray, optional
        Variance array for weighted fitting
    zodi_scale_min : float
        Minimum allowed zodiacal scaling factor
    zodi_scale_max : float
        Maximum allowed zodiacal scaling factor

    Returns
    -------
    corrected_image : np.ndarray
        Background-subtracted image
    scale_factor : float
        Applied scaling factor for the zodiacal model
    """
    # Create mask for background estimation
    bg_mask = create_background_mask(flags)

    # Check if enough pixels are available for zodiacal estimation
    n_bg_pixels = np.sum(bg_mask)
    n_total_pixels = bg_mask.size
    bg_fraction = n_bg_pixels / n_total_pixels if n_total_pixels > 0 else 0.0

    if bg_fraction < 0.5:
        logger.warning(
            f"Insufficient background pixels for zodiacal estimation "
            f"({n_bg_pixels}/{n_total_pixels} = {bg_fraction * 100:.1f}% < 50%) - using fallback scale factor 1.0"
        )
        scale_factor = 1.0
    else:
        # Estimate zodiacal scaling factor
        scale_factor = estimate_zodiacal_scaling(image, zodi, bg_mask, variance)

        # Validate scale factor
        if scale_factor <= zodi_scale_min or scale_factor > zodi_scale_max:
            logger.warning(
                f"Unusual scaling factor {scale_factor:.4f} (outside [{zodi_scale_min}, {zodi_scale_max}]) - using 1.0"
            )
            scale_factor = 1.0

    # Apply scaled subtraction
    corrected_image = image - (scale_factor * zodi)

    logger.info(f"Subtracted zodiacal background with scaling factor {scale_factor:.4f}")

    return corrected_image, scale_factor
