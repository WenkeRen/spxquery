"""
Helper utility functions for SPXQuery package.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def save_yaml(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save dictionary to YAML file.

    Parameters
    ----------
    data : Dict[str, Any]
        Data to save
    filepath : Path
        Output file path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.debug(f"Saved YAML to {filepath}")


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """
    Load dictionary from YAML file.

    Parameters
    ----------
    filepath : Path
        Input file path

    Returns
    -------
    Dict[str, Any]
        Loaded data
    """
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    logger.debug(f"Loaded YAML from {filepath}")
    return data


def format_file_size(size_bytes: float) -> str:
    """
    Format file size in human-readable format.

    Parameters
    ----------
    size_bytes : float
        Size in bytes

    Returns
    -------
    str
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def validate_directory(path: Path, create: bool = True) -> bool:
    """
    Validate and optionally create directory.

    Parameters
    ----------
    path : Path
        Directory path
    create : bool
        Whether to create directory if it doesn't exist

    Returns
    -------
    bool
        True if directory exists or was created
    """
    if path.exists():
        if not path.is_dir():
            logger.error(f"{path} exists but is not a directory")
            return False
        return True

    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False

    return False


def get_file_list(directory: Path, pattern: str = "*.fits") -> list[Path]:
    """
    Get list of files matching pattern in directory.

    Parameters
    ----------
    directory : Path
        Directory to search
    pattern : str
        Glob pattern for files

    Returns
    -------
    list[Path]
        List of matching file paths
    """
    if not directory.exists():
        return []

    files = sorted(directory.rglob(pattern))
    return files


# Cutout-related helper functions


def validate_cutout_size(size_str: str) -> bool:
    """
    Validate cutout size parameter format.

    Valid formats:
    - Single value: "200", "0.1", "3.5"
    - Two values: "100,200", "0.5,1.0"
    - With units: "200px", "100,200pixels", "3arcmin", "0.1deg"

    Parameters
    ----------
    size_str : str
        Size parameter string

    Returns
    -------
    bool
        True if format is valid
    """
    import re

    if not size_str or not isinstance(size_str, str):
        return False

    # Pattern: number[,number][units]
    # Units: px, pix, pixels, arcsec, arcmin, deg, rad
    pattern = r"^(\d+\.?\d*)(,\d+\.?\d*)?(px|pix|pixels|arcsec|arcmin|deg|rad)?$"

    match = re.match(pattern, size_str.strip())
    if not match:
        return False

    # Extract values and check they're positive
    values = size_str.split(",")
    try:
        # Remove units from the last value if present
        last_val = values[-1]
        for unit in ["px", "pix", "pixels", "arcsec", "arcmin", "deg", "rad"]:
            if last_val.endswith(unit):
                last_val = last_val[: -len(unit)]
                break
        values[-1] = last_val

        # Check all values are positive numbers
        for val in values:
            num = float(val.strip())
            if num <= 0:
                return False
    except (ValueError, IndexError):
        return False

    return True


def validate_cutout_center(center_str: str) -> bool:
    """
    Validate cutout center parameter format.

    Valid formats:
    - Degrees (default): "70,20", "304.5,42.3"
    - Pixels: "1020,1020px", "500,600pixels"
    - Other angular: "1.5,0.8rad", "304.5,42.3deg"

    Parameters
    ----------
    center_str : str
        Center parameter string

    Returns
    -------
    bool
        True if format is valid
    """
    import re

    if not center_str or not isinstance(center_str, str):
        return False

    # Pattern: number,number[units]
    # Units: px, pix, pixels, deg, rad, arcsec, arcmin (though arcsec/arcmin unusual for center)
    pattern = r"^(-?\d+\.?\d*),(-?\d+\.?\d*)(px|pix|pixels|deg|rad|arcsec|arcmin)?$"

    match = re.match(pattern, center_str.strip())
    if not match:
        return False

    # Extract and validate coordinates
    try:
        coords = center_str.split(",")
        x_str = coords[0].strip()
        y_str = coords[1].strip()

        # Remove units from y if present
        for unit in ["px", "pix", "pixels", "deg", "rad", "arcsec", "arcmin"]:
            if y_str.endswith(unit):
                y_str = y_str[: -len(unit)]
                break

        x = float(x_str)
        y = float(y_str)

        # If units are degrees (default or explicit), validate Dec range
        if "px" not in center_str and "pix" not in center_str:
            # Assume degrees, check Dec in [-90, 90]
            # Y is declination in astronomical coordinates
            if not -90 <= y <= 90:
                logger.warning(f"Declination {y} outside valid range [-90, 90]")
                return False

        return True

    except (ValueError, IndexError):
        return False


def format_cutout_url_params(
    cutout_size: str | None, cutout_center: str | None, source_ra: float, source_dec: float
) -> str:
    """
    Format cutout parameters as URL query string.

    If cutout_size is None, returns empty string (no cutout).
    If cutout_size is specified but cutout_center is None, uses source position.
    If both specified, uses provided center.

    Parameters
    ----------
    cutout_size : str or None
        Size parameter (e.g., "200px", "3arcmin")
    cutout_center : str or None
        Center parameter (e.g., "70,20") or None to use source position
    source_ra : float
        Source RA in degrees (used if cutout_center is None)
    source_dec : float
        Source Dec in degrees (used if cutout_center is None)

    Returns
    -------
    str
        URL query string (e.g., "?size=200px" or "?center=70,20&size=200px")
        Empty string if cutout_size is None

    Examples
    --------
    >>> format_cutout_url_params("200px", None, 304.69, 42.44)
    '?center=304.69,42.44&size=200px'

    >>> format_cutout_url_params("3arcmin", "70,20", 304.69, 42.44)
    '?center=70,20&size=3arcmin'

    >>> format_cutout_url_params(None, None, 304.69, 42.44)
    ''
    """
    if not cutout_size:
        return ""

    # Use source position if center not specified
    if not cutout_center:
        cutout_center = f"{source_ra},{source_dec}"

    # Format URL parameters
    # Always include center for clarity, even if it matches source position
    params = f"?center={cutout_center}&size={cutout_size}"

    return params


def estimate_cutout_size_mb(cutout_size: str | None, full_size_mb: float = 71.6, min_size_mb: float = 5.0) -> float:
    """
    Estimate cutout file size based on dimensions.

    Assumes full SPHEREx image is 2040x2040 pixels (~71.6 MB).
    Estimates cutout size proportional to pixel area, with minimum 5 MB
    due to auxiliary data (PSF, WCS, etc.) that's always included.

    SPHEREx pixel scale: ~6.2 arcsec/pixel

    Parameters
    ----------
    cutout_size : str or None
        Size parameter (e.g., "200px", "3arcmin", "0.1deg")
        If None, returns full_size_mb
    full_size_mb : float
        Size of full image in MB (default 71.6 for SPHEREx)
    min_size_mb : float
        Minimum cutout size in MB (default 5.0, due to auxiliary data)

    Returns
    -------
    float
        Estimated cutout size in MB

    Examples
    --------
    >>> estimate_cutout_size_mb("200px")
    5.0  # minimum size due to auxiliary data

    >>> estimate_cutout_size_mb("500px")
    6.1  # (500*500)/(2040*2040) * 71.6 ≈ 4.3, but min 5.0

    >>> estimate_cutout_size_mb("1000px")
    17.2  # (1000*1000)/(2040*2040) * 71.6

    >>> estimate_cutout_size_mb("3arcmin")
    9.7  # 3 arcmin ≈ 29 pixels, but minimum 5 MB applies

    >>> estimate_cutout_size_mb(None)
    71.6  # full image
    """
    if not cutout_size:
        return full_size_mb

    try:
        # Convert angular units to pixels
        # SPHEREx pixel scale: ~6.2 arcsec/pixel
        pixel_scale_arcsec = 6.2

        size_str = cutout_size.lower().strip()

        # Parse dimensions and units
        if "arcsec" in size_str:
            # Remove unit and parse
            value_str = size_str.replace("arcsec", "").strip()
            values = [float(x.strip()) for x in value_str.split(",")]
            # Convert arcsec to pixels
            dims_pixels = [v / pixel_scale_arcsec for v in values]
        elif "arcmin" in size_str:
            value_str = size_str.replace("arcmin", "").strip()
            values = [float(x.strip()) for x in value_str.split(",")]
            # Convert arcmin to pixels (1 arcmin = 60 arcsec)
            dims_pixels = [v * 60.0 / pixel_scale_arcsec for v in values]
        elif "deg" in size_str:
            value_str = size_str.replace("deg", "").strip()
            values = [float(x.strip()) for x in value_str.split(",")]
            # Convert degrees to pixels (1 deg = 3600 arcsec)
            dims_pixels = [v * 3600.0 / pixel_scale_arcsec for v in values]
        elif "rad" in size_str:
            value_str = size_str.replace("rad", "").strip()
            values = [float(x.strip()) for x in value_str.split(",")]
            # Convert radians to pixels (1 rad = 206265 arcsec)
            dims_pixels = [v * 206265.0 / pixel_scale_arcsec for v in values]
        elif "px" in size_str or "pix" in size_str:
            # Already in pixels
            for unit in ["pixels", "pixel", "pix", "px"]:
                size_str = size_str.replace(unit, "")
            dims_pixels = [float(x.strip()) for x in size_str.split(",")]
        else:
            # No units specified, assume degrees (IRSA default)
            values = [float(x.strip()) for x in size_str.split(",")]
            dims_pixels = [v * 3600.0 / pixel_scale_arcsec for v in values]

        # Calculate cutout area
        if len(dims_pixels) == 1:
            # Square cutout
            cutout_pixels = dims_pixels[0] * dims_pixels[0]
        elif len(dims_pixels) == 2:
            # Rectangular cutout
            cutout_pixels = dims_pixels[0] * dims_pixels[1]
        else:
            logger.warning(f"Invalid dimensions in cutout_size: {cutout_size}")
            return full_size_mb

        # Calculate size ratio
        full_pixels = 2040 * 2040  # SPHEREx full image size
        size_ratio = cutout_pixels / full_pixels

        # Estimate based on pixel ratio
        estimated_size = full_size_mb * size_ratio

        # Apply minimum size constraint (auxiliary data: PSF, WCS, headers)
        final_size = max(estimated_size, min_size_mb)

        logger.debug(
            f"Estimated cutout size: {final_size:.2f} MB for {cutout_size} "
            f"({cutout_pixels:.0f} pixels, ratio={size_ratio:.4f})"
        )

        return final_size

    except (ValueError, AttributeError) as e:
        logger.warning(f"Could not estimate cutout size for '{cutout_size}': {e}")
        return full_size_mb


# Quality control helper functions


def create_flag_mask(bad_flags: list[int]) -> int:
    """
    Convert list of bad flag bit positions to a single integer mask.

    Parameters
    ----------
    bad_flags : list[int]
        List of bad flag bit positions

    Returns
    -------
    int
        Integer mask with all bad flag bits set

    Examples
    --------
    >>> create_flag_mask([0, 1, 2])
    7  # 0b0111

    >>> create_flag_mask([0, 2, 4])
    21  # 0b10101
    """
    mask = 0
    for bit in bad_flags:
        mask |= 1 << bit
    return mask


def check_flag_bits(flag: int, bad_flags_mask: int) -> bool:
    """
    Check if any bad flag bits are set in the given flag value.

    Parameters
    ----------
    flag : int
        Combined flag bitmap value
    bad_flags_mask : int
        Mask with bad flag bits set (created by create_flag_mask)

    Returns
    -------
    bool
        True if any bad flags are set, False otherwise

    Examples
    --------
    >>> mask = create_flag_mask([0, 1, 2])
    >>> check_flag_bits(0b0000, mask)  # No flags set
    False

    >>> check_flag_bits(0b0001, mask)  # Bit 0 is set
    True

    >>> check_flag_bits(0b1000, mask)  # Only bit 3 is set (not in bad mask)
    False
    """
    return (flag & bad_flags_mask) != 0


def apply_quality_filters(
    photometry_results: list, sigma_threshold: float = 5.0, bad_flags: list[int] | None = None
) -> tuple[list, dict]:
    """
    Apply quality control filters to photometry results.

    Filters based on:
    1. SNR threshold: flux/flux_err >= sigma_threshold
    2. Flag rejection: reject points with any bad flags set

    Parameters
    ----------
    photometry_results : list[PhotometryResult]
        Input photometry measurements
    sigma_threshold : float
        Minimum SNR (flux/flux_err) to accept (default: 5.0)
    bad_flags : list[int], optional
        List of bad flag bit positions to reject
        Default: [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19]

    Returns
    -------
    filtered_results : list[PhotometryResult]
        Filtered photometry results
    filter_stats : dict
        Statistics about filtering (rejected counts by reason)

    Examples
    --------
    >>> from spxquery.core.config import PhotometryResult
    >>> results = [
    ...     PhotometryResult(..., flux=10, flux_error=1, flag=0),  # Good
    ...     PhotometryResult(..., flux=10, flux_error=5, flag=0),  # Low SNR
    ...     PhotometryResult(..., flux=10, flux_error=1, flag=0b0001),  # Bad flag
    ... ]
    >>> filtered, stats = apply_quality_filters(results, sigma_threshold=5.0)
    >>> len(filtered)
    1
    """
    if bad_flags is None:
        bad_flags = [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19]

    # Convert bad_flags list to mask once for efficient checking
    bad_flags_mask = create_flag_mask(bad_flags)

    filtered = []
    rejected_snr = 0
    rejected_flag = 0
    rejected_both = 0

    for result in photometry_results:
        # Calculate SNR
        if result.flux_error > 0:
            snr = result.flux / result.flux_error
        else:
            snr = 0.0

        # Check filters
        fails_snr = snr < sigma_threshold
        fails_flag = check_flag_bits(result.flag, bad_flags_mask)

        if fails_snr and fails_flag:
            rejected_both += 1
        elif fails_snr:
            rejected_snr += 1
        elif fails_flag:
            rejected_flag += 1
        else:
            # Passed all filters
            filtered.append(result)

    filter_stats = {
        "total_input": len(photometry_results),
        "total_output": len(filtered),
        "rejected_snr": rejected_snr,
        "rejected_flag": rejected_flag,
        "rejected_both": rejected_both,
        "total_rejected": rejected_snr + rejected_flag + rejected_both,
        "sigma_threshold": sigma_threshold,
        "bad_flags": bad_flags,
        "bad_flags_mask": bad_flags_mask,
    }

    logger.info(
        f"Quality filtering: {len(photometry_results)} -> {len(filtered)} measurements "
        f"({filter_stats['total_rejected']} rejected: {rejected_snr} SNR, {rejected_flag} flags, {rejected_both} both)"
    )

    return filtered, filter_stats


@dataclass
class ClassifiedPhotometry:
    """Quality-classified photometry points."""

    good_regular: List
    rejected_regular: List
    good_upper_limits: List
    rejected_upper_limits: List


def classify_photometry_by_quality(
    photometry_results: List,
    sigma_threshold: float,
    bad_flags_mask: Optional[int] = None,
    separate_upper_limits: bool = True,
) -> ClassifiedPhotometry:
    """
    Classify photometry points as good/rejected based on SNR and flag filters.

    This function separates photometry measurements into four categories:
    - Good regular measurements (SNR >= threshold, no bad flags, not upper limit)
    - Rejected regular measurements (SNR < threshold or has bad flags, not upper limit)
    - Good upper limits (SNR >= threshold, no bad flags, is upper limit)
    - Rejected upper limits (SNR < threshold or has bad flags, is upper limit)

    Parameters
    ----------
    photometry_results : List[PhotometryResult]
        Photometry measurements to classify
    sigma_threshold : float
        Minimum SNR (flux/flux_err) threshold
    bad_flags_mask : int, optional
        Integer mask with bad flag bits set (from create_flag_mask)
        If None, no flag filtering is applied
    separate_upper_limits : bool
        If True, separate upper limits from regular measurements

    Returns
    -------
    ClassifiedPhotometry
        Classified photometry points in four categories

    Examples
    --------
    >>> from spxquery.core.config import PhotometryResult
    >>> from spxquery.utils.helpers import create_flag_mask, classify_photometry_by_quality
    >>>
    >>> # Create sample data
    >>> results = [...]  # List of PhotometryResult objects
    >>>
    >>> # Create flag mask for bad flags
    >>> bad_flags_mask = create_flag_mask([0, 1, 2, 6, 7, 9, 10, 11, 15])
    >>>
    >>> # Classify photometry
    >>> classified = classify_photometry_by_quality(
    ...     results,
    ...     sigma_threshold=5.0,
    ...     bad_flags_mask=bad_flags_mask,
    ...     separate_upper_limits=True
    ... )
    >>>
    >>> print(f"Good regular: {len(classified.good_regular)}")
    >>> print(f"Rejected regular: {len(classified.rejected_regular)}")
    """
    # Initialize classification lists
    good_regular = []
    rejected_regular = []
    good_upper_limits = []
    rejected_upper_limits = []

    for p in photometry_results:
        # Calculate SNR
        snr = p.flux / p.flux_error if p.flux_error > 0 else 0.0

        # Check filters
        fails_snr = snr < sigma_threshold
        fails_flag = check_flag_bits(p.flag, bad_flags_mask) if bad_flags_mask is not None else False
        is_rejected = fails_snr or fails_flag

        # Classify based on quality and upper limit status
        if separate_upper_limits and p.is_upper_limit:
            if is_rejected:
                rejected_upper_limits.append(p)
            else:
                good_upper_limits.append(p)
        else:
            if is_rejected:
                rejected_regular.append(p)
            else:
                good_regular.append(p)

    return ClassifiedPhotometry(
        good_regular=good_regular,
        rejected_regular=rejected_regular,
        good_upper_limits=good_upper_limits,
        rejected_upper_limits=rejected_upper_limits,
    )
