"""
Helper utility functions for SPXQuery package.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save dictionary to JSON file.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Data to save
    filepath : Path
        Output file path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug(f"Saved JSON to {filepath}")


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Parameters
    ----------
    filepath : Path
        Input file path
    
    Returns
    -------
    Dict[str, Any]
        Loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.debug(f"Loaded JSON from {filepath}")
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
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
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
    pattern = r'^(\d+\.?\d*)(,\d+\.?\d*)?(px|pix|pixels|arcsec|arcmin|deg|rad)?$'

    match = re.match(pattern, size_str.strip())
    if not match:
        return False

    # Extract values and check they're positive
    values = size_str.split(',')
    try:
        # Remove units from the last value if present
        last_val = values[-1]
        for unit in ['px', 'pix', 'pixels', 'arcsec', 'arcmin', 'deg', 'rad']:
            if last_val.endswith(unit):
                last_val = last_val[:-len(unit)]
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
    pattern = r'^(-?\d+\.?\d*),(-?\d+\.?\d*)(px|pix|pixels|deg|rad|arcsec|arcmin)?$'

    match = re.match(pattern, center_str.strip())
    if not match:
        return False

    # Extract and validate coordinates
    try:
        coords = center_str.split(',')
        x_str = coords[0].strip()
        y_str = coords[1].strip()

        # Remove units from y if present
        for unit in ['px', 'pix', 'pixels', 'deg', 'rad', 'arcsec', 'arcmin']:
            if y_str.endswith(unit):
                y_str = y_str[:-len(unit)]
                break

        x = float(x_str)
        y = float(y_str)

        # If units are degrees (default or explicit), validate Dec range
        if 'px' not in center_str and 'pix' not in center_str:
            # Assume degrees, check Dec in [-90, 90]
            # Y is declination in astronomical coordinates
            if not -90 <= y <= 90:
                logger.warning(f"Declination {y} outside valid range [-90, 90]")
                return False

        return True

    except (ValueError, IndexError):
        return False


def format_cutout_url_params(
    cutout_size: str | None,
    cutout_center: str | None,
    source_ra: float,
    source_dec: float
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


def estimate_cutout_size_mb(
    cutout_size: str | None,
    full_size_mb: float = 70.0
) -> float:
    """
    Estimate cutout file size based on pixel dimensions.

    Assumes full SPHEREx image is 2040x2040 pixels (~70 MB).
    Estimates cutout size proportional to pixel area.

    Parameters
    ----------
    cutout_size : str or None
        Size parameter (e.g., "200px", "500,600px")
        If None or cannot parse, returns full_size_mb
    full_size_mb : float
        Size of full image in MB (default 70.0 for SPHEREx)

    Returns
    -------
    float
        Estimated cutout size in MB

    Examples
    --------
    >>> estimate_cutout_size_mb("200px")
    0.68  # approximately (200*200)/(2040*2040) * 70

    >>> estimate_cutout_size_mb(None)
    70.0
    """
    if not cutout_size:
        return full_size_mb

    try:
        # Parse pixel dimensions
        # Only estimate for pixel units (angular units depend on pixel scale)
        if 'px' not in cutout_size.lower() and 'pix' not in cutout_size.lower():
            logger.debug(f"Cannot estimate size for angular units: {cutout_size}")
            return full_size_mb

        # Remove units
        size_str = cutout_size.lower()
        for unit in ['pixels', 'pixel', 'pix', 'px']:
            size_str = size_str.replace(unit, '')

        # Parse dimensions
        dims = [float(x.strip()) for x in size_str.split(',')]

        if len(dims) == 1:
            # Square cutout
            cutout_pixels = dims[0] * dims[0]
        elif len(dims) == 2:
            # Rectangular cutout
            cutout_pixels = dims[0] * dims[1]
        else:
            return full_size_mb

        # Calculate size ratio
        full_pixels = 2040 * 2040  # SPHEREx image size
        size_ratio = cutout_pixels / full_pixels

        estimated_size = full_size_mb * size_ratio

        logger.debug(f"Estimated cutout size: {estimated_size:.2f} MB for {cutout_size}")

        return estimated_size

    except (ValueError, AttributeError) as e:
        logger.warning(f"Could not estimate cutout size for '{cutout_size}': {e}")
        return full_size_mb