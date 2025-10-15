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