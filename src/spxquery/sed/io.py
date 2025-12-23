"""
Save/load utilities for SED reconstruction results.

This module provides top-level convenience functions for saving and loading
SED reconstruction results with proper folder structure.
"""

import logging
from pathlib import Path
from typing import Union

from .data_structures import EnsembleResult, SEDReconstructionResult

logger = logging.getLogger(__name__)


def sed_save_all(
    output_dir: Union[str, Path],
    reconstruction_result: Union[SEDReconstructionResult, EnsembleResult],
    save_members: bool = False,
) -> None:
    """
    Save complete SED reconstruction results with proper folder structure.

    This is a convenience function that delegates to the appropriate class method
    based on the result type (single or ensemble reconstruction).

    Parameters
    ----------
    output_dir : Path or str
        Directory to save results.
    reconstruction_result : SEDReconstructionResult or EnsembleResult
        Reconstruction result to save.
    save_members : bool, optional
        For EnsembleResult only: If True, save individual ensemble member results
        to results/members/ folder. Default: False.

    Examples
    --------
    >>> from spxquery.sed import sed_save_all
    >>> sed_save_all("output/my_reconstruction", result)

    For ensemble with member results:
    >>> sed_save_all("output/ensemble_reconstruction", ensemble_result, save_members=True)
    """
    output_dir = Path(output_dir)

    if isinstance(reconstruction_result, EnsembleResult):
        logger.info("Saving ensemble reconstruction results...")
        reconstruction_result.save_all(output_dir, save_members=save_members)
    elif isinstance(reconstruction_result, SEDReconstructionResult):
        logger.info("Saving single SED reconstruction results...")
        reconstruction_result.save_all(output_dir)
    else:
        raise TypeError(f"Expected SEDReconstructionResult or EnsembleResult, got {type(reconstruction_result)}")

    logger.info(f"Successfully saved results to {output_dir}")


def sed_load_all(output_dir: Union[str, Path]) -> Union[SEDReconstructionResult, EnsembleResult]:
    """
    Load SED reconstruction results from saved directory.

    This function automatically detects whether the saved data corresponds to
    a single SEDReconstructionResult or an EnsembleResult by checking the
    ensemble_size attribute in the config.

    Parameters
    ----------
    output_dir : Path or str
        Directory containing saved reconstruction results.

    Returns
    -------
    SEDReconstructionResult or EnsembleResult
        Loaded reconstruction result.

    Raises
    ------
    FileNotFoundError
        If required files or directories are missing.

    Examples
    --------
    >>> from spxquery.sed import sed_load_all
    >>> result = sed_load_all("output/my_reconstruction")
    >>> print(result.flux)
    """
    output_dir = Path(output_dir)

    # Check config to determine result type
    config_path = output_dir / "config" / "sed_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Import here to avoid circular dependency
    from .config import SEDConfig

    config = SEDConfig.from_yaml_file(config_path)
    ensemble_size = getattr(config, "ensemble_size", 1)

    # Delegate to appropriate class method
    if ensemble_size > 1:
        logger.info(f"Loading ensemble reconstruction (ensemble_size={ensemble_size})...")
        result = EnsembleResult.load_all(output_dir)
    else:
        logger.info("Loading single SED reconstruction...")
        result = SEDReconstructionResult.load_all(output_dir)

    logger.info(f"Successfully loaded results from {output_dir}")

    return result
