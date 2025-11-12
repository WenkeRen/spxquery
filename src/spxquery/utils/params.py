"""
Utilities for exporting and importing advanced parameter templates.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..core.config import AdvancedConfig

logger = logging.getLogger(__name__)


def export_default_parameters(output_path: Union[str, Path], filename: str = "spxquery_default_params.yaml") -> Path:
    """
    Export default advanced parameters to YAML file with comments.

    This creates a template file that users can modify to customize
    photometry, visualization, and download parameters. The template
    does NOT include source-specific information (ra/dec/name).

    Parameters
    ----------
    output_path : str or Path
        Directory to save the parameter file, or full path to YAML file.
        If directory, saves as {output_path}/{filename}.
        If file path ending in .yaml/.yml, uses that path directly.
    filename : str, optional
        Filename to use if output_path is a directory.
        Default: "spxquery_default_params.yaml"

    Returns
    -------
    Path
        Full path to the created parameter file

    Examples
    --------
    >>> # Save to directory with default filename
    >>> params_file = export_default_parameters("my_folder")
    >>> # Returns: Path("my_folder/spxquery_default_params.yaml")

    >>> # Save to specific file
    >>> params_file = export_default_parameters("my_folder/custom_params.yaml")
    >>> # Returns: Path("my_folder/custom_params.yaml")

    >>> # Save to current directory
    >>> params_file = export_default_parameters(".")
    >>> # Returns: Path("./spxquery_default_params.yaml")
    """
    from ..core.config import AdvancedConfig, QueryConfig, Source

    output_path = Path(output_path)

    # Determine final file path
    if output_path.suffix in [".yaml", ".yml"]:
        # Full path to YAML file provided
        final_path = output_path
    else:
        # Directory provided, append filename
        output_path.mkdir(parents=True, exist_ok=True)
        final_path = output_path / filename

    # Build up default Source and QueryConfig
    source = Source(ra=0.0, dec=0.0, name="MySource")
    query_config = QueryConfig(source=source, output_dir=output_path)

    # Create default config
    default_config = AdvancedConfig(query=query_config)

    # Save to file
    default_config.to_yaml_file(final_path)

    logger.info(f"Default parameters exported to {final_path}")
    print(f"\n{'=' * 70}")
    print("Advanced Parameters Template Exported")
    print(f"{'=' * 70}")
    print(f"File location: {final_path}")
    print("\nThis template contains customizable parameters for:")
    print("  • Photometry (aperture, background annulus, sigma clipping)")
    print("  • Visualization (colormaps, marker sizes, figure settings)")
    print("  • Downloads (chunk size, timeouts, retries)")
    print("\nNOTE: This template does NOT include source information (ra/dec).")
    print("      You must provide source coordinates when running the pipeline.")
    print("\nNext steps:")
    print(f"  1. Edit {final_path.name} to customize parameters (YAML format with comments)")
    print("  2. Use the file in your pipeline with:")
    print(f"     • QueryConfig(source=..., advanced_params_file='{final_path}')")
    print(f"     • run_pipeline(ra=..., dec=..., advanced_params_file='{final_path}')")
    print(f"{'=' * 70}\n")

    return final_path


def load_advanced_config(filepath: Path) -> "AdvancedConfig":
    """
    Load advanced configuration from YAML file.

    Parameters
    ----------
    filepath : Path
        Path to YAML parameter file

    Returns
    -------
    AdvancedConfig
        Loaded configuration

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If file contains invalid parameters
    """
    from ..core.config import AdvancedConfig

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Parameter file not found: {filepath}")

    try:
        config = AdvancedConfig.from_yaml_file(filepath)
        logger.info(f"Loaded advanced parameters from {filepath}")
        return config
    except Exception as e:
        raise ValueError(f"Failed to load parameters from {filepath}: {e}")
