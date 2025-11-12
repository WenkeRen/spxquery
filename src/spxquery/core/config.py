"""
Configuration and data models for SPXQuery package.
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord

from .. import __version__

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Astronomical source coordinates."""

    ra: float  # Right ascension in degrees
    dec: float  # Declination in degrees
    name: Optional[str] = None

    def __post_init__(self):
        if not 0 <= self.ra <= 360:
            raise ValueError(f"RA must be between 0 and 360 degrees, got {self.ra}")
        if not -90 <= self.dec <= 90:
            raise ValueError(f"Dec must be between -90 and 90 degrees, got {self.dec}")

    def to_skycoord(self) -> SkyCoord:
        """
        Convert source coordinates to astropy SkyCoord object.

        Returns
        -------
        SkyCoord
            Astropy SkyCoord object with ICRS frame coordinates.

        Examples
        --------
        >>> source = Source(ra=304.69, dec=42.44, name="Deneb")
        >>> coord = source.to_skycoord()
        >>> print(coord.ra, coord.dec)
        304d41m24s 42d26m24s
        """
        return SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame="icrs")


@dataclass
class PhotometryConfig:
    """
    Advanced photometry configuration.

    Attributes
    ----------
    aperture_method : str
        Method for determining aperture size.
        Options: 'fixed' (use aperture_diameter), 'fwhm' (use PSF FWHM).
        Default: 'fwhm'.
    aperture_diameter : float
        Aperture diameter in pixels.
        Default: 3.0. Used when aperture_method='fixed', or as fallback when FWHM estimation fails.
    fwhm_multiplier : float
        Multiplier for FWHM to determine aperture diameter.
        Default: 2.5. Aperture diameter = FWHM × fwhm_multiplier.
        Only used when aperture_method='fwhm'.
    max_processing_workers : int
        Number of parallel workers for photometry processing.
        Default: 10. Adjust based on CPU cores.
    bad_flags : List[int]
        List of flag bit positions to reject in quality control.
        Default: [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19]. These flags indicate problematic pixels.
    annulus_inner_offset : float
        Gap between aperture edge and inner annulus radius (pixels).
        Default: 1.414 (√2). Reduce for crowded fields, increase for extended sources.
    min_annulus_area : int
        Minimum area for background annulus (pixels).
        Default: 10. Increase for better statistics.
    max_outer_radius : float
        Maximum outer radius for background annulus (pixels).
        Default: 5.0. Increase for faint sources.
    min_usable_pixels : int
        Minimum number of unflagged pixels required in annulus.
        Default: 10. Increase for higher quality.
    bg_sigma_clip_sigma : float
        Sigma threshold for sigma-clipped background statistics.
        Default: 3.0. Common values: 2.5 (strict), 3.0 (standard), 3.5 (lenient).
    bg_sigma_clip_maxiters : int
        Maximum iterations for sigma clipping of background.
        Default: 3. Usually 1-5 is sufficient.
    subtract_zodi : bool
        Whether to subtract zodiacal light background from images before photometry.
        Default: True. Set to False to work with raw images.
    zodi_scale_min : float
        Minimum allowed zodiacal scaling factor.
        Default: 0.0. Negative values may indicate model failure.
    zodi_scale_max : float
        Maximum allowed zodiacal scaling factor.
        Default: 10.0. Increase if studying high-zodiacal periods.
    pixel_scale_fallback : float
        Fallback pixel scale (arcsec/pixel) when WCS fails.
        Default: 6.2 (SPHEREx). Change for other missions.
    max_annulus_attempts : int
        Maximum attempts to expand annulus when insufficient pixels.
        Default: 5. Rarely needs adjustment.
    annulus_expansion_step : float
        Step size in pixels when expanding annulus.
        Default: 0.5. Usually 0.3-1.0 is reasonable.
    background_method : str
        Background estimation method.
        Options: 'annulus' (ring around source), 'window' (rectangular region).
        Default: 'annulus'. Use 'window' for crowded fields.
    window_size : Union[int, Tuple[int, int]]
        Size of background window in pixels (for window method).
        Default: 50. If int: square window. If tuple: (height, width).
        Pixels intersecting the aperture are automatically excluded.
    """

    aperture_method: str = "fwhm"
    aperture_diameter: float = 3.0
    fwhm_multiplier: float = 2.5
    max_processing_workers: int = 10
    bad_flags: List[int] = field(default_factory=lambda: [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19])
    annulus_inner_offset: float = 1.414
    min_annulus_area: int = 10
    max_outer_radius: float = 5.0
    min_usable_pixels: int = 10
    bg_sigma_clip_sigma: float = 3.0
    bg_sigma_clip_maxiters: int = 3
    subtract_zodi: bool = True
    zodi_scale_min: float = 0.0
    zodi_scale_max: float = 10.0
    pixel_scale_fallback: float = 6.2
    max_annulus_attempts: int = 5
    annulus_expansion_step: float = 0.5
    background_method: str = "annulus"
    window_size: Union[int, Tuple[int, int]] = 50

    def __post_init__(self):
        """Validate parameters."""
        # Validate aperture method
        valid_methods = ["fixed", "fwhm"]
        if self.aperture_method not in valid_methods:
            raise ValueError(f"aperture_method must be one of {valid_methods}, got '{self.aperture_method}'")

        # Validate aperture parameters
        if self.aperture_diameter <= 0:
            raise ValueError(f"aperture_diameter must be > 0, got {self.aperture_diameter}")
        if self.fwhm_multiplier <= 0:
            raise ValueError(f"fwhm_multiplier must be > 0, got {self.fwhm_multiplier}")

        if self.max_processing_workers <= 0:
            raise ValueError(f"max_processing_workers must be > 0, got {self.max_processing_workers}")
        if self.annulus_inner_offset < 0:
            raise ValueError(f"annulus_inner_offset must be >= 0, got {self.annulus_inner_offset}")
        if self.min_annulus_area <= 0:
            raise ValueError(f"min_annulus_area must be > 0, got {self.min_annulus_area}")
        if self.max_outer_radius <= 0:
            raise ValueError(f"max_outer_radius must be > 0, got {self.max_outer_radius}")
        if self.min_usable_pixels <= 0:
            raise ValueError(f"min_usable_pixels must be > 0, got {self.min_usable_pixels}")
        if self.bg_sigma_clip_sigma <= 0:
            raise ValueError(f"bg_sigma_clip_sigma must be > 0, got {self.bg_sigma_clip_sigma}")
        if self.bg_sigma_clip_maxiters <= 0:
            raise ValueError(f"bg_sigma_clip_maxiters must be > 0, got {self.bg_sigma_clip_maxiters}")
        if self.zodi_scale_max <= self.zodi_scale_min:
            raise ValueError("zodi_scale_max must be > zodi_scale_min")
        if self.pixel_scale_fallback <= 0:
            raise ValueError(f"pixel_scale_fallback must be > 0, got {self.pixel_scale_fallback}")

        # Validate background method
        valid_bg_methods = ["annulus", "window"]
        if self.background_method not in valid_bg_methods:
            raise ValueError(f"background_method must be one of {valid_bg_methods}, got '{self.background_method}'")

        # Validate window parameters
        if isinstance(self.window_size, int):
            if self.window_size <= 0:
                raise ValueError(f"window_size must be > 0, got {self.window_size}")
        elif isinstance(self.window_size, tuple):
            if len(self.window_size) != 2:
                raise ValueError(f"window_size tuple must have 2 elements, got {len(self.window_size)}")
            if any(s <= 0 for s in self.window_size):
                raise ValueError(f"window_size elements must be > 0, got {self.window_size}")
        else:
            raise TypeError(f"window_size must be int or tuple, got {type(self.window_size)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhotometryConfig":
        """Create from dictionary."""
        # Only use keys that exist in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass
class VisualizationConfig:
    """
    Advanced visualization configuration.

    Attributes
    ----------
    sigma_threshold : float
        Minimum SNR (flux/flux_err) for quality control filtering.
        Default: 5.0. Measurements below this are marked as rejected.
    use_magnitude : bool
        If True, plot AB magnitude instead of flux.
        Default: False (plot flux in μJy).
    show_errorbars : bool
        If True, show error bars on plots.
        Default: True.
    wavelength_cmap : str
        Matplotlib colormap name for wavelength coding in light curves.
        Default: "rainbow". Alternatives: "viridis", "plasma", "cividis".
    date_cmap : str
        Matplotlib colormap name for date coding in spectra.
        Default: "viridis". Should differ from wavelength_cmap.
    sigma_clip_sigma : float
        Sigma threshold for outlier removal in plots.
        Default: 3.0. Set to 100+ to disable outlier removal.
    sigma_clip_maxiters : int
        Maximum iterations for sigma clipping.
        Default: 10. Usually sufficient.
    ylim_percentile_min : float
        Lower percentile for smart y-axis limits (0-100).
        Default: 1.0. Use 0.0 to show all data.
    ylim_percentile_max : float
        Upper percentile for smart y-axis limits (0-100).
        Default: 99.0. Use 100.0 to show all data.
    ylim_padding_fraction : float
        Padding fraction added to y-axis range.
        Default: 0.1 (10%). Usually 0.05-0.2.
    marker_size_good : float
        Marker size for good measurements.
        Default: 1.5. Increase for print publications.
    marker_size_rejected : float
        Marker size for rejected measurements.
        Default: 2.0. Should be visible but not dominant.
    marker_size_upper_limit : float
        Marker size for upper limit arrows.
        Default: 3.0. Should be clearly visible.
    errorbar_alpha : float
        Transparency for error bars (0-1).
        Default: 0.2. Increase for print, decrease for screen.
    marker_alpha : float
        Transparency for markers (0-1).
        Default: 0.9. Usually keep near 1.0.
    errorbar_linewidth : float
        Line width for error bars in points.
        Default: 0.5. Increase for print publications.
    figsize : Tuple[float, float]
        Figure size in inches (width, height).
        Default: (10, 8). Common journal sizes: (7.5, 6), (3.5, 3).
    dpi : int
        Resolution in dots per inch for saved figures.
        Default: 150. Use 300 for print publications.
    """

    sigma_threshold: float = 5.0
    use_magnitude: bool = False
    show_errorbars: bool = True
    wavelength_cmap: str = "rainbow"
    date_cmap: str = "viridis"
    sigma_clip_sigma: float = 3.0
    sigma_clip_maxiters: int = 10
    ylim_percentile_min: float = 1.0
    ylim_percentile_max: float = 99.0
    ylim_padding_fraction: float = 0.1
    marker_size_good: float = 1.5
    marker_size_rejected: float = 2.0
    marker_size_upper_limit: float = 3.0
    errorbar_alpha: float = 0.2
    marker_alpha: float = 0.9
    errorbar_linewidth: float = 0.5
    figsize: Tuple[float, float] = (10, 8)
    dpi: int = 150

    def __post_init__(self):
        """Validate parameters."""
        # Validate quality control
        if self.sigma_threshold <= 0:
            raise ValueError(f"sigma_threshold must be > 0, got {self.sigma_threshold}")

        # Validate colormaps
        import matplotlib.cm as cm

        try:
            cm.get_cmap(self.wavelength_cmap)
        except ValueError:
            raise ValueError(f"Invalid wavelength_cmap: '{self.wavelength_cmap}'")
        try:
            cm.get_cmap(self.date_cmap)
        except ValueError:
            raise ValueError(f"Invalid date_cmap: '{self.date_cmap}'")

        # Validate numeric ranges
        if not 0 <= self.ylim_percentile_min <= 100:
            raise ValueError(f"ylim_percentile_min must be 0-100, got {self.ylim_percentile_min}")
        if not 0 <= self.ylim_percentile_max <= 100:
            raise ValueError(f"ylim_percentile_max must be 0-100, got {self.ylim_percentile_max}")
        if self.ylim_percentile_min >= self.ylim_percentile_max:
            raise ValueError("ylim_percentile_min must be < ylim_percentile_max")
        if not 0 <= self.errorbar_alpha <= 1:
            raise ValueError(f"errorbar_alpha must be 0-1, got {self.errorbar_alpha}")
        if not 0 <= self.marker_alpha <= 1:
            raise ValueError(f"marker_alpha must be 0-1, got {self.marker_alpha}")
        if self.dpi <= 0:
            raise ValueError(f"dpi must be > 0, got {self.dpi}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert tuple to list for JSON serialization
        if isinstance(data["figsize"], tuple):
            data["figsize"] = list(data["figsize"])
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualizationConfig":
        """Create from dictionary."""
        # Convert figsize list back to tuple
        if "figsize" in data and isinstance(data["figsize"], list):
            data["figsize"] = tuple(data["figsize"])
        # Only use keys that exist in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass
class DownloadConfig:
    """
    Advanced download configuration.

    Attributes
    ----------
    max_download_workers : int
        Number of parallel workers for file downloads.
        Default: 4. Adjust based on network bandwidth and connection limits.
    cutout_size : Optional[str]
        Image cutout size for downloads (e.g., "200px", "3arcmin").
        Default: None (download full images).
    cutout_center : Optional[str]
        Cutout center coordinates (e.g., "70,20", "300.5,120px").
        Default: None (use source position).
    chunk_size : int
        Download chunk size in bytes.
        Default: 8192 (8 KB). Increase for fast connections.
    timeout : int
        HTTP request timeout in seconds.
        Default: 300 (5 minutes). Increase for slow connections.
    max_retries : int
        Number of retry attempts for failed downloads.
        Default: 3. Increase for unreliable connections.
    retry_delay : int
        Delay between retry attempts in seconds.
        Default: 5. Consider exponential backoff for many retries.
    user_agent : str
        User agent string for HTTP requests.
        Default: "SPXQuery/<version>". Usually no need to change.
    """

    max_download_workers: int = 4
    cutout_size: Optional[str] = None
    cutout_center: Optional[str] = None
    chunk_size: int = 8192
    timeout: int = 300
    max_retries: int = 3
    retry_delay: int = 5
    user_agent: str = field(default_factory=lambda: f"SPXQuery/{__version__}")

    def __post_init__(self):
        """Validate parameters."""
        if self.max_download_workers <= 0:
            raise ValueError(f"max_download_workers must be > 0, got {self.max_download_workers}")

        # Validate cutout parameters
        if self.cutout_size:
            from ..utils.helpers import validate_cutout_size

            if not validate_cutout_size(self.cutout_size):
                raise ValueError(
                    f"Invalid cutout_size format: '{self.cutout_size}'. "
                    "Expected format: <value>[,<value>][units], e.g., '200px', '3arcmin', '0.1'"
                )

        if self.cutout_center:
            from ..utils.helpers import validate_cutout_center

            if not validate_cutout_center(self.cutout_center):
                raise ValueError(
                    f"Invalid cutout_center format: '{self.cutout_center}'. "
                    "Expected format: <x>,<y>[units], e.g., '70,20', '300.5,120px'"
                )

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.retry_delay < 0:
            raise ValueError(f"retry_delay must be >= 0, got {self.retry_delay}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DownloadConfig":
        """Create from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass
class QueryConfig:
    """
    Query-specific configuration (sub-config of AdvancedConfig).

    This class handles only query-specific parameters. It is contained within
    AdvancedConfig, not the other way around.

    Parameters
    ----------
    source : Source
        Source coordinates and name
    output_dir : Path
        Output directory for results and data
    bands : Optional[List[str]]
        List of bands to query (e.g., ['D1', 'D2']). None = all bands.
    """

    source: Source
    output_dir: Path = field(default_factory=Path.cwd)
    bands: Optional[List[str]] = None

    def __post_init__(self):
        # Convert to Path if string
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Validate bands
        valid_bands = ["D1", "D2", "D3", "D4", "D5", "D6"]
        if self.bands:
            invalid = set(self.bands) - set(valid_bands)
            if invalid:
                raise ValueError(f"Invalid bands: {invalid}. Valid bands are: {valid_bands}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": {"ra": float(self.source.ra), "dec": float(self.source.dec), "name": self.source.name},
            "output_dir": str(self.output_dir),
            "bands": self.bands,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryConfig":
        """Create from dictionary."""
        source = Source(
            ra=data["source"]["ra"],
            dec=data["source"]["dec"],
            name=data["source"].get("name"),
        )
        return cls(
            source=source,
            output_dir=Path(data.get("output_dir", Path.cwd())),
            bands=data.get("bands"),
        )


@dataclass
class AdvancedConfig:
    """
    Central configuration "bus" for all advanced parameters.

    This class manages ALL sub-configurations including query parameters.
    It provides intelligent parameter routing so users can update any parameter
    without needing to know which sub-config it belongs to.

    Attributes
    ----------
    query : QueryConfig
        Query-specific parameters (source, output_dir, bands)
    photometry : PhotometryConfig
        Photometry-related parameters
    visualization : VisualizationConfig
        Visualization and plotting parameters
    download : DownloadConfig
        Download and cutout parameters
    pipeline_stages : List[str]
        List of pipeline stages to execute.
        Default: ['query', 'download', 'processing', 'visualization']
    """

    query: QueryConfig
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    pipeline_stages: List[str] = field(default_factory=lambda: ["query", "download", "processing", "visualization"])
    _params_file: Optional[Path] = field(default=None, init=False, repr=False)
    _auto_loaded: bool = field(default=False, init=False, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query.to_dict(),
            "photometry": self.photometry.to_dict(),
            "visualization": self.visualization.to_dict(),
            "download": self.download.to_dict(),
            "pipeline_stages": self.pipeline_stages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvancedConfig":
        """Create from dictionary."""
        return cls(
            query=QueryConfig.from_dict(data.get("query", {})),
            photometry=PhotometryConfig.from_dict(data.get("photometry", {})),
            visualization=VisualizationConfig.from_dict(data.get("visualization", {})),
            download=DownloadConfig.from_dict(data.get("download", {})),
            pipeline_stages=data.get("pipeline_stages", ["query", "download", "processing", "visualization"]),
        )

    def update(self, **kwargs) -> None:
        """
        Intelligently update parameters across all sub-configs.

        This method automatically determines which sub-config each parameter
        belongs to and updates it accordingly. Users don't need to know the
        internal structure.

        Parameters
        ----------
        **kwargs
            Parameter names and values to update. Can include:
            - Query parameters: source, output_dir, bands
            - Photometry parameters: aperture_diameter, etc.
            - Visualization parameters: sigma_threshold, use_magnitude, etc.
            - Download parameters: cutout_size, max_download_workers, etc.
            - Pipeline control: pipeline_stages

        Raises
        ------
        ValueError
            If a parameter name doesn't exist in any sub-config

        Examples
        --------
        >>> config = AdvancedConfig(query=QueryConfig(source=Source(ra=10, dec=20)))
        >>> config.update(aperture_diameter=5.0, use_magnitude=True, bands=['D1', 'D2'])
        >>> # Automatically routes to correct sub-configs:
        >>> # - aperture_diameter -> photometry
        >>> # - use_magnitude -> visualization
        >>> # - bands -> query
        """
        # Build mapping of parameter names to sub-configs
        param_map = {}

        # Get all fields from each sub-config
        for field_name in QueryConfig.__dataclass_fields__:
            param_map[field_name] = "query"

        for field_name in PhotometryConfig.__dataclass_fields__:
            param_map[field_name] = "photometry"

        for field_name in VisualizationConfig.__dataclass_fields__:
            param_map[field_name] = "visualization"

        for field_name in DownloadConfig.__dataclass_fields__:
            param_map[field_name] = "download"

        # Special handling for pipeline_stages (belongs to AdvancedConfig itself)
        if "pipeline_stages" in kwargs:
            self.pipeline_stages = kwargs.pop("pipeline_stages")

        # Route each parameter to its sub-config
        query_updates = {}
        photometry_updates = {}
        visualization_updates = {}
        download_updates = {}
        unknown_params = []

        for param_name, value in kwargs.items():
            if param_name in param_map:
                target = param_map[param_name]
                if target == "query":
                    query_updates[param_name] = value
                elif target == "photometry":
                    photometry_updates[param_name] = value
                elif target == "visualization":
                    visualization_updates[param_name] = value
                elif target == "download":
                    download_updates[param_name] = value
            else:
                unknown_params.append(param_name)

        # Raise error for unknown parameters
        if unknown_params:
            raise ValueError(
                f"Unknown parameter(s): {unknown_params}. "
                f"Valid parameters are in QueryConfig, PhotometryConfig, VisualizationConfig, or DownloadConfig."
            )

        # Update each sub-config by creating new instances
        if query_updates:
            # Handle Source object conversion
            if "source" in query_updates and isinstance(query_updates["source"], Source):
                # Convert Source object to dictionary
                query_updates["source"] = {
                    "ra": query_updates["source"].ra,
                    "dec": query_updates["source"].dec,
                    "name": query_updates["source"].name,
                }

            # Handle Path object conversion
            if "output_dir" in query_updates and isinstance(query_updates["output_dir"], Path):
                query_updates["output_dir"] = str(query_updates["output_dir"])

            current_dict = self.query.to_dict()
            current_dict.update(query_updates)
            self.query = QueryConfig.from_dict(current_dict)

        if photometry_updates:
            current_dict = self.photometry.to_dict()
            current_dict.update(photometry_updates)
            self.photometry = PhotometryConfig.from_dict(current_dict)

        if visualization_updates:
            current_dict = self.visualization.to_dict()
            current_dict.update(visualization_updates)
            self.visualization = VisualizationConfig.from_dict(current_dict)

        if download_updates:
            current_dict = self.download.to_dict()
            current_dict.update(download_updates)
            self.download = DownloadConfig.from_dict(current_dict)

        logger.info(f"Updated parameters: {list(kwargs.keys())}")

    def to_yaml_file(self, filepath: Path) -> None:
        """Save to YAML file with comments."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            # Add header comment
            f.write("# SPXQuery Advanced Configuration\n")
            f.write("# Edit values below and load via advanced_params_file parameter\n\n")

            # Write YAML
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml_file(cls, filepath: Path) -> "AdvancedConfig":
        """Load from YAML file."""
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        
        # Handle both nested and direct config formats
        if isinstance(data, dict) and "config" in data:
            config_data = data["config"]
        else:
            config_data = data
        
        config = cls.from_dict(config_data)
        config._params_file = filepath
        config._auto_loaded = True
        logger.info(f"Loaded advanced parameters from {filepath}")
        return config

    @classmethod
    def from_saved_state(cls, source_name: str, output_dir: Path, **user_overrides) -> "AdvancedConfig":
        """
        Create AdvancedConfig by loading from saved state with optional overrides.

        Parameters are loaded with priority: user_overrides > saved state > defaults.

        Parameters
        ----------
        source_name : str
            Name of the source (used to find {source_name}.yaml)
        output_dir : Path
            Output directory where state file is located
        **user_overrides
            Any parameters to override from saved state (can include parameters
            from any sub-config: query, photometry, visualization, download)

        Returns
        -------
        AdvancedConfig
            Complete configuration loaded from state with overrides applied

        Examples
        --------
        >>> # Load everything from saved state
        >>> config = AdvancedConfig.from_saved_state("cloverleaf", Path("output"))
        >>>
        >>> # Load from state but override some parameters
        >>> config = AdvancedConfig.from_saved_state(
        ...     "cloverleaf", Path("output"),
        ...     aperture_diameter=5.0,
        ...     use_magnitude=True,
        ...     bands=['D1', 'D2']
        ... )
        """
        from ..utils.helpers import load_yaml

        output_dir = Path(output_dir)
        state_file = output_dir / f"{source_name}.yaml"

        if not state_file.exists():
            # No saved state - create from scratch with user overrides
            ra = user_overrides.pop("ra", 0.0)
            dec = user_overrides.pop("dec", 0.0)
            bands = user_overrides.pop("bands", None)

            source = Source(ra=ra, dec=dec, name=source_name)
            query_config = QueryConfig(source=source, output_dir=output_dir, bands=bands)
            config = cls(query=query_config)

            # Apply any remaining overrides using update()
            if user_overrides:
                config.update(**user_overrides)

            config._auto_loaded = False
            return config

        # Load saved state
        saved_data = load_yaml(state_file)
        saved_full_config = saved_data.get("config", {})

        # Extract query data from saved state
        saved_source = saved_full_config.get("source", {})
        saved_output_dir = saved_full_config.get("output_dir", str(output_dir))
        saved_bands = saved_full_config.get("bands")

        # Build query config with priority: user > saved > defaults
        ra = user_overrides.pop("ra", saved_source.get("ra", 0.0))
        dec = user_overrides.pop("dec", saved_source.get("dec", 0.0))
        bands = user_overrides.pop("bands", saved_bands)

        source = Source(ra=ra, dec=dec, name=source_name)
        query_config = QueryConfig(source=source, output_dir=Path(saved_output_dir), bands=bands)

        # Build sub-configs from saved state or defaults
        photometry_data = {}
        visualization_data = {}
        download_data = {}

        # Map old flat structure to new sub-config structure
        photometry_params = PhotometryConfig.__dataclass_fields__.keys()
        visualization_params = VisualizationConfig.__dataclass_fields__.keys()
        download_params = DownloadConfig.__dataclass_fields__.keys()

        for key, value in saved_full_config.items():
            if key in photometry_params:
                photometry_data[key] = value
            elif key in visualization_params:
                visualization_data[key] = value
            elif key in download_params:
                download_data[key] = value

        # Create sub-configs
        photometry = PhotometryConfig.from_dict(photometry_data) if photometry_data else PhotometryConfig()
        visualization = (
            VisualizationConfig.from_dict(visualization_data) if visualization_data else VisualizationConfig()
        )
        download = DownloadConfig.from_dict(download_data) if download_data else DownloadConfig()

        # Get pipeline stages from saved state
        pipeline_stages = saved_data.get("pipeline_stages", ["query", "download", "processing", "visualization"])

        # Create AdvancedConfig
        config = cls(
            query=query_config,
            photometry=photometry,
            visualization=visualization,
            download=download,
            pipeline_stages=pipeline_stages,
        )

        # Apply user overrides for any remaining parameters
        if user_overrides:
            config.update(**user_overrides)

        config._auto_loaded = True
        logger.info(f"Loaded configuration from {state_file} with {len(user_overrides)} overrides")
        return config

    @classmethod
    def create(
        cls,
        source: Source,
        output_dir: Path,
        bands: Optional[List[str]] = None,
        **advanced_params,
    ) -> "AdvancedConfig":
        """
        Convenient factory method to create AdvancedConfig from basic parameters.

        This is the recommended way to create a new AdvancedConfig for users who
        want to specify some advanced parameters without creating all sub-configs.

        Parameters
        ----------
        source : Source
            Source coordinates and name
        output_dir : Path
            Output directory for results
        bands : Optional[List[str]]
            List of bands to query (e.g., ['D1', 'D2']). None = all bands.
        **advanced_params
            Any additional parameters to override from defaults.
            Can include parameters from photometry, visualization, or download configs.

        Returns
        -------
        AdvancedConfig
            Complete configuration with specified parameters

        Examples
        --------
        >>> # Create with defaults
        >>> source = Source(ra=304.69, dec=42.44, name="My_Star")
        >>> config = AdvancedConfig.create(source, Path("output"))
        >>>
        >>> # Create with custom parameters
        >>> config = AdvancedConfig.create(
        ...     source=Source(ra=304.69, dec=42.44, name="My_Star"),
        ...     output_dir=Path("output"),
        ...     bands=['D1', 'D2'],
        ...     aperture_diameter=5.0,
        ...     use_magnitude=True,
        ...     cutout_size='200px'
        ... )
        """
        query_config = QueryConfig(source=source, output_dir=output_dir, bands=bands)
        config = cls(query=query_config)

        # Apply any advanced parameter overrides
        if advanced_params:
            config.update(**advanced_params)

        return config


@dataclass
class ObservationInfo:
    """Information about a single SPHEREx observation."""

    obs_id: str
    band: str
    mjd: float
    wavelength_min: float  # microns
    wavelength_max: float  # microns
    download_url: str  # Base download URL (cutout params appended during download)
    t_min: float  # MJD
    t_max: float  # MJD

    @property
    def wavelength_center(self) -> float:
        """Central wavelength in microns."""
        return (self.wavelength_min + self.wavelength_max) / 2

    @property
    def bandwidth(self) -> float:
        """Bandwidth in microns."""
        return self.wavelength_max - self.wavelength_min


@dataclass
class QueryResults:
    """Results from SPHEREx archive query."""

    observations: List[ObservationInfo]
    query_time: datetime
    source: Source
    total_size_gb: float
    time_span_days: float
    band_counts: Dict[str, int]

    def __len__(self):
        return len(self.observations)

    def filter_by_band(self, bands: List[str]) -> "QueryResults":
        """Return new QueryResults filtered by bands."""
        filtered_obs = [obs for obs in self.observations if obs.band in bands]
        return QueryResults(
            observations=filtered_obs,
            query_time=self.query_time,
            source=self.source,
            total_size_gb=0.0,  # File sizes unknown until download
            time_span_days=self.time_span_days,
            band_counts={band: sum(1 for obs in filtered_obs if obs.band == band) for band in bands},
        )


@dataclass
class PhotometryResult:
    """Result from aperture photometry on a single observation."""

    obs_id: str
    mjd: float
    flux: float  # microJansky (uJy)
    flux_error: float  # microJansky (uJy)
    wavelength: float  # microns
    bandwidth: float  # microns
    flag: int  # Combined flag bitmap
    pix_x: float  # Pixel X coordinate
    pix_y: float  # Pixel Y coordinate
    band: str
    mag_ab: Optional[float] = None  # AB magnitude
    mag_ab_error: Optional[float] = None  # AB magnitude error

    @property
    def is_upper_limit(self) -> bool:
        """Check if measurement should be treated as upper limit."""
        return self.flux_error > self.flux


@dataclass
class DownloadResult:
    """Result from file download attempt."""

    url: str
    local_path: Path
    success: bool
    error: Optional[str] = None
    size_mb: Optional[float] = None


@dataclass
class PipelineState:
    """
    State for resumable pipeline execution.

    This class uses AdvancedConfig as the complete configuration container,
    which includes query, photometry, visualization, download, and pipeline stages.
    """

    stage: str  # Current stage: 'query', 'download', 'processing', 'visualization', 'complete'
    config: AdvancedConfig
    query_results: Optional[QueryResults] = None
    downloaded_files: List[Path] = field(default_factory=list)
    photometry_results: List[PhotometryResult] = field(default_factory=list)
    csv_path: Optional[Path] = None
    plot_path: Optional[Path] = None
    completed_stages: List[str] = field(default_factory=list)  # Track completed stages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage": self.stage,
            "completed_stages": self.completed_stages,
            "config": self.config.to_dict(),
            "query_results": {
                "observations": [
                    {
                        "obs_id": obs.obs_id,
                        "band": obs.band,
                        "mjd": float(obs.mjd),
                        "wavelength_min": float(obs.wavelength_min),
                        "wavelength_max": float(obs.wavelength_max),
                        "download_url": obs.download_url,
                        "t_min": float(obs.t_min),
                        "t_max": float(obs.t_max),
                    }
                    for obs in self.query_results.observations
                ]
                if self.query_results
                else [],
                "query_time": self.query_results.query_time.isoformat() if self.query_results else None,
                "total_size_gb": float(self.query_results.total_size_gb) if self.query_results else 0,
                "time_span_days": float(self.query_results.time_span_days) if self.query_results else 0,
                "band_counts": self.query_results.band_counts if self.query_results else {},
            }
            if self.query_results
            else None,
            "downloaded_files": [str(p) for p in self.downloaded_files],
            "photometry_results": [
                {
                    "obs_id": pr.obs_id,
                    "mjd": float(pr.mjd),
                    "flux": float(pr.flux),
                    "flux_error": float(pr.flux_error),
                    "wavelength": float(pr.wavelength),
                    "bandwidth": float(pr.bandwidth),
                    "flag": int(pr.flag),
                    "pix_x": float(pr.pix_x),
                    "pix_y": float(pr.pix_y),
                    "band": pr.band,
                    "mag_ab": float(pr.mag_ab) if pr.mag_ab is not None else None,
                    "mag_ab_error": float(pr.mag_ab_error) if pr.mag_ab_error is not None else None,
                }
                for pr in self.photometry_results
            ],
            "csv_path": str(self.csv_path) if self.csv_path else None,
            "plot_path": str(self.plot_path) if self.plot_path else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        """Create from dictionary."""
        # Reconstruct AdvancedConfig
        # Handle both old flat format and new nested format
        config_data = data["config"]

        # If config has nested structure (new format), use it directly
        if "query" in config_data:
            config = AdvancedConfig.from_dict(config_data)
        else:
            # Old flat format - need to restructure
            # Extract source from old format
            source_data = config_data.get("source", {})
            source = Source(
                ra=source_data.get("ra", 0.0),
                dec=source_data.get("dec", 0.0),
                name=source_data.get("name"),
            )

            # Build query config
            query_config = QueryConfig(
                source=source,
                output_dir=Path(config_data.get("output_dir", Path.cwd())),
                bands=config_data.get("bands"),
            )

            # Build photometry config from old flat format
            photometry_params = {}
            for key in PhotometryConfig.__dataclass_fields__:
                if key in config_data:
                    photometry_params[key] = config_data[key]
            photometry = PhotometryConfig.from_dict(photometry_params) if photometry_params else PhotometryConfig()

            # Build visualization config from old flat format
            visualization_params = {}
            for key in VisualizationConfig.__dataclass_fields__:
                if key in config_data:
                    visualization_params[key] = config_data[key]
            visualization = (
                VisualizationConfig.from_dict(visualization_params) if visualization_params else VisualizationConfig()
            )

            # Build download config from old flat format
            download_params = {}
            for key in DownloadConfig.__dataclass_fields__:
                if key in config_data:
                    download_params[key] = config_data[key]
            download = DownloadConfig.from_dict(download_params) if download_params else DownloadConfig()

            # Get pipeline stages
            pipeline_stages = data.get("pipeline_stages", ["query", "download", "processing", "visualization"])

            # Create AdvancedConfig
            config = AdvancedConfig(
                query=query_config,
                photometry=photometry,
                visualization=visualization,
                download=download,
                pipeline_stages=pipeline_stages,
            )

        # Reconstruct query results
        query_results = None
        if data.get("query_results"):
            observations = [ObservationInfo(**obs) for obs in data["query_results"]["observations"]]
            query_results = QueryResults(
                observations=observations,
                query_time=datetime.fromisoformat(data["query_results"]["query_time"]),
                source=config.query.source,
                total_size_gb=data["query_results"]["total_size_gb"],
                time_span_days=data["query_results"]["time_span_days"],
                band_counts=data["query_results"]["band_counts"],
            )

        # Reconstruct photometry results
        photometry_results = [PhotometryResult(**pr) for pr in data.get("photometry_results", [])]

        return cls(
            stage=data["stage"],
            config=config,
            query_results=query_results,
            downloaded_files=[Path(p) for p in data.get("downloaded_files", [])],
            photometry_results=photometry_results,
            csv_path=Path(data["csv_path"]) if data.get("csv_path") else None,
            plot_path=Path(data["plot_path"]) if data.get("plot_path") else None,
            completed_stages=data.get("completed_stages", []),
        )
