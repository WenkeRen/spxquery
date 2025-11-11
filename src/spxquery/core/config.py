"""
Configuration and data models for SPXQuery package.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import astropy.units as u
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

    @property
    def coord(self) -> SkyCoord:
        """Return source coordinates as SkyCoord."""
        return SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame="icrs")


@dataclass
class PhotometryConfig:
    """
    Advanced photometry configuration.

    Attributes
    ----------
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
    """

    annulus_inner_offset: float = 1.414
    min_annulus_area: int = 10
    max_outer_radius: float = 5.0
    min_usable_pixels: int = 10
    bg_sigma_clip_sigma: float = 3.0
    bg_sigma_clip_maxiters: int = 3
    zodi_scale_min: float = 0.0
    zodi_scale_max: float = 10.0
    pixel_scale_fallback: float = 6.2
    max_annulus_attempts: int = 5
    annulus_expansion_step: float = 0.5

    def __post_init__(self):
        """Validate parameters."""
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

    chunk_size: int = 8192
    timeout: int = 300
    max_retries: int = 3
    retry_delay: int = 5
    user_agent: str = field(default_factory=lambda: f"SPXQuery/{__version__}")

    def __post_init__(self):
        """Validate parameters."""
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
class ModelPhotometryConfig:
    """
    Model-based photometry configuration using AstroPhot.

    Supports various source models (point source, Sersic profile, etc.) for
    improved photometric precision. Addresses SPHEREx undersampling (6.2"/pixel
    vs 5" FWHM PSF) using 10x oversampled PSF images.

    Attributes
    ----------
    enabled : bool
        Enable model-based photometry availability. Default: False.
        Note: Even when True, modeling photometry must be explicitly called
        via run_modeling_photometry() and is NOT part of standard pipeline.
    model_type : str
        Source model to fit. Default: "point_source".
        Options: "point_source" (PSF convolution), "sersic" (galaxy profile),
        "exponential", "gaussian", etc. (see AstroPhot documentation).
    fitting_method : str
        AstroPhot fitting method. Default: "LM" (Levenberg-Marquardt).
        Options: "LM", "Nested", "HMC" (see AstroPhot documentation).
    psf_oversample_factor : int
        PSF oversampling factor. Default: 10 (SPHEREx standard).
        Do not change unless using non-standard PSF data.
    fitting_box_size : int
        Size of fitting region in pixels (image scale, not PSF scale).
        Default: 30. Must be >= 20 pixels (2× PSF effective size).
        Increase for extended sources, decrease for crowded fields.
    initial_flux_from_aperture : bool
        Use aperture photometry result as initial flux guess. Default: True.
        Improves convergence but requires running aperture photometry first.
    save_residuals : bool
        Save model fitting residual images to FITS files. Default: False.
        Useful for quality assessment but increases disk usage.
    save_fitting_plots : bool
        Save model and residual plots for each observation. Default: False.
        Creates PNG files showing fitted model and normalized residuals.
    residuals_dir : str
        Directory name for residual FITS files (relative to output_dir).
        Default: "model_residuals". Organized by band: model_residuals/D1/, etc.
    plots_dir : str
        Directory name for fitting plots (relative to output_dir).
        Default: "model_plots". Organized by band: model_plots/D1/, etc.
    background_method : str
        Background estimation method. Default: "local_annulus".
        Options: "local_annulus" (annulus around source), "global" (image median),
        "plane" (fit plane to image), "none" (assume background-subtracted).
    max_iterations : int
        Maximum fitting iterations. Default: 100.
        Increase for difficult fits, decrease for speed.
    convergence_tolerance : float
        Convergence criterion for fitting. Default: 1e-6.
        Smaller values require tighter convergence but may not improve results.
    """

    enabled: bool = False
    model_type: str = "point_source"
    fitting_method: str = "LM"
    psf_oversample_factor: int = 10
    fitting_box_size: int = 30
    initial_flux_from_aperture: bool = True
    save_residuals: bool = False
    save_fitting_plots: bool = False
    residuals_dir: str = "model_residuals"
    plots_dir: str = "model_plots"
    background_method: str = "local_annulus"
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6

    def __post_init__(self):
        """Validate parameters."""
        # Validate model type
        valid_models = ["point_source", "sersic", "exponential", "gaussian"]
        if self.model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}, got '{self.model_type}'")

        # Validate fitting method
        valid_methods = ["LM", "Nested", "HMC"]
        if self.fitting_method not in valid_methods:
            raise ValueError(f"fitting_method must be one of {valid_methods}, got '{self.fitting_method}'")

        # Validate numeric parameters
        if self.psf_oversample_factor <= 0:
            raise ValueError(f"psf_oversample_factor must be > 0, got {self.psf_oversample_factor}")
        if self.fitting_box_size <= 0:
            raise ValueError(f"fitting_box_size must be > 0, got {self.fitting_box_size}")

        # Validate fitting_box_size is large enough for PSF
        # PSF is typically 101 pixels at 10x oversampling = 10.1 pixels effective
        # Need at least 2x PSF size for robust fitting
        min_fitting_box = 20
        if self.fitting_box_size < min_fitting_box:
            raise ValueError(
                f"fitting_box_size must be >= {min_fitting_box} pixels "
                f"(2× PSF effective size), got {self.fitting_box_size}"
            )
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {self.max_iterations}")
        if self.convergence_tolerance <= 0:
            raise ValueError(f"convergence_tolerance must be > 0, got {self.convergence_tolerance}")

        # Validate background method
        valid_bg_methods = ["local_annulus", "global", "plane", "none"]
        if self.background_method not in valid_bg_methods:
            raise ValueError(f"background_method must be one of {valid_bg_methods}, got '{self.background_method}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelPhotometryConfig":
        """Create from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass
class AdvancedConfig:
    """
    Container for all advanced configuration options.

    This class groups all advanced configuration objects together
    for easier management and serialization.
    """

    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    model_photometry: ModelPhotometryConfig = field(default_factory=ModelPhotometryConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "photometry": self.photometry.to_dict(),
            "visualization": self.visualization.to_dict(),
            "download": self.download.to_dict(),
            "model_photometry": self.model_photometry.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvancedConfig":
        """Create from dictionary."""
        return cls(
            photometry=PhotometryConfig.from_dict(data.get("photometry", {})),
            visualization=VisualizationConfig.from_dict(data.get("visualization", {})),
            download=DownloadConfig.from_dict(data.get("download", {})),
            model_photometry=ModelPhotometryConfig.from_dict(data.get("model_photometry", {})),
        )

    def to_json_file(self, filepath: Path) -> None:
        """Save to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json_file(cls, filepath: Path) -> "AdvancedConfig":
        """Load from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class QueryConfig:
    """
    Configuration for SPHEREx data query and processing.

    Smart loading: If a saved state file exists for the source, parameters are loaded
    with priority: user-provided > saved state > defaults.
    """

    source: Source
    output_dir: Path = field(default_factory=Path.cwd)
    bands: Optional[List[str]] = None  # e.g., ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    aperture_diameter: float = 3.0  # pixels (default 3 pixel diameter)
    max_download_workers: int = 4
    max_processing_workers: int = 10  # Number of workers for photometry processing
    cutout_size: Optional[str] = None  # e.g., "200px", "100,200px", "3arcmin", "0.1"
    cutout_center: Optional[str] = None  # e.g., "70,20", "300.5,120px" (optional, defaults to source position)
    sigma_threshold: float = 5.0  # Minimum SNR (flux/flux_err) for quality control
    bad_flags: List[int] = field(default_factory=lambda: [0, 1, 2, 6, 7, 9, 10, 11, 15])  # Flags to reject
    use_magnitude: bool = False  # If True, plot AB magnitude instead of flux (default: False)
    show_errorbars: bool = True  # If True, show errorbars on plots (default: True)
    advanced_params_file: Optional[Path] = None  # Path to JSON file with advanced parameters
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)  # Advanced configuration
    _auto_loaded: bool = field(default=False, init=False, repr=False)  # Internal flag

    def __post_init__(self):
        # Convert to Path if string
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Load advanced parameters from file if provided
        if self.advanced_params_file is not None:
            from ..utils.params import load_advanced_config

            self.advanced_params_file = Path(self.advanced_params_file)
            self.advanced = load_advanced_config(self.advanced_params_file)
            logger.info(f"Loaded advanced parameters from {self.advanced_params_file}")

        # Validate bands
        valid_bands = ["D1", "D2", "D3", "D4", "D5", "D6"]
        if self.bands:
            invalid = set(self.bands) - set(valid_bands)
            if invalid:
                raise ValueError(f"Invalid bands: {invalid}. Valid bands are: {valid_bands}")

        # Validate aperture diameter
        if self.aperture_diameter <= 0:
            raise ValueError(f"Aperture diameter must be positive, got {self.aperture_diameter}")

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

        # Validate quality control parameters
        if self.sigma_threshold <= 0:
            raise ValueError(f"Sigma threshold must be positive, got {self.sigma_threshold}")

    @classmethod
    def from_saved_state(cls, source_name: str, output_dir: Path, **user_overrides) -> "QueryConfig":
        """
        Create QueryConfig by loading from saved state with optional overrides.

        Parameters are loaded with priority: user_overrides > saved state > defaults.

        Parameters
        ----------
        source_name : str
            Name of the source (used to find {source_name}.json)
        output_dir : Path
            Output directory where state file is located
        **user_overrides
            Any parameters to override from saved state

        Returns
        -------
        QueryConfig
            Configuration loaded from state with overrides applied

        Examples
        --------
        >>> # Load everything from saved state
        >>> config = QueryConfig.from_saved_state("cloverleaf", Path("output"))
        >>>
        >>> # Load from state but override aperture_diameter
        >>> config = QueryConfig.from_saved_state(
        ...     "cloverleaf", Path("output"),
        ...     aperture_diameter=5.0
        ... )
        """
        from ..utils.helpers import load_json

        output_dir = Path(output_dir)
        state_file = output_dir / f"{source_name}.json"

        if not state_file.exists():
            # No saved state - create from scratch with user overrides
            source = Source(ra=user_overrides.get("ra", 0.0), dec=user_overrides.get("dec", 0.0), name=source_name)
            # Remove ra/dec from overrides since they're in source
            user_overrides.pop("ra", None)
            user_overrides.pop("dec", None)

            return cls(source=source, output_dir=output_dir, **user_overrides)

        # Load saved state
        saved_data = load_json(state_file)
        saved_config = saved_data.get("config", {})

        # Create source from saved or user data
        if "source" in saved_config:
            source = Source(
                ra=user_overrides.get("ra", saved_config["source"]["ra"]),
                dec=user_overrides.get("dec", saved_config["source"]["dec"]),
                name=source_name,
            )
        else:
            source = Source(ra=user_overrides.get("ra", 0.0), dec=user_overrides.get("dec", 0.0), name=source_name)

        # Remove ra/dec from overrides
        user_overrides.pop("ra", None)
        user_overrides.pop("dec", None)

        # Build kwargs with priority: user > saved > defaults
        kwargs = {
            "source": source,
            "output_dir": output_dir,
        }

        # For each parameter, use: user_overrides > saved_config > class default
        param_names = [
            "bands",
            "aperture_diameter",
            "max_download_workers",
            "max_processing_workers",
            "cutout_size",
            "cutout_center",
            "sigma_threshold",
            "bad_flags",
            "use_magnitude",
            "show_errorbars",
        ]

        for param in param_names:
            if param in user_overrides:
                # User explicitly provided this parameter
                kwargs[param] = user_overrides[param]
            elif param in saved_config:
                # Load from saved state
                kwargs[param] = saved_config[param]
            # Otherwise, use class default (don't set in kwargs)

        config = cls(**kwargs)
        config._auto_loaded = True  # Mark as auto-loaded
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
    """
    Result from photometry on a single observation.

    Supports both aperture and model-based photometry methods.
    """

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
    photometry_method: str = "aperture"  # "aperture" or "model"
    model_type: Optional[str] = None  # Model used: "point_source", "sersic", etc.
    model_chi2: Optional[float] = None  # Reduced chi-squared for model fit
    psf_zone_id: Optional[int] = None  # PSF zone used (1-121) for point source models

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
    """State for resumable pipeline execution."""

    stage: str  # Current stage: 'query', 'download', 'processing', 'visualization', 'complete'
    config: QueryConfig
    query_results: Optional[QueryResults] = None
    downloaded_files: List[Path] = field(default_factory=list)
    photometry_results: List[PhotometryResult] = field(default_factory=list)
    csv_path: Optional[Path] = None
    plot_path: Optional[Path] = None
    completed_stages: List[str] = field(default_factory=list)  # Track completed stages
    pipeline_stages: List[str] = field(
        default_factory=lambda: ["query", "download", "processing", "visualization"]
    )  # Configurable stages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage": self.stage,
            "completed_stages": self.completed_stages,
            "pipeline_stages": self.pipeline_stages,
            "config": {
                "source": {"ra": self.config.source.ra, "dec": self.config.source.dec, "name": self.config.source.name},
                "output_dir": str(self.config.output_dir),
                "bands": self.config.bands,
                "aperture_diameter": self.config.aperture_diameter,
                "max_download_workers": self.config.max_download_workers,
                "max_processing_workers": self.config.max_processing_workers,
                "cutout_size": self.config.cutout_size,
                "cutout_center": self.config.cutout_center,
                "sigma_threshold": self.config.sigma_threshold,
                "bad_flags": self.config.bad_flags,
                "use_magnitude": self.config.use_magnitude,
                "show_errorbars": self.config.show_errorbars,
            },
            "query_results": {
                "observations": [
                    {
                        "obs_id": obs.obs_id,
                        "band": obs.band,
                        "mjd": obs.mjd,
                        "wavelength_min": obs.wavelength_min,
                        "wavelength_max": obs.wavelength_max,
                        "download_url": obs.download_url,
                        "t_min": obs.t_min,
                        "t_max": obs.t_max,
                    }
                    for obs in self.query_results.observations
                ]
                if self.query_results
                else [],
                "query_time": self.query_results.query_time.isoformat() if self.query_results else None,
                "total_size_gb": self.query_results.total_size_gb if self.query_results else 0,
                "time_span_days": self.query_results.time_span_days if self.query_results else 0,
                "band_counts": self.query_results.band_counts if self.query_results else {},
            }
            if self.query_results
            else None,
            "downloaded_files": [str(p) for p in self.downloaded_files],
            "photometry_results": [
                {
                    "obs_id": pr.obs_id,
                    "mjd": pr.mjd,
                    "flux": pr.flux,
                    "flux_error": pr.flux_error,
                    "wavelength": pr.wavelength,
                    "bandwidth": pr.bandwidth,
                    "flag": pr.flag,
                    "pix_x": pr.pix_x,
                    "pix_y": pr.pix_y,
                    "band": pr.band,
                    "mag_ab": pr.mag_ab,
                    "mag_ab_error": pr.mag_ab_error,
                    "photometry_method": pr.photometry_method,
                    "model_type": pr.model_type,
                    "model_chi2": pr.model_chi2,
                    "psf_zone_id": pr.psf_zone_id,
                }
                for pr in self.photometry_results
            ],
            "csv_path": str(self.csv_path) if self.csv_path else None,
            "plot_path": str(self.plot_path) if self.plot_path else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        """Create from dictionary."""
        # Reconstruct config
        source = Source(
            ra=data["config"]["source"]["ra"],
            dec=data["config"]["source"]["dec"],
            name=data["config"]["source"].get("name"),
        )
        config = QueryConfig(
            source=source,
            output_dir=Path(data["config"]["output_dir"]),
            bands=data["config"].get("bands"),
            aperture_diameter=data["config"]["aperture_diameter"],
            max_download_workers=data["config"]["max_download_workers"],
            max_processing_workers=data["config"].get("max_processing_workers", 10),
            cutout_size=data["config"].get("cutout_size"),
            cutout_center=data["config"].get("cutout_center"),
            sigma_threshold=data["config"].get("sigma_threshold", 5.0),
            bad_flags=data["config"].get("bad_flags", [0, 1, 2, 6, 7, 9, 10, 11, 15]),
            use_magnitude=data["config"].get("use_magnitude", False),
            show_errorbars=data["config"].get("show_errorbars", True),
        )

        # Reconstruct query results
        query_results = None
        if data.get("query_results"):
            observations = [ObservationInfo(**obs) for obs in data["query_results"]["observations"]]
            query_results = QueryResults(
                observations=observations,
                query_time=datetime.fromisoformat(data["query_results"]["query_time"]),
                source=source,
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
            pipeline_stages=data.get("pipeline_stages", ["query", "download", "processing", "visualization"]),
        )
