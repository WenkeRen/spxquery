"""
Configuration and data models for SPXQuery package.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    Configuration for pysersic model-based photometry.

    All parameters are exposed via custom_params.json for advanced users.
    This follows the same philosophy as PhotometryConfig and VisualizationConfig.

    Attributes
    ----------
    profile_type : str
        Source profile model. Default: 'pointsource' (PRIMARY use case for undersampled SPHEREx data).
        Options: 'pointsource', 'sersic', 'exp', 'dev', 'sersic_exp', 'doublesersic'
    sky_type : str
        Background model. Default: 'tilted-plane' (accounts for gradients).
        Options: 'tilted-plane', 'flat', 'none'
    inference_method : str
        Bayesian inference method. Default: 'map' (fast).
        Options: 'map', 'laplace', 'svi-mvn', 'svi-flow', 'nuts'
    num_samples : int
        Number of posterior samples for MCMC/SVI. Default: 1000.
    num_warmup : int
        Number of MCMC warmup steps. Default: 500.
    num_chains : int
        Number of MCMC chains. Default: 2.
    target_accept_prob : float
        NUTS target acceptance probability. Default: 0.8.
    max_tree_depth : int
        NUTS maximum tree depth. Default: 10.
    svi_num_steps : int
        Number of SVI optimization steps. Default: 10000.
    svi_learning_rate : float
        SVI learning rate. Default: 0.01.
    use_aperture_prior : bool
        Use aperture photometry as flux/position prior. Default: True.
    allow_position_shift : bool
        Allow center to move from input position. Default: True.
    position_prior_sigma : float
        Gaussian sigma for position prior (pixels). Default: 2.0.
    flux_prior_scale_factor : float
        Multiply aperture flux_err by this for prior width. Default: 3.0.
    cutout_size : int
        Size of cutout around source for fitting (pixels). Default: 50.
        Applied during processing after FITS download. Square cutout: cutout_size x cutout_size.
    psf_oversample : int
        SPHEREx PSF oversampling factor. Default: 10.
    psf_selection : str
        PSF selection method. Default: 'nearest'.
        Options: 'nearest' (find closest PSF), 'interpolate' (future), 'center' (testing only)
    psf_downsample_method : str
        PSF downsampling method. Default: 'rebin'.
        Options: 'rebin' (block_reduce), 'interpolate' (scipy zoom)
    loss_function : str
        Loss function for fitting. Default: 'gaussian'.
        Options: 'gaussian', 'student_t', 'gaussian_mixture'
    student_t_df : int
        Degrees of freedom for Student-t loss. Default: 5.
    renderer : str
        PSF rendering strategy. Default: 'hybrid'.
        Options: 'hybrid', 'fourier', 'pixel'
    hybrid_frac_start : float
        HybridRenderer: smallest Gaussian fraction of r_eff. Default: 1e-2.
    hybrid_frac_end : float
        HybridRenderer: largest Gaussian fraction. Default: 15.0.
    hybrid_n_sigma : int
        HybridRenderer: number of Gaussian components. Default: 15.
    hybrid_num_pixel_render : int
        HybridRenderer: components to render in pixel space. Default: 3.
    hybrid_precision : int
        HybridRenderer: decomposition precision. Default: 10.
    use_mpi : bool
        Use MPI for parallelization (requires mpi4py). Default: False.
    use_gpu : bool
        Use GPU acceleration via JAX (requires jaxlib[cuda]). Default: False.
    device_id : int
        GPU device ID if use_gpu=True. Default: 0.
    output_filename : str
        Output CSV filename. Default: 'model_lightcurve.csv'.
    save_model_params : bool
        Save Sersic parameters to separate CSV. Default: True.
    save_chains : bool
        Deprecated: use save_posterior_chains. Default: False.
    save_diagnostic_plots : bool
        Save detailed plots for each source. Default: False.
    plot_image : bool
        Plot input image/mask/PSF (if diagnostics enabled). Default: True.
    plot_residual : bool
        Plot model residuals (if diagnostics enabled). Default: True.
    plot_corner : bool
        Corner plot for posterior (slow, MCMC/SVI only). Default: False.
    diagnostic_plot_format : str
        File format for diagnostic plots. Default: 'png'.
        Options: 'png', 'pdf', 'jpg'
    diagnostic_plot_dpi : int
        DPI for saved figures. Default: 150.
    save_posterior_chains : bool
        Save full MCMC/SVI chains (use pysersic ASDF format). Default: False.
        Chains saved to: diagnostics/chains/{band}/{filename}_chains.asdf
    """

    # Model specification
    profile_type: str = "pointsource"
    sky_type: str = "tilted-plane"

    # Inference method
    inference_method: str = "map"
    num_samples: int = 1000
    num_warmup: int = 500
    num_chains: int = 2

    # MCMC sampler parameters (advanced)
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10

    # SVI parameters (advanced)
    svi_num_steps: int = 10000
    svi_learning_rate: float = 0.01

    # Prior configuration
    use_aperture_prior: bool = True
    allow_position_shift: bool = True
    position_prior_sigma: float = 2.0
    flux_prior_scale_factor: float = 3.0

    # Cutout configuration
    cutout_size: int = 50  # Size of cutout around source for fitting (pixels)

    # PSF handling
    psf_oversample: int = 10
    psf_selection: str = "nearest"
    psf_downsample_method: str = "rebin"

    # Loss function
    loss_function: str = "gaussian"
    student_t_df: int = 5

    # Rendering (advanced)
    renderer: str = "hybrid"
    hybrid_frac_start: float = 1e-2
    hybrid_frac_end: float = 15.0
    hybrid_n_sigma: int = 15
    hybrid_num_pixel_render: int = 3
    hybrid_precision: int = 10

    # Parallelization (future)
    use_mpi: bool = False
    use_gpu: bool = False
    device_id: int = 0

    # Output
    output_filename: str = "model_lightcurve.csv"
    save_model_params: bool = True
    save_chains: bool = False  # Deprecated

    # Diagnostic visualization (pysersic built-in plots)
    save_diagnostic_plots: bool = False
    plot_image: bool = True
    plot_residual: bool = True
    plot_corner: bool = False
    diagnostic_plot_format: str = "png"
    diagnostic_plot_dpi: int = 150

    # Posterior chain storage (similar to diagnostic plots)
    save_posterior_chains: bool = False

    def __post_init__(self):
        """Validate parameters."""
        # Validate profile_type
        valid_profiles = ["pointsource", "sersic", "exp", "dev", "sersic_exp", "doublesersic"]
        if self.profile_type not in valid_profiles:
            raise ValueError(f"profile_type must be one of {valid_profiles}, got '{self.profile_type}'")

        # Validate sky_type
        valid_sky_types = ["tilted-plane", "flat", "none"]
        if self.sky_type not in valid_sky_types:
            raise ValueError(f"sky_type must be one of {valid_sky_types}, got '{self.sky_type}'")

        # Validate inference_method
        valid_methods = ["map", "laplace", "svi-mvn", "svi-flow", "nuts"]
        if self.inference_method not in valid_methods:
            raise ValueError(f"inference_method must be one of {valid_methods}, got '{self.inference_method}'")

        # Validate numeric parameters
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {self.num_samples}")
        if self.num_warmup < 0:
            raise ValueError(f"num_warmup must be >= 0, got {self.num_warmup}")
        if self.num_chains <= 0:
            raise ValueError(f"num_chains must be > 0, got {self.num_chains}")
        if not 0 < self.target_accept_prob < 1:
            raise ValueError(f"target_accept_prob must be in (0, 1), got {self.target_accept_prob}")
        if self.max_tree_depth <= 0:
            raise ValueError(f"max_tree_depth must be > 0, got {self.max_tree_depth}")
        if self.svi_num_steps <= 0:
            raise ValueError(f"svi_num_steps must be > 0, got {self.svi_num_steps}")
        if self.svi_learning_rate <= 0:
            raise ValueError(f"svi_learning_rate must be > 0, got {self.svi_learning_rate}")
        if self.position_prior_sigma <= 0:
            raise ValueError(f"position_prior_sigma must be > 0, got {self.position_prior_sigma}")
        if self.flux_prior_scale_factor <= 0:
            raise ValueError(f"flux_prior_scale_factor must be > 0, got {self.flux_prior_scale_factor}")
        if self.cutout_size <= 0:
            raise ValueError(f"cutout_size must be > 0, got {self.cutout_size}")
        if self.psf_oversample <= 0:
            raise ValueError(f"psf_oversample must be > 0, got {self.psf_oversample}")
        if self.student_t_df <= 0:
            raise ValueError(f"student_t_df must be > 0, got {self.student_t_df}")
        if self.device_id < 0:
            raise ValueError(f"device_id must be >= 0, got {self.device_id}")
        if self.diagnostic_plot_dpi <= 0:
            raise ValueError(f"diagnostic_plot_dpi must be > 0, got {self.diagnostic_plot_dpi}")

        # Validate psf_selection
        valid_psf_methods = ["nearest", "interpolate", "center"]
        if self.psf_selection not in valid_psf_methods:
            raise ValueError(f"psf_selection must be one of {valid_psf_methods}, got '{self.psf_selection}'")

        # Validate psf_downsample_method
        valid_downsample = ["rebin", "interpolate"]
        if self.psf_downsample_method not in valid_downsample:
            raise ValueError(f"psf_downsample_method must be one of {valid_downsample}, got '{self.psf_downsample_method}'")

        # Validate loss_function
        valid_loss = ["gaussian", "student_t", "gaussian_mixture"]
        if self.loss_function not in valid_loss:
            raise ValueError(f"loss_function must be one of {valid_loss}, got '{self.loss_function}'")

        # Validate renderer
        valid_renderers = ["hybrid", "fourier", "pixel"]
        if self.renderer not in valid_renderers:
            raise ValueError(f"renderer must be one of {valid_renderers}, got '{self.renderer}'")

        # Validate diagnostic_plot_format
        valid_formats = ["png", "pdf", "jpg"]
        if self.diagnostic_plot_format not in valid_formats:
            raise ValueError(f"diagnostic_plot_format must be one of {valid_formats}, got '{self.diagnostic_plot_format}'")

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
