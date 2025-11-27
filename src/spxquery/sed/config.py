"""
Configuration dataclasses for SED reconstruction.

This module provides configuration classes for spectral reconstruction
from SPHEREx narrow-band photometry using convex optimization.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# SPHEREx detector wavelength ranges (microns) - hardcoded from mission specifications
# These ranges are used for spectral reconstruction instead of data-derived boundaries
# Ref: https://doi.org/10.1117/12.3018463
DETECTOR_WAVELENGTH_RANGES = {
    "D1": (0.75, 1.12),  # Band 1: 0.75-1.12 μm, R=41
    "D2": (1.10, 1.65),  # Band 2: 1.10-1.65 μm, R=41
    "D3": (1.63, 2.44),  # Band 3: 1.63-2.44 μm, R=41
    "D4": (2.40, 3.85),  # Band 4: 2.40-3.85 μm, R=35
    "D5": (3.81, 4.43),  # Band 5: 3.81-4.43 μm, R=110
    "D6": (4.41, 5.01),  # Band 6: 4.41-5.01 μm, R=130
}


@dataclass
class SEDConfig:
    """
    Configuration for SED reconstruction from narrow-band photometry.

    This class controls all aspects of spectral reconstruction using
    regularized least-squares optimization with Stationary Wavelet Transform (SWT)
    multi-scale regularization and CVXPY.

    The SWT implementation uses a 4-group hyperparameter system:
    - Group A: Approximation coefficients (low-frequency continuum)
    - Group B: Coarse detail coefficients (large-scale features)
    - Group C: Medium detail coefficients (emission lines, main features)
    - Group D: Fine detail coefficients (high-frequency noise)

    Parameters
    ----------
    lambda_continuum : float
        Regularization weight for approximation coefficients (low-frequency continuum).
        Lower values preserve more continuum structure. Default: 0.1.
    lambda_low_features : float
        Regularization weight for coarse detail coefficients (large-scale features).
        Controls broad emission/absorption features. Default: 1.0.
    lambda_main_features : float
        Regularization weight for medium detail coefficients (emission lines).
        Controls main spectral features and line strengths. Default: 5.0.
    lambda_noise : float
        Regularization weight for fine detail coefficients (high-frequency noise).
        Higher values suppress more noise. Default: 100.0.
    wavelet_family : str
        Wavelet basis function family. Default: 'sym6' (Symlet-6).
        Common choices: 'sym4', 'sym6', 'sym8', 'db4', 'db6', 'db8'.
    wavelet_level : Optional[int]
        Number of wavelet decomposition levels. If None, auto-detect using
        pywt.dwt_max_level(). Default: None (auto-detect).
    wavelet_boundary_mode : str
        Boundary mode for wavelet transform. Default: 'symmetric'.
        Options: 'symmetric' (recommended for spectra), 'reflect', 'periodic', 'zero', 'smooth'.
        Symmetric mode reflects signal at boundaries to avoid edge artifacts.
    resolution_samples : int
        Number of wavelength bins in reconstructed spectrum.
        Must be even for SWT (symmetric padding requirements).
        Default: 1020 (approximately 2-pixel sampling of detector).
    auto_tune : bool
        Enable automatic hyperparameter tuning via grouped grid search.
        If False, uses grouped lambda values as specified. Default: False.
    lambda_continuum_grid : List[float]
        Grid search values for continuum regularization when auto_tune=True.
        Default: [0.01, 0.1, 1.0].
    lambda_low_features_grid : List[float]
        Grid search values for low-feature regularization when auto_tune=True.
        Default: [0.1, 1.0, 10.0].
    lambda_main_features_grid : List[float]
        Grid search values for main-feature regularization when auto_tune=True.
        Default: [1.0, 10.0, 100.0].
    lambda_noise_grid : List[float]
        Grid search values for noise regularization when auto_tune=True.
        Default: [10.0, 100.0, 1000.0].
    validation_fraction : float
        Fraction of data reserved for validation during tuning.
        Must be between 0 and 1. Default: 0.2 (80/20 train/test split).
    sigma_threshold : float
        Minimum SNR (flux/flux_error) for quality filtering.
        Measurements below this threshold are excluded. Default: 5.0.
    bad_flags : List[int]
        Pixel flag bits to reject during data loading.
        Default: [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19] (standard SPHEREx masks).
    enable_sigma_clip : bool
        Enable rolling window sigma clipping to remove outliers using
        MAD-based robust statistics (Median Absolute Deviation).
        Default: True.
    sigma_clip_sigma : float
        Number of MAD-equivalent standard deviations for sigma clipping threshold.
        Measurements beyond this threshold (in rolling windows) are rejected as outliers.
        Default: 3.0 (3-sigma clipping with robust statistics).
    sigma_clip_window : int
        Rolling window size (number of measurements) for local MAD calculation.
        Should be odd for centered windows. Larger windows are more robust
        but less sensitive to local variations. Default: 21.
    sigma_clip_max_iterations : int
        Maximum number of iterative sigma clipping passes per band.
        Clipping repeats until no outliers found or max iterations reached.
        Prevents infinite loops. Default: 10.
    filter_profile : str
        Narrow-band filter shape. Options: 'boxcar' (rectangular window).
        Future: 'gaussian', 'tophat'. Default: 'boxcar'.
    solver : str
        CVXPY solver backend. Options: 'CLARABEL', 'SCS', 'OSQP', 'ECOS'.
        Default: 'CLARABEL' (recommended, improved version of ECOS).
    solver_verbose : bool
        Print solver iteration progress. Default: False.
    epsilon_weight : float
        Small constant added to weights to avoid division by zero.
        w_i = 1 / (sigma_i + epsilon). Default: 1e-10.

    Attributes
    ----------
    All parameters above are stored as instance attributes.

    Examples
    --------
    >>> config = SEDConfig(lambda_continuum=0.1, lambda_noise=100.0, auto_tune=False)
    >>> config.to_yaml_file("my_config.yaml")
    >>> loaded = SEDConfig.from_yaml_file("my_config.yaml")
    """

    # SWT 4-group hyperparameters
    lambda_continuum: float = 0.1
    lambda_low_features: float = 1.0
    lambda_main_features: float = 5.0
    lambda_noise: float = 100.0

    # Backward compatibility (deprecated)
    lambda_low: float = 0.1
    lambda_detail: float = 10.0

    # Wavelet configuration
    wavelet_family: str = "sym6"
    wavelet_level: Optional[int] = None
    wavelet_boundary_mode: str = "symmetric"

    # Resolution control
    resolution_samples: int = 1020

    # Auto-tuning parameters
    auto_tune: bool = False
    lambda_continuum_grid: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0])
    lambda_low_features_grid: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    lambda_main_features_grid: List[float] = field(default_factory=lambda: [1.0, 10.0, 100.0])
    lambda_noise_grid: List[float] = field(default_factory=lambda: [10.0, 100.0, 1000.0])
    validation_fraction: float = 0.2

    # Backward compatibility (deprecated grid parameters)
    lambda_low_grid: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    lambda_detail_grid: List[float] = field(default_factory=lambda: [1.0, 10.0, 100.0, 1000.0])

    # Quality control
    sigma_threshold: float = 3.0
    bad_flags: List[int] = field(default_factory=lambda: [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19])

    # Sigma clipping (outlier removal using rolling MAD-based robust statistics)
    enable_sigma_clip: bool = True
    sigma_clip_sigma: float = 3.0
    sigma_clip_window: int = 21
    sigma_clip_max_iterations: int = 10

    # Physical modeling
    filter_profile: str = "boxcar"

    # Solver configuration
    solver: str = "CLARABEL"
    solver_verbose: bool = False
    epsilon_weight: float = 1e-10

    # Spatial weighting for L1 regularization (to reduce Gibbs Phenomenon at edges)
    spatial_weight_enabled: bool = True
    padding_region_weight: float = 0.0  # Completely suppress regularization in padding
    science_region_weight: float = 1.0  # Full regularization in scientific regions
    transition_width: int = 0  # Width of transition zone (pixels), 0 = hard cutoffs

    # ==========================================
    # PyTorch Solver Configuration (Deep Spectral Prior)
    # ==========================================
    solver_type: str = "cvxpy"  # Options: "cvxpy", "torch"

    # Global reconstruction parameters
    wavelength_range: Tuple[float, float] = (0.75, 5.0)
    global_resolution: int = 3000
    device: str = "mps"  # Options: "cpu", "cuda", "mps"

    # Optimization parameters
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    epochs: int = 3000

    # Deep Prior Architecture
    dip_noise_std: float = 0.1
    dip_filters: int = 32
    dip_depth: int = 3

    # Regularization (CWT)
    regularization_weight: float = 1.0
    cwt_scales: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate SWT 4-group regularization weights
        if self.lambda_continuum < 0:
            raise ValueError(f"lambda_continuum must be non-negative, got {self.lambda_continuum}")
        if self.lambda_low_features < 0:
            raise ValueError(f"lambda_low_features must be non-negative, got {self.lambda_low_features}")
        if self.lambda_main_features < 0:
            raise ValueError(f"lambda_main_features must be non-negative, got {self.lambda_main_features}")
        if self.lambda_noise < 0:
            raise ValueError(f"lambda_noise must be non-negative, got {self.lambda_noise}")

        # Validate backward compatibility parameters
        if self.lambda_low < 0:
            raise ValueError(f"lambda_low must be non-negative, got {self.lambda_low}")
        if self.lambda_detail < 0:
            raise ValueError(f"lambda_detail must be non-negative, got {self.lambda_detail}")

        # Validate wavelet configuration
        import pywt

        try:
            valid_wavelets = pywt.wavelist(kind="discrete")
        except AttributeError:
            # Fallback for older PyWavelets versions
            valid_wavelets = ["sym4", "sym6", "sym8", "db4", "db6", "db8", "coif1", "coif2", "bior1.3", "rbio1.3"]

        if self.wavelet_family not in valid_wavelets:
            raise ValueError(
                f"wavelet_family '{self.wavelet_family}' not recognized. "
                f"Valid choices include: {', '.join(valid_wavelets[:15])}... "
                f"(check pywt.wavelist() for full list)"
            )

        if self.wavelet_level is not None:
            if self.wavelet_level < 1:
                raise ValueError(f"wavelet_level must be >= 1 if specified, got {self.wavelet_level}")

        # Validate resolution
        if self.resolution_samples <= 0:
            raise ValueError(f"resolution_samples must be positive, got {self.resolution_samples}")
        if self.resolution_samples % 2 != 0:
            raise ValueError(
                f"resolution_samples must be even for SWT (got {self.resolution_samples}). "
                f"SWT requires symmetric padding, which works best with even signal lengths."
            )
        if self.resolution_samples > 10000:
            raise ValueError(
                f"resolution_samples too large ({self.resolution_samples}), may cause memory issues. Consider < 10000."
            )

        # Validate validation fraction
        if not 0 < self.validation_fraction < 1:
            raise ValueError(f"validation_fraction must be in (0, 1), got {self.validation_fraction}")

        # Validate grid search ranges
        if self.auto_tune:
            # Validate new grouped parameters
            if not self.lambda_continuum_grid:
                raise ValueError("lambda_continuum_grid cannot be empty when auto_tune=True")
            if not self.lambda_low_features_grid:
                raise ValueError("lambda_low_features_grid cannot be empty when auto_tune=True")
            if not self.lambda_main_features_grid:
                raise ValueError("lambda_main_features_grid cannot be empty when auto_tune=True")
            if not self.lambda_noise_grid:
                raise ValueError("lambda_noise_grid cannot be empty when auto_tune=True")
            if any(x < 0 for x in self.lambda_continuum_grid):
                raise ValueError("All lambda_continuum_grid values must be non-negative")
            if any(x < 0 for x in self.lambda_low_features_grid):
                raise ValueError("All lambda_low_features_grid values must be non-negative")
            if any(x < 0 for x in self.lambda_main_features_grid):
                raise ValueError("All lambda_main_features_grid values must be non-negative")
            if any(x < 0 for x in self.lambda_noise_grid):
                raise ValueError("All lambda_noise_grid values must be non-negative")

            # Validate backward compatibility parameters
            if not self.lambda_low_grid:
                raise ValueError("lambda_low_grid cannot be empty when auto_tune=True")
            if not self.lambda_detail_grid:
                raise ValueError("lambda_detail_grid cannot be empty when auto_tune=True")
            if any(x < 0 for x in self.lambda_low_grid):
                raise ValueError("All lambda_low_grid values must be non-negative")
            if any(x < 0 for x in self.lambda_detail_grid):
                raise ValueError("All lambda_detail_grid values must be non-negative")

        # Validate quality control
        if self.sigma_threshold < 0:
            raise ValueError(f"sigma_threshold must be non-negative, got {self.sigma_threshold}")

        # Validate sigma clipping parameters
        if self.sigma_clip_sigma <= 0:
            raise ValueError(f"sigma_clip_sigma must be positive, got {self.sigma_clip_sigma}")
        if self.sigma_clip_window < 3:
            raise ValueError(f"sigma_clip_window must be >= 3, got {self.sigma_clip_window}")
        if self.sigma_clip_window % 2 == 0:
            raise ValueError(
                f"sigma_clip_window should be odd for centered rolling windows, got {self.sigma_clip_window}"
            )
        if self.sigma_clip_max_iterations < 1:
            raise ValueError(f"sigma_clip_max_iterations must be >= 1, got {self.sigma_clip_max_iterations}")

        # Validate wavelet boundary mode
        valid_modes = [
            "zero",
            "constant",
            "symmetric",
            "periodic",
            "smooth",
            "periodization",
            "reflect",
            "antisymmetric",
            "antireflect",
        ]
        if self.wavelet_boundary_mode not in valid_modes:
            raise ValueError(f"wavelet_boundary_mode must be one of {valid_modes}, got '{self.wavelet_boundary_mode}'")

        # Validate filter profile
        valid_filters = ["boxcar"]
        if self.filter_profile not in valid_filters:
            raise ValueError(f"filter_profile must be one of {valid_filters}, got '{self.filter_profile}'")

        # Validate solver
        valid_solvers = ["ECOS", "SCS", "OSQP", "CLARABEL"]
        if self.solver not in valid_solvers:
            raise ValueError(f"solver must be one of {valid_solvers}, got '{self.solver}'")

        # Validate epsilon
        if self.epsilon_weight <= 0:
            raise ValueError(f"epsilon_weight must be positive, got {self.epsilon_weight}")

        # Validate spatial weighting parameters
        if not (0.0 <= self.padding_region_weight <= 1.0):
            raise ValueError(f"padding_region_weight must be between 0.0 and 1.0, got {self.padding_region_weight}")
        if not (0.0 <= self.science_region_weight <= 1.0):
            raise ValueError(f"science_region_weight must be between 0.0 and 1.0, got {self.science_region_weight}")
        if self.transition_width < 0:
            raise ValueError(f"transition_width must be non-negative, got {self.transition_width}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of configuration.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SEDConfig":
        """
        Create configuration from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with configuration parameters.

        Returns
        -------
        SEDConfig
            Configuration instance.
        """
        # Filter to only valid fields to handle extra keys gracefully
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_yaml_file(self, filepath: Path) -> Path:
        """
        Save configuration to YAML file.

        Parameters
        ----------
        filepath : Path
            Output YAML file path.

        Returns
        -------
        Path
            Path to written file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        return filepath

    @classmethod
    def from_yaml_file(cls, filepath: Path) -> "SEDConfig":
        """
        Load configuration from YAML file.

        Parameters
        ----------
        filepath : Path
            Input YAML file path.

        Returns
        -------
        SEDConfig
            Configuration instance.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def copy_with_overrides(self, **kwargs) -> "SEDConfig":
        """
        Create a copy with specified parameters overridden.

        Parameters
        ----------
        **kwargs
            Parameters to override in the copy.

        Returns
        -------
        SEDConfig
            New configuration instance with overrides applied.

        Examples
        --------
        >>> config = SEDConfig(lambda_low=0.1)
        >>> tuned = config.copy_with_overrides(lambda_low=1.0, lambda_detail=50.0)
        """
        current_dict = self.to_dict()
        current_dict.update(kwargs)
        return self.from_dict(current_dict)


def export_default_sed_config(output_dir: Path, filename: str = "sed_config.yaml") -> Path:
    """
    Export default SED configuration template to YAML file.

    Users can customize this template and load it for reconstruction.

    Parameters
    ----------
    output_dir : Path
        Directory to save configuration file.
    filename : str
        Output filename. Default: 'sed_config.yaml'.

    Returns
    -------
    Path
        Path to exported configuration file.

    Examples
    --------
    >>> config_path = export_default_sed_config("output/")
    >>> # User edits sed_config.yaml
    >>> config = SEDConfig.from_yaml_file(config_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SEDConfig()
    filepath = output_dir / filename
    config.to_yaml_file(filepath)

    return filepath
