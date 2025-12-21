"""
Configuration dataclasses for SED reconstruction.

This module provides configuration classes for spectral reconstruction
from SPHEREx narrow-band photometry using PyTorch-based Deep Image Prior.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    PyTorch-based Deep Image Prior (DIP) with Continuous Wavelet Transform (CWT)
    regularization for global spectrum reconstruction across all SPHEREx detector bands.

    Parameters
    ----------
    wavelength_range : Tuple[float, float]
        Global wavelength range for reconstruction (microns). Default: (0.75, 5.0).
    global_resolution : int
        Number of wavelength bins in reconstructed global spectrum.
        Default: 3000 (high-resolution across full SPHEREx range).
    device : str
        PyTorch computation device. Options: 'cpu', 'cuda', 'mps'.
        Default: 'mps' (Apple Silicon GPU). Use 'cuda' for NVIDIA GPUs.
    optimizer : str
        PyTorch optimizer for DIP training. Default: 'Adam'.
    learning_rate : float
        Peak learning rate for optimizer. Default: 0.001.
    epochs : int
        Number of training iterations. Default: 3000.
    learning_rate_scheduler_type : str
        Learning rate scheduler type. Options: 'none', 'cosine', 'cosine_warmup', 'warmup'.
        Default: 'cosine_warmup' (5% linear warmup + cosine decay).
    learning_rate_warmup_epochs : int
        Number of epochs for linear warmup phase. Default: 150 (5% of 3000).
    learning_rate_min_factor : float
        Minimum learning rate as fraction of peak for cosine decay.
        Default: 0.01 (1% of peak learning rate).
    dip_noise_std : float
        Standard deviation of input noise for Deep Image Prior. Default: 0.1.
    dip_filters : int
        Number of filters in U-Net architecture. Default: 32.
    dip_depth : int
        Depth of U-Net architecture (number of downsampling stages). Default: 3.
    dip_noise_jitter_initial_ratio : Optional[float]
        Initial jitter as ratio of dip_noise_std. Default: 0.3.
        None means no jitter is applied. 0.3 means jitter = 0.3 x dip_noise_std.
    dip_noise_jitter_min_ratio : Optional[float]
        Minimum jitter ratio for linear decay. Default: None.
        None means no decay is applied. Otherwise, jitter decays linearly from
        initial_ratio to min_ratio over the course of training.
    regularization_weight : float
        Weight for CWT regularization term. Default: 1.0.
    cwt_scales : List[float]
        Scales for Continuous Wavelet Transform regularization. Default: [1.0, 2.0, 3.0].
    sigma_threshold : float
        Minimum SNR (flux/flux_error) for quality filtering.
        Measurements below this threshold are excluded. Default: 3.0.
    bad_flags : List[int]
        Pixel flag bits to reject during data loading.
        Default: [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19] (standard SPHEREx masks).
    enable_sigma_clip : bool
        Enable rolling window sigma clipping to remove outliers using
        MAD-based robust statistics (Median Absolute Deviation). Default: True.
    sigma_clip_sigma : float
        Number of MAD-equivalent standard deviations for sigma clipping threshold.
        Default: 3.0.
    sigma_clip_window : int
        Rolling window size for local MAD calculation. Should be odd. Default: 21.
    sigma_clip_max_iterations : int
        Maximum number of iterative sigma clipping passes. Default: 10.
    filter_profile : str
        Narrow-band filter shape. Currently only 'boxcar' supported. Default: 'boxcar'.
    epsilon_weight : float
        Small constant added to avoid division by zero. Default: 1e-10.
    wandb_run : Any, optional
        wandb run instance for experiment tracking. If provided, metrics will be logged
        to wandb during training. Default: None (no wandb logging).
    wandb_log_frequency : int
        How often to log training metrics to wandb (in epochs). Default: 100.
    wandb_log_spectrum_evolution : bool
        Whether to log spectrum snapshots during training. Default: True.
    wandb_spectrum_evolution_frequency : int
        How often to log spectrum snapshots (in epochs). Default: 500.
    wandb_track_convergence : bool
        Whether to track convergence metrics for stopping decisions. Default: True.

    Attributes
    ----------
    All parameters above are stored as instance attributes.

    Examples
    --------
    >>> config = SEDConfig(epochs=5000, regularization_weight=2.0)
    >>> config.to_yaml_file("my_config.yaml")
    >>> loaded = SEDConfig.from_yaml_file("my_config.yaml")

    >>> # With wandb integration
    >>> import wandb
    >>> wandb.init(project="sed-reconstruction")
    >>> config = SEDConfig(epochs=5000, wandb_run=wandb)
    >>> # Metrics will be logged to wandb during training
    """

    # Global reconstruction parameters
    wavelength_range: Tuple[float, float] = (0.75, 5.0)
    global_resolution: int = 3000
    device: str = "mps"  # Options: "cpu", "cuda", "mps"

    # Optimization parameters
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    epochs: int = 3000

    # Learning rate scheduling parameters
    learning_rate_scheduler_type: str = "cosine_warmup"  # Options: "none", "cosine", "cosine_warmup"
    learning_rate_warmup_epochs: int = 150  # 5% of default 3000 epochs
    learning_rate_min_factor: float = 0.01  # Minimum LR as fraction of peak (1%)

    # Deep Prior Architecture
    dip_noise_std: float = 0.1
    dip_filters: int = 32
    dip_depth: int = 3

    # Noise jittering parameters
    dip_noise_jitter_initial_ratio: Optional[float] = 0.3  # Initial jitter as ratio of dip_noise_std, None means no jitter
    dip_noise_jitter_min_ratio: Optional[float] = None  # Minimum jitter ratio, None means no decay

    # Regularization (CWT)
    regularization_weight: float = 1.0
    cwt_scales: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])

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

    # Numerical stability
    epsilon_weight: float = 1e-10  # Small constant added to avoid division by zero

    # Optional wandb integration parameters
    wandb_run: Any = None  # wandb run instance for experiment tracking (optional)
    wandb_log_frequency: int = 100  # How often to log metrics (in epochs)
    wandb_log_spectrum_evolution: bool = True  # Whether to log spectrum snapshots during training
    wandb_spectrum_evolution_frequency: int = 500  # How often to log spectrum snapshots (in epochs)
    wandb_track_convergence: bool = True  # Whether to track convergence metrics for stopping decisions

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate global reconstruction parameters
        if not (0.75 <= self.wavelength_range[0] < self.wavelength_range[1] <= 5.0):
            raise ValueError(
                f"wavelength_range must be within SPHEREx coverage (0.75-5.0 μm), got {self.wavelength_range}"
            )
        if self.global_resolution <= 0:
            raise ValueError(f"global_resolution must be positive, got {self.global_resolution}")
        if self.global_resolution > 10000:
            raise ValueError(
                f"global_resolution too large ({self.global_resolution}), may cause memory issues. Consider < 10000."
            )

        # Validate device
        valid_devices = ["cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got '{self.device}'")

        # Validate optimization parameters
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")

        # Validate learning rate scheduling parameters
        valid_schedulers = ["none", "cosine", "cosine_warmup", "warmup"]
        if self.learning_rate_scheduler_type not in valid_schedulers:
            raise ValueError(f"learning_rate_scheduler_type must be one of {valid_schedulers}, got '{self.learning_rate_scheduler_type}'")
        if self.learning_rate_warmup_epochs < 0:
            raise ValueError(f"learning_rate_warmup_epochs must be non-negative, got {self.learning_rate_warmup_epochs}")
        if self.learning_rate_warmup_epochs >= self.epochs:
            raise ValueError(f"learning_rate_warmup_epochs ({self.learning_rate_warmup_epochs}) must be less than epochs ({self.epochs})")
        if self.learning_rate_min_factor <= 0 or self.learning_rate_min_factor >= 1:
            raise ValueError(f"learning_rate_min_factor must be between 0 and 1 (exclusive), got {self.learning_rate_min_factor}")

        # Validate Deep Prior Architecture
        if self.dip_noise_std < 0:
            raise ValueError(f"dip_noise_std must be non-negative, got {self.dip_noise_std}")
        if self.dip_filters <= 0:
            raise ValueError(f"dip_filters must be positive, got {self.dip_filters}")
        if self.dip_depth < 1 or self.dip_depth > 10:
            raise ValueError(f"dip_depth must be between 1 and 10, got {self.dip_depth}")

        # Validate noise jittering parameters
        if self.dip_noise_jitter_initial_ratio is not None:
            if self.dip_noise_jitter_initial_ratio < 0:
                raise ValueError(
                    f"dip_noise_jitter_initial_ratio must be non-negative, "
                    f"got {self.dip_noise_jitter_initial_ratio}"
                )

        if self.dip_noise_jitter_min_ratio is not None:
            if self.dip_noise_jitter_min_ratio < 0:
                raise ValueError(
                    f"dip_noise_jitter_min_ratio must be non-negative, "
                    f"got {self.dip_noise_jitter_min_ratio}"
                )
            if self.dip_noise_jitter_initial_ratio is not None:
                if self.dip_noise_jitter_min_ratio > self.dip_noise_jitter_initial_ratio:
                    raise ValueError(
                        f"dip_noise_jitter_min_ratio ({self.dip_noise_jitter_min_ratio}) "
                        f"must be <= dip_noise_jitter_initial_ratio ({self.dip_noise_jitter_initial_ratio})"
                    )

        # Validate regularization
        if self.regularization_weight < 0:
            raise ValueError(f"regularization_weight must be non-negative, got {self.regularization_weight}")
        if not self.cwt_scales:
            raise ValueError("cwt_scales cannot be empty")
        if any(scale <= 0 for scale in self.cwt_scales):
            raise ValueError("All cwt_scales values must be positive")

        # Validate quality control
        if self.sigma_threshold < 0:
            raise ValueError(f"sigma_threshold must be non-negative, got {self.sigma_threshold}")

        # Validate sigma clipping parameters
        if self.enable_sigma_clip:
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

        # Validate filter profile
        valid_filters = ["boxcar"]
        if self.filter_profile not in valid_filters:
            raise ValueError(f"filter_profile must be one of {valid_filters}, got '{self.filter_profile}'")

        # Validate epsilon_weight
        if self.epsilon_weight <= 0:
            raise ValueError(f"epsilon_weight must be positive, got {self.epsilon_weight}")

        # Validate wandb parameters
        if self.wandb_log_frequency <= 0:
            raise ValueError(f"wandb_log_frequency must be positive, got {self.wandb_log_frequency}")
        if self.wandb_spectrum_evolution_frequency <= 0:
            raise ValueError(f"wandb_spectrum_evolution_frequency must be positive, got {self.wandb_spectrum_evolution_frequency}")

    def is_wandb_enabled(self) -> bool:
        """
        Check if wandb logging is enabled.

        Returns
        -------
        bool
            True if wandb_run is not None, False otherwise.
        """
        return self.wandb_run is not None

    def log_to_wandb(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to wandb if wandb is enabled.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Dictionary of metrics to log.
        step : Optional[int]
            Step number for the metrics. If None, wandb will use internal step counter.
        """
        if self.is_wandb_enabled():
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                # Fail silently if wandb logging fails
                import warnings
                warnings.warn(f"Failed to log to wandb: {e}", RuntimeWarning)

    def log_spectrum_to_wandb(self, spectrum_data, wavelength_data, step: int,
                            title: str = "Spectrum") -> None:
        """
        Log spectrum plot to wandb if wandb is enabled.

        Parameters
        ----------
        spectrum_data : array_like
            Spectrum data to plot.
        wavelength_data : array_like
            Wavelength data for the spectrum.
        step : int
            Step number for the plot.
        title : str
            Title for the plot.
        """
        if self.is_wandb_enabled() and self.wandb_log_spectrum_evolution:
            try:
                import matplotlib.pyplot as plt

                # Create a temporary figure
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(wavelength_data, spectrum_data, 'b-', linewidth=1.5)
                ax.set_xlabel('Wavelength (μm)')
                ax.set_ylabel('Flux')
                ax.set_title(f'{title} - Step {step}')
                ax.grid(True, alpha=0.3)

                # Import wandb Image
                from wandb import Image
                # Log to wandb
                self.wandb_run.log({title.lower(): Image(fig)}, step=step)

                # Close the figure to free memory
                plt.close(fig)
            except Exception as e:
                # Fail silently if wandb logging fails
                import warnings
                warnings.warn(f"Failed to log spectrum to wandb: {e}", RuntimeWarning)

    def to_wandb_config(self) -> Dict[str, Any]:
        """
        Convert configuration to wandb-compatible dictionary (excluding wandb_run).

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary suitable for wandb.init(config=...).
        """
        # Manually build config dict to avoid serialization issues with wandb module
        config_dict = {
            'wavelength_range': self.wavelength_range,
            'global_resolution': self.global_resolution,
            'device': self.device,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'learning_rate_scheduler_type': self.learning_rate_scheduler_type,
            'learning_rate_warmup_epochs': self.learning_rate_warmup_epochs,
            'learning_rate_min_factor': self.learning_rate_min_factor,
            'dip_noise_std': self.dip_noise_std,
            'dip_filters': self.dip_filters,
            'dip_depth': self.dip_depth,
            'dip_noise_jitter_initial_ratio': self.dip_noise_jitter_initial_ratio,
            'dip_noise_jitter_min_ratio': self.dip_noise_jitter_min_ratio,
            'regularization_weight': self.regularization_weight,
            'cwt_scales': self.cwt_scales,
            'sigma_threshold': self.sigma_threshold,
            'bad_flags': self.bad_flags,
            'enable_sigma_clip': self.enable_sigma_clip,
            'sigma_clip_sigma': self.sigma_clip_sigma,
            'sigma_clip_window': self.sigma_clip_window,
            'sigma_clip_max_iterations': self.sigma_clip_max_iterations,
            'filter_profile': self.filter_profile,
            'epsilon_weight': self.epsilon_weight,
            'wandb_log_frequency': self.wandb_log_frequency,
            'wandb_log_spectrum_evolution': self.wandb_log_spectrum_evolution,
            'wandb_spectrum_evolution_frequency': self.wandb_spectrum_evolution_frequency,
            'wandb_track_convergence': self.wandb_track_convergence,
        }

        # Handle wandb_run separately
        config_dict['wandb_run_active'] = self.wandb_run is not None

        return config_dict

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
        >>> config = SEDConfig(epochs=1000, regularization_weight=2.0)
        >>> tuned = config.copy_with_overrides(epochs=5000, learning_rate=0.01)
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
