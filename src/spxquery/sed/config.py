"""
Configuration dataclasses for SED reconstruction.

This module provides configuration classes for spectral reconstruction
from SPHEREx narrow-band photometry using PyTorch-based Deep Image Prior.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings

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
    dip_kernel_size : int
        Convolution kernel size for U-Net (must be odd). Default: 5.
        Larger kernels increase receptive field but also computational cost.
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
    regularization_method : str
        Regularization loss function for CWT sparsity. Options: 'absolute', 'log', 'cauchy'.
        Default: 'cauchy'. 'absolute' uses simple L1 norm without adaptive parameters.
        'log' and 'cauchy' use adaptive epsilon/gamma with warmup logic.
    reg_sensitivity_factor : float
        Sensitivity factor (kappa) for adaptive methods. For 'log': epsilon = k * sigma_noise.
        For 'cauchy': gamma = k * sigma_noise. Default: 3.0. Higher values preserve more signal.
        Only used for 'log' and 'cauchy' methods.
    reg_warmup_floor : float
        Minimum floor for epsilon/gamma during warmup phase (adaptive methods only).
        Provides loose constraint to prevent large gradients early in training. Default: 0.1.
    reg_normal_floor : float
        Minimum floor for epsilon/gamma after warmup (adaptive methods only).
        Provides numerical stability while allowing sensitivity to noise. Default: 1e-4.
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
    use_ema : bool
        Enable/disable Exponential Moving Average for spectrum quality control. Default: True.
        EMA provides smoother, more stable spectrum outputs by maintaining a running average
        of the generated spectra during training.
    ema_decay : float
        Decay rate for Exponential Moving Average. Default: 0.99 (99%).
        Higher values (0.99+) trust history more for smoother results.
        Lower values (0.95-0.98) respond faster to recent changes.
    ema_start_epoch : Optional[int]
        Epoch to start EMA tracking. Default: None (uses learning_rate_warmup_epochs).
        If provided, EMA tracking begins after this many epochs.
        If None, automatically uses the same value as learning_rate_warmup_epochs.

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
    dip_kernel_size: int = 5  # Convolution kernel size for U-Net (must be odd)

    # Noise jittering parameters
    dip_noise_jitter_initial_ratio: Optional[float] = (
        0.3  # Initial jitter as ratio of dip_noise_std, None means no jitter
    )
    dip_noise_jitter_min_ratio: Optional[float] = None  # Minimum jitter ratio, None means no decay

    # Regularization (CWT)
    regularization_weight: float = 1.0
    cwt_scales: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])

    # Regularization method and adaptive parameters
    regularization_method: str = "cauchy"  # Options: "absolute", "log", "cauchy"
    reg_sensitivity_factor: float = 3.0  # k where epsilon/gamma = k * sigma_noise (log, cauchy only)
    reg_warmup_floor: float = 0.1  # Loose constraint during warmup (log, cauchy only)
    reg_normal_floor: float = 1e-4  # Tight constraint after warmup (log, cauchy only)

    # Quality control
    sigma_threshold: float = 3.0
    bad_flags: List[int] = field(default_factory=lambda: [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19])

    # Sigma clipping (outlier removal using rolling MAD-based robust statistics)
    enable_sigma_clip: bool = True
    sigma_clip_sigma: float = 3.0
    sigma_clip_window: int = 21
    sigma_clip_max_iterations: int = 10

    # Filter profile
    filter_profile: str = "boxcar"

    # Numerical stability
    epsilon_weight: float = 1e-10  # Small constant added to avoid division by zero

    # Optional wandb integration parameters
    wandb_run: Any = None  # wandb run instance for experiment tracking (optional)
    wandb_log_frequency: int = 100  # How often to log metrics and save data (in epochs)
    wandb_save_raw_data: bool = True  # Save raw spectrum data instead of PNG images
    wandb_save_model_artifacts: bool = True  # Save model state and input noise as artifacts
    wandb_track_convergence: bool = True  # Whether to track convergence metrics for stopping decisions

    # EMA (Exponential Moving Average) parameters for spectrum quality control
    use_ema: bool = True  # Enable/disable Exponential Moving Average (default: True)
    ema_decay: float = 0.99  # EMA decay rate (default: 99%, trusting history more)
    ema_start_epoch: Optional[int] = None  # EMA startup epoch (None = use learning_rate_warmup_epochs)

    # Ensembling parameters for improved reconstruction robustness
    ensemble_size: int = 1  # Number of ensemble members (default: 1 for single reconstruction)
    ensemble_random_seed: Optional[int] = None  # Base seed for reproducible ensemble generation
    ensemble_strategy: str = "independent"  # Ensembling approach (currently only "independent" supported)
    ensemble_save_members: bool = True  # Whether to save individual ensemble member results
    ensemble_n_workers: Optional[int] = (
        None  # Number of parallel workers for ensemble processing (None = sequential, 1 = sequential, >1 = parallel)
    )
    ensemble_perturb_observations: bool = (
        False  # Perturb observations with Gaussian noise during ensemble processing (default: False)
    )

    # Ensemble robustness controls (optional)
    # These are designed to mitigate rare "hang" cases in parallel workers.
    # Defaults preserve existing behavior (no timeout, no retries).
    ensemble_member_timeout_seconds: Optional[float] = 300  # e.g. 300 for 5 minutes
    ensemble_max_retries: int = 3  # retries per member on timeout/crash (0 = disabled)
    ensemble_retry_backoff_seconds: float = 0.0  # sleep before retry (seconds)

    # Early stopping parameters (dual-criteria: chi2 + normality test)
    enable_early_stopping: bool = False  # Enable/disable early stopping
    early_stop_check_steps: int = 50  # Check early stopping criteria every N epochs (after warmup)
    early_stop_cooldown_epoch: int = 300  # Number of cooldown epochs to run at the end (jump to last N epochs)
    early_stop_target_chi2: float = 1.05  # Target chi2 threshold for perfect early stop
    early_stop_lowest_chi2: float = 0.85  # Lowest chi2 threshold for force early stop (regardless of p-value)

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate global reconstruction parameters
        if not (0.75 <= self.wavelength_range[0] < self.wavelength_range[1] <= 5.0):
            raise ValueError(
                f"wavelength_range must be within SPHEREx coverage (0.75-5.0 μm), got {self.wavelength_range}"
            )
        if self.global_resolution <= 0:
            raise ValueError(f"global_resolution must be positive, got {self.global_resolution}")
        if self.global_resolution > 50000:
            warnings.warn(
                f"global_resolution too large ({self.global_resolution}), may cause memory issues. Consider < 50000.",
                UserWarning,
                stacklevel=2,
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
            raise ValueError(
                f"learning_rate_scheduler_type must be one of {valid_schedulers}, got '{self.learning_rate_scheduler_type}'"
            )
        if self.learning_rate_warmup_epochs < 0:
            raise ValueError(
                f"learning_rate_warmup_epochs must be non-negative, got {self.learning_rate_warmup_epochs}"
            )
        if self.learning_rate_warmup_epochs >= self.epochs:
            raise ValueError(
                f"learning_rate_warmup_epochs ({self.learning_rate_warmup_epochs}) must be less than epochs ({self.epochs})"
            )
        if self.learning_rate_min_factor <= 0 or self.learning_rate_min_factor >= 1:
            raise ValueError(
                f"learning_rate_min_factor must be between 0 and 1 (exclusive), got {self.learning_rate_min_factor}"
            )

        # Validate Deep Prior Architecture
        if self.dip_noise_std < 0:
            raise ValueError(f"dip_noise_std must be non-negative, got {self.dip_noise_std}")
        if self.dip_filters <= 0:
            raise ValueError(f"dip_filters must be positive, got {self.dip_filters}")
        if self.dip_depth < 1 or self.dip_depth > 10:
            raise ValueError(f"dip_depth must be between 1 and 10, got {self.dip_depth}")
        if self.dip_kernel_size < 1 or self.dip_kernel_size > 11:
            raise ValueError(f"dip_kernel_size must be between 1 and 11, got {self.dip_kernel_size}")
        if self.dip_kernel_size % 2 == 0:
            raise ValueError(f"dip_kernel_size must be odd, got {self.dip_kernel_size}")

        # Validate noise jittering parameters
        if self.dip_noise_jitter_initial_ratio is not None:
            if self.dip_noise_jitter_initial_ratio < 0:
                raise ValueError(
                    f"dip_noise_jitter_initial_ratio must be non-negative, got {self.dip_noise_jitter_initial_ratio}"
                )

        if self.dip_noise_jitter_min_ratio is not None:
            if self.dip_noise_jitter_min_ratio < 0:
                raise ValueError(
                    f"dip_noise_jitter_min_ratio must be non-negative, got {self.dip_noise_jitter_min_ratio}"
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

        # Validate regularization method
        valid_methods = ["absolute", "log", "cauchy"]
        if self.regularization_method not in valid_methods:
            raise ValueError(
                f"regularization_method must be one of {valid_methods}, got '{self.regularization_method}'"
            )

        # Validate adaptive parameters for log and cauchy methods
        if self.regularization_method in ["log", "cauchy"]:
            if self.reg_sensitivity_factor <= 0:
                raise ValueError(f"reg_sensitivity_factor must be positive, got {self.reg_sensitivity_factor}")
            if self.reg_warmup_floor < 0:
                raise ValueError(f"reg_warmup_floor must be non-negative, got {self.reg_warmup_floor}")
            if self.reg_normal_floor < 0:
                raise ValueError(f"reg_normal_floor must be non-negative, got {self.reg_normal_floor}")

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
        valid_filters = ["boxcar", "gaussian"]  # Will expand as more profiles are implemented
        if self.filter_profile not in valid_filters:
            raise ValueError(f"filter_profile must be one of {valid_filters}, got '{self.filter_profile}'")

        # Validate epsilon_weight
        if self.epsilon_weight <= 0:
            raise ValueError(f"epsilon_weight must be positive, got {self.epsilon_weight}")

        # Validate wandb parameters
        if self.wandb_log_frequency <= 0:
            raise ValueError(f"wandb_log_frequency must be positive, got {self.wandb_log_frequency}")

        # Validate EMA parameters
        if self.use_ema:
            if not (0.9 <= self.ema_decay <= 0.999):
                raise ValueError(f"ema_decay must be between 0.9 and 0.999 for stable EMA, got {self.ema_decay}")
            if self.ema_start_epoch is not None:
                if self.ema_start_epoch < 0:
                    raise ValueError(f"ema_start_epoch must be non-negative, got {self.ema_start_epoch}")
                if self.ema_start_epoch >= self.epochs:
                    raise ValueError(
                        f"ema_start_epoch ({self.ema_start_epoch}) must be less than epochs ({self.epochs})"
                    )

        # Validate ensembling parameters
        if self.ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {self.ensemble_size}")
        if self.ensemble_size > 20:
            warnings.warn(
                f"ensemble_size ({self.ensemble_size}) is too large, may cause memory issues. Consider < 20.",
                UserWarning,
                stacklevel=2,
            )

        # Validate ensemble strategy
        valid_strategies = ["independent"]  # Will expand as more strategies are implemented
        if self.ensemble_strategy not in valid_strategies:
            raise ValueError(f"ensemble_strategy must be one of {valid_strategies}, got '{self.ensemble_strategy}'")

        # Validate ensemble n_workers
        if self.ensemble_n_workers is not None:
            if self.ensemble_n_workers < 1:
                raise ValueError(f"ensemble_n_workers must be >= 1 or None, got {self.ensemble_n_workers}")
            # Warn if n_workers exceeds ensemble_size (extra workers won't be used)
            if self.ensemble_n_workers > self.ensemble_size:
                import warnings

                warnings.warn(
                    f"ensemble_n_workers ({self.ensemble_n_workers}) > ensemble_size ({self.ensemble_size}). "
                    f"Only {self.ensemble_size} workers will be used effectively.",
                    UserWarning,
                    stacklevel=2,
                )

        # Validate ensemble robustness controls
        if self.ensemble_member_timeout_seconds is not None and self.ensemble_member_timeout_seconds <= 0:
            raise ValueError(
                "ensemble_member_timeout_seconds must be > 0 or None, "
                f"got {self.ensemble_member_timeout_seconds}"
            )
        if self.ensemble_max_retries < 0:
            raise ValueError(f"ensemble_max_retries must be >= 0, got {self.ensemble_max_retries}")
        if self.ensemble_retry_backoff_seconds < 0:
            raise ValueError(
                f"ensemble_retry_backoff_seconds must be >= 0, got {self.ensemble_retry_backoff_seconds}"
            )

        # Warn about wandb conflicts for ensemble runs
        if self.ensemble_size > 1 and self.is_wandb_enabled():
            import warnings

            warnings.warn(
                "Ensembling with wandb logging: Only the first ensemble member will be logged to wandb to prevent conflicts.",
                UserWarning,
                stacklevel=2,
            )

        # Validate early stopping parameters
        if self.enable_early_stopping:
            if self.early_stop_check_steps <= 0:
                raise ValueError(f"early_stop_check_steps must be positive, got {self.early_stop_check_steps}")
            if self.early_stop_cooldown_epoch >= self.epochs:
                raise ValueError(
                    f"early_stop_cooldown_epoch ({self.early_stop_cooldown_epoch}) must be less than epochs ({self.epochs})"
                )
            # Ensure there's enough room for early stopping to actually trigger
            # Earliest possible trigger = warmup_end + check_steps
            # Cooldown starts at epochs - cooldown_epoch
            warmup_end = self.learning_rate_warmup_epochs
            earliest_check = warmup_end + self.early_stop_check_steps
            cooldown_start = self.epochs - self.early_stop_cooldown_epoch
            if earliest_check >= cooldown_start:
                raise ValueError(
                    f"Insufficient epochs for early stopping: warmup ends at {warmup_end}, "
                    f"cooldown starts at {cooldown_start}, but first check is at {earliest_check}. "
                    f"Reduce learning_rate_warmup_epochs or early_stop_cooldown_epoch."
                )
            if self.early_stop_lowest_chi2 >= self.early_stop_target_chi2:
                raise ValueError(
                    f"early_stop_lowest_chi2 ({self.early_stop_lowest_chi2}) must be less than "
                    f"early_stop_target_chi2 ({self.early_stop_target_chi2})"
                )

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

    def log_spectrum_data_to_wandb(
        self, spectrum_data, wavelength_data, step: int, data_type: str = "spectrum"
    ) -> None:
        """
        Log raw spectrum data to wandb if wandb is enabled and data saving is on.

        Parameters
        ----------
        spectrum_data : array_like
            Spectrum data to save.
        wavelength_data : array_like
            Wavelength data for the spectrum.
        step : int
            Step number for the data.
        data_type : str
            Type identifier for the spectrum data.
        """
        if self.is_wandb_enabled() and self.wandb_save_raw_data:
            try:
                import json

                import wandb

                # Convert to lists for JSON serialization
                data_dict = {
                    "spectrum": spectrum_data.tolist() if hasattr(spectrum_data, "tolist") else list(spectrum_data),
                    "wavelength": wavelength_data.tolist()
                    if hasattr(wavelength_data, "tolist")
                    else list(wavelength_data),
                    "epoch": step,
                    "data_type": data_type,
                }

                # Create wandb Table for summary info
                table = wandb.Table(columns=["epoch", "data_type", "n_pixels", "flux_min", "flux_max"])
                table.add_data(
                    step, data_type, len(spectrum_data), float(min(spectrum_data)), float(max(spectrum_data))
                )

                # Log both the table and create an artifact for the full data
                self.wandb_run.log(
                    {
                        f"{data_type}_summary": table,
                        f"{data_type}_data_step": step,
                    },
                    step=step,
                )

                # Create and log artifact for the full data
                artifact = wandb.Artifact(f"{data_type}_epoch_{step}", type="spectrum_data")
                with artifact.new_file(f"{data_type}_epoch_{step}.json", mode="w") as f:
                    json.dump(data_dict, f)
                self.wandb_run.log_artifact(artifact, aliases=[f"epoch_{step}"])

            except Exception as e:
                # Fail silently if wandb logging fails
                import warnings

                warnings.warn(f"Failed to log spectrum data to wandb: {e}", RuntimeWarning)

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
            "wavelength_range": self.wavelength_range,
            "global_resolution": self.global_resolution,
            "device": self.device,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "learning_rate_scheduler_type": self.learning_rate_scheduler_type,
            "learning_rate_warmup_epochs": self.learning_rate_warmup_epochs,
            "learning_rate_min_factor": self.learning_rate_min_factor,
            "dip_noise_std": self.dip_noise_std,
            "dip_filters": self.dip_filters,
            "dip_depth": self.dip_depth,
            "dip_kernel_size": self.dip_kernel_size,
            "dip_noise_jitter_initial_ratio": self.dip_noise_jitter_initial_ratio,
            "dip_noise_jitter_min_ratio": self.dip_noise_jitter_min_ratio,
            "regularization_weight": self.regularization_weight,
            "cwt_scales": self.cwt_scales,
            "regularization_method": self.regularization_method,
            "reg_sensitivity_factor": self.reg_sensitivity_factor,
            "reg_warmup_floor": self.reg_warmup_floor,
            "reg_normal_floor": self.reg_normal_floor,
            "sigma_threshold": self.sigma_threshold,
            "bad_flags": self.bad_flags,
            "enable_sigma_clip": self.enable_sigma_clip,
            "sigma_clip_sigma": self.sigma_clip_sigma,
            "sigma_clip_window": self.sigma_clip_window,
            "sigma_clip_max_iterations": self.sigma_clip_max_iterations,
            "filter_profile": self.filter_profile,
            "epsilon_weight": self.epsilon_weight,
            "wandb_log_frequency": self.wandb_log_frequency,
            "wandb_save_raw_data": self.wandb_save_raw_data,
            "wandb_save_model_artifacts": self.wandb_save_model_artifacts,
            "wandb_track_convergence": self.wandb_track_convergence,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "ema_start_epoch": self.ema_start_epoch,
            "ensemble_size": self.ensemble_size,
            "ensemble_strategy": self.ensemble_strategy,
            "ensemble_save_members": self.ensemble_save_members,
            "ensemble_n_workers": self.ensemble_n_workers,
            "ensemble_perturb_observations": self.ensemble_perturb_observations,
            "enable_early_stopping": self.enable_early_stopping,
            "early_stop_check_steps": self.early_stop_check_steps,
            "early_stop_cooldown_epoch": self.early_stop_cooldown_epoch,
            "early_stop_target_chi2": self.early_stop_target_chi2,
            "early_stop_lowest_chi2": self.early_stop_lowest_chi2,
        }

        # Handle wandb_run separately
        config_dict["wandb_run_active"] = self.wandb_run is not None

        return config_dict

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of configuration.
        """
        import typing
        from dataclasses import asdict, fields

        data = asdict(self)

        # Convert tuples to lists for YAML serialization
        # (yaml.safe_load can't handle !!python/tuple tags)
        for field in fields(self.__class__):
            if field.name in data:
                origin = typing.get_origin(field.type)
                if origin is tuple:
                    if isinstance(data[field.name], tuple):
                        data[field.name] = list(data[field.name])

        return data

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
        import typing
        from dataclasses import fields

        # Filter to only valid fields to handle extra keys gracefully
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Convert lists back to tuples for tuple-type fields (YAML limitation)
        for field in fields(cls):
            if field.name in filtered_data:
                # Check if the field type is a Tuple
                origin = typing.get_origin(field.type)
                if origin is tuple:
                    # Convert list to tuple if needed
                    value = filtered_data[field.name]
                    if isinstance(value, list):
                        filtered_data[field.name] = tuple(value)

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
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

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
