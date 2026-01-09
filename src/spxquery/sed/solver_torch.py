"""
PyTorch solver for Deep Image Prior reconstruction.
"""

import logging
import math
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import SEDConfig
from .data_structures import GlobalSpectralData
from .regularization import GaussianCWT

logger = logging.getLogger(__name__)


def _coo_spmv(
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    values: torch.Tensor,
    x: torch.Tensor,
    n_rows: int,
) -> torch.Tensor:
    """Sparse matrix-vector multiply for COO buffers.

    Computes y = H @ x where H is represented by COO triplets (row_idx, col_idx, values).
    This implementation is differentiable w.r.t. x and works on backends that lack
    torch sparse ops (e.g., MPS).
    """
    if row_idx.dtype != torch.long:
        row_idx = row_idx.long()
    if col_idx.dtype != torch.long:
        col_idx = col_idx.long()

    y = torch.zeros(n_rows, device=x.device, dtype=x.dtype)
    y.index_add_(0, row_idx, values.to(dtype=x.dtype) * x[col_idx])
    return y


def set_random_seed(seed: int):
    """
    Set random seeds for reproducible ensemble generation.

    Parameters
    ----------
    seed : int
        Random seed to use for all random number generators.
    """
    logger.debug(f"Setting random seed to {seed}")

    # Set Python random seed
    import random

    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)

    # Set PyTorch CUDA random seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cauchy_loss(x: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Cauchy (Lorentzian) loss function: log(1 + (x/gamma)^2).

    Why this fixes the 'Line Broadening' and 'High-Freq Leakage' problem:
    1. Small x (x << gamma): Approx (x/gamma)^2 -> L2 Norm.
       - Strong smoothing for background noise.
    2. Large x (x >> gamma): Approx 2*log(x/gamma) -> Logarithmic growth.
       - Gradient dL/dx approx 2/x.
       - As x increases (strong line signal), the penalty gradient goes to 0.

    This effectively tells the solver: "If the CWT coefficient is large enough (meaning it's likely
    part of a real line, even in high-freq scales), stop punishing it."
    """
    return torch.sum(torch.log(1 + (x / gamma) ** 2))


def log_loss(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Logarithmic loss function: sum(log(1 + |x|/epsilon)).

    Similar to L1 but with sublinear growth for large values, providing
    a balance between L1 and L2 regularization. The epsilon parameter
    controls the transition point from linear (for small |x|) to logarithmic
    behavior (for large |x|).

    Behavior:
    - Small |x| (|x| << epsilon): Approx |x|/epsilon -> L1-like
    - Large |x| (|x| >> epsilon): Approx log(|x|/epsilon) -> Logarithmic growth
    """
    return torch.sum(torch.log(1 + torch.abs(x) / epsilon))


def create_learning_rate_scheduler(optimizer, config: SEDConfig):
    """
    Create a learning rate scheduler based on configuration.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        PyTorch optimizer to wrap with scheduler.
    config : SEDConfig
        Configuration containing scheduler parameters.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler or None
        Learning rate scheduler instance, or None if no scheduler is specified.
    """
    if config.learning_rate_scheduler_type == "none":
        return None

    class CosineWarmupScheduler:
        """
        Custom learning rate scheduler with linear warmup and cosine decay.

        This scheduler implements:
        - Linear warmup from 0 to peak learning rate
        - Cosine annealing from peak to minimum learning rate
        """

        def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_factor):
            self.optimizer = optimizer
            self.warmup_epochs = warmup_epochs
            self.total_epochs = total_epochs
            self.min_lr_factor = min_lr_factor
            self.base_lrs = [group["lr"] for group in optimizer.param_groups]
            self.current_epoch = 0

        def step(self):
            """Update learning rates based on current epoch."""
            if self.current_epoch < self.warmup_epochs:
                # Linear warmup phase
                lr_factor = self.current_epoch / self.warmup_epochs
            else:
                # Cosine decay phase
                decay_progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                decay_progress = min(decay_progress, 1.0)  # Clamp to [0, 1]
                lr_factor = self.min_lr_factor + 0.5 * (1 - self.min_lr_factor) * (
                    1 + math.cos(math.pi * decay_progress)
                )

            # Update learning rates for all parameter groups
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.base_lrs[i] * lr_factor

            self.current_epoch += 1

        def get_last_lr(self):
            """Get current learning rates."""
            return [group["lr"] for group in self.optimizer.param_groups]

        def state_dict(self):
            """Get scheduler state dictionary for saving/loading."""
            return {
                "warmup_epochs": self.warmup_epochs,
                "total_epochs": self.total_epochs,
                "min_lr_factor": self.min_lr_factor,
                "base_lrs": self.base_lrs,
                "current_epoch": self.current_epoch,
            }

        def load_state_dict(self, state_dict):
            """Load scheduler state from dictionary."""
            self.warmup_epochs = state_dict["warmup_epochs"]
            self.total_epochs = state_dict["total_epochs"]
            self.min_lr_factor = state_dict["min_lr_factor"]
            self.base_lrs = state_dict["base_lrs"]
            self.current_epoch = state_dict["current_epoch"]

    class WarmupOnlyScheduler:
        """
        Custom learning rate scheduler with linear warmup and constant learning rate.

        This scheduler implements:
        - Linear warmup from 0 to peak learning rate
        - Constant learning rate after warmup phase (no decay)
        """

        def __init__(self, optimizer, warmup_epochs):
            self.optimizer = optimizer
            self.warmup_epochs = warmup_epochs
            self.base_lrs = [group["lr"] for group in optimizer.param_groups]
            self.current_epoch = 0

        def step(self):
            """Update learning rates based on current epoch."""
            if self.current_epoch < self.warmup_epochs:
                # Linear warmup phase
                lr_factor = self.current_epoch / self.warmup_epochs
            else:
                # Constant learning rate phase
                lr_factor = 1.0

            # Update learning rates for all parameter groups
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.base_lrs[i] * lr_factor

            self.current_epoch += 1

        def get_last_lr(self):
            """Get current learning rates."""
            return [group["lr"] for group in self.optimizer.param_groups]

        def state_dict(self):
            """Get scheduler state dictionary for saving/loading."""
            return {
                "warmup_epochs": self.warmup_epochs,
                "base_lrs": self.base_lrs,
                "current_epoch": self.current_epoch,
            }

        def load_state_dict(self, state_dict):
            """Load scheduler state from dictionary."""
            self.warmup_epochs = state_dict["warmup_epochs"]
            self.base_lrs = state_dict["base_lrs"]
            self.current_epoch = state_dict["current_epoch"]

    if config.learning_rate_scheduler_type == "cosine":
        # Cosine annealing without warmup
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.learning_rate * config.learning_rate_min_factor
        )
    elif config.learning_rate_scheduler_type == "cosine_warmup":
        # Linear warmup + cosine decay
        return CosineWarmupScheduler(
            optimizer,
            warmup_epochs=config.learning_rate_warmup_epochs,
            total_epochs=config.epochs,
            min_lr_factor=config.learning_rate_min_factor,
        )
    elif config.learning_rate_scheduler_type == "warmup":
        # Linear warmup + constant learning rate (no decay)
        return WarmupOnlyScheduler(
            optimizer,
            warmup_epochs=config.learning_rate_warmup_epochs,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.learning_rate_scheduler_type}")


class SpectralUNet(nn.Module):
    """
    1D U-Net architecture for spectral generation.
    Acts as a Deep Image Prior (Deep Spectral Prior).
    """

    def __init__(
        self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 32, depth: int = 3, kernel_size: int = 5
    ):
        super().__init__()

        self.depth = depth

        padding = kernel_size // 2

        # Encoders
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        curr_filters = base_filters
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, curr_filters, kernel_size=kernel_size, padding=padding, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv1d(curr_filters, curr_filters, kernel_size=kernel_size, padding=padding, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        for i in range(depth):
            # Downsample
            self.downsamples.append(nn.MaxPool1d(2))

            # Double filters
            next_filters = curr_filters * 2
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(
                        curr_filters, next_filters, kernel_size=kernel_size, padding=padding, padding_mode="reflect"
                    ),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(
                        next_filters, next_filters, kernel_size=kernel_size, padding=padding, padding_mode="reflect"
                    ),
                    nn.LeakyReLU(0.2),
                )
            )
            curr_filters = next_filters

        # Decoders
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in range(depth):
            curr_filters = curr_filters // 2

            # Upsample
            self.upsamples.append(nn.Upsample(scale_factor=2, mode="linear", align_corners=True))
            # self.upsamples.append(nn.ConvTranspose1d(prev_filters, curr_filters, kernel_size=4, stride=2, padding=1))
            # self.upsamples.append(nn.Upsample(scale_factor=2, mode="nearest"))

            # Convolution (input channels = prev_filters from upsample + prev_filters from skip = 2*prev_filters?? No)
            # Skip connection brings 'curr_filters' channels. Upsample brings 'prev_filters' (which is 2*curr).
            # So concatenation has 3 * curr_filters?
            # Standard UNet: Encoder i outputs F filters. Decoder i takes F filters from
            # upsample and F from skip. Total 2F.

            self.decoders.append(
                nn.Sequential(
                    nn.Conv1d(
                        curr_filters * 3, curr_filters, kernel_size=kernel_size, padding=padding, padding_mode="reflect"
                    ),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(
                        curr_filters, curr_filters, kernel_size=kernel_size, padding=padding, padding_mode="reflect"
                    ),
                    nn.LeakyReLU(0.2),
                )
            )

        # Output layer
        self.final_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Initial
        x1 = self.initial_conv(x)

        # Encoder path
        skips = [x1]
        out = x1
        for i in range(self.depth):
            out = self.downsamples[i](out)
            out = self.encoders[i](out)
            skips.append(out)

        # Decoder path
        # skips has [x1, e1, e2, ... e_depth]
        # e_depth is the bottom. We start decoding from it.

        for i in range(self.depth):
            # Upsample
            out = self.upsamples[i](out)

            # Get skip connection (from second to last)
            skip = skips[-(i + 2)]

            # Handle size mismatch due to odd dimensions during downsampling
            if out.shape[-1] != skip.shape[-1]:
                out = F.interpolate(out, size=skip.shape[-1], mode="linear", align_corners=True)

            # Concatenate
            out = torch.cat([out, skip], dim=1)

            # Convolve
            out = self.decoders[i](out)

        # Final output
        out = self.final_conv(out)
        # return F.relu(out)  # Enforce non-negativity
        return out


class SpectralModel(nn.Module):
    """
    Wrapper for SpectralUNet that holds static noise and optional jitter.

    The input to the network consists of:
    - Static noise: Fixed component with dip_noise_std
    - Optional jitter: Additional component that can decay linearly
    """

    def __init__(self, n_pixels: int, config: SEDConfig):
        super().__init__()
        self.net = SpectralUNet(
            in_channels=1,
            out_channels=1,
            base_filters=config.dip_filters,
            depth=config.dip_depth,
            kernel_size=config.dip_kernel_size,
        )
        self.config = config
        self.n_pixels = n_pixels

        # Static noise component (fixed throughout training)
        # Shape: (1, 1, N)
        self.register_buffer("z_static", torch.randn(1, 1, n_pixels) * config.dip_noise_std)

        # Dynamic jitter component (will be updated during training)
        self.register_buffer("z_jitter", torch.zeros(1, 1, n_pixels))

        # Current jitter standard deviation
        self.register_buffer("current_jitter_std", torch.tensor(0.0))

        # Initialize jitter
        self.update_jitter(0)

    def update_jitter(self, epoch: int):
        """
        Update the jitter component based on current epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch (0-indexed).
        """
        if self.config.dip_noise_jitter_initial_ratio is None:
            # No jitter enabled
            self.current_jitter_std = torch.tensor(0.0)
            self.z_jitter.zero_()
            return

        # Calculate initial jitter standard deviation
        initial_jitter_std = self.config.dip_noise_std * self.config.dip_noise_jitter_initial_ratio

        if self.config.dip_noise_jitter_min_ratio is None:
            # No decay, use constant jitter
            self.current_jitter_std = torch.tensor(initial_jitter_std)
        else:
            # Linear decay from initial ratio to minimum ratio
            if epoch < self.config.learning_rate_warmup_epochs:
                # During warmup, use full initial jitter
                current_jitter_std = initial_jitter_std
            else:
                # Linear decay phase
                decay_progress = (epoch - self.config.learning_rate_warmup_epochs) / (
                    self.config.epochs - self.config.learning_rate_warmup_epochs
                )
                decay_progress = min(decay_progress, 1.0)  # Clamp to [0, 1]

                # Linear interpolation from initial to minimum jitter
                current_ratio = self.config.dip_noise_jitter_min_ratio + (
                    self.config.dip_noise_jitter_initial_ratio - self.config.dip_noise_jitter_min_ratio
                ) * (1 - decay_progress)
                current_jitter_std = self.config.dip_noise_std * current_ratio

            self.current_jitter_std = torch.tensor(current_jitter_std)

        # Generate new jitter with current standard deviation
        if self.current_jitter_std > 0:
            # Generate jitter on the same device as static noise
            device = self.z_static.device
            self.z_jitter = torch.randn(1, 1, self.n_pixels, device=device) * self.current_jitter_std
        else:
            self.z_jitter.zero_()

    @property
    def z(self):
        """
        Combined input noise (static + jitter).

        Returns
        -------
        torch.Tensor
            Combined noise tensor with shape (1, 1, N).
        """
        return self.z_static + self.z_jitter

    def forward(self):
        # Output shape: (N,)
        return self.net(self.z).squeeze()

    def forward_fixed_noise(self):
        """
        Generate spectrum using only fixed noise (no jitter).

        This method is useful for consistent evaluation and logging,
        as it produces deterministic output independent of the current jitter state.

        Returns
        -------
        torch.Tensor
            Spectrum generated with fixed noise only, shape (N,).
        """
        return self.net(self.z_static).squeeze()


class EMATracker:
    """
    Exponential Moving Average (EMA) tracker for spectrum smoothing.

    This class maintains an EMA of generated spectra during training to provide
    smoother, more stable outputs. The EMA reduces noise and variance in the
    final spectrum while preserving the essential features.

    The EMA formula is:
    new_average = decay * old_average + (1 - decay) * new_value
    """

    def __init__(self, decay: float = 0.99):
        """
        Initialize EMA tracker.

        Parameters
        ----------
        decay : float
            Decay rate for EMA. 0.99 means high trust in history (smoother results),
            while 0.95 would respond faster to recent changes (less smoothing).
            Must be between 0.9 and 0.999 for stable EMA behavior.
        """
        if not (0.9 <= decay <= 0.999):
            raise ValueError(f"EMA decay must be between 0.9 and 0.999 for stable behavior, got {decay}")

        self.decay = decay
        self.shadow = None  # EMA-averaged spectrum
        self.is_initialized = False

    def update(self, current_spectrum: torch.Tensor):
        """
        Update EMA with current spectrum.

        Parameters
        ----------
        current_spectrum : torch.Tensor
            Current spectrum tensor from model output, shape (N,).
        """
        # Ensure input is detached from computation graph
        current_spectrum = current_spectrum.detach()

        if not self.is_initialized:
            # First update: EMA equals current spectrum
            self.shadow = current_spectrum.clone()
            self.is_initialized = True
        else:
            # EMA update: new_average = decay * old_average + (1 - decay) * new_value
            self.shadow = self.decay * self.shadow + (1 - self.decay) * current_spectrum

    def get_ema_spectrum(self):
        """
        Get EMA-smoothed spectrum.

        Returns
        -------
        torch.Tensor or None
            EMA-averaged spectrum, shape (N,), or None if not yet initialized.
        """
        return self.shadow.clone() if self.is_initialized else None

    def get_l1_change(self, current_spectrum: torch.Tensor) -> float:
        """
        Calculate L1 distance between current spectrum and EMA spectrum.

        Parameters
        ----------
        current_spectrum : torch.Tensor
            Current spectrum tensor, shape (N,).

        Returns
        -------
        float
            L1 distance (sum of absolute differences).
        """
        if not self.is_initialized:
            return 0.0

        with torch.no_grad():
            l1_change = torch.sum(torch.abs(current_spectrum - self.shadow)).item()
        return l1_change

    def get_l2_change(self, current_spectrum: torch.Tensor) -> float:
        """
        Calculate L2 distance between current spectrum and EMA spectrum.

        Parameters
        ----------
        current_spectrum : torch.Tensor
            Current spectrum tensor, shape (N,).

        Returns
        -------
        float
            L2 distance (sqrt of sum of squared differences).
        """
        if not self.is_initialized:
            return 0.0

        with torch.no_grad():
            l2_change = torch.norm(current_spectrum - self.shadow).item()
        return l2_change


def _compute_chi_squared_per_obs(residuals: torch.Tensor, weights: torch.Tensor) -> float:
    """
    Compute unnormalized chi-squared per observation.

    Computes chi2 to match validation.py: sum((weights * residuals)^2) / n_obs

    Parameters
    ----------
    residuals : torch.Tensor
        UNNORMALIZED residuals (observations - predictions), shape (n_obs,).
    weights : torch.Tensor
        Observation weights, shape (n_obs,).

    Returns
    -------
    float
        Chi-squared per observation: sum((weights * residuals)^2) / n_obs
    """
    # Compute weighted residuals first, then square (matches validation.py)
    weighted_residuals = weights * residuals
    chi_squared = torch.sum(weighted_residuals**2).item()
    n_obs = residuals.shape[0]
    return chi_squared / n_obs


def _compute_normality_pvalue(residuals: torch.Tensor) -> float:
    """
    Compute normality test p-value using scipy statistical tests.

    Uses Shapiro-Wilk test for N < 5000, otherwise uses D'Agostino's normaltest.
    High p-values (> 0.05) indicate residuals are normally distributed.

    Parameters
    ----------
    residuals : torch.Tensor
        Residuals (observations - predictions), shape (n_obs,).
        Will be converted to numpy array for scipy functions.

    Returns
    -------
    float
        P-value from normality test. Returns 0.0 if test fails.
    """
    from scipy.stats import normaltest, shapiro

    # Convert to numpy (detaches from computation graph)
    residuals_np = residuals.detach().cpu().numpy()

    try:
        # Use Shapiro-Wilk for small samples, normaltest for larger
        if len(residuals_np) < 5000:
            _, pvalue = shapiro(residuals_np)
        else:
            _, pvalue = normaltest(residuals_np)
        return float(pvalue)
    except Exception as e:
        # If normality test fails, log warning and return 0.0
        import warnings

        warnings.warn(f"Normality test failed: {e}", RuntimeWarning)
        return 0.0


def solve_global_reconstruction(
    data: GlobalSpectralData,
    config: SEDConfig,
    ensemble_member: Optional[int] = None,
    progress: bool = True,
    progress_queue=None,
    progress_interval: int = 10,
):
    """
    Solve for global spectrum using Deep Image Prior.

    Parameters
    ----------
    data : GlobalSpectralData
        Global spectral observation data with sparse H matrix.
    config : SEDConfig
        Reconstruction configuration.
    ensemble_member : Optional[int]
        Ensemble member index (0-based) if running as part of ensemble.

    Returns
    -------
    tuple
        (result_spectrum, solver_status, solver_time) where:
        - result_spectrum: torch.Tensor, reconstructed spectrum (shape: global_resolution)
        - solver_status: str, status message
        - solver_time: float, computation time in seconds
    """
    # Set random seed for reproducible ensemble generation
    if config.ensemble_random_seed is not None:
        set_random_seed(config.ensemble_random_seed)

    device = torch.device(config.device if torch.backends.mps.is_available() or torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # PyTorch MPS backend often lacks sparse ops (mv/addmv). Instead of CPU fallback (which
    # causes per-epoch device sync), use a differentiable COO SpMV implemented with gather + index_add_.
    use_coo_spmv = device.type == "mps"
    if use_coo_spmv:
        logger.info(
            "MPS detected: using custom COO SpMV (gather + index_add_) to avoid sparse CPU fallback and device sync."
        )

    # Move observation data and sparse components to the main device once (avoid per-epoch transfers)
    H_indices = data.H_indices.to(device)
    H_values = data.H_values.to(device)
    observations = data.observations.to(device)
    weights = data.weights.to(device)

    # Prepare either a torch sparse tensor (CUDA/CPU) or COO buffers (MPS)
    if use_coo_spmv:
        H_row = H_indices[0].long()
        H_col = H_indices[1].long()
        H_val = H_values
        n_obs = int(data.H_shape[0])
        H_sparse = None
    else:
        H_sparse = torch.sparse_coo_tensor(H_indices, H_values, data.H_shape, device=device)

    # Initialize model on main device (can be MPS)
    n_pixels = config.global_resolution
    model = SpectralModel(n_pixels, config).to(device)

    # Initialize CWT regularization on main device
    cwt = GaussianCWT(config.cwt_scales, device=str(device)).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Learning rate scheduler
    scheduler = create_learning_rate_scheduler(optimizer, config)
    if scheduler is not None:
        logger.info(f"Using learning rate scheduler: {config.learning_rate_scheduler_type}")
        if config.learning_rate_scheduler_type in ["cosine_warmup", "cosine", "warmup"]:
            logger.info(
                f"Warmup epochs: {config.learning_rate_warmup_epochs}, Min LR factor: {config.learning_rate_min_factor}"
            )

    # Start timing
    start_time = time.time()

    # Training loop
    logger.info(f"Starting DIP optimization ({config.epochs} epochs)...")

    # Compute data characteristic scale for adaptive normalization (scalar)
    # Robust normalization using median absolute deviation (less sensitive to outliers)
    median_obs = torch.median(observations)
    mad = torch.median(torch.abs(observations - median_obs))
    flux_scale = mad * 1.4826 + 1e-6  # 1.4826 converts MAD to std equivalent for normal distribution
    # Copy to main device for model normalization use
    flux_scale_device = flux_scale.to(device)

    logger.info(f"Data flux scale (MAD-based): {flux_scale.item():.4e}")

    # Model evolution tracking for convergence analysis
    # NOTE: This is intended for wandb/diagnostics only. If wandb is not enabled,
    # we avoid extra device->CPU sync and spectrum cloning.
    wandb_enabled = config.is_wandb_enabled()
    track_convergence = wandb_enabled and config.wandb_track_convergence
    previous_spectrum = None
    spectrum_changes = []

    # Initialize EMA tracker if enabled
    ema_tracker = None
    ema_start_epoch = (
        config.ema_start_epoch if config.ema_start_epoch is not None else config.learning_rate_warmup_epochs
    )
    if config.use_ema:
        ema_tracker = EMATracker(decay=config.ema_decay)
        logger.info(f"EMA enabled with decay={config.ema_decay}, start_epoch={ema_start_epoch}")
    else:
        logger.info("EMA disabled")

    # Initialize wandb logging if enabled
    if wandb_enabled:
        try:
            # Log initial configuration
            config.log_to_wandb(config.to_wandb_config())
            config.log_to_wandb(
                {
                    "training_started": True,
                    "total_epochs": config.epochs,
                    "learning_rate": config.learning_rate,
                    "regularization_weight": config.regularization_weight,
                }
            )

            # Save initial model state and input noise as artifacts if enabled
            if config.wandb_save_model_artifacts:
                import json

                import wandb

                # Save initial model state
                model_artifact = wandb.Artifact("model_initial", type="model")
                with model_artifact.new_file("model_initial.pth", mode="wb") as f:
                    torch.save(model.state_dict(), f)
                config.wandb_run.log_artifact(model_artifact, aliases=["initial"])

                # Save input static noise
                noise_artifact = wandb.Artifact("input_static_noise", type="noise")
                noise_data = {
                    "z_static": model.z_static.detach().cpu().numpy().tolist(),
                    "noise_std": config.dip_noise_std,
                    "n_pixels": model.n_pixels,
                }
                with noise_artifact.new_file("static_noise.json", mode="w") as f:
                    json.dump(noise_data, f)
                config.wandb_run.log_artifact(noise_artifact)

        except Exception as e:
            import warnings

            warnings.warn(f"Failed to initialize wandb logging: {e}", RuntimeWarning)

    # Progress reporting options:
    # - If progress_queue is provided, the worker will NOT create a tqdm instance.
    #   Instead it will periodically push ("progress", member_index, epoch_done) to the queue.
    # - Otherwise, use local tqdm if progress=True.

    # Calculate total epochs for progress bar (Phase 1 + Phase 2 if SGLD enabled)
    total_epochs = config.epochs + (config.sgld_epochs if config.enable_sgld else 0)

    if progress_queue is not None:
        epoch_iter = range(config.epochs)
    elif progress:
        # Single reconstruction: safe default.
        # For ensemble members, prefer disabling progress and letting the parent track members.
        if ensemble_member is not None:
            position = ensemble_member
            desc = f"Member {ensemble_member + 1}"
            leave = True
        else:
            position = None
            desc = "Phase 1: MAP Optimization" if config.enable_sgld else "Optimizing Spectrum"
            leave = True

        # Create progress bar with total epochs (both phases if SGLD enabled)
        epoch_iter = tqdm(range(total_epochs), desc=desc, position=position, leave=leave)
    else:
        epoch_iter = iter(range(config.epochs))  # Convert to iterator for next() calls
    # Determine if adaptive parameters are needed (log and cauchy methods only)
    use_adaptive = config.regularization_method in ["log", "cauchy"]

    # Initialize early stopping info
    early_stop_info: Dict[str, any] = {
        "status": None,  # "PES", "FES", or None
        "trigger_epoch": None,
        "chi2": None,
        "pvalue": None,
    }

    # Initialize cached chi2 value for progress bar (computed only during early stopping checks)
    chi2_per_obs_cached = float("inf")  # Start with infinity (will be updated on first check)
    pvalue_normality_cached = 0.0

    # Track cooldown phase for early stopping
    in_cooldown_phase = False  # Flag to indicate we're in cooldown phase
    cooldown_start_epoch = None  # When cooldown phase started (for breaking after N epochs)

    for epoch in epoch_iter:
        # Skip training during the gap between early stopping trigger and cooldown phase
        if in_cooldown_phase and cooldown_start_epoch is not None and epoch < cooldown_start_epoch:
            # Advance scheduler to keep LR in sync
            if scheduler is not None:
                scheduler.step()
            continue  # Skip this epoch without training

        # Check if cooldown phase is complete (early stopping)
        if in_cooldown_phase and cooldown_start_epoch is not None and epoch >= cooldown_start_epoch:
            epochs_in_cooldown = epoch - cooldown_start_epoch + 1  # +1 because current epoch counts
            if epochs_in_cooldown > config.early_stop_cooldown_epoch:
                logger.info(
                    f"Cooldown phase complete at epoch {epoch} "
                    f"({epochs_in_cooldown - 1} cooldown epochs run). "
                    f"Early stopping: {early_stop_info['status']} at epoch {early_stop_info['trigger_epoch']}"
                )
                break

        # Update jitter component based on current epoch
        model.update_jitter(epoch)

        optimizer.zero_grad()

        # Generate spectrum on main device
        spectrum = model()  # (N,)

        # 1. Data Fidelity Loss
        # Compute predicted observations: y_pred = H @ spectrum
        if use_coo_spmv:
            y_pred = _coo_spmv(H_row, H_col, H_val, spectrum, n_obs)
        else:
            y_pred = torch.mv(H_sparse, spectrum)

        # Normalized weighted MSE: loss_data = sum( w * ((y - y_pred) / scale)^2 )
        diff = (observations - y_pred) / flux_scale
        loss_data = torch.sum(weights * (diff**2))

        # 2. Regularization Loss (CWT Sparsity on Normalized Spectrum)
        # Normalize spectrum before applying CWT to maintain consistent regularization strength
        cwt_coeffs = cwt(spectrum / flux_scale_device)

        # Set adaptive floor based on warmup phase (log and cauchy only)
        if use_adaptive:
            if epoch < config.learning_rate_warmup_epochs:
                current_floor = torch.tensor(config.reg_warmup_floor, device=device)
            else:
                current_floor = torch.tensor(config.reg_normal_floor, device=device)

        # Compute regularization loss per CWT scale
        loss_reg = 0.0

        if config.regularization_method == "absolute":
            # Simple L1: no adaptive parameters needed
            for coeff in cwt_coeffs:
                loss_reg += torch.sum(torch.abs(coeff))

        elif config.regularization_method == "log":
            # Log-sum with adaptive epsilon
            for coeff in cwt_coeffs:
                with torch.no_grad():
                    # MAD-based noise estimation (robust against outliers)
                    mad_sigma = 1.4826 * torch.median(torch.abs(coeff - torch.median(coeff)))
                    adaptive_epsilon = config.reg_sensitivity_factor * mad_sigma
                    effective_epsilon = torch.max(adaptive_epsilon, current_floor)

                # Multiply by effective_epsilon to maintain consistent reg strength
                loss_reg += effective_epsilon * log_loss(coeff, epsilon=effective_epsilon.item())

        elif config.regularization_method == "cauchy":
            # Cauchy loss with adaptive gamma
            for coeff in cwt_coeffs:
                with torch.no_grad():
                    # MAD-based noise estimation (robust against outliers)
                    mad_sigma = 1.4826 * torch.median(torch.abs(coeff - torch.median(coeff)))
                    adaptive_gamma = config.reg_sensitivity_factor * mad_sigma
                    effective_gamma = torch.max(adaptive_gamma, current_floor)

                # Multiply by effective_gamma to maintain consistent reg strength
                loss_reg += effective_gamma * cauchy_loss(coeff, gamma=effective_gamma.item())

        # Apply regularization weight
        loss_reg = loss_reg * config.regularization_weight

        # Total Loss
        loss = loss_data + loss_reg

        # Backward
        loss.backward()

        # Gradient clipping to prevent explosion
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # EMA update (after optimizer step, using the fixed noise spectrum)
        if ema_tracker is not None and epoch >= ema_start_epoch:
            # Use fixed noise spectrum for EMA to get consistent tracking
            fixed_spectrum = model.forward_fixed_noise()
            ema_tracker.update(fixed_spectrum)

        # Learning rate scheduler step
        if scheduler is not None:
            scheduler.step()

        # Early stopping check (after warmup, every N epochs)
        if (
            config.enable_early_stopping
            and epoch >= config.learning_rate_warmup_epochs
            and epoch % config.early_stop_check_steps == 0
            and early_stop_info["status"] is None  # Only check if not already triggered
        ):
            # Compute raw residuals (not scaled) to match validation.py chi2 calculation
            # validation.py computes: chi2 = sum((weights * (y - H@spectrum))^2) / n_obs
            residuals_raw = observations - y_pred

            # Calculate chi-squared per observation (matches validation.py formula)
            chi2 = _compute_chi_squared_per_obs(residuals_raw, weights)

            # Update cached chi2 for progress bar display
            chi2_per_obs_cached = chi2

            # Calculate normality test p-value (use raw residuals)
            pvalue = _compute_normality_pvalue(residuals_raw)

            # Update cached pvalue for progress bar display
            pvalue_normality_cached = pvalue

            # Check perfect early stop criteria (chi2 < target AND pvalue > 0.05)
            if chi2 < config.early_stop_target_chi2 and pvalue > 0.05:
                early_stop_info["status"] = "PES"
                early_stop_info["trigger_epoch"] = epoch
                early_stop_info["chi2"] = chi2
                early_stop_info["pvalue"] = pvalue

                cooldown_start_epoch = config.epochs - config.early_stop_cooldown_epoch

                logger.info(
                    f"Perfect Early Stop (PES) triggered at epoch {epoch}: "
                    f"chi2={chi2:.3f} (target={config.early_stop_target_chi2}), "
                    f"pvalue={pvalue:.3f}. Entering cooldown phase: "
                    f"{cooldown_start_epoch} → {config.epochs} ({config.early_stop_cooldown_epoch} epochs)"
                )

                # Set cooldown flag (epochs between now and cooldown_start will be skipped)
                in_cooldown_phase = True

            # Check force early stop criteria (chi2 < lowest, regardless of pvalue)
            elif chi2 < config.early_stop_lowest_chi2:
                early_stop_info["status"] = "FES"
                early_stop_info["trigger_epoch"] = epoch
                early_stop_info["chi2"] = chi2
                early_stop_info["pvalue"] = pvalue

                cooldown_start_epoch = config.epochs - config.early_stop_cooldown_epoch

                logger.info(
                    f"Force Early Stop (FES) triggered at epoch {epoch}: "
                    f"chi2={chi2:.3f} (lowest={config.early_stop_lowest_chi2}). "
                    f"Entering cooldown phase: {cooldown_start_epoch} → {config.epochs} "
                    f"({config.early_stop_cooldown_epoch} epochs)"
                )

                # Set cooldown flag (epochs between now and cooldown_start will be skipped)
                in_cooldown_phase = True

        # Logging and model evolution tracking
        current_loss = loss.item()

        # Parent-managed per-member progress (safe for notebooks + multiprocessing).
        # Send a small postfix dict that can be rendered via tqdm.set_postfix() in the parent.
        if progress_queue is not None and ensemble_member is not None:
            if progress_interval <= 1 or epoch % progress_interval == 0 or epoch == config.epochs - 1:
                try:
                    current_lr = optimizer.param_groups[0]["lr"]
                    postfix = {
                        "Loss": f"{current_loss:.3e}",
                        "Reg": f"{loss_reg.item():.3e}",
                        "LR": f"{current_lr:.2e}",
                        "Chi2": f"{chi2_per_obs_cached:.3f}",
                        "Pval": f"{pvalue_normality_cached:.2f}",
                    }
                    if early_stop_info["status"] is not None:
                        postfix["ES"] = early_stop_info["status"]
                    progress_queue.put_nowait(("progress", ensemble_member, epoch + 1, postfix))
                except Exception:
                    # Best-effort progress reporting; drop updates if queue is unavailable/full.
                    pass

        # Track model evolution for convergence analysis (wandb only)
        if track_convergence and previous_spectrum is not None:
            # Calculate spectrum change metrics
            spectrum_cpu = spectrum.detach().cpu().numpy()
            prev_spectrum_cpu = previous_spectrum.cpu().numpy()

            # L1 and L2 differences
            l1_diff = np.mean(np.abs(spectrum_cpu - prev_spectrum_cpu))
            l2_diff = np.sqrt(np.mean((spectrum_cpu - prev_spectrum_cpu) ** 2))

            # Relative change (normalized by current spectrum magnitude)
            current_magnitude = np.mean(np.abs(spectrum_cpu))
            relative_change = l1_diff / (current_magnitude + 1e-10)

            spectrum_changes.append(
                {
                    "epoch": epoch,
                    "l1_difference": l1_diff,
                    "l2_difference": l2_diff,
                    "relative_change": relative_change,
                    "current_magnitude": current_magnitude,
                }
            )

        # Update previous spectrum only if it will be used
        if track_convergence:
            previous_spectrum = spectrum.detach().clone()
        else:
            previous_spectrum = None

        # Wandb logging (do not even prepare metrics / convert tensors if wandb is disabled)
        if wandb_enabled and epoch % config.wandb_log_frequency == 0:
            # Get current learning rate for logging
            current_lr = optimizer.param_groups[0]["lr"]

            # Prepare metrics for logging
            metrics = {
                "epoch": epoch,
                "total_loss": current_loss,
                "data_loss": loss_data.item(),
                "regularization_loss": loss_reg.item(),
                "learning_rate": current_lr,
            }

            # Add convergence metrics if available
            if track_convergence and spectrum_changes:
                latest_change = spectrum_changes[-1]
                metrics.update(
                    {
                        "spectrum_l1_change": latest_change["l1_difference"],
                        "spectrum_l2_change": latest_change["l2_difference"],
                        "spectrum_relative_change": latest_change["relative_change"],
                        "spectrum_magnitude": latest_change["current_magnitude"],
                    }
                )

            # Add jitter information
            if hasattr(model, "current_jitter_std"):
                metrics["jitter_std"] = model.current_jitter_std.item()

            # Add EMA metrics if enabled and active
            if ema_tracker is not None and epoch >= ema_start_epoch and ema_tracker.is_initialized:
                fixed_spectrum = model.forward_fixed_noise()
                ema_l1_change = ema_tracker.get_l1_change(fixed_spectrum)
                ema_l2_change = ema_tracker.get_l2_change(fixed_spectrum)
                metrics.update(
                    {
                        "ema_l1_change": ema_l1_change,
                        "ema_l2_change": ema_l2_change,
                        "ema_active": True,
                        "ema_epoch": epoch - ema_start_epoch + 1,  # How many epochs of EMA tracking
                    }
                )

                # Log EMA spectrum evolution periodically
                if epoch % config.wandb_log_frequency == 0:
                    ema_spectrum = ema_tracker.get_ema_spectrum()
                    if ema_spectrum is not None:
                        ema_spectrum_cpu = ema_spectrum.detach().cpu().numpy()
                        wavelength_cpu = data.global_wavelength_grid.cpu().numpy()
                        config.log_spectrum_data_to_wandb(
                            ema_spectrum_cpu, wavelength_cpu, epoch, data_type="ema_spectrum"
                        )
            elif ema_tracker is not None:
                # EMA enabled but not yet active
                metrics["ema_active"] = False

            # Log to wandb
            config.log_to_wandb(metrics, step=epoch)

            # Log spectrum evolution if enabled
            if epoch % config.wandb_log_frequency == 0:
                # Use fixed noise only for consistent spectrum logging
                fixed_spectrum = model.forward_fixed_noise()
                spectrum_cpu = fixed_spectrum.detach().cpu().numpy()
                wavelength_cpu = data.global_wavelength_grid.cpu().numpy()
                config.log_spectrum_data_to_wandb(spectrum_cpu, wavelength_cpu, epoch, data_type="spectrum")

        # Progress bar logging (existing functionality)
        if epoch % progress_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            log_dict = {
                "Loss": f"{current_loss:.3e}",
                "Reg": f"{loss_reg.item():.3e}",
                "LR": f"{current_lr:.2e}",
                "Chi2": f"{chi2_per_obs_cached:.3f}",
                "Pval": f"{pvalue_normality_cached:.2f}",
            }
            # Add early stopping status to progress bar if triggered
            if early_stop_info["status"] is not None:
                log_dict["ES"] = early_stop_info["status"]
            # Only update postfix if we're using tqdm.
            if progress_queue is None and progress:
                epoch_iter.set_postfix(log_dict)

    # End of Phase 1
    phase1_epochs_completed = epoch + 1
    logger.info(f"Phase 1 complete: {phase1_epochs_completed} epochs")

    # Get final spectrum, preferring EMA if available and active
    if ema_tracker is not None and ema_tracker.is_initialized:
        ema_spectrum = ema_tracker.get_ema_spectrum()
        if ema_spectrum is not None:
            best_fit_spectrum = ema_spectrum
            logger.info("Using EMA-smoothed spectrum as final result")
        else:
            best_fit_spectrum = model.forward_fixed_noise().detach()
            logger.warning("EMA tracker initialized but no EMA spectrum available, using fixed noise spectrum")
    else:
        best_fit_spectrum = model.forward_fixed_noise().detach()
        if ema_tracker is not None:
            logger.info(
                f"EMA tracker did not initialize (completed epochs: {config.epochs}, start_epoch: {ema_start_epoch})"
            )

    # ========== PHASE 2: SGLD SAMPLING (Uncertainty Quantification) ==========
    sgld_samples = []
    sgld_mean = None
    sgld_std = None

    if config.enable_sgld:
        logger.info(
            f"Starting Phase 2: SGLD Sampling (total {config.sgld_epochs} epochs: "
            f"{config.sgld_burnin_epochs} burn-in + "
            f"{config.sgld_epochs - config.sgld_burnin_epochs} sampling)"
        )

        # Set fixed learning rate for SGLD
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.sgld_lr

        logger.info(f"SGLD learning rate set to {config.sgld_lr:.2e}")

        # Update progress bar description for Phase 2
        if progress and progress_queue is None:
            if ensemble_member is not None:
                epoch_iter.set_description(f"Member {ensemble_member + 1}: Phase 2 SGLD")
            else:
                epoch_iter.set_description("Phase 2: SGLD Sampling")

        # Phase 2A: Burn-in period (no noise, no sampling)
        logger.info(f"Phase 2A: Burn-in period ({config.sgld_burnin_epochs} epochs, no noise)")
        for sgld_epoch in range(config.sgld_burnin_epochs):
            # Update jitter (continue from Phase 1's final epoch)
            model.update_jitter(phase1_epochs_completed + sgld_epoch)

            optimizer.zero_grad()

            # Generate spectrum
            spectrum = model()

            # Data fidelity loss
            if use_coo_spmv:
                y_pred = _coo_spmv(H_row, H_col, H_val, spectrum, n_obs)
            else:
                y_pred = torch.mv(H_sparse, spectrum)

            diff = (observations - y_pred) / flux_scale
            loss_data = torch.sum(weights * (diff**2))

            # Regularization loss
            cwt_coeffs = cwt(spectrum / flux_scale_device)

            if use_adaptive:
                if sgld_epoch < config.learning_rate_warmup_epochs:
                    current_floor = torch.tensor(config.reg_warmup_floor, device=device)
                else:
                    current_floor = torch.tensor(config.reg_normal_floor, device=device)

            loss_reg = 0.0
            if config.regularization_method == "absolute":
                for coeff in cwt_coeffs:
                    loss_reg += torch.sum(torch.abs(coeff))
            elif config.regularization_method == "log":
                for coeff in cwt_coeffs:
                    with torch.no_grad():
                        mad_sigma = 1.4826 * torch.median(torch.abs(coeff - torch.median(coeff)))
                        adaptive_epsilon = config.reg_sensitivity_factor * mad_sigma
                        effective_epsilon = torch.max(adaptive_epsilon, current_floor)
                    loss_reg += effective_epsilon * log_loss(coeff, epsilon=effective_epsilon.item())
            elif config.regularization_method == "cauchy":
                for coeff in cwt_coeffs:
                    with torch.no_grad():
                        mad_sigma = 1.4826 * torch.median(torch.abs(coeff - torch.median(coeff)))
                        adaptive_gamma = config.reg_sensitivity_factor * mad_sigma
                        effective_gamma = torch.max(adaptive_gamma, current_floor)
                    loss_reg += effective_gamma * cauchy_loss(coeff, gamma=effective_gamma.item())

            loss_reg = loss_reg * config.regularization_weight
            loss = loss_data + loss_reg

            # Backward
            loss.backward()

            # NO gradient noise injection during burn-in

            # Optimizer step (no scheduler in Phase 2)
            optimizer.step()

            # NO sample collection during burn-in

            # Update progress bar (manually, since we're iterating over range)
            if progress and progress_queue is None:
                epoch_iter.update(1)

            # Progress reporting
            if progress_queue is not None and ensemble_member is not None:
                if sgld_epoch % progress_interval == 0 or sgld_epoch == config.sgld_burnin_epochs - 1:
                    try:
                        postfix = {"Phase": "Burn-in", "Loss": f"{loss.item():.3e}", "LR": f"{config.sgld_lr:.2e}"}
                        # Total progress = Phase 1 completed + Phase 2 current epoch
                        total_progress = phase1_epochs_completed + sgld_epoch + 1
                        progress_queue.put_nowait(("progress", ensemble_member, total_progress, postfix))
                    except Exception:
                        pass

            # Log to wandb periodically
            if wandb_enabled and sgld_epoch % config.wandb_log_frequency == 0:
                metrics = {
                    "sgld_epoch": sgld_epoch,
                    "sgld_phase": "burnin",
                    "sgld_loss": loss.item(),
                    "sgld_data_loss": loss_data.item(),
                    "sgld_reg_loss": loss_reg.item(),
                    "sgld_lr": config.sgld_lr,
                }
                config.log_to_wandb(metrics, step=phase1_epochs_completed + sgld_epoch)

        logger.info("Phase 2A complete: Burn-in finished, starting sampling phase")

        # Phase 2B: Sampling period (with noise, collect samples)
        sampling_epochs = config.sgld_epochs - config.sgld_burnin_epochs
        logger.info(
            f"Phase 2B: Sampling period ({sampling_epochs} epochs, collect every {config.sgld_collect_interval} epochs)"
        )

        for sgld_epoch in range(sampling_epochs):
            actual_sgld_epoch = config.sgld_burnin_epochs + sgld_epoch

            # Update jitter (continue from Phase 1's final epoch)
            model.update_jitter(phase1_epochs_completed + actual_sgld_epoch)

            optimizer.zero_grad()

            # Generate spectrum
            spectrum = model()

            # Data fidelity loss
            if use_coo_spmv:
                y_pred = _coo_spmv(H_row, H_col, H_val, spectrum, n_obs)
            else:
                y_pred = torch.mv(H_sparse, spectrum)

            diff = (observations - y_pred) / flux_scale
            loss_data = torch.sum(weights * (diff**2))

            # Regularization loss
            cwt_coeffs = cwt(spectrum / flux_scale_device)

            if use_adaptive:
                if actual_sgld_epoch < config.learning_rate_warmup_epochs:
                    current_floor = torch.tensor(config.reg_warmup_floor, device=device)
                else:
                    current_floor = torch.tensor(config.reg_normal_floor, device=device)

            loss_reg = 0.0
            if config.regularization_method == "absolute":
                for coeff in cwt_coeffs:
                    loss_reg += torch.sum(torch.abs(coeff))
            elif config.regularization_method == "log":
                for coeff in cwt_coeffs:
                    with torch.no_grad():
                        mad_sigma = 1.4826 * torch.median(torch.abs(coeff - torch.median(coeff)))
                        adaptive_epsilon = config.reg_sensitivity_factor * mad_sigma
                        effective_epsilon = torch.max(adaptive_epsilon, current_floor)
                    loss_reg += effective_epsilon * log_loss(coeff, epsilon=effective_epsilon.item())
            elif config.regularization_method == "cauchy":
                for coeff in cwt_coeffs:
                    with torch.no_grad():
                        mad_sigma = 1.4826 * torch.median(torch.abs(coeff - torch.median(coeff)))
                        adaptive_gamma = config.reg_sensitivity_factor * mad_sigma
                        effective_gamma = torch.max(adaptive_gamma, current_floor)
                    loss_reg += effective_gamma * cauchy_loss(coeff, gamma=effective_gamma.item())

            loss_reg = loss_reg * config.regularization_weight
            loss = loss_data + loss_reg

            # Backward
            loss.backward()

            # SGLD: Inject adaptive gradient noise
            with torch.no_grad():
                grad_norm = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).norm()
                noise_std = config.sgld_noise_factor * grad_norm

                # Inject Gaussian noise into gradients
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param.grad) * noise_std
                        param.grad.add_(noise)

            # Optimizer step (no scheduler in Phase 2)
            optimizer.step()

            # Collect sample every N epochs during sampling phase
            if sgld_epoch % config.sgld_collect_interval == 0:
                with torch.no_grad():
                    current_spectrum = model().detach().cpu()
                    sgld_samples.append(current_spectrum)

            # Progress reporting
            if progress_queue is not None and ensemble_member is not None:
                if sgld_epoch % progress_interval == 0 or sgld_epoch == sampling_epochs - 1:
                    try:
                        postfix = {
                            "Phase": "Sampling",
                            "Samples": f"{len(sgld_samples)}",
                            "Loss": f"{loss.item():.3e}",
                            "GradNorm": f"{grad_norm:.3e}",
                        }
                        # Total progress = Phase 1 completed + Phase 2 current epoch
                        total_progress = phase1_epochs_completed + actual_sgld_epoch + 1
                        progress_queue.put_nowait(("progress", ensemble_member, total_progress, postfix))
                    except Exception:
                        pass

            # Log to wandb periodically
            if wandb_enabled and sgld_epoch % config.wandb_log_frequency == 0:
                metrics = {
                    "sgld_epoch": actual_sgld_epoch,
                    "sgld_phase": "sampling",
                    "sgld_loss": loss.item(),
                    "sgld_data_loss": loss_data.item(),
                    "sgld_reg_loss": loss_reg.item(),
                    "sgld_grad_norm": grad_norm.item(),
                    "sgld_lr": config.sgld_lr,
                    "sgld_samples_collected": len(sgld_samples),
                }
                config.log_to_wandb(metrics, step=phase1_epochs_completed + actual_sgld_epoch)

            # Update progress bar (manually, since we're iterating over range)
            if progress and progress_queue is None:
                epoch_iter.update(1)

        # Phase 2 complete: compute statistics
        if sgld_samples:
            logger.info(f"Phase 2 complete: collected {len(sgld_samples)} samples")

            # Stack samples and compute statistics
            sgld_samples_tensor = torch.stack(sgld_samples)  # Shape: (n_samples, n_wavelength)
            sgld_mean = sgld_samples_tensor.mean(dim=0)  # Shape: (n_wavelength,)
            sgld_std = sgld_samples_tensor.std(dim=0)  # Shape: (n_wavelength,)

            logger.info(
                f"SGLD statistics: "
                f"mean flux range: [{sgld_mean.min():.3f}, {sgld_mean.max():.3f}] μJy, "
                f"std uncertainty range: [{sgld_std.min():.3f}, {sgld_std.max():.3f}] μJy"
            )

            # Log SGLD results to wandb if enabled
            if config.is_wandb_enabled():
                try:
                    import wandb

                    wandb_summary = {
                        "sgld/n_samples": len(sgld_samples),
                        "sgld/mean_flux_min": float(sgld_mean.min()),
                        "sgld/mean_flux_max": float(sgld_mean.max()),
                        "sgld/std_min": float(sgld_std.min()),
                        "sgld/std_max": float(sgld_std.max()),
                    }
                    config.log_to_wandb(wandb_summary, step=phase1_epochs_completed + config.sgld_epochs)

                    # Log mean and std spectra if raw data saving is enabled
                    if config.wandb_save_raw_data:
                        wavelength_cpu = data.global_wavelength_grid.cpu().numpy()
                        sgld_mean_np = sgld_mean.cpu().numpy()
                        sgld_std_np = sgld_std.cpu().numpy()

                        config.log_spectrum_data_to_wandb(
                            sgld_mean_np,
                            wavelength_cpu,
                            phase1_epochs_completed + config.sgld_epochs,
                            data_type="sgld_mean",
                        )
                        config.log_spectrum_data_to_wandb(
                            sgld_std_np,
                            wavelength_cpu,
                            phase1_epochs_completed + config.sgld_epochs,
                            data_type="sgld_std",
                        )

                except Exception as e:
                    import warnings

                    warnings.warn(f"Failed to log SGLD results to wandb: {e}", RuntimeWarning)

    # End timing
    end_time = time.time()
    solver_time = end_time - start_time

    # Create status message
    solver_status = "success"

    logger.info(
        f"Optimization complete. Status: {solver_status}, Time: {solver_time:.2f}s, "
        f"Phase 1: {phase1_epochs_completed} epochs"
    )

    if config.enable_sgld:
        logger.info(f"Phase 2: {len(sgld_samples)} SGLD samples collected")

    # Final wandb logging and convergence analysis
    if wandb_enabled:
        try:
            # Log final results
            final_metrics = {
                "training_completed": True,
                "final_status": solver_status,
                "total_time_seconds": solver_time,
                "phase1_epochs_completed": phase1_epochs_completed,
                "epochs_per_second": phase1_epochs_completed / solver_time,
            }

            # Add EMA final information
            if ema_tracker is not None:
                final_metrics.update(
                    {
                        "ema_enabled": True,
                        "ema_start_epoch": ema_start_epoch,
                        "ema_was_initialized": ema_tracker.is_initialized,
                        "ema_final_used": ema_tracker.is_initialized and ema_tracker.get_ema_spectrum() is not None,
                        "ema_decay": config.ema_decay,
                    }
                )

                if ema_tracker.is_initialized:
                    final_metrics["ema_activated_at_epoch"] = ema_start_epoch
                    final_metrics["ema_tracking_epochs"] = phase1_epochs_completed - ema_start_epoch
            else:
                final_metrics["ema_enabled"] = False

            # Add SGLD information if enabled
            if config.enable_sgld:
                final_metrics.update(
                    {
                        "sgld_enabled": True,
                        "sgld_epochs": config.sgld_epochs,
                        "sgld_samples_collected": len(sgld_samples),
                    }
                )
            else:
                final_metrics["sgld_enabled"] = False

            # Add convergence analysis if tracking was enabled
            if track_convergence and spectrum_changes:
                # Compute convergence statistics
                l1_changes = [change["l1_difference"] for change in spectrum_changes]
                l2_changes = [change["l2_difference"] for change in spectrum_changes]
                relative_changes = [change["relative_change"] for change in spectrum_changes]

                final_metrics.update(
                    {
                        "convergence_mean_l1_change": np.mean(l1_changes),
                        "convergence_std_l1_change": np.std(l1_changes),
                        "convergence_final_l1_change": l1_changes[-1],
                        "convergence_mean_l2_change": np.mean(l2_changes),
                        "convergence_std_l2_change": np.std(l2_changes),
                        "convergence_final_l2_change": l2_changes[-1],
                        "convergence_mean_relative_change": np.mean(relative_changes),
                        "convergence_std_relative_change": np.std(relative_changes),
                        "convergence_final_relative_change": relative_changes[-1],
                    }
                )

                # Stopping criteria assessment
                recent_changes = relative_changes[-min(10, len(relative_changes)) :]  # Last 10 changes
                recent_stability = np.std(recent_changes)
                recent_mean = np.mean(recent_changes)

                final_metrics.update(
                    {
                        "stopping_recent_stability": recent_stability,
                        "stopping_recent_mean_change": recent_mean,
                        "stopping_recommendation": "stable" if recent_stability < 1e-4 else "unstable",
                    }
                )

            config.log_to_wandb(final_metrics)

            # Save final model state and spectrum data as artifacts if enabled
            if config.wandb_save_model_artifacts:
                import wandb

                # Save final model state
                final_model_artifact = wandb.Artifact("model_final", type="model")
                with final_model_artifact.new_file("model_final.pth", mode="wb") as f:
                    torch.save(model.state_dict(), f)
                config.wandb_run.log_artifact(final_model_artifact, aliases=["final"])

            # Log final spectrum data if enabled
            if config.wandb_save_raw_data:
                best_fit_spectrum_cpu = best_fit_spectrum.cpu().numpy()
                wavelength_cpu = data.global_wavelength_grid.cpu().numpy()
                config.log_spectrum_data_to_wandb(
                    best_fit_spectrum_cpu, wavelength_cpu, phase1_epochs_completed, data_type="final_spectrum"
                )

                # Log EMA spectrum if available and different from best fit
                if ema_tracker is not None and ema_tracker.is_initialized:
                    ema_final_spectrum = ema_tracker.get_ema_spectrum()
                    if ema_final_spectrum is not None:
                        ema_spectrum_cpu = ema_final_spectrum.detach().cpu().numpy()
                        config.log_spectrum_data_to_wandb(
                            ema_spectrum_cpu, wavelength_cpu, phase1_epochs_completed, data_type="final_ema_spectrum"
                        )

        except Exception as e:
            import warnings

            warnings.warn(f"Failed to log final results to wandb: {e}", RuntimeWarning)

    # Log early stopping status if triggered
    if early_stop_info["status"] is not None:
        logger.info(
            f"Early stopping completed: {early_stop_info['status']} at epoch {early_stop_info['trigger_epoch']}, "
            f"chi2={early_stop_info['chi2']:.3f}, pvalue={early_stop_info['pvalue']:.3f}"
        )

    # Return results based on whether SGLD was enabled
    if config.enable_sgld and sgld_mean is not None and sgld_std is not None:
        # Return with SGLD data
        # best_fit_spectrum: Phase 1 MAP result (clean)
        # sgld_samples_tensor: All samples from Phase 2
        # sgld_mean: Mean of SGLD samples
        # sgld_std: Standard deviation of SGLD samples (uncertainty)
        sgld_samples_tensor = torch.stack(sgld_samples)
        return (
            best_fit_spectrum,  # Clean MAP result from Phase 1
            sgld_samples_tensor.cpu().numpy(),  # All SGLD samples
            sgld_mean.cpu().numpy(),  # SGLD mean
            sgld_std.cpu().numpy(),  # SGLD std (uncertainty)
            solver_status,
            solver_time,
            early_stop_info,
        )
    else:
        # Standard return without SGLD
        return best_fit_spectrum, solver_status, solver_time, early_stop_info
