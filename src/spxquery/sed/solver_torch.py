"""
PyTorch solver for Deep Image Prior reconstruction.
"""

import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import SEDConfig
from .data_structures import GlobalSpectralData
from .regularization import GaussianCWT

logger = logging.getLogger(__name__)


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

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 32, depth: int = 3):
        super().__init__()

        self.depth = depth

        # Encoders
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        curr_filters = base_filters
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, curr_filters, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv1d(curr_filters, curr_filters, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        for i in range(depth):
            # Downsample
            self.downsamples.append(nn.MaxPool1d(2))

            # Double filters
            next_filters = curr_filters * 2
            self.encoders.append(
                nn.Sequential(
                    nn.Conv1d(curr_filters, next_filters, kernel_size=3, padding=1, padding_mode="reflect"),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(next_filters, next_filters, kernel_size=3, padding=1, padding_mode="reflect"),
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
            # Standard UNet: Encoder i outputs F filters. Decoder i takes F filters from upsample and F from skip. Total 2F.

            self.decoders.append(
                nn.Sequential(
                    nn.Conv1d(curr_filters * 3, curr_filters, kernel_size=3, padding=1, padding_mode="reflect"),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(curr_filters, curr_filters, kernel_size=3, padding=1, padding_mode="reflect"),
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
        self.net = SpectralUNet(in_channels=1, out_channels=1, base_filters=config.dip_filters, depth=config.dip_depth)
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


def solve_global_reconstruction(data: GlobalSpectralData, config: SEDConfig):
    """
    Solve for global spectrum using Deep Image Prior.

    Parameters
    ----------
    data : GlobalSpectralData
        Global spectral observation data with sparse H matrix.
    config : SEDConfig
        Reconstruction configuration.

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

    # Check for MPS sparse limitations
    # PyTorch on MPS often lacks sparse operations like mv / addmv
    use_sparse_cpu_fallback = device.type == "mps"

    if use_sparse_cpu_fallback:
        logger.info(
            "MPS detected: Falling back to CPU for sparse matrix operations to avoid 'aten::addmv_' NotImplementedError."
        )
        sparse_device = torch.device("cpu")
    else:
        sparse_device = device

    # Move sparse matrix components to sparse_device
    H_indices = data.H_indices.to(sparse_device)
    H_values = data.H_values.to(sparse_device)
    observations = data.observations.to(sparse_device)
    weights = data.weights.to(sparse_device)

    # Construct sparse H tensor on sparse_device
    H_sparse = torch.sparse_coo_tensor(H_indices, H_values, data.H_shape, device=sparse_device)

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
    if config.is_wandb_enabled():
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

    pbar = tqdm(range(config.epochs), desc="Optimizing Spectrum")

    # --- Self-Adaptive Parameters ---
    # cauchy_k: Sensitivity factor (kappa).
    # gamma = k * sigma_noise.
    # Typically, k=3 means "preserve signals > 3 sigma".
    # Since we are using flux_scale to normalize, we can use a simpler heuristic or dynamic estimation.
    cauchy_k = 3.0

    # [Protection] Safe floor for gamma during warmup and normal training
    # During warmup, we use a much larger floor to be "forgiving".
    warmup_gamma_floor = 0.1  # Very loose constraint (10% of flux) during warmup
    normal_gamma_floor = 1e-4  # Tight constraint (numerical stability) after warmup

    for epoch in pbar:
        # Update jitter component based on current epoch
        model.update_jitter(epoch)

        optimizer.zero_grad()

        # Generate spectrum on main device
        spectrum = model()  # (N,)

        # 1. Data Fidelity Loss
        # If fallback is active, move spectrum to CPU for sparse MV
        if use_sparse_cpu_fallback:
            spectrum_for_mv = spectrum.to(sparse_device)
        else:
            spectrum_for_mv = spectrum

        # Compute predicted observations: y_pred = H @ spectrum
        y_pred = torch.mv(H_sparse, spectrum_for_mv)

        # Normalized weighted MSE: loss_data = sum( w * ((y - y_pred) / scale)^2 )
        diff = (observations - y_pred) / flux_scale
        loss_data = torch.sum(weights * (diff**2))

        # 2. Regularization Loss (CWT Sparsity on Normalized Spectrum)
        # Normalize spectrum before applying CWT to maintain consistent regularization strength
        cwt_coeffs = cwt(spectrum / flux_scale_device)
        loss_reg = 0.0
        # for coeff in cwt_coeffs:
        #     loss_reg += cauchy_loss(coeff, gamma=base_gamma)
        # loss_reg += torch.sum(torch.abs(coeff))
        # loss_reg += torch.sum(torch.log(1 + torch.abs(coeff) / 1e-6))

        # --- Dynamic Noise Estimation (MAD) ---
        # We estimate the 'noise level' of the current CWT scale dynamically.
        # MAD is robust against the emission lines (outliers).
        # Note: We use .detach() because we don't want to optimize the gamma itself.
        # Determine Gamma Floor based on Epoch (Warmup Logic)
        if epoch < config.learning_rate_warmup_epochs:
            # During warmup, ensure gamma doesn't drop below a safe, large value.
            # This prevents huge gradients from CWT noise early on.
            current_gamma_floor = torch.tensor(warmup_gamma_floor, device=device)

            # Optional: Linear ramp-up of Reg Weight
            # ramp_factor = epoch / config.learning_rate_warmup_epochs
        else:
            # After warmup, allow gamma to go low (be sensitive), just keeping numerical stability.
            current_gamma_floor = torch.tensor(normal_gamma_floor, device=device)
            # ramp_factor = 1.0

        for coeff in cwt_coeffs:
            with torch.no_grad():
                median_val = torch.median(coeff)
                mad_sigma = 1.4826 * torch.median(torch.abs(coeff - median_val))

                # Calculate target adaptive gamma
                adaptive_gamma = cauchy_k * mad_sigma

                # Apply Protection Floor
                # During warmup, this will likely be dominated by `warmup_gamma_floor` (0.1)
                # After warmup, this will be dominated by `adaptive_gamma` (actual noise)
                effective_gamma = torch.max(adaptive_gamma, current_gamma_floor)

            loss_reg += cauchy_loss(coeff, gamma=effective_gamma)

        loss_reg = loss_reg * config.regularization_weight

        # Total Loss
        # Move loss_data to main device to add with loss_reg
        loss = loss_data.to(device) + loss_reg

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

        # Logging and model evolution tracking
        current_loss = loss.item()

        # Track model evolution for convergence analysis
        if config.wandb_track_convergence and previous_spectrum is not None:
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

        # Update previous spectrum
        previous_spectrum = spectrum.detach().clone()

        # Wandb logging
        if epoch % config.wandb_log_frequency == 0:
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
            if config.wandb_track_convergence and spectrum_changes:
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
        if epoch % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            log_dict = {
                "Loss": f"{current_loss:.4e}",
                "Data": f"{loss_data.item():.4e}",
                "Reg": f"{loss_reg.item():.4e}",
                "LR": f"{current_lr:.2e}",
            }
            pbar.set_postfix(log_dict)

    # End timing
    end_time = time.time()
    solver_time = end_time - start_time

    # Get final spectrum, preferring EMA if available and active
    if ema_tracker is not None and ema_tracker.is_initialized:
        ema_spectrum = ema_tracker.get_ema_spectrum()
        if ema_spectrum is not None:
            final_spectrum = ema_spectrum
            logger.info("Using EMA-smoothed spectrum as final result")
        else:
            final_spectrum = model.forward_fixed_noise().detach()
            logger.warning("EMA tracker initialized but no EMA spectrum available, using fixed noise spectrum")
    else:
        final_spectrum = model.forward_fixed_noise().detach()
        if ema_tracker is not None:
            logger.info(
                f"EMA tracker did not initialize (completed epochs: {config.epochs}, start_epoch: {ema_start_epoch})"
            )

    # Create status message
    solver_status = "success"

    logger.info(
        f"Optimization complete. Status: {solver_status}, Time: {solver_time:.2f}s, Final Loss: {current_loss:.4e}"
    )

    # Final wandb logging and convergence analysis
    if config.is_wandb_enabled():
        try:
            # Log final results
            final_metrics = {
                "training_completed": True,
                "final_status": solver_status,
                "total_time_seconds": solver_time,
                "final_loss": current_loss,
                "total_epochs_completed": config.epochs,
                "epochs_per_second": config.epochs / solver_time,
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
                    final_metrics["ema_tracking_epochs"] = config.epochs - ema_start_epoch
            else:
                final_metrics["ema_enabled"] = False

            # Add convergence analysis if tracking was enabled
            if config.wandb_track_convergence and spectrum_changes:
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
                final_spectrum_cpu = final_spectrum.cpu().numpy()
                wavelength_cpu = data.global_wavelength_grid.cpu().numpy()
                config.log_spectrum_data_to_wandb(
                    final_spectrum_cpu, wavelength_cpu, config.epochs, data_type="final_spectrum"
                )

                # Log EMA spectrum if available and different from final spectrum
                if ema_tracker is not None and ema_tracker.is_initialized:
                    ema_final_spectrum = ema_tracker.get_ema_spectrum()
                    if ema_final_spectrum is not None:
                        ema_spectrum_cpu = ema_final_spectrum.detach().cpu().numpy()
                        config.log_spectrum_data_to_wandb(
                            ema_spectrum_cpu, wavelength_cpu, config.epochs, data_type="final_ema_spectrum"
                        )

        except Exception as e:
            import warnings

            warnings.warn(f"Failed to log final results to wandb: {e}", RuntimeWarning)

    return final_spectrum.cpu(), solver_status, solver_time
