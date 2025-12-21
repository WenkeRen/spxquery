"""
PyTorch solver for Deep Image Prior reconstruction.
"""

import logging
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import SEDConfig
from .data_structures import GlobalSpectralData
from .regularization import GaussianCWT

logger = logging.getLogger(__name__)


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
            prev_filters = curr_filters
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
                current_ratio = (
                    self.config.dip_noise_jitter_min_ratio +
                    (self.config.dip_noise_jitter_initial_ratio - self.config.dip_noise_jitter_min_ratio) *
                    (1 - decay_progress)
                )
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
        if config.learning_rate_scheduler_type in ["cosine_warmup", "cosine"]:
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

    best_loss = float("inf")
    best_spectrum = None

    pbar = tqdm(range(config.epochs), desc="Optimizing Spectrum")

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
        for coeff in cwt_coeffs:
            loss_reg += torch.sum(torch.abs(coeff))

        loss_reg = loss_reg * config.regularization_weight

        # Total Loss
        # Move loss_data to main device to add with loss_reg
        loss = loss_data.to(device) + loss_reg

        # Backward
        loss.backward()

        # Gradient clipping to prevent explosion
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Learning rate scheduler step
        if scheduler is not None:
            scheduler.step()

        # Logging
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_spectrum = spectrum.detach().clone()

        if epoch % 100 == 0:
            # Get current learning rate for logging
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

    # Create status message
    if best_spectrum is not None:
        solver_status = "success"
    else:
        solver_status = "failed"
        best_spectrum = torch.zeros(config.global_resolution)

    logger.info(f"Optimization complete. Status: {solver_status}, Time: {solver_time:.2f}s, Best Loss: {best_loss:.4e}")

    return best_spectrum.cpu(), solver_status, solver_time
