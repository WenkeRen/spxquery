"""
PyTorch solver for Deep Image Prior reconstruction.
"""

import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import SEDConfig
from .data_structures import GlobalSpectralData
from .regularization import GaussianCWT

logger = logging.getLogger(__name__)


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
    Wrapper for SpectralUNet that holds the fixed input noise.
    """

    def __init__(self, n_pixels: int, config: SEDConfig):
        super().__init__()
        self.net = SpectralUNet(in_channels=1, out_channels=1, base_filters=config.dip_filters, depth=config.dip_depth)

        # Fixed input noise
        # Shape: (1, 1, N)
        self.register_buffer("z", torch.randn(1, 1, n_pixels) * config.dip_noise_std)

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
    use_sparse_cpu_fallback = (device.type == 'mps')
    
    if use_sparse_cpu_fallback:
        logger.info("MPS detected: Falling back to CPU for sparse matrix operations to avoid 'aten::addmv_' NotImplementedError.")
        sparse_device = torch.device('cpu')
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

    # Start timing
    start_time = time.time()

    # Training loop
    logger.info(f"Starting DIP optimization ({config.epochs} epochs)...")

    # Compute data characteristic scale for adaptive normalization (scalar)
    flux_scale = torch.std(observations) + 1e-6  # Add small value to avoid division by zero
    # Copy to main device for model normalization use
    flux_scale_device = flux_scale.to(device)

    logger.info(f"Data flux scale (std): {flux_scale.item():.4e}")

    best_loss = float("inf")
    best_spectrum = None

    pbar = tqdm(range(config.epochs), desc="Optimizing Spectrum")

    for epoch in pbar:
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
        optimizer.step()

        # Logging
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_spectrum = spectrum.detach().clone()

        if epoch % 100 == 0:
            pbar.set_postfix(
                {"Loss": f"{current_loss:.4e}", "Data": f"{loss_data.item():.4e}", "Reg": f"{loss_reg.item():.4e}"}
            )

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
