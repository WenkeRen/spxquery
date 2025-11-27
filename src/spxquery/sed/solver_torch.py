"""
PyTorch solver for Deep Image Prior reconstruction.
"""

import logging
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import SEDConfig
from .data_structures import PixelObservationData
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
            nn.Conv1d(in_channels, curr_filters, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(curr_filters, curr_filters, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        
        for i in range(depth):
            # Downsample
            self.downsamples.append(nn.MaxPool1d(2))
            
            # Double filters
            next_filters = curr_filters * 2
            self.encoders.append(nn.Sequential(
                nn.Conv1d(curr_filters, next_filters, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2),
                nn.Conv1d(next_filters, next_filters, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2)
            ))
            curr_filters = next_filters
            
        # Decoders
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(depth):
            prev_filters = curr_filters
            curr_filters = curr_filters // 2
            
            # Upsample
            self.upsamples.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=True))
            
            # Convolution (input channels = prev_filters from upsample + prev_filters from skip = 2*prev_filters?? No)
            # Skip connection brings 'curr_filters' channels. Upsample brings 'prev_filters' (which is 2*curr).
            # So concatenation has 3 * curr_filters? 
            # Standard UNet: Encoder i outputs F filters. Decoder i takes F filters from upsample and F from skip. Total 2F.
            
            self.decoders.append(nn.Sequential(
                nn.Conv1d(curr_filters * 3, curr_filters, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2),
                nn.Conv1d(curr_filters, curr_filters, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2)
            ))
            
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
            skip = skips[-(i+2)]
            
            # Handle size mismatch due to odd dimensions during downsampling
            if out.shape[-1] != skip.shape[-1]:
                out = F.interpolate(out, size=skip.shape[-1], mode='linear', align_corners=True)
                
            # Concatenate
            out = torch.cat([out, skip], dim=1)
            
            # Convolve
            out = self.decoders[i](out)
            
        # Final output
        out = self.final_conv(out)
        return F.softplus(out) # Enforce non-negativity


class SpectralModel(nn.Module):
    """
    Wrapper for SpectralUNet that holds the fixed input noise.
    """
    def __init__(self, n_pixels: int, config: SEDConfig):
        super().__init__()
        self.net = SpectralUNet(
            in_channels=1, 
            out_channels=1, 
            base_filters=config.dip_filters,
            depth=config.dip_depth
        )
        
        # Fixed input noise
        # Shape: (1, 1, N)
        self.register_buffer('z', torch.randn(1, 1, n_pixels) * config.dip_noise_std)
        
    def forward(self):
        # Output shape: (N,)
        return self.net(self.z).squeeze()


def solve_global_reconstruction(
    data: PixelObservationData,
    config: SEDConfig
) -> torch.Tensor:
    """
    Solve for global spectrum using Deep Image Prior.
    
    Parameters
    ----------
    data : PixelObservationData
        Pixel-level observation data.
    config : SEDConfig
        Reconstruction configuration.
        
    Returns
    -------
    torch.Tensor
        Reconstructed spectrum (shape: global_resolution).
    """
    device = torch.device(config.device if torch.backends.mps.is_available() or torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Move data to device
    pixel_indices = data.pixel_indices.to(device)
    pixel_fluxes = data.pixel_fluxes.to(device)
    pixel_weights = data.pixel_weights.to(device)
    
    # Initialize model
    n_pixels = config.global_resolution
    model = SpectralModel(n_pixels, config).to(device)
    
    # Initialize CWT regularization
    cwt = GaussianCWT(config.cwt_scales, device=str(device)).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    logger.info(f"Starting DIP optimization ({config.epochs} epochs)...")
    
    best_loss = float('inf')
    best_spectrum = None
    
    pbar = tqdm(range(config.epochs), desc="Optimizing Spectrum")
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Generate spectrum
        spectrum = model() # (N,)
        
        # 1. Data Fidelity Loss
        # Gather predicted fluxes at observed pixel indices
        pred_fluxes = spectrum[pixel_indices]
        
        # Weighted MSE
        # loss_data = sum( w * (y - y_pred)^2 )
        diff = pixel_fluxes - pred_fluxes
        loss_data = torch.sum(pixel_weights * (diff ** 2))
        
        # 2. Regularization Loss (CWT Sparsity)
        # L1 norm of wavelet coefficients
        cwt_coeffs = cwt(spectrum)
        loss_reg = 0.0
        for coeff in cwt_coeffs:
            loss_reg += torch.sum(torch.abs(coeff))
            
        loss_reg = loss_reg * config.regularization_weight
        
        # Total Loss
        loss = loss_data + loss_reg
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Logging
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_spectrum = spectrum.detach().clone()
            
        if epoch % 100 == 0:
            pbar.set_postfix({
                'Loss': f"{current_loss:.4e}", 
                'Data': f"{loss_data.item():.4e}", 
                'Reg': f"{loss_reg.item():.4e}"
            })
            
    logger.info(f"Optimization complete. Best Loss: {best_loss:.4e}")
    
    return best_spectrum.cpu()
