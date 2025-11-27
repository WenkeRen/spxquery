"""
Regularization modules for Deep Spectral Prior reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

class GaussianCWT(nn.Module):
    """
    Differentiable Continuous Wavelet Transform using Gaussian (Mexican Hat) wavelets.
    
    Implemented as a fixed 1D convolution layer.
    """
    
    def __init__(self, scales: List[float], device: str = 'cpu'):
        """
        Initialize Gaussian CWT module.
        
        Parameters
        ----------
        scales : List[float]
            List of scales (sigma) for the wavelets.
        device : str
            Device to place the kernels on.
        """
        super().__init__()
        self.scales = scales
        self.device = device
        
        # Create filters for each scale
        self.filters = nn.ModuleList()
        
        for scale in scales:
            # Determine kernel size (sufficient to capture the wavelet)
            # +/- 4 sigma covers >99.9% of the wavelet energy
            radius = int(4.0 * scale)
            kernel_size = 2 * radius + 1
            
            # Generate Ricker (Mexican Hat) wavelet
            t = torch.linspace(-4.0 * scale, 4.0 * scale, kernel_size, device=device)
            
            # Ricker wavelet formula: A * (1 - (t/s)^2) * exp(-0.5 * (t/s)^2)
            # Normalized to have unit L2 norm or similar? 
            # Standard definition:
            # psi(t) = (2 / (sqrt(3*sigma) * pi^0.25)) * (1 - (t/sigma)^2) * exp(-(t^2)/(2*sigma^2))
            
            A = 2 / (np.sqrt(3 * scale) * (np.pi ** 0.25))
            t_sq = (t / scale) ** 2
            kernel = A * (1 - t_sq) * torch.exp(-0.5 * t_sq)
            
            # Normalize sum to 0 (wavelet property) - Ricker naturally sums to ~0
            # But for discrete kernel, enforce sum=0 strictly
            kernel = kernel - kernel.mean()
            
            # Create Conv1d layer
            # in_channels=1, out_channels=1, kernel_size=kernel_size
            conv = nn.Conv1d(1, 1, kernel_size, padding='same', padding_mode='reflect', bias=False)
            conv.weight.data = kernel.view(1, 1, -1)
            conv.weight.requires_grad = False # Fixed weights
            
            self.filters.append(conv)
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply CWT to input signal.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape (N_pixels,) or (1, N_pixels) or (B, 1, N_pixels).
            
        Returns
        -------
        List[torch.Tensor]
            List of wavelet coefficient tensors, one for each scale.
        """
        # Ensure input is (B, 1, N)
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
            
        coeffs = []
        for conv in self.filters:
            out = conv(x)
            coeffs.append(out)
            
        return coeffs
