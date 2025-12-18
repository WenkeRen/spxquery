"""
Data structures for SED reconstruction.
"""

from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass
class GlobalSpectralData:
    """
    Container for global spectral reconstruction data.
    
    Stores the sparse measurement matrix H, observed fluxes y, and weights w.
    H maps the global spectral grid x to observations y: y = H @ x.
    
    Attributes
    ----------
    H_indices : torch.Tensor
        Indices for sparse H matrix (2, nnz).
    H_values : torch.Tensor
        Values for sparse H matrix (nnz).
    H_shape : Tuple[int, int]
        Shape of H matrix (M_observations, N_spectral_bins).
    observations : torch.Tensor
        Observed flux densities y (M_observations).
    weights : torch.Tensor
        Observation weights w (M_observations).
    global_wavelength_grid : torch.Tensor
        The global wavelength grid (N_spectral_bins).
    """
    H_indices: torch.Tensor
    H_values: torch.Tensor
    H_shape: Tuple[int, int]
    observations: torch.Tensor
    weights: torch.Tensor
    global_wavelength_grid: torch.Tensor