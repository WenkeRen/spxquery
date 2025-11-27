"""
Data structures for SED reconstruction.
"""

from dataclasses import dataclass
import torch

@dataclass
class PixelObservationData:
    """
    Container for flattened pixel-level observation data for global reconstruction.
    
    Attributes
    ----------
    pixel_indices : torch.LongTensor
        Indices into the global spectral grid (shape: N_total_samples).
    pixel_fluxes : torch.FloatTensor
        Target flux density for each sample (shape: N_total_samples).
    pixel_weights : torch.FloatTensor
        Weight for each sample (shape: N_total_samples).
    global_wavelength_grid : torch.FloatTensor
        The global wavelength grid (shape: global_resolution).
    """
    pixel_indices: torch.Tensor
    pixel_fluxes: torch.Tensor
    pixel_weights: torch.Tensor
    global_wavelength_grid: torch.Tensor
