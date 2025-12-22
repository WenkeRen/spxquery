"""
SED (Spectral Energy Distribution) reconstruction module for SPHEREx.

This module provides tools to reconstruct high-resolution spectra from
SPHEREx narrow-band photometry using PyTorch-based Deep Image Prior
with Continuous Wavelet Transform regularization.

The reconstruction problem is formulated as a global optimization:
    min_θ ||W(y - H·G_θ(z))||_2^2 + R(G_θ(z))

where:
    - Data fidelity: weighted chi-squared (L2 norm)
    - G_θ(z): Deep Image Prior neural network generating spectrum
    - R(G_θ(z)): CWT regularization for multi-scale constraints
    - W: Observation weights based on measurement uncertainties

Main Classes
------------
SEDConfig : Configuration dataclass with PyTorch DIP parameters
SEDReconstructor : Main orchestrator for the reconstruction pipeline
SEDReconstructionResult : Container for reconstruction outputs
ValidationMetrics : Quality assessment metrics

Main Functions
--------------
export_default_sed_config : Export configuration template for customization

Examples
--------
Basic usage with default settings:

>>> from spxquery.sed import SEDConfig, SEDReconstructor
>>> config = SEDConfig(epochs=3000, regularization_weight=1.0)
>>> reconstructor = SEDReconstructor(config)
>>> result = reconstructor.reconstruct_from_csv("lightcurve.csv")
>>> result.save_all("output/")

With custom Deep Image Prior architecture:

>>> config = SEDConfig(
...     dip_filters=64,  # More filters in U-Net
...     dip_depth=4,     # Deeper network
...     regularization_weight=2.0,
...     cwt_scales=[1.0, 2.0, 4.0, 8.0]
... )
>>> reconstructor = SEDReconstructor(config)
>>> result = reconstructor.reconstruct_from_csv("lightcurve.csv")

"""

# Version
__version__ = "0.1.0"

# Configuration
from .config import SEDConfig, export_default_sed_config

# Main reconstruction classes
from .reconstruction import (
    SEDReconstructor,
    SEDReconstructionResult,
)

# Data structures
from .data_loader import BandData
from .validation import ValidationMetrics

# Public API
__all__ = [
    # Version
    "__version__",
    # Configuration
    "SEDConfig",
    "export_default_sed_config",
    # Main classes
    "SEDReconstructor",
    "SEDReconstructionResult",
    # Data structures
    "BandData",
    "ValidationMetrics",
]
