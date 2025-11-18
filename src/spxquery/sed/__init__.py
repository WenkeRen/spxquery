"""
SED (Spectral Energy Distribution) reconstruction module for SPHEREx.

This module provides tools to reconstruct high-resolution spectra from
SPHEREx narrow-band photometry using regularized convex optimization.

The reconstruction problem is formulated as:
    min_x ( ||w(y - Hx)||_2^2 + lambda1*||x||_1 + lambda2*||D2 x||_2^2 )

where:
    - Data fidelity: weighted chi-squared (L2 norm)
    - L1 regularization: sparsity prior for emission lines
    - L2 regularization: smoothness prior for continuum

Main Classes
------------
SEDConfig : Configuration dataclass with all reconstruction parameters
SEDReconstructor : Main orchestrator for the reconstruction pipeline
SEDReconstructionResult : Container for reconstruction outputs
BandReconstructionResult : Single-band reconstruction result

Main Functions
--------------
reconstruct_sed_from_csv : Convenience function for one-line reconstruction
export_default_sed_config : Export configuration template for customization

Examples
--------
Basic usage with default settings:

>>> from spxquery.sed import SEDConfig, SEDReconstructor
>>> config = SEDConfig()
>>> reconstructor = SEDReconstructor(config)
>>> result = reconstructor.reconstruct_from_csv("lightcurve.csv")
>>> result.save_all("output/")

With auto-tuning enabled:

>>> config = SEDConfig(auto_tune=True)
>>> reconstructor = SEDReconstructor(config)
>>> result = reconstructor.reconstruct_from_csv("lightcurve.csv")

Using custom hyperparameters:

>>> config = SEDConfig(lambda1=0.5, lambda2=50.0)
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
    BandReconstructionResult,
    reconstruct_sed_from_csv,
)

# Data structures
from .data_loader import BandData
from .validation import ValidationMetrics
from .solver import ReconstructionResult
from .tuning import TuningResult

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
    "BandReconstructionResult",
    # Convenience function
    "reconstruct_sed_from_csv",
    # Data structures
    "BandData",
    "ValidationMetrics",
    "ReconstructionResult",
    "TuningResult",
]
