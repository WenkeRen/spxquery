"""Processing modules for SPXQuery package."""

from .variability import (
    generate_empirical_sed,
    estimate_magnitude_at_wavelength,
    process_lightcurve_variability
)
from .variability_pipeline import (
    read_lightcurve_data,
    run_variability_pipeline
)

__all__ = [
    'generate_empirical_sed',
    'estimate_magnitude_at_wavelength', 
    'process_lightcurve_variability',
    'read_lightcurve_data',
    'run_variability_pipeline'
]