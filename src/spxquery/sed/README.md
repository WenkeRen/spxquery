# SED Reconstruction Module

High-resolution spectral reconstruction from SPHEREx narrow-band photometry using PyTorch Deep Image Prior optimization.

## Overview

This module reconstructs high-resolution spectra from SPHEREx's randomly-sampled narrow-band measurements using Deep Image Prior (DIP) neural network optimization with Continuous Wavelet Transform (CWT) regularization. The method is particularly powerful for the NEP/SEP deep fields where individual sources have ~40,000 measurements spanning 0.75-5.0 microns, reconstructed as a unified global spectrum.

## Mathematical Formulation

The reconstruction solves:

$$
\min_\theta \left( ||W(y - \mathcal{P}(G_\theta(z)))||_2^2 + R(G_\theta(z)) \right)
$$

where:

- **Data fidelity**: Weighted chi-squared (L2 norm) between observed photometry and neural network output
- **Deep Image Prior**: U-Net neural network $G_\theta$ with fixed input noise $z$ generates spectrum
- **CWT regularization**: Multi-scale sparsity prior using Continuous Wavelet Transform with Mexican Hat wavelets
- **Global reconstruction**: Single unified spectrum spanning 0.75-5.0 μm

## Quick Start

### Python API

```python
from spxquery.sed import SEDConfig, SEDReconstructor

# Basic usage with default settings
config = SEDConfig()
reconstructor = SEDReconstructor(config)
result = reconstructor.reconstruct_from_csv("lightcurve.csv")
result.save_all("output/")

# Custom DIP parameters
config = SEDConfig(
    epochs=5000,
    learning_rate=0.001,
    regularization_weight=2.0,
    global_resolution=4000
)
reconstructor = SEDReconstructor(config)
result = reconstructor.reconstruct_from_csv("lightcurve.csv")
```

### Export Configuration Template

```python
from spxquery.sed import export_default_sed_config

config_path = export_default_sed_config("output/")
# Edit output/sed_config.yaml to customize parameters
```

## Module Structure

```
sed/
├── __init__.py          - Module exports and public API
├── config.py            - SEDConfig dataclass with PyTorch DIP parameters
├── data_loader.py       - CSV loading and quality filtering
├── data_structures.py   - GlobalSpectralData container for sparse matrices
├── matrices.py          - Frequency-normalized measurement matrix construction
├── solver_torch.py      - PyTorch Deep Image Prior optimization
├── regularization.py    - Continuous Wavelet Transform with Mexican Hat wavelets
├── reconstruction.py    - Main SEDReconstructor orchestrator
├── validation.py        - Quality assessment and residual analysis
├── plots.py             - Diagnostic visualization
└── README.md            - This file
```

## Configuration Parameters

### Deep Image Prior Parameters

- `dip_filters` (int): Number of filters in U-Net architecture (default: 32)
- `dip_depth` (int): Depth of U-Net network (default: 3)
- `dip_noise_std` (float): Standard deviation of input noise (default: 0.1)

### Optimization Parameters

- `epochs` (int): Number of optimization iterations (default: 3000)
- `learning_rate` (float): Adam optimizer learning rate (default: 0.001)
- `learning_rate_scheduler_type` (str): 'none', 'cosine', 'cosine_warmup', or 'warmup' (default: 'cosine_warmup')
- `learning_rate_warmup_epochs` (int): Warmup epochs for cosine scheduler (default: 150)

### Regularization Parameters

- `regularization_weight` (float): CWT regularization strength (default: 1.0)
- `cwt_scales` (list): Wavelet scales for multi-scale constraints (default: [1.0, 2.0, 3.0])

### Reconstruction Parameters

- `global_resolution` (int): Output wavelength bins across 0.75-5.0 μm (default: 3000)
- `wavelength_range` (tuple): Reconstruction wavelength range in μm (default: (0.75, 5.0))
- `device` (str): PyTorch device ('cpu', 'cuda', 'mps') (default: 'mps')

### Quality Control

- `sigma_threshold` (float): Minimum SNR for quality filtering (default: 3.0)
- `bad_flags` (list): Pixel flags to reject (default: standard SPHEREx bad flags)
- `enable_sigma_clip` (bool): MAD-based outlier removal (default: True)

## Outputs

### CSV File: `sed_reconstruction.csv`

Columns:

- `wavelength` (microns): Global wavelength grid (0.75-5.0 μm)
- `flux` (microJansky): Reconstructed flux density
- `flux_error` (microJansky): Estimated reconstruction uncertainty

### YAML File: `sed_metadata.yaml`

Contains:

- Source information (name, RA, Dec)
- Deep Image Prior configuration (network architecture, optimization parameters)
- Quality metrics (chi-squared reduced, residuals statistics)
- Dataset information (number of measurements per band, quality control statistics)

### Diagnostic Plots

- `sed_diagnostic_summary.png`: Multi-panel overview with residuals and quality metrics
- `sed_reconstruction.png`: Reconstructed spectrum with confidence intervals

## Workflow

1. **Data Loading**: Load lightcurve CSV from SPXQuery processing pipeline
2. **Quality Filtering**: Apply SNR threshold and bad pixel flags
3. **Band Aggregation**: Combine all detector bands into global dataset
4. **Matrix Construction**: Build sparse measurement matrix H with frequency normalization
5. **Deep Image Prior Optimization**: Train U-Net network with CWT regularization
6. **Quality Assessment**: Compute chi-squared, residuals, and validation metrics
7. **Output**: Save global spectrum CSV, metadata YAML, and diagnostic plots

## Quality Metrics

### Reconstruction Quality

- `chi_squared_reduced`: Weighted residual sum of squares / degrees of freedom
  - Ideal: ≈ 1.0
  - > 2.0: Poor fit or underestimated errors
  - < 0.5: Overfitting or overestimated errors

- `line_flux_recovery`: Fraction of known emission line flux recovered (>95% for S/N>10)
- `continuum_deviation`: Maximum deviation from smooth continuum (<3%)

### Convergence Metrics

- Training loss progression and learning rate schedule
- CWT regularization term magnitude
- Data fidelity term convergence

## Integration with SPXQuery

This module is designed as a standalone tool but integrates with SPXQuery:

**Input**: Consumes `lightcurve.csv` from SPXQuery processing pipeline

**Columns Required**:

- `flux`, `flux_error` (microJansky)
- `wavelength`, `bandwidth` (microns)
- `band` (D1-D6)
- `flag` (pixel quality bitmap)
- `snr` (signal-to-noise ratio)

## Dependencies

- `torch`: Neural network optimization and automatic differentiation
- `numpy`: Numerical arrays and operations
- `scipy`: Sparse matrix construction and scientific computing
- `pandas`: CSV I/O and data manipulation
- `matplotlib`: Publication-quality visualization
- `astropy`: Physical constants and astronomical calculations
- `pyyaml`: Configuration file management

## Performance Notes

- **GPU Acceleration**: Automatic MPS/CUDA detection and fallback to CPU
- **Memory**: Sparse matrices keep memory usage reasonable (< 2 GB for typical sources)
- **Speed**: Global reconstruction ~10-30 seconds with GPU/MPS acceleration
- **Scalability**: Configurable resolution and network architecture for different requirements

## Hardware Requirements

### Recommended
- **GPU**: NVIDIA CUDA (RTX 3060+), Apple Silicon (M1/M2/M3), or Intel integrated GPU
- **RAM**: 8GB+ for typical datasets
- **Storage**: Minimal (input CSV + output files)

### Minimum
- **CPU**: Multi-core processor (slower but functional)
- **RAM**: 4GB for small datasets
- **Storage**: Same as recommended

## Usage Examples

### Basic Reconstruction

```python
from spxquery.sed import SEDReconstructor, SEDConfig

# Default configuration (3000 epochs, MPS device)
config = SEDConfig()
reconstructor = SEDReconstructor(config)
result = reconstructor.reconstruct_from_csv("my_lightcurve.csv")

# Access results
wavelengths = result.wavelengths
spectrum = result.spectrum
quality = result.validation_metrics

print(f"Chi-squared reduced: {quality.chi_squared_reduced:.3f}")
print(f"Reconstruction quality: {quality.assess_quality()}")
```

### Advanced Configuration

```python
config = SEDConfig(
    # Deep Image Prior architecture
    dip_filters=64,
    dip_depth=4,

    # Optimization parameters
    epochs=5000,
    learning_rate=0.0005,
    learning_rate_scheduler_type='cosine_warmup',

    # Regularization
    regularization_weight=2.0,
    cwt_scales=[0.5, 1.0, 2.0, 4.0],

    # Quality control
    sigma_threshold=5.0,
    enable_sigma_clip=True,

    # Device selection
    device='cuda'  # Force CUDA GPU
)

reconstructor = SEDReconstructor(config)
result = reconstructor.reconstruct_from_csv("lightcurve.csv")
result.save_all("high_quality_output/")
```

## References

See `SpecRefine.md` for detailed mathematical derivation and implementation details.

## Support

For issues or questions:

- GitHub Issues: [spxquery repository]
- Documentation: [spxquery.readthedocs.io]