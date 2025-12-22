# SED Reconstruction Module

High-resolution spectral reconstruction from SPHEREx narrow-band photometry using PyTorch Deep Image Prior optimization with ensemble support, EMA smoothing, and wandb integration.

## Overview

This module reconstructs high-resolution spectra from SPHEREx's randomly-sampled narrow-band measurements using Deep Image Prior (DIP) neural network optimization with Continuous Wavelet Transform (CWT) regularization. The method is particularly powerful for the NEP/SEP deep fields where individual sources have ~40,000 measurements spanning 0.75-5.0 microns, reconstructed as a unified global spectrum.

## Mathematical Formulation

The reconstruction solves:

$$
\min_\theta \left( ||W(y - \mathcal{P}(G_\theta(z)))||_2^2 + R(G_\theta(z)) \right)
$$

where:

- **Data fidelity**: Weighted chi-squared (L2 norm) between observed photometry and neural network output
- **Deep Image Prior**: U-Net neural network $G_\theta$ with fixed input noise $z$ and optional jitter generates spectrum
- **CWT regularization**: Multi-scale sparsity prior using Continuous Wavelet Transform with Mexican Hat wavelets
- **Global reconstruction**: Single unified spectrum spanning 0.75-5.0 μm
- **Ensemble support**: Multiple independent reconstructions for robust uncertainty quantification
- **EMA smoothing**: Exponential Moving Average for stable, high-quality spectrum output

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

# Ensemble reconstruction for uncertainty quantification
config = SEDConfig(
    epochs=3000,
    ensemble_size=5,  # Run 5 independent reconstructions
    ensemble_random_seed=42  # For reproducible results
)
reconstructor = SEDReconstructor(config)
ensemble_result = reconstructor.reconstruct_from_csv("lightcurve.csv")
ensemble_result.save_all("ensemble_output/")
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
├── config.py            - SEDConfig dataclass with comprehensive PyTorch DIP parameters
├── data_loader.py       - CSV loading, quality filtering, and rolling MAD sigma clipping
├── data_structures.py   - Data containers: GlobalSpectralData, SEDReconstructionResult, EnsembleResult
├── matrices.py          - Frequency-normalized sparse measurement matrix construction
├── solver_torch.py      - PyTorch Deep Image Prior with EMA, ensemble support, and wandb integration
├── regularization.py    - Continuous Wavelet Transform with Mexican Hat wavelets
├── reconstruction.py    - Main SEDReconstructor orchestrator with ensemble/single auto-detection
├── validation.py        - Quality assessment, residual analysis, and train/validation splitting
├── plots.py             - Diagnostic visualization (in development)
└── README.md            - This file
```

## Configuration Parameters

### Deep Image Prior Parameters

- `dip_filters` (int): Number of filters in U-Net architecture (default: 32)
- `dip_depth` (int): Depth of U-Net network (default: 3)
- `dip_noise_std` (float): Standard deviation of input noise (default: 0.1)
- `dip_noise_jitter_initial_ratio` (float): Initial jitter as ratio of dip_noise_std (default: 0.3)
- `dip_noise_jitter_min_ratio` (float): Minimum jitter ratio for linear decay (default: None)

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
- `sigma_clip_sigma` (float): Sigma threshold for outlier detection (default: 3.0)
- `sigma_clip_window` (int): Rolling window size for local MAD calculation (default: 21)
- `sigma_clip_max_iterations` (int): Maximum number of iterative sigma clipping passes (default: 10)

### EMA (Exponential Moving Average) Parameters

- `use_ema` (bool): Enable EMA smoothing for spectrum quality control (default: True)
- `ema_decay` (float): EMA decay rate between 0.9 and 0.999 (default: 0.99)
- `ema_start_epoch` (int): EMA startup epoch (default: None, uses learning_rate_warmup_epochs)

### Ensemble Parameters

- `ensemble_size` (int): Number of ensemble members for uncertainty quantification (default: 1)
- `ensemble_random_seed` (int): Base seed for reproducible ensemble generation (default: None)
- `ensemble_strategy` (str): Ensembling approach (default: "independent")
- `ensemble_save_members` (bool): Whether to save individual ensemble member results (default: True)

### Weights & Biases Integration

- `wandb_run` (wandb.Run): wandb run instance for experiment tracking (default: None)
- `wandb_log_frequency` (int): How often to log training metrics (default: 100)
- `wandb_save_raw_data` (bool): Save raw spectrum data instead of PNG images (default: True)
- `wandb_save_model_artifacts` (bool): Save model state and input noise as artifacts (default: True)
- `wandb_track_convergence` (bool): Track convergence metrics for stopping decisions (default: True)

## Outputs

### Single Reconstruction Outputs

#### CSV File: `sed_reconstruction.csv`

Columns:

- `wavelength_microns`: Global wavelength grid (0.75-5.0 μm)
- `flux_microjansky`: Reconstructed flux density

#### YAML File: `sed_metadata.yaml`

Contains:

- Source information (name, RA, Dec)
- Deep Image Prior configuration (network architecture, optimization parameters)
- Quality metrics (chi-squared reduced, residuals statistics)
- Dataset information (number of measurements per band, quality control statistics)

### Ensemble Reconstruction Outputs

#### Mean Spectrum CSV: `ensemble_mean_spectrum.csv`

Columns:

- `wavelength_microns`: Global wavelength grid (0.75-5.0 μm)
- `flux_microjansky`: Mean ensemble flux density
- `flux_uncertainty_microjansky`: Standard deviation across ensemble (uncertainty estimate)
- `median_flux_microjansky`: Median ensemble flux density

#### Individual Member Spectra: `member_XX_spectrum.csv`

Separate CSV files for each ensemble member with:

- `wavelength_microns`: Global wavelength grid
- `flux_microjansky`: Individual member flux density

#### JSON File: `ensemble_metadata.json`

Contains:

- Ensemble configuration (size, strategy, random seed)
- Statistical summaries (mean/std uncertainty, convergence metrics)
- Individual member validation metrics

### Diagnostic Plots

- `sed_diagnostic_summary.png`: Multi-panel overview with residuals and quality metrics
- `sed_reconstruction.png`: Reconstructed spectrum with confidence intervals (in development)

## Workflow

1. **Data Loading**: Load lightcurve CSV from SPXQuery processing pipeline
2. **Quality Filtering**: Apply SNR threshold, bad pixel flags, and rolling MAD sigma clipping
3. **Band Aggregation**: Combine all detector bands into global dataset
4. **Matrix Construction**: Build sparse measurement matrix H with frequency normalization
5. **Deep Image Prior Optimization**: Train U-Net network with CWT regularization and optional EMA
6. **Quality Assessment**: Compute chi-squared, residuals, and validation metrics
7. **Ensemble Processing**: (Optional) Run multiple independent reconstructions for uncertainty quantification
8. **Output**: Save global spectrum CSV, metadata YAML, and diagnostic plots

## Rolling MAD Sigma Clipping

The module implements sophisticated outlier detection using rolling window Median Absolute Deviation (MAD) statistics:

- **Per-band processing**: Each SPHEREx detector band is processed independently
- **Wavelength-sorted data**: Measurements sorted by wavelength for meaningful local statistics
- **Iterative refinement**: Multiple passes to catch outliers masked by extreme values
- **Robust statistics**: MAD scaled to equivalent standard deviation (multiply by 1.4826)
- **Adaptive windows**: Configurable window size with minimum requirements for edge handling

Quality filters are applied in logical progression:

1. Bad pixel flags (remove known problematic data)
2. NaN values in critical columns
3. Non-positive flux_error values
4. SNR threshold filtering
5. Rolling MAD sigma clipping (statistical outlier removal)

## Quality Metrics

### Reconstruction Quality

The module computes several metrics to evaluate the quality of the reconstructed spectrum.

**Important Note**: For Deep Image Prior reconstruction with high-resolution reconstruction ($N \gg M$), standard reduced chi-squared ($\chi^2_\nu = \chi^2 / (M - N)$) is not statistically valid because degrees of freedom become negative or undefined. Instead, we use $\chi^2/M$ which provides meaningful assessment even when $N > M$.

- **`chi_squared_per_obs`**: $\chi^2 / M$. Average weighted residual squared per observation.
  - Ideal: $\approx 1.0$
  - $> 2.0$: Poor fit or underestimated errors.
  - $< 0.5$: Overfitting or overestimated errors.

- **`negative_flux_fraction`**: Fraction of spectral bins with negative flux.
  - Ideal: $0.0$ (Physical spectra must be non-negative).
  - Warning threshold: $> 5\%$.

- **`smoothness_tv`**: Normalized Total Variation (TV) of the spectrum.
  - Measures spectral roughness/oscillation.
  - Lower values indicate smoother spectra (preferred for continuum).
  - Very high values suggest noise fitting or instability.

- **`residual_rms`**: Root Mean Square of weighted residuals.
  - Expected to be close to 1.0 if weights are properly calibrated.
  - Deviations indicate systematic issues with error estimation.

- **`residual_oscillation`**: The von Neumann Ratio (or Mean Square Successive Difference) test p-value on weighted residuals.
  - The Statistic ($M$ or $\delta^2/s^2$): $\mathcal{M} = \frac{\sum_{i=1}^{n-1} (r_{i+1} - r_i)^2}{\sum_{i=1}^n (r_i - \bar{r})^2}$ Where $r$ is your residual vector.
  - $> 0.05$: Residuals are consistent with Gaussian noise (good).
  - $< 0.05$: Residuals are non-Gaussian (systematic errors or outliers).

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

## Advanced Features

### Weights & Biases Integration

The module supports comprehensive experiment tracking with wandb:

```python
import wandb
from spxquery.sed import SEDConfig, SEDReconstructor

# Initialize wandb run
wandb.init(project="sed-reconstruction", config={})

# Configure with wandb integration
config = SEDConfig(
    epochs=3000,
    wandb_run=wandb,  # Pass wandb run for automatic logging
    wandb_log_frequency=50,  # Log every 50 epochs
    wandb_save_raw_data=True,  # Save spectrum evolution
    wandb_track_convergence=True,  # Track convergence metrics
)

reconstructor = SEDReconstructor(config)
result = reconstructor.reconstruct_from_csv("lightcurve.csv")

# All training metrics, spectrum evolution, and final results automatically logged to wandb
```

### Convergence Tracking

The module tracks spectrum evolution during training for convergence analysis:

```python
config = SEDConfig(
    epochs=3000,
    wandb_track_convergence=True,  # Enable L1/L2 change tracking
    use_ema=True,  # Enable EMA smoothing
    ema_decay=0.99,  # High trust in history for smooth results
)

# Convergence metrics automatically logged:
# - L1/L2 changes between consecutive spectra
# - Relative changes normalized by spectrum magnitude
# - EMA distance from current spectrum
# - Stopping criteria recommendations
```

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

print(f"Chi-squared per observation: {quality.chi_squared_per_obs:.3f}")
```

### Advanced Configuration

```python
config = SEDConfig(
    # Deep Image Prior architecture
    dip_filters=64,
    dip_depth=4,
    dip_noise_jitter_initial_ratio=0.3,  # Enable input noise jitter
    dip_noise_jitter_min_ratio=0.1,      # Linear decay of jitter

    # Optimization parameters
    epochs=5000,
    learning_rate=0.0005,
    learning_rate_scheduler_type='cosine_warmup',
    learning_rate_warmup_epochs=200,

    # Regularization
    regularization_weight=2.0,
    cwt_scales=[0.5, 1.0, 2.0, 4.0],

    # Quality control with enhanced sigma clipping
    sigma_threshold=5.0,
    enable_sigma_clip=True,
    sigma_clip_sigma=3.5,
    sigma_clip_window=25,
    sigma_clip_max_iterations=15,

    # EMA smoothing
    use_ema=True,
    ema_decay=0.995,
    ema_start_epoch=200,

    # Ensemble for uncertainty quantification
    ensemble_size=7,
    ensemble_random_seed=12345,

    # Device selection
    device='cuda'  # Force CUDA GPU
)

reconstructor = SEDReconstructor(config)
ensemble_result = reconstructor.reconstruct_from_csv("lightcurve.csv")
ensemble_result.save_all("ensemble_output/")
```

### Ensemble Reconstruction Analysis

```python
from spxquery.sed import SEDConfig, SEDReconstructor
import numpy as np

# Configure ensemble reconstruction
config = SEDConfig(
    epochs=3000,
    ensemble_size=5,
    ensemble_random_seed=42,
    use_ema=True,
)

reconstructor = SEDReconstructor(config)
ensemble_result = reconstructor.reconstruct_from_csv("lightcurve.csv")

# Access ensemble statistics
mean_spectrum = ensemble_result.mean_flux
uncertainty = ensemble_result.std_flux
median_spectrum = ensemble_result.median_flux

# Calculate SNR spectrum
snr_spectrum = mean_spectrum / uncertainty
high_snr_mask = snr_spectrum > 5.0

print(f"Mean reconstruction quality across ensemble:")
for i, member in enumerate(ensemble_result.member_results):
    chi2 = member.validation_metrics.chi_squared_per_obs
    print(f"  Member {i+1}: χ²/M = {chi2:.3f}")

print(f"Ensemble mean χ²/M = {np.mean([r.validation_metrics.chi_squared_per_obs for r in ensemble_result.member_results]):.3f} ± {np.std([r.validation_metrics.chi_squared_per_obs for r in ensemble_result.member_results]):.3f}")
```

## Key Classes and Functions

### Core Classes

- **SEDConfig**: Comprehensive configuration dataclass with 40+ parameters for all reconstruction aspects
- **SEDReconstructor**: Main orchestrator class that coordinates the entire reconstruction pipeline
- **SEDReconstructionResult**: Container for single reconstruction results with validation metrics
- **EnsembleResult**: Container for ensemble reconstruction results with uncertainty quantification
- **BandData**: Data container for individual SPHEREx detector band measurements
- **GlobalSpectralData**: Sparse matrix container for global reconstruction dataset
- **ValidationMetrics**: Comprehensive quality assessment including chi-squared and residual analysis
- **EMATracker**: Exponential Moving Average implementation for spectrum smoothing

### Key Functions

- **export_default_sed_config()**: Export configuration template for customization
- **apply_rolling_mad_sigma_clip_single_band()**: Rolling MAD-based outlier removal
- **SpectralEvaluator**: Comprehensive reconstruction quality assessment with physically meaningful metrics
- **solve_global_reconstruction()**: PyTorch Deep Image Prior optimization engine

### Automatic Detection

The module automatically detects and handles:

- **Ensemble vs Single**: Based on `config.ensemble_size` parameter
- **Device Selection**: Automatic MPS/CUDA/CPU detection with fallback handling
- **Sparse Matrix Optimization**: CPU fallback for MPS sparse operation limitations
- **wandb Integration**: Only first ensemble member logged to prevent conflicts

## References

See `SpecRefine.md` for detailed mathematical derivation and implementation details.

## Support

For issues or questions:

- GitHub Issues: [spxquery repository]
- Documentation: [spxquery.readthedocs.io]
