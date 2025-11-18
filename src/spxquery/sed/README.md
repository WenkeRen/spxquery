# SED Reconstruction Module

High-resolution spectral reconstruction from SPHEREx narrow-band photometry using convex optimization.

## Overview

This module reconstructs high-resolution spectra from SPHEREx's randomly-sampled narrow-band measurements using regularized least-squares optimization. The method is particularly powerful for the NEP/SEP deep fields where individual sources have ~40,000 measurements spanning 0.75-5.0 microns.

## Mathematical Formulation

The reconstruction solves:

$$
min_x ( ||w(y - Hx)||_2^2 + \sum_{k=0}^{J} \lambda_k ||\Psi_k x||_1 )
$$

where:

- **Data fidelity**: Weighted chi-squared (L2 norm)
- **SWT regularization**: Multi-scale sparsity prior using Stationary Wavelet Transform
- **4-group system**: Different regularization weights for continuum, large features, emission lines, and noise

## Quick Start

### Python API

```python
from spxquery.sed import SEDConfig, SEDReconstructor

# Basic usage with default settings
config = SEDConfig()
reconstructor = SEDReconstructor(config)
result = reconstructor.reconstruct_from_csv("lightcurve.csv")
result.save_all("output/")

# With auto-tuning
config = SEDConfig(auto_tune=True)
reconstructor = SEDReconstructor(config)
result = reconstructor.reconstruct_from_csv("lightcurve.csv")

# Custom hyperparameters
config = SEDConfig(lambda_continuum=0.05, lambda_main_features=20.0, resolution_samples=2040)
reconstructor = SEDReconstructor(config)
result = reconstructor.reconstruct_from_csv("lightcurve.csv")
```

## Module Structure

```
sed/
├── __init__.py          - Module exports
├── config.py            - SEDConfig dataclass
├── data_loader.py       - CSV loading and quality filtering
├── matrices.py          - H matrix and SWT operator construction
├── solver.py            - CVXPY optimization
├── validation.py        - Residual analysis and diagnostics
├── tuning.py            - Grid search hyperparameter tuning
├── reconstruction.py    - Main SEDReconstructor orchestrator
├── plots.py             - Diagnostic visualization
└── README.md            - This file
```

## Configuration Parameters

### Key Parameters

- `lambda_continuum` (float): Approximation coefficients regularization (default: 0.1)
- `lambda_low_features` (float): Coarse detail regularization (default: 1.0)
- `lambda_main_features` (float): Medium detail regularization (default: 5.0)
- `lambda_noise` (float): Fine detail regularization (default: 100.0)
- `resolution_samples` (int): Output wavelength bins (default: 1020)
- `auto_tune` (bool): Enable grid search tuning (default: False)
- `sigma_threshold` (float): Minimum SNR for quality filtering (default: 5.0)
- `wavelet_family` (str): Wavelet basis function (default: 'sym6')

### Export Configuration Template

```python
from spxquery.sed import export_default_sed_config

config_path = export_default_sed_config("output/")
# Edit output/sed_config.yaml to customize parameters
```

## Outputs

### CSV File: `sed_reconstruction.csv`

Columns:

- `wavelength` (microns): Wavelength grid
- `flux` (microJansky): Reconstructed flux density
- `band` (str): Source band (D1-D6)

### YAML File: `sed_metadata.yaml`

Contains:

- Source information (name, RA, Dec)
- Hyperparameters (lambda_vector per band)
- Wavelet decomposition information
- Quality metrics (chi-squared reduced)

### Diagnostic Plots (optional)

- `sed_diagnostic_summary.png`: Multi-panel overview
- `sed_band_comparison.png`: Individual band spectra

## Workflow

1. **Data Loading**: Load lightcurve CSV from SPXQuery processing pipeline
2. **Quality Filtering**: Apply SNR threshold and bad pixel flags
3. **Band Separation**: Split data into 6 detector bands (D1-D6)
4. **Matrix Construction**: Build measurement matrix H and SWT operators
5. **Optimization**: Solve convex problem (manual or auto-tuned hyperparameters)
6. **Validation**: Compute chi-squared, residuals, quality metrics
7. **Output**: Save CSV, YAML, and diagnostic plots

## Auto-Tuning

When `auto_tune=True`:

1. Split data into 80% training, 20% validation
2. Grid search over 4D hyperparameter space (continuum × low_features × main_features × noise)
3. Select hyperparameters minimizing validation error
4. Reconstruct on full dataset with optimal parameters

Default grids:

- `lambda_continuum_grid`: [0.01, 0.1, 1.0]
- `lambda_low_features_grid`: [0.1, 1.0, 10.0]
- `lambda_main_features_grid`: [1.0, 10.0, 100.0]
- `lambda_noise_grid`: [10.0, 100.0, 1000.0]

## Quality Metrics

### Chi-Squared

- `chi_squared`: Sum of squared weighted residuals
- `chi_squared_reduced`: Chi-squared / degrees of freedom
- Ideal: chi_squared_reduced ≈ 1.0
- > 2.0: Poor fit or underestimated errors
- < 0.5: Overfitting or overestimated errors

### Residuals

- Raw residuals: y - H@x
- Weighted residuals: w*(y - H@x)
- Weighted residuals should be approximately N(0, 1) if model is correct

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

- numpy: Numerical arrays
- scipy: Sparse matrices
- pandas: CSV I/O
- cvxpy: Convex optimization
- matplotlib: Visualization
- pyyaml: Configuration files

## Performance Notes

- **Memory**: Sparse matrices keep memory usage reasonable (< 1 GB for typical sources)
- **Speed**: Single band reconstruction ~5-60 seconds depending on solver
- **Auto-tuning**: Grid search multiplies time by number of combinations (typically 16)
- **Solver**: CLARABEL (default, improved ECOS) is fast and stable; SCS for large problems

## Future Extensions

Potential enhancements:

- Gaussian filter profiles (currently only boxcar)
- Cross-validation with K-folds
- Adaptive grid search
- Parallel band processing
- GPU acceleration for large datasets

## References

See `SpecRefine.md` for detailed mathematical derivation and implementation plan.

## Support

For issues or questions:

- GitHub Issues: [spxquery repository]
- Documentation: [spxquery.readthedocs.io]
