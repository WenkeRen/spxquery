# SED Reconstruction Module

High-resolution spectral reconstruction from SPHEREx narrow-band photometry using convex optimization.

## Overview

This module reconstructs high-resolution spectra from SPHEREx's randomly-sampled narrow-band measurements using regularized least-squares optimization. The method is particularly powerful for the NEP/SEP deep fields where individual sources have ~40,000 measurements spanning 0.75-5.0 microns.

## Mathematical Formulation

The reconstruction solves:

```
min_x ( ||w(y - Hx)||_2^2 + lambda1*||x||_1 + lambda2*||D2 x||_2^2 )
```

where:
- **Data fidelity**: Weighted chi-squared (L2 norm)
- **L1 regularization**: Sparsity prior for emission lines
- **L2 regularization**: Smoothness prior for continuum

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
config = SEDConfig(auto_tune=True, stitch_bands=True)
reconstructor = SEDReconstructor(config)
result = reconstructor.reconstruct_from_csv("lightcurve.csv")

# Custom hyperparameters
config = SEDConfig(lambda1=0.5, lambda2=50.0, resolution_samples=2040)
reconstructor = SEDReconstructor(config)
result = reconstructor.reconstruct_from_csv("lightcurve.csv")
```

## Module Structure

```
sed/
├── __init__.py          - Module exports
├── config.py            - SEDConfig dataclass
├── data_loader.py       - CSV loading and quality filtering
├── matrices.py          - H matrix and D2 operator construction
├── solver.py            - CVXPY optimization
├── validation.py        - Residual analysis and diagnostics
├── tuning.py            - Grid search hyperparameter tuning
├── stitching.py         - Multi-band normalization and merging
├── reconstruction.py    - Main SEDReconstructor orchestrator
├── plots.py             - Diagnostic visualization
└── README.md            - This file
```

## Configuration Parameters

### Key Parameters

- `lambda1` (float): L1 regularization weight (default: 1.0)
- `lambda2` (float): L2 regularization weight (default: 10.0)
- `resolution_samples` (int): Output wavelength bins (default: 1020)
- `auto_tune` (bool): Enable grid search tuning (default: False)
- `stitch_bands` (bool): Auto-stitch 6 bands (default: True)
- `sigma_threshold` (float): Minimum SNR for quality filtering (default: 5.0)

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
- `band` (str): Source band (D1-D6 or stitched)

### YAML File: `sed_metadata.yaml`

Contains:
- Source information (name, RA, Dec)
- Hyperparameters (lambda1, lambda2 per band)
- Normalization factors (eta_D1...eta_D6)
- Quality metrics (chi-squared reduced)

### Diagnostic Plots (optional)

- `sed_diagnostic_summary.png`: Multi-panel overview
- `sed_band_comparison.png`: Individual band spectra
- `sed_stitched_spectrum.png`: Full stitched SED

## Workflow

1. **Data Loading**: Load lightcurve CSV from SPXQuery processing pipeline
2. **Quality Filtering**: Apply SNR threshold and bad pixel flags
3. **Band Separation**: Split data into 6 detector bands (D1-D6)
4. **Matrix Construction**: Build measurement matrix H, smoothness operator D2
5. **Optimization**: Solve convex problem (manual or auto-tuned hyperparameters)
6. **Validation**: Compute chi-squared, residuals, quality metrics
7. **Stitching**: Normalize and merge bands into continuous spectrum
8. **Output**: Save CSV, YAML, and diagnostic plots

## Auto-Tuning

When `auto_tune=True`:

1. Split data into 80% training, 20% validation
2. Grid search over lambda1_grid x lambda2_grid
3. Select hyperparameters minimizing validation error
4. Reconstruct on full dataset with optimal parameters

Default grids:
- `lambda1_grid`: [0.01, 0.1, 1.0, 10.0]
- `lambda2_grid`: [1.0, 10.0, 100.0, 1000.0]

## Multi-Band Stitching

SPHEREx has 6 detector bands with different flux calibrations. Stitching procedure:

1. Use D1 as reference (eta_D1 = 1.0)
2. For each subsequent band (D2-D6):
   - Find wavelength overlap with previous bands
   - Compute normalization factor (eta) from median flux ratio
   - Apply normalization and append to stitched spectrum

Output includes normalization factors for reproducibility.

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
