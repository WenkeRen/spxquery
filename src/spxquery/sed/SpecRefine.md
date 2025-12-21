# Spectral Reconstruction from SPHEREx Time-Domain Photometry - PyTorch Deep Image Prior Implementation

This document describes the mathematical framework for refining coarse-grained SPHEREx time-domain photometry into high-resolution spectra using PyTorch-based **Deep Image Prior (DIP)** optimization with **Continuous Wavelet Transform (CWT)** regularization. The DIP implementation provides a unified global reconstruction approach across all SPHEREx detector bands with differentiable neural network optimization.

**Key Technologies**: Python, NumPy, SciPy, Matplotlib, PyTorch, Astropy

## Scientific Background

SPHEREx obtains narrow-band photometry through Linear Variable Filters (LVFs) that provide spectral resolution R~35-130 across 6 wavelength bands (0.75-5.0 μm). While the nominal mission provides ~102 broad bands per source, the deep survey regions in NEP and SEP provide up to 40,000 repeated measurements with randomly distributed narrowband centers across the full wavelength range. This dense sampling enables reconstruction of high-resolution spectra from the time-domain photometry.

### Deep Image Prior Approach

The **Deep Image Prior (DIP)** approach leverages neural network architecture bias as an implicit regularizer:

- **Global reconstruction**: Unified spectrum spanning 0.75-5.0 μm reconstructed simultaneously
- **Neural network flexibility**: U-Net architecture captures complex spectral features adaptively
- **Differentiable optimization**: End-to-end gradient-based optimization with automatic regularization
- **Multi-scale CWT regularization**: Continuous wavelet transform provides natural multi-scale constraints

### SPHEREx Detector Specifications

SPHEREx uses six detector bands to cover the 0.75-5.0 μm spectral range:

| Band | Wavelength Range (μm) | Resolution (R=λ/Δλ) | Notes |
|------|----------------------|---------------------|-------|
| 1    | 0.75 - 1.09          | 41                  | Short-wave infrared |
| 2    | 1.10 - 1.62          | 41                  | Short-wave infrared |
| 3    | 1.63 - 2.41          | 41                  | Short-wave infrared |
| 4    | 2.42 - 3.82          | 35                  | Mid-wave infrared |
| 5    | 3.83 - 4.41          | 110                 | Mid-wave infrared, high resolution |
| 6    | 4.42 - 5.00          | 130                 | Mid-wave infrared, high resolution |

**Key Parameters:**

- **Resolution R**: Defined as λ/Δλ, where Δλ is the effective narrowband width
- **Narrowband width**: For R=41 bands, Δλ ≈ λ/41 ≈ 2-3% of center wavelength
- **Detector array**: 2040×2040 pixels with LVF-based wavelength mapping along y-axis

### Data Structure

The spxquery package extracts single-epoch aperture photometry from SPHEREx level2 images. Each measurement represents a narrowband photometric observation with the following fields:

- **flux**: Photometric flux (μJy)
- **flux_err**: Photometric uncertainty (μJy)
- **wavelength**: Narrowband center wavelength (μm)
- **bandwidth**: Narrowband width (μm)
- **band**: Detector band ID (1-6)
- **flag**: Quality flag (see spxquery documentation)

### Processing Pipeline

The reconstruction pipeline operates in distinct stages:

1. **Data Preprocessing**: Clean data by removing NaNs and poor-quality measurements (based on flags, flux/flux_err thresholds)
2. **Global Dataset Construction**: Aggregate all detector band measurements into unified observation matrix:
   - Frequency-normalized measurement matrix H with proper energy conservation
   - Weight vector w based on measurement uncertainties
   - Global wavelength grid spanning all detector bands
3. **Deep Image Prior Optimization**: Single global optimization across all bands:
   - U-Net neural network architecture with fixed input noise
   - CWT regularization for multi-scale spectral constraints
   - PyTorch automatic differentiation and optimization
   - Quality assessment with chi-squared metrics

## Mathematical Modeling

### Physical Intuition

Astronomical spectra are inherently **multi-scale signals**:

1. **Continuum**: Very low-frequency signal, smooth variations across wavelengths
2. **Emission/Absorption Lines**: Medium-frequency features with variable widths
3. **Observational Noise**: High-frequency component to be suppressed

Traditional optimization methods struggle with the ill-posed nature of reconstructing high-resolution spectra from sparse photometry. The **Deep Image Prior** approach leverages neural network architecture bias as an implicit regularizer, while **Continuous Wavelet Transform** provides explicit multi-scale constraints on the solution.

### Core Optimization Problem

$$
\min_\theta \left( \underbrace{||W(y - \mathcal{P}(G_\theta(z)))||_2^2}_{\text{Data Fidelity}} + \underbrace{R(G_\theta(z))}_{\text{CWT Regularization}} \right)
$$

**Notation:**

- $G_\theta(z)$: Deep Image Prior neural network with weights $\theta$ and fixed input noise $z$
- $\mathcal{P}$: Projection operator mapping global spectrum to observed photometry
- $x = G_\theta(z) \in \mathbb{R}^N$: Reconstructed high-resolution spectrum (output)
- $y \in \mathbb{R}^M$: Observed narrowband photometry (known)
- $H \in \mathbb{R}^{M \times N}$: Measurement matrix with frequency step normalization
- $W \in \mathbb{R}^{M \times M}$: Weight diagonal matrix, $W_{ii} = 1/\sigma_i$
- $R(x)$: CWT regularization promoting sparsity in wavelet domain
- $z \in \mathbb{R}^N$: Fixed random noise vector (network input)

### Frequency-Consistent Measurement Matrix

The measurement matrix incorporates **frequency step normalization** for energy conservation:

$$H_{ij} = T_i(\lambda_j) \times \frac{\Delta\nu_j}{\sum_{k \in W_i} \Delta\nu_k}$$

where $\Delta\nu_j$ is the frequency step at wavelength $\lambda_j$ and $W_i$ is the wavelength window for measurement $i$. This ensures:

- Rows sum to 1.0 (energy conservation)
- Proper handling of non-uniform frequency sampling
- Consistent units between input photometry (μJy) and output spectrum (μJy)

### Deep Image Prior Architecture

The implementation uses a **1D U-Net architecture** for spectral generation:

- **Encoder-Decoder Structure**: Multi-scale feature extraction with skip connections
- **Downsampling**: Max pooling layers reduce spatial dimensions while increasing feature channels
- **Skip Connections**: Preserve high-frequency details during decoding
- **Reflective Padding**: Maintains boundary conditions without artifacts
- **Configurable Depth**: Default 3 downsampling stages with 32 base filters

**Network Architecture:**

- Input: Fixed random noise vector $z$ with standard deviation 0.1
- Output: Reconstructed spectrum $x = G_\theta(z)$
- Optimization: Adam optimizer with learning rate scheduling

### Continuous Wavelet Transform Regularization

The implementation uses **Gaussian (Mexican Hat) wavelets** for multi-scale regularization:

- **Differentiable**: Implemented as fixed Conv1d layers with gradients disabled
- **Multiple Scales**: Default scales [1.0, 2.0, 3.0] capture different frequency bands
- **L1 Sparsity**: Promotes sparse wavelet coefficients across all scales
- **Reflective Padding**: Maintains proper boundary conditions for convolution

**Wavelet Properties:**

- Mexican Hat wavelet: $\psi(t) = \frac{2}{\sqrt{3\sigma}\pi^{1/4}}(1 - \frac{t^2}{\sigma^2})e^{-t^2/(2\sigma^2)}$
- Zero mean property enforced for discrete kernels
- Scales adapt to spectral resolution (3000 bins across 0.75-5.0 μm)

## Implementation Overview

The PyTorch-based spectral reconstruction has been fully implemented in the spxquery.sed module with the following key components:

### Core Modules

- **`config.py`**: SEDConfig class with PyTorch DIP and CWT regularization parameters
- **`matrices.py`**: Frequency-normalized measurement matrix construction for global reconstruction
- **`data_structures.py`**: GlobalSpectralData container for sparse H matrix and observations
- **`solver_torch.py`**: PyTorch Deep Image Prior optimization with U-Net architecture
- **`regularization.py`**: Continuous Wavelet Transform using Mexican Hat wavelets
- **`reconstruction.py`**: Main orchestration with global dataset construction and validation
- **`data_loader.py`**: SPHEREx photometry loading with quality control and sigma clipping
- **`validation.py`**: Quality assessment metrics including chi-squared and residual analysis

### Dependencies

```bash
pip install numpy scipy matplotlib torch astropy pandas
```

**Key Libraries:**

- **PyTorch**: Neural network optimization and automatic differentiation
- **NumPy**: Numerical operations and array handling
- **SciPy**: Sparse matrix construction and scientific computing
- **Astropy**: Physical constants and astronomical calculations
- **Matplotlib**: Publication-quality visualization

## Key Implementation Details

### Global Dataset Construction

The core dataset building function aggregates all detector bands into a unified global reconstruction problem:

```python
def build_global_observation_data(
    all_band_data: Dict[str, BandData],
    config: SEDConfig
) -> GlobalSpectralData:
    """
    Build global dataset for reconstruction across all SPHEREx bands.

    Returns:
        GlobalSpectralData with sparse H matrix, observations y, weights w,
        and global wavelength grid spanning 0.75-5.0 μm
    """
```

**Key features:**

- **Unified wavelength grid**: Single high-resolution grid (default 3000 bins) across full SPHEREx range
- **Frequency normalization**: Proper energy conservation with frequency step weighting
- **Sparse representation**: Efficient COO sparse matrix format for PyTorch compatibility
- **Quality control**: Automatic weight computation based on measurement uncertainties

### PyTorch Deep Image Prior Solver

The core optimization uses neural network architecture as implicit regularizer:

```python
def solve_global_reconstruction(
    data: GlobalSpectralData,
    config: SEDConfig
) -> Tuple[torch.Tensor, str, float]:
    """
    Solve global reconstruction using Deep Image Prior optimization.

    Returns:
        result_spectrum: Reconstructed spectrum (global_resolution)
        solver_status: Optimization status message
        solver_time: Computation time in seconds
    """
```

**Key components:**

- **SpectralUNet**: 1D U-Net with encoder-decoder architecture and skip connections
- **Adam optimizer**: Gradient-based optimization with configurable learning rate
- **Learning rate scheduling**: Cosine annealing with optional warmup phase
- **Adaptive normalization**: MAD-based robust scaling for data fidelity term
- **Multi-device support**: Automatic MPS/CPU fallback for sparse operations

### Continuous Wavelet Transform Regularization

The CWT implementation provides multi-scale spectral constraints:

```python
class GaussianCWT(nn.Module):
    """
    Differentiable Continuous Wavelet Transform using Mexican Hat wavelets.

    Implements fixed Conv1d layers with disabled gradients for regularization.
    """
```

**Regularization features:**

- **Multiple scales**: Configurable scale list for different frequency bands
- **Mexican Hat wavelets**: Optimal for spike-like spectral features (emission/absorption lines)
- **L1 sparsity**: Promotes sparse wavelet coefficients across all scales
- **Differentiable**: Integrated into PyTorch computational graph

### Data Loading and Quality Control

The data loading system handles SPHEREx time-domain photometry with robust quality control:

```python
@dataclass
class BandData:
    """Prepared photometry data for one detector band."""
    band: str
    flux: np.ndarray
    flux_err: np.ndarray
    wavelength_center: np.ndarray
    bandwidth: np.ndarray
    weights: np.ndarray
    n_measurements: int
```

**Quality control features:**

- **Sigma thresholding**: Minimum SNR filtering (default: 3.0)
- **Flag filtering**: Configurable bad pixel flag rejection
- **Sigma clipping**: MAD-based robust outlier removal with rolling windows
- **Automatic band detection**: Processes all available SPHEREx detector bands

## Usage Example

The complete reconstruction pipeline is available through a simple interface:

```python
from spxquery.sed import SEDReconstructor, SEDConfig

# Configure PyTorch DIP reconstruction
config = SEDConfig(
    epochs=3000,                  # Number of optimization iterations
    regularization_weight=1.0,    # CWT regularization strength
    cwt_scales=[1.0, 2.0, 3.0],   # Multi-scale wavelet constraints
    learning_rate=0.001,          # Adam optimizer learning rate
    global_resolution=3000,       # Output spectrum resolution
    device='mps',                 # Use Apple Silicon GPU
    sigma_threshold=3.0,          # Minimum SNR for input data
)

# Run reconstruction
reconstructor = SEDReconstructor(config)
result = reconstructor.reconstruct_from_csv('lightcurve.csv')

# Save results
result.save_all('output/')
```

**Output files:**

- `sed_reconstruction.csv`: Global reconstructed spectrum (wavelength, flux)
- `sed_metadata.yaml`: Complete metadata with DIP parameters and quality metrics
- Quality assessment plots and validation reports

### Configuration Options

The system supports extensive customization through the SEDConfig class:

**Deep Image Prior parameters:**

- `dip_filters`: Number of filters in U-Net architecture (default: 32)
- `dip_depth`: Depth of U-Net network (default: 3)
- `dip_noise_std`: Standard deviation of input noise (default: 0.1)

**Optimization parameters:**

- `learning_rate`: Adam optimizer learning rate (default: 0.001)
- `epochs`: Number of optimization iterations (default: 3000)
- `learning_rate_scheduler_type`: 'none', 'cosine', 'cosine_warmup', or 'warmup'

**Regularization parameters:**

- `regularization_weight`: CWT regularization strength (default: 1.0)
- `cwt_scales`: List of wavelet scales (default: [1.0, 2.0, 3.0])

**Quality control:**

- `sigma_threshold`: Minimum SNR for photometry (default: 3.0)
- `bad_flags`: Pixel flags to reject
- `enable_sigma_clip`: MAD-based outlier removal

### Code Structure

**Core implementation modules:**

- `config.py`: SEDConfig class with PyTorch DIP and CWT regularization parameters
- `matrices.py`: Frequency-normalized measurement matrix construction for global reconstruction
- `data_structures.py`: GlobalSpectralData container for sparse H matrix and observations
- `solver_torch.py`: PyTorch Deep Image Prior optimization with U-Net architecture
- `regularization.py`: Continuous Wavelet Transform using Mexican Hat wavelets
- `reconstruction.py`: Main orchestration with global dataset construction and validation
- `data_loader.py`: SPHEREx photometry loading with quality control and sigma clipping
- `validation.py`: Quality assessment metrics including chi-squared and residual analysis

The system is fully implemented and tested, ready for production use with SPHEREx deep survey data.
