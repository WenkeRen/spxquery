# Execution Plan: Deep Image Prior Reconstruction (PyTorch)

## 1. Objective
Implement a **Deep Image Prior (DIP)** approach for spectral reconstruction. Unlike standard gradient descent on pixel values, this method optimizes the **weights of a generative neural network (1D U-Net)** to map fixed random noise to the observed spectrum. This architectural prior inherently enforces local smoothness and correlation, eliminating Gibbs phenomena.

**Key Strategy Change**: Instead of solving the inverse problem $y = Hx$ using sparse matrix multiplication, we will project observations onto the spectral grid *a priori*. The problem is reformulated as fitting the network to a massive dataset of individual "pixel observations" derived from the original photometry. This structure allows for massive parallelism and avoids sparse matrix overhead.

## 2. Architecture Changes

### A. Configuration (`src/spxquery/sed/config.py`)
Update `SEDConfig` to support the new solver backend and global parameters.
- **New Fields**:
  - `solver_type`: `str` = "torch" (Options: "cvxpy", "torch")
  - `wavelength_range`: `Tuple[float, float]` = (0.75, 5.0)
  - `global_resolution`: `int` = 3000
  - `device`: `str` = "mps" (Options: "cpu", "cuda", "mps")
  - **Optimization**:
    - `optimizer`: `str` = "Adam"
    - `learning_rate`: `float` = 0.001
    - `epochs`: `int` = 3000
  - **Deep Prior Architecture**:
    - `dip_noise_std`: `float` = 0.1
    - `dip_filters`: `int` = 32
    - `dip_depth`: `int` = 3
  - **Regularization**:
    - `regularization_weight`: `float` = 1.0 (Weight for CWT sparsity loss)
    - `cwt_scales`: `List[float]` = [1.0, 2.0, 3.0]

### B. Data Structures (`src/spxquery/sed/data_structures.py` - New File)
Create a container for the flattened pixel-level dataset.
- **`PixelObservationData`**:
  - `pixel_indices`: LongTensor of shape `(N_total_samples,)`. Indices into the 3000-bin spectral grid.
  - `pixel_fluxes`: FloatTensor of shape `(N_total_samples,)`. The target flux density for each pixel sample.
  - `pixel_weights`: FloatTensor of shape `(N_total_samples,)`. The weight for each pixel sample.
  - `global_wavelength_grid`: 3000-bin wavelength grid (PyTorch Tensor).

### C. Data Preparation & Matrix Conversion (`src/spxquery/sed/matrices.py`)
Refactor to build the "Pixel Unit" dataset instead of a sparse matrix.
- **`build_pixel_observation_dataset(all_band_data, config) -> PixelObservationData`**:
  1. **Global Grid**: Generate `wavelength_grid` (3000 bins, 0.75-5.0µm).
  2. **Decomposition**:
     - Iterate through all observations $i$ in all bands.
     - For each observation, identify the set of spectral grid pixels $P_i$ it covers (where filter response > 0).
     - **Flux Distribution**: Assign the observed flux density $y_i$ to each pixel $j \in P_i$ as the target $f_{ij} = y_i$ (assuming flux density is roughly constant across the narrowband).
     - **Weight Distribution**: Distribute the observation weight $w_i$ to pixels. To conserve the total constraining power, normalize by coverage: $w_{ij} \approx w_i / |P_i|$.
     - Collect tuples `(j, f_{ij}, w_{ij})` into a massive list.
  3. **Tensor Construction**:
     - Convert lists to PyTorch tensors: `pixel_indices`, `pixel_fluxes`, `pixel_weights`.
     - Move to `config.device`.

### D. Regularization Module (`src/spxquery/sed/regularization.py` - New File)
Implement differentiable CWT for additional constraints.
- **`GaussianCWT(nn.Module)`**:
  - Implemented via `torch.nn.Conv1d` with fixed weights (discretized Mexican Hat / Ricker wavelet).
  - Use `padding_mode='reflect'` to avoid boundary artifacts.

### E. Deep Image Prior Solver (`src/spxquery/sed/solver_torch.py` - New File)
The optimization loop now resembles a regression training loop.
- **`SpectralUNet(nn.Module)`**:
  - 1D U-Net architecture generating the full spectrum ($N=3000$).
  - Output passes through `Softplus` to enforce non-negativity.
- **`SpectralModel(nn.Module)`**:
  - Wraps `SpectralUNet` with fixed input noise `z`.
- **`solve_global_reconstruction(...)`**:
  - Initialize model and optimizer.
  - **Optimization Loop**:
    - `x_spectrum = model()` (Generate 3000-point spectrum).
    - **Gather**: Extract predicted values for all pixel samples:
      `x_pred_samples = x_spectrum[pixel_indices]`
    - **Data Loss**: Calculate weighted MSE directly on samples:
      `Loss_data = torch.sum(pixel_weights * (x_pred_samples - pixel_fluxes)**2)`
    - **Reg Loss**: `Loss_reg = RegWeight * L1(CWT(x_spectrum))`
    - Backward & Step.
  - Return `x_spectrum`.

### F. Orchestrator Update (`src/spxquery/sed/reconstruction.py`)
- Update `reconstruct_from_csv` to check `solver_type`.
- If "torch":
  - Load bands.
  - Call `build_pixel_observation_dataset`.
  - Run `solve_global_reconstruction`.
  - Format output.

## 3. Detailed Implementation Steps

### Step 1: Setup
- Update `SEDConfig` and dependencies.

### Step 2: Data Processing
- Implement `build_pixel_observation_dataset` in `matrices.py`.
- This replaces the complex sparse matrix logic with loop-based expansion of observations into pixel samples.

### Step 3: Model Implementation
- Implement `SpectralUNet` in `solver_torch.py`.
- Implement `GaussianCWT` in `regularization.py`.

### Step 4: Training Logic
- Implement the training loop using the index-select method (`x[indices]`) which is extremely fast on GPU/MPS.

### Step 5: Integration
- Wire up `SEDReconstructor`.

## 4. Advantages of Revised Plan
- **Efficiency**: Avoids sparse matrix multiplication completely. The forward pass becomes a simple indexing operation (`gather`), which is highly optimized on GPUs.
- **Simplicity**: The "projection" logic decouples the observation geometry from the solver. The solver just sees a weighted regression problem on the spectral grid.
- **Parallelism**: The dataset construction is trivially parallelizable, and the optimization step is standard deep learning training.