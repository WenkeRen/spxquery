# -*- coding: utf-8 -*-

"""
# Time       : 2025-12-30 14:07
# Author     : Wenke Ren
# File       : sed_rct_benchmark.py
# Version    : Python 3.13
# Description: This code is a parallel benchmark script for SED reconstruction using data.
"""

# %% Setup and Imports
# Standard libraries
# Cell 1: Force complete module reload
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add benchmark tools to path (not part of spxquery package)
sys.path.insert(0, str(Path.cwd() / "tools"))

# Import SED reconstruction modules from spxquery

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import SED reconstruction modules from spxquery
# Import benchmark simulation tools
from tools.models import SPHEREX_BANDS, SpectralModelConfig, generate_spectral_model
from tools.simobs import convert_model_flux_to_uJy, simulate_spherex_observations

from spxquery.sed import SEDConfig, SEDReconstructor
from spxquery.sed.plots import plot_sed_reconstruction_dashboard

# Check device availability
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configure matplotlib
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (12, 5)

print("All imports successful!")
print(f"SPHEREx has {len(SPHEREX_BANDS)} bands: {list(SPHEREX_BANDS.keys())}")

# ==========================================================================
# %% Generate Synthetic Spectral Model
# Create model configuration
print("Creating spectral model configuration...")
model_config = SpectralModelConfig(
    # Continuum parameters
    magnitude=18,  # AB magnitude at reference wavelength
    slope=0.5,  # Power-law slope (F_lambda ~ lambda^slope)
    # Primary emission line
    line1_center=2.0,  # microns (e.g., Br gamma at 2.166 microns)
    line1_ew=20.0,  # Equivalent width (Angstroms)
    line1_fwhm_vel=1500.0,  # FWHM velocity (km/s)
    # Secondary emission line
    line2_flux_ratio=0.0,  # Flux ratio to line 1
    line2_separation_fwhm=3.0,  # Separation from line 1 in units of FWHM
    line2_same_fwhm=True,  # Use same FWHM as line 1
    # Wavelength grid
    band_id="D3",  # Use band D3 for reference
    oversample_factor=500,  # Oversample factor for model generation
)

# Generate spectrum
wavelength_model, flux_model = generate_spectral_model(model_config)
print("Generated model spectrum:")
print(f"  Wavelength range: {wavelength_model.min():.3f} - {wavelength_model.max():.3f} microns")
print(f"  Number of pixels: {len(wavelength_model)}")
print(f"  Flux range: {flux_model.min():.2e} - {flux_model.max():.2e} erg/s/cm^2/Angstrom")

# Prepare for simulation (stack into format expected by simobs)
model_spectrum = np.vstack([wavelength_model, flux_model])
print(f"\nModel spectrum ready for simulation (shape: {model_spectrum.shape})")

# ==========================================================================
# %% Simulated SPHEREx Observation

# Configuration for simulation
N_VISIT = 100  # Number of visit to this band
FILTER_PROFILE = "boxcar"  # or "gaussian"
RANDOM_SEED = 42  # For reproducibility
BG_SURFACE_BRIGHTNESS = 0.1  # MJy/sr

print(f"Simulating SPHEREx observations with {N_VISIT} visits per band...")
print(f"Filter profile: {FILTER_PROFILE}")
print()

# Store all band data
all_band_data = {}

# Simulate observations for a single band
band_id = model_config.band_id
band_info = SPHEREX_BANDS[band_id]
print(f"Processing {band_id} ({band_info['wave_min']:.2f}-{band_info['wave_max']:.2f} microns)...")

# Create wavelength grid for this band
wl_min = band_info["wave_min"]
wl_max = band_info["wave_max"]

# Evenly spaced wavelength centers
np.random.seed(RANDOM_SEED)
N_OBS_PER_BAND = int((wl_max - wl_min) / ((wl_max + wl_min) / 2) * band_info["R"])
wl_centers = np.random.uniform(wl_min, wl_max, N_VISIT * N_OBS_PER_BAND)

# Bandwidths (approximate based on resolving power)
bandwidths = wl_centers / band_info["R"]

# Simulate observations
band_data = simulate_spherex_observations(
    model_spectrum=model_spectrum,
    wavelength_centers=wl_centers,
    bandwidths=bandwidths,
    band_id=band_id,
    filter_profile=FILTER_PROFILE,
    sigma_det=None,  # Auto-load from calibration files
    abs_gain=None,  # Auto-load from calibration files
    num_pixels=5,  # Aperture size
    bg_surface_brightness=BG_SURFACE_BRIGHTNESS,
    T_int=113.58,  # Integration time (s)
    N_reads=77,
    random_seed=RANDOM_SEED + hash(band_id) % 1000,  # Different seed per band
)

all_band_data[band_id] = band_data
print(f"  Generated {band_data.n_measurements} observations")
print(f"    Flux range: {band_data.flux.min():.2f} - {band_data.flux.max():.2f} uJy")
print(f"    Error range: {band_data.flux_error.min():.2f} - {band_data.flux_error.max():.2f} uJy")

print(f"\nSimulation complete! Generated data for {len(all_band_data)} bands.")
print(f"\nTotal measurements: {sum(bd.n_measurements for bd in all_band_data.values())}")

# ==========================================================================
# %% SED Reconstruction (iteratively)


def sedconfig_list(
    cwt_scales_list=[[1.0, 2.0, 4.0, 8.0], [1.0, 3.0, 5.0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0]],
    sensitive_factor_list=[2.0, 3.0, 4.0],
    noise_std_list=[0.1, 0.3, 0.5],
):
    config_list = []
    for sensitive_factor in sensitive_factor_list:
        for cwt_scales in cwt_scales_list:
            for noise_std in noise_std_list:
                config = SEDConfig(
                    device=device,
                    # Global grid settings
                    wavelength_range=(wl_min, wl_max),
                    global_resolution=2000,  # High-resolution output
                    ensemble_size=8,  # Number of ensemble members
                    ensemble_random_seed=3407,  # Base seed for reproducible ensembles
                    ensemble_n_workers=8,  # Number of parallel workers for ensemble processing
                    ensemble_perturb_observations=True,  # Perturb observations for each ensemble member
                    # Optimization parameters
                    epochs=10000,  # Number of iterations
                    learning_rate=0.001,  # Learning rate for Adam
                    learning_rate_min_factor=0.05,  # Minimum LR factor for scheduling
                    learning_rate_scheduler_type="cosine_warmup",  # Learning rate scheduler type
                    learning_rate_warmup_epochs=300,  # Warmup epochs for scheduler
                    # Settings for EMA smoothing
                    use_ema=True,
                    ema_decay=0.99,
                    ema_start_epoch=2000,
                    # Deep Image Prior architecture
                    dip_filters=32,  # Base filters in U-Net
                    dip_depth=3,  # Depth of U-Net
                    dip_noise_std=noise_std,  # Input noise standard deviation
                    dip_noise_jitter_initial_ratio=0.3,  # Initial jitter ratio
                    dip_noise_jitter_min_ratio=None,  # No minimum jitter
                    # Regularization
                    regularization_weight=1e-3,
                    cwt_scales=cwt_scales,
                    # Regularization method and adaptive parameters
                    regularization_method="log",  # Options: "absolute", "log", "cauchy"
                    reg_sensitivity_factor=sensitive_factor,  # k where epsilon/gamma = k * sigma_noise (log, cauchy only)
                    reg_warmup_floor=0.1,  # Loose constraint during warmup (log, cauchy only)
                    reg_normal_floor=1e-4,  # Tight constraint after warmup (log, cauchy only)
                    # Quality control
                    sigma_threshold=3.0,
                    enable_sigma_clip=True,
                    # Filter profile (must match simulation)
                    filter_profile=FILTER_PROFILE,
                )
                config_list.append(config)
    return config_list


# ==========================================================================
# %% Benchmark function for a single configuration
def benchmark_single(config_recon: SEDConfig, output_dir: Path, all_band_data: dict):
    # Initialize reconstructor
    reconstructor = SEDReconstructor(config_recon)

    # Run reconstruction from BandData objects
    print("\nStarting Deep Image Prior optimization...")
    print("This may take a few minutes depending on your hardware...")
    print()

    result = reconstructor.reconstruct_from_data(
        band_data_dict=all_band_data,
        metadata={
            "benchmark_test": True,
            "model_config": str(model_config),
            "n_obs_per_band": N_OBS_PER_BAND,
            "filter_profile": FILTER_PROFILE,
        },
    )

    flux_truth_uJy = convert_model_flux_to_uJy(wavelength_model, flux_model)
    model_spectrum_uJy = np.vstack([wavelength_model, flux_truth_uJy])
    plots_dir = (
        output_dir
        / f"plots_noi{config_recon.dip_noise_std}_scales{len(config_recon.cwt_scales)}_sen{config_recon.reg_sensitivity_factor}.png"
    )
    plot_sed_reconstruction_dashboard(result, plots_dir, true_spectrum=model_spectrum_uJy)
    return None


# ==========================================================================
# %% Run Bench in parallel
from typing import Tuple

from tqdm import tqdm


def _benchmark_worker(task_args: Tuple[int, SEDConfig, Path, dict]):
    idx, config, base_dir, band_data = task_args
    base_dir.mkdir(parents=True, exist_ok=True)
    benchmark_single(config, base_dir, band_data)
    return idx


if __name__ == "__main__":
    configs = sedconfig_list()
    output_base_dir = Path(project_root) / "sed_benchmark_outputs"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    task_args = [(idx, config, output_base_dir, all_band_data) for idx, config in enumerate(configs, start=1)]

    # Parallel execution (commented out)
    # if len(task_args) > 1:
    #     workers = 1
    #     print(f"Running in parallel with {workers} workers...")
    #     with ProcessPoolExecutor(max_workers=workers) as executor:
    #         list(
    #             tqdm(
    #                 executor.map(_benchmark_worker, task_args),
    #                 total=len(task_args),
    #                 desc="Benchmarking SED configs (Parallel)",
    #             )
    #         )
    # else:
    #     for args in tqdm(task_args, desc="Benchmarking SED configs (Serial)"):
    #         _benchmark_worker(args)

    # Serial execution
    for args in tqdm(task_args, desc="Benchmarking SED configs"):
        _benchmark_worker(args)

    print("Benchmarking complete!")

    print("Benchmarking complete!")
