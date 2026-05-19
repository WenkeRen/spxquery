# Drizzle3D — SPHEREx 3D Spectral Image Combination

Drizzle3D combines multiple SPHEREx observations into a single data cube (X, Y, λ)
using a decoupled spatial + spectral drizzle algorithm with inverse-variance weighting.

## Interactive Tutorial

A step-by-step Jupyter notebook is available with full Colab support:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WenkeRen/spxquery/blob/main/example/drizzle3d_demo/ngc4395_pipeline.ipynb)

The notebook walks through the complete pipeline on NGC 4395 (Detector D3):
configuration, grid construction, IRSA query, data download, drizzling, and
interactive visualization with mpld3 (Colab) or ipympl (local).

Source: [`example/drizzle3d_demo/ngc4395_pipeline.ipynb`](https://github.com/WenkeRen/spxquery/tree/main/example/drizzle3d_demo)

---

## Pipeline Overview

```
                         Drizzle3DConfig
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
        build_output_wcs  build_z_grid    query_observations
        (spatial WCS)     (λ bin grid)    (IRSA TAP)
              │                │                 │
              │                │                 ▼
              │                │        download_observations
              │                │                 │
              └───────┬────────┘                 │
                      ▼                          │
                  For each input FITS file ◄─────┘
                      │
              ┌───────┴───────┐
              ▼               ▼
     _read_input_fits   _extract_wavelength_maps
     (image/var/flag)   (λ_c, Δλ per pixel)
              │               │
              └───────┬───────┘
                      ▼
           compute_spatial_mapping
           (bilinear XY overlap)
                      │
                      ▼
              drizzle_image (×N files)
              (vectorized Z overlap + np.add.at)
                      │
                      ▼
                DrizzleCube (in-memory)
                      │
                      ▼
                  save_cube (7-HDU FITS)
```

### Pipeline Steps

1. **Configure** — `Drizzle3DConfig` defines the sky region, detector, drizzle
   kernel parameters, and output settings.
2. **Build grids** — A 2-D TAN spatial WCS and a 1-D spectral (Z) bin grid are
   constructed from the config.
3. **Query & download** — IRSA TAP is queried for SPHEREx observations
   intersecting the target region; matching FITS files are downloaded.
4. **Per-image drizzle** — Each input FITS is read, its spatial and spectral
   WCS are extracted, and pixel-to-output mapping is computed. Valid
   contributions are accumulated into a shared `DrizzleCube` using vectorized
   NumPy operations.
5. **Save** — The finalized cube is written as a 7-HDU FITS file.

---

## Quick Start

```python
from spxquery.drizzle3d import Drizzle3DConfig, drizzle

config = Drizzle3DConfig(
    center_ra=186.4536,    # NGC 4395
    center_dec=33.5468,
    width=30.0,            # arcmin
    height=30.0,
    detector=3,            # D3 only (1.66–2.44 μm)
    xy_shrink=0.8,
    output_dir="output",
)

results = drizzle(config)
# → {3: Path("output/drizzle_D3.fits")}
```

### Step-by-Step Usage

For finer control, run each stage independently:

```python
from spxquery.drizzle3d import Drizzle3DConfig
from spxquery.drizzle3d.grid import build_output_wcs
from spxquery.drizzle3d.spectral import build_z_grid
from spxquery.drizzle3d.query import query_observations, download_observations
from spxquery.drizzle3d.pipeline import drizzle_detector

# 1. Config
config = Drizzle3DConfig(
    center_ra=186.4536, center_dec=33.5468,
    width=30.0, height=30.0, detector=3,
    xy_shrink=0.8, output_dir="output",
)

# 2. Grids
output_wcs = build_output_wcs(config)
zgrid = build_z_grid(config.detector, config.z_oversample)

# 3. Query & download
obs_by_det = query_observations(config)
fits_paths = download_observations(
    obs_by_det[3], output_dir="output", skip_existing=True,
)

# 4. Drizzle
outpath = drizzle_detector(fits_paths, config, 3, output_wcs)
```

---

## Configuration

`Drizzle3DConfig` is a validated dataclass that controls every aspect of the pipeline.
It supports YAML round-tripping for reproducibility.

### Key Parameters

| Group | Parameters | Purpose |
|---|---|---|
| Sky region | `center_ra`, `center_dec`, `width`, `height` | Where and how big the output cube is |
| Detector | `detector` (0=all, 1–6=single) | Which SPHEREx detector to process |
| Droplet shrink | `xy_shrink`, `z_shrink` | Input pixel footprint scaling (0, 1] |
| Output grid | `xy_oversample`, `z_oversample`, `z_lambda_edges` | Output resolution control |
| Query | `mjd_range`, `max_images`, `download_workers` | IRSA TAP query constraints |
| Processing | `subtract_zodi`, `static_zodi`, `exclude_flags` | Pre-drizzle data cleaning |
| Accumulation | `ivar_max`, `min_overlap` | Weighting and quality cuts |

### Computed Properties

- `effective_pixscale()` → `BASE_PIXSCALE / xy_oversample` (arcsec/pixel)
- `output_nx()`, `output_ny()` → grid dimensions in pixels
- `spatial_radius_deg()` → half-diagonal for TAP region queries
- `effective_z_shrink()` → `z_shrink` if set, else `xy_shrink`

### YAML Workflow

```python
# Export
config.to_yaml_file("my_config.yaml")

# Edit the YAML in a text editor, then reload
config = Drizzle3DConfig.from_yaml_file("my_config.yaml")
```

### Local Data Mirror

Instead of downloading from IRSA, point `data_mirror` to a local directory that
mirrors the IRSA IBE structure:

```
data_mirror/
├── qr2/
│   └── level2/
│       └── 2025W20_2D/
│           └── .../xxx.fits
└── qr3/
    └── level2/
        └── ...
```

```python
config = Drizzle3DConfig(
    ..., data_mirror="/data/spherex_mirror",
)
```

---

## Algorithm Details

### Spatial Mapping (XY)

`compute_spatial_mapping()` uses **bilinear weight distribution**:

1. Transform all input pixel centers → sky coordinates via `input_wcs`.
2. Project onto the output tangent plane via `output_wcs.world_to_pixel_values()`.
3. For each valid input pixel (within output bounds ± 2-pixel margin), compute
   the 2×2 neighborhood of surrounding output pixels.
4. Bilinear weight: `w = max(1 - |Δx|, 0) × max(1 - |Δy|, 0) × xy_shrink²`.
   Only non-zero weights (`|Δx| < 1`, `|Δy| < 1`) are kept.

Each input pixel produces 1–4 output contributions.

### Spectral Mapping (Z)

`drizzle_image()` computes spectral overlap via a vectorized **top-hat kernel**:

- Top-hat range: `[λ_c − ½Δλ·z_shrink, λ_c + ½Δλ·z_shrink]`
- Overlap length with each Z bin: `max(min(hi, z_edge[j+1]) - max(lo, z_edge[j]), 0)`
- Normalize each row so `Σf_z ≈ 1`

### Accumulation

The `DrizzleCube` maintains per-voxel accumulators:

| Array | Accumulates |
|---|---|
| `flux_weighted` | `Σ w × f_xy × f_z × F_i` |
| `weight_total` | `Σ w × f_xy × f_z` |
| `var_accum` | `Σ (w × f_xy × f_z)² × σ²` |
| `count_map` | Number of contributing input pixels |
| `and_mask` | Bitwise AND of input FLAGS (conservative) |
| `or_mask` | Bitwise OR of input FLAGS (inclusive) |

Final outputs:
- `flux = flux_weighted / weight_total` (inverse-variance weighted mean)
- `variance = var_accum / weight_total²`

Accumulation uses `np.add.at` for correct handling of duplicate indices.

---

## Output Format

`save_cube()` writes a 7-HDU FITS file:

| HDU | Name | Content |
|-----|------|---------|
| 0 | PRIMARY | Metadata headers (detector, N_INPUTS, shrink factors, wavelength range) |
| 1 | SCI | float32 — flux-weighted mean surface brightness [MJy/sr] |
| 2 | VARIANCE | float32 — per-voxel variance [MJy²/sr²] |
| 3 | AND_MASK | uint32 — conservative bitwise AND of input flags |
| 4 | OR_MASK | uint32 — inclusive bitwise OR of input flags |
| 5 | COUNT | uint16 — number of contributing input pixels |
| 6 | WAVELENGTH | BinTableHDU — INDEX, LAMBDA, LAMBDA_MIN, LAMBDA_MAX, DLAMBDA, NU, DNU, R_EFF |

SCI and VARIANCE HDUs include an approximate 3-D WCS (TAN + WAVE) for
compatibility with DS9, CARTA, and other FITS viewers. The exact per-plane
wavelengths are in the WAVELENGTH extension.

### Reading Output

```python
from spxquery.drizzle3d.io import load_cube

data = load_cube("output/drizzle_D3.fits")
sci = data["sci"]         # (n_z, n_y, n_x) in MJy/sr
var = data["variance"]    # (n_z, n_y, n_x) in (MJy/sr)²
lam = data["wavelength"]["LAMBDA"]  # central wavelength per Z bin [μm]
hdr = data["header"]      # PRIMARY header with metadata
```

---

## Spectral Grid Modes

Three modes dispatched by `build_z_grid()`:

| Mode | Trigger | Algorithm |
|---|---|---|
| Native | `z_oversample=1.0`, no custom edges | Uses WL_MIN/WL_MAX from spectral table (17 bins/detector) |
| Oversampled | `z_oversample > 1.0` | Logarithmic spacing at constant R_eff = R̄ × z_oversample |
| Custom | `z_lambda_edges` provided | User-supplied monotonically increasing bin edges |
