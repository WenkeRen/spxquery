# Drizzle3D — SPHEREx 3D Spectral Image Drizzle

Combines multiple SPHEREx observations into a single data cube (X, Y, λ)
using a decoupled spatial + spectral drizzle algorithm with inverse-variance
weighting.

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
              (vectorized Z overlap + np.add.at accumulation)
                      │
                      ▼
                DrizzleCube (in-memory)
                      │
                      ▼
                  save_cube (7-HDU FITS)
```

### High-Level Flow

1. **Configure** — `Drizzle3DConfig` defines the sky region, detector, drizzle
   kernel parameters, and output settings.
2. **Build grids** — A 2-D TAN spatial WCS and a 1-D spectral (Z) bin grid are
   constructed from the config.
3. **Query & download** — IRSA TAP is queried for SPHEREx observations
   intersecting the target region; matching FITS files are downloaded in
   parallel.
4. **Per-image drizzle** — Each input FITS is read, its spatial and spectral
   WCS are extracted, and pixel-to-output mapping is computed. Valid
   contributions are accumulated into a shared `DrizzleCube` using vectorized
   numpy operations.
5. **Save** — The finalized cube is written as a 7-HDU FITS file.

---

## Module Reference

### `config.py` — Drizzle3DConfig

The single entry point for all pipeline parameters. Validated on construction
via `__post_init__`.

**Key parameter groups:**

| Group | Parameters | Purpose |
|---|---|---|
| Sky region | `center_ra`, `center_dec`, `width`, `height` | Where and how big the output cube is |
| Detector | `detector` (0=all, 1–6=single) | Which SPHEREx detector to process |
| Droplet shrink | `xy_shrink`, `z_shrink` | Input pixel footprint scaling (0, 1] |
| Output grid | `xy_oversample`, `z_oversample`, `z_lambda_edges` | Output resolution control |
| Query | `mjd_range`, `max_images`, `download_workers` | IRSA TAP query constraints |
| Processing | `subtract_zodi`, `exclude_flags` | Pre-drizzle data cleaning |
| Accumulation | `ivar_max`, `min_overlap` | Weighting and quality cuts |

**Computed properties:**

- `effective_pixscale()` → `BASE_PIXSCALE / xy_oversample` (arcsec/pixel)
- `output_nx()`, `output_ny()` → grid dimensions in pixels
- `spatial_radius_deg()` → half-diagonal for TAP region queries
- `effective_z_shrink()` → `z_shrink` if set, else `xy_shrink`

Supports JSON serialization via `to_json()` / `from_json()`.

---

### `grid.py` — Output Spatial WCS

`build_output_wcs(config)` constructs a 2-D gnomonic (TAN) projection:

- CRPIX is placed at the geometric center of the output image so that
  `(nx//2, ny//2)` maps to `(center_ra, center_dec)`.
- CDELT is `±pixscale_deg` (negative for RA axis).
- Pixel scale defaults to 6.15 arcsec/pixel at `xy_oversample=1.0`.

---

### `spectral.py` — Spectral (Z) Axis Grid

Reads SPHEREx's `SPECTRAL_CHANNELS` table (102 rows for D1–D6 × 17 subchannels)
from the bundled aux FITS file and builds the output wavelength bin grid.

**`ZGrid` dataclass** — holds bin edges, centers, and widths. Provides utility
methods: `frequencies()`, `delta_nu()`, `resolving_power()`.

**Three grid modes** dispatched by `build_z_grid()`:

| Mode | Trigger | Algorithm |
|---|---|---|
| Native | `z_oversample=1.0`, no custom edges | Uses WL_MIN/WL_MAX from spectral table as bin edges (17 bins/detector) |
| Oversampled | `z_oversample > 1.0` | Logarithmic spacing at constant R_eff = R̄ × z_oversample |
| Custom | `z_lambda_edges` provided | User-supplied monotonically increasing bin edges |

---

### `spatial.py` — Spatial (XY) Mapping

`compute_spatial_mapping()` determines which output pixel(s) each input pixel
contributes to, using **bilinear weight distribution**.

**Algorithm:**

1. Transform all input pixel centers → sky coordinates via `input_wcs`.
2. Project onto the output tangent plane via `output_wcs.world_to_pixel_values()`.
3. For each valid input pixel (within output bounds ± 2-pixel margin), compute
   the 2×2 neighborhood of surrounding output pixels.
4. Bilinear weight: `w = max(1 - |Δx|, 0) × max(1 - |Δy|, 0) × xy_shrink²`.
   Only non-zero weights (|Δx| < 1, |Δy| < 1) are kept.

**Returns** 5 arrays (all flat, same length):

| Return | Type | Meaning |
|---|---|---|
| `valid_mask` | bool (ny_in, nx_in) | Which input pixels overlap the output grid |
| `pixel_idx` | int64 | Flat index into the input image for each contribution |
| `out_y`, `out_x` | int32 | Output pixel coordinates |
| `f_xy` | float64 | Spatial overlap fraction |

Each input pixel produces 1–4 output contributions (bilinear distribution).

---

### `accumulate.py` — Core Drizzle Engine

#### `DrizzleCube`

In-memory accumulation buffer of shape `(n_z, n_y, n_x)`. Contains:

| Array | Dtype | Accumulates |
|---|---|---|
| `flux_weighted` | float64 | Σ w × f_xy × f_z × F_i |
| `weight_total` | float64 | Σ w × f_xy × f_z |
| `var_accum` | float64 | Σ (w × f_xy × f_z)² × σ² |
| `count_map` | uint16 | Number of contributing input pixels |
| `and_mask` | uint32 | Bitwise AND of input FLAGS (conservative) |
| `or_mask` | uint32 | Bitwise OR of input FLAGS (inclusive) |

Final outputs are computed as properties:
- `flux = flux_weighted / weight_total` (inverse-variance weighted mean)
- `variance = var_accum / weight_total²`

#### `drizzle_image()` — Per-Image Accumulation

Processes one input image into the cube. Fully vectorized (no Python loops
over pixels):

1. **Filter** spatial contributions by valid per-pixel data
   (finite λ_c, Δλ > 0, variance > 0, not excluded by flags).
2. **Group** by unique input pixel via `np.unique`.
3. **Vectorized spectral overlap** — dense `(n_unique, n_z)` matrix:
   - Top-hat kernel: `[λ_c − ½Δλ·z_shrink, λ_c + ½Δλ·z_shrink]`
   - Overlap length with each Z bin: `max(min(hi, z_edge[j+1]) − max(lo, z_edge[j]), 0)`
   - Normalize each row so Σf_z ≈ 1.
4. **Cross-product** spatial × spectral: `(n_contrib, n_z)` with
   `total_f = f_xy × f_z`, `w = ivar × total_f`.
5. **Unbuffered accumulation** via `np.add.at` (handles duplicate indices
   correctly) and `np.bitwise_and.at` / `np.bitwise_or.at` for masks.

---

### `query.py` — IRSA TAP Query & Download

#### `query_observations(config)`

Queries the IRSA TAP service (`spherex.artifact` + `spherex.plane` tables) using
`CONTAINS(POINT, poly)` to find observations whose footprint covers the target
position. Results are grouped into `{detector: [DrizzleObservation, ...]}`.

Supports filtering by:
- Detector band (`energy_bandpassname`)
- MJD time range
- Safety cap via `max_images`

#### `download_observations(observations, ...)`

Downloads FITS files into `output_dir/images/D{N}/` with:
- `skip_existing` to avoid re-downloading
- Sequential download with progress bar (tqdm)

---

### `pipeline.py` — Orchestrator

#### `drizzle(config)` — Top-Level Entry Point

Full end-to-end pipeline:

```
build_output_wcs → query_observations → [per detector: download → drizzle_detector → save_cube]
```

Returns `{detector: output_path}` for each successfully processed detector.

#### `drizzle_detector(fits_paths, config, detector, output_wcs)`

Per-detector pipeline loop:

1. Build Z grid.
2. Allocate empty `DrizzleCube`.
3. For each input FITS: read data → spatial mapping → wavelength extraction →
   `drizzle_image()`.
4. Finalize masks and save.

#### Helper Functions

- `_read_input_fits(filepath, subtract_zodi)` — Opens a SPHEREx MEF FITS,
  reads IMAGE, VARIANCE, FLAGS, ZODI extensions, and extracts spatial + spectral
  WCS. Optionally subtracts zodiacal background.
- `_extract_wavelength_maps(spectral_wcs, shape)` — Converts the spectral WCS
  (alternative 'W' key) into per-pixel `(λ_c, Δλ)` maps using
  `pixel_to_world()`.

---

### `io.py` — FITS I/O

#### `save_cube(cube, path)` — 7-HDU Output Format

| HDU | Name | Content |
|-----|------|---------|
| 0 | PRIMARY | Metadata headers (detector, N_INPUTS, shrink factors, wavelength range, etc.) |
| 1 | SCI | float32 — flux-weighted mean surface brightness [MJy/sr] |
| 2 | VARIANCE | float32 — per-voxel variance [MJy²/sr²] |
| 3 | AND_MASK | uint32 — conservative bitwise AND of input flags |
| 4 | OR_MASK | uint32 — inclusive bitwise OR of input flags |
| 5 | COUNT | uint16 — number of contributing input pixels |
| 6 | WAVELENGTH | BinTableHDU — INDEX, LAMBDA, LAMBDA_MIN, LAMBDA_MAX, DLAMBDA, NU, DNU, R_EFF |

SCI and VARIANCE HDUs include an approximate 3-D WCS (TAN + WAVE) for
compatibility with DS9, CARTA, and other FITS viewers. The exact per-plane
wavelengths are in the WAVELENGTH extension.

#### `load_cube(path)` — Reader

Returns a dict with keys: `header`, `sci`, `variance`, `and_mask`, `or_mask`,
`count`, `wavelength`.

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
# → {"3": Path("output/drizzle_D3.fits")}
```

See `example/drizzle3d_demo/ngc4395_drizzle.py` for a full 6-step demo
(config → grids → query → download → drizzle → visualize) and
`example/drizzle3d_demo/ngc4395_explore.ipynb` for an interactive Jupyter
notebook with Z-bin sliders and click-to-spectrum exploration.
