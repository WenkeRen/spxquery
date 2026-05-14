# Batch Photometry

The batch photometry module (`spxquery.batch`) enables multi-source aperture photometry over a sky region. Unlike the single-source pipeline which downloads small cutouts around one target, the batch module queries for **full-frame images** covering a circular region, then extracts photometry for **all catalog sources** in each image simultaneously.

## When to Use Batch vs. Single-Source

| Feature | Single-Source (`SPXQueryPipeline`) | Batch (`spxquery.batch`) |
|---------|-------------------------------------|--------------------------|
| Targets | One source per run | Multiple sources from catalog CSV |
| Downloads | Cutout images (~100 KB each) | Full-frame images (~70 MB each) |
| Query | Point search (CONESearch) | Region search (CIRCLE + INTERSECTS) |
| Output | Per-source light curve + plot | Per-source light curves (CSV only) |
| Use case | Detailed analysis of one object | Survey of many objects in a region |

## Quick Start

```python
from spxquery.batch import run_batch

run_batch(
    catalog="sources.csv",
    center_ra=270.0,
    center_dec=66.56,
    radius=1.0,
    bands=["D3", "D4"],
)
```

This single call will:
1. Query the IRSA TAP service for full-frame images covering a 1° radius circle
2. Download matching images in D3 and D4 bands
3. Extract aperture photometry for all catalog sources in each image
4. Aggregate per-image results into per-source light curves

## Catalog Format

The source catalog must be a CSV file with columns `targetid`, `ra`, `dec`:

```csv
targetid,ra,dec
39633458707826492,265.623,66.531
39633451346821630,266.445,65.636
39633453829850190,266.794,65.983
```

Additional columns (flux, redshift, etc.) are ignored. Coordinates must be in degrees (ICRS).

## Configuration

```python
from pathlib import Path
from spxquery.batch import BatchConfig, run_batch
from spxquery.core.config import PhotometryConfig

config = BatchConfig(
    # Sky region
    center_ra=270.0,
    center_dec=66.56,
    radius=1.0,
    catalog_path=Path("sources.csv"),

    # Query filters
    coverage_mode="any",
    bands=["D3", "D4"],
    mjd_range=(60800, 61000),

    # Safety
    max_images=500,

    # Output
    output_dir=Path("batch_output"),

    # Parallelism
    max_download_workers=4,
    max_extract_workers=12,

    # Photometry parameters (forwarded to extraction)
    photometry=PhotometryConfig(
        aperture_method="fwhm",
        fwhm_multiplier=2.5,
        background_method="window",
        window_size=30,
        subtract_zodi=True,
    ),
)

run_batch(config)
```

### Configuration Parameters

#### Region and Query

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `center_ra` | float | required | Region center RA in degrees (0–360) |
| `center_dec` | float | required | Region center Dec in degrees (−90 to +90) |
| `radius` | float | required | Search radius in degrees |
| `catalog_path` | Path | required | CSV file with `targetid`, `ra`, `dec` columns |
| `coverage_mode` | str | `"any"` | `"any"` = image overlaps region; `"full"` = image fully contains region |
| `bands` | list[str] | `None` | Filter by band, e.g. `["D1", "D3"]`. `None` = all bands |
| `mjd_range` | tuple | `None` | Time filter as `(mjd_min, mjd_max)`. `None` = no filter |
| `max_images` | int | 500 | Raise error if query returns more images than this |

#### Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_download_workers` | int | 4 | Parallel download threads |
| `max_extract_workers` | int | 12 | Parallel extraction processes |
| `output_dir` | Path | `"batch_output"` | Root directory for all outputs |
| `num_buckets` | int | 64 | Hash-partition buckets for aggregation |
| `keep_bucket_files` | bool | `False` | Keep temporary bucket CSVs after aggregation |
| `photometry` | PhotometryConfig | defaults | Photometry parameters (see {doc}`parameters`) |

## Coverage Modes

The coverage mode controls which images are selected from the archive:

### `any` (INTERSECTS)

Selects images whose footprint **overlaps** with the search circle. This is the most inclusive mode — it returns all images that touch the region, even if only a small corner overlaps.

```
Query: INTERSECTS(p.poly, CIRCLE('ICRS', ra, dec, radius)) = 1
```

Use when you want maximum coverage and don't mind some images only partially covering your region.

### `full` (CONTAINS)

Selects images that **fully contain** the search circle. This ensures every returned image covers the entire region, so all catalog sources are present in every image.

```
Query: CONTAINS(CIRCLE('ICRS', ra, dec, radius), p.poly) = 1
```

Use when you need complete source coverage across all images (e.g., for consistent light curves). Since SPHEREx full-frame images are ~3.7° across, a `full` search with radius < 1° will return a smaller but more complete subset.

## Band and Time Filtering

### Band Selection

SPHEREx has 6 detectors covering different wavelength ranges:

| Band | Wavelength (μm) | Resolving Power |
|------|-----------------|-----------------|
| D1 | 0.75–1.09 | R ≈ 39 |
| D2 | 1.10–1.62 | R ≈ 41 |
| D3 | 1.63–2.41 | R ≈ 41 |
| D4 | 2.42–3.82 | R ≈ 35 |
| D5 | 3.83–4.41 | R ≈ 112 |
| D6 | 4.42–5.00 | R ≈ 128 |

To query only specific bands:

```python
config = BatchConfig(
    ...,
    bands=["D3", "D4"],  # Only near-infrared
)
```

Setting `bands=None` (default) queries all 6 bands.

### MJD Range

Filter observations by Modified Julian Date to restrict to a specific time window:

```python
config = BatchConfig(
    ...,
    mjd_range=(60800, 61000),  # ~200 days
)
```

This is applied as a post-query filter. Use it to limit the number of downloaded images when the region has extensive temporal coverage.

## Pipeline Stages

The batch pipeline has four stages: **Query → Download → Extract → Aggregate**.

### Query

Queries the IRSA TAP service using ADQL spatial predicates. The search region is defined by a circle (center RA/Dec + radius). Results include download URLs, observation IDs, band information, and time stamps.

### Download

Downloads full-frame FITS images from IRSA. Uses the same parallel download engine as the single-source pipeline, but without cutout parameters.

### Extract

For each image, the extraction stage:

1. Reads the MEF file once (IMAGE, FLAGS, VARIANCE, ZODI extensions)
2. Projects all catalog sources onto the image via batch WCS transformation
3. Filters to sources within the field of view
4. Extracts aperture photometry for each in-FOV source using pre-computed shared arrays:
   - Background quality mask (combined bitmask)
   - Error array (sqrt of variance)
   - Pixel scale
5. Writes per-image CSV files

### Aggregate

Combines per-image CSVs into per-source light curves using hash-partitioned bucket aggregation:

1. Partition all per-image rows into hash buckets by `target_id`
2. Sort each bucket by `(target_id, mjd)`
3. Write one CSV per source

This approach avoids loading the entire dataset into memory.

## Output Structure

```
batch_output/
├── images/                              # Downloaded full-frame FITS files
│   ├── level2_2025W25_1B_0263_4D3_*.fits
│   └── ...
├── per_image/                           # Per-image photometry CSVs
│   ├── level2_2025W25_1B_0263_4D3_*_photometry.csv
│   └── ...
└── lightcurves/                         # Per-source light curves
    ├── 39633458707826492.csv
    ├── 39633451346821630.csv
    └── ...
```

### Per-Image CSV Columns

Each per-image CSV contains photometry for all in-FOV sources from one observation:

| Column | Unit | Description |
|--------|------|-------------|
| `target_id` | — | Source identifier from catalog |
| `ra`, `dec` | deg | Source coordinates |
| `obs_id` | — | Observation ID |
| `band` | — | Detector band (D1–D6) |
| `mjd` | days | Modified Julian Date |
| `x`, `y` | pixels | Pixel coordinates on image |
| `flux` | μJy | Background-subtracted flux |
| `flux_error` | μJy | Flux uncertainty |
| `mag_ab` | mag | AB magnitude |
| `mag_ab_error` | mag | Magnitude uncertainty |
| `wavelength` | μm | Central wavelength |
| `bandwidth` | μm | Bandpass width |
| `flag` | — | Combined pixel flags (bitwise OR) |
| `bg_level` | uJy/arcsec² | Estimated background per pixel |
| `bg_error` | uJy/arcsec² | Background uncertainty |
| `aperture_radius` | pixels | Aperture radius used |
| `filename` | — | Source FITS filename |

### Light Curve CSV Columns

Each light curve CSV contains all observations for one source across all images:

```
obs_id,band,mjd,x,y,flux,flux_error,mag_ab,mag_ab_error,wavelength,bandwidth,flag,bg_level,bg_error,aperture_radius
```

## Step-by-Step API

For more control over individual stages:

```python
from pathlib import Path
from spxquery.batch import BatchPipeline, BatchConfig
from spxquery.core.config import PhotometryConfig

config = BatchConfig(
    center_ra=270.0,
    center_dec=66.56,
    radius=1.0,
    catalog_path=Path("sources.csv"),
    bands=["D3"],
    coverage_mode="full",
    output_dir=Path("batch_output"),
)

pipeline = BatchPipeline(config)

# Run stages individually
pipeline.run_query()       # TAP query → observations list
pipeline.run_download()    # Parallel download → images/
pipeline.run_extract()     # Multi-source extraction → per_image/
pipeline.run_aggregate()   # Bucket aggregation → lightcurves/

# Or run all at once
pipeline.run_all()
```

### Incremental Execution

The extract stage supports incremental processing — if a per-image CSV already exists, that image is skipped:

```python
# First run: processes all images
pipeline.run_extract()

# Later: only new images are processed
pipeline.run_extract()  # skip_existing=True by default
```

## Performance

The batch extraction is optimized for processing many sources across many images:

- **Pre-computed shared arrays**: Error map, background quality mask, and pixel scale are computed once per image (not per source)
- **Local cutout photometry**: Aperture photometry operates on small cutouts instead of full 2040×2040 images
- **Batch WCS projection**: All source coordinates are projected in a single WCS call
- **Combined bitmask flag filtering**: Single bitwise operation replaces per-flag loops
- **Bucket-based aggregation**: Memory-efficient aggregation via hash partitioning

Typical performance (single-threaded, single image):

| Sources in FOV | Time per image |
|----------------|---------------|
| 5 | ~110 ms |
| 17 | ~85 ms |
| 34 | ~85 ms |

I/O (reading FITS from disk) dominates the per-image cost. With 12 parallel workers, throughput scales near-linearly for I/O-unbound cases.
