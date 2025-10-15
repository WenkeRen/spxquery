# SPXQuery - SPHEREx Time-Domain Analysis Package

SPXQuery is a Python package designed to automate SPHEREx spectral image data query, download, and time-domain analysis for astronomical sources.

## Features

- **Automated TAP queries** to IRSA SPHEREx archive
- **Parallel URL resolution** with progress tracking and caching
- **Parallel downloads** with progress tracking
- **FITS MEF processing** with proper spectral WCS handling
- **Aperture photometry** extraction with zodiacal background subtraction
- **Time-domain analysis** with light curve generation
- **Visualization** of spectra and light curves
- **Resumable pipeline** with state persistence and URL caching
- **Step-by-step execution** support

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

```python
from spxquery.core.pipeline import run_pipeline

# Run full pipeline for a source with image cutout
run_pipeline(
    ra=304.693508808,      # Right ascension in degrees
    dec=42.4436872991,     # Declination in degrees
    output_dir="my_output",
    source_name="My_Star",
    cutout_size="200px"    # Download 200x200 pixel cutouts instead of full images
)
```

## Usage Examples

### 1. Simple Full Pipeline

```python
from spxquery import Source, QueryConfig, SPXQueryPipeline

# Define source
source = Source(ra=304.69, dec=42.44, name="Variable_Star")

# Configure pipeline
config = QueryConfig(
    source=source,
    output_dir="output",
    bands=['D1', 'D2', 'D3'],  # Specific bands only
    aperture_diameter=3.0,     # 3 pixel diameter
    cutout_size="200px"        # Download 200x200 pixel cutouts (optional)
)

# Run pipeline
pipeline = SPXQueryPipeline(config)
pipeline.run_full_pipeline()
```

### 2. Step-by-Step Execution

```python
# Create pipeline
pipeline = SPXQueryPipeline(config)

# Run individual stages
pipeline.run_query()      # Query IRSA archive
pipeline.run_download()   # Download FITS files
pipeline.run_processing() # Extract photometry
pipeline.run_visualization() # Create plots
```

### 3. Resume After Interruption

```python
# Resume from saved state
pipeline = SPXQueryPipeline(config)
pipeline.resume()
```

### 4. Using Image Cutouts

Image cutouts significantly reduce download time and storage requirements:

```python
# Example 1: 200-pixel square cutout (recommended for point sources)
config = QueryConfig(
    source=source,
    output_dir="output",
    cutout_size="200px"  # ~0.7 MB per file vs ~70 MB full image
)

# Example 2: Rectangular cutout in arcminutes
config = QueryConfig(
    source=source,
    output_dir="output",
    cutout_size="5,10arcmin"  # 5×10 arcminute rectangular cutout
)

# Example 3: Cutout with custom center (useful for offset sources)
config = QueryConfig(
    source=source,
    output_dir="output",
    cutout_size="3arcmin",
    cutout_center="304.7,42.5"  # RA, Dec in degrees
)

# Example 4: Full images (no cutout)
config = QueryConfig(
    source=source,
    output_dir="output"
    # Omit cutout_size to download full images
)
```

**Cutout Size Formats:**
- Pixels: `"200px"`, `"100,200pixels"`, `"500pix"`
- Angular: `"3arcmin"`, `"30,45arcsec"`, `"0.1deg"`, `"0.05"` (degrees default)
- SPHEREx pixel scale: ~6.2 arcsec/pixel

**Storage Comparison:**
- Full image: ~70 MB
- 200px cutout: ~0.7 MB (99% reduction)
- 500px cutout: ~4.3 MB (94% reduction)
- 1000px cutout: ~17 MB (76% reduction)


## Output Structure

```
output_dir/
├── data/                    # Downloaded FITS files
│   ├── D1/                  # Band D1 files
│   ├── D2/                  # Band D2 files
│   └── ...
├── results/                 # Analysis results
│   ├── query_summary.json   # Query metadata
│   ├── download_urls.json   # Cached download URLs
│   ├── lightcurve.csv       # Photometry results
│   └── combined_plot.png    # Visualization
└── pipeline_state.json      # Resumable state
```

## Light Curve CSV Format

The output CSV contains:
- `obs_id`: Observation identifier
- `mjd`: Modified Julian Date
- `flux`: Flux in MJy/sr
- `flux_error`: Flux uncertainty
- `wavelength`: Central wavelength (μm)
- `bandwidth`: Bandwidth (μm)
- `band`: SPHEREx band (D1-D6)
- `flag`: Combined flag bitmap
- `pix_x`, `pix_y`: Pixel coordinates
- `is_upper_limit`: True if error > flux

## SPHEREx Bands

- **D1**: 0.75-1.09 μm (R=39)
- **D2**: 1.10-1.62 μm (R=41)
- **D3**: 1.63-2.41 μm (R=41)
- **D4**: 2.42-3.82 μm (R=35)
- **D5**: 3.83-4.41 μm (R=112)
- **D6**: 4.42-5.00 μm (R=128)

## Advanced Usage

### Custom Query with Parallel URL Resolution

```python
from spxquery.core.query import query_spherex_observations, get_download_urls
from pathlib import Path

# Query observations
results = query_spherex_observations(
    source=Source(ra=304.69, dec=42.44),
    bands=['D2', 'D3']
)

# Get download URLs with caching and parallel processing
urls = get_download_urls(
    results,
    max_workers=8,
    show_progress=True,
    cache_file=Path("my_urls.json")
)
```

### Custom Photometry

```python
from spxquery.processing.photometry import extract_source_photometry

result = extract_source_photometry(
    mef_file=Path("data/D2/obs_123.fits"),
    source=Source(ra=304.69, dec=42.44),
    aperture_radius=2.0  # Custom aperture
)
```

## Troubleshooting

1. **TAP Query Timeout**: Increase timeout in query module
2. **Download Failures**: Check network, reduce max_workers
3. **Memory Issues**: Process files in batches
4. **Missing Dependencies**: Run `pip install -r requirements.txt`

## License

This package is provided as-is for SPHEREx data analysis.

## Acknowledgments

Based on SPHEREx data from NASA/IPAC Infrared Science Archive (IRSA).