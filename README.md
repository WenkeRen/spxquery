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
- **Image cutout support** for reduced download sizes
- **Quality control filtering** for photometry visualization

## Installation

Install dependencies and the package in development mode:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from spxquery.core.pipeline import run_pipeline

# Run full pipeline for a source with image cutout
run_pipeline(
    ra=304.693508808,
    dec=42.4436872991,
    output_dir="my_output",
    source_name="My_Star",
    cutout_size="200px"  # Optional: download cutouts instead of full images
)
```

## Basic Usage

### Full Pipeline with Configuration

```python
from spxquery import Source, QueryConfig, SPXQueryPipeline

source = Source(ra=304.69, dec=42.44, name="Variable_Star")
config = QueryConfig(
    source=source,
    output_dir="output",
    bands=['D1', 'D2', 'D3'],  # Optional: specific bands only
    aperture_diameter=3.0,     # Aperture size in pixels
    cutout_size="200px",       # Optional: download cutouts
    sigma_threshold=5.0,       # QC: minimum SNR (default: 5.0)
    bad_flags=[0,1,2,6,7,9,10,11,15]  # QC: bad pixel flags (default)
)

pipeline = SPXQueryPipeline(config)
pipeline.run_full_pipeline()
```

### Step-by-Step Execution

Execute individual pipeline stages for more control:

```python
pipeline = SPXQueryPipeline(config)
pipeline.run_query()          # Query IRSA archive
pipeline.run_download()       # Download FITS files
pipeline.run_processing()     # Extract photometry
pipeline.run_visualization()  # Create plots
```

### Resume After Interruption

```python
pipeline = SPXQueryPipeline(config)
pipeline.resume()  # Resume from saved state
```

## Image Cutouts

Cutouts significantly reduce download time and storage by downloading only the region of interest instead of full 2040×2040 images (~70 MB each).

**Cutout Parameters:**
- `cutout_size`: Specify size with units (e.g., "200px", "100,200px", "3arcmin", "0.1deg")
  - Single value: square cutout; two values: rectangular cutout
  - Units: px/pix/pixels (pixels), arcsec/arcmin/deg/rad (angular), default: degrees
- `cutout_center`: Optional center coordinates (defaults to source position)
  - Format: "x,y[units]" (e.g., "304.5,42.3" for RA/Dec in degrees)

**Storage Comparison:**
- Full image: ~70 MB
- 200px cutout: ~0.7 MB (99% reduction)
- 500px cutout: ~4.3 MB (94% reduction)
- 1000px cutout: ~17 MB (76% reduction)

SPHEREx pixel scale: ~6.2 arcsec/pixel

## Quality Control

Quality control filters affect visualization only - ALL measurements are saved to CSV.

**Parameters:**
- `sigma_threshold`: Minimum SNR (flux/flux_err) - default: 5.0
- `bad_flags`: Pixel flag bits to reject - default: [0,1,2,6,7,9,10,11,15]

**Visualization:**
- Good measurements: filled circles with error bars
- Rejected measurements: small gray crosses (for visual inspection)
- CSV output: contains ALL measurements regardless of QC status

**Default bad flags reject:**
- Saturation (bit 0)
- Bad/hot pixels (bits 1, 2)
- Cosmic rays (bit 6)
- Non-linearity (bit 7)
- Edge effects (bits 9, 10, 11)
- Other quality issues (bit 15)

## Output Structure

```
output_dir/
├── data/                    # Downloaded FITS files (organized by band)
├── results/                 # Analysis results
│   ├── query_summary.json   # Query metadata
│   ├── download_urls.json   # Cached download URLs
│   ├── lightcurve.csv       # Photometry results
│   └── combined_plot.png    # Visualization
└── pipeline_state.json      # Resumable state
```

## Light Curve CSV Columns

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

| Band | Wavelength Range | Resolving Power |
|------|------------------|-----------------|
| D1   | 0.75-1.09 μm     | R=39           |
| D2   | 1.10-1.62 μm     | R=41           |
| D3   | 1.63-2.41 μm     | R=41           |
| D4   | 2.42-3.82 μm     | R=35           |
| D5   | 3.83-4.41 μm     | R=112          |
| D6   | 4.42-5.00 μm     | R=128          |

## Advanced Usage

### Custom Query with URL Caching

```python
from spxquery.core.query import query_spherex_observations, get_download_urls

results = query_spherex_observations(
    source=Source(ra=304.69, dec=42.44),
    bands=['D2', 'D3']
)

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

- **TAP Query Timeout**: Increase timeout in query module
- **Download Failures**: Check network connection, reduce max_workers
- **Memory Issues**: Process files in batches or use cutouts
- **Missing Dependencies**: Run `pip install -r requirements.txt`

## License

This package is provided as-is for SPHEREx data analysis.

## Acknowledgments

Based on SPHEREx data from NASA/IPAC Infrared Science Archive (IRSA).
