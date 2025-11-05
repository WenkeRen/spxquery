# Pipeline Architecture

SPXQuery uses a flexible, resumable pipeline architecture that processes SPHEREx data through four distinct stages.

## Four-Stage Pipeline

The pipeline executes in this order:

### 1. Query Stage

Query the IRSA SPHEREx archive using TAP (Table Access Protocol).

**What it does:**
- Searches for observations matching your source coordinates (RA/Dec)
- Filters by spectral bands (D1-D6)
- Resolves datalink URLs for each observation
- Saves query results and metadata

**Output:**
- `results/query_summary.json` - Observation metadata, time span, data size
- `results/download_urls.json` - Cached datalink URLs for downloads

**Key features:**
- Automatic coordinate matching within search radius
- Band filtering (query specific bands or all)
- URL caching to avoid repeated datalink queries

### 2. Download Stage

Download FITS files from IRSA with optional cutout support.

**What it does:**
- Downloads spectral images via HTTP
- Applies cutout parameters if specified (reduces file size by 90%)
- Organizes files by band (data/D1/, data/D2/, etc.)
- Tracks download progress with parallel workers

**Output:**
- `data/D*/` - FITS files organized by spectral band
- Download progress logging

**Key features:**
- Parallel downloads (configurable workers, default: 4)
- Skip existing files to enable resume
- Retry logic with exponential backoff
- Progress tracking for large datasets

### 3. Processing Stage

Extract aperture photometry from FITS files.

**What it does:**
- Parses Multi-Extension FITS (MEF) structure
- Extracts flux using circular aperture photometry
- Estimates and subtracts zodiacal background
- Handles pixel flags for quality assessment
- Computes flux uncertainties from variance maps

**Output:**
- `results/photometry.json` - Per-observation photometry results
- Photometry metadata (aperture size, background estimation)

**Key features:**
- Automatic background annulus sizing
- Zodiacal light subtraction (from ZODI extension)
- Pixel flag tracking (FLAGS extension)
- Spectral WCS handling for wavelength extraction
- Parallel processing (configurable workers, default: 10)

### 4. Visualization Stage

Generate publication-quality plots with quality control.

**What it does:**
- Creates combined spectral and temporal plots
- Applies quality filtering (SNR threshold, bad pixel flags)
- Marks rejected measurements with visual indicators
- Generates light curve CSV file

**Output:**
- `results/combined_plot.png` - Multi-panel visualization
- `results/lightcurve.csv` - Time-series photometry data

**Key features:**
- Quality control: good measurements (filled circles) vs. rejected (gray crosses)
- Customizable colormaps, marker sizes, and figure parameters
- Respects user's matplotlibrc settings
- Optional magnitude vs. flux plotting

## Pipeline Execution Modes

SPXQuery supports three execution modes:

### One-Click Execution

Run all stages automatically:

```python
from spxquery.core.pipeline import run_pipeline

run_pipeline(
    ra=304.69,
    dec=42.44,
    output_dir="output",
    cutout_size="200px"
)
```

### Step-by-Step Execution

Run individual stages with dependency checking:

```python
from spxquery import SPXQueryPipeline, Source, QueryConfig

source = Source(ra=304.69, dec=42.44, name="my_source")
config = QueryConfig(source=source, output_dir="output")
pipeline = SPXQueryPipeline(config)

# Run stages individually
pipeline.run_query()
pipeline.run_download(skip_existing=True)
pipeline.run_processing()
pipeline.run_visualization()
```

The pipeline automatically checks dependencies - you cannot run `processing` before completing `download`.

### Resumable Execution

The pipeline saves state after each stage to `{source_name}.json`. Resume from interruptions:

```python
# Load configuration from saved state
config = QueryConfig.from_saved_state(
    source_name="my_source",
    output_dir="output"
)

pipeline = SPXQueryPipeline(config)
pipeline.resume()  # Automatically runs remaining stages
```

**What gets saved:**
- Completed stages
- Query results (observations, time span, data size)
- Downloaded file paths
- Photometry results
- All configuration parameters

## Stage Dependencies

The pipeline enforces these dependencies:

- **query**: No dependencies (always runs first)
- **download**: Requires `query`
- **processing**: Requires `query` + `download`
- **visualization**: Requires `query` + `download` + `processing`

If you try to run a stage without its dependencies, the pipeline will raise an error.

## Customizing Pipeline Stages

You can customize which stages to run:

```python
# Only query and download, skip processing
pipeline = SPXQueryPipeline(config, pipeline_stages=["query", "download"])
pipeline.run_full_pipeline()
```

This is useful for:
- Downloading data for later analysis
- Re-running specific stages with different parameters
- Integrating SPXQuery into custom workflows

## State Persistence

State files (`{source_name}.json`) contain:

```json
{
  "stage": "complete",
  "completed_stages": ["query", "download", "processing", "visualization"],
  "pipeline_stages": ["query", "download", "processing", "visualization"],
  "query_results": {
    "observations": [...],
    "time_span_days": 33.1,
    "total_size_gb": 0.51
  },
  "downloaded_files": [...],
  "photometry_results": [...]
}
```

This enables:
- Resume after interruptions (network failures, crashes)
- Audit trail of completed work
- Configuration recovery (auto-load parameters from saved state)

## Error Handling

The pipeline handles common errors gracefully:

- **Network failures**: Retry logic with exponential backoff (configurable)
- **Missing files**: Skip and continue processing remaining files
- **Invalid FITS**: Log error and skip observation
- **Photometry failures**: Mark as bad and continue
- **Interrupted execution**: Resume from last completed stage

Errors are logged to help diagnose issues without stopping the entire pipeline.
