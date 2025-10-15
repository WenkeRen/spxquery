# SPXQuery Package Implementation Plan

## Overview
SPXQuery is a Python package designed to automate SPHEREx spectral image data query, download, and time-domain analysis for astronomical sources.

## Package Requirements
See `requirements.txt` for all dependencies. Key packages:
- `astropy`: FITS handling, WCS, coordinates
- `pyvo`: TAP queries to IRSA
- `photutils`: Aperture photometry
- `tqdm`: Progress bars
- `pandas`: Data handling for CSV output
- `matplotlib`: Visualization

## Architecture

### Directory Structure
```
spxquery/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── config.py          # Configuration and data models
│   ├── query.py           # TAP query functionality
│   ├── download.py        # Parallel download manager
│   └── pipeline.py        # Main pipeline orchestrator
├── processing/
│   ├── __init__.py
│   ├── fits_handler.py    # FITS MEF handling
│   ├── photometry.py      # Aperture photometry extraction
│   └── lightcurve.py      # Light curve generation
├── visualization/
│   ├── __init__.py
│   └── plots.py           # Spectrum and light curve plotting
└── utils/
    ├── __init__.py
    └── helpers.py         # Utility functions
```

## Module Specifications

### 1. Core Module

#### config.py - Data Models and Configuration
```python
@dataclass
class Source:
    ra: float  # Right ascension in degrees
    dec: float  # Declination in degrees
    name: Optional[str] = None

@dataclass
class QueryConfig:
    source: Source
    output_dir: Path = Path.cwd()
    bands: Optional[List[str]] = None  # e.g., ['D1', 'D2']
    aperture_radius: float = 3.0  # pixels
    
@dataclass
class ObservationInfo:
    obs_id: str
    band: str
    mjd: float
    wavelength: float
    bandwidth: float
    access_url: str
    file_size_mb: float

@dataclass
class PhotometryResult:
    obs_id: str
    mjd: float
    flux: float
    flux_error: float
    wavelength: float
    bandwidth: float
    flag: int
    pix_x: float
    pix_y: float
```

#### query.py - TAP Query Module
Functions:
- `query_spherex_observations(source: Source, bands: Optional[List[str]]) -> QueryResults`
  - Execute TAP query to IRSA
  - Filter by coordinate overlap
  - Optional band filtering
- `analyze_query_results(results: QueryResults) -> QuerySummary`
  - Count observations per band
  - Calculate time span
  - Estimate total download size
- `get_download_urls(query_results: QueryResults) -> List[DownloadInfo]`
  - Process datalink URLs
  - Extract actual FITS URLs

#### download.py - Download Manager
Functions:
- `parallel_download(urls: List[str], output_dir: Path, max_workers: int = 4) -> DownloadResults`
  - Use ThreadPoolExecutor
  - Show progress with tqdm
  - Handle failures gracefully
- `download_file(url: str, output_path: Path, retry: int = 3) -> bool`
  - Download with retries
  - Verify file integrity

#### pipeline.py - Main Pipeline
Classes:
- `SPXQueryPipeline`
  - Methods:
    - `__init__(config: QueryConfig)`
    - `run_full_pipeline()` - Execute all steps
    - `run_query()` - Query and save results
    - `run_download()` - Download from saved query
    - `run_processing()` - Process downloaded files
    - `run_visualization()` - Create plots
    - `save_state()` / `load_state()` - Persistence

### 2. Processing Module

#### fits_handler.py - FITS MEF Handler
Functions:
- `read_spherex_mef(filepath: Path) -> SPHERExMEF`
  - Read all extensions
  - Parse headers
- `get_spectral_wcs(hdulist: HDUList) -> WCS`
  - Load spectral WCS
  - Disable SIP distortion
- `get_wavelength_at_position(wcs: WCS, x: float, y: float) -> Tuple[float, float]`
  - Return wavelength and bandwidth
- `subtract_zodiacal_background(image: np.ndarray, zodi: np.ndarray) -> np.ndarray`

#### photometry.py - Photometry Extraction
Functions:
- `extract_aperture_photometry(image: np.ndarray, x: float, y: float, radius: float) -> Tuple[float, float]`
  - Circular aperture photometry
  - Return flux and error
- `process_flags(flags: np.ndarray, x: float, y: float, radius: float) -> int`
  - OR flags within aperture
  - Return combined flag bitmap
- `extract_source_photometry(mef_file: Path, source: Source, aperture_radius: float) -> PhotometryResult`
  - Complete photometry extraction
  - Handle coordinate transformation

#### lightcurve.py - Light Curve Generation
Functions:
- `generate_lightcurve(photometry_results: List[PhotometryResult]) -> pd.DataFrame`
  - Compile all measurements
  - Sort by MJD
  - Handle upper limits
- `save_lightcurve_csv(df: pd.DataFrame, output_path: Path)`
  - Save with metadata
  - Include all required columns

### 3. Visualization Module

#### plots.py - Plotting Functions
Functions:
- `plot_spectrum(photometry_results: List[PhotometryResult]) -> Figure`
  - X-axis: wavelength (μm)
  - Y-axis: flux (MJy/sr)
  - X-error bars: bandwidth
  - Y-error bars: flux error
  - Upper limits for high errors
- `plot_lightcurve(photometry_results: List[PhotometryResult]) -> Figure`
  - X-axis: MJD
  - Y-axis: flux
  - Color-code by wavelength (0.75-5.0 μm)
- `create_combined_plot(photometry_results: List[PhotometryResult], output_path: Path)`
  - Two subplots: spectrum (top), lightcurve (bottom)

### 4. Utils Module

#### helpers.py - Utility Functions
- `mjd_from_spherex_time(t_min: float, t_max: float) -> float`
- `format_flag_binary(flag: int) -> str`
- `estimate_file_size(band: str) -> float`
- `validate_coordinates(ra: float, dec: float) -> bool`

## Implementation Steps

1. **Phase 1: Core Infrastructure**
   - Set up package structure
   - Implement data models
   - Create configuration system

2. **Phase 2: Query System**
   - TAP query implementation
   - Datalink URL processing
   - Query result analysis

3. **Phase 3: Download System**
   - Parallel download manager
   - Progress tracking
   - Error handling

4. **Phase 4: FITS Processing**
   - MEF reader
   - Spectral WCS handling
   - Zodiacal background subtraction

5. **Phase 5: Photometry**
   - Aperture photometry
   - Flag processing
   - Coordinate transformation

6. **Phase 6: Output Generation**
   - Light curve compilation
   - CSV export
   - Visualization

7. **Phase 7: Pipeline Integration**
   - Main pipeline class
   - State persistence
   - Step-by-step execution

## Key Technical Considerations

1. **Spectral WCS Handling**
   - Must disable SIP distortion for spectral WCS
   - Use lookup table for wavelength mapping

2. **Coordinate Systems**
   - Primary WCS: RA/DEC with SIP distortion
   - Alternative WCS 'W': Spectral coordinates

3. **Data Volume**
   - Each image ~70MB
   - Implement efficient download management

4. **Flag Handling**
   - 14 different flag types
   - Use bitmap operations

5. **Time-Domain Analysis**
   - MJD calculation: (t_max + t_min) / 2
   - Sort by observation time

## Testing Strategy

1. Unit tests for each module
2. Integration tests for pipeline
3. Mock TAP service for testing
4. Sample FITS files for processing tests

## Error Handling

1. Network errors during query/download
2. Invalid FITS files
3. Coordinate transformation failures
4. Missing data handling
5. User interruption recovery

## Performance Considerations

1. Parallel downloads (default 4 workers)
2. Efficient FITS reading (avoid loading full data when possible)
3. Progress tracking for long operations
4. Memory-efficient processing for large datasets