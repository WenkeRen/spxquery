# SPXQuery Cutout Feature Implementation

## Update Date: 2025-10-15

## Overview
This document tracks the implementation of image cutout functionality for the SPXQuery package. This feature allows users to download image cutouts instead of full FITS files, significantly reducing data volume and download times.

## Motivation
SPHEREx FITS files are approximately 70MB each (2040×2040 pixels). For point source analysis, downloading the entire image is often unnecessary and wasteful. The IRSA Image Cutout Service allows requesting specific image regions, reducing both:
- Download time and bandwidth
- Local storage requirements
- Processing time for photometry

## Feature Description

### IRSA Cutout Service
The IRSA cutout service allows appending URL parameters to FITS download URLs:

**Size Parameter Format:**
```
size=<value>[,<value>][units]
```
- First value (x): full-width along NAXIS1 (image columns)
- Second value (y): full-height along NAXIS2 (image rows)
- If only one value: used for both width and height
- Units: `px/pix/pixels` (pixels), `arcsec/arcmin/deg/rad` (angular)
- Default units: degrees
- Negative sizes are illegal

**Center Parameter Format:**
```
center=<x>,<y>[units]
```
- If units are pixels: pixel coordinates of cutout center
- If units are angular: J2000 RA (x) and Dec (y) coordinates
- Default units: degrees
- Declination must be in [-90, 90] degrees
- RA values are range-reduced (any value accepted)

**Example URLs:**
```
# 200 pixel square cutout
https://irsa.ipac.caltech.edu/ibe/data/wise/...fits?size=200px

# 100x200 pixel cutout
https://irsa.ipac.caltech.edu/ibe/data/wise/...fits?size=100,200px

# 3 arcminute square cutout
https://irsa.ipac.caltech.edu/ibe/data/wise/...fits?size=3arcmin

# With explicit center
https://irsa.ipac.caltech.edu/ibe/data/wise/...fits?center=70,20&size=200pix
```

## Implementation Design

### 1. Configuration Changes

#### QueryConfig (config.py)
Added new optional parameters:
```python
@dataclass
class QueryConfig:
    # ... existing fields ...
    cutout_size: Optional[str] = None  # e.g., "200px", "100,200px", "3arcmin", "0.1"
    cutout_center: Optional[str] = None  # e.g., "70,20", "300.5,120px" (optional, defaults to source position)
```

**Design Decisions:**
- Use string format to preserve user input exactly as IRSA expects
- Make both parameters optional (None = download full image)
- Center defaults to None, automatically set to source RA/Dec when size is specified
- Store as strings for JSON serialization compatibility

#### PipelineState (config.py)
Updated serialization methods:
- `to_dict()`: Include cutout_size and cutout_center in config dictionary
- `from_dict()`: Restore cutout parameters from saved state

### 2. Helper Functions

#### Cutout Utilities (utils/helpers.py)
New functions:
```python
def validate_cutout_size(size_str: str) -> bool:
    """Validate cutout size parameter format."""

def validate_cutout_center(center_str: str) -> bool:
    """Validate cutout center parameter format."""

def format_cutout_url_params(cutout_size: Optional[str],
                              cutout_center: Optional[str],
                              source_ra: float,
                              source_dec: float) -> str:
    """
    Format cutout parameters as URL query string.

    Returns empty string if no cutout requested.
    Returns "?size=..." if only size specified (uses source position).
    Returns "?center=...&size=..." if center explicitly provided.
    """

def estimate_cutout_size_mb(cutout_size: Optional[str],
                             full_size_mb: float) -> float:
    """
    Estimate cutout file size based on pixel dimensions.

    Returns full_size_mb if cutout_size is None or cannot be estimated.
    """
```

### 3. URL Resolution Changes

#### query.py Modifications
Modified `_resolve_single_url()` function:
```python
def _resolve_single_url(obs: ObservationInfo,
                       cutout_size: Optional[str] = None,
                       cutout_center: Optional[str] = None,
                       source_ra: float = None,
                       source_dec: float = None) -> Tuple[ObservationInfo, Optional[str]]:
    """
    Resolve datalink URL to actual download URL with optional cutout parameters.
    """
    # ... existing datalink resolution ...

    # Append cutout parameters if specified
    if cutout_size:
        cutout_params = format_cutout_url_params(cutout_size, cutout_center, source_ra, source_dec)
        download_url = download_url + cutout_params

    return (obs, download_url)
```

Modified `get_download_urls()` function:
```python
def get_download_urls(
    query_results: QueryResults,
    max_workers: int = 8,
    show_progress: bool = True,
    cache_file: Optional[Path] = None,
    cutout_size: Optional[str] = None,  # NEW
    cutout_center: Optional[str] = None  # NEW
) -> List[Tuple[ObservationInfo, str]]:
    """
    Process datalink URLs with optional cutout parameters.
    """
    # Pass cutout parameters to _resolve_single_url
```

**Note on URL Caching:**
- Cache file now depends on cutout parameters
- Different cutout sizes should use different cache files
- Cache filename could include cutout hash to avoid conflicts

### 4. Pipeline Integration

#### pipeline.py Modifications
Updated to pass cutout parameters through the pipeline:
```python
class SPXQueryPipeline:
    def run_download(self):
        # Pass config.cutout_size and config.cutout_center to get_download_urls()
```

## Files Modified

### Core Changes
1. **spxquery/core/config.py**
   - Add `cutout_size` and `cutout_center` to `QueryConfig`
   - Update `PipelineState.to_dict()` serialization
   - Update `PipelineState.from_dict()` deserialization
   - Update `QueryConfig.__post_init__()` validation

2. **spxquery/core/query.py**
   - Modify `_resolve_single_url()` to accept cutout parameters
   - Modify `get_download_urls()` to accept and pass cutout parameters
   - Import cutout formatting helpers

3. **spxquery/core/pipeline.py**
   - Pass cutout parameters from config to `get_download_urls()`
   - Update cache filename logic to handle cutout variations

4. **spxquery/utils/helpers.py**
   - Add `validate_cutout_size()`
   - Add `validate_cutout_center()`
   - Add `format_cutout_url_params()`
   - Add `estimate_cutout_size_mb()`

### Documentation Changes
5. **spxquery/README.md**
   - Add cutout usage examples
   - Document cutout parameters
   - Update quick start with cutout example

6. **CLAUDE.md** (root level)
   - Add cutout feature to SPXQuery section
   - Document IRSA cutout service integration

## Edge Cases and Error Handling

### 1. Invalid Size Parameters
**Issue:** User provides malformed size string
**Handling:**
- Validate in `QueryConfig.__post_init__()`
- Raise `ValueError` with clear message
- Example regex: `r"^\d+(\.\d+)?(,\d+(\.\d+)?)?(px|pix|pixels|arcsec|arcmin|deg|rad)?$"`

### 2. Invalid Center Parameters
**Issue:** User provides malformed center string
**Handling:**
- Validate format and Dec range [-90, 90]
- Raise `ValueError` for invalid coordinates
- Warn if RA/Dec far from source position

### 3. Cutout Larger Than Image
**Issue:** Requested cutout exceeds image dimensions (2040×2040 pixels)
**Handling:**
- IRSA will return full image automatically
- Log warning to user
- No error thrown (graceful degradation)

### 4. Cutout at Image Edge
**Issue:** Cutout center near edge may result in partial cutout
**Handling:**
- IRSA handles automatically (returns available pixels)
- No special handling needed
- Document behavior in README

### 5. Angular Size Conversion
**Issue:** Angular size depends on pixel scale, varies by field
**Handling:**
- User responsible for appropriate size selection
- Document SPHEREx pixel scale (~6.2 arcsec/pixel)
- Provide conversion helper if needed

### 6. URL Caching with Cutouts
**Issue:** Cached URLs without cutout params used incorrectly
**Handling:**
- Include cutout params in cache filename
- Example: `download_urls_200px.json` vs `download_urls_full.json`
- Clear cache when cutout parameters change

### 7. Source Outside Image Bounds
**Issue:** Source position not in image, cutout fails
**Handling:**
- IRSA will return 404 error
- Download will fail gracefully (captured in DownloadResult)
- Log error, continue with other observations

## JSON Structure for State Persistence

### PipelineState JSON with Cutout Parameters
```json
{
  "stage": "download",
  "config": {
    "source": {
      "ra": 304.693508808,
      "dec": 42.4436872991,
      "name": "Variable_Star"
    },
    "output_dir": "/path/to/output",
    "bands": ["D1", "D2", "D3"],
    "aperture_diameter": 3.0,
    "max_download_workers": 4,
    "max_processing_workers": 10,
    "cutout_size": "200px",
    "cutout_center": null
  },
  "query_results": { ... },
  "downloaded_files": [],
  "photometry_results": [],
  "csv_path": null,
  "plot_path": null
}
```

### URL Cache JSON with Cutouts
Filename convention: `download_urls_{cutout_hash}.json`
```json
{
  "_metadata": {
    "cutout_size": "200px",
    "cutout_center": null,
    "created": "2025-10-15T10:30:00"
  },
  "obs_12345": "https://irsa.ipac.caltech.edu/ibe/data/spherex/.../file.fits?size=200px",
  "obs_12346": "https://irsa.ipac.caltech.edu/ibe/data/spherex/.../file.fits?size=200px"
}
```

## Usage Examples

### Example 1: 200-pixel Square Cutout
```python
from spxquery import Source, QueryConfig, SPXQueryPipeline

source = Source(ra=304.69, dec=42.44, name="My_Star")
config = QueryConfig(
    source=source,
    output_dir="output",
    cutout_size="200px"  # 200x200 pixel cutout
)

pipeline = SPXQueryPipeline(config)
pipeline.run_full_pipeline()
```

### Example 2: Rectangular Cutout in Arcminutes
```python
config = QueryConfig(
    source=source,
    output_dir="output",
    cutout_size="5,10arcmin"  # 5x10 arcminute cutout
)
```

### Example 3: Cutout with Custom Center
```python
config = QueryConfig(
    source=source,
    output_dir="output",
    cutout_size="3arcmin",
    cutout_center="304.7,42.5"  # RA, Dec in degrees
)
```

### Example 4: Full Image (No Cutout)
```python
config = QueryConfig(
    source=source,
    output_dir="output"
    # cutout_size=None means download full images
)
```

## Performance Impact

### Storage Savings
- Full image: ~70 MB per observation
- 200px cutout: ~0.7 MB per observation (99% reduction)
- 500px cutout: ~4.3 MB per observation (94% reduction)

### Download Time Savings
Proportional to file size reduction.

### Processing Impact
- Faster FITS I/O
- Faster photometry (smaller arrays)
- No impact on photometry accuracy (same data)

## Testing Plan

### Unit Tests
1. Test `validate_cutout_size()` with valid/invalid inputs
2. Test `validate_cutout_center()` with valid/invalid inputs
3. Test `format_cutout_url_params()` output format
4. Test `estimate_cutout_size_mb()` calculations

### Integration Tests
1. Test full pipeline with cutout parameters
2. Test state serialization/deserialization with cutouts
3. Test URL caching with different cutout sizes
4. Test graceful handling of oversized cutouts

### Manual Testing
1. Download cutout and verify dimensions
2. Verify photometry results match full image
3. Test with various size formats (px, arcmin, deg)
4. Test edge cases (source near edge, outside image)

## Future Enhancements

### Potential Improvements
1. **Automatic Size Selection**: Choose cutout size based on aperture diameter
2. **Multi-Center Cutouts**: Support multiple sources in batch mode
3. **Cutout Preview**: Show cutout region on sky plot before download
4. **Adaptive Sizing**: Adjust size based on source extent/morphology
5. **Cutout Validation**: Pre-check if source is within image bounds

### Related Features
1. **Mosaic Support**: Combine cutouts from multiple overlapping observations
2. **Background Estimation**: Use cutout edges for local background
3. **PSF Extraction**: Ensure PSF HDU covers cutout region

## References

- IRSA Cutout Service Documentation: https://irsa.ipac.caltech.edu/ibe/cutouts.html
- SPHEREx Data Products: https://irsa.ipac.caltech.edu/data/SPHEREx/
- SPHEREx Pixel Scale: ~6.2 arcsec/pixel

## Change Log

### 2025-10-15: Initial Implementation
- Added cutout_size and cutout_center to QueryConfig
- Implemented URL parameter formatting
- Updated pipeline to support cutouts
- Created comprehensive documentation
- Added validation and error handling

---

**Implementation Status:**  Planned, =§ In Progress,  Complete

- [=§] Config changes (config.py)
- [  ] Helper functions (utils/helpers.py)
- [  ] Query modifications (query.py)
- [  ] Pipeline integration (pipeline.py)
- [  ] Documentation updates (README.md, CLAUDE.md)
- [  ] Testing and validation
