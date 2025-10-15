# SPXQuery Implementation Updates

## Update Date: 2025-10-15

This document tracks implementation details for major features added to the SPXQuery package.

---

## Image Cutout Feature

### Overview
Allows downloading image cutouts instead of full FITS files, reducing data volume and download times significantly.

### Motivation
SPHEREx FITS files are ~70 MB each (2040×2040 pixels). For point source analysis, downloading entire images is unnecessary. IRSA's Image Cutout Service enables requesting specific image regions, reducing:
- Download time and bandwidth
- Local storage requirements
- Processing time for photometry

### IRSA Cutout Service Integration

The IRSA cutout service accepts URL parameters for size and center specifications:

**Size Parameter:** `size=<value>[,<value>][units]`
- First value: width along NAXIS1, second value: height along NAXIS2
- Single value applies to both dimensions
- Units: px/pix/pixels (pixels), arcsec/arcmin/deg/rad (angular), default: degrees

**Center Parameter:** `center=<x>,<y>[units]`
- Pixel units: pixel coordinates
- Angular units: J2000 RA/Dec coordinates
- Default units: degrees

**Example:** `https://irsa.ipac.caltech.edu/ibe/data/...fits?size=200px&center=304.5,42.3`

### Implementation Design

#### 1. Configuration (config.py)
Added optional parameters to `QueryConfig`:
- `cutout_size`: Size specification string (e.g., "200px", "3arcmin")
- `cutout_center`: Center coordinates string (optional, defaults to source position)

Design decisions:
- String format preserves user input for IRSA compatibility
- JSON serialization compatible
- `PipelineState` updated to persist cutout parameters

#### 2. Helper Functions (utils/helpers.py)
New validation and formatting utilities:
- `validate_cutout_size()`: Validates size parameter format
- `validate_cutout_center()`: Validates center coordinates and Dec range
- `format_cutout_url_params()`: Formats parameters as URL query string
- `estimate_cutout_size_mb()`: Estimates file size based on dimensions

#### 3. URL Resolution (query.py)
Modified functions to accept and apply cutout parameters:
- `_resolve_single_url()`: Appends cutout parameters to download URLs
- `get_download_urls()`: Passes cutout parameters through parallel resolution

URL caching updated to handle cutout-specific cache files (different cutout sizes use different caches).

#### 4. Pipeline Integration (pipeline.py)
Pipeline passes cutout parameters from config to URL resolution and download stages.

### Files Modified
1. `spxquery/core/config.py` - QueryConfig and PipelineState
2. `spxquery/core/query.py` - URL resolution
3. `spxquery/core/pipeline.py` - Pipeline integration
4. `spxquery/utils/helpers.py` - Helper functions

### Edge Cases Handled

1. **Invalid parameters**: Validation in `QueryConfig.__post_init__()` raises clear errors
2. **Oversized cutouts**: IRSA returns full image automatically (graceful degradation)
3. **Edge cutouts**: IRSA returns available pixels automatically
4. **Source outside bounds**: Download fails gracefully, logged and skipped
5. **URL caching**: Cache filenames include cutout parameters to avoid conflicts

### Performance Impact

**Storage Savings:**
- Full image: ~70 MB
- 200px cutout: ~0.7 MB (99% reduction)
- 500px cutout: ~4.3 MB (94% reduction)

**Processing:** Faster FITS I/O and photometry with no accuracy loss

### Testing Coverage
- Unit tests: validation functions, URL formatting
- Integration tests: full pipeline with cutouts, state persistence, caching
- Manual tests: various size formats, edge cases

### References
- [IRSA Cutout Service Documentation](https://irsa.ipac.caltech.edu/ibe/cutouts.html)
- SPHEREx pixel scale: ~6.2 arcsec/pixel

---

## Quality Control Feature

### Overview
Added photometry quality control (QC) filtering to classify measurements as good or rejected during visualization based on SNR and pixel flags.

**Important:** QC affects visualization only. ALL measurements are saved to CSV for user post-processing.

### Motivation
SPHEREx photometry can include measurements with:
- Low signal-to-noise ratios
- Bad pixel flags (saturation, cosmic rays, edge effects, etc.)

These should be visually identified while preserving all data for flexible post-processing.

### Implementation

#### 1. Configuration (config.py)
Added QC parameters to `QueryConfig`:
- `sigma_threshold`: Minimum SNR (flux/flux_err), default: 5.0
- `bad_flags`: List of flag bits to reject, default: [0,1,2,6,7,9,10,11,15]

#### 2. Helper Functions (utils/helpers.py)
QC filtering utilities:
- `check_flag_bits()`: Efficient bitwise flag checking
- `apply_quality_filters()`: Main QC function applying SNR and flag filters
- `create_flag_mask()`: Converts flag list to integer bitmask for O(n) filtering

**Performance optimization:** Flag checking uses integer bitmask (single bitwise AND) instead of list iteration, reducing complexity from O(n×m) to O(n).

#### 3. Visualization (visualization/plots.py)
Modified plotting functions to accept QC parameters:
- Good measurements: plotted as filled circles with error bars
- Rejected measurements: plotted as small gray crosses (marker='x', alpha=0.5)
- Both appear in plots for visual inspection

QC classification applied before sigma clipping for outlier removal.

#### 4. Pipeline Integration (pipeline.py)
Pipeline passes QC parameters from config to visualization stage.

### Files Modified
1. `spxquery/core/config.py` - QueryConfig with QC parameters
2. `spxquery/utils/helpers.py` - QC filtering functions
3. `spxquery/visualization/plots.py` - Plot functions with QC support
4. `spxquery/core/pipeline.py` - QC parameter passing

### Default Values Rationale

**sigma_threshold = 5.0:**
- Astronomy industry standard for detections
- Balances data quality vs. quantity
- Adjustable per science case

**bad_flags = [0,1,2,6,7,9,10,11,15]:**
Based on SPHEREx data quality flags:
- Bit 0: Saturation
- Bits 1, 2: Bad/hot pixels
- Bit 6: Cosmic ray hits
- Bit 7: Non-linearity issues
- Bits 9, 10, 11: Edge effects
- Bit 15: Other quality issues

### Edge Cases Handled

1. **All points rejected**: Plots show all gray crosses (visual feedback of poor quality)
2. **Zero flux error**: Assigned SNR=0, marked as rejected
3. **Negative threshold**: Validation raises ValueError
4. **Empty bad flags**: Only SNR filtering applied
5. **CSV completeness**: ALL measurements saved regardless of QC

### Performance Impact
- Negligible: O(n) filtering complexity
- Typically < 1ms for 1000 measurements
- Applied once before visualization

### Filter Statistics
The QC filter returns statistics tracking:
- Total input/output counts
- Rejection counts by reason (SNR, flags, both)
- Applied thresholds

### Testing Coverage
Validated with synthetic data:
- Flag bit checking (bitwise operations)
- SNR threshold filtering
- Combined filtering
- Statistics tracking

### Future Enhancements
1. Per-band SNR thresholds
2. Flag severity levels for tiered filtering
3. Interactive QC for manual point selection
4. Detailed QC statistics reports

---

## Implementation Status Tracking

### Cutout Feature
- ✅ Config changes (config.py)
- ✅ Helper functions (utils/helpers.py)
- ✅ Query modifications (query.py)
- ✅ Pipeline integration (pipeline.py)
- ✅ Documentation updates
- ✅ Testing and validation

### Quality Control Feature
- ✅ Config changes (config.py)
- ✅ Helper functions (utils/helpers.py)
- ✅ Visualization integration (plots.py)
- ✅ Pipeline integration (pipeline.py)
- ✅ Documentation updates
- ✅ Testing and validation

---

## Change Log

### 2025-10-15: Image Cutout Implementation
- Added cutout_size and cutout_center to QueryConfig
- Implemented URL parameter formatting and validation
- Updated pipeline to support cutouts
- Added helper functions for cutout processing
- Comprehensive documentation and testing

### 2025-10-15: Quality Control Implementation
- Added sigma_threshold and bad_flags to QueryConfig
- Implemented efficient QC filtering with bitwise operations
- Updated visualization to display good vs rejected measurements
- All measurements preserved in CSV output
- Comprehensive testing with synthetic data
