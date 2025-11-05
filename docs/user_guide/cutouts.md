# Image Cutouts

SPXQuery supports IRSA image cutouts to download only the region of interest instead of full SPHEREx images, providing dramatic storage savings and faster processing.

## Why Use Cutouts?

Full SPHEREx images are 2040×2040 pixels (~70 MB per file). For point source photometry, you typically only need a small region around the source.

**Storage savings example:**

| Cutout Size | File Size | Reduction | Use Case |
|-------------|-----------|-----------|----------|
| Full image | ~70 MB | 0% | Extended sources, large-scale structure |
| 500px square | ~10 MB (minimum) | ~85% | Extended sources with margins |
| 200px square | ~5 MB (minimum) | ~93% | Point sources (recommended) |

For a typical SPHEREx time-series with 100 observations:
- **Full images**: ~7 GB
- **200px cutouts**: ~500 MB (minimum file size ≈5 MB per file; ~14× reduction vs full images)

## Cutout Parameters

### Basic Usage

Specify cutout size in the pipeline configuration:

```python
from spxquery.core.pipeline import run_pipeline

run_pipeline(
    ra=304.69,
    dec=42.44,
    output_dir="output",
    cutout_size="200px"  # Square 200×200 pixel cutout
)
```

### Size Specifications

The `cutout_size` parameter accepts several formats:

**Square cutouts (single value):**
```python
cutout_size="200px"      # 200×200 pixels
cutout_size="3arcmin"    # 3×3 arcminutes
cutout_size="0.1"        # 0.1×0.1 degrees (default unit)
```

**Rectangular cutouts (two values):**
```python
cutout_size="200,400px"  # 200×400 pixels (width × height)
cutout_size="2,5arcmin"  # 2×5 arcminutes
```

**Supported units:**
- `px`, `pix`, `pixels` - Pixels
- `arcsec` - Arcseconds
- `arcmin` - Arcminutes
- `deg` - Degrees
- `rad` - Radians
- No unit (default: degrees)

### SPHEREx Pixel Scale

SPHEREx has a pixel scale of approximately **6.2 arcsec/pixel**.

**Conversion examples:**
- 100 pixels ≈ 620 arcsec ≈ 10.3 arcmin ≈ 0.17 degrees
- 200 pixels ≈ 1240 arcsec ≈ 20.7 arcmin ≈ 0.34 degrees
- 500 pixels ≈ 3100 arcsec ≈ 51.7 arcmin ≈ 0.86 degrees

## Cutout Centering

By default, cutouts are centered on the source coordinates (RA/Dec). You can specify a custom center:

```python
from spxquery import Source, QueryConfig

source = Source(ra=304.69, dec=42.44, name="my_source")
config = QueryConfig(
    source=source,
    output_dir="output",
    cutout_size="200px",
    cutout_center="304.7,42.5"  # Custom center (RA, Dec in degrees)
)
```

**Custom center formats:**
```python
cutout_center="304.7,42.5"        # RA/Dec in degrees (default)
cutout_center="304.7,42.5deg"     # Explicit degrees
cutout_center="1020,1020px"       # Pixel coordinates
cutout_center="10.3,5.2arcmin"    # Arcminutes from reference
```

<!-- ## Recommended Cutout Sizes

### Point Sources

For isolated point sources (stars, quasars, unresolved AGN):

```python
cutout_size="200px"  # Recommended
```

**Rationale:**
- Provides ~100 pixels radius around source
- Sufficient for aperture photometry with background annulus
- Includes margin for astrometric uncertainties
- ~93% storage reduction vs. full images (subject to minimum file-size limit)

### Extended Sources

For galaxies or extended emission:

```python
cutout_size="500px"  # For sources up to ~1 arcmin extent
cutout_size="1000px"  # For larger extended sources
```

### Crowded Fields

For sources in crowded regions (stellar clusters, galactic plane):

```python
cutout_size="400px"  # Larger margin to assess contamination
``` -->

## How Cutouts Work

### Behind the Scenes

1. **Query stage**: Pipeline queries IRSA for full image URLs
2. **Cutout parameter addition**: Pipeline appends cutout parameters to datalink URLs:
   ```
   https://irsa.ipac.caltech.edu/...?POS=304.69,42.44&SIZE=200,200&...
   ```
3. **Server-side processing**: IRSA generates cutout on-the-fly
4. **Download stage**: Pipeline downloads only the cutout region

### File Format Preservation

Cutout FITS files preserve the full Multi-Extension FITS (MEF) structure:
- **IMAGE**: Cutout region with updated WCS
- **FLAGS**: Corresponding pixel flags
- **VARIANCE**: Variance estimates for cutout
- **ZODI**: Zodiacal model for cutout region
- **PSF**: Full PSF cube (not cutout)
- **WCS-WAVE**: Full wavelength lookup table

The World Coordinate System (WCS) is automatically adjusted to reflect the cutout region.

## Cutout Validation

SPXQuery validates cutout parameters before querying:

```python
# Valid
cutout_size="200px"          # ✓
cutout_size="3arcmin"        # ✓
cutout_size="0.1"            # ✓
cutout_size="100,200px"      # ✓

# Invalid (raises ValueError)
cutout_size="0px"            # ✗ Zero size
cutout_size="-100px"         # ✗ Negative size
cutout_size="2040,2040px"    # ⚠ Warning: same as full image
cutout_size="5000px"         # ✗ Exceeds detector size
```

## Limitations

### Maximum Size

Cutouts cannot exceed the detector size (2040×2040 pixels):

```python
cutout_size="2040px"  # Maximum (but defeats purpose of cutouts)
```

If you need the full field of view, omit the `cutout_size` parameter:

```python
run_pipeline(
    ra=304.69,
    dec=42.44,
    output_dir="output"
    # No cutout_size = download full images
)
```

<!-- ### Cutout Availability

Not all archives support cutouts. IRSA SPHEREx archive supports cutouts for all SPHEREx data products. -->

### Edge Cases

If the source is near the detector edge, the cutout may be smaller than requested:
- Pipeline logs a warning
- Processing continues with available data
- Quality control flags identify affected observations

## Performance Impact

<!-- ### Download Time

Approximate download times (100 Mbps connection):

| Cutout Size | Time per File | 100 Files Total |
|-------------|---------------|-----------------|
| Full image (70 MB) | ~6 sec | ~10 minutes |
| 500px (~5 MB, minimum) | ~0.4 sec | ~40 seconds |
| 200px (~5 MB, minimum) | ~0.4 sec | ~40 seconds |
| 100px (~5 MB, minimum) | ~0.4 sec | ~40 seconds | -->

### Processing Time

Cutouts also reduce processing time:

- Smaller FITS files load faster
- Fewer pixels for photometry operations
- Reduced memory footprint

**Typical speedup:** 5-10× faster for 200px cutouts vs. full images

## Storage Management

### File Organization

Downloaded files are organized by band:

```
output/
├── data/
│   ├── D1/
│   │   ├── observation1_cutout.fits
│   │   └── observation2_cutout.fits
│   ├── D2/
│   │   └── ...
│   └── ...
└── results/
    └── query_summary.json  # Contains actual file sizes
```

### Checking Storage Usage

After download, check the query summary:

```python
import json

with open("output/results/query_summary.json") as f:
    summary = json.load(f)
    print(f"Total size: {summary['total_size_gb']:.2f} GB")
    print(f"Band sizes: {summary['band_sizes']}")
```

<!-- ## Examples

### Minimal Storage (Point Source)

```python
run_pipeline(
    ra=213.94,
    dec=11.50,
    output_dir="output",
    cutout_size="100px",  # Minimal size for point source
    aperture_diameter=2.0
)
```

### Balanced (Standard Point Source)

```python
run_pipeline(
    ra=213.94,
    dec=11.50,
    output_dir="output",
    cutout_size="200px",  # Recommended default
    aperture_diameter=3.0
)
```

### Extended Source

```python
run_pipeline(
    ra=213.94,
    dec=11.50,
    output_dir="output",
    cutout_size="500px",  # Larger margin for extended emission
    aperture_diameter=5.0
)
```

### Custom Rectangle

```python
run_pipeline(
    ra=213.94,
    dec=11.50,
    output_dir="output",
    cutout_size="300,600px",  # Elongated cutout (e.g., for jets)
    aperture_diameter=3.0
)
``` -->
<!-- 
## Best Practices

1. **Start with 200px** - Good default for most point sources
2. **Check first observation** - Verify cutout contains your source
3. **Consider source extent** - Use larger cutouts for extended sources
4. **Account for astrometry** - Include margin for coordinate uncertainties (~10-20 pixels)
5. **Monitor logs** - Check for edge-case warnings during download
6. **Test before bulk runs** - Download a few observations first to verify cutout size -->

## See Also

- [Pipeline Architecture](pipeline.md) - How cutouts are applied in the download stage
- [Quality Control](quality_control.md) - Identifying affected observations
- [Parameters](parameters.md) - Other configuration options
