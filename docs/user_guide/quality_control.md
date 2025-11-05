# Quality Control

SPXQuery applies quality control filtering to identify reliable photometric measurements and flag problematic data.

## Overview

Quality control operates on two criteria:

1. **Signal-to-Noise Ratio (SNR)** - Filters low-significance detections
2. **Pixel Flags** - Rejects measurements affected by instrumental or processing issues

**Important:** Quality filtering applies **only to visualization**. All measurements are saved to the CSV file, allowing users to apply custom filtering for their analysis.

## Signal-to-Noise Ratio (SNR)

### Definition

SNR is computed as:

```
SNR = flux / flux_error
```

Where:
- `flux` is the aperture-corrected flux (MJy/sr)
- `flux_error` is the combined uncertainty from photon noise and background variance

### SNR Threshold

The `sigma_threshold` parameter sets the minimum SNR for "good" measurements:

```python
from spxquery.core.pipeline import run_pipeline

run_pipeline(
    ra=304.69,
    dec=42.44,
    output_dir="output",
    sigma_threshold=5.0  # Default: 5 sigma detection threshold
)
```

**Typical values:**
- **3.0** - Marginal detections (relaxed)
- **5.0** - Standard detection threshold (default, recommended)
- **10.0** - High-confidence detections only (strict)

### Effect on Visualization

In the combined plot:
- **Good measurements** (SNR ≥ threshold): Filled circles, colored by wavelength/date
- **Rejected measurements** (SNR < threshold): Gray crosses (×)

This allows you to see both the reliable measurements and the rejected data points for context.

## Pixel Flags

### SPHEREx Flag System

The SPHEREx FLAGS extension uses a bitmap where each bit represents a different quality issue. Multiple flags can be set for a single pixel.

### Default Bad Flags

SPXQuery uses this default set of bad pixel flags:

```python
bad_flags = [0, 1, 2, 6, 7, 9, 10, 11, 15]
```

**Flag definitions:**

| Bit | Flag Name | Description |
|-----|-----------|-------------|
| 0 | TRANSIENT | Transient event detected (cosmic ray, etc.) |
| 1 | OVERFLOW | Pixel overflow/saturation |
| 2 | SUR_ERROR | Sample-up-the-ramp error |
| 6 | NONFUNC | Non-functional pixel |
| 7 | DICHROIC | Dichroic reflection artifact |
| 9 | MISSING_DATA | Missing data |
| 10 | HOT | Hot pixel |
| 11 | COLD | Cold pixel |
| 15 | NONLINEAR | Non-linear response |

### Other Available Flags

SPHEREx provides additional flags that are **not** rejected by default:

| Bit | Flag Name | Description | Why Not Default |
|-----|-----------|-------------|-----------------|
| 4 | PHANTOM | Phantom source detected | May not affect photometry |
| 5 | REFERENCE | Reference pixel | Informational |
| 12 | FULLSAMPLE | Full sample available | Quality indicator, not rejection criterion |
| 14 | PHANMISS | Phantom or missing | Overlap with bits 0, 9 |
| 17 | PERSIST | Detector persistence | Low impact for most sources |
| 19 | OUTLIER | Statistical outlier | May be real variability |
| 21 | SOURCE | Source detected | Informational |

### Customizing Bad Flags

**Relaxed filtering** (accept more data):
```python
run_pipeline(
    ra=304.69,
    dec=42.44,
    output_dir="output",
    bad_flags=[0, 1, 2]  # Only reject saturated/bad pixels
)
```

**Strict filtering** (reject more):
```python
run_pipeline(
    ra=304.69,
    dec=42.44,
    output_dir="output",
    bad_flags=[0, 1, 2, 4, 6, 7, 9, 10, 11, 14, 15, 17]  # Add PHANTOM, PHANMISS, PERSIST
)
```

**No flag filtering** (use all data):
```python
run_pipeline(
    ra=304.69,
    dec=42.44,
    output_dir="output",
    bad_flags=[]  # Accept all flags
)
```

### How Flag Filtering Works

The FLAGS extension in SPHEREx FITS files contains integer values where each bit represents a flag. A pixel is rejected if **any** of the specified flag bits are set.

**Example:**
```
pixel_flag = 2097152  # Binary: 1000000000000000000000 (bit 21 set)
bad_flags = [0, 1, 2]

# Check if any bad flags are set
for bit in bad_flags:
    if pixel_flag & (1 << bit):
        reject_pixel()  # Reject if bit is set

# Result: Not rejected (bit 21 is not in bad_flags)
```

## Quality Assessment Workflow

### 1. Check Distribution

After running the pipeline, examine the light curve CSV to assess quality:

```python
import pandas as pd

df = pd.read_csv("output/results/lightcurve.csv", comment="#")

# Check SNR distribution
print("SNR statistics:")
print(df['snr'].describe())

# Check flag distribution
print("\nFlag counts:")
print(df['flag'].value_counts())
```

### 2. Identify Patterns

Look for systematic issues:

```python
# Identify low-SNR measurements
low_snr = df[df['snr'] < 5.0]
print(f"Low SNR: {len(low_snr)} / {len(df)} ({100*len(low_snr)/len(df):.1f}%)")

# Check which flags are most common
import numpy as np

def decode_flags(flag_value):
    """Extract which bits are set."""
    return [bit for bit in range(32) if flag_value & (1 << bit)]

# Get all set flags across dataset
all_flags = []
for flag in df['flag']:
    all_flags.extend(decode_flags(flag))

flag_counts = pd.Series(all_flags).value_counts()
print("\nMost common flag bits:")
print(flag_counts.head(10))
```

### 3. Adjust Filtering

Based on the assessment, adjust quality control parameters:

```python
# If too few good measurements, relax threshold
run_pipeline(..., sigma_threshold=3.0)

# If specific flag is problematic, add to bad_flags
run_pipeline(..., bad_flags=[0, 1, 2, 6, 7, 9, 10, 11, 15, 17])
```

## Visualization Quality Indicators

### Combined Plot

The visualization shows three types of data points:

1. **Good measurements** (filled circles)
   - SNR ≥ `sigma_threshold`
   - No bad pixel flags set
   - Colored by wavelength (left panel) or date (right panel)

2. **Rejected measurements** (gray crosses ×)
   - SNR < `sigma_threshold` OR
   - Bad pixel flags set
   - Shown for context but not used in trend analysis

3. **Upper limits** (downward arrows, if applicable)
   - Non-detections (negative flux or SNR < threshold)
   - Plotted at 3σ upper limit

### Interpreting the Plot

**High rejection rate:**
- Many gray crosses → adjust `sigma_threshold` or `bad_flags`
- Check if source is too faint for aperture size

**Clustered rejections:**
- Rejections at specific wavelengths → instrumental issue
- Rejections at specific dates → transient contamination

**No rejections:**
- All measurements pass quality control
- May indicate overly relaxed filtering

## CSV Output Format

The light curve CSV contains all measurements with quality flags:

```csv
obs_id,mjd,flux,flux_error,wavelength,bandwidth,band,flag,snr,is_upper_limit
2025W25_1B_0062_1,60842.269794,1007.005,43.199,1.940,0.048,D3,2097152,23.3,False
...
```

**Quality-related columns:**
- `flag` - Integer bitmap of pixel flags
- `snr` - Signal-to-noise ratio (flux / flux_error)
- `is_upper_limit` - Boolean indicating non-detection

Users can apply custom filtering:

```python
import pandas as pd

df = pd.read_csv("output/results/lightcurve.csv", comment="#")

# Custom filtering
good = df[(df['snr'] >= 5.0) & (df['flag'] == 0)]

# Or more complex criteria
def has_bad_flags(flag_value, bad_flags=[0, 1, 2]):
    return any(flag_value & (1 << bit) for bit in bad_flags)

df['is_good'] = (df['snr'] >= 5.0) & ~df['flag'].apply(has_bad_flags)
good = df[df['is_good']]
```

<!-- ## Best Practices

### Initial Analysis

1. **Use defaults first**: `sigma_threshold=5.0`, default `bad_flags`
2. **Examine the plot**: Check rejection patterns
3. **Review CSV**: Assess SNR and flag distributions

### Adjusting Thresholds

1. **For faint sources**: Lower `sigma_threshold` to 3.0
2. **For bright sources**: Raise `sigma_threshold` to 10.0 for cleaner sample
3. **For variability studies**: Use stricter filtering to avoid false variations

### Flag Selection

1. **Start with defaults**: Good balance for most cases
2. **Add flags cautiously**: Adding flags reduces sample size
3. **Remove flags carefully**: Removing flags may include bad data
4. **Document choices**: Record your quality criteria in analysis notes

### Common Scenarios

**Faint variable source:**
```python
sigma_threshold=3.0  # Accept marginal detections
bad_flags=[0, 1, 2, 6, 7, 9]  # Remove hot/cold pixel flags
```

**Bright stable source:**
```python
sigma_threshold=10.0  # High-confidence only
bad_flags=[0, 1, 2, 6, 7, 9, 10, 11, 15, 17]  # Strict rejection
```

**Crowded field:**
```python
sigma_threshold=5.0
bad_flags=[0, 1, 2, 4, 6, 7, 9, 10, 11, 15]  # Add PHANTOM flag
``` -->

<!-- ## Limitations

### What Quality Control Cannot Fix

- **Contamination**: Nearby sources within aperture
- **Background subtraction errors**: Incorrect zodiacal scaling
- **Astrometric errors**: Source offset from expected position
- **Astrophysical variability**: Real flux changes (not defects)

These issues require manual inspection or custom analysis.

### When to Manually Inspect

Inspect individual FITS files when:
- High rejection rate (>50%)
- Unexpected flux variations
- Systematic trends in rejections
- Light curve shows outliers -->

## See Also

- [Pipeline Architecture](pipeline.md) - How quality control is applied
- [Parameters](parameters.md) - Customizing quality thresholds
- [Tutorial](../tutorials/quickstart_demo.ipynb) - Practical examples
