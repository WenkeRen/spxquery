# Parameter Configuration

SPXQuery uses a flexible parameter system that separates basic pipeline parameters from advanced configuration options.

## Parameter Organization

Parameters are organized into two categories:

### Basic Parameters

These are commonly used pipeline parameters passed directly to functions:

- **Source parameters**: `ra`, `dec`, `source_name`
- **Pipeline control**: `output_dir`, `bands`, `aperture_diameter`
- **Download control**: `cutout_size`, `max_download_workers`, `skip_existing_downloads`
- **Quality control**: `sigma_threshold`, `bad_flags`
- **Visualization**: `use_magnitude`, `show_errorbars`

Example:

```python
from spxquery.core.pipeline import run_pipeline

run_pipeline(
    ra=213.94,
    dec=11.50,
    output_dir="output",
    cutout_size="200px",
    aperture_diameter=3.0,
    sigma_threshold=5.0
)
```

### Advanced Parameters

These are less frequently modified parameters organized into three configuration classes:

1. **PhotometryConfig** - Aperture photometry and background estimation
2. **VisualizationConfig** - Plot appearance and layout
3. **DownloadConfig** - HTTP download behavior

Advanced parameters are configured via JSON files (see [JSON Configuration](#json-configuration) below).

## Three-Tier Priority System

SPXQuery uses a priority hierarchy to determine parameter values:

### Priority Order (Highest → Lowest)

1. **Explicit function arguments** - Parameters passed directly
2. **JSON configuration file** - Loaded via `advanced_params_file`
3. **Built-in defaults** - Default values in the code

### Example

```python
# Scenario 1: Use defaults
config = QueryConfig(source=source, output_dir="output")
# Result: Uses all built-in defaults

# Scenario 2: Override with JSON
config = QueryConfig(
    source=source,
    output_dir="output",
    advanced_params_file="my_params.json"  # Contains custom values
)
# Result: Uses JSON values, falls back to defaults if not specified

# Scenario 3: Explicit override
config = QueryConfig(
    source=source,
    output_dir="output",
    advanced_params_file="my_params.json",
    aperture_diameter=5.0  # Explicit override
)
# Result: Uses aperture_diameter=5.0, ignores JSON value for this parameter
```

## Configuration Classes

### PhotometryConfig

Controls aperture photometry and background estimation.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `annulus_inner_offset` | 1.414 | Gap between aperture and background annulus (in aperture radii) |
| `min_annulus_area` | 10 | Minimum usable pixels in background annulus |
| `bg_sigma_clip_sigma` | 3.0 | Sigma threshold for background sigma clipping |
| `bg_sigma_clip_maxiters` | 3 | Maximum iterations for sigma clipping |
| `pixel_scale_fallback` | 6.2 | Fallback pixel scale (arcsec/pixel) when WCS unavailable |
| `max_annulus_attempts` | 5 | Maximum attempts to find valid background annulus |
| `annulus_expansion_step` | 0.5 | Annulus expansion step (in aperture radii) |

**When to customize:**

- Non-standard aperture geometries
- Sources in crowded fields (adjust annulus sizing)
- Stricter/looser background estimation

### VisualizationConfig

Controls plot appearance and layout.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wavelength_cmap` | "rainbow" | Colormap for wavelength coding |
| `date_cmap` | "viridis" | Colormap for observation date coding |
| `figsize` | (10, 8) | Figure dimensions (width, height) in inches |
| `dpi` | 150 | Resolution for saved figures |
| `marker_size_good` | 1.5 | Marker size for good measurements |
| `marker_size_rejected` | 2.0 | Marker size for rejected measurements |
| `marker_alpha` | 0.9 | Marker transparency (0-1) |
| `errorbar_alpha` | 0.2 | Error bar transparency (0-1) |

**When to customize:**

- Publication requirements (DPI, figure size)
- Color vision accessibility (different colormaps)
- Visual emphasis (marker sizes, transparency)

### DownloadConfig

Controls HTTP download behavior.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 8192 | Download chunk size (bytes) |
| `timeout` | 300 | HTTP timeout (seconds) |
| `max_retries` | 3 | Maximum retry attempts on failure |
| `retry_delay` | 5 | Delay between retries (seconds) |
| `user_agent` | "SPXQuery/0.2.0" | HTTP User-Agent header |

**When to customize:**

- Slow networks (increase timeout, reduce chunk size)
- Unreliable connections (increase retries/delay)
- Rate limiting issues (adjust retry delay)

## JSON Configuration

### Export Default Template

Create a JSON template with all default values:

```python
from spxquery.utils.params import export_default_parameters

params_file = export_default_parameters(
    output_dir="config",
    filename="my_params.json"
)
```

This creates a file like:

```json
{
  "photometry": {
    "annulus_inner_offset": 1.414,
    "min_annulus_area": 10,
    "bg_sigma_clip_sigma": 3.0,
    ...
  },
  "visualization": {
    "wavelength_cmap": "rainbow",
    "date_cmap": "viridis",
    "figsize": [10, 8],
    "dpi": 150,
    ...
  },
  "download": {
    "chunk_size": 8192,
    "timeout": 300,
    "max_retries": 3,
    ...
  }
}
```

### Customize Parameters

Edit the JSON file to change specific values:

```json
{
  "photometry": {
    "bg_sigma_clip_sigma": 2.5
  },
  "visualization": {
    "dpi": 300,
    "figsize": [7.5, 6],
    "wavelength_cmap": "plasma"
  }
}
```

You only need to specify parameters you want to change - omitted parameters use defaults.

### Load Custom Configuration

Use the JSON file in your pipeline:

**Method 1: run_pipeline() function**

```python
from spxquery.core.pipeline import run_pipeline

run_pipeline(
    ra=213.94,
    dec=11.50,
    output_dir="output",
    advanced_params_file="config/my_params.json"
)
```

**Method 2: QueryConfig class**

```python
from spxquery import SPXQueryPipeline, Source, QueryConfig

source = Source(ra=213.94, dec=11.50, name="my_source")
config = QueryConfig(
    source=source,
    output_dir="output",
    advanced_params_file="config/my_params.json"
)

pipeline = SPXQueryPipeline(config)
pipeline.run_full_pipeline()
```

<!-- ## Common Use Cases

### Publication-Quality Plots

```json
{
  "visualization": {
    "dpi": 300,
    "figsize": [7.5, 6],
    "wavelength_cmap": "plasma",
    "marker_size_good": 2.0,
    "errorbar_linewidth": 0.8
  }
}
```

### Stricter Background Estimation

```json
{
  "photometry": {
    "bg_sigma_clip_sigma": 2.5,
    "bg_sigma_clip_maxiters": 5,
    "min_annulus_area": 20
  }
}
```

### Slow Network Optimization

```json
{
  "download": {
    "chunk_size": 4096,
    "timeout": 600,
    "max_retries": 5,
    "retry_delay": 10
  }
}
```

### Color Vision Accessibility

```json
{
  "visualization": {
    "wavelength_cmap": "viridis",
    "date_cmap": "cividis"
  }
}
``` -->

## Parameter Validation

SPXQuery validates parameters at configuration time:

- **Type checking**: Parameters must match expected types
- **Range validation**: Values must be within valid ranges (e.g., DPI > 0)
- **Dependency checking**: Related parameters are checked for consistency

Invalid configurations raise `ValueError` with descriptive error messages.

<!-- ## Best Practices

1. **Start with defaults** - Only customize parameters when needed
2. **Use JSON files** - Keep project-specific settings in version-controlled JSON files
3. **Document changes** - Add comments in JSON (use separate documentation file) explaining why you changed defaults
4. **Test incrementally** - Change one parameter at a time to understand effects
5. **Check logs** - SPXQuery logs which configuration file was loaded and parameter values used -->

## See Also

- [Pipeline Architecture](pipeline.md) - How parameters are used in each stage
- [Quality Control](quality_control.md) - Quality filtering parameters
- [Cutouts](cutouts.md) - Cutout size parameter details
