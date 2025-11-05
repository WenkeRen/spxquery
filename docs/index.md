# SPXQuery Documentation

**SPXQuery** is a Python package for automated SPHEREx spectral image data query, download, and time-domain photometry analysis.

SPHEREx (NASA Astrophysics Medium Explorer) obtains 0.75-5 μm spectroscopy across the entire sky using Linear Variable Filters (LVFs). This package provides tools to:

- Query SPHEREx data from [IRSA (NASA/IPAC Infrared Science Archive)](https://irsa.ipac.caltech.edu/Missions/spherex.html)
- Download spectral images with optional cutouts for reduced file sizes
- Perform aperture photometry with zodiacal background subtraction
- Generate time-series light curves and visualizations
- Apply quality control filtering for robust photometry

## Features

- **Automated TAP queries** to IRSA SPHEREx archive
- **Image cutout support** for 99% storage reduction (200px cutout vs full 2040×2040 image)
- **Parallel downloads** with progress tracking and resumable pipeline
- **Aperture photometry** with configurable parameters and background estimation
- **Quality control** with SNR thresholds and pixel flag filtering
- **Publication-quality plots** with customizable visualization parameters

## Quick Start

Install via pip:

```bash
pip install spxquery
```

Basic usage:

```python
from spxquery import SPXQueryPipeline, Source, QueryConfig

# Define your astronomical source
source = Source(ra=304.69, dec=42.44, name="My_Star")

# Configure the query with cutout support
config = QueryConfig(
    source=source,
    output_dir="output",
    cutout_size="200px",  # Download only 200×200 pixel region
    sigma_threshold=5.0,  # Minimum SNR for quality control
)

# Run the full pipeline
pipeline = SPXQueryPipeline(config)
pipeline.run_full_pipeline()
```

This will query IRSA, download cutout images, perform photometry, and generate light curve plots.

## Documentation

```{toctree}
:maxdepth: 2
:caption: Tutorial

tutorials/quickstart_demo
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/pipeline
user_guide/parameters
user_guide/cutouts
user_guide/quality_control
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

## Links

- [GitHub Repository](https://github.com/wenke-astro/spxquery)
- [PyPI Package](https://pypi.org/project/spxquery/)

## Indices and Tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
