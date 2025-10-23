# Installation Guide

## PyPI Installation (Coming Soon)

SPXQuery will soon be available on PyPI. Once published, you'll be able to install it with:

```bash
pip install spxquery
```

**Note**: This installation method is not yet available. Please use the manual installation method below.

## Manual Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/WenkeRen/spxquery.git
cd spxquery
```

### Step 2: Install the package

```bash
pip install .
```

This will automatically install SPXQuery and all required dependencies.

### Step 3: Verify installation

Test that the package is installed correctly:

```bash
python -c "import spxquery; print(spxquery.__version__)"
```

You should see: `0.1.0`

## Using SPXQuery

After installation, you can use SPXQuery in two ways:

### In Python scripts

```python
from spxquery import Source, QueryConfig, SPXQueryPipeline

# Create a source and configuration
source = Source(ra=304.69, dec=42.44, name="my_target")
config = QueryConfig(
    source=source,
    output_dir="output",
    cutout_size="200px"
)

# Run the pipeline
pipeline = SPXQueryPipeline(config)
pipeline.run_full_pipeline()
```

### From the command line

```bash
spxquery --help
```

## Uninstalling

To remove SPXQuery from your system:

```bash
pip uninstall spxquery
```

## Requirements

- Python 3.8 or later (supports Python 3.8-3.13)
- Internet connection for downloading SPHEREx data from IRSA

All other dependencies will be installed automatically.

## Troubleshooting

### ImportError after installation

If you encounter import errors, make sure you're not running Python from within the `spxquery` source directory. Navigate to a different directory and try again:

```bash
cd ~
python -c "import spxquery; print(spxquery.__version__)"
```

### Permission errors during installation

If you get permission errors, try installing with the `--user` flag:

```bash
pip install --user .
```

### Need help?

For issues or questions, please visit: https://github.com/WenkeRen/spxquery/issues
