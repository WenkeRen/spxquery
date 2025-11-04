# Installation Guide

## Prerequisites

SPXQuery requires:
- **Python 3.11 or later** (supports Python 3.11, 3.12, 3.13)
- Internet connection for downloading SPHEREx data from IRSA

## Installation from Source (pip)

This is the recommended method for most users.

### Step 1: Clone the repository

```bash
git clone https://github.com/WenkeRen/spxquery.git
cd spxquery
```

### Step 2: Install the package

`pip` can install directly from `pyproject.toml`:

```bash
pip install .
```

This will automatically install SPXQuery and all required dependencies (numpy, astropy, matplotlib, photutils, etc.).

### Step 3: Verify installation

Test that the package is installed correctly:

```bash
python -c "import spxquery; print(spxquery.__version__)"
```

You should see: `0.2.0`

## Installation with Poetry (Optional)

If you're a developer contributing to SPXQuery or prefer using Poetry for dependency management:

### Step 1: Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

For Windows PowerShell:

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Verify Poetry installation:

```bash
poetry --version
```

### Step 2: Install with Poetry

```bash
git clone https://github.com/WenkeRen/spxquery.git
cd spxquery
poetry install
```

This installs:
- SPXQuery package from `src/spxquery/`
- All runtime dependencies
- Development tools (pytest, black, flake8, mypy)

### Step 3: Verify installation

```bash
poetry run python -c "import spxquery; print(spxquery.__version__)"
```

## PyPI Installation (Coming Soon)

SPXQuery will be available on PyPI in the future. Once published, you'll be able to install it with:

```bash
pip install spxquery
```

**Note**: This installation method is not yet available. Please use the installation from source method above.

## Using SPXQuery

### If installed with pip

Use SPXQuery directly in Python scripts:

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

Or use the command-line interface:

```bash
spxquery --help
```

### If installed with Poetry

You have two options:

**Option 1: Activate Poetry shell** (recommended)

```bash
poetry shell
# Now use SPXQuery normally
python my_script.py
spxquery --help
exit  # When done
```

**Option 2: Use `poetry run` prefix**

```bash
poetry run python my_script.py
poetry run spxquery --help
```

## Development Workflow (Poetry users)

### Running tests

```bash
# Install with dev dependencies
poetry install --with dev

# Run tests with coverage
poetry run pytest
poetry run pytest --cov=spxquery --cov-report=html
```

### Code formatting

```bash
poetry run black .
poetry run isort .
poetry run flake8 src/
```

### Adding dependencies

```bash
# Add runtime dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name
```

## Uninstalling

### If installed with pip

```bash
pip uninstall spxquery
```

### If installed with Poetry

```bash
# Remove the virtual environment
poetry env remove python

# Or manually delete the project directory
cd ..
rm -rf spxquery
```

## Troubleshooting

### ImportError after installation

**For pip users**: Make sure you're not running Python from within the `spxquery` source directory. Navigate to a different directory and try again:

```bash
cd ~
python -c "import spxquery; print(spxquery.__version__)"
```

**For Poetry users**: Ensure you're either:
1. Running commands with `poetry run` prefix, OR
2. Inside an activated Poetry shell (`poetry shell`)

### Permission errors during pip installation

If you get permission errors with pip, try:

```bash
pip install --user .
```

Or use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install .
```

### Python version mismatch

SPXQuery requires Python 3.11+. Check your Python version:

```bash
python --version
```

If needed, install a compatible version or use pyenv:

```bash
# Install pyenv (macOS/Linux)
curl https://pyenv.run | bash

# Install Python 3.11
pyenv install 3.11

# Use it for this project
cd spxquery
pyenv local 3.11
pip install .  # or poetry install
```

### Poetry-specific issues

**Poetry not found after installation**: Add Poetry to your PATH. For bash/zsh:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Restart your terminal after adding to PATH.

**Lock file errors**: If you encounter Poetry lock file errors:

```bash
poetry lock --no-update
poetry install
```

### Need help?

For issues or questions, please visit: https://github.com/WenkeRen/spxquery/issues
