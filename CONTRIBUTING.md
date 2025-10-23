# Contributing to SPXQuery

Thank you for your interest in contributing to SPXQuery! This document provides guidelines for contributing to the project.

## Development Setup

### Using Poetry (Recommended)

1. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:
```bash
git clone https://github.com/WenkeRen/spxquery.git
cd spxquery
```

3. Install dependencies:
```bash
poetry install --with dev
```

4. Activate the Poetry environment:
```bash
poetry shell
```

### Using pip (Alternative)

1. Clone the repository:
```bash
git clone https://github.com/WenkeRen/spxquery.git
cd spxquery
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

## Code Style

We follow standard Python style guidelines:
- Use [Black](https://github.com/psf/black) for code formatting (100 char line length)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Follow [PEP 8](https://pep8.org/) conventions

Run formatters before committing:
```bash
# With Poetry
poetry run black .
poetry run isort .

# Or after activating environment
black .
isort .
```

## Testing

Run tests using pytest:
```bash
# With Poetry
poetry run pytest

# Or after activating environment
pytest
```

For coverage report:
```bash
poetry run pytest --cov=spxquery --cov-report=html
```

## Development Workflow

### Adding Dependencies

```bash
# Add a runtime dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Update dependencies
poetry update
```

### Making Changes

1. Create a new branch for your feature/fix:
```bash
git checkout -b feature-name
```

2. Make your changes
3. Format code:
```bash
poetry run black .
poetry run isort .
```

4. Run tests:
```bash
poetry run pytest
```

5. Commit with clear messages:
```bash
git add .
git commit -m "feat: add new feature description"
```

## Submitting Changes

1. Push to your fork:
```bash
git push origin feature-name
```

2. Submit a pull request to the main repository
3. Ensure all tests pass and code follows style guidelines
4. Wait for code review and address feedback

## Commit Message Guidelines

Follow conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```bash
git commit -m "feat: add cutout size validation"
git commit -m "fix: correct wavelength calculation in photometry"
git commit -m "docs: update installation instructions for Poetry"
```

## Reporting Issues

When reporting issues, please include:
- Python version and operating system
- SPXQuery version (`python -c "import spxquery; print(spxquery.__version__)"`)
- Poetry version (`poetry --version`) if using Poetry
- Minimal reproducible example
- Full error traceback

## Code Review Process

1. All pull requests require review before merging
2. Ensure CI tests pass
3. Address reviewer comments
4. Maintainers will merge once approved

## Questions?

Open an issue with the "question" label for general questions about the project.

## Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Black Code Style](https://black.readthedocs.io/)
- [Conventional Commits](https://www.conventionalcommits.org/)
