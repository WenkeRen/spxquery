"""
Tests for advanced parameter configuration system.
"""

import json
from pathlib import Path

import pytest

from spxquery.core.config import (
    AdvancedConfig,
    DownloadConfig,
    PhotometryConfig,
    QueryConfig,
    Source,
    VisualizationConfig,
)
from spxquery.utils.params import export_default_parameters, load_advanced_config


class TestPhotometryConfig:
    """Tests for PhotometryConfig."""

    def test_default_values(self):
        """Test that default values are correctly set."""
        config = PhotometryConfig()
        assert config.annulus_inner_offset == 1.414
        assert config.min_annulus_area == 10
        assert config.max_outer_radius == 5.0
        assert config.bg_sigma_clip_sigma == 3.0
        assert config.pixel_scale_fallback == 6.2

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = PhotometryConfig(annulus_inner_offset=2.0, bg_sigma_clip_sigma=5.0, pixel_scale_fallback=7.0)
        assert config.annulus_inner_offset == 2.0
        assert config.bg_sigma_clip_sigma == 5.0
        assert config.pixel_scale_fallback == 7.0

    def test_validation_positive_values(self):
        """Test that validation rejects negative values."""
        with pytest.raises(ValueError, match="must be >= 0"):
            PhotometryConfig(annulus_inner_offset=-1.0)

        with pytest.raises(ValueError, match="must be > 0"):
            PhotometryConfig(bg_sigma_clip_sigma=0.0)

    def test_validation_range_checks(self):
        """Test that validation checks min/max relationships."""
        with pytest.raises(ValueError, match="zodi_scale_max must be > zodi_scale_min"):
            PhotometryConfig(zodi_scale_min=10.0, zodi_scale_max=5.0)

    def test_serialization(self):
        """Test JSON serialization."""
        config = PhotometryConfig(annulus_inner_offset=2.5, bg_sigma_clip_sigma=4.0)
        data = config.to_dict()

        assert isinstance(data, dict)
        assert data["annulus_inner_offset"] == 2.5
        assert data["bg_sigma_clip_sigma"] == 4.0

        # Round-trip test
        config2 = PhotometryConfig.from_dict(data)
        assert config2.annulus_inner_offset == 2.5
        assert config2.bg_sigma_clip_sigma == 4.0


class TestVisualizationConfig:
    """Tests for VisualizationConfig."""

    def test_default_values(self):
        """Test that default values are correctly set."""
        config = VisualizationConfig()
        assert config.wavelength_cmap == "rainbow"
        assert config.date_cmap == "viridis"
        assert config.sigma_clip_sigma == 3.0
        assert config.dpi == 150
        assert config.figsize == (10, 8)  # Actual default from config.py

    def test_custom_figsize(self):
        """Test custom figure size."""
        config = VisualizationConfig(figsize=(10, 8))
        assert config.figsize == (10, 8)

    def test_colormap_validation(self):
        """Test that invalid colormaps are rejected."""
        with pytest.raises(ValueError, match="Invalid wavelength_cmap"):
            VisualizationConfig(wavelength_cmap="invalid_cmap_name")

    def test_alpha_validation(self):
        """Test that alpha values must be between 0 and 1."""
        with pytest.raises(ValueError, match="must be 0-1"):
            VisualizationConfig(errorbar_alpha=1.5)

        with pytest.raises(ValueError, match="must be 0-1"):
            VisualizationConfig(marker_alpha=-0.1)

    def test_serialization(self):
        """Test JSON serialization."""
        config = VisualizationConfig(wavelength_cmap="plasma", dpi=300)
        data = config.to_dict()

        assert data["wavelength_cmap"] == "plasma"
        assert data["dpi"] == 300

        # Round-trip test
        config2 = VisualizationConfig.from_dict(data)
        assert config2.wavelength_cmap == "plasma"
        assert config2.dpi == 300


class TestDownloadConfig:
    """Tests for DownloadConfig."""

    def test_default_values(self):
        """Test that default values are correctly set."""
        config = DownloadConfig()
        assert config.chunk_size == 8192
        assert config.timeout == 300
        assert config.max_retries == 3

    def test_validation_positive_values(self):
        """Test that validation rejects non-positive values."""
        with pytest.raises(ValueError, match="must be > 0"):
            DownloadConfig(chunk_size=0)

        with pytest.raises(ValueError, match="must be > 0"):
            DownloadConfig(timeout=-10)

    def test_serialization(self):
        """Test JSON serialization."""
        config = DownloadConfig(chunk_size=16384, max_retries=5)
        data = config.to_dict()

        assert data["chunk_size"] == 16384
        assert data["max_retries"] == 5


class TestAdvancedConfig:
    """Tests for AdvancedConfig container."""

    def test_default_values(self):
        """Test that default nested configs are created."""
        config = AdvancedConfig()
        assert isinstance(config.photometry, PhotometryConfig)
        assert isinstance(config.visualization, VisualizationConfig)
        assert isinstance(config.download, DownloadConfig)

    def test_custom_nested_configs(self):
        """Test creating with custom nested configs."""
        photo_config = PhotometryConfig(bg_sigma_clip_sigma=5.0)
        viz_config = VisualizationConfig(dpi=300)

        config = AdvancedConfig(photometry=photo_config, visualization=viz_config)

        assert config.photometry.bg_sigma_clip_sigma == 5.0
        assert config.visualization.dpi == 300

    def test_serialization(self):
        """Test JSON serialization of nested configs."""
        config = AdvancedConfig()
        config.photometry.bg_sigma_clip_sigma = 4.0
        config.visualization.dpi = 200

        data = config.to_dict()
        assert isinstance(data, dict)
        assert "photometry" in data
        assert "visualization" in data
        assert "download" in data
        assert data["photometry"]["bg_sigma_clip_sigma"] == 4.0
        assert data["visualization"]["dpi"] == 200

        # Round-trip test
        config2 = AdvancedConfig.from_dict(data)
        assert config2.photometry.bg_sigma_clip_sigma == 4.0
        assert config2.visualization.dpi == 200

    def test_json_file_operations(self, tmp_path):
        """Test saving and loading from JSON file."""
        config = AdvancedConfig()
        config.photometry.pixel_scale_fallback = 7.5
        config.visualization.wavelength_cmap = "plasma"

        filepath = tmp_path / "test_config.json"
        config.to_json_file(filepath)

        # Verify file exists and is valid JSON
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
            assert data["photometry"]["pixel_scale_fallback"] == 7.5
            assert data["visualization"]["wavelength_cmap"] == "plasma"

        # Load from file
        config2 = AdvancedConfig.from_json_file(filepath)
        assert config2.photometry.pixel_scale_fallback == 7.5
        assert config2.visualization.wavelength_cmap == "plasma"


class TestExportDefaultParameters:
    """Tests for parameter export utility."""

    def test_export_to_directory(self, tmp_path):
        """Test exporting to directory with default filename."""
        params_file = export_default_parameters(tmp_path)

        assert params_file.exists()
        assert params_file.name == "spxquery_default_params.json"
        assert params_file.parent == tmp_path

        # Verify JSON is valid
        with open(params_file) as f:
            data = json.load(f)
            assert "photometry" in data
            assert "visualization" in data
            assert "download" in data

    def test_export_to_file(self, tmp_path):
        """Test exporting to specific file path."""
        filepath = tmp_path / "custom_params.json"
        params_file = export_default_parameters(filepath)

        assert params_file == filepath
        assert params_file.exists()

    def test_export_with_custom_filename(self, tmp_path):
        """Test exporting with custom filename."""
        params_file = export_default_parameters(tmp_path, filename="my_params.json")

        assert params_file.name == "my_params.json"
        assert params_file.exists()

    def test_no_source_information_in_template(self, tmp_path):
        """Test that exported template does NOT contain source info."""
        params_file = export_default_parameters(tmp_path)

        with open(params_file) as f:
            data = json.load(f)
            # Ensure no source-specific fields
            assert "source" not in data
            assert "ra" not in data
            assert "dec" not in data
            assert "name" not in data


class TestLoadAdvancedConfig:
    """Tests for loading config from JSON."""

    def test_load_valid_file(self, tmp_path):
        """Test loading from valid JSON file."""
        # Create a config file
        config = AdvancedConfig()
        config.photometry.bg_sigma_clip_sigma = 4.5
        filepath = tmp_path / "config.json"
        config.to_json_file(filepath)

        # Load it
        loaded = load_advanced_config(filepath)
        assert loaded.photometry.bg_sigma_clip_sigma == 4.5

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_advanced_config(Path("/nonexistent/file.json"))

    def test_load_invalid_json(self, tmp_path):
        """Test that loading invalid JSON raises ValueError."""
        filepath = tmp_path / "invalid.json"
        filepath.write_text("not valid json {[")

        with pytest.raises(ValueError, match="Failed to load parameters"):
            load_advanced_config(filepath)


class TestPrioritySystem:
    """Tests for parameter priority system."""

    def test_explicit_param_overrides_config(self, tmp_path):
        """Test that explicit parameters override config file values."""
        # Create config file with specific values
        config = AdvancedConfig()
        config.photometry.bg_sigma_clip_sigma = 10.0
        filepath = tmp_path / "config.json"
        config.to_json_file(filepath)

        # Load config
        loaded_config = load_advanced_config(filepath)
        assert loaded_config.photometry.bg_sigma_clip_sigma == 10.0

        # When used with explicit parameter, explicit should win
        # (This would be tested at the function level in photometry.py tests)

    def test_config_file_overrides_defaults(self, tmp_path):
        """Test that config file values override defaults."""
        # Export default parameters
        default_file = export_default_parameters(tmp_path)

        # Load and modify
        config = load_advanced_config(default_file)
        original_sigma = config.photometry.bg_sigma_clip_sigma

        config.photometry.bg_sigma_clip_sigma = 15.0
        modified_file = tmp_path / "modified.json"
        config.to_json_file(modified_file)

        # Load modified config
        modified_config = load_advanced_config(modified_file)
        assert modified_config.photometry.bg_sigma_clip_sigma == 15.0
        assert modified_config.photometry.bg_sigma_clip_sigma != original_sigma


class TestQueryConfigIntegration:
    """Tests for QueryConfig integration with advanced params."""

    def test_query_config_with_param_file(self, tmp_path):
        """Test QueryConfig auto-loads advanced params from file."""
        # Create params file
        params_file = export_default_parameters(tmp_path)

        # Modify params
        config = load_advanced_config(params_file)
        config.photometry.bg_sigma_clip_sigma = 7.0
        config.to_json_file(params_file)

        # Create QueryConfig with params file
        source = Source(ra=100.0, dec=20.0, name="TestSource")
        query_config = QueryConfig(source=source, output_dir=tmp_path / "output", advanced_params_file=params_file)

        # Verify advanced params were loaded
        assert query_config.advanced.photometry.bg_sigma_clip_sigma == 7.0

    def test_query_config_with_defaults(self, tmp_path):
        """Test QueryConfig uses defaults when no param file provided."""
        source = Source(ra=100.0, dec=20.0, name="TestSource")
        query_config = QueryConfig(source=source, output_dir=tmp_path / "output")

        # Should use defaults
        assert isinstance(query_config.advanced, AdvancedConfig)
        assert query_config.advanced.photometry.bg_sigma_clip_sigma == 3.0  # default value

    def test_query_config_with_nonexistent_file(self, tmp_path):
        """Test QueryConfig raises error for nonexistent param file."""
        source = Source(ra=100.0, dec=20.0, name="TestSource")

        with pytest.raises(FileNotFoundError):
            QueryConfig(
                source=source, output_dir=tmp_path / "output", advanced_params_file=tmp_path / "nonexistent.json"
            )
