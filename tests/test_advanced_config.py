"""
Tests for advanced parameter configuration system.
"""

from pathlib import Path

import pytest
import yaml

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
        assert config.aperture_method == "fwhm"  # Default changed to fwhm in v0.2.2
        assert config.aperture_diameter == 3.0
        assert config.fwhm_multiplier == 2.5
        assert config.background_method == "annulus"
        assert config.window_size == 50
        assert config.annulus_inner_offset == 1.414
        assert config.min_annulus_area == 10
        assert config.max_outer_radius == 5.0
        assert config.bg_sigma_clip_sigma == 3.0
        assert config.pixel_scale_fallback == 6.2
        assert config.subtract_zodi is True

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = PhotometryConfig(
            aperture_method="fwhm",
            fwhm_multiplier=3.0,
            background_method="window",
            window_size=100,
            annulus_inner_offset=2.0,
            bg_sigma_clip_sigma=5.0,
            pixel_scale_fallback=7.0
        )
        assert config.aperture_method == "fwhm"
        assert config.fwhm_multiplier == 3.0
        assert config.background_method == "window"
        assert config.window_size == 100
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

    def test_aperture_method_validation(self):
        """Test that aperture_method only accepts 'fixed' or 'fwhm'."""
        with pytest.raises(ValueError, match="aperture_method must be one of"):
            PhotometryConfig(aperture_method="invalid")

    def test_background_method_validation(self):
        """Test that background_method only accepts 'annulus' or 'window'."""
        with pytest.raises(ValueError, match="background_method must be one of"):
            PhotometryConfig(background_method="invalid")

    def test_serialization(self):
        """Test YAML serialization."""
        config = PhotometryConfig(
            aperture_method="fwhm",
            background_method="window",
            annulus_inner_offset=2.5,
            bg_sigma_clip_sigma=4.0
        )
        data = config.to_dict()

        assert isinstance(data, dict)
        assert data["aperture_method"] == "fwhm"
        assert data["background_method"] == "window"
        assert data["annulus_inner_offset"] == 2.5
        assert data["bg_sigma_clip_sigma"] == 4.0

        # Round-trip test
        config2 = PhotometryConfig.from_dict(data)
        assert config2.aperture_method == "fwhm"
        assert config2.background_method == "window"
        assert config2.annulus_inner_offset == 2.5
        assert config2.bg_sigma_clip_sigma == 4.0


class TestVisualizationConfig:
    """Tests for VisualizationConfig."""

    def test_default_values(self):
        """Test that default values are correctly set."""
        config = VisualizationConfig()
        assert config.sigma_threshold == 5.0
        assert config.use_magnitude is False
        assert config.show_errorbars is True
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
        """Test YAML serialization."""
        config = VisualizationConfig(
            sigma_threshold=3.0,
            use_magnitude=True,
            wavelength_cmap="plasma",
            dpi=300
        )
        data = config.to_dict()

        assert data["sigma_threshold"] == 3.0
        assert data["use_magnitude"] is True
        assert data["wavelength_cmap"] == "plasma"
        assert data["dpi"] == 300

        # Round-trip test
        config2 = VisualizationConfig.from_dict(data)
        assert config2.sigma_threshold == 3.0
        assert config2.use_magnitude is True
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
        """Test YAML serialization."""
        config = DownloadConfig(chunk_size=16384, max_retries=5)
        data = config.to_dict()

        assert data["chunk_size"] == 16384
        assert data["max_retries"] == 5

        # Round-trip test
        config2 = DownloadConfig.from_dict(data)
        assert config2.chunk_size == 16384
        assert config2.max_retries == 5


class TestAdvancedConfig:
    """Tests for AdvancedConfig container."""

    def test_default_values(self):
        """Test that default nested configs are created."""
        source = Source(ra=100.0, dec=20.0, name="TestSource")
        query_config = QueryConfig(source=source, output_dir="output")
        config = AdvancedConfig(query=query_config)
        assert isinstance(config.query, QueryConfig)
        assert isinstance(config.photometry, PhotometryConfig)
        assert isinstance(config.visualization, VisualizationConfig)
        assert isinstance(config.download, DownloadConfig)

    def test_custom_nested_configs(self):
        """Test creating with custom nested configs."""
        source = Source(ra=100.0, dec=20.0, name="TestSource")
        query_config = QueryConfig(source=source, output_dir="output")
        photo_config = PhotometryConfig(bg_sigma_clip_sigma=5.0)
        viz_config = VisualizationConfig(dpi=300)

        config = AdvancedConfig(query=query_config, photometry=photo_config, visualization=viz_config)

        assert config.photometry.bg_sigma_clip_sigma == 5.0
        assert config.visualization.dpi == 300

    def test_serialization(self):
        """Test YAML serialization of nested configs."""
        source = Source(ra=100.0, dec=20.0, name="TestSource")
        query_config = QueryConfig(source=source, output_dir="output")
        config = AdvancedConfig(query=query_config)
        config.photometry.bg_sigma_clip_sigma = 4.0
        config.photometry.aperture_method = "fixed"  # Changed from fwhm since we're testing modification
        config.visualization.dpi = 200
        config.visualization.sigma_threshold = 3.0

        data = config.to_dict()
        assert isinstance(data, dict)
        assert "query" in data
        assert "photometry" in data
        assert "visualization" in data
        assert "download" in data
        assert data["photometry"]["bg_sigma_clip_sigma"] == 4.0
        assert data["photometry"]["aperture_method"] == "fixed"
        assert data["visualization"]["dpi"] == 200
        assert data["visualization"]["sigma_threshold"] == 3.0

        # Round-trip test
        config2 = AdvancedConfig.from_dict(data)
        assert config2.photometry.bg_sigma_clip_sigma == 4.0
        assert config2.photometry.aperture_method == "fixed"
        assert config2.visualization.dpi == 200
        assert config2.visualization.sigma_threshold == 3.0

    def test_yaml_file_operations(self, tmp_path):
        """Test saving and loading from YAML file."""
        source = Source(ra=100.0, dec=20.0, name="TestSource")
        query_config = QueryConfig(source=source, output_dir=tmp_path / "output")
        config = AdvancedConfig(query=query_config)
        config.photometry.pixel_scale_fallback = 7.5
        config.photometry.background_method = "window"
        config.visualization.wavelength_cmap = "plasma"
        config.visualization.use_magnitude = True

        filepath = tmp_path / "test_config.yaml"
        config.to_yaml_file(filepath)

        # Verify file exists and is valid YAML
        assert filepath.exists()
        with open(filepath) as f:
            data = yaml.safe_load(f)
            assert data["photometry"]["pixel_scale_fallback"] == 7.5
            assert data["photometry"]["background_method"] == "window"
            assert data["visualization"]["wavelength_cmap"] == "plasma"
            assert data["visualization"]["use_magnitude"] is True

        # Load from file
        config2 = AdvancedConfig.from_yaml_file(filepath)
        assert config2.photometry.pixel_scale_fallback == 7.5
        assert config2.photometry.background_method == "window"
        assert config2.visualization.wavelength_cmap == "plasma"
        assert config2.visualization.use_magnitude is True


class TestExportDefaultParameters:
    """Tests for parameter export utility."""

    def test_export_to_directory(self, tmp_path):
        """Test exporting to directory with default filename."""
        params_file = export_default_parameters(tmp_path)

        assert params_file.exists()
        assert params_file.name == "spxquery_default_params.yaml"
        assert params_file.parent == tmp_path

        # Verify YAML is valid
        with open(params_file) as f:
            data = yaml.safe_load(f)
            assert "photometry" in data
            assert "visualization" in data
            assert "download" in data

    def test_export_to_file(self, tmp_path):
        """Test exporting to specific file path."""
        filepath = tmp_path / "custom_params.yaml"
        params_file = export_default_parameters(filepath)

        assert params_file == filepath
        assert params_file.exists()

    def test_export_with_custom_filename(self, tmp_path):
        """Test exporting with custom filename."""
        params_file = export_default_parameters(tmp_path, filename="my_params.yaml")

        assert params_file.name == "my_params.yaml"
        assert params_file.exists()

    def test_no_source_information_in_template(self, tmp_path):
        """Test that exported template does NOT contain source info."""
        params_file = export_default_parameters(tmp_path)

        with open(params_file) as f:
            data = yaml.safe_load(f)
            # Ensure no source-specific fields
            assert "source" not in data
            assert "ra" not in data
            assert "dec" not in data
            assert "name" not in data


class TestLoadAdvancedConfig:
    """Tests for loading config from YAML."""

    def test_load_valid_file(self, tmp_path):
        """Test loading from valid YAML file."""
        # Create a config file
        source = Source(ra=100.0, dec=20.0, name="TestSource")
        query_config = QueryConfig(source=source, output_dir=tmp_path / "output")
        config = AdvancedConfig(query=query_config)
        config.photometry.bg_sigma_clip_sigma = 4.5
        config.photometry.aperture_method = "fixed"  # Changed from fwhm
        filepath = tmp_path / "config.yaml"
        config.to_yaml_file(filepath)

        # Load it
        loaded = load_advanced_config(filepath)
        assert loaded.photometry.bg_sigma_clip_sigma == 4.5
        assert loaded.photometry.aperture_method == "fixed"

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_advanced_config(Path("/nonexistent/file.yaml"))

    def test_load_invalid_yaml(self, tmp_path):
        """Test that loading invalid YAML raises ValueError."""
        filepath = tmp_path / "invalid.yaml"
        filepath.write_text("not valid yaml: [[[")

        with pytest.raises(ValueError, match="Failed to load parameters"):
            load_advanced_config(filepath)


class TestPrioritySystem:
    """Tests for parameter priority system."""

    def test_explicit_param_overrides_config(self, tmp_path):
        """Test that explicit parameters override config file values."""
        # Create config file with specific values
        source = Source(ra=100.0, dec=20.0, name="TestSource")
        query_config = QueryConfig(source=source, output_dir=tmp_path / "output")
        config = AdvancedConfig(query=query_config)
        config.photometry.bg_sigma_clip_sigma = 10.0
        config.photometry.aperture_method = "fixed"
        filepath = tmp_path / "config.yaml"
        config.to_yaml_file(filepath)

        # Load config
        loaded_config = load_advanced_config(filepath)
        assert loaded_config.photometry.bg_sigma_clip_sigma == 10.0
        assert loaded_config.photometry.aperture_method == "fixed"

        # When used with explicit parameter, explicit should win
        # (This would be tested at the function level in photometry.py tests)

    def test_config_file_overrides_defaults(self, tmp_path):
        """Test that config file values override defaults."""
        # Export default parameters
        default_file = export_default_parameters(tmp_path)

        # Load and modify
        config = load_advanced_config(default_file)
        original_sigma = config.photometry.bg_sigma_clip_sigma
        original_method = config.photometry.background_method

        config.photometry.bg_sigma_clip_sigma = 15.0
        config.photometry.background_method = "window"
        modified_file = tmp_path / "modified.yaml"
        config.to_yaml_file(modified_file)

        # Load modified config
        modified_config = load_advanced_config(modified_file)
        assert modified_config.photometry.bg_sigma_clip_sigma == 15.0
        assert modified_config.photometry.background_method == "window"
        assert modified_config.photometry.bg_sigma_clip_sigma != original_sigma
        assert modified_config.photometry.background_method != original_method


class TestAdvancedConfigIntegration:
    """Tests for AdvancedConfig integration and loading."""

    def test_advanced_config_from_param_file(self, tmp_path):
        """Test loading AdvancedConfig from YAML file."""
        # Create params file
        params_file = export_default_parameters(tmp_path)

        # Modify params
        config = load_advanced_config(params_file)
        config.photometry.bg_sigma_clip_sigma = 7.0
        config.photometry.aperture_method = "fixed"
        config.visualization.sigma_threshold = 3.0
        config.to_yaml_file(params_file)

        # Load modified config
        loaded_config = load_advanced_config(params_file)

        # Verify params were loaded
        assert loaded_config.photometry.bg_sigma_clip_sigma == 7.0
        assert loaded_config.photometry.aperture_method == "fixed"
        assert loaded_config.visualization.sigma_threshold == 3.0

    def test_advanced_config_with_defaults(self, tmp_path):
        """Test AdvancedConfig created with default sub-configs."""
        source = Source(ra=100.0, dec=20.0, name="TestSource")
        query_config = QueryConfig(source=source, output_dir=tmp_path / "output")
        config = AdvancedConfig(query=query_config)

        # Should use defaults
        assert config.photometry.bg_sigma_clip_sigma == 3.0  # default value
        assert config.photometry.aperture_method == "fwhm"  # default value (changed in v0.2.2)
        assert config.visualization.sigma_threshold == 5.0  # default value

    def test_advanced_config_load_nonexistent_file(self, tmp_path):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_advanced_config(tmp_path / "nonexistent.yaml")
