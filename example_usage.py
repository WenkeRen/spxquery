"""
Example usage of SPXQuery package for SPHEREx time-domain analysis.
"""

from pathlib import Path

from spxquery import QueryConfig, Source, SPXQueryPipeline
from spxquery.core.pipeline import run_pipeline
from spxquery.utils.helpers import setup_logging


# Example 1: Simple usage with convenience function
def example_simple():
    """Simple example using the convenience function."""
    print("Example 1: Simple usage")
    print("-" * 50)

    # Define source coordinates (example: a known variable star)
    ra = 304.693508808
    dec = 42.4436872991

    # Run pipeline with default settings (downloads full images)
    run_pipeline(
        ra=ra,
        dec=dec,
        output_dir=Path("example_output"),
        source_name="Example_Star",
        log_level="INFO",
        max_processing_workers=6,  # Use 6 workers for photometry processing
    )


# Example 2: Image cutout usage (5 arcminute square)
def example_cutout():
    """Example using image cutouts to reduce download size."""
    print("\nExample 2: Using image cutouts (5 arcmin square)")
    print("-" * 50)

    # Set up logging
    setup_logging("INFO")

    # Create source and configuration with cutout
    source = Source(ra=304.693508808, dec=42.4436872991, name="Point_Source")

    config = QueryConfig(
        source=source,
        output_dir=Path("cutout_output"),
        bands=["D1", "D2", "D3"],  # Only query specific bands
        aperture_diameter=3.0,  # 3 pixel diameter
        max_download_workers=4,
        max_processing_workers=8,
        cutout_size="5arcmin",  # 5 arcminute square cutout
    )

    print("\nConfiguration:")
    print(f"  Source: RA={source.ra}, Dec={source.dec}")
    print(f"  Cutout size: {config.cutout_size}")
    print(f"  Cutout center: {config.cutout_center or 'Source position (default)'}")
    print('\nNote: 5 arcmin cutout at SPHEREx pixel scale (~6.2"/pixel)')
    print("      Approximate size: ~48×48 pixels")
    print("      Storage: Much smaller than full 2040×2040 pixel image\n")

    # Create and run pipeline
    pipeline = SPXQueryPipeline(config)
    pipeline.run_full_pipeline()


# Example 3: Step-by-step execution with practical cutout size
def example_step_by_step():
    """Example showing step-by-step execution with image cutouts."""
    print("\nExample 3: Step-by-step execution with cutouts")
    print("-" * 50)

    # Set up logging
    setup_logging("INFO")

    # Create source and configuration
    source = Source(ra=304.693508808, dec=42.4436872991, name="Variable_Star_123")

    config = QueryConfig(
        source=source,
        output_dir=Path("stepwise_output"),
        bands=["D1", "D2", "D3"],  # Only query specific bands
        aperture_diameter=3.0,  # 3 pixel diameter
        max_download_workers=4,
        max_processing_workers=8,  # Use 8 workers for photometry processing
        cutout_size="200px",  # 200×200 pixel cutout (recommended for point sources)
    )

    print("\nConfiguration:")
    print(f"  Cutout size: {config.cutout_size}")
    print("  Storage savings: ~70 MB → ~0.7 MB per file (99% reduction)\n")

    # Create pipeline
    pipeline = SPXQueryPipeline(config)

    # Run stages individually
    print("\n1. Running query...")
    pipeline.run_query()

    print("\n2. Running download...")
    pipeline.run_download()

    print("\n3. Running processing...")
    pipeline.run_processing()

    print("\n4. Running visualization...")
    pipeline.run_visualization()


# Example 4: Resume from interruption
def example_resume():
    """Example showing how to resume after interruption."""
    print("\nExample 4: Resume from saved state")
    print("-" * 50)

    run_pipeline(
        ra=304.693508808,
        dec=42.4436872991,
        output_dir=Path("example_output"),
        resume=True,  # Resume from saved state
        log_level="INFO",
    )


# Example 5: Custom processing with cutouts
def example_custom_processing():
    """Example with custom processing steps and cutout URLs."""
    print("\nExample 5: Custom processing with cutouts")
    print("-" * 50)

    from spxquery.core.query import get_download_urls, query_spherex_observations

    # Set up
    setup_logging("INFO")
    source = Source(ra=304.693508808, dec=42.4436872991)

    # Query only
    results = query_spherex_observations(source, bands=["D2"])
    print(f"Found {len(results)} observations")

    # Get URLs with caching, parallel processing, and cutout parameters
    from pathlib import Path

    cache_file = Path("example_urls_cutout.json")
    urls = get_download_urls(
        results,
        max_workers=4,
        show_progress=True,
        cache_file=cache_file,
        cutout_size="5arcmin",  # 5 arcminute square cutout
        cutout_center=None,  # Use source position
    )
    print(f"Got {len(urls)} download URLs with cutout parameters")

    # Show example URL with cutout parameters
    if urls:
        obs, url = urls[0]
        print(f"\nExample observation: {obs.obs_id}")
        print(f"Download URL: {url[:120]}...")  # Show first 120 chars
        print("\nNote: URL includes cutout parameters (center and size)")


# Example 6: Quality control filtering
def example_quality_control():
    """Example showing quality control filtering options."""
    print("\nExample 6: Quality Control Filtering")
    print("-" * 50)

    setup_logging("INFO")
    source = Source(ra=304.693508808, dec=42.4436872991, name="QC_Test_Source")

    # Example A: Default QC settings (recommended)
    print("\nA. Default QC settings:")
    config_default = QueryConfig(
        source=source,
        output_dir=Path("qc_default"),
        cutout_size="200px",
        sigma_threshold=5.0,  # Default: mark measurements with SNR < 5
        bad_flags=[0, 1, 2, 6, 7, 9, 10, 11, 15],  # Default bad flags
    )
    print(f"   sigma_threshold: {config_default.sigma_threshold}")
    print(f"   bad_flags: {config_default.bad_flags}")
    print("   Filters: SNR < 5.0 and bad pixel flags")

    # Example B: Stringent QC (high quality requirements)
    print("\nB. Stringent QC (high quality):")
    config_strict = QueryConfig(
        source=source,
        output_dir=Path("qc_strict"),
        cutout_size="200px",
        sigma_threshold=10.0,  # Higher SNR requirement
    )
    print(f"   sigma_threshold: {config_strict.sigma_threshold}")
    print("   Higher SNR threshold for cleaner data")

    # Example C: Relaxed QC (more data points)
    print("\nC. Relaxed QC (more data points):")
    config_relaxed = QueryConfig(
        source=source,
        output_dir=Path("qc_relaxed"),
        cutout_size="200px",
        sigma_threshold=3.0,  # Lower SNR requirement
        bad_flags=[0, 1, 2],  # Only most critical flags
    )
    print(f"   sigma_threshold: {config_relaxed.sigma_threshold}")
    print(f"   bad_flags: {config_relaxed.bad_flags}")
    print("   More permissive filtering for fainter sources")

    print("\n**Quality Control Filters Applied During Visualization:**")
    print("  - Good points: Plotted as filled circles (normal markers)")
    print("  - Rejected points: Plotted as small gray crosses")
    print("  - CSV output: Contains ALL measurements (no data removed)")
    print("\n  Quality criteria:")
    print("    1. SNR filter: Marks flux/flux_err < sigma_threshold")
    print("    2. Flag filter: Marks pixels with bad quality flags")
    print("\n  Default bad flags:")
    print("    - 0,1,2: Saturation, bad pixels")
    print("    - 6,7: Cosmic rays, non-linearity")
    print("    - 9,10,11: Edge effects")
    print("    - 15: Other data quality issues")


# Example 7: Various cutout size formats
def example_cutout_formats():
    """Example showing different cutout size formats."""
    print("\nExample 7: Different cutout size formats")
    print("-" * 50)

    setup_logging("INFO")
    source = Source(ra=304.693508808, dec=42.4436872991, name="Test_Source")

    # Example A: Square pixel cutout (recommended for point sources)
    print("\nA. 200-pixel square cutout:")
    config_px = QueryConfig(
        source=source,
        output_dir=Path("cutout_200px"),
        cutout_size="200px",
    )
    print(f"   cutout_size: {config_px.cutout_size}")
    print("   Estimated size: ~0.7 MB per file (vs 70 MB full image)")

    # Example B: Rectangular pixel cutout
    print("\nB. 300×400 pixel rectangular cutout:")
    config_rect = QueryConfig(
        source=source,
        output_dir=Path("cutout_rect"),
        cutout_size="300,400px",
    )
    print(f"   cutout_size: {config_rect.cutout_size}")
    print("   Estimated size: ~2.0 MB per file")

    # Example C: Angular cutout (arcminutes)
    print("\nC. 3 arcminute square cutout:")
    config_arcmin = QueryConfig(
        source=source,
        output_dir=Path("cutout_3arcmin"),
        cutout_size="3arcmin",
    )
    print(f"   cutout_size: {config_arcmin.cutout_size}")
    print('   Approximately 29x29 pixels at SPHEREx scale (~6.2"/pixel)')

    # Example D: 5 arcminute cutout (good balance of coverage and efficiency)
    print("\nD. 5 arcminute square cutout:")
    config_5arcmin = QueryConfig(
        source=source,
        output_dir=Path("cutout_5arcmin"),
        cutout_size="5arcmin",
    )
    print(f"   cutout_size: {config_5arcmin.cutout_size}")
    print('   Approximately 48x48 pixels at SPHEREx scale (~6.2"/pixel)')
    print("   Good balance between coverage and file size")

    # Example E: Custom center position
    print("\nE. Cutout with custom center:")
    config_custom = QueryConfig(
        source=source,
        output_dir=Path("cutout_custom_center"),
        cutout_size="200px",
        cutout_center="304.7,42.5",  # Slightly offset from source
    )
    print(f"   cutout_size: {config_custom.cutout_size}")
    print(f"   cutout_center: {config_custom.cutout_center}")
    print("   (Center offset from source position)")


if __name__ == "__main__":
    # Run examples
    print("SPXQuery Package Examples")
    print("=" * 50)

    # Uncomment the example you want to run:

    # example_simple()  # Basic usage without cutouts
    # example_cutout()  # 5 arcmin square cutout example
    # example_step_by_step()  # Step-by-step with 200px cutout
    # example_resume()  # Resume from saved state
    # example_custom_processing()  # Custom processing with 5 arcmin cutout
    example_quality_control()  # Quality control filtering options
    # example_cutout_formats()  # Various cutout size formats
