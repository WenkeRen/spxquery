#!/usr/bin/env python
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

    print(f"\nConfiguration:")
    print(f"  Source: RA={source.ra}, Dec={source.dec}")
    print(f"  Cutout size: {config.cutout_size}")
    print(f"  Cutout center: {config.cutout_center or 'Source position (default)'}")
    print(f"\nNote: 5 arcmin cutout at SPHEREx pixel scale (~6.2\"/pixel)")
    print(f"      Approximate size: ~48×48 pixels")
    print(f"      Storage: Much smaller than full 2040×2040 pixel image\n")

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

    print(f"\nConfiguration:")
    print(f"  Cutout size: {config.cutout_size}")
    print(f"  Storage savings: ~70 MB → ~0.7 MB per file (99% reduction)\n")

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
        print(f"\nNote: URL includes cutout parameters (center and size)")


# Example 6: Various cutout size formats
def example_cutout_formats():
    """Example showing different cutout size formats."""
    print("\nExample 6: Different cutout size formats")
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
    print(f"   Estimated size: ~0.7 MB per file (vs 70 MB full image)")

    # Example B: Rectangular pixel cutout
    print("\nB. 300×400 pixel rectangular cutout:")
    config_rect = QueryConfig(
        source=source,
        output_dir=Path("cutout_rect"),
        cutout_size="300,400px",
    )
    print(f"   cutout_size: {config_rect.cutout_size}")
    print(f"   Estimated size: ~2.0 MB per file")

    # Example C: Angular cutout (arcminutes)
    print("\nC. 3 arcminute square cutout:")
    config_arcmin = QueryConfig(
        source=source,
        output_dir=Path("cutout_3arcmin"),
        cutout_size="3arcmin",
    )
    print(f"   cutout_size: {config_arcmin.cutout_size}")
    print(f"   Approximately 29x29 pixels at SPHEREx scale (~6.2\"/pixel)")

    # Example D: 5 arcminute cutout (good balance of coverage and efficiency)
    print("\nD. 5 arcminute square cutout:")
    config_5arcmin = QueryConfig(
        source=source,
        output_dir=Path("cutout_5arcmin"),
        cutout_size="5arcmin",
    )
    print(f"   cutout_size: {config_5arcmin.cutout_size}")
    print(f"   Approximately 48x48 pixels at SPHEREx scale (~6.2\"/pixel)")
    print(f"   Good balance between coverage and file size")

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
    print(f"   (Center offset from source position)")


if __name__ == "__main__":
    # Run examples
    print("SPXQuery Package Examples")
    print("=" * 50)

    # Uncomment the example you want to run:

    # example_simple()  # Basic usage without cutouts
    example_cutout()  # 5 arcmin square cutout example
    # example_step_by_step()  # Step-by-step with 200px cutout
    # example_resume()  # Resume from saved state
    # example_custom_processing()  # Custom processing with 5 arcmin cutout
    # example_cutout_formats()  # Various cutout size formats
