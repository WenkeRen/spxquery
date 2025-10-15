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

    # Run pipeline with default settings
    run_pipeline(
        ra=ra,
        dec=dec,
        output_dir=Path("example_output"),
        source_name="Example_Star",
        log_level="INFO",
        max_processing_workers=6,  # Use 6 workers for photometry processing
    )


# Example 2: Step-by-step execution
def example_step_by_step():
    """Example showing step-by-step execution."""
    print("\nExample 2: Step-by-step execution")
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
    )

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


# Example 3: Resume from interruption
def example_resume():
    """Example showing how to resume after interruption."""
    print("\nExample 3: Resume from saved state")
    print("-" * 50)

    run_pipeline(
        ra=304.693508808,
        dec=42.4436872991,
        output_dir=Path("example_output"),
        resume=True,  # Resume from saved state
        log_level="INFO",
    )


# Example 4: Custom processing
def example_custom_processing():
    """Example with custom processing steps."""
    print("\nExample 4: Custom processing")
    print("-" * 50)

    from spxquery.core.query import get_download_urls, query_spherex_observations

    # Set up
    setup_logging("INFO")
    source = Source(ra=304.693508808, dec=42.4436872991)

    # Query only
    results = query_spherex_observations(source, bands=["D2"])
    print(f"Found {len(results)} observations")

    # Get URLs with caching and parallel processing
    from pathlib import Path

    cache_file = Path("example_urls.json")
    urls = get_download_urls(results, max_workers=4, show_progress=True, cache_file=cache_file)
    print(f"Got {len(urls)} download URLs")

    # Process a single file (example)
    if urls:
        obs, url = urls[0]
        print(f"Example observation: {obs.obs_id}")
        print(f"Download URL: {url[:80]}...")  # Show first 80 chars


if __name__ == "__main__":
    # Run examples
    print("SPXQuery Package Examples")
    print("=" * 50)

    # Uncomment the example you want to run:

    example_simple()
    # example_step_by_step()
    # example_resume()
    # example_custom_processing()
