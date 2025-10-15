"""
Main pipeline orchestrator for SPXQuery package.
"""

import logging
from pathlib import Path
from typing import Optional, List

from ..core.config import QueryConfig, PipelineState
from ..core.query import query_spherex_observations, get_download_urls, print_query_summary
from ..core.download import parallel_download, print_download_summary
from ..processing.photometry import process_all_observations
from ..processing.lightcurve import (
    generate_lightcurve_dataframe, save_lightcurve_csv, print_lightcurve_summary,
    load_lightcurve_from_csv
)
from ..visualization.plots import create_combined_plot
from ..utils.helpers import setup_logging, save_json, load_json, get_file_list

logger = logging.getLogger(__name__)


class SPXQueryPipeline:
    """
    Main pipeline for SPHEREx data query, download, and analysis.
    
    Supports both full automatic execution and step-by-step mode.
    """
    
    def __init__(self, config: QueryConfig):
        """
        Initialize pipeline with configuration.
        
        Parameters
        ----------
        config : QueryConfig
            Pipeline configuration
        """
        self.config = config
        self.state = PipelineState(stage='query', config=config)
        
        # Set up directories
        self.data_dir = config.output_dir / 'data'
        self.results_dir = config.output_dir / 'results'
        self.state_file = config.output_dir / 'pipeline_state.json'
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized pipeline for source at RA={config.source.ra}, Dec={config.source.dec}")
    
    def save_state(self) -> None:
        """Save current pipeline state to disk."""
        state_dict = self.state.to_dict()
        save_json(state_dict, self.state_file)
        logger.info(f"Saved pipeline state: stage={self.state.stage}")
    
    def load_state(self) -> bool:
        """
        Load pipeline state from disk.
        
        Returns
        -------
        bool
            True if state was loaded successfully
        """
        if not self.state_file.exists():
            return False
        
        try:
            state_dict = load_json(self.state_file)
            self.state = PipelineState.from_dict(state_dict)
            logger.info(f"Loaded pipeline state: stage={self.state.stage}")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def run_full_pipeline(self) -> None:
        """Run the complete pipeline from query to visualization."""
        logger.info("Starting full pipeline execution")
        
        self.run_query()
        self.run_download()
        self.run_processing()
        self.run_visualization()
        
        logger.info("Pipeline execution complete")
    
    def run_query(self) -> None:
        """Execute query stage."""
        logger.info("Running query stage")
        
        # Query SPHEREx archive
        query_results = query_spherex_observations(
            self.config.source,
            self.config.bands
        )
        
        # Print summary
        print_query_summary(query_results)
        
        # Save query results
        query_info = {
            'source': {
                'ra': self.config.source.ra,
                'dec': self.config.source.dec,
                'name': self.config.source.name
            },
            'query_time': query_results.query_time.isoformat(),
            'n_observations': len(query_results),
            'time_span_days': query_results.time_span_days,
            'total_size_gb': query_results.total_size_gb,
            'band_counts': query_results.band_counts
        }
        save_json(query_info, self.results_dir / 'query_summary.json')
        
        # Update state
        self.state.query_results = query_results
        self.state.stage = 'download'
        self.save_state()
    
    def run_download(self) -> None:
        """Execute download stage."""
        if not self.state.query_results:
            raise RuntimeError("No query results available. Run query stage first.")
        
        logger.info("Running download stage")
        
        # Get download URLs with caching
        url_cache_file = self.results_dir / 'download_urls.json'
        download_info = get_download_urls(
            self.state.query_results,
            max_workers=self.config.max_download_workers,
            show_progress=True,
            cache_file=url_cache_file,
            cutout_size=self.config.cutout_size,
            cutout_center=self.config.cutout_center
        )
        
        if not download_info:
            logger.warning("No download URLs found")
            self.state.stage = 'processing'
            self.save_state()
            return
        
        # Download files
        download_results = parallel_download(
            download_info,
            self.data_dir,
            max_workers=self.config.max_download_workers
        )
        
        # Print summary
        print_download_summary(download_results)
        
        # Update state with downloaded files
        self.state.downloaded_files = [
            r.local_path for r in download_results if r.success
        ]
        self.state.stage = 'processing'
        self.save_state()
    
    def run_processing(self) -> None:
        """Execute processing stage."""
        logger.info("Running processing stage")
        
        # Get list of downloaded files
        if not self.state.downloaded_files:
            # Try to find files in data directory
            self.state.downloaded_files = get_file_list(self.data_dir, "*.fits")
        
        if not self.state.downloaded_files:
            logger.warning("No FITS files found for processing")
            self.state.stage = 'visualization'
            self.save_state()
            return
        
        logger.info(f"Processing {len(self.state.downloaded_files)} FITS files")
        
        # Process all files
        # Convert diameter to radius for photometry function
        photometry_results = process_all_observations(
            self.state.downloaded_files,
            self.config.source,
            aperture_radius=self.config.aperture_diameter / 2.0,  # Convert diameter to radius
            subtract_zodi=True,
            max_workers=self.config.max_processing_workers
        )
        
        if not photometry_results:
            logger.warning("No photometry results obtained")
            self.state.stage = 'complete'
            self.save_state()
            return
        
        # Generate light curve
        df = generate_lightcurve_dataframe(photometry_results, self.config.source)
        
        # Save light curve CSV
        csv_path = self.results_dir / 'lightcurve.csv'
        save_lightcurve_csv(df, csv_path)
        
        # Print summary
        print_lightcurve_summary(df)
        
        # Update state
        self.state.photometry_results = photometry_results
        self.state.csv_path = csv_path
        self.state.stage = 'visualization'
        self.save_state()
    
    def run_visualization(self) -> None:
        """Execute visualization stage."""
        # Check if photometry results are available in memory
        if not self.state.photometry_results:
            # Try to load from saved lightcurve CSV
            csv_path = self.results_dir / 'lightcurve.csv'
            if csv_path.exists():
                logger.info("Loading photometry results from saved lightcurve CSV")
                self.state.photometry_results = load_lightcurve_from_csv(csv_path)
                self.state.csv_path = csv_path
            
        if not self.state.photometry_results:
            logger.warning("No photometry results available for visualization")
            self.state.stage = 'complete'
            self.save_state()
            return
        
        logger.info("Running visualization stage")

        # Create combined plot with quality control filters
        plot_path = self.results_dir / 'combined_plot.png'
        create_combined_plot(
            self.state.photometry_results,
            plot_path,
            apply_quality_filters=True,
            sigma_threshold=self.config.sigma_threshold,
            bad_flags=self.config.bad_flags
        )

        # Update state
        self.state.plot_path = plot_path
        self.state.stage = 'complete'
        self.save_state()

        logger.info(f"Visualization saved to {plot_path}")
    
    def resume(self) -> None:
        """Resume pipeline from saved state."""
        if not self.load_state():
            logger.warning("No saved state found. Starting from beginning.")
            self.run_full_pipeline()
            return
        
        logger.info(f"Resuming from stage: {self.state.stage}")
        
        # Resume from appropriate stage
        stage_map = {
            'query': self.run_full_pipeline,
            'download': lambda: (self.run_download(), self.run_processing(), self.run_visualization()),
            'processing': lambda: (self.run_processing(), self.run_visualization()),
            'visualization': self.run_visualization,
            'complete': lambda: logger.info("Pipeline already complete")
        }
        
        stage_func = stage_map.get(self.state.stage)
        if stage_func:
            stage_func()
        else:
            logger.error(f"Unknown stage: {self.state.stage}")


def run_pipeline(
    ra: float,
    dec: float,
    output_dir: Optional[Path] = None,
    bands: Optional[List[str]] = None,
    aperture_diameter: float = 3.0,
    source_name: Optional[str] = None,
    resume: bool = False,
    log_level: str = "INFO",
    max_processing_workers: int = 10,
    cutout_size: Optional[str] = None,
    cutout_center: Optional[str] = None,
    sigma_threshold: float = 5.0,
    bad_flags: Optional[List[int]] = None
) -> None:
    """
    Convenience function to run the pipeline.

    Parameters
    ----------
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    output_dir : Path, optional
        Output directory (default: current directory)
    bands : List[str], optional
        Bands to query (e.g., ['D1', 'D2'])
    aperture_diameter : float
        Aperture diameter in pixels (default: 3)
    source_name : str, optional
        Name of the source
    resume : bool
        Whether to resume from saved state
    log_level : str
        Logging level
    max_processing_workers : int
        Number of worker processes for photometry (default: 10)
    cutout_size : str, optional
        Cutout size parameter (e.g., "200px", "3arcmin")
    cutout_center : str, optional
        Cutout center parameter (e.g., "70,20") or None to use source position
    sigma_threshold : float
        Minimum SNR (flux/flux_err) for quality control (default: 5.0)
    bad_flags : List[int], optional
        List of bad flag bit positions to reject (default: [0, 1, 2, 6, 7, 9, 10, 11, 15])
    """
    # Set up logging
    setup_logging(log_level)
    
    # Create configuration
    from ..core.config import Source
    
    source = Source(ra=ra, dec=dec, name=source_name)
    config = QueryConfig(
        source=source,
        output_dir=output_dir or Path.cwd(),
        bands=bands,
        aperture_diameter=aperture_diameter,
        max_processing_workers=max_processing_workers,
        cutout_size=cutout_size,
        cutout_center=cutout_center,
        sigma_threshold=sigma_threshold,
        bad_flags=bad_flags if bad_flags is not None else [0, 1, 2, 6, 7, 9, 10, 11, 15]
    )
    
    # Create and run pipeline
    pipeline = SPXQueryPipeline(config)
    
    if resume:
        pipeline.resume()
    else:
        pipeline.run_full_pipeline()