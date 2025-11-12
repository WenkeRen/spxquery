"""
Main pipeline orchestrator for SPXQuery package with flexible, resumable execution.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

from ..core.config import AdvancedConfig, DownloadResult, PipelineState
from ..core.download import parallel_download, print_download_summary
from ..core.query import print_query_summary, query_spherex_observations
from ..processing.lightcurve import (
    generate_lightcurve_dataframe,
    load_lightcurve_from_csv,
    print_lightcurve_summary,
    save_lightcurve_csv,
)
from ..processing.photometry import process_all_observations
from ..utils.helpers import format_cutout_url_params, get_file_list, load_yaml, save_yaml, setup_logging
from ..visualization.plots import create_combined_plot

logger = logging.getLogger(__name__)


class SPXQueryPipeline:
    """
    Main pipeline for SPHEREx data query, download, and analysis.

    Supports:
    - Flexible stage configuration (add/remove pipeline stages)
    - Full automatic execution or step-by-step mode
    - Resumable execution with state persistence
    - Dependency checking for manual execution
    """

    # Define stage dependencies
    STAGE_DEPENDENCIES = {
        "query": [],
        "download": ["query"],
        "processing": ["query", "download"],
        "visualization": ["query", "download", "processing"],
    }

    def __init__(self, config: AdvancedConfig, pipeline_stages: Optional[List[str]] = None):
        """
        Initialize pipeline with configuration.

        Parameters
        ----------
        config : AdvancedConfig
            Complete pipeline configuration including query, photometry, visualization, and download settings
        pipeline_stages : List[str], optional
            List of stages to execute. If None, uses config.pipeline_stages.
            Default: ['query', 'download', 'processing', 'visualization']
            Allows customization of pipeline flow (e.g., skip visualization, add custom stages)
        """
        self.config = config

        # Set pipeline stages (explicit parameter > config > default)
        if pipeline_stages is not None:
            # Override config's pipeline_stages if explicitly provided
            self.config.pipeline_stages = pipeline_stages

        # Initialize state
        self.state = PipelineState(stage="query", config=config, completed_stages=[])

        # Set up directories
        self.data_dir = config.query.output_dir / "data"
        self.results_dir = config.query.output_dir / "results"
        # State file named after source for easy identification
        source_name = config.query.source.name or f"source_{config.query.source.ra:.4f}_{config.query.source.dec:.4f}"
        self.state_file = config.query.output_dir / f"{source_name}.yaml"

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized pipeline for source at RA={config.query.source.ra}, Dec={config.query.source.dec}")
        logger.info(f"Pipeline stages: {self.config.pipeline_stages}")
        logger.info(f"State file: {self.state_file.name}")

    def save_state(self) -> None:
        """Save current pipeline state to disk."""
        state_dict = self.state.to_dict()
        save_yaml(state_dict, self.state_file)
        logger.info(f"Saved pipeline state: stage={self.state.stage}, completed={self.state.completed_stages}")

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
            state_dict = load_yaml(self.state_file)
            self.state = PipelineState.from_dict(state_dict)
            logger.info(f"Loaded pipeline state: stage={self.state.stage}, completed={self.state.completed_stages}")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    def check_dependencies(self, stage: str) -> None:
        """
        Check if all dependencies for a stage are satisfied.

        Parameters
        ----------
        stage : str
            Stage name to check

        Raises
        ------
        RuntimeError
            If dependencies are not satisfied
        """
        if stage not in self.STAGE_DEPENDENCIES:
            logger.warning(f"Unknown stage '{stage}', cannot check dependencies")
            return

        required_stages = self.STAGE_DEPENDENCIES[stage]
        missing_stages = [s for s in required_stages if s not in self.state.completed_stages]

        if missing_stages:
            raise RuntimeError(
                f"Cannot run stage '{stage}': missing dependencies {missing_stages}. "
                f"Completed stages: {self.state.completed_stages}. "
                f"Please run the following stages first: {missing_stages}"
            )

    def mark_stage_complete(self, stage: str) -> None:
        """
        Mark a stage as completed.

        Parameters
        ----------
        stage : str
            Stage name
        """
        if stage not in self.state.completed_stages:
            self.state.completed_stages.append(stage)
            logger.info(f"Marked stage '{stage}' as complete")

    def get_status_message(self) -> str:
        """
        Get a human-readable status message.

        Returns
        -------
        str
            Status message describing completed and pending stages
        """
        all_stages = self.config.pipeline_stages
        completed = self.state.completed_stages
        pending = [s for s in all_stages if s not in completed]

        msg = "\nPipeline Status:\n"
        msg += f"  Completed stages: {completed if completed else 'None'}\n"
        msg += f"  Pending stages: {pending if pending else 'None'}\n"
        msg += f"  Current stage: {self.state.stage}\n"

        return msg

    def print_status(self) -> None:
        """Print current pipeline status."""
        print(self.get_status_message())

    def _update_file_sizes_from_download(self, download_results: List[DownloadResult]) -> None:
        """
        Update QueryResults with actual total file size from downloaded files.

        After download completes, calculate actual total size and update query summary.
        This is simpler than per-file mapping and provides the key information users need.

        Parameters
        ----------
        download_results : List[DownloadResult]
            Download results with actual file sizes
        """
        if not self.state.query_results or not download_results:
            return

        # Calculate total size from successful downloads
        actual_total_mb = sum(result.size_mb for result in download_results if result.success and result.size_mb)
        actual_total_gb = actual_total_mb / 1024.0
        old_total_gb = self.state.query_results.total_size_gb

        # Update total size in state
        self.state.query_results.total_size_gb = actual_total_gb

        # Log the size comparison
        successful_count = sum(1 for r in download_results if r.success)
        logger.info(f"Downloaded {successful_count} files")
        logger.info(f"Total data size: {old_total_gb:.2f} GB (estimated) → {actual_total_gb:.2f} GB (actual)")

        # Save updated query summary with actual total size
        query_info = {
            "source": {
                "ra": float(self.config.query.source.ra),
                "dec": float(self.config.query.source.dec),
                "name": self.config.query.source.name,
            },
            "query_time": self.state.query_results.query_time.isoformat(),
            "n_observations": len(self.state.query_results),
            "time_span_days": float(self.state.query_results.time_span_days),
            "total_size_gb": float(actual_total_gb),
            "total_size_gb_estimated": False,  # Now using actual sizes
            "band_counts": self.state.query_results.band_counts,
        }
        save_yaml(query_info, self.results_dir / "query_summary.yaml")

    def run_full_pipeline(self, skip_existing_downloads: bool = True) -> None:
        """
        Run the complete pipeline through all configured stages.

        Parameters
        ----------
        skip_existing_downloads : bool
            If True, skip already downloaded files. If False, re-download everything.
        """
        logger.info("Starting full pipeline execution")
        logger.info(f"Pipeline stages: {self.config.pipeline_stages}")

        # Execute each stage in order
        for stage in self.config.pipeline_stages:
            if stage == "query":
                self.run_query()
            elif stage == "download":
                self.run_download(skip_existing=skip_existing_downloads)
            elif stage == "processing":
                self.run_processing()
            elif stage == "visualization":
                self.run_visualization()
            else:
                logger.warning(f"Unknown stage '{stage}', skipping")

        self.state.stage = "complete"
        self.save_state()
        logger.info("Pipeline execution complete")

    def run_query(self) -> None:
        """Execute query stage."""
        logger.info("Running query stage")

        # Query SPHEREx archive with cutout size for accurate file size estimation
        query_results = query_spherex_observations(
            self.config.query.source, self.config.query.bands, cutout_size=self.config.download.cutout_size
        )

        # Print summary
        print_query_summary(query_results)

        # Save query results
        query_info = {
            "source": {
                "ra": float(self.config.query.source.ra),
                "dec": float(self.config.query.source.dec),
                "name": self.config.query.source.name,
            },
            "query_time": query_results.query_time.isoformat(),
            "n_observations": len(query_results),
            "time_span_days": float(query_results.time_span_days),
            "total_size_gb": float(query_results.total_size_gb),
            "total_size_gb_estimated": True,  # Mark as estimated
            "band_counts": query_results.band_counts,
        }
        save_yaml(query_info, self.results_dir / "query_summary.yaml")

        # Update state
        self.state.query_results = query_results
        self.state.stage = "download"
        self.mark_stage_complete("query")
        self.save_state()

    def run_download(self, skip_existing: bool = True) -> None:
        """
        Execute download stage.

        Parameters
        ----------
        skip_existing : bool
            If True, skip files that already exist. If False, re-download all files.
        """
        # Check dependencies
        self.check_dependencies("download")

        if not self.state.query_results:
            raise RuntimeError("No query results available. Run query stage first.")

        logger.info(f"Running download stage (skip_existing={skip_existing})")

        # Construct download URLs with cutout parameters appended on-the-fly
        download_info = []
        for obs in self.state.query_results.observations:
            url = obs.download_url  # Base URL from query

            # Append cutout parameters if specified
            if self.config.download.cutout_size:
                cutout_params = format_cutout_url_params(
                    self.config.download.cutout_size,
                    self.config.download.cutout_center,
                    self.config.query.source.ra,
                    self.config.query.source.dec,
                )
                url = url + cutout_params
                logger.debug(f"Added cutout to {obs.obs_id}: {cutout_params}")

            download_info.append((obs, url))

        if not download_info:
            logger.warning("No observations to download")
            self.state.stage = "processing"
            self.mark_stage_complete("download")
            self.save_state()
            return

        # Download files
        download_results = parallel_download(
            download_info,
            self.data_dir,
            max_workers=self.config.download.max_download_workers,
            skip_existing=skip_existing,
            download_config=self.config.download,
        )

        # Print summary
        print_download_summary(download_results)

        # Update state with downloaded files
        self.state.downloaded_files = [r.local_path for r in download_results if r.success]

        # Update QueryResults with actual file sizes from download
        self._update_file_sizes_from_download(download_results)

        self.state.stage = "processing"
        self.mark_stage_complete("download")
        self.save_state()

    def run_processing(self) -> None:
        """Execute processing stage."""
        # Check dependencies
        self.check_dependencies("processing")

        logger.info("Running processing stage")

        # Get list of downloaded files
        if not self.state.downloaded_files:
            # Try to find files in data directory
            self.state.downloaded_files = get_file_list(self.data_dir, "*.fits")

        if not self.state.downloaded_files:
            logger.warning("No FITS files found for processing")
            self.state.stage = "visualization"
            self.mark_stage_complete("processing")
            self.save_state()
            return

        logger.info(f"Processing {len(self.state.downloaded_files)} FITS files")

        # Process all files
        # All parameters (aperture sizing, subtract_zodi, max_workers, etc.) come from config
        photometry_results = process_all_observations(
            self.state.downloaded_files,
            self.config.query.source,
            photometry_config=self.config.photometry,
        )

        if not photometry_results:
            logger.warning("No photometry results obtained")
            self.state.stage = "complete"
            self.mark_stage_complete("processing")
            self.save_state()
            return

        # Generate light curve
        df = generate_lightcurve_dataframe(photometry_results, self.config.query.source)

        # Save light curve CSV
        csv_path = self.results_dir / "lightcurve.csv"
        save_lightcurve_csv(df, csv_path)

        # Print summary
        print_lightcurve_summary(df)

        # Update state
        self.state.photometry_results = photometry_results
        self.state.csv_path = csv_path
        self.state.stage = "visualization"
        self.mark_stage_complete("processing")
        self.save_state()

    def run_visualization(self) -> None:
        """Execute visualization stage."""
        # Check dependencies
        self.check_dependencies("visualization")

        # Check if photometry results are available in memory
        if not self.state.photometry_results:
            # Try to load from saved lightcurve CSV
            csv_path = self.results_dir / "lightcurve.csv"
            if csv_path.exists():
                logger.info("Loading photometry results from saved lightcurve CSV")
                self.state.photometry_results = load_lightcurve_from_csv(csv_path)
                self.state.csv_path = csv_path

        if not self.state.photometry_results:
            logger.warning("No photometry results available for visualization")
            self.state.stage = "complete"
            self.mark_stage_complete("visualization")
            self.save_state()
            return

        logger.info("Running visualization stage")

        # Filter photometry results by configured bands
        photometry_results = self.state.photometry_results
        if self.config.query.bands is not None:
            # Only keep results for bands in config
            original_count = len(photometry_results)
            photometry_results = [r for r in photometry_results if r.band in self.config.query.bands]
            logger.info(
                f"Filtered photometry results by bands {self.config.query.bands}: "
                f"{original_count} -> {len(photometry_results)} measurements"
            )

            if not photometry_results:
                logger.warning(f"No photometry results match configured bands {self.config.query.bands}")
                self.state.stage = "complete"
                self.mark_stage_complete("visualization")
                self.save_state()
                return

        # Create combined plot with quality control filters
        plot_path = self.results_dir / "combined_plot.png"
        create_combined_plot(
            photometry_results,  # Use filtered results
            plot_path,
            apply_quality_filters=True,
            sigma_threshold=self.config.visualization.sigma_threshold,
            bad_flags=self.config.photometry.bad_flags,
            use_magnitude=self.config.visualization.use_magnitude,
            show_errorbars=self.config.visualization.show_errorbars,
            visualization_config=self.config.visualization,  # Pass visualization config
        )

        # Update state
        self.state.plot_path = plot_path
        self.state.stage = "complete"
        self.mark_stage_complete("visualization")
        self.save_state()

        logger.info(f"Visualization saved to {plot_path}")

    def resume(self, skip_existing_downloads: bool = True) -> None:
        """
        Resume pipeline from saved state.

        Parameters
        ----------
        skip_existing_downloads : bool
            If True, skip files that already exist during download. If False, re-download.
        """
        if not self.load_state():
            logger.warning("No saved state found. Starting from beginning.")
            self.run_full_pipeline(skip_existing_downloads=skip_existing_downloads)
            return

        logger.info("Resuming from saved state")
        self.print_status()

        # Get remaining stages
        remaining_stages = [s for s in self.config.pipeline_stages if s not in self.state.completed_stages]

        if not remaining_stages:
            logger.info("All stages already complete")
            return

        logger.info(f"Running remaining stages: {remaining_stages}")

        # Execute remaining stages
        for stage in remaining_stages:
            if stage == "query":
                self.run_query()
            elif stage == "download":
                self.run_download(skip_existing=skip_existing_downloads)
            elif stage == "processing":
                self.run_processing()
            elif stage == "visualization":
                self.run_visualization()
            else:
                logger.warning(f"Unknown stage '{stage}', skipping")

        self.state.stage = "complete"
        self.save_state()
        logger.info("Resume complete")


def run_pipeline(
    ra: float,
    dec: float,
    output_dir: Optional[Path] = None,
    bands: Optional[List[str]] = None,
    aperture_diameter: float = 3.0,
    source_name: Optional[str] = None,
    resume: bool = False,
    log_level: str = "INFO",
    max_download_workers: int = 4,
    max_processing_workers: int = 10,
    cutout_size: Optional[str] = None,
    cutout_center: Optional[str] = None,
    sigma_threshold: float = 5.0,
    bad_flags: Optional[List[int]] = None,
    use_magnitude: bool = False,
    show_errorbars: bool = True,
    skip_existing_downloads: bool = True,
    pipeline_stages: Optional[List[str]] = None,
    advanced_params_file: Optional[Union[str, Path]] = None,
) -> None:
    """
    Convenience function to run the pipeline with sensible defaults.

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
    max_download_workers : int
        Number of worker threads for downloading (default: 4)
    max_processing_workers : int
        Number of worker processes for photometry (default: 10)
    cutout_size : str, optional
        Cutout size parameter (e.g., "200px", "3arcmin")
    cutout_center : str, optional
        Cutout center parameter (e.g., "70,20") or None to use source position
    sigma_threshold : float
        Minimum SNR (flux/flux_err) for quality control (default: 5.0)
    bad_flags : List[int], optional
        List of bad flag bit positions to reject (default: [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19])
    use_magnitude : bool
        If True, plot AB magnitude instead of flux (default: False)
    show_errorbars : bool
        If True, show errorbars on plots (default: True)
    skip_existing_downloads : bool
        If True, skip already downloaded files. If False, re-download all (default: True)
    pipeline_stages : List[str], optional
        List of stages to execute (default: ['query', 'download', 'processing', 'visualization'])
    advanced_params_file : str or Path, optional
        Path to JSON file with advanced parameters (photometry, visualization, download settings).
        If provided, these parameters are loaded with priority: user input > JSON file > defaults

    Examples
    --------
    >>> # Basic usage
    >>> run_pipeline(ra=304.69, dec=42.44, output_dir="output")

    >>> # With advanced parameters
    >>> from spxquery.utils.params import export_default_parameters
    >>> params_file = export_default_parameters("output")
    >>> # Edit output/spxquery_default_params.json
    >>> run_pipeline(ra=304.69, dec=42.44, advanced_params_file=params_file)
    """
    # Set up logging
    setup_logging(log_level)

    # Create configuration
    from ..core.config import Source

    source = Source(ra=ra, dec=dec, name=source_name)

    # Load advanced parameters from file if provided
    if advanced_params_file:
        from ..utils.params import load_advanced_config

        config = load_advanced_config(Path(advanced_params_file))

        # Override with explicitly provided parameters
        # Build update dict only for non-default parameters
        updates = {}
        updates["source"] = source  # Always use provided source
        updates["output_dir"] = output_dir or Path.cwd()

        if bands is not None:
            updates["bands"] = bands
        if aperture_diameter != 3.0:  # Non-default
            updates["aperture_diameter"] = aperture_diameter
        if max_download_workers != 4:  # Non-default
            updates["max_download_workers"] = max_download_workers
        if max_processing_workers != 10:  # Non-default
            updates["max_processing_workers"] = max_processing_workers
        if cutout_size is not None:
            updates["cutout_size"] = cutout_size
        if cutout_center is not None:
            updates["cutout_center"] = cutout_center
        if sigma_threshold != 5.0:  # Non-default
            updates["sigma_threshold"] = sigma_threshold
        if bad_flags is not None:
            updates["bad_flags"] = bad_flags
        if use_magnitude != False:  # Non-default
            updates["use_magnitude"] = use_magnitude
        if show_errorbars != True:  # Non-default
            updates["show_errorbars"] = show_errorbars

        # Update query config source and output_dir directly
        config.query.source = source
        config.query.output_dir = output_dir or Path.cwd()
        if bands is not None:
            config.query.bands = bands

        # Apply other updates via intelligent routing
        if len(updates) > 3:  # More than just source, output_dir, bands
            remaining_updates = {k: v for k, v in updates.items() if k not in ["source", "output_dir", "bands"]}
            if remaining_updates:
                config.update(**remaining_updates)
    else:
        # No advanced params file - create with defaults and provided parameters
        config = AdvancedConfig.create(
            source=source,
            output_dir=output_dir or Path.cwd(),
            bands=bands,
            aperture_diameter=aperture_diameter,
            max_download_workers=max_download_workers,
            max_processing_workers=max_processing_workers,
            cutout_size=cutout_size,
            cutout_center=cutout_center,
            sigma_threshold=sigma_threshold,
            bad_flags=bad_flags if bad_flags is not None else [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19],
            use_magnitude=use_magnitude,
            show_errorbars=show_errorbars,
        )

    # Set pipeline stages if provided
    if pipeline_stages is not None:
        config.pipeline_stages = pipeline_stages

    # Create and run pipeline
    pipeline = SPXQueryPipeline(config)

    if resume:
        pipeline.resume(skip_existing_downloads=skip_existing_downloads)
    else:
        pipeline.run_full_pipeline(skip_existing_downloads=skip_existing_downloads)
