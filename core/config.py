"""
Configuration and data models for SPXQuery package.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Source:
    """Astronomical source coordinates."""
    ra: float  # Right ascension in degrees
    dec: float  # Declination in degrees
    name: Optional[str] = None
    
    def __post_init__(self):
        if not 0 <= self.ra <= 360:
            raise ValueError(f"RA must be between 0 and 360 degrees, got {self.ra}")
        if not -90 <= self.dec <= 90:
            raise ValueError(f"Dec must be between -90 and 90 degrees, got {self.dec}")


@dataclass
class QueryConfig:
    """Configuration for SPHEREx data query and processing."""
    source: Source
    output_dir: Path = field(default_factory=Path.cwd)
    bands: Optional[List[str]] = None  # e.g., ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    aperture_diameter: float = 3.0  # pixels (default 3 pixel diameter)
    max_download_workers: int = 4
    max_processing_workers: int = 10  # Number of workers for photometry processing
    cutout_size: Optional[str] = None  # e.g., "200px", "100,200px", "3arcmin", "0.1"
    cutout_center: Optional[str] = None  # e.g., "70,20", "300.5,120px" (optional, defaults to source position)
    
    def __post_init__(self):
        # Convert to Path if string
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Validate bands
        valid_bands = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
        if self.bands:
            invalid = set(self.bands) - set(valid_bands)
            if invalid:
                raise ValueError(f"Invalid bands: {invalid}. Valid bands are: {valid_bands}")

        # Validate aperture diameter
        if self.aperture_diameter <= 0:
            raise ValueError(f"Aperture diameter must be positive, got {self.aperture_diameter}")

        # Validate cutout parameters
        if self.cutout_size:
            from ..utils.helpers import validate_cutout_size
            if not validate_cutout_size(self.cutout_size):
                raise ValueError(f"Invalid cutout_size format: '{self.cutout_size}'. "
                               "Expected format: <value>[,<value>][units], e.g., '200px', '3arcmin', '0.1'")

        if self.cutout_center:
            from ..utils.helpers import validate_cutout_center
            if not validate_cutout_center(self.cutout_center):
                raise ValueError(f"Invalid cutout_center format: '{self.cutout_center}'. "
                               "Expected format: <x>,<y>[units], e.g., '70,20', '300.5,120px'")


@dataclass
class ObservationInfo:
    """Information about a single SPHEREx observation."""
    obs_id: str
    band: str
    mjd: float
    ra: float
    dec: float
    wavelength_min: float  # microns
    wavelength_max: float  # microns
    access_url: str
    file_size_mb: float
    t_min: float  # MJD
    t_max: float  # MJD
    
    @property
    def wavelength_center(self) -> float:
        """Central wavelength in microns."""
        return (self.wavelength_min + self.wavelength_max) / 2
    
    @property
    def bandwidth(self) -> float:
        """Bandwidth in microns."""
        return self.wavelength_max - self.wavelength_min


@dataclass
class QueryResults:
    """Results from SPHEREx archive query."""
    observations: List[ObservationInfo]
    query_time: datetime
    source: Source
    total_size_gb: float
    time_span_days: float
    band_counts: Dict[str, int]
    
    def __len__(self):
        return len(self.observations)
    
    def filter_by_band(self, bands: List[str]) -> 'QueryResults':
        """Return new QueryResults filtered by bands."""
        filtered_obs = [obs for obs in self.observations if obs.band in bands]
        return QueryResults(
            observations=filtered_obs,
            query_time=self.query_time,
            source=self.source,
            total_size_gb=sum(obs.file_size_mb for obs in filtered_obs) / 1024,
            time_span_days=self.time_span_days,
            band_counts={band: sum(1 for obs in filtered_obs if obs.band == band) 
                        for band in bands}
        )


@dataclass
class PhotometryResult:
    """Result from aperture photometry on a single observation."""
    obs_id: str
    mjd: float
    flux: float  # MJy/sr
    flux_error: float  # MJy/sr
    wavelength: float  # microns
    bandwidth: float  # microns
    flag: int  # Combined flag bitmap
    pix_x: float  # Pixel X coordinate
    pix_y: float  # Pixel Y coordinate
    band: str
    mag_ab: Optional[float] = None  # AB magnitude
    mag_ab_error: Optional[float] = None  # AB magnitude error
    
    @property
    def is_upper_limit(self) -> bool:
        """Check if measurement should be treated as upper limit."""
        return self.flux_error > self.flux


@dataclass
class DownloadResult:
    """Result from file download attempt."""
    url: str
    local_path: Path
    success: bool
    error: Optional[str] = None
    size_mb: Optional[float] = None


@dataclass
class PipelineState:
    """State for resumable pipeline execution."""
    stage: str  # 'query', 'download', 'processing', 'visualization', 'complete'
    config: QueryConfig
    query_results: Optional[QueryResults] = None
    downloaded_files: List[Path] = field(default_factory=list)
    photometry_results: List[PhotometryResult] = field(default_factory=list)
    csv_path: Optional[Path] = None
    plot_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'stage': self.stage,
            'config': {
                'source': {
                    'ra': self.config.source.ra,
                    'dec': self.config.source.dec,
                    'name': self.config.source.name
                },
                'output_dir': str(self.config.output_dir),
                'bands': self.config.bands,
                'aperture_diameter': self.config.aperture_diameter,
                'max_download_workers': self.config.max_download_workers,
                'max_processing_workers': self.config.max_processing_workers,
                'cutout_size': self.config.cutout_size,
                'cutout_center': self.config.cutout_center
            },
            'query_results': {
                'observations': [
                    {
                        'obs_id': obs.obs_id,
                        'band': obs.band,
                        'mjd': obs.mjd,
                        'ra': obs.ra,
                        'dec': obs.dec,
                        'wavelength_min': obs.wavelength_min,
                        'wavelength_max': obs.wavelength_max,
                        'access_url': obs.access_url,
                        'file_size_mb': obs.file_size_mb,
                        't_min': obs.t_min,
                        't_max': obs.t_max
                    } for obs in self.query_results.observations
                ] if self.query_results else [],
                'query_time': self.query_results.query_time.isoformat() if self.query_results else None,
                'total_size_gb': self.query_results.total_size_gb if self.query_results else 0,
                'time_span_days': self.query_results.time_span_days if self.query_results else 0,
                'band_counts': self.query_results.band_counts if self.query_results else {}
            } if self.query_results else None,
            'downloaded_files': [str(p) for p in self.downloaded_files],
            'photometry_results': [
                {
                    'obs_id': pr.obs_id,
                    'mjd': pr.mjd,
                    'flux': pr.flux,
                    'flux_error': pr.flux_error,
                    'wavelength': pr.wavelength,
                    'bandwidth': pr.bandwidth,
                    'flag': pr.flag,
                    'pix_x': pr.pix_x,
                    'pix_y': pr.pix_y,
                    'band': pr.band,
                    'mag_ab': pr.mag_ab,
                    'mag_ab_error': pr.mag_ab_error
                } for pr in self.photometry_results
            ],
            'csv_path': str(self.csv_path) if self.csv_path else None,
            'plot_path': str(self.plot_path) if self.plot_path else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineState':
        """Create from dictionary."""
        # Reconstruct config
        source = Source(
            ra=data['config']['source']['ra'],
            dec=data['config']['source']['dec'],
            name=data['config']['source'].get('name')
        )
        config = QueryConfig(
            source=source,
            output_dir=Path(data['config']['output_dir']),
            bands=data['config'].get('bands'),
            aperture_diameter=data['config']['aperture_diameter'],
            max_download_workers=data['config']['max_download_workers'],
            max_processing_workers=data['config'].get('max_processing_workers', 10),
            cutout_size=data['config'].get('cutout_size'),
            cutout_center=data['config'].get('cutout_center')
        )
        
        # Reconstruct query results
        query_results = None
        if data.get('query_results'):
            observations = [
                ObservationInfo(**obs) for obs in data['query_results']['observations']
            ]
            query_results = QueryResults(
                observations=observations,
                query_time=datetime.fromisoformat(data['query_results']['query_time']),
                source=source,
                total_size_gb=data['query_results']['total_size_gb'],
                time_span_days=data['query_results']['time_span_days'],
                band_counts=data['query_results']['band_counts']
            )
        
        # Reconstruct photometry results
        photometry_results = [
            PhotometryResult(**pr) for pr in data.get('photometry_results', [])
        ]
        
        return cls(
            stage=data['stage'],
            config=config,
            query_results=query_results,
            downloaded_files=[Path(p) for p in data.get('downloaded_files', [])],
            photometry_results=photometry_results,
            csv_path=Path(data['csv_path']) if data.get('csv_path') else None,
            plot_path=Path(data['plot_path']) if data.get('plot_path') else None
        )