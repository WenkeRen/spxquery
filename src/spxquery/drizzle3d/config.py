"""
Drizzle3D configuration for SPHEREx 3D spectral image drizzle.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


@dataclass
class Drizzle3DConfig:
    """Configuration for SPHEREx 3D spectral image drizzle.

    Sky-region-driven: user specifies where on the sky and how big the output
    cube should be. The module queries IRSA, downloads matching observations,
    and drizzles them.

    Two independent parameter groups:
    - **Shrink** (xy_shrink / z_shrink): controls input droplet size during
      overlap calculation. Frequently tuned based on data volume.
    - **Oversample** (xy_oversample / z_oversample): controls output grid
      resolution. Default = native SPHEREx resolution. Rarely changed.
    """

    # ── Target sky region (required) ───────────────────────────────────────
    center_ra: float  # Output projection center RA [deg]
    center_dec: float  # Output projection center Dec [deg]
    width: float  # Output region width [arcmin]
    height: float  # Output region height [arcmin]

    # ── Detector / band selection ──────────────────────────────────────────
    detector: int = 0  # 0 = process all detectors D1–D6; 1–6 = single detector

    # ── Droplet shrink (input kernel, frequently tuned) ────────────────────
    xy_shrink: float = 0.8  # Spatial droplet shrink (0, 1]
    z_shrink: Optional[float] = None  # Spectral; None = use xy_shrink

    # ── Output grid oversample (rarely changed) ────────────────────────────
    xy_oversample: float = 1.0  # Spatial: 1.0 → 6.15"/pix, 2.0 → 3.075"/pix
    z_oversample: float = 1.0  # Spectral: 1.0 → 17 native bins, 2.0 → ~34
    z_lambda_edges: Optional[np.ndarray] = None  # Custom Z bin edges [μm]

    # ── Query / download ───────────────────────────────────────────────────
    mjd_range: Optional[Tuple[float, float]] = None  # (mjd_min, mjd_max); None = all
    max_images: int = 500  # Safety cap
    download_workers: int = 4  # Parallel download threads
    skip_existing: bool = True  # Skip already-downloaded FITS files
    data_mirror: Optional[Path] = None  # Local mirror root (contains qr2/, qr3/, ...)
    # Points to the directory that holds data release folders (qr2, qr3, ...).
    # The path mapping strips the IRSA IBE prefix (/ibe/data/spherex/) from the
    # download URL and prepends data_mirror, e.g.:
    #   URL:    .../ibe/data/spherex/qr2/level2/.../xxx.fits
    #   Mirror: data_mirror/qr2/level2/.../xxx.fits
    # Set to None (default) to download from IRSA via HTTP.

    # ── Pre-drizzle processing ─────────────────────────────────────────────
    subtract_zodi: bool = True  # Subtract ZODI extension before drizzling
    static_zodi: bool = False  # True = use zodi model as-is (scale=1.0), skip fitting
    exclude_flags: List[int] = field(default_factory=list)
    # Flag bit positions: pixels with ANY of these flags are excluded.
    # Default: empty (no exclusion).
    # Example: [0, 1, 2] to exclude TRANSIENT, OVERFLOW, SUR_ERROR.

    # ── Accumulation ───────────────────────────────────────────────────────
    ivar_max: float = 1e10  # Inverse-variance cap
    min_overlap: float = 0.0  # Minimum f_xy × f_z to accumulate

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir: Path = Path("drizzle_output")
    overwrite: bool = False  # Overwrite existing output FITS files

    # ── Constants ──────────────────────────────────────────────────────────
    BASE_PIXSCALE: float = field(default=6.15, init=False, repr=False)  # arcsec/pixel

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.center_ra <= 360:
            raise ValueError(f"center_ra must be 0–360 deg, got {self.center_ra}")
        if not -90 <= self.center_dec <= 90:
            raise ValueError(f"center_dec must be -90 to 90 deg, got {self.center_dec}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width, height must be > 0 arcmin, got ({self.width}, {self.height})")
        if not 0 < self.xy_shrink <= 1.0:
            raise ValueError(f"xy_shrink must be in (0, 1], got {self.xy_shrink}")
        z_s = self.z_shrink if self.z_shrink is not None else self.xy_shrink
        if not 0 < z_s <= 1.0:
            raise ValueError(f"z_shrink must be in (0, 1], got {z_s}")
        if self.xy_oversample <= 0:
            raise ValueError(f"xy_oversample must be > 0, got {self.xy_oversample}")
        if self.z_oversample <= 0:
            raise ValueError(f"z_oversample must be > 0, got {self.z_oversample}")
        if self.detector != 0 and not 1 <= self.detector <= 6:
            raise ValueError(f"detector must be 0 (all) or 1–6, got {self.detector}")
        if not 0 <= self.min_overlap < 1:
            raise ValueError(f"min_overlap must be in [0, 1), got {self.min_overlap}")
        if self.ivar_max <= 0:
            raise ValueError(f"ivar_max must be > 0, got {self.ivar_max}")

    def effective_pixscale(self) -> float:
        """Output pixel scale [arcsec] after oversampling."""
        return self.BASE_PIXSCALE / self.xy_oversample

    def effective_z_shrink(self) -> float:
        """Spectral droplet shrink factor (z_shrink if set, else xy_shrink)."""
        return self.z_shrink if self.z_shrink is not None else self.xy_shrink

    def output_nx(self) -> int:
        """Output grid width in pixels."""
        return int(self.width * 60 / self.effective_pixscale())

    def output_ny(self) -> int:
        """Output grid height in pixels."""
        return int(self.height * 60 / self.effective_pixscale())

    def spatial_radius_deg(self) -> float:
        """Half-diagonal of the output region in degrees (for TAP INTERSECTS query)."""
        half_diag_arcmin = (self.width**2 + self.height**2) ** 0.5 / 2
        return half_diag_arcmin / 60.0

    # Fields excluded from serialization (non-init or non-serializable)
    _SERIAL_EXCLUDE = frozenset({"z_lambda_edges", "BASE_PIXSCALE"})

    def _to_dict(self) -> Dict:
        """Convert to a plain dict for serialization (YAML/JSON).

        Strips non-serializable fields (z_lambda_edges ndarray) and
        non-init fields (BASE_PIXSCALE). Converts Path → str and
        tuple → list for YAML compatibility.
        """
        d = asdict(self)
        for key in self._SERIAL_EXCLUDE:
            d.pop(key, None)
        d["output_dir"] = str(d["output_dir"])
        if d.get("data_mirror") is not None:
            d["data_mirror"] = str(d["data_mirror"])
        if d.get("mjd_range") is not None:
            d["mjd_range"] = list(d["mjd_range"])
        return d

    @classmethod
    def _from_dict(cls, data: Dict) -> "Drizzle3DConfig":
        """Reconstruct config from a deserialized dict.

        Converts str → Path, list → tuple (mjd_range), and strips
        non-init fields that may have been left in the dict.
        """
        data = dict(data)  # shallow copy
        data["output_dir"] = Path(data["output_dir"])
        if data.get("data_mirror") is not None:
            data["data_mirror"] = Path(data["data_mirror"])
        if data.get("mjd_range") is not None:
            data["mjd_range"] = tuple(data["mjd_range"])
        data.setdefault("z_lambda_edges", None)
        data.setdefault("data_mirror", None)
        for key in cls._SERIAL_EXCLUDE:
            data.pop(key, None)
        return cls(**data)

    def to_json(self) -> str:
        """Serialize config to JSON (excludes array fields)."""
        return json.dumps(self._to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Drizzle3DConfig":
        """Deserialize config from JSON string."""
        return cls._from_dict(json.loads(json_str))

    def to_yaml_file(self, filepath: Path) -> None:
        """Save config to a YAML file with descriptive header comments.

        Parameters
        ----------
        filepath : Path
            Output YAML file path. Parent directories are created if needed.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write("# SPHEREx Drizzle3D Configuration\n")
            f.write("# Edit values below, then load via Drizzle3DConfig.from_yaml_file()\n")
            f.write("#\n")
            f.write("# Notes:\n")
            f.write("#   - z_lambda_edges (custom spectral edges) is not stored here;\n")
            f.write("#     set it programmatically after loading if needed.\n")
            f.write("#   - mjd_range is a list [min, max] or null.\n")
            f.write("#   - exclude_flags is a list of flag bit positions.\n\n")
            yaml.dump(self._to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml_file(cls, filepath: Path) -> "Drizzle3DConfig":
        """Load config from a YAML file.

        Parameters
        ----------
        filepath : Path
            Path to the YAML config file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file does not contain a valid YAML dict.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML dict, got {type(data).__name__}")
        return cls._from_dict(data)
