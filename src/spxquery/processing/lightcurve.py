"""
Light curve generation and CSV export for SPHEREx time-domain analysis.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..core.config import PhotometryResult, Source
from ..utils.spherex_mef import format_flag_binary

logger = logging.getLogger(__name__)


def generate_lightcurve_dataframe(photometry_results: List[PhotometryResult], source: Source) -> pd.DataFrame:
    """
    Generate light curve DataFrame from photometry results.

    Parameters
    ----------
    photometry_results : List[PhotometryResult]
        List of photometry measurements
    source : Source
        Source information

    Returns
    -------
    pd.DataFrame
        Light curve data with all measurements
    """
    if not photometry_results:
        logger.warning("No photometry results to generate light curve")
        return pd.DataFrame()

    # Convert to records
    records = []
    for result in photometry_results:
        record = {
            "obs_id": result.obs_id,
            "mjd": result.mjd,
            "flux": result.flux,
            "flux_error": result.flux_error,
            "mag_ab": result.mag_ab,
            "mag_ab_error": result.mag_ab_error,
            "wavelength": result.wavelength,
            "bandwidth": result.bandwidth,
            "band": result.band,
            "flag": result.flag,
            "flag_binary": format_flag_binary(result.flag),
            "pix_x": result.pix_x,
            "pix_y": result.pix_y,
            "is_upper_limit": result.is_upper_limit,
            "snr": result.flux / result.flux_error if result.flux_error > 0 else 0,
        }
        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Sort by MJD
    df = df.sort_values("mjd").reset_index(drop=True)

    # Add source information as attributes
    df.attrs["source_ra"] = source.ra
    df.attrs["source_dec"] = source.dec
    df.attrs["source_name"] = source.name or f"RA{source.ra:.4f}_Dec{source.dec:.4f}"
    df.attrs["generated_at"] = datetime.now().isoformat()

    logger.info(f"Generated light curve with {len(df)} measurements")

    return df


def save_lightcurve_csv(df: pd.DataFrame, output_path: Path, include_metadata: bool = True) -> None:
    """
    Save light curve DataFrame to CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Light curve data
    output_path : Path
        Path for output CSV file
    include_metadata : bool
        Whether to include metadata header
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if include_metadata and df.attrs:
        # Write metadata as comments
        with open(output_path, "w") as f:
            f.write("# SPHEREx Light Curve Data\n")
            f.write(f"# Source: {df.attrs.get('source_name', 'Unknown')}\n")
            f.write(f"# RA: {df.attrs.get('source_ra', 'N/A')}\n")
            f.write(f"# Dec: {df.attrs.get('source_dec', 'N/A')}\n")
            f.write(f"# Generated: {df.attrs.get('generated_at', 'N/A')}\n")
            f.write(f"# Number of observations: {len(df)}\n")
            f.write("#\n")

            # Write column descriptions
            f.write("# Column descriptions:\n")
            f.write("# - obs_id: Observation identifier\n")
            f.write("# - mjd: Modified Julian Date\n")
            f.write("# - flux: Flux density in microJansky (μJy)\n")
            f.write("# - flux_error: Flux density uncertainty in microJansky (μJy)\n")
            f.write("# - mag_ab: AB magnitude\n")
            f.write("# - mag_ab_error: AB magnitude uncertainty\n")
            f.write("# - wavelength: Central wavelength in microns\n")
            f.write("# - bandwidth: Bandwidth in microns\n")
            f.write("# - band: SPHEREx detector band (D1-D6)\n")
            f.write("# - flag: Combined flag bitmap (integer)\n")
            f.write("# - flag_binary: Flag bitmap in binary format\n")
            f.write("# - pix_x, pix_y: Pixel coordinates\n")
            f.write("# - is_upper_limit: True if flux_error > flux\n")
            f.write("# - snr: Signal-to-noise ratio\n")
            f.write("#\n")

            # Write the data
            df.to_csv(f, index=False)
    else:
        # Simple CSV without metadata
        df.to_csv(output_path, index=False)

    logger.info(f"Saved light curve to {output_path}")


def summarize_lightcurve(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for light curve.

    Parameters
    ----------
    df : pd.DataFrame
        Light curve data

    Returns
    -------
    Dict[str, Any]
        Summary statistics
    """
    if df.empty:
        return {}

    summary = {
        "n_observations": len(df),
        "n_bands": df["band"].nunique(),
        "bands": sorted(df["band"].unique()),
        "mjd_range": (df["mjd"].min(), df["mjd"].max()),
        "time_span_days": df["mjd"].max() - df["mjd"].min(),
        "wavelength_range": (df["wavelength"].min(), df["wavelength"].max()),
        "n_upper_limits": df["is_upper_limit"].sum(),
        "mean_snr": df["snr"].mean(),
        "observations_per_band": df["band"].value_counts().to_dict(),
    }

    return summary


def load_lightcurve_from_csv(csv_path: Path) -> List[PhotometryResult]:
    """
    Load photometry results from saved lightcurve CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the lightcurve CSV file

    Returns
    -------
    List[PhotometryResult]
        List of photometry measurements
    """
    if not csv_path.exists():
        logger.warning(f"Lightcurve CSV file not found: {csv_path}")
        return []

    try:
        # Read CSV file, skipping comment lines
        df = pd.read_csv(csv_path, comment="#")

        # Convert DataFrame back to PhotometryResult objects
        photometry_results = []
        for _, row in df.iterrows():
            # Handle AB magnitude columns (may be None/NaN for older CSV files)
            mag_ab = row.get("mag_ab")
            mag_ab = None if pd.isna(mag_ab) else float(mag_ab)

            mag_ab_error = row.get("mag_ab_error")
            mag_ab_error = None if pd.isna(mag_ab_error) else float(mag_ab_error)

            result = PhotometryResult(
                obs_id=row["obs_id"],
                mjd=float(row["mjd"]),
                flux=float(row["flux"]),
                flux_error=float(row["flux_error"]),
                wavelength=float(row["wavelength"]),
                bandwidth=float(row["bandwidth"]),
                band=str(row["band"]),
                flag=int(row["flag"]),
                pix_x=float(row["pix_x"]),
                pix_y=float(row["pix_y"]),
                mag_ab=mag_ab,
                mag_ab_error=mag_ab_error,
            )
            # Note: is_upper_limit is a computed property (flux_error > flux), not a constructor parameter
            photometry_results.append(result)

        logger.info(f"Loaded {len(photometry_results)} photometry results from {csv_path}")
        return photometry_results

    except Exception as e:
        logger.error(f"Failed to load lightcurve from CSV: {e}")
        return []


def print_lightcurve_summary(df: pd.DataFrame) -> None:
    """
    Print summary of light curve data.

    Parameters
    ----------
    df : pd.DataFrame
        Light curve data
    """
    if df.empty:
        print("No light curve data available")
        return

    summary = summarize_lightcurve(df)

    print(f"\n{'=' * 60}")
    print("Light Curve Summary")
    print(f"{'=' * 60}")
    print(f"Total observations: {summary['n_observations']}")
    print(f"Time span: {summary['time_span_days']:.1f} days")
    print(f"MJD range: {summary['mjd_range'][0]:.2f} - {summary['mjd_range'][1]:.2f}")
    print(f"Wavelength range: {summary['wavelength_range'][0]:.2f} - {summary['wavelength_range'][1]:.2f} μm")
    print(f"Number of bands: {summary['n_bands']}")
    print(f"Upper limits: {summary['n_upper_limits']}")
    print(f"Mean SNR: {summary['mean_snr']:.1f}")

    print("\nObservations per band:")
    for band, count in sorted(summary["observations_per_band"].items()):
        print(f"  {band}: {count}")

    print(f"{'=' * 60}\n")
