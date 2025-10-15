"""
Pipeline for processing SPHEREx quasar variability analysis.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List
import warnings

from .variability import process_lightcurve_variability
from ..visualization.variability_plots import plot_variability_qa


def read_lightcurve_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read lightcurve data from CSV file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the lightcurve CSV file
        
    Returns
    -------
    data : pd.DataFrame
        Lightcurve data
    """
    # Read CSV, skipping comment lines
    data = pd.read_csv(file_path, comment='#')
    
    # Ensure required columns exist
    required_cols = ['mjd', 'mag_ab', 'mag_ab_error', 'wavelength', 'bandwidth', 'band']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter out upper limits if requested
    if 'is_upper_limit' in data.columns:
        data = data[~data['is_upper_limit']]
    
    return data


def extract_source_info(file_path: Union[str, Path]) -> Dict[str, str]:
    """
    Extract source information from file header.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the lightcurve CSV file
        
    Returns
    -------
    info : dict
        Source information (name, RA, Dec, etc.)
    """
    info = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            if 'Source:' in line:
                info['source_name'] = line.split('Source:')[1].strip()
            elif 'RA:' in line:
                info['ra'] = float(line.split('RA:')[1].strip())
            elif 'Dec:' in line:
                info['dec'] = float(line.split('Dec:')[1].strip())
    
    return info


def process_detector_variability(data: pd.DataFrame,
                               detector: str,
                               output_dir: Union[str, Path],
                               source_name: str = "Source",
                               target_wavelength: Optional[float] = None,
                               min_points: int = 50,
                               save_plots: bool = True,
                               save_data: bool = True) -> Optional[Dict]:
    """
    Process variability for a single detector.
    
    Parameters
    ----------
    data : pd.DataFrame
        Lightcurve data for the detector
    detector : str
        Detector name (e.g., 'D3')
    output_dir : str or Path
        Output directory for results
    source_name : str, optional
        Source name for labeling
    target_wavelength : float, optional
        Target wavelength for correction
    min_points : int, optional
        Minimum points required for processing
    save_plots : bool, optional
        Whether to save QA plots
    save_data : bool, optional
        Whether to save processed data
        
    Returns
    -------
    results : dict or None
        Processing results, or None if insufficient data
    """
    print(f"\nProcessing detector {detector}...")
    
    # Check if enough data points
    if len(data) < min_points:
        print(f"  Skipping {detector}: only {len(data)} points (< {min_points})")
        return None
    
    try:
        # Process variability
        results = process_lightcurve_variability(
            data,
            target_wavelength=target_wavelength,
            min_points=min_points
        )
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save QA plots
        if save_plots:
            plot_path = output_path / f"{source_name}_{detector}_variability_qa.png"
            fig = plot_variability_qa(
                results,
                source_name=f"{source_name} - {detector}",
                save_path=str(plot_path)
            )
            plt.close(fig)
        
        # Save processed data
        if save_data:
            # Save corrected lightcurve
            lc_path = output_path / f"{source_name}_{detector}_corrected_lightcurve.csv"
            results['corrected_lightcurve'].to_csv(lc_path, index=False)
            
            # Save daily statistics
            daily_path = output_path / f"{source_name}_{detector}_daily_stats.csv"
            results['daily_stats'].to_csv(daily_path, index=False)
            
            # Save SED model points
            sed_path = output_path / f"{source_name}_{detector}_sed_model.csv"
            sed_df = pd.DataFrame({
                'wavelength': results['sed_model']['wave_grid'],
                'magnitude': results['sed_model']['sed_mag']
            })
            sed_df.to_csv(sed_path, index=False)
            
            print(f"  Results saved to: {output_path}")
        
        # Print summary statistics
        daily_mags = results['daily_stats']['mag_median'].values
        print(f"  Processed {len(data)} observations -> {len(daily_mags)} daily epochs")
        print(f"  Magnitude range: {np.min(daily_mags):.3f} - {np.max(daily_mags):.3f}")
        print(f"  Standard deviation: {np.std(daily_mags):.3f} mag")
        
        return results
        
    except Exception as e:
        print(f"  Error processing {detector}: {str(e)}")
        warnings.warn(f"Failed to process detector {detector}: {str(e)}")
        return None


def run_variability_pipeline(lightcurve_file: Union[str, Path],
                           output_dir: Union[str, Path],
                           detectors: Optional[List[str]] = None,
                           target_wavelength: Optional[float] = None,
                           min_points: int = 50,
                           save_plots: bool = True,
                           save_data: bool = True) -> Dict[str, Dict]:
    """
    Run the full variability analysis pipeline.
    
    Parameters
    ----------
    lightcurve_file : str or Path
        Path to the lightcurve CSV file
    output_dir : str or Path
        Output directory for results
    detectors : list of str, optional
        Specific detectors to process. If None, processes all with sufficient data
    target_wavelength : float, optional
        Target wavelength for correction. If None, uses median per detector
    min_points : int, optional
        Minimum points required per detector (default: 50)
    save_plots : bool, optional
        Whether to save QA plots (default: True)
    save_data : bool, optional
        Whether to save processed data (default: True)
        
    Returns
    -------
    all_results : dict
        Dictionary mapping detector names to their results
    """
    print(f"Starting variability analysis pipeline...")
    print(f"Input file: {lightcurve_file}")
    print(f"Output directory: {output_dir}")
    
    # Read data and source info
    data = read_lightcurve_data(lightcurve_file)
    source_info = extract_source_info(lightcurve_file)
    source_name = source_info.get('source_name', 'Unknown')
    
    print(f"\nSource: {source_name}")
    print(f"Total observations: {len(data)}")
    
    # Get unique detectors
    available_detectors = data['band'].unique()
    if detectors is None:
        detectors = available_detectors
    else:
        # Validate requested detectors
        invalid = [d for d in detectors if d not in available_detectors]
        if invalid:
            warnings.warn(f"Detectors not found in data: {invalid}")
        detectors = [d for d in detectors if d in available_detectors]
    
    print(f"Processing detectors: {detectors}")
    
    # Process each detector
    all_results = {}
    
    for detector in detectors:
        detector_data = data[data['band'] == detector].copy()
        
        results = process_detector_variability(
            detector_data,
            detector,
            output_dir,
            source_name=source_name.replace(' ', '_').replace('/', '_'),
            target_wavelength=target_wavelength,
            min_points=min_points,
            save_plots=save_plots,
            save_data=save_data
        )
        
        if results is not None:
            all_results[detector] = results
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Pipeline completed!")
    print(f"Successfully processed: {list(all_results.keys())}")
    print(f"Output saved to: {output_dir}")
    
    return all_results


# Make matplotlib import conditional to avoid issues if not installed
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    warnings.warn("matplotlib not available, plotting disabled")