"""
Parallel download manager for SPHEREx FITS files.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
from urllib.request import Request, urlopen

from tqdm import tqdm

from .config import DownloadResult, ObservationInfo

logger = logging.getLogger(__name__)

# Download configuration
CHUNK_SIZE = 8192  # 8KB chunks
DEFAULT_TIMEOUT = 300  # 5 minutes
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def download_file(
    url: str, output_path: Path, timeout: int = DEFAULT_TIMEOUT, retries: int = MAX_RETRIES
) -> DownloadResult:
    """
    Download a single file with retry logic.

    Parameters
    ----------
    url : str
        URL to download
    output_path : Path
        Local path to save file
    timeout : int
        Download timeout in seconds
    retries : int
        Number of retry attempts

    Returns
    -------
    DownloadResult
        Result of download attempt
    """
    attempt = 0
    last_error = None

    while attempt <= retries:
        try:
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Set up request with headers
            request = Request(url, headers={"User-Agent": "SPXQuery/0.1.0"})

            # Open connection
            with urlopen(request, timeout=timeout) as response:
                # Get file size if available
                total_size = int(response.headers.get("Content-Length", 0))

                # Download file
                with open(output_path, "wb") as f:
                    downloaded = 0
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                # Verify download
                actual_size = output_path.stat().st_size
                if total_size > 0 and actual_size != total_size:
                    raise RuntimeError(f"Download incomplete: expected {total_size} bytes, got {actual_size}")

                return DownloadResult(url=url, local_path=output_path, success=True, size_mb=actual_size / 1024 / 1024)

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}")

            # Remove partial file
            if output_path.exists():
                output_path.unlink()

            attempt += 1
            if attempt <= retries:
                time.sleep(RETRY_DELAY)

    # All attempts failed
    return DownloadResult(
        url=url, local_path=output_path, success=False, error=f"Failed after {retries + 1} attempts: {last_error}"
    )


def parallel_download(
    download_info: List[Tuple[ObservationInfo, str]],
    output_dir: Path,
    max_workers: int = 4,
    show_progress: bool = True,
    skip_existing: bool = True,
) -> List[DownloadResult]:
    """
    Download multiple files in parallel with progress tracking.

    Parameters
    ----------
    download_info : List[Tuple[ObservationInfo, str]]
        List of (observation, download_url) tuples
    output_dir : Path
        Directory to save downloaded files
    max_workers : int
        Maximum number of parallel downloads
    show_progress : bool
        Whether to show progress bar
    skip_existing : bool
        If True, skip files that already exist. If False, re-download all files.

    Returns
    -------
    List[DownloadResult]
        Results for all download attempts
    """
    logger.info(f"Starting parallel download of {len(download_info)} files")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    # Organize downloads by band
    for band in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        (output_dir / band).mkdir(exist_ok=True)

    results = []

    # Set up progress bar
    if show_progress:
        pbar = tqdm(total=len(download_info), desc="Downloading", unit="files")

    # Download files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all downloads
        future_to_info = {}
        for obs, url in download_info:
            # Generate output filename
            filename = f"{obs.obs_id}.fits"
            output_path = output_dir / obs.band / filename

            # Skip if already downloaded (and skip_existing is True)
            if skip_existing and output_path.exists():
                logger.info(f"Skipping {filename} - already exists")
                results.append(
                    DownloadResult(
                        url=url, local_path=output_path, success=True, size_mb=output_path.stat().st_size / 1024 / 1024
                    )
                )
                if show_progress:
                    pbar.update(1)
                continue

            # Submit download task
            future = executor.submit(download_file, url, output_path)
            future_to_info[future] = (obs, url)

        # Process completed downloads
        for future in as_completed(future_to_info):
            obs, url = future_to_info[future]

            try:
                result = future.result()
                results.append(result)

                if result.success:
                    logger.info(f"Downloaded {obs.obs_id} ({result.size_mb:.1f} MB)")
                else:
                    logger.error(f"Failed to download {obs.obs_id}: {result.error}")

            except Exception as e:
                logger.error(f"Unexpected error downloading {obs.obs_id}: {e}")
                results.append(
                    DownloadResult(
                        url=url, local_path=output_dir / obs.band / f"{obs.obs_id}.fits", success=False, error=str(e)
                    )
                )

            if show_progress:
                pbar.update(1)

    if show_progress:
        pbar.close()

    # Summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    total_size = sum(r.size_mb for r in results if r.success and r.size_mb)

    logger.info(f"Download complete: {successful} successful, {failed} failed")
    logger.info(f"Total size: {total_size:.1f} MB")

    return results


def print_download_summary(results: List[DownloadResult]) -> None:
    """
    Print a summary of download results.

    Parameters
    ----------
    results : List[DownloadResult]
        Download results to summarize
    """
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\n{'=' * 60}")
    print("Download Summary")
    print(f"{'=' * 60}")
    print(f"Total files: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        total_size = sum(r.size_mb for r in successful if r.size_mb)
        print(f"Total downloaded: {total_size:.1f} MB ({total_size / 1024:.2f} GB)")

    if failed:
        print("\nFailed downloads:")
        for r in failed[:5]:  # Show first 5 failures
            print(f"  - {r.local_path.name}: {r.error}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    print(f"{'=' * 60}\n")
