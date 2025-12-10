"""Generic file download utilities.

This module provides both async and sync download functions using httpx.
It is used by CMF and Pucobre downloaders for fetching PDFs and XBRL files.

Functions
---------
download_file : Async download with httpx.AsyncClient
download_file_sync : Sync download with httpx.Client (for CLI scripts)

Notes
-----
Both functions auto-create parent directories and follow redirects.
Logging is configured via puco_eeff.config.setup_logging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from puco_eeff.config import setup_logging

if TYPE_CHECKING:
    from pathlib import Path

# Module-level logger for download operations
logger = setup_logging(__name__)


async def download_file(url: str, destination: Path, request_timeout: float = 60.0) -> Path:
    """Download a file from URL to destination asynchronously.

    Uses httpx.AsyncClient for non-blocking I/O. Preferred when running
    in an async context (e.g., with asyncio.run or inside async functions).

    Parameters
    ----------
    url : str
        URL to download from (supports redirects).
    destination : Path
        Absolute path where the file should be saved.
    request_timeout : float, optional
        HTTP request timeout in seconds. Default 60.0.

    Returns
    -------
    Path
        Path to the downloaded file (same as destination).

    Raises
    ------
    httpx.HTTPError
        If HTTP request fails (4xx, 5xx, connection error).

    Notes
    -----
    Parent directories are created automatically if they don't exist.
    File is written atomically in binary mode.
    """
    # Ensure destination directory exists before writing
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading: %s", url)
    logger.debug("Destination: %s", destination)

    # Async client with redirect following for CDN-hosted files
    async with httpx.AsyncClient(timeout=request_timeout, follow_redirects=True) as async_http:
        http_response = await async_http.get(url)
        http_response.raise_for_status()  # Raise on 4xx/5xx

        # Write response content in binary mode
        with destination.open("wb") as output_file:
            output_file.write(http_response.content)

    file_stats = destination.stat()
    logger.info("Downloaded: %s (%d bytes)", destination.name, file_stats.st_size)
    return destination


def download_file_sync(url: str, destination: Path, timeout: float = 60.0) -> Path:
    """Download file synchronously using blocking I/O.

    Uses httpx.Client for blocking requests. Use this when running in
    synchronous code without an event loop, such as in CLI scripts.

    Parameters
    ----------
    url : str
        URL to download from (supports redirects).
    destination : Path
        Absolute path where the file should be saved.
    timeout : float, optional
        HTTP request timeout in seconds. Default 60.0.

    Returns
    -------
    Path
        Path to the downloaded file (same as destination).

    Raises
    ------
    httpx.HTTPError
        If HTTP request fails (4xx, 5xx, connection error).

    Notes
    -----
    Parent directories are created automatically if they don't exist.
    This is the blocking version - use download_file for async contexts.
    """
    # Ensure destination directory exists before writing
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading: %s", url)
    logger.debug("Destination: %s", destination)

    # Sync client for non-async code paths
    with httpx.Client(timeout=timeout, follow_redirects=True) as sync_client:
        resp = sync_client.get(url)
        resp.raise_for_status()  # Raise on 4xx/5xx
        destination.write_bytes(resp.content)

    downloaded_size = destination.stat().st_size
    logger.info(f"Downloaded: {destination.name} ({downloaded_size} bytes)")
    return destination
