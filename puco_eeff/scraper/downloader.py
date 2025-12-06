"""Generic file download utilities."""

from __future__ import annotations

from pathlib import Path

import httpx

from puco_eeff.config import setup_logging

logger = setup_logging(__name__)


async def download_file(url: str, destination: Path, timeout: float = 60.0) -> Path:
    """Download a file from URL to destination.

    Args:
        url: URL to download from
        destination: Path to save the file
        timeout: Request timeout in seconds

    Returns:
        Path to the downloaded file

    Raises:
        httpx.HTTPError: If download fails
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading: {url}")
    logger.debug(f"Destination: {destination}")

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        with open(destination, "wb") as f:
            f.write(response.content)

    logger.info(f"Downloaded: {destination.name} ({destination.stat().st_size} bytes)")
    return destination


def download_file_sync(url: str, destination: Path, timeout: float = 60.0) -> Path:
    """Synchronous version of download_file.

    Args:
        url: URL to download from
        destination: Path to save the file
        timeout: Request timeout in seconds

    Returns:
        Path to the downloaded file

    Raises:
        httpx.HTTPError: If download fails
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading: {url}")
    logger.debug(f"Destination: {destination}")

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

        with open(destination, "wb") as f:
            f.write(response.content)

    logger.info(f"Downloaded: {destination.name} ({destination.stat().st_size} bytes)")
    return destination
