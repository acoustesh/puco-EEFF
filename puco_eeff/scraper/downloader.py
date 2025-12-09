"""Generic file download utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from puco_eeff.config import setup_logging

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logging(__name__)


async def download_file(url: str, destination: Path, request_timeout: float = 60.0) -> Path:
    """Download a file from URL to destination.

    Args:
        url: URL to download from
        destination: Path to save the file
        request_timeout: Request timeout in seconds

    Returns:
        Path to the downloaded file

    Raises:
        httpx.HTTPError: If download fails

    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading: %s", url)
    logger.debug("Destination: %s", destination)

    async with httpx.AsyncClient(timeout=request_timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        with destination.open("wb") as f:
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

    logger.info("Downloading: %s", url)
    logger.debug("Destination: %s", destination)

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

        with destination.open("wb") as f:
            f.write(response.content)

    logger.info(f"Downloaded: {destination.name} ({destination.stat().st_size} bytes)")
    return destination
