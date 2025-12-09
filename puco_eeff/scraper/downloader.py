"""Generic file download utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from puco_eeff.config import setup_logging

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logging(__name__)


async def download_file(url: str, destination: Path, request_timeout: float = 60.0) -> Path:
    """Download a file from URL to destination asynchronously.

    Uses httpx.AsyncClient for non-blocking I/O. Preferred when running
    in an async context (e.g., with asyncio.run or inside async functions).

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

    async with httpx.AsyncClient(timeout=request_timeout, follow_redirects=True) as async_http:
        http_response = await async_http.get(url)
        http_response.raise_for_status()

        with destination.open("wb") as output_file:
            output_file.write(http_response.content)

    file_stats = destination.stat()
    logger.info("Downloaded: %s (%d bytes)", destination.name, file_stats.st_size)
    return destination


def download_file_sync(url: str, destination: Path, timeout: float = 60.0) -> Path:
    """Synchronous file download using blocking I/O.

    Uses httpx.Client for blocking requests. Use this when running in
    synchronous code without an event loop, such as in CLI scripts.

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

    with httpx.Client(timeout=timeout, follow_redirects=True) as sync_client:
        resp = sync_client.get(url)
        resp.raise_for_status()
        destination.write_bytes(resp.content)

    downloaded_size = destination.stat().st_size
    logger.info(f"Downloaded: {destination.name} ({downloaded_size} bytes)")
    return destination
