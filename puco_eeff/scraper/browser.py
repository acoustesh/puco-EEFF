"""Browser automation utilities using Playwright."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

if TYPE_CHECKING:
    from playwright.sync_api import Playwright


def create_browser(playwright: Playwright, headless: bool = True) -> Browser:
    """Create a Chromium browser instance.

    Args:
        playwright: Playwright instance
        headless: Run browser in headless mode (default: True for server use)

    Returns:
        Browser instance
    """
    return playwright.chromium.launch(
        headless=headless,
        args=[
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-sandbox",
        ],
    )


def create_browser_context(browser: Browser) -> BrowserContext:
    """Create a browser context with appropriate settings.

    Args:
        browser: Browser instance

    Returns:
        BrowserContext configured for scraping
    """
    return browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        accept_downloads=True,
    )


@contextmanager
def browser_session(headless: bool = True) -> Generator[tuple[Browser, Page], None, None]:
    """Context manager for a complete browser session.

    Args:
        headless: Run browser in headless mode

    Yields:
        Tuple of (Browser, Page) for interaction
    """
    with sync_playwright() as playwright:
        browser = create_browser(playwright, headless=headless)
        context = create_browser_context(browser)
        page = context.new_page()

        try:
            yield browser, page
        finally:
            context.close()
            browser.close()


def wait_for_download(page: Page, trigger_action: Any, download_path: str) -> str:
    """Wait for a download to complete and save to specified path.

    Args:
        page: Page instance
        trigger_action: Callable that triggers the download
        download_path: Path to save the downloaded file

    Returns:
        Path to the downloaded file
    """
    with page.expect_download() as download_info:
        trigger_action()

    download = download_info.value
    download.save_as(download_path)

    return download_path
