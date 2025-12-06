#!/bin/bash
# Setup script for headless Chromium browser on Ubuntu 24.04
# This script installs Playwright's Chromium browser for headless operation

set -e

echo "=== puco-EEFF Browser Setup ==="
echo "Target: Ubuntu 24.04 (headless compatible)"
echo ""

# Check if running on Linux
if [[ "$(uname)" != "Linux" ]]; then
    echo "Warning: This script is designed for Linux (Ubuntu 24.04)"
    echo "On other systems, run: poetry run playwright install chromium"
    exit 1
fi

# Install system dependencies for Playwright/Chromium
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2

# Install Playwright browsers
echo ""
echo "Installing Playwright Chromium browser..."
poetry run playwright install chromium

# Install Playwright system dependencies (for headless)
echo ""
echo "Installing Playwright system dependencies..."
poetry run playwright install-deps chromium

echo ""
echo "=== Setup Complete ==="
echo "Chromium is ready for headless operation."
echo ""
echo "To verify installation, run:"
echo "  poetry run python -c \"from playwright.sync_api import sync_playwright; print('OK')\""
