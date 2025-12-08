"""Mistral OCR integration for PDF image extraction."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from puco_eeff.config import AUDIT_DIR, get_mistral_client, setup_logging

logger = setup_logging(__name__)

OCR_MODEL = "mistral-ocr-latest"


def ocr_with_mistral(
    image_path: Path | None = None,
    image_base64: str | None = None,
    pdf_path: Path | None = None,
    page_number: int | None = None,
    prompt: str | None = None,
    save_response: bool = True,
    audit_dir: Path | None = None,
) -> dict[str, Any]:
    """Extract text and tables from an image using Mistral OCR.

    Args:
        image_path: Path to an image file
        image_base64: Base64-encoded image data
        pdf_path: Path to a PDF file (Mistral can process PDFs directly)
        page_number: Page number if processing a specific page
        prompt: Custom prompt for OCR extraction
        save_response: Whether to save the response for audit
        audit_dir: Directory to save audit files

    Returns:
        Dictionary with extracted content and metadata
    """
    if not any([image_path, image_base64, pdf_path]):
        raise ValueError("Must provide image_path, image_base64, or pdf_path")

    client = get_mistral_client()

    # Prepare the content based on input type
    if pdf_path:
        content = _prepare_pdf_content(pdf_path, page_number)
        source_desc = f"PDF: {pdf_path.name}" + (f" (page {page_number})" if page_number else "")
    elif image_path:
        content = _prepare_image_content(image_path)
        source_desc = f"Image: {image_path.name}"
    else:
        content = _prepare_base64_content(image_base64)  # type: ignore[arg-type]
        source_desc = "Base64 image"

    # Default prompt for financial statement extraction
    default_prompt = """Extract all text and tables from this image/document.
For tables, preserve the structure with rows and columns clearly separated.
Include all numbers, dates, and text exactly as shown.
Format tables as markdown tables when possible."""

    messages = [
        {
            "role": "user",
            "content": [
                content,
                {"type": "text", "text": prompt or default_prompt},
            ],
        }
    ]

    logger.info(f"Calling Mistral OCR for: {source_desc}")
    logger.debug(f"Model: {OCR_MODEL}")

    try:
        response = client.chat.complete(
            model=OCR_MODEL,
            messages=messages,
        )

        result = {
            "success": True,
            "provider": "mistral",
            "model": OCR_MODEL,
            "source": source_desc,
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }

        logger.info("Mistral OCR extraction successful")
        logger.debug(
            f"Tokens used: {response.usage.prompt_tokens} + {response.usage.completion_tokens}"
        )

    except Exception as e:
        logger.error(f"Mistral OCR failed: {e}")
        result = {
            "success": False,
            "provider": "mistral",
            "model": OCR_MODEL,
            "source": source_desc,
            "error": str(e),
        }

    # Save response for audit
    if save_response:
        _save_audit_response(result, audit_dir or AUDIT_DIR)

    return result


def _prepare_pdf_content(pdf_path: Path, page_number: int | None) -> dict[str, Any]:
    """Prepare PDF content for Mistral API.

    Args:
        pdf_path: Path to PDF file
        page_number: Optional specific page

    Returns:
        Content dictionary for API
    """
    with open(pdf_path, "rb") as f:
        pdf_base64 = base64.standard_b64encode(f.read()).decode("utf-8")

    content: dict[str, Any] = {
        "type": "image_url",
        "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"},
    }

    # Note: page selection may need to be handled differently depending on API support
    if page_number:
        logger.debug(f"Requesting page {page_number} (API support may vary)")

    return content


def _prepare_image_content(image_path: Path) -> dict[str, Any]:
    """Prepare image content for Mistral API.

    Args:
        image_path: Path to image file

    Returns:
        Content dictionary for API
    """
    # Determine MIME type
    suffix = image_path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(suffix, "image/png")

    with open(image_path, "rb") as f:
        image_base64 = base64.standard_b64encode(f.read()).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
    }


def _prepare_base64_content(image_base64: str) -> dict[str, Any]:
    """Prepare base64 image content for Mistral API.

    Args:
        image_base64: Base64-encoded image

    Returns:
        Content dictionary for API
    """
    # Assume PNG if no prefix provided
    if not image_base64.startswith("data:"):
        image_base64 = f"data:image/png;base64,{image_base64}"

    return {
        "type": "image_url",
        "image_url": {"url": image_base64},
    }


def _save_audit_response(result: dict[str, Any], audit_dir: Path, model: str = "mistral") -> None:
    """Save OCR response to audit directory.

    Args:
        result: OCR result dictionary
        audit_dir: Directory to save audit files
        model: Model name for filename (default: "mistral")
    """
    from datetime import datetime

    audit_dir.mkdir(parents=True, exist_ok=True)

    # Create a safe filename from model name
    model_safe = model.replace("/", "_").replace(".", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ocr_{model_safe}_{timestamp}.json"
    filepath = audit_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.debug(f"Audit response saved: {filepath}")
