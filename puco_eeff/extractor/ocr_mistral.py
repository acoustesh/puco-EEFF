"""Mistral OCR integration for PDF image extraction."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from puco_eeff.config import AUDIT_DIR, get_mistral_client, setup_logging

if TYPE_CHECKING:
    from pathlib import Path

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

    Returns
    -------
        Dictionary with extracted content and metadata

    """
    if not any([image_path, image_base64, pdf_path]):
        msg = "Must provide image_path, image_base64, or pdf_path"
        raise ValueError(msg)

    client = get_mistral_client()

    # Prepare the content based on input type
    if pdf_path:
        content = _prepare_pdf_content(pdf_path, page_number)
        source_desc = f"PDF: {pdf_path.name}" + (f" (page {page_number})" if page_number else "")
    elif image_path:
        content = _prepare_image_content(image_path)
        source_desc = f"Image: {image_path.name}"
    else:
        content = _prepare_image_content(image_base64, is_base64=True)  # type: ignore[arg-type]
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
        },
    ]

    logger.info("Calling Mistral OCR for: %s", source_desc)
    logger.debug("Model: %s", OCR_MODEL)

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
            f"Tokens used: {response.usage.prompt_tokens} + {response.usage.completion_tokens}",
        )

    except Exception as e:
        logger.exception("Mistral OCR failed: %s", e)
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

    Returns
    -------
        Content dictionary for API

    """
    with pdf_path.open("rb") as f:
        pdf_base64 = base64.standard_b64encode(f.read()).decode("utf-8")

    content: dict[str, Any] = {
        "type": "image_url",
        "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"},
    }

    # Note: page selection may need to be handled differently depending on API support
    if page_number:
        logger.debug("Requesting page %s (API support may vary)", page_number)

    return content


# MIME type mapping for common image formats
_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _prepare_image_content(
    image_source: Path | str,
    *,
    is_base64: bool = False,
    default_mime: str = "image/png",
) -> dict[str, Any]:
    """Prepare image content for Mistral API.

    Args:
        image_source: Either a Path to an image file, or a base64-encoded string
        is_base64: If True, treat image_source as base64 string; if False, as file path
        default_mime: Default MIME type when not determinable (default: image/png)

    Returns
    -------
        Content dictionary for API with {"type": "image_url", "image_url": {"url": ...}}

    """
    if is_base64:
        # Handle base64 string input
        base64_str = str(image_source)
        if base64_str.startswith("data:"):
            data_url = base64_str
        else:
            data_url = f"data:{default_mime};base64,{base64_str}"
    else:
        # Handle file path input
        image_path = Path(image_source)
        mime_type = _MIME_TYPES.get(image_path.suffix.lower(), default_mime)
        with image_path.open("rb") as f:
            encoded = base64.standard_b64encode(f.read()).decode("utf-8")
        data_url = f"data:{mime_type};base64,{encoded}"

    return {"type": "image_url", "image_url": {"url": data_url}}


def _save_audit_response(result: dict[str, Any], audit_dir: Path, model: str = "mistral") -> None:
    """Save OCR response to audit directory.

    Args:
        result: OCR result dictionary
        audit_dir: Directory to save audit files
        model: Model name for filename (default: "mistral")

    """
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Create a safe filename from model name
    model_safe = model.replace("/", "_").replace(".", "_")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"ocr_{model_safe}_{timestamp}.json"
    filepath = audit_dir / filename

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.debug("Audit response saved: %s", filepath)
