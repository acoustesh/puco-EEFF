"""Mistral OCR integration for PDF and image extraction.

This module provides thin wrappers around the Mistral vision/chat API to turn
PDFs or images into text and markdown-formatted tables. Responses can be
persisted to the audit directory for traceability.
"""

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
    """Extract text and tables using the Mistral OCR model.

    Parameters
    ----------
    image_path
        Path to a single image file to encode and send.
    image_base64
        Raw base64-encoded image string (with or without ``data:`` prefix).
    pdf_path
        Path to a PDF file; if provided, takes precedence over image inputs.
    page_number
        Optional page number hint for the PDF request (may be ignored by the
        API, but useful for logging and downstream consumers).
    prompt
        Custom extraction prompt to override the default “extract all text and
        tables” instruction.
    save_response
        Whether to persist the provider response to the audit directory.
    audit_dir
        Directory used when ``save_response`` is ``True``; defaults to the
        global audit directory.

    Returns
    -------
    dict[str, Any]
        Result payload containing ``success`` flag, ``content`` string,
        provider metadata, and optional token usage or error details.
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
    """Convert a PDF file into the data URL structure expected by Mistral.

    Parameters
    ----------
    pdf_path
        Path to the PDF on disk.
    page_number
        Optional page indicator logged for traceability; the API may ignore it.

    Returns
    -------
    dict[str, Any]
        Payload entry suitable for ``messages[0]["content"]`` with a
        ``type: image_url`` pointing to a PDF data URL.
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
    """Normalize an image input into the Mistral ``image_url`` payload.

    Parameters
    ----------
    image_source
        Path to an image on disk or the raw base64 string to embed.
    is_base64
        When ``True``, treat ``image_source`` as pre-encoded base64 content;
        otherwise read from the filesystem.
    default_mime
        Fallback MIME type used when the extension is unknown.

    Returns
    -------
    dict[str, Any]
        ``{"type": "image_url", "image_url": {"url": ...}}`` payload ready
        for use inside a chat message.
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
    """Persist OCR response for traceability.

    Parameters
    ----------
    result
        Result payload returned by the provider call.
    audit_dir
        Destination directory; created if it does not exist.
    model
        Model name used to derive a safe filename.
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
