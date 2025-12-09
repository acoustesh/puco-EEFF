"""OCR fallback using OpenRouter (Anthropic Claude and OpenAI GPT-4V)."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from puco_eeff.config import AUDIT_DIR, get_config, get_openrouter_client, setup_logging
from puco_eeff.extractor.ocr_mistral import _save_audit_response, ocr_with_mistral

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logging(__name__)


@dataclass
class OCRRequest:
    """Encapsulates OCR request parameters."""

    image_path: Path | None = None
    image_base64: str | None = None
    pdf_path: Path | None = None
    page_number: int | None = None
    prompt: str | None = None
    audit_dir: Path | None = None


def _try_mistral_ocr(request: OCRRequest, save_response: bool) -> dict[str, Any]:
    """Attempt OCR using Mistral."""
    return ocr_with_mistral(
        image_path=request.image_path,
        image_base64=request.image_base64,
        pdf_path=request.pdf_path,
        page_number=request.page_number,
        prompt=request.prompt,
        save_response=save_response,
        audit_dir=request.audit_dir,
    )


def _attempt_single_ocr(
    provider: str,
    model: str,
    request: OCRRequest,
    save_response: bool,
) -> dict[str, Any]:
    """Execute a single OCR attempt for a specific provider."""
    if provider == "mistral":
        return _try_mistral_ocr(request, save_response)
    # OpenRouter provider - call _ocr_with_openrouter directly
    return _ocr_with_openrouter(
        model=model,
        image_path=request.image_path,
        image_base64=request.image_base64,
        prompt=request.prompt,
        audit_dir=request.audit_dir if save_response else None,
    )


def _run_with_retries(
    provider: str,
    model: str,
    request: OCRRequest,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    save_responses: bool,
    all_responses: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Run OCR attempts with exponential backoff retry logic."""
    for attempt in range(max_attempts):
        logger.info(f"OCR attempt {attempt + 1}/{max_attempts} with {provider}/{model}")

        try:
            result = _attempt_single_ocr(provider, model, request, save_responses)

            if result.get("success"):
                all_responses.append(result)
                result["all_responses"] = all_responses if save_responses else []
                return result

            logger.warning(f"OCR failed: {result.get('error', 'Unknown error')}")
            all_responses.append(result)

        except Exception as e:
            logger.exception("OCR exception: %s", e)
            all_responses.append({
                "success": False,
                "provider": provider,
                "model": model,
                "error": str(e),
                "attempt": attempt + 1,
            })

        if attempt < max_attempts - 1:
            delay = min(base_delay * (2**attempt), max_delay)
            logger.debug("Waiting %ss before retry...", delay)
            time.sleep(delay)

    logger.warning("All %s attempts failed for %s/%s", max_attempts, provider, model)
    return None


def ocr_with_fallback(
    image_path: Path | None = None,
    image_base64: str | None = None,
    pdf_path: Path | None = None,
    page_number: int | None = None,
    prompt: str | None = None,
    save_all_responses: bool = True,
    audit_dir: Path | None = None,
) -> dict[str, Any]:
    """Extract text using OCR with fallback chain.

    Tries Mistral OCR first, then falls back to OpenRouter providers
    (Anthropic Claude, OpenAI GPT-4V) with exponential retry.
    """
    config = get_config()
    ocr_config = config["ocr"]
    retry_config = ocr_config["retry"]

    request = OCRRequest(
        image_path=image_path,
        image_base64=image_base64,
        pdf_path=pdf_path,
        page_number=page_number,
        prompt=prompt,
        audit_dir=audit_dir or AUDIT_DIR,
    )

    all_responses: list[dict[str, Any]] = []
    providers = [{"provider": "mistral", "model": "mistral-ocr-latest"}, *ocr_config["fallback"]]

    for provider_info in providers:
        result = _run_with_retries(
            provider=provider_info["provider"],
            model=provider_info["model"],
            request=request,
            max_attempts=retry_config["max_attempts"],
            base_delay=retry_config["base_delay_seconds"],
            max_delay=retry_config["max_delay_seconds"],
            save_responses=save_all_responses,
            all_responses=all_responses,
        )
        if result:
            return result

    logger.error("All OCR providers failed")
    return {
        "success": False,
        "error": "All OCR providers failed after maximum retries",
        "all_responses": all_responses,
    }


def _ocr_with_openrouter(
    model: str,
    image_path: Path | None = None,
    image_base64: str | None = None,
    prompt: str | None = None,
    audit_dir: Path | None = None,
) -> dict[str, Any]:
    """Perform OCR using OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "anthropic/claude-3.5-sonnet")
        image_path: Path to an image file
        image_base64: Base64-encoded image data
        prompt: Custom prompt for OCR extraction
        audit_dir: Directory to save audit files (None to skip)

    Returns
    -------
        Dictionary with extracted content and metadata

    """
    client = get_openrouter_client()

    # Prepare image content
    if image_path:
        image_data = _encode_image(image_path)
        source_desc = f"Image: {image_path.name}"
    elif image_base64:
        image_data = (
            image_base64
            if image_base64.startswith("data:")
            else f"data:image/png;base64,{image_base64}"
        )
        source_desc = "Base64 image"
    else:
        msg = "Must provide image_path or image_base64"
        raise ValueError(msg)

    default_prompt = """Extract all text and tables from this image.
For tables, preserve the structure with rows and columns clearly separated.
Include all numbers, dates, and text exactly as shown.
Format tables as markdown tables when possible."""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data},
                },
                {
                    "type": "text",
                    "text": prompt or default_prompt,
                },
            ],
        },
    ]

    logger.info("Calling OpenRouter OCR: %s", model)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
        )

        result = {
            "success": True,
            "provider": "openrouter",
            "model": model,
            "source": source_desc,
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }

        logger.info("OpenRouter OCR successful with %s", model)

    except Exception as e:
        logger.exception("OpenRouter OCR failed: %s", e)
        result = {
            "success": False,
            "provider": "openrouter",
            "model": model,
            "source": source_desc,
            "error": str(e),
        }

    # Save response for audit
    if audit_dir:
        _save_audit_response(result, audit_dir, model)

    return result


def _encode_image(image_path: Path) -> str:
    """Encode an image file to base64 data URL.

    Args:
        image_path: Path to image file

    Returns
    -------
        Base64 data URL string

    """
    suffix = image_path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(suffix, "image/png")

    with image_path.open("rb") as f:
        image_base64 = base64.standard_b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{image_base64}"
