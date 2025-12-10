"""Source tracking for audit and provenance.

This module tracks the origin of each extracted value for audit purposes.
Every field can be traced back to its source: XBRL fact, PDF page, or OCR result.

Classes
-------
SourceInfo
    Dataclass holding source metadata (file, location, method, confidence).
SourceTracker
    Aggregates SourceInfo entries per field and persists to JSON.

Functions
---------
create_source_mapping
    Convenience function to create a single source mapping dict.

Notes
-----
Source mappings are saved to audit/{period}/source_mapping.json.
This enables full traceability for regulatory compliance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from puco_eeff.config import AUDIT_DIR, setup_logging

if TYPE_CHECKING:
    from pathlib import Path

# Module logger for source tracking operations
logger = setup_logging(__name__)


@dataclass
class SourceInfo:
    """Information about a data source.

    Attributes
    ----------
    source_type : str
        Type of source: "xml", "pdf_text", or "pdf_ocr".
    file_path : str
        Absolute path to the source file.
    location : str
        Location within file: XPath for XML, "page N, section X" for PDF.
    extraction_method : str
        Method used: "direct", "pdfplumber", "mistral_ocr", "openrouter_ocr".
    confidence : float
        Confidence score from 0.0 to 1.0. Direct extraction = 1.0.
    timestamp : str
        ISO 8601 timestamp when source was recorded.
    raw_value : str or None
        Original extracted string before parsing/normalization.
    """

    source_type: str  # "xml", "pdf_text", "pdf_ocr"
    file_path: str
    location: str  # XPath for XML, page/section for PDF
    extraction_method: str  # "direct", "pdfplumber", "mistral_ocr", etc.
    confidence: float = 1.0  # 1.0 for direct extraction, lower for OCR
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    raw_value: str | None = None  # Original value before normalization


@dataclass
class SourceTracker:
    """Track data sources for audit trail.

    Maintains a mapping from field names to their source information.
    Supports multiple sources per field (e.g., PDF and XBRL for validation).

    Attributes
    ----------
    period : str
        Period identifier, e.g., "2024_Q3".
    mappings : dict[str, list[SourceInfo]]
        Field name to list of sources. Multiple sources allow cross-validation.
    """

    period: str  # e.g., "2024_Q3"
    mappings: dict[str, list[SourceInfo]] = field(default_factory=dict)

    def add_source(
        self,
        field_name: str,
        source_type: str,
        file_path: str,
        location: str,
        extraction_method: str,
        confidence: float = 1.0,
        raw_value: str | None = None,
    ) -> None:
        """Add a source mapping for a field.

        Parameters
        ----------
        field_name : str
            Name of the data field (e.g., "total_costo_venta").
        source_type : str
            Type of source: "xml", "pdf_text", or "pdf_ocr".
        file_path : str
            Path to the source file.
        location : str
            Location within file (XPath, page number, section).
        extraction_method : str
            Method used for extraction.
        confidence : float, optional
            Confidence score. Default 1.0.
        raw_value : str or None, optional
            Raw extracted value before normalization.
        """
        # Create SourceInfo with provided metadata
        source = SourceInfo(
            source_type=source_type,
            file_path=file_path,
            location=location,
            extraction_method=extraction_method,
            confidence=confidence,
            raw_value=raw_value,
        )

        # Initialize list if first source for this field
        if field_name not in self.mappings:
            self.mappings[field_name] = []

        self.mappings[field_name].append(source)
        logger.debug("Added source for '%s': %s @ %s", field_name, source_type, location)

    def get_primary_source(self, field_name: str) -> SourceInfo | None:
        """Get the primary (highest confidence) source for a field.

        Parameters
        ----------
        field_name : str
            Name of the data field.

        Returns
        -------
        SourceInfo or None
            Primary source if found, None if no sources exist.

        Notes
        -----
        Prioritizes by confidence, then by source type (XML > PDF > OCR).
        """
        sources = self.mappings.get(field_name, [])
        if not sources:
            return None

        # Prefer highest confidence, then XML over other types
        return max(sources, key=lambda s: (s.confidence, s.source_type == "xml"))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "period": self.period,
            "generated_at": datetime.now(UTC).isoformat(),
            "mappings": {
                field: [asdict(s) for s in sources] for field, sources in self.mappings.items()
            },
        }

    def save(self, output_dir: Path | None = None) -> Path:
        """Save source mappings to audit directory.

        Parameters
        ----------
        output_dir : Path or None, optional
            Directory to save to. Defaults to AUDIT_DIR/{period}.

        Returns
        -------
        Path
            Path to saved JSON file.
        """
        # Default to audit directory organized by period
        save_dir = output_dir if output_dir is not None else AUDIT_DIR / self.period
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / "source_mapping.json"

        # Write with UTF-8 encoding for Spanish characters
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("Saved source mapping to: %s", filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> SourceTracker:
        """Load source tracker from file.

        Parameters
        ----------
        filepath : Path
            Path to source_mapping.json file.

        Returns
        -------
        SourceTracker
            Reconstructed tracker instance.
        """
        with filepath.open(encoding="utf-8") as f:
            data = json.load(f)

        tracker = cls(period=data["period"])

        # Reconstruct all source mappings from JSON
        for field_name, sources in data.get("mappings", {}).items():
            for source_data in sources:
                tracker.add_source(
                    field_name=field_name,
                    source_type=source_data["source_type"],
                    file_path=source_data["file_path"],
                    location=source_data["location"],
                    extraction_method=source_data["extraction_method"],
                    confidence=source_data.get("confidence", 1.0),
                    raw_value=source_data.get("raw_value"),
                )

        return tracker


def create_source_mapping(
    period: str,
    field_name: str,
    source_type: str,
    file_path: str,
    location: str,
    extraction_method: str,
    confidence: float = 1.0,
    raw_value: str | None = None,
) -> dict[str, Any]:
    """Create a single source mapping entry.

    Convenience function for creating source mapping dictionaries
    without instantiating a full SourceTracker.

    Parameters
    ----------
    period : str
        Period identifier (e.g., "2024_Q3").
    field_name : str
        Name of the data field.
    source_type : str
        Type of source: "xml", "pdf_text", or "pdf_ocr".
    file_path : str
        Path to source file.
    location : str
        Location within file.
    extraction_method : str
        Method used for extraction.
    confidence : float, optional
        Confidence score. Default 1.0.
    raw_value : str or None, optional
        Raw extracted value.

    Returns
    -------
    dict[str, Any]
        Source mapping dictionary suitable for JSON serialization.
    """
    return {
        "period": period,
        "field_name": field_name,
        "source": {
            "type": source_type,
            "file": file_path,
            "location": location,
            "method": extraction_method,
            "confidence": confidence,
            "raw_value": raw_value,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    }
