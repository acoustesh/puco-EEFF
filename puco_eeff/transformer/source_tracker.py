"""Source tracking for audit and provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from puco_eeff.config import AUDIT_DIR, setup_logging

logger = setup_logging(__name__)


@dataclass
class SourceInfo:
    """Information about a data source."""

    source_type: str  # "xml", "pdf_text", "pdf_ocr"
    file_path: str
    location: str  # XPath for XML, page/section for PDF
    extraction_method: str  # "direct", "pdfplumber", "mistral_ocr", etc.
    confidence: float = 1.0  # Confidence score (1.0 for direct extraction)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    raw_value: str | None = None


@dataclass
class SourceTracker:
    """Track data sources for audit trail."""

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

        Args:
            field_name: Name of the data field
            source_type: Type of source (xml, pdf_text, pdf_ocr)
            file_path: Path to source file
            location: Location within file (XPath, page number, section)
            extraction_method: Method used for extraction
            confidence: Confidence score
            raw_value: Raw extracted value before normalization
        """
        source = SourceInfo(
            source_type=source_type,
            file_path=file_path,
            location=location,
            extraction_method=extraction_method,
            confidence=confidence,
            raw_value=raw_value,
        )

        if field_name not in self.mappings:
            self.mappings[field_name] = []

        self.mappings[field_name].append(source)
        logger.debug(f"Added source for '{field_name}': {source_type} @ {location}")

    def get_primary_source(self, field_name: str) -> SourceInfo | None:
        """Get the primary (highest confidence) source for a field.

        Args:
            field_name: Name of the data field

        Returns:
            Primary SourceInfo or None if not found
        """
        sources = self.mappings.get(field_name, [])
        if not sources:
            return None

        # Return highest confidence source (XML > PDF text > OCR)
        return max(sources, key=lambda s: (s.confidence, s.source_type == "xml"))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "period": self.period,
            "generated_at": datetime.now().isoformat(),
            "mappings": {
                field: [
                    {
                        "source_type": s.source_type,
                        "file_path": s.file_path,
                        "location": s.location,
                        "extraction_method": s.extraction_method,
                        "confidence": s.confidence,
                        "timestamp": s.timestamp,
                        "raw_value": s.raw_value,
                    }
                    for s in sources
                ]
                for field, sources in self.mappings.items()
            },
        }

    def save(self, output_dir: Path | None = None) -> Path:
        """Save source mappings to audit directory.

        Args:
            output_dir: Directory to save to (defaults to AUDIT_DIR/period)

        Returns:
            Path to saved file
        """
        save_dir = output_dir if output_dir is not None else AUDIT_DIR / self.period
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / "source_mapping.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved source mapping to: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> SourceTracker:
        """Load source tracker from file.

        Args:
            filepath: Path to source mapping JSON file

        Returns:
            SourceTracker instance
        """
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        tracker = cls(period=data["period"])

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

    Convenience function for creating source mapping dictionaries.

    Args:
        period: Period identifier (e.g., "2024_Q3")
        field_name: Name of the data field
        source_type: Type of source (xml, pdf_text, pdf_ocr)
        file_path: Path to source file
        location: Location within file
        extraction_method: Method used for extraction
        confidence: Confidence score
        raw_value: Raw extracted value

    Returns:
        Source mapping dictionary
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
            "timestamp": datetime.now().isoformat(),
        },
    }
