# puco-EEFF: Estados Financieros Extraction Pipeline

Pipeline para extraer Estados Financieros (EEFF) de Pucobre desde PDFs públicos y XBRL/XML de la CMF Chile, produciendo libros Excel multi-hoja con trazabilidad completa.

## Features

- **Descarga automática**: PDFs desde pucobre.cl y XBRL desde cmfchile.cl (Consolidado, IFRS)
- **Extracción inteligente**: Prioriza XBRL/XML, usa pdfplumber para tablas, OCR como fallback
- **OCR con fallback**: Mistral OCR → OpenRouter/Claude → OpenRouter/GPT (3 reintentos exponenciales)
- **Auditoría completa**: Guarda todas las respuestas OCR y mapeo de fuentes en `audit/`
- **Validaciones automáticas**: Suma de líneas, cross-validations, comparación PDF vs XBRL
- **Re-ejecución parcial**: Cada hoja se guarda independientemente, permite re-procesar hojas individuales
- **Output estructurado**: JSON por período + `EEFF_YYYY.xlsx` con múltiples trimestres

## Quick Start (Sheet1 end-to-end)

```bash
git clone <repo-url>
cd puco-EEFF
poetry install
chmod +x setup_browser.sh && ./setup_browser.sh  # instala Chromium para Playwright
cp .env.example .env && $EDITOR .env             # añade API keys

# Ejecuta Sheet1: descarga (si falta), extrae PDF/XBRL, valida, guarda JSON
python -m puco_eeff.main_sheet1 --year 2025 --quarter 3

# Reutilizar descargas existentes
python -m puco_eeff.main_sheet1 --year 2025 --quarter 3 --skip-download

# Procesar todos los trimestres de un año
python -m puco_eeff.main_sheet1 --year 2025

# Validar contra valores de referencia conocidos
python -m puco_eeff.main_sheet1 --year 2024 --quarter 2 --validate-reference

# Modo estricto para CI (falla si validaciones no pasan)
python -m puco_eeff.main_sheet1 --year 2024 -q 2 --fail-on-sum-mismatch
```

Entradas/salidas principales:
- Descargas: `data/raw/pdf/`, `data/raw/xbrl/`
- Extraídos por período: `data/processed/sheet1_IIQ2024.json`
- Excel combinado: `data/output/EEFF_2024.xlsx`
- Auditoría OCR y trazabilidad: `audit/2024_Q2/`

## Architecture

### High-Level Data Flow

```
┌─────────────────┐
│  CMF / Pucobre  │  ← Web sources
└────────┬────────┘
         │ (Playwright/httpx)
         ▼
┌─────────────────┐
│ scraper/        │  ← Browser automation, file downloads
│  - browser.py   │     Creates headless Chromium sessions
│  - cmf_downloader.py       Downloads XBRL + PDF from CMF
│  - pucobre_downloader.py   Downloads combined PDF, splits
└────────┬────────┘
         │ (raw PDFs + XBRL → data/raw/)
         ▼
┌─────────────────┐
│ extractor/      │  ← PDF parsing (pdfplumber), XBRL parsing (lxml)
│  - extraction.py           Main extraction orchestration
│  - pdf_parser.py           pdfplumber table/text extraction
│  - table_parser.py         Chilean number format, fuzzy matching
│  - xbrl_parser.py          lxml-based XBRL fact extraction
│  - ocr_mistral.py          Mistral Vision OCR
│  - ocr_fallback.py         Retry chain: Mistral → Claude → GPT
│  - validation_core.py      Sum/cross/reference validations
└────────┬────────┘
         │ (SectionBreakdown, ExtractionResult)
         ▼
┌─────────────────┐
│ sheets/         │  ← Sheet-specific config + validation
│  - sheet1.py    │     27-row Ingresos y Costos structure
│                 │     Config loaders for extraction/XBRL/reference
└────────┬────────┘
         │ (Sheet1Data dataclass)
         ▼
┌─────────────────┐
│ transformer/    │  ← Normalization and provenance tracking
│  - formatter.py            Maps to standard row structure
│  - normalizer.py           Parses Chilean/US number formats
│  - source_tracker.py       Tracks PDF page/XBRL fact sources
└────────┬────────┘
         │ (normalized data + source metadata)
         ▼
┌─────────────────┐
│ writer/         │  ← JSON/CSV/Excel output
│  - sheet_writer.py         Saves JSON (Roman numeral naming)
│  - workbook_combiner.py    Combines quarters → Excel
└────────┬────────┘
         │
         ▼
   data/processed/sheet1_IIQ2024.json
   data/output/EEFF_2024.xlsx
```

### Package Structure

```
puco_eeff/
├── __init__.py             # Package version, public API
├── config.py               # Path resolution, config loaders, API clients
├── main_sheet1.py          # CLI orchestrator for Sheet1 workflow
│
├── scraper/                # Web scraping and file downloads
│   ├── browser.py          # Playwright session management
│   ├── cmf_downloader.py   # CMF XBRL/PDF downloads
│   ├── pucobre_downloader.py   PDF download and splitting
│   └── downloader.py       # httpx-based downloads
│
├── extractor/              # PDF/XBRL parsing and OCR
│   ├── extraction.py       # PDF section extraction (pdfplumber)
│   ├── extraction_pipeline.py   Unified PDF+XBRL extraction
│   ├── pdf_parser.py       # Low-level pdfplumber interface
│   ├── table_parser.py     # Chilean number parsing, fuzzy matching
│   ├── xbrl_parser.py      # lxml-based XBRL fact extraction
│   ├── ocr_mistral.py      # Mistral Vision OCR
│   ├── ocr_fallback.py     # Multi-provider OCR retry chain
│   └── validation_core.py  # Sum/cross/reference validations
│
├── sheets/                 # Sheet-specific logic and config
│   └── sheet1.py           # Sheet1: Ingresos y Costos (27 rows)
│                            Config: extraction.json, fields.json,
│                                     xbrl_mappings.json, reference_data.json
│
├── transformer/            # Normalization and provenance
│   ├── formatter.py        # Maps extracted data to standard rows
│   ├── normalizer.py       # Number/text normalization
│   └── source_tracker.py   # Tracks PDF page/XBRL sources
│
└── writer/                 # Output generation
    ├── sheet_writer.py     # JSON/CSV serialization
    └── workbook_combiner.py    Multi-quarter Excel assembly
```

### Where to Put Sheet-Specific Logic

**New sheets** should follow the Sheet1 pattern:

1. **Config files** in `config/<sheet_name>/`:
   - `extraction.json`: PDF section patterns, search strategy
   - `fields.json`: Field definitions, row mapping
   - `xbrl_mappings.json`: XBRL fact mappings
   - `reference_data.json`: Known-good values for validation

2. **Python module** in `puco_eeff/sheets/<sheet_name>.py`:
   - Dataclass for the sheet structure
   - Config accessor functions (similar to Sheet1)
   - Validation rules (sum checks, cross-validations)

3. **Main orchestrator** following `main_sheet1.py` pattern

## Configuration

### API Keys (`.env`)

| Variable | Servicio | Uso |
|----------|----------|-----|
| `MISTRAL_API_KEY` | Mistral AI | OCR primario |
| `ANTHROPIC_API_KEY` | Anthropic | Fallback vía OpenRouter |
| `OPENROUTER_API_KEY` | OpenRouter | Gateway para fallbacks |
| `OPENAI_API_KEY` | OpenAI | Fallback vía OpenRouter |

### Config (`config/config.json`)

- URLs de fuentes (pucobre.cl, cmfchile.cl)
- Filtros XBRL (Consolidado, IFRS)
- Patrones de secciones PDF (ej: `"note 20.b"`)
- XPath expressions para XML
- Configuración OCR y reintentos

## Operación y debugging

- **Sólo descargar**: ver `instructions/01_download_all.md` (usa `download_all_documents` o CLI de `main_sheet1`).
- **Validar referencia**: `python -m puco_eeff.main_sheet1 --year 2025 --quarter 3 --validate-reference`.
- **OCR fallback**: Mistral → OpenRouter/Claude → OpenRouter/GPT-4o-mini; respuestas quedan en `audit/`.
- **Combinar Excel**: `python - <<'PY'
from puco_eeff.writer.workbook_combiner import combine_sheet1_quarters
combine_sheet1_quarters(year=2025)
PY`

## QA / Tests

```bash
# Run full test suite
poetry run pytest

# Test subset by marker or pattern
poetry run pytest tests/test_environment.py              # Dependencies check
poetry run pytest tests/test_config_integrity.py         # Config validation
poetry run pytest tests/test_comment_density.py          # 6-17% comment density
poetry run pytest tests/test_docstring_format.py         # NumPy docstring style
poetry run pytest tests/test_code_complexity.py          # CC/Cognitive/MI metrics
poetry run pytest tests/test_cost_extractor.py           # Sheet1 extraction

# Code quality checks
poetry run ruff check puco_eeff --select D              # Docstring conventions
poetry run mypy puco_eeff                                # Type checking
poetry run black --check puco_eeff tests                 # Code formatting

# Update baselines (after intentional changes)
poetry run pytest --update-baselines
```

### Test Validation Guarantees

- **test_comment_density.py**: Enforces 6-17% comment density (configurable per file)
- **test_docstring_format.py**: All functions/classes/modules have NumPy-style docstrings
- **test_config_integrity.py**: JSON configs parse correctly, required fields present
- **test_code_complexity.py**: Cyclomatic complexity ≤15, Cognitive complexity ≤15, MI ≥13
- **test_cost_extractor.py**: Sheet1 extraction, validation, and serialization
- **test_formatter.py**: Row mapping, sum validations, reference checks

### Baseline Management

Tests use baselines in `tests/baselines/complexity_baselines.json` to grandfather existing code while preventing regressions:

- **Docstring violations**: Empty baseline means zero violations allowed
- **Comment density**: Aggregate must be 6-26%, per-file baselines for outliers
- **Complexity**: Functions exceeding CC/COG thresholds tracked individually
- **Similarity**: High-similarity function pairs tracked to prevent duplication

Update baselines with `pytest --update-baselines` and commit via PR for review.

## Workflow

1. **Descarga**: PDF + XBRL/XML del período seleccionado
2. **Por cada hoja** (3-5 hojas):
   - Localizar fuentes de datos (XML paths, secciones PDF)
   - Extraer datos (XBRL directo, o OCR con fallback)
   - Formatear como CSV/JSON
3. **Combinar**: Todas las hojas → `EEFF_YYYY_QN.xlsx`

## Requisitos del Sistema

- **OS**: Ubuntu 24.04 (headless compatible)
- **Python**: 3.12+
- **Dependencias del sistema**: Chromium (instalado via `setup_browser.sh`)

## Desarrollo

```bash
# Formatear código
poetry run black puco_eeff tests
poetry run isort puco_eeff tests

# Lint
poetry run ruff check puco_eeff tests

# Type check
poetry run mypy puco_eeff

# Tests
poetry run pytest
```

## Licencia

MIT
