# Data Mapping - Pucobre Financial Statements

## Overview

This document maps the data structure between what was requested (Q1 2024 example) and what was found in actual CMF documents (Q3 2024 verified).

## Key Findings

### Data Source Summary

| Data Type | Source | Location |
|-----------|--------|----------|
| Ingresos de actividades ordinarias | XBRL | `Revenue` fact |
| Costo de Venta (total) | XBRL | `CostOfSales` fact |
| Costo de Venta (desglose) | PDF | Nota 21, Page 71 |
| Gastos Admin y Ventas (total) | XBRL | `AdministrativeExpense` fact |
| Gastos Admin y Ventas (desglose) | PDF | Nota 22, Page 71 |
| Ganancia bruta | XBRL | `GrossProfit` fact |
| Ganancia del periodo | XBRL | `ProfitLoss` fact |

### Q1 2024 User Example vs Q3 2024 Actual Structure

#### Costo de Venta - Line Items Comparison

| User Example (Q1 2024) | Actual PDF (Q3 2024) | Status |
|------------------------|----------------------|--------|
| Gastos en personal | Gastos en personal | ✓ Match |
| Materiales y repuestos | Materiales y repuestos | ✓ Match |
| Energía eléctrica | Energía eléctrica | ✓ Match |
| Servicios de terceros | Servicios de terceros | ✓ Match |
| Depreciación y amort. | Depreciación y amort. del periodo | ✓ Match |
| Depreciación Activos en leasing | Depreciación Activos en leasing (Nota 20) | ✓ Match |
| Depreciación Arrendamientos | Depreciación Arrendamientos (Nota 20) | ✓ Match |
| Servicios mineros de terceros | Servicios mineros de terceros | ✓ Match |
| Fletes y otros | Fletes y otros gastos operacionales | ✓ Match |
| Gastos Diferidos | Gastos Diferidos, ajustes existencias y otros | ✓ Match |
| Obligaciones por convenios colectivos | Obligaciones por convenios colectivos | ✓ Match |

**Note**: Q3 2024 has the same 11 line items as Q1 2024 example.

#### Gastos Admin y Ventas - Line Items Comparison

| User Example (Q1 2024) | Actual PDF (Q3 2024) | Status |
|------------------------|----------------------|--------|
| Gastos en personal | Gastos en personal | ✓ Match |
| Materiales y repuestos | Materiales y repuestos | ✓ Match |
| Servicios de terceros | Servicios de terceros | ✓ Match |
| Provisión gratificación legal | Provisión gratificación legal y otros | ✓ Match |
| Gastos comercialización | Gastos comercialización | ✓ Match |
| Otros gastos | Otros gastos | ✓ Match |

**Note**: Q3 2024 has the same 6 line items as Q1 2024 example.

## Sample Values Comparison

### Q1 2024 (User Example) - Values in MUSD

```
Ingresos de actividades ordinarias:  80,767

COSTO DE VENTA:
  Gastos en personal              -9,857
  Materiales y repuestos         -10,986
  Energía eléctrica               -4,727
  Servicios de terceros          -11,723
  Depreciación y amort.          -10,834
  Depreciación Activos en leasing   -485
  Depreciación Arrendamientos       -711
  Servicios mineros de terceros   -3,675
  Fletes y otros                  -2,489
  Gastos Diferidos                  -875
  Obligaciones convenios colect.  -6,620
  TOTAL COSTO VENTA:             -62,982

GASTOS ADM Y VENTAS:
  Gastos en personal              -1,831
  Materiales y repuestos             -56
  Servicios de terceros           -1,250
  Provisión gratificación legal     -966
  Gastos comercialización           -805
  Otros gastos                      -229
  TOTAL GASTOS ADM:               -5,137
```

### Q3 2024 (Actual Extracted) - Values in MUSD (YTD 9 months)

```
Ingresos de actividades ordinarias: 231,472

COSTO DE VENTA (Nota 21):
  Gastos en personal             -30,294
  Materiales y repuestos         -37,269
  Energía eléctrica              -14,710
  Servicios de terceros          -39,213
  Depreciación y amort.          -33,178
  Depreciación Activos en leasing -1,354
  Depreciación Arrendamientos     -2,168
  Servicios mineros de terceros  -19,577
  Fletes y otros gastos          -7,405
  Gastos Diferidos               +20,968  (positive adjustment)
  Obligaciones convenios colect.  -6,662
  TOTAL COSTO VENTA:            -170,862

GASTOS ADM Y VENTAS (Nota 22):
  Gastos en personal              -5,727
  Materiales y repuestos            -226
  Servicios de terceros           -4,474
  Provisión gratificación legal   -2,766
  Gastos comercialización         -3,506
  Otros gastos                      -664
  TOTAL GASTOS ADM:              -17,363
```

## Number Formatting

### Chilean Financial Documents

| Format | Meaning | Example |
|--------|---------|---------|
| `1.234` | 1,234 (thousands separator) | 30.294 = 30,294 |
| `1,5` | 1.5 (decimal separator) | 1,5 = 1.5 |
| `(30.294)` | Negative value | -30,294 |
| `MUS$` | Miles de USD | Thousands of dollars |
| `MUSD` | Miles de USD | Same as MUS$ |

### Parsing Rules

1. **Parentheses = Negative**: `(30.294)` → `-30294`
2. **Period = Thousands**: `30.294` → `30294`
3. **Remove currency symbols**: `MUS$`, `US$`, `$`
4. **Scale**: Values are already in thousands (MUSD/MUS$)

## Column Mapping

### PDF Table Columns (Nota 21 & 22)

| Column Index | Header | Description |
|--------------|--------|-------------|
| 0 | Concepto | Line item description |
| 1 | 01-01-YYYY 30-09-YYYY | YTD Current Year |
| 2 | 01-01-YYYY 30-09-YYYY | YTD Prior Year |
| 3 | 01-07-YYYY 30-09-YYYY | Q3 Current Year |
| 4 | 01-07-YYYY 30-09-YYYY | Q3 Prior Year |

**For extraction**: Use Column 1 (YTD current year) as the primary value.

## XBRL Fact Name Mapping

> **Note**: XBRL fact names and scaling are now configured in `config/xbrl_specs.json`.
> The `fact_mappings` section defines primary and fallback fact names for each field.

| Internal Field | Primary XBRL Fact | Fallbacks | Apply Scaling |
|----------------|-------------------|-----------|---------------|
| ingresos_ordinarios | RevenueFromContractsWithCustomers | Revenue, IngresosPorActividadesOrdinarias | true |
| total_costo_venta | CostOfSales | CostoDeVentas | true |
| total_gasto_admin | AdministrativeExpense | GastosDeAdministracion | true |
| gross_profit | GrossProfit | UtilidadBruta | true |
| profit_loss | ProfitLoss | GananciaPerdida | true |

**Scaling Factor**: 1000 (XBRL values in USD → divide by 1000 to get MUS$)

| Spanish Name | XBRL Fact | Context |
|--------------|-----------|---------|
| Ingresos de actividades ordinarias | `RevenueFromContractsWithCustomers` | Duration (YTD) |
| Costo de ventas | `CostOfSales` | Duration (YTD) |
| Ganancia bruta | `GrossProfit` | Duration (YTD) |
| Gastos de administración y ventas | `AdministrativeExpense` | Duration (YTD) |
| Ganancia (pérdida) | `ProfitLoss` | Duration (YTD) |
| Total activos | `Assets` | Instant (end of period) |
| Total pasivos | `Liabilities` | Instant (end of period) |
| Patrimonio total | `Equity` | Instant (end of period) |

## Validation Rules

### Unified Validation API

All validation is handled by `run_sheet1_validations()`, a single entry point that runs three types of config-driven validations:

```python
from puco_eeff.extractor.cost_extractor import run_sheet1_validations

report = run_sheet1_validations(
    data,                          # Sheet1Data with extracted values
    xbrl_totals,                   # XBRL totals dict (or None)
    run_sum_validations=True,      # Enable sum checks (default: True)
    run_pdf_xbrl_validations=True, # Enable PDF↔XBRL comparison (default: True)
    run_cross_validations=True,    # Enable cross-validation formulas (default: True)
    use_xbrl_fallback=True,        # Set missing PDF values from XBRL (default: True)
)

# Check for failures
if report.has_failures():
    print(f"Failed: {len([v for v in report.sum_validations if not v.match])} sum validations")
```

#### 1. PDF ↔ XBRL Total Comparison
- `CostOfSales` (XBRL) = `total_costo_venta` (PDF)
- `AdministrativeExpense` (XBRL) = `total_gasto_admin` (PDF)
- `RevenueFromContractsWithCustomers` (XBRL) = `ingresos_ordinarios` (PDF fallback)

Configured via `pdf_xbrl_validations` in `config/sheet1/xbrl_mappings.json`.

#### 2. Sum Validations (Config-Driven)
Validates that extracted totals match the sum of their line items.
Configured in `config/sheet1/xbrl_mappings.json`:

```json
{
  "validation_rules": {
    "sum_tolerance": 1,
    "total_validations": [
      {
        "total_field": "total_costo_venta",
        "sum_fields": ["cv_gastos_personal", "cv_materiales", "cv_energia", "..."],
        "description": "Nota 21 - Costo de Venta"
      },
      {
        "total_field": "total_gasto_admin",
        "sum_fields": ["ga_gastos_personal", "ga_materiales", "..."],
        "description": "Nota 22 - Gastos de Administración y Ventas"
      }
    ]
  }
}
```

#### 3. Cross-Validations (Accounting Identities)
Validates relationships between fields. Example:

```json
{
  "cross_validations": [
    {
      "description": "Gross Profit = Revenue - Cost of Sales",
      "formula": "gross_profit == ingresos_ordinarios - abs(total_costo_venta)",
      "tolerance": 1
    }
  ]
}
```

**Formula evaluation:** Safe AST-based parsing (no `eval()`), supports: variables, integers, `+`, `-`, `abs()`.

### Deprecated Functions

> **⚠️ Deprecation:** `validate_extraction()` is deprecated. Use `run_sheet1_validations()` instead.

### Opt-in Reference Validation

Compare extracted values against known-good reference data. Enabled via `--validate-reference` flag:

```bash
python -m puco_eeff.main_sheet1 -y 2024 -q 2 --validate-reference
```

Reference data stored in `config/sheet1/reference_data.json`.

### Validation Tolerance

All validations use tolerance-based comparison (default: 1 MUSD):
- **Global tolerance:** `validation_rules.sum_tolerance`
- **Per-rule tolerance:** Individual cross-validation rules can override

### Validation Report

Running extraction produces a validation report:

```
═══════════════════════════════════════════════════════════════
                    VALIDATION REPORT
═══════════════════════════════════════════════════════════════

Sum Validations:
  ✓ Nota 21 - Costo de Venta: Sum matches total (-126,202)
  ✓ Nota 22 - Gastos de Administración y Ventas: Sum matches total (-11,632)

PDF ↔ XBRL Validations:
  ✓ Total Costo de Venta: -126,202 (PDF) = -126,202 (XBRL)
  ✓ Total Gasto Admin: -11,632 (PDF) = -11,632 (XBRL)

Cross-Validations:
  ⚠ Gross Profit = Revenue - Cost of Sales: Skipped - missing: gross_profit

Reference Validation: (not run - use --validate-reference to enable)

═══════════════════════════════════════════════════════════════
```

## Period-Specific Notes

### Q1 Reports (Trimestre 1)
- Downloaded month: March (03)
- YTD period: 01-01 to 31-03
- 3 months of data

### Q2 Reports (Trimestre 2)
- Downloaded month: June (06)
- YTD period: 01-01 to 30-06
- 6 months of data

### Q3 Reports (Trimestre 3)
- Downloaded month: September (09)
- YTD period: 01-01 to 30-09
- 9 months of data

### Q4 Reports / Annual
- Downloaded month: December (12)
- YTD period: 01-01 to 31-12
- 12 months of data (full year)

## File Naming Convention

| Document Type | File Pattern |
|---------------|--------------|
| Análisis Razonado | `analisis_razonado_YYYY_QN.pdf` |
| Estados Financieros | `estados_financieros_YYYY_QN.pdf` |
| XBRL Data | `estados_financieros_YYYY_QN.xml` |

## Known Variations by Period

Some line items may vary between periods:

1. **"Obligaciones por convenios colectivos"**: May be 0 or absent in some quarters (only present when collective bargaining occurs)
2. **"Gastos Diferidos, ajustes existencias y otros"**: Can be positive (adjustment) or negative
3. **Line item names**: May have slight variations (e.g., "y otros" suffix)

## Extraction Priority

1. **XBRL first**: Use for all aggregate totals (most reliable, structured data)
2. **PDF tables**: Use pdfplumber for detailed line items
3. **OCR fallback**: Only when PDF extraction fails (image-based tables)

Always validate OCR results against XBRL totals.

## Configuration Architecture

Configuration is organized into a hierarchical structure:

```
config/
├── config.json              # Shared project config: file patterns, period types, sources
├── xbrl_specs.json          # XBRL namespaces, global scaling factor
├── extraction_specs.json    # Legacy (being deprecated) - use sheet-specific configs
│
└── sheet1/                  # Sheet1-specific configuration
    ├── fields.json          # Field definitions, 27-row mapping
    ├── extraction.json      # PDF section rules (nota_21, nota_22)
    ├── xbrl_mappings.json   # XBRL fact mappings, validation rules
    └── reference_data.json  # Known-good values per period
```

### config/config.json (Shared)

Project-wide settings:

```json
{
  "sources": ["cmf", "pucobre"],
  "period_types": ["quarterly"],
  "file_patterns": {
    "estados_financieros_pdf": "estados_financieros_{year}_Q{quarter}.pdf",
    "estados_financieros_xbrl": "estados_financieros_{year}_Q{quarter}.xbrl"
  }
}
```

### config/sheet1/fields.json (Field Definitions)

Defines the 27-row structure and field metadata:

```json
{
  "name": "Ingresos y Costos",
  "value_fields": {
    "ingresos_ordinarios": {"type": "int", "section": "ingresos", "row": 1},
    "cv_gastos_personal": {"type": "int", "section": "nota_21", "row": 4},
    "total_costo_venta": {"type": "int", "section": "nota_21", "row": 15, "is_total": true},
    "ga_gastos_personal": {"type": "int", "section": "nota_22", "row": 20},
    "total_gasto_admin": {"type": "int", "section": "nota_22", "row": 27, "is_total": true}
  },
  "row_mapping": {
    "1": {"field": "ingresos_ordinarios", "label": "Ingresos de actividades ordinarias M USD"},
    "4": {"field": "cv_gastos_personal", "label": "Gastos en personal", "section": "costo_venta"}
  }
}
```

### config/sheet1/extraction.json (PDF Rules)

Keyword-based PDF field matching:

```json
{
  "sections": {
    "nota_21": {
      "title": "Costo de Venta",
      "search_patterns": ["21. costo", "21 costo", "nota 21"],
      "table_identifiers": {
        "unique_items": ["energía eléctrica", "servicios mineros", "fletes"],
        "exclude_items": ["gratificación", "comercialización"]
      },
      "field_mappings": {
        "cv_gastos_personal": {"match_keywords": ["gastos en personal"]},
        "cv_deprec_leasing": {"match_keywords": ["leasing"]},
        "cv_depreciacion_amort": {
          "match_keywords": ["amort"],
          "exclude_keywords": ["leasing", "arrendamiento"]
        }
      }
    },
    "nota_22": {
      "title": "Gastos de Administración y Ventas",
      "search_patterns": ["22. gastos", "22 gastos", "nota 22"],
      "field_mappings": {
        "ga_gratificacion": {"match_keywords": ["gratificación", "gratificacion"]},
        "ga_comercializacion": {"match_keywords": ["comercialización", "comercializacion"]}
      }
    }
  }
}
```

**Matching rules:**
- `match_keywords`: At least one keyword must match (case-insensitive)
- `exclude_keywords`: If any match, skip this field mapping
- Use unique identifiers to avoid overlapping matches

### config/sheet1/xbrl_mappings.json (XBRL Facts)

XBRL fact name mappings, section-to-field mappings, and validation:

```json
{
  "section_total_mapping": {
    "_description": "Maps PDF section_id to Sheet1Data total field name",
    "nota_21": "total_costo_venta",
    "nota_22": "total_gasto_admin"
  },
  "fact_mappings": {
    "ingresos_ordinarios": {
      "primary": "RevenueFromContractsWithCustomers",
      "fallbacks": ["Revenue", "IngresosPorActividadesOrdinarias"],
      "context_type": "duration",
      "apply_scaling": true
    },
    "total_costo_venta": {
      "primary": "CostOfSales",
      "fallbacks": ["CostoDeVentas"],
      "apply_scaling": true
    },
    "total_gasto_admin": {
      "primary": "AdministrativeExpense",
      "fallbacks": ["GastosDeAdministracion"],
      "apply_scaling": true
    }
  },
  "validation_rules": {
    "sum_tolerance": 1,
    "total_validations": [
      {"total_field": "total_costo_venta", "sum_fields": ["cv_gastos_personal", "cv_materiales", "..."]},
      {"total_field": "total_gasto_admin", "sum_fields": ["ga_gastos_personal", "ga_materiales", "..."]}
    ]
  }
}
```

**Key fields:**
- `section_total_mapping`: Maps PDF section IDs (nota_21, nota_22) to Sheet1Data total field names. Used by `sections_to_sheet1data()` to convert extraction results to Sheet1Data.
- `apply_scaling`: Set to `true` for fields that need conversion from USD to MUS$ (÷1000)
- `fallbacks`: Alternative XBRL fact names tried if primary not found
- `context_type`: "duration" for YTD values, "instant" for balance sheet values

### config/sheet1/reference_data.json (Validation)

Known-good values for validation by period:

```json
{
  "2024_Q2": {
    "verified": true,
    "verified_date": "2024-12-06",
    "values": {
      "ingresos_ordinarios": 179165,
      "total_costo_venta": -126202,
      "total_gasto_admin": -11632
    }
  },
  "2024_Q1": {
    "verified": true,
    "source": "Estados Financieros PDF - pucobre.cl (no XBRL)",
    "values": {"..."}
  }
}
```

### Loading Configuration (Python API)

```python
from puco_eeff.config import load_sheet_config, get_period_paths

# Load sheet1 configs
fields = load_sheet_config("sheet1", "fields")
extraction = load_sheet_config("sheet1", "extraction")
xbrl_mappings = load_sheet_config("sheet1", "xbrl_mappings")
reference = load_sheet_config("sheet1", "reference_data")

# Get file paths for a period
paths = get_period_paths(2024, 2)
# paths["raw_pdf"], paths["raw_xbrl"], paths["processed"]
```

This architecture allows learning from each quarter's extraction experience and supports adding new sheets (sheet2, sheet3) with their own configs.
