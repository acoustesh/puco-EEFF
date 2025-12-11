# 05 - Locate Data Sources for Sheet 2 (Cuadro Resumen KPIs)

## Objective

Identify and document data sources for Sheet 2 - production and operational KPIs from Análisis Razonado.

## Context

Sheet 2 contains the **Cuadro Resumen de KPIs** with:
- Revenue breakdown by product (Cobre concentrados, Cobre cátodos, Oro, Plata)
- EBITDA
- Production metrics (libras vendidas, cobre fino, toneladas procesadas)
- Cost metrics (Cash Cost, Non-Cash Cost, Costo Unitario Total)
- Price metrics (Precio efectivo de venta)

## Data Sources

### Primary Source: Análisis Razonado PDF

The preferred source is the standalone `analisis_razonado_{year}_Q{quarter}.pdf` which contains:
- Summary tables with quarterly KPIs
- Revenue breakdown by product type
- Operational indicators

**Location**: `data/raw/pdf/analisis_razonado_2024_Q2.pdf`

### Fallback Source: Estados Financieros PDF

Some quarters have the Análisis Razonado embedded at the **end** of the Estados Financieros PDF (typically pages 80+).

**Location**: `data/raw/pdf/estados_financieros_2024_Q2.pdf`

### XBRL Cross-Validation

XBRL provides `Revenue` fact for cross-validation of `total_ingresos`:
```python
from puco_eeff.extractor.xbrl_parser import parse_xbrl_file, get_facts_by_name
from puco_eeff.config import get_period_paths

year, quarter = 2024, 2
paths = get_period_paths(year, quarter)
xbrl_path = paths["raw_xbrl"] / f"estados_financieros_{year}_Q{quarter}.xbrl"

facts = parse_xbrl_file(xbrl_path)
revenue = get_facts_by_name(facts, "RevenueFromContractsWithCustomers")
print(f"XBRL Revenue: {revenue}")
```

## Field Mapping

| CSV Row | Field Name | Type | Unit | Source |
|---------|------------|------|------|--------|
| Cobre en concentrados M USD | `cobre_concentrados` | int | MUS$ | PDF |
| Cobre en Cátodos | `cobre_catodos` | int | MUS$ | PDF |
| Oro subproducto | `oro_subproducto` | int | MUS$ | PDF |
| Plata subproducto | `plata_subproducto` | int | MUS$ | PDF |
| Total | `total_ingresos` | int | MUS$ | PDF + XBRL validation |
| Ebitda del periodo | `ebitda` | int | MUS$ | PDF |
| Libras de cobre vendido | `libras_vendidas` | float | MM lbs | PDF |
| Cobre Fino Obtenido | `cobre_fino` | float | MM lbs | PDF |
| Precio efectivo de venta | `precio_efectivo` | float | US$/lb | PDF |
| Cash Cost | `cash_cost` | float | US$/lb | PDF |
| Costo unitario Total | `costo_unitario_total` | float | US$/lb | PDF |
| Non Cash cost | `non_cash_cost` | float | US$/lb | PDF |
| Total Toneladas Procesadas | `toneladas_procesadas` | float | miles | PDF |
| Oro en Onzas | `oro_onzas` | float | miles oz | PDF |

## Configuration Files

All configuration is in `config/sheet2/`:

- **fields.json**: Field definitions, types, units, row mapping
- **extraction.json**: PDF search patterns, field mappings with keywords
- **xbrl_mappings.json**: XBRL fact mappings for cross-validation
- **reference_data.json**: Verified values for IQ2024-IIIQ2025

## Decimal Format

Spanish locale uses:
- Comma (`,`) as decimal separator: `19,5` = 19.5
- Period (`.`) as thousands separator: `64.057` = 64057

The `parse_spanish_number()` function in `sheet2.py` handles this conversion.

## Next Steps

Proceed to `06_extract_sheet2.md`
