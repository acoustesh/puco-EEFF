# 06 - Extract Data for Sheet 2 (Cuadro Resumen KPIs)

## Objective

Extract production and operational KPIs from Análisis Razonado PDF.

## Quick Start

```bash
# Extract Sheet2 for Q2 2024
python -m puco_eeff.main_sheet2 -y 2024 -q 2

# With reference validation
python -m puco_eeff.main_sheet2 -y 2024 -q 2 --validate-reference

# Skip download (use existing files)
python -m puco_eeff.main_sheet2 -y 2024 -q 2 --skip-download
```

## Extraction Steps

### 1. Programmatic Extraction

```python
from puco_eeff.sheets.sheet2 import extract_sheet2, print_sheet2_report

year, quarter = 2024, 2
data, issues = extract_sheet2(year, quarter)

if data:
    print_sheet2_report(data)
    print(f"\nValidation issues: {issues}")
else:
    print(f"Extraction failed: {issues}")
```

### 2. Access Extracted Values

```python
# Revenue breakdown (MUS$)
print(f"Cobre concentrados: {data.cobre_concentrados:,}")
print(f"Cobre cátodos:      {data.cobre_catodos:,}")
print(f"Oro subproducto:    {data.oro_subproducto:,}")
print(f"Plata subproducto:  {data.plata_subproducto:,}")
print(f"Total ingresos:     {data.total_ingresos:,}")

# Operational metrics
print(f"EBITDA:             {data.ebitda:,} MUS$")
print(f"Libras vendidas:    {data.libras_vendidas} MM lbs")
print(f"Cash Cost:          ${data.cash_cost}/lb")
print(f"Precio efectivo:    ${data.precio_efectivo}/lb")
```

### 3. Save to JSON

```python
from puco_eeff.sheets.sheet2 import save_sheet2_to_json

output_path = save_sheet2_to_json(data)
print(f"Saved to: {output_path}")
# Output: data/processed/sheet2_IIQ2024.json
```

## Field Matching

PDF labels are matched to fields using keywords from `config/sheet2/extraction.json`:

```python
from puco_eeff.sheets.sheet2 import match_concepto_to_field_sheet2

# Example: match PDF label to field
field = match_concepto_to_field_sheet2("Cobre en concentrados M USD", "resumen_ingresos")
print(field)  # "cobre_concentrados"

field = match_concepto_to_field_sheet2("Cash Cost (US$/lb)", "indicadores_operacionales")
print(field)  # "cash_cost"
```

## Spanish Number Parsing

The `parse_spanish_number()` function handles Spanish locale:

```python
from puco_eeff.sheets.sheet2 import parse_spanish_number

# Comma as decimal separator
parse_spanish_number("19,5")    # 19.5 (float)
parse_spanish_number("3,97")   # 3.97 (float)

# Period as thousands separator
parse_spanish_number("64.057") # 64057 (int)
parse_spanish_number("1.342")  # 1342 (int)
```

## Validation

### Sum Validation

Revenue products should sum to total:
```python
from puco_eeff.sheets.sheet2 import validate_sheet2_sums

issues = validate_sheet2_sums(data)
if issues:
    print("Sum validation failed:")
    for issue in issues:
        print(f"  - {issue}")
```

### Reference Validation

Compare against known-good values from `reference_data.json`:
```python
from puco_eeff.sheets.sheet2 import validate_sheet2_against_reference

issues = validate_sheet2_against_reference(data)
if issues is None:
    print("No reference data for this period")
elif not issues:
    print("✓ All values match reference data")
else:
    print("Reference mismatches:")
    for issue in issues:
        print(f"  - {issue}")
```

## Output Format

JSON output in `data/processed/sheet2_{quarter}Q{year}.json`:

```json
{
  "quarter": "IIQ2024",
  "year": 2024,
  "quarter_num": 2,
  "source": "reference_data",
  "cobre_concentrados": 142391,
  "cobre_catodos": 15022,
  "oro_subproducto": 20223,
  "plata_subproducto": 1529,
  "total_ingresos": 179165,
  "ebitda": 65483,
  "libras_vendidas": 38.4,
  "cobre_fino": 38.0,
  "precio_efectivo": 4.3,
  "cash_cost": 2.59,
  "costo_unitario_total": 3.57,
  "non_cash_cost": 0.98,
  "toneladas_procesadas": 2683,
  "oro_onzas": 8.2
}
```

## Next Steps

Proceed to `07_format_sheet2.md`
