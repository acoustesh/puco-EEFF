# 05 - Format Sheet 1 for Excel

## Objective

Format the extracted Estado de Situación Financiera data into a clean structure ready for the final Excel workbook.

## Prerequisites

- Sheet 1 data extracted: `data/processed/YYYY_QN_sheet1.json`

## Steps

### 1. Load Extracted Data

```python
from puco_eeff.writer.sheet_writer import load_sheet_data
from puco_eeff.transformer.normalizer import normalize_financial_data

year, quarter = 2024, 3  # Adjust as needed
period = f"{year}_Q{quarter}"

# Load the extracted data
df = load_sheet_data("sheet1", period)
print(f"Loaded {len(df)} rows")
print(df)
```

### 2. Normalize Data

```python
# Normalize numeric values
df_normalized = normalize_financial_data(
    df,
    numeric_columns=["valor", "valor_anterior"]  # Adjust column names
)

print("Normalized data:")
print(df_normalized)
```

### 3. Apply Standard Structure

```python
import pandas as pd

# Define standard structure for Balance General
standard_structure = [
    # Activos
    {"concepto": "ACTIVOS", "nivel": 0, "tipo": "header"},
    {"concepto": "Activos corrientes", "nivel": 1, "tipo": "subtotal"},
    {"concepto": "Efectivo y equivalentes al efectivo", "nivel": 2, "tipo": "item"},
    {"concepto": "Otros activos financieros corrientes", "nivel": 2, "tipo": "item"},
    {"concepto": "Deudores comerciales y otras cuentas por cobrar", "nivel": 2, "tipo": "item"},
    {"concepto": "Inventarios", "nivel": 2, "tipo": "item"},
    {"concepto": "Total activos corrientes", "nivel": 1, "tipo": "total"},
    
    {"concepto": "Activos no corrientes", "nivel": 1, "tipo": "subtotal"},
    {"concepto": "Propiedades, planta y equipo", "nivel": 2, "tipo": "item"},
    {"concepto": "Activos intangibles", "nivel": 2, "tipo": "item"},
    {"concepto": "Total activos no corrientes", "nivel": 1, "tipo": "total"},
    
    {"concepto": "TOTAL ACTIVOS", "nivel": 0, "tipo": "grand_total"},
    
    # Pasivos
    {"concepto": "PASIVOS", "nivel": 0, "tipo": "header"},
    {"concepto": "Pasivos corrientes", "nivel": 1, "tipo": "subtotal"},
    {"concepto": "Otros pasivos financieros corrientes", "nivel": 2, "tipo": "item"},
    {"concepto": "Cuentas por pagar comerciales", "nivel": 2, "tipo": "item"},
    {"concepto": "Total pasivos corrientes", "nivel": 1, "tipo": "total"},
    
    {"concepto": "Pasivos no corrientes", "nivel": 1, "tipo": "subtotal"},
    {"concepto": "Otros pasivos financieros no corrientes", "nivel": 2, "tipo": "item"},
    {"concepto": "Total pasivos no corrientes", "nivel": 1, "tipo": "total"},
    
    {"concepto": "TOTAL PASIVOS", "nivel": 0, "tipo": "grand_total"},
    
    # Patrimonio
    {"concepto": "PATRIMONIO", "nivel": 0, "tipo": "header"},
    {"concepto": "Capital emitido", "nivel": 1, "tipo": "item"},
    {"concepto": "Ganancias acumuladas", "nivel": 1, "tipo": "item"},
    {"concepto": "Otras reservas", "nivel": 1, "tipo": "item"},
    {"concepto": "TOTAL PATRIMONIO", "nivel": 0, "tipo": "grand_total"},
    
    {"concepto": "TOTAL PASIVOS Y PATRIMONIO", "nivel": 0, "tipo": "grand_total"},
]
```

### 4. Map Extracted Values to Structure

```python
def map_to_structure(extracted_df, structure):
    """Map extracted values to standard structure."""
    result = []
    
    for item in structure:
        row = item.copy()
        # Try to find matching value in extracted data
        concepto_lower = item["concepto"].lower()
        
        for _, extracted_row in extracted_df.iterrows():
            if concepto_lower in str(extracted_row.get("concepto", "")).lower():
                row["valor_actual"] = extracted_row.get("valor")
                row["valor_anterior"] = extracted_row.get("valor_anterior")
                break
        
        result.append(row)
    
    return pd.DataFrame(result)

formatted_df = map_to_structure(df_normalized, standard_structure)
print(formatted_df)
```

### 5. Format Numbers

```python
def format_currency(value):
    """Format value as currency (thousands)."""
    if pd.isna(value) or value is None:
        return ""
    try:
        # Convert to millions or thousands as appropriate
        num = float(value)
        return f"{num:,.0f}"
    except (ValueError, TypeError):
        return str(value)

# Apply formatting (for display only, keep raw values for Excel)
formatted_df["valor_display"] = formatted_df["valor_actual"].apply(format_currency)
```

### 6. Validate Totals

```python
def validate_balance_sheet(df):
    """Validate balance sheet totals."""
    errors = []
    
    # Get key totals
    total_activos = df[df["concepto"] == "TOTAL ACTIVOS"]["valor_actual"].values
    total_pasivos = df[df["concepto"] == "TOTAL PASIVOS"]["valor_actual"].values
    total_patrimonio = df[df["concepto"] == "TOTAL PATRIMONIO"]["valor_actual"].values
    
    if len(total_activos) > 0 and len(total_pasivos) > 0 and len(total_patrimonio) > 0:
        activos = float(total_activos[0] or 0)
        pasivos = float(total_pasivos[0] or 0)
        patrimonio = float(total_patrimonio[0] or 0)
        
        # Check: Activos = Pasivos + Patrimonio
        diff = abs(activos - (pasivos + patrimonio))
        if diff > 1:  # Allow small rounding differences
            errors.append(f"Balance check failed: {activos} != {pasivos} + {patrimonio} (diff: {diff})")
    
    return errors

validation_errors = validate_balance_sheet(formatted_df)
if validation_errors:
    print("Validation errors:")
    for e in validation_errors:
        print(f"  - {e}")
else:
    print("✓ Balance sheet validation passed")
```

### 7. Save Formatted Data

```python
from puco_eeff.writer.sheet_writer import save_sheet_data, write_sheet_to_csv

# Save formatted JSON
save_sheet_data(
    sheet_name="estado_situacion_financiera",
    data=formatted_df,
    period=period
)

# Also save as CSV for review
write_sheet_to_csv(
    sheet_name="estado_situacion_financiera",
    data=formatted_df,
    period=period
)

print(f"✓ Sheet 1 formatted and saved")
```

## Output

- Formatted JSON: `data/processed/YYYY_QN_estado_situacion_financiera.json`
- Review CSV: `data/processed/YYYY_QN_estado_situacion_financiera.csv`

## Column Structure for Final Sheet

| Column | Description |
|--------|-------------|
| concepto | Row label/description |
| nivel | Indentation level (0-2) |
| tipo | Row type (header, subtotal, item, total, grand_total) |
| valor_actual | Current period value |
| valor_anterior | Previous period value (if available) |

## Next Steps

After completing this instruction:
1. Review the CSV file for accuracy
2. Adjust structure mapping if needed
3. Proceed to `06_locate_sheet2.md` for the next sheet
