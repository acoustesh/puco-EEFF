# puco-EEFF: Estados Financieros Extraction Pipeline

Pipeline para extraer Estados Financieros (EEFF) de Pucobre desde PDFs públicos y XBRL/XML de la CMF Chile, produciendo libros Excel multi-hoja con trazabilidad completa.

## Features

- **Descarga automática**: PDFs desde pucobre.cl y XBRL desde cmfchile.cl (Consolidado, IFRS)
- **Extracción inteligente**: Prioriza XML/XBRL, usa OCR para PDFs cuando es necesario
- **OCR con fallback**: Mistral OCR → OpenRouter/Anthropic → OpenRouter/OpenAI (3 reintentos exponenciales)
- **Auditoría completa**: Guarda todas las respuestas OCR y mapeo de fuentes
- **Re-ejecución parcial**: Cada hoja se guarda independientemente, permite re-procesar hojas individuales
- **Output estructurado**: `EEFF_YYYY_QN.xlsx` con múltiples hojas

## Quick Start

### 1. Clonar e instalar dependencias

```bash
git clone <repo-url>
cd puco-EEFF
poetry install
```

### 2. Instalar navegador headless

```bash
chmod +x setup_browser.sh
./setup_browser.sh
```

### 3. Configurar API keys

```bash
cp .env.example .env
# Editar .env con tus API keys
```

### 4. Ejecutar

El pipeline se ejecuta siguiendo las instrucciones en `instructions/` de forma secuencial.

## Estructura del Proyecto

```
puco-EEFF/
├── puco_eeff/              # Código fuente (package)
│   ├── config.py           # Configuración y clientes API
│   ├── scraper/            # Descarga PDFs + XBRL
│   ├── extractor/          # Parsing PDF + OCR
│   ├── transformer/        # Normalización y tracking
│   └── writer/             # Output Excel
├── config/
│   └── config.json         # URLs, selectores, mapeos
├── data/
│   ├── raw/                # PDFs y XBRL descargados
│   ├── processed/          # Datos extraídos por hoja
│   └── output/             # Excel final
├── audit/                  # Respuestas OCR y trazabilidad
├── notebooks/              # Exploración
├── logs/                   # Logs de ejecución
├── instructions/           # Guías para ejecución secuencial
└── tests/
```

## Configuración

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
