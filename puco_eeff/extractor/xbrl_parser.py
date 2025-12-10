"""XBRL/XML parsing utilities for financial statements.

Parses CMF/Pucobre XBRL instance documents with lxml, extracts facts and
contexts, and surfaces aggregate values used for Sheet1 cross-validation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from lxml import etree

from puco_eeff.config import get_config, setup_logging

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logging(__name__)


def _get_xbrl_config(config: dict | None = None) -> tuple[dict[str, str], list[str]]:
    """Return namespaces and aggregate fact names from config.

    Parameters
    ----------
    config : dict | None, optional
        Preloaded config; when ``None`` it is fetched via :func:`get_config`.

    Returns
    -------
    tuple[dict[str, str], list[str]]
        Namespace mapping and aggregate fact identifiers.
    """
    if config is None:
        config = get_config()

    xbrl_config = config.get("xbrl", {})

    namespaces = xbrl_config.get(
        "namespaces",
        {
            "xbrli": "http://www.xbrl.org/2003/instance",
            "link": "http://www.xbrl.org/2003/linkbase",
            "xlink": "http://www.w3.org/1999/xlink",
            "ifrs-full": "http://xbrl.ifrs.org/taxonomy/2023-03-23/ifrs-full",
        },
    )

    aggregate_facts = xbrl_config.get(
        "aggregate_facts",
        [
            "RevenueFromContractsWithCustomers",
            "Revenue",
            "CostOfSales",
            "GrossProfit",
            "AdministrativeExpense",
            "SellingExpense",
            "ProfitLoss",
            "ProfitLossBeforeTax",
        ],
    )

    return namespaces, aggregate_facts


# Common XBRL namespaces (module-level for backward compatibility)
NAMESPACES = {
    "xbrli": "http://www.xbrl.org/2003/instance",
    "link": "http://www.xbrl.org/2003/linkbase",
    "xlink": "http://www.w3.org/1999/xlink",
    "ifrs-full": "http://xbrl.ifrs.org/taxonomy/2023-03-23/ifrs-full",
}


def parse_xbrl_file(file_path: Path) -> dict[str, Any]:
    """Parse an XBRL/XML instance into facts and contexts.

    Handles both ``.xml`` and ``.xbrl`` files; retries with ISO-8859-1 decoding
    when the initial parse fails. Uses lxml to retain namespaces and context
    references.

    Parameters
    ----------
    file_path : Path
        Path to the XBRL/XML instance.

    Returns
    -------
    dict[str, Any]
        Parsed payload with ``facts``, ``contexts``, detected ``namespaces``,
        and the ``source_file`` path.

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    lxml.etree.XMLSyntaxError
        If parsing fails even after the ISO-8859-1 fallback.
    """
    logger.info("Parsing XBRL file: %s", file_path)

    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        msg = f"XBRL file not found: {file_path}"
        raise FileNotFoundError(msg)

    # Read file content and parse with encoding detection
    content = file_path.read_bytes()

    # Try parsing - lxml handles encoding declaration automatically
    try:
        root = etree.fromstring(content)
    except etree.XMLSyntaxError as e:
        # Fallback: try with explicit ISO-8859-1 encoding
        logger.debug("Initial parse failed, trying ISO-8859-1: %s", e)
        content_str = content.decode("iso-8859-1")
        root = etree.fromstring(content_str.encode("utf-8"))

    # Detect namespaces from the document
    doc_namespaces = _detect_namespaces(root)
    logger.debug(f"Detected namespaces: {list(doc_namespaces.keys())}")

    # Extract facts from XBRL
    facts = _extract_facts(root, doc_namespaces)
    logger.info(f"Extracted {len(facts)} facts from XBRL")

    # Extract contexts (periods, entities)
    contexts = _extract_contexts(root, doc_namespaces)
    logger.debug(f"Extracted {len(contexts)} contexts")

    return {
        "facts": facts,
        "contexts": contexts,
        "namespaces": doc_namespaces,
        "source_file": str(file_path),
    }


def _detect_namespaces(root: etree._Element) -> dict[str, str]:
    """Merge common namespaces with those declared in the document root.

    Parameters
    ----------
    root : etree._Element
        Parsed XBRL root element.

    Returns
    -------
    dict[str, str]
        Namespace prefixes mapped to URIs.
    """
    namespaces = dict(NAMESPACES)  # Start with common namespaces

    # Add namespaces from the document
    namespaces.update({prefix: uri for prefix, uri in root.nsmap.items() if prefix is not None})

    return namespaces


def _extract_facts(root: etree._Element, namespaces: dict[str, str]) -> list[dict[str, Any]]:
    """Collect fact elements with context and unit references.

    Parameters
    ----------
    root : etree._Element
        Parsed XBRL root element.
    namespaces : dict[str, str]
        Namespace mapping (unused here but kept for future filters).

    Returns
    -------
    list[dict[str, Any]]
        Facts with ``name``, ``namespace``, ``value``, ``context_ref``,
        ``unit_ref``, and ``decimals`` fields.
    """
    facts = []

    # Iterate through all elements that have a contextRef (these are facts)
    for element in root.iter():
        context_ref = element.get("contextRef")
        if context_ref is not None:
            fact = {
                "name": _get_local_name(element),
                "namespace": element.tag.split("}")[0].strip("{") if "}" in element.tag else "",
                "value": element.text,
                "context_ref": context_ref,
                "unit_ref": element.get("unitRef"),
                "decimals": element.get("decimals"),
            }
            facts.append(fact)

    return facts


def _extract_contexts(
    root: etree._Element,
    namespaces: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Extract period/entity contexts referenced by facts.

    Parameters
    ----------
    root : etree._Element
        Parsed XBRL root element.
    namespaces : dict[str, str]
        Namespace mapping for XPath queries.

    Returns
    -------
    dict[str, dict[str, Any]]
        Context metadata keyed by ``contextRef`` identifiers.
    """
    contexts: dict[str, dict[str, Any]] = {}

    # Find all context elements
    for context in root.findall(".//xbrli:context", namespaces):
        context_id = context.get("id")
        if context_id is None:
            continue

        context_info: dict[str, Any] = {"id": context_id}

        # Extract period information
        period = context.find("xbrli:period", namespaces)
        if period is not None:
            instant = period.find("xbrli:instant", namespaces)
            start_date = period.find("xbrli:startDate", namespaces)
            end_date = period.find("xbrli:endDate", namespaces)

            if instant is not None:
                context_info["period_type"] = "instant"
                context_info["instant"] = instant.text
            elif start_date is not None and end_date is not None:
                context_info["period_type"] = "duration"
                context_info["start_date"] = start_date.text
                context_info["end_date"] = end_date.text

        # Extract entity information
        entity = context.find("xbrli:entity", namespaces)
        if entity is not None:
            identifier = entity.find("xbrli:identifier", namespaces)
            if identifier is not None:
                context_info["entity_id"] = identifier.text
                context_info["entity_scheme"] = identifier.get("scheme")

        contexts[context_id] = context_info

    return contexts


def _get_local_name(element: etree._Element) -> str:
    """Return the element tag without namespace prefix."""
    if "}" in element.tag:
        return element.tag.split("}")[1]
    return element.tag


def extract_by_xpath(file_path: Path, xpath_expr: str) -> list[str]:
    """Evaluate an XPath expression against an XBRL/XML file.

    Parameters
    ----------
    file_path : Path
        XML file to parse.
    xpath_expr : str
        XPath expression using document namespaces.

    Returns
    -------
    list[str]
        Extracted string values in document order.
    """
    logger.debug("Extracting with XPath: %s", xpath_expr)

    # Read file content and parse with encoding detection
    content = file_path.read_bytes()
    try:
        root = etree.fromstring(content)
    except etree.XMLSyntaxError:
        content_str = content.decode("iso-8859-1")
        root = etree.fromstring(content_str.encode("utf-8"))

    # Use document namespaces
    namespaces = _detect_namespaces(root)

    results = root.xpath(xpath_expr, namespaces=namespaces)

    # Convert results to strings
    values = []
    for result in results:
        if isinstance(result, etree._Element):
            values.append(result.text or "")
        else:
            values.append(str(result))

    logger.debug(f"XPath returned {len(values)} values")
    return values


def get_facts_by_name(
    data: dict[str, Any],
    name_pattern: str,
    exact: bool = False,
) -> list[dict[str, Any]]:
    """Return facts whose names match a pattern.

    Parameters
    ----------
    data : dict[str, Any]
        Parsed payload from :func:`parse_xbrl_file`.
    name_pattern : str
        Fact name to match exactly or by substring.
    exact : bool, optional
        When ``True`` requires exact equality; otherwise case-insensitive substring match.

    Returns
    -------
    list[dict[str, Any]]
        Matching facts enriched with their context (when available).
    """
    matching_facts = []
    contexts = data.get("contexts", {})

    for fact in data.get("facts", []):
        fact_name = fact.get("name", "")

        matches = fact_name == name_pattern if exact else name_pattern.lower() in fact_name.lower()

        if matches:
            # Enrich fact with context information
            enriched_fact = dict(fact)
            context_ref = fact.get("context_ref")
            if context_ref and context_ref in contexts:
                enriched_fact["context"] = contexts[context_ref]
            matching_facts.append(enriched_fact)

    return matching_facts


def get_units(data: dict[str, Any]) -> dict[str, str]:
    """Infer unit identifiers present in parsed facts.

    Parameters
    ----------
    data : dict[str, Any]
        Parsed payload from :func:`parse_xbrl_file`.

    Returns
    -------
    dict[str, str]
        Unit IDs mapped to descriptive labels (currency hinted by namespace when possible).
    """
    # Units are stored in the facts with unitRef attribute
    units: dict[str, str] = {}

    for fact in data.get("facts", []):
        unit_ref = fact.get("unit_ref")
        if unit_ref and unit_ref not in units:
            # Try to infer unit type from the namespace
            if "iso4217" in str(fact.get("namespace", "")):
                units[unit_ref] = f"Currency: {unit_ref}"
            else:
                units[unit_ref] = unit_ref

    return units


def summarize_facts(data: dict[str, Any]) -> dict[str, int]:
    """Count facts by inferred CamelCase category prefix.

    Parameters
    ----------
    data : dict[str, Any]
        Parsed payload from :func:`parse_xbrl_file`.

    Returns
    -------
    dict[str, int]
        Category counts sorted descending.
    """
    categories: dict[str, int] = {}

    for fact in data.get("facts", []):
        name = fact.get("name", "Other")
        # Extract category from CamelCase name (take first capital-starting word)
        import re

        parts = re.findall(r"[A-Z][a-z]*", name)
        category = parts[0] if parts else "Other"
        categories[category] = categories.get(category, 0) + 1

    return dict(sorted(categories.items(), key=lambda x: -x[1]))


def extract_xbrl_aggregates(
    xml_path: Path,
    config: dict | None = None,
) -> dict[str, Any]:
    """Extract key aggregates (e.g., Revenue, CostOfSales) for the latest period.

    Parameters
    ----------
    xml_path : Path
        XBRL instance path.
    config : dict | None, optional
        Optional preloaded config; defaults to :func:`get_config`.

    Returns
    -------
    dict[str, Any]
        Aggregates dict including ``source_file``, ``period`` window, aggregate
        values, and the full fact list for debugging.
    """
    _, aggregate_fact_names = _get_xbrl_config(config)

    # Parse the XBRL file
    data = parse_xbrl_file(xml_path)
    contexts = data.get("contexts", {})

    # Find the current period context (duration type, most recent end date)
    current_context = None
    latest_end_date = ""

    for ctx_id, ctx_info in contexts.items():
        if ctx_info.get("period_type") == "duration":
            end_date = ctx_info.get("end_date", "")
            if end_date > latest_end_date:
                latest_end_date = end_date
                current_context = ctx_id

    logger.debug("Using context: %s (end: %s)", current_context, latest_end_date)

    # Extract aggregate values
    aggregates: dict[str, int | None] = {}

    for fact_name in aggregate_fact_names:
        matching = get_facts_by_name(data, fact_name, exact=True)

        # Filter to current period context
        for fact in matching:
            if fact.get("context_ref") == current_context:
                try:
                    value = int(float(fact.get("value", 0)))
                    aggregates[fact_name] = value
                    logger.debug(f"{fact_name}: {value:,}")
                except (ValueError, TypeError):
                    aggregates[fact_name] = None
                break

    # Get period info
    period_info = {}
    if current_context and current_context in contexts:
        ctx = contexts[current_context]
        period_info = {
            "start": ctx.get("start_date"),
            "end": ctx.get("end_date"),
        }

    return {
        "source_file": str(xml_path),
        "period": period_info,
        "aggregates": aggregates,
        "all_facts": data.get("facts", []),
    }


def save_xbrl_aggregates(
    aggregates: dict[str, Any],
    output_path: Path,
    config: dict | None = None,
) -> Path:
    """Persist XBRL aggregate output to JSON (without full fact list).

    Parameters
    ----------
    aggregates : dict[str, Any]
        Output from :func:`extract_xbrl_aggregates`.
    output_path : Path
        Destination JSON path.
    config : dict | None, optional
        Unused placeholder for API consistency.

    Returns
    -------
    Path
        Written file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a clean version for saving (exclude all_facts to keep file small)
    save_data = {
        "source_file": aggregates.get("source_file"),
        "period": aggregates.get("period"),
        "aggregates": aggregates.get("aggregates"),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    logger.info("Saved XBRL aggregates to: %s", output_path)
    return output_path
