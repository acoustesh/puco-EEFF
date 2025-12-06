"""XBRL/XML parser for financial statements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lxml import etree

from puco_eeff.config import setup_logging

logger = setup_logging(__name__)

# Common XBRL namespaces
NAMESPACES = {
    "xbrli": "http://www.xbrl.org/2003/instance",
    "link": "http://www.xbrl.org/2003/linkbase",
    "xlink": "http://www.w3.org/1999/xlink",
    "ifrs-full": "http://xbrl.ifrs.org/taxonomy/2023-03-23/ifrs-full",
}


def parse_xbrl_file(file_path: Path) -> dict[str, Any]:
    """Parse an XBRL/XML file and extract financial data.

    Args:
        file_path: Path to the XBRL/XML file

    Returns:
        Dictionary containing extracted financial data
    """
    logger.info(f"Parsing XBRL file: {file_path}")

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"XBRL file not found: {file_path}")

    tree = etree.parse(str(file_path))  # noqa: S320
    root = tree.getroot()

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
    """Detect namespaces used in the XBRL document.

    Args:
        root: Root element of the XML tree

    Returns:
        Dictionary of namespace prefixes to URIs
    """
    namespaces = dict(NAMESPACES)  # Start with common namespaces

    # Add namespaces from the document
    for prefix, uri in root.nsmap.items():
        if prefix is not None:
            namespaces[prefix] = uri

    return namespaces


def _extract_facts(root: etree._Element, namespaces: dict[str, str]) -> list[dict[str, Any]]:
    """Extract XBRL facts (financial data points).

    Args:
        root: Root element of the XML tree
        namespaces: Namespace mapping

    Returns:
        List of fact dictionaries
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


def _extract_contexts(root: etree._Element, namespaces: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Extract XBRL contexts (periods and entity information).

    Args:
        root: Root element of the XML tree
        namespaces: Namespace mapping

    Returns:
        Dictionary mapping context IDs to context information
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
    """Get the local name of an element (without namespace).

    Args:
        element: XML element

    Returns:
        Local name string
    """
    if "}" in element.tag:
        return element.tag.split("}")[1]
    return element.tag


def extract_by_xpath(file_path: Path, xpath_expr: str) -> list[str]:
    """Extract values using an XPath expression.

    Args:
        file_path: Path to the XML file
        xpath_expr: XPath expression to evaluate

    Returns:
        List of extracted values as strings
    """
    logger.debug(f"Extracting with XPath: {xpath_expr}")

    tree = etree.parse(str(file_path))  # noqa: S320
    root = tree.getroot()

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
