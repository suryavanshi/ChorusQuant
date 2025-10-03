"""High-accuracy table extraction utilities for financial documents."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency guard
    import camelot  # type: ignore
except ImportError:  # pragma: no cover
    camelot = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover
    pdfplumber = None  # type: ignore[assignment]


CamelotReader = Callable[..., Any]
PdfPlumberOpen = Callable[[Path], Any]


@dataclass(slots=True)
class NormalizedCell:
    """Container for a single table cell with normalization metadata."""

    raw: str
    text: str
    numeric: Optional[float] = None
    currency: Optional[str] = None
    is_percentage: bool = False


@dataclass(slots=True)
class TieOutResult:
    """Validation output for a subtotal/total row."""

    label: str
    column_index: int
    column_name: str
    expected: float
    observed: float
    difference: float
    passed: bool
    tolerance: float


@dataclass(slots=True)
class TableQAReport:
    """Diagnostics gathered during post-processing and QA."""

    column_types: List[str] = field(default_factory=list)
    column_currencies: List[Optional[str]] = field(default_factory=list)
    header_rows: int = 0
    header_recovered: bool = False
    tie_out_results: List[TieOutResult] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractedTable:
    """Normalized table along with QA metadata and provenance."""

    columns: List[str]
    rows: List[List[str]]
    raw_rows: List[List[str]]
    page_numbers: List[int]
    flavor: str
    qa: TableQAReport


_CURRENCY_SYMBOLS = {
    "$": "USD",
    "US$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "C$": "CAD",
    "A$": "AUD",
    "CA$": "CAD",
    "HK$": "HKD",
    "CHF": "CHF",
    "₹": "INR",
    "₩": "KRW",
    "₺": "TRY",
}

_CURRENCY_CODES = {
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CAD",
    "AUD",
    "CHF",
    "CNY",
    "HKD",
    "SEK",
    "NOK",
    "DKK",
    "INR",
    "KRW",
    "SGD",
    "MXN",
    "BRL",
}


def extract_tables(
    path: Path,
    *,
    pages: Optional[Sequence[int]] = None,
    read_pdf: Optional[CamelotReader] = None,
    open_pdf: Optional[PdfPlumberOpen] = None,
    camelot_kwargs: Optional[dict[str, Any]] = None,
) -> List[ExtractedTable]:
    """Extract tables from a PDF using Camelot with pdfplumber-assisted heuristics."""

    camelot_reader = read_pdf or (camelot.read_pdf if camelot is not None else None)
    if camelot_reader is None:  # pragma: no cover - dependency missing in runtime env
        raise RuntimeError("Camelot is required for table extraction.")

    pdf_opener = open_pdf or (pdfplumber.open if pdfplumber is not None else None)
    if pdf_opener is None:  # pragma: no cover - dependency missing in runtime env
        raise RuntimeError("pdfplumber is required for table extraction.")

    camelot_kwargs = camelot_kwargs or {}

    tables: List[ExtractedTable] = []
    with pdf_opener(path) as pdf:
        total_pages = len(pdf.pages)
        page_candidates = list(pages) if pages is not None else list(range(1, total_pages + 1))

        for page_number in page_candidates:
            if page_number < 1 or page_number > total_pages:
                continue

            page = pdf.pages[page_number - 1]
            has_ruled_lines = _page_has_ruled_lines(page)
            flavors = ["lattice", "stream"] if has_ruled_lines else ["stream", "lattice"]
            seen_table_for_page = False

            for flavor in flavors:
                try:
                    camelot_tables = camelot_reader(
                        str(path),
                        flavor=flavor,
                        pages=str(page_number),
                        process_background=True,
                        **camelot_kwargs,
                    )
                except Exception:  # pragma: no cover - Camelot edge cases
                    continue

                if not camelot_tables:
                    continue

                for camelot_table in camelot_tables:
                    extracted = _convert_camelot_table(
                        camelot_table,
                        page_number=page_number,
                        flavor=flavor,
                        plumber_page=page,
                    )
                    tables.append(extracted)
                    seen_table_for_page = True

                if seen_table_for_page:
                    break

    return tables


def _page_has_ruled_lines(page: Any) -> bool:
    """Return True when the pdfplumber page contains vertical ruling lines."""

    edges: Iterable[Any] = getattr(page, "edges", []) or []
    rects: Iterable[Any] = getattr(page, "rects", []) or []

    for edge in edges:
        x0 = _safe_float(edge, "x0")
        x1 = _safe_float(edge, "x1")
        if abs(x0 - x1) <= 1.0:
            return True

    for rect in rects:
        x0 = _safe_float(rect, "x0")
        x1 = _safe_float(rect, "x1")
        if abs(x0 - x1) <= 1.0:
            return True

    return False


def _safe_float(obj: Any, key: str) -> float:
    """Best-effort float coercion for either dict-like or attr-like objects."""

    if hasattr(obj, key):
        try:
            return float(getattr(obj, key))
        except (TypeError, ValueError):
            return 0.0
    if isinstance(obj, dict) and key in obj:
        try:
            return float(obj[key])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _convert_camelot_table(
    table: Any,
    *,
    page_number: int,
    flavor: str,
    plumber_page: Any,
) -> ExtractedTable:
    raw_rows = _table_to_rows(table)
    normalized_rows = [[_normalize_cell(value) for value in row] for row in raw_rows]

    header_row_count = _estimate_header_rows(normalized_rows)
    header_cells = normalized_rows[:header_row_count] if header_row_count else []
    data_cells = normalized_rows[header_row_count:] if header_row_count else normalized_rows

    if header_cells:
        pre_columns = _compose_headers(header_cells, plumber_page=plumber_page)
    else:
        width = len(normalized_rows[0]) if normalized_rows else 0
        pre_columns = ["" for _ in range(width)]

    column_types = _infer_column_types(data_cells)
    column_currencies = _infer_column_currencies(data_cells)

    if not pre_columns and column_types:
        pre_columns = ["" for _ in column_types]
    elif len(pre_columns) < len(column_types):
        pre_columns.extend(["" for _ in range(len(column_types) - len(pre_columns))])

    columns, header_recovered = _recover_headers(pre_columns, column_types)

    data_rows = [[cell.text for cell in row] for row in data_cells]
    raw_data_rows = [[cell.raw for cell in row] for row in data_cells]

    tie_outs = _evaluate_tie_outs(data_cells, columns, column_types)

    issues: List[str] = []
    for result in tie_outs:
        if not result.passed:
            issues.append(
                f"Tie-out failed for {result.label!r} in column {result.column_name!r} "
                f"(Δ={result.difference:.2f}, tol={result.tolerance:.2f})"
            )

    qa = TableQAReport(
        column_types=column_types,
        column_currencies=column_currencies,
        header_rows=header_row_count,
        header_recovered=header_recovered,
        tie_out_results=tie_outs,
        issues=issues,
    )

    return ExtractedTable(
        columns=columns,
        rows=data_rows,
        raw_rows=raw_data_rows,
        page_numbers=[page_number],
        flavor=flavor,
        qa=qa,
    )


def _table_to_rows(table: Any) -> List[List[str]]:
    """Convert Camelot table data frame into a list of raw strings."""

    data_frame = getattr(table, "df", table)

    if hasattr(data_frame, "values"):
        values = data_frame.values
        if hasattr(values, "tolist"):
            rows = values.tolist()
        else:
            rows = [list(row) for row in values]
    elif hasattr(data_frame, "tolist"):
        rows = data_frame.tolist()
    else:
        rows = list(data_frame)

    cleaned_rows: List[List[str]] = []
    for row in rows:
        cleaned_row: List[str] = []
        for value in row:
            if value is None:
                cleaned_row.append("")
                continue
            if isinstance(value, float) and math.isnan(value):
                cleaned_row.append("")
                continue
            cleaned_row.append(str(value))
        cleaned_rows.append(cleaned_row)

    return cleaned_rows


def _normalize_cell(raw: str) -> NormalizedCell:
    """Normalize a table cell by cleaning whitespace and parsing numerics."""

    raw_text = raw if raw is not None else ""
    raw_text = raw_text.replace("\u00a0", " ")
    compact_text = " ".join(raw_text.strip().split())

    if not compact_text:
        return NormalizedCell(raw="", text="")

    text_without_parens = compact_text
    is_parenthetical = False
    if text_without_parens.startswith("(") and text_without_parens.endswith(")"):
        is_parenthetical = True
        text_without_parens = text_without_parens[1:-1].strip()

    currency, remainder = _extract_currency(text_without_parens)

    remainder = (
        remainder.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )

    is_percentage = False
    if remainder.endswith("%"):
        is_percentage = True
        remainder = remainder[:-1].strip()

    numeric_value = _parse_number(remainder)
    if numeric_value is None:
        return NormalizedCell(
            raw=compact_text,
            text=compact_text,
            currency=currency,
            numeric=None,
            is_percentage=is_percentage,
        )

    if is_parenthetical:
        numeric_value *= -1

    text_value = _format_numeric(numeric_value)
    text = f"{text_value}%" if is_percentage else text_value

    return NormalizedCell(
        raw=compact_text,
        text=text,
        numeric=numeric_value,
        currency=currency,
        is_percentage=is_percentage,
    )


def _extract_currency(text: str) -> Tuple[Optional[str], str]:
    """Extract currency symbols or codes from the supplied string."""

    if not text:
        return None, ""

    for symbol, code in sorted(_CURRENCY_SYMBOLS.items(), key=lambda item: -len(item[0])):
        if text.startswith(symbol):
            return code, text[len(symbol) :].strip()
        if text.endswith(symbol):
            return code, text[: -len(symbol)].strip()

    tokens = text.split()
    if tokens:
        first, last = tokens[0], tokens[-1]
        if first in _CURRENCY_CODES:
            return first, " ".join(tokens[1:]).strip()
        if last in _CURRENCY_CODES:
            return last, " ".join(tokens[:-1]).strip()

    return None, text


def _parse_number(value: str) -> Optional[float]:
    """Parse locale-formatted numerics into floats."""

    if not value:
        return None

    text = value.replace("'", "")
    text = text.replace(" ", "")

    if text.endswith("-") and text[:-1]:
        text = f"-{text[:-1]}"

    decimal_sep = "."
    thousands_sep: Optional[str] = None

    if "," in text and "." in text:
        last_comma = text.rfind(",")
        last_dot = text.rfind(".")
        if last_comma > last_dot:
            decimal_sep = ","
            thousands_sep = "."
        else:
            decimal_sep = "."
            thousands_sep = ","
    elif "," in text:
        parts = text.split(",")
        digits_only = all(part.isdigit() for part in parts if part)
        if digits_only and len(parts) > 1:
            if len(parts) == 2 and len(parts[-1]) <= 2:
                decimal_sep = ","
                thousands_sep = "."
            elif all(len(part) == 3 for part in parts[1:]):
                decimal_sep = "."
                thousands_sep = ","
            else:
                decimal_sep = ","
                thousands_sep = "."
        else:
            decimal_sep = ","
            thousands_sep = "."
    elif "." in text:
        parts = text.split(".")
        digits_only = all(part.isdigit() for part in parts if part)
        if digits_only and len(parts) > 1:
            if len(parts) == 2 and len(parts[-1]) <= 2:
                decimal_sep = "."
            elif all(len(part) == 3 for part in parts[1:]):
                thousands_sep = "."
            else:
                decimal_sep = "."
        else:
            decimal_sep = "."

    if thousands_sep:
        text = text.replace(thousands_sep, "")

    if decimal_sep != ".":
        text = text.replace(decimal_sep, ".")

    if text in {"", ".", "-", "+"}:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def _format_numeric(value: float) -> str:
    """Format numeric output while removing trailing zeros."""

    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text in {"", "-"}:
        text = "0"
    return text


def _estimate_header_rows(rows: List[List[NormalizedCell]]) -> int:
    """Estimate how many rows at the top correspond to headers."""

    header_rows = 0
    for row in rows:
        if not row:
            header_rows += 1
            continue
        numeric_count = sum(1 for cell in row if cell.numeric is not None)
        if numeric_count >= max(1, len(row) // 2):
            break
        header_rows += 1
    return header_rows


def _compose_headers(
    header_rows: List[List[NormalizedCell]],
    *,
    plumber_page: Any,
) -> List[str]:
    """Combine multiple header lines, optionally consulting pdfplumber for context."""

    if not header_rows:
        return []

    column_count = max(len(row) for row in header_rows)
    combined: List[str] = []
    for col_idx in range(column_count):
        parts: List[str] = []
        for row in header_rows:
            if col_idx >= len(row):
                continue
            text = row[col_idx].text
            if text:
                parts.append(text)
        combined.append(" ".join(part for part in parts if part).strip())

    if plumber_page is not None and all(not header for header in combined):
        context_lines = _extract_context_headers(plumber_page)
        combined = [
            header or context_lines[idx] if idx < len(context_lines) else header
            for idx, header in enumerate(combined)
        ]

    return combined


def _extract_context_headers(page: Any) -> List[str]:
    """Leverage pdfplumber to recover header lines when Camelot fails."""

    text = getattr(page, "extract_text", lambda: "")()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()][:5]


def _recover_headers(headers: List[str], column_types: List[str]) -> Tuple[List[str], bool]:
    """Ensure headers are populated and unique, generating fallbacks when required."""

    if not headers:
        headers = ["" for _ in column_types]

    recovered = False
    result: List[str] = []
    seen: dict[str, int] = {}

    for idx, header in enumerate(headers):
        if header:
            name = header
        else:
            recovered = True
            inferred = column_types[idx] if idx < len(column_types) else "column"
            name = inferred.capitalize() if inferred else f"Column {idx + 1}"
        base_name = name
        counter = 1
        while name in seen:
            counter += 1
            name = f"{base_name}_{counter}"
            recovered = True
        seen[name] = idx
        result.append(name)

    return result, recovered


def _infer_column_types(rows: List[List[NormalizedCell]]) -> List[str]:
    """Infer column types based on normalized cell metadata."""

    if not rows:
        return []

    column_count = max(len(row) for row in rows)
    column_types: List[str] = []

    for col_idx in range(column_count):
        numeric = 0
        percentage = 0
        non_empty = 0
        currencies: set[str] = set()

        for row in rows:
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            if cell.text:
                non_empty += 1
            if cell.numeric is not None:
                numeric += 1
            if cell.is_percentage:
                percentage += 1
            if cell.currency:
                currencies.add(cell.currency)

        if non_empty == 0:
            column_types.append("empty")
            continue

        numeric_ratio = numeric / non_empty
        percentage_ratio = percentage / non_empty

        if percentage_ratio >= 0.5:
            column_types.append("percentage")
        elif numeric_ratio >= 0.6 and currencies:
            column_types.append("currency")
        elif numeric_ratio >= 0.6:
            column_types.append("numeric")
        else:
            column_types.append("text")

    return column_types


def _infer_column_currencies(rows: List[List[NormalizedCell]]) -> List[Optional[str]]:
    """Infer dominant currency per column when numeric data is present."""

    if not rows:
        return []

    column_count = max(len(row) for row in rows)
    column_currencies: List[Optional[str]] = []

    for col_idx in range(column_count):
        counts: dict[str, int] = {}
        total_numeric = 0
        for row in rows:
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            if cell.numeric is not None:
                total_numeric += 1
            if cell.currency:
                counts[cell.currency] = counts.get(cell.currency, 0) + 1
        if not counts or total_numeric == 0:
            column_currencies.append(None)
            continue
        currency, count = max(counts.items(), key=lambda item: item[1])
        coverage = count / max(1, total_numeric)
        column_currencies.append(currency if coverage >= 0.5 else None)

    return column_currencies


def _evaluate_tie_outs(
    rows: List[List[NormalizedCell]],
    columns: List[str],
    column_types: List[str],
    *,
    abs_tolerance: float = 1.0,
    rel_tolerance: float = 0.01,
) -> List[TieOutResult]:
    """Perform subtotal/total tie-out checks on numeric columns."""

    if not rows:
        return []

    numeric_columns = [
        idx
        for idx, column_type in enumerate(column_types)
        if column_type in {"numeric", "currency", "percentage"}
    ]

    if not numeric_columns:
        return []

    results: List[TieOutResult] = []
    running_totals = {index: 0.0 for index in numeric_columns}

    for row in rows:
        label = row[0].text if row else ""
        normalized_label = label.lower()
        is_total = any(keyword in normalized_label for keyword in ("total", "subtotal"))

        if is_total:
            for column_index in numeric_columns:
                cell = row[column_index] if column_index < len(row) else None
                expected = cell.numeric if cell and cell.numeric is not None else None
                observed = running_totals[column_index]

                if expected is None:
                    running_totals[column_index] = 0.0
                    continue

                tolerance = max(abs_tolerance, rel_tolerance * max(abs(expected), 1.0))
                difference = observed - expected
                passed = abs(difference) <= tolerance

                results.append(
                    TieOutResult(
                        label=label or "Total",
                        column_index=column_index,
                        column_name=columns[column_index] if column_index < len(columns) else str(column_index),
                        expected=expected,
                        observed=observed,
                        difference=difference,
                        passed=passed,
                        tolerance=tolerance,
                    )
                )

                running_totals[column_index] = 0.0

            continue

        for column_index in numeric_columns:
            if column_index >= len(row):
                continue
            cell = row[column_index]
            if cell.numeric is not None:
                running_totals[column_index] += cell.numeric

    return results


__all__ = [
    "ExtractedTable",
    "TableQAReport",
    "TieOutResult",
    "extract_tables",
]
