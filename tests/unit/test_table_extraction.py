from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pytest

from agents.ingest.tables import ExtractedTable, extract_tables


@dataclass
class LayoutCase:
    name: str
    tables: Dict[Tuple[str, str], List[List[List[str]]]]
    edges: Sequence[dict]
    rects: Sequence[dict]
    expected_flavor: str
    expected_columns: List[str]
    expected_rows: List[List[str]]
    expected_types: List[str]
    expected_currencies: List[str | None]
    expect_header_recovered: bool


class DummyValues:
    def __init__(self, rows: List[List[str]]) -> None:
        self._rows = rows

    def tolist(self) -> List[List[str]]:
        return [list(row) for row in self._rows]


class DummyDataFrame:
    def __init__(self, rows: List[List[str]]) -> None:
        self._values = DummyValues(rows)

    @property
    def values(self) -> DummyValues:
        return self._values


class DummyCamelotTable:
    def __init__(self, rows: List[List[str]]) -> None:
        self.df = DummyDataFrame(rows)


class CamelotReaderStub:
    def __init__(self, tables: Dict[Tuple[str, str], List[List[List[str]]]]) -> None:
        self._tables = tables
        self.calls: List[Tuple[str, str]] = []

    def __call__(self, path: str, *, flavor: str, pages: str, **_: dict) -> List[DummyCamelotTable]:
        self.calls.append((flavor, pages))
        payload = self._tables.get((flavor, pages), [])
        return [DummyCamelotTable(rows) for rows in payload]


class DummyPage:
    def __init__(self, *, edges: Iterable[dict] | None = None, rects: Iterable[dict] | None = None, text: str = "") -> None:
        self.edges = list(edges or [])
        self.rects = list(rects or [])
        self._text = text

    def extract_text(self) -> str:
        return self._text


class DummyPdf:
    def __init__(self, pages: List[DummyPage]) -> None:
        self.pages = pages

    def __enter__(self) -> "DummyPdf":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial context manager
        return False


def _open_pdf_factory(edges: Sequence[dict], rects: Sequence[dict], text: str = ""):
    page = DummyPage(edges=edges, rects=rects, text=text)
    pdf = DummyPdf([page])

    def _opener(_: Path) -> DummyPdf:
        return pdf

    return _opener


@pytest.mark.parametrize(
    "case",
    [
        LayoutCase(
            name="ruled_basic",
            tables={
                ("lattice", "1"): [
                    [
                        ["Holding", "Market Value", "Weight %"],
                        ["Alpha Fund", "$1,200.00", "25%"],
                        ["Beta Fund", "$3,600.00", "75%"],
                        ["Total", "$4,800.00", "100%"],
                    ]
                ]
            },
            edges=[{"x0": 10, "x1": 10.0}],
            rects=[],
            expected_flavor="lattice",
            expected_columns=["Holding", "Market Value", "Weight %"],
            expected_rows=[
                ["Alpha Fund", "1200", "25%"],
                ["Beta Fund", "3600", "75%"],
                ["Total", "4800", "100%"],
            ],
            expected_types=["text", "currency", "percentage"],
            expected_currencies=[None, "USD", None],
            expect_header_recovered=False,
        ),
        LayoutCase(
            name="multi_header",
            tables={
                ("lattice", "1"): [
                    [
                        ["Security", "Market", "Market"],
                        ["Name", "Value", "Value (%)"],
                        ["Gamma", "$2,000.00", "60%"],
                        ["Delta", "$1,333.33", "40%"],
                        ["Total", "$3,333.33", "100%"],
                    ]
                ]
            },
            edges=[{"x0": 20, "x1": 20.1}],
            rects=[],
            expected_flavor="lattice",
            expected_columns=["Security Name", "Market Value", "Market Value (%)"],
            expected_rows=[
                ["Gamma", "2000", "60%"],
                ["Delta", "1333.33", "40%"],
                ["Total", "3333.33", "100%"],
            ],
            expected_types=["text", "currency", "percentage"],
            expected_currencies=[None, "USD", None],
            expect_header_recovered=False,
        ),
        LayoutCase(
            name="no_vertical_lines",
            tables={
                ("stream", "1"): [
                    [
                        ["Holding", "Value"],
                        ["Alpha", "$1,000"],
                        ["Beta", "$2,000"],
                        ["Total", "$3,000"],
                    ]
                ]
            },
            edges=[],
            rects=[],
            expected_flavor="stream",
            expected_columns=["Holding", "Value"],
            expected_rows=[
                ["Alpha", "1000"],
                ["Beta", "2000"],
                ["Total", "3000"],
            ],
            expected_types=["text", "currency"],
            expected_currencies=[None, "USD"],
            expect_header_recovered=False,
        ),
        LayoutCase(
            name="no_header_numeric_first_row",
            tables={
                ("lattice", "1"): [
                    [
                        ["Omega", "$500", "10%"],
                        ["Sigma", "$4,500", "90%"],
                        ["Total", "$5,000", "100%"],
                    ]
                ]
            },
            edges=[{"x0": 5, "x1": 5.1}],
            rects=[],
            expected_flavor="lattice",
            expected_columns=["Text", "Currency", "Percentage"],
            expected_rows=[
                ["Omega", "500", "10%"],
                ["Sigma", "4500", "90%"],
                ["Total", "5000", "100%"],
            ],
            expected_types=["text", "currency", "percentage"],
            expected_currencies=[None, "USD", None],
            expect_header_recovered=True,
        ),
        LayoutCase(
            name="european_negative",
            tables={
                ("stream", "1"): [
                    [
                        ["Holding", "Value", "Weight %"],
                        ["Alpha", "€1.234,56", "12,5%"],
                        ["Beta", "(€2.000,44)", "87,5%"],
                        ["Total", "€ -765,88", "100,0%"],
                    ]
                ]
            },
            edges=[],
            rects=[],
            expected_flavor="stream",
            expected_columns=["Holding", "Value", "Weight %"],
            expected_rows=[
                ["Alpha", "1234.56", "12.5%"],
                ["Beta", "-2000.44", "87.5%"],
                ["Total", "-765.88", "100%"],
            ],
            expected_types=["text", "currency", "percentage"],
            expected_currencies=[None, "EUR", None],
            expect_header_recovered=False,
        ),
    ],
)
def test_golden_holdings_layouts(case: LayoutCase) -> None:
    reader = CamelotReaderStub(case.tables)
    pdf_opener = _open_pdf_factory(case.edges, case.rects)

    tables = extract_tables(Path("dummy.pdf"), read_pdf=reader, open_pdf=pdf_opener)
    assert tables, f"no tables extracted for case {case.name}"

    table = tables[0]
    assert isinstance(table, ExtractedTable)
    assert table.flavor == case.expected_flavor
    assert table.columns == case.expected_columns
    assert table.rows == case.expected_rows
    assert table.qa.column_types == case.expected_types
    assert table.qa.column_currencies == case.expected_currencies
    assert table.qa.header_recovered is case.expect_header_recovered

    if table.qa.tie_out_results:
        assert all(result.passed for result in table.qa.tie_out_results)

    if case.name == "ruled_basic":
        assert reader.calls[0][0] == "lattice"
    if case.name == "no_vertical_lines":
        assert reader.calls[0][0] == "stream"


def test_header_context_recovery_when_all_headers_blank() -> None:
    reader = CamelotReaderStub(
        {
            ("lattice", "1"): [
                [
                    ["", ""],
                    ["Alpha", "$1,000"],
                    ["Total", "$1,000"],
                ]
            ]
        }
    )
    pdf_opener = _open_pdf_factory(edges=[{"x0": 1, "x1": 1.0}], rects=[], text="Name\nValue")

    tables = extract_tables(Path("dummy.pdf"), read_pdf=reader, open_pdf=pdf_opener)
    table = tables[0]
    assert table.columns == ["Name", "Value"]
    assert table.qa.header_recovered is False
