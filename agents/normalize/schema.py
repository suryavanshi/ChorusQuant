"""Pydantic schemas shared across agents."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

BBox = Tuple[float, float, float, float]


class Evidence(BaseModel):
    """Structured pointer to source material supporting an assertion."""

    text: str = Field(..., description="Extracted snippet that supports the claim.")
    page: int = Field(..., ge=1, description="1-indexed page number where the evidence was found.")
    bbox: Optional[BBox] = Field(
        default=None,
        description="Optional bounding box (x0, y0, x1, y1) in page coordinates.",
    )

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, value: Optional[Sequence[float]]) -> Optional[BBox]:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("Bounding box must contain exactly four float values.")
        return tuple(float(coord) for coord in value)  # type: ignore[return-value]


class Table(BaseModel):
    """Normalized table representation extracted from documents."""

    title: Optional[str] = Field(default=None, description="Optional table caption or title.")
    columns: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    page_span: List[int] = Field(default_factory=list, description="Pages that contain the table.")
    evidence: List[Evidence] = Field(default_factory=list, description="Evidence supporting table contents.")


class PortfolioHolding(BaseModel):
    """Domain model capturing a single portfolio position."""

    name: str = Field(..., description="Name of the holding or security.")
    ticker: Optional[str] = Field(default=None, description="Ticker symbol when available.")
    value: float = Field(..., ge=0.0, description="Market value of the position.")
    currency: str = Field(default="USD", description="Currency for the market value.")
    weight_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Portfolio weight expressed as a percentage.",
    )
    evidence: List[Evidence] = Field(default_factory=list, description="Supporting evidence spans.")


class FeeItem(BaseModel):
    """Domain model capturing an individual fee or expense item."""

    description: str = Field(..., description="Label for the fee.")
    amount: float = Field(..., description="Fee amount expressed in the given currency.")
    currency: str = Field(default="USD", description="Currency the fee amount is denominated in.")
    frequency: Optional[str] = Field(
        default=None,
        description="Cadence for the fee (e.g. monthly, annual).",
    )
    evidence: List[Evidence] = Field(default_factory=list, description="Supporting evidence spans.")


class DocumentData(BaseModel):
    """Normalized view of a financial document for downstream analysis."""

    pages: List[str] = Field(default_factory=list, description="Plain-text content split per page.")
    tables: List[Table] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list, description="Paths to extracted images or charts.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata such as document id, source, processing flags, etc.",
    )
    portfolio: List[PortfolioHolding] = Field(
        default_factory=list, description="Structured portfolio holdings extracted from the document."
    )
    fees: List[FeeItem] = Field(default_factory=list, description="Structured list of fee disclosures.")

    @property
    def document_id(self) -> str:
        """Convenience accessor for the document identifier."""

        value = self.metadata.get("document_id")
        return str(value) if value is not None else "unknown"

