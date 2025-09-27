"""Pydantic schemas shared across agents."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Structured pointer to source material supporting an assertion."""

    text: str = Field(..., description="Extracted snippet that supports the claim.")
    page: int = Field(..., ge=1, description="1-indexed page number where the evidence was found.")
    bbox: Optional[Sequence[float]] = Field(
        default=None,
        description="Optional bounding box (x0, y0, x1, y1) in page coordinates.",
    )


class Table(BaseModel):
    """Normalized table representation extracted from documents."""

    title: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    page_span: List[int] = Field(default_factory=list, description="Pages that contain the table.")
    evidence: List[Evidence] = Field(default_factory=list, description="Evidence supporting table contents.")


class DocumentData(BaseModel):
    """Normalized view of a financial document for downstream analysis."""

    text_pages: List[str] = Field(default_factory=list, description="Plain-text content split per page.")
    tables: List[Table] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list, description="Paths to extracted images or charts.")
    meta: Dict[str, str] = Field(default_factory=dict, description="Metadata such as document id, source, etc.")

    @property
    def document_id(self) -> str:
        """Convenience accessor for the document identifier."""

        return self.meta.get("document_id", "unknown")
