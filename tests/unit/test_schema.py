"""Unit tests for Pydantic schemas used across the pipeline."""
from __future__ import annotations

import pytest

from agents.normalize.schema import DocumentData, Evidence, FeeItem, PortfolioHolding


def test_document_data_defaults() -> None:
    document = DocumentData()
    assert document.pages == []
    assert document.tables == []
    assert document.images == []
    assert document.metadata == {}
    assert document.document_id == "unknown"


def test_evidence_bbox_validation() -> None:
    evidence = Evidence(text="snippet", page=1, bbox=(0.0, 0.0, 10.0, 10.0))
    assert evidence.bbox == (0.0, 0.0, 10.0, 10.0)

    with pytest.raises(ValueError):
        Evidence(text="bad", page=1, bbox=(0.0, 1.0, 2.0))


def test_domain_models_accept_nested_evidence() -> None:
    evidence = Evidence(text="holding", page=2)
    holding = PortfolioHolding(name="Example Fund", value=1000.0, currency="USD", evidence=[evidence])
    fee = FeeItem(description="Management Fee", amount=5.0, evidence=[evidence])

    document = DocumentData(portfolio=[holding], fees=[fee])
    assert document.portfolio[0].evidence[0].text == "holding"
    assert document.fees[0].description == "Management Fee"


def test_document_data_metadata_roundtrip() -> None:
    document = DocumentData(metadata={"document_id": "abc123"}, pages=["Hello"])
    assert document.document_id == "abc123"
    payload = document.model_dump()
    assert payload["metadata"]["document_id"] == "abc123"
