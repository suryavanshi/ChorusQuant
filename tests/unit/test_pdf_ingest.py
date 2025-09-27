from __future__ import annotations

import base64
from pathlib import Path

import fitz

from agents.ingest.pdf_text import extract_pdf_text
from agents.ingest.sniff_mime import sniff_path


_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9pXt1l0AAAAASUVORK5CYII="
)


def _make_text_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Header", fontsize=16)
    page.insert_text((72, 144), "left one\nleft two", fontsize=12)
    page.insert_text((300, 144), "right one\nright two", fontsize=12)
    path = tmp_path / "layout.pdf"
    doc.save(path)
    return path


def _make_scanned_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    pix = fitz.Pixmap(fitz.csRGB, (0, 0, 400, 600), 0)
    pix.clear_with(255)
    page.insert_image(page.rect, pixmap=pix)
    path = tmp_path / "scan.pdf"
    doc.save(path)
    return path


def test_sniff_mime_pdf_features(tmp_path):
    pdf_path = _make_text_pdf(tmp_path)
    result = sniff_path(pdf_path)

    assert result.kind == "pdf"
    assert len(result.page_features) == 1

    page_feature = result.page_features[0]
    assert page_feature.word_count == 9
    assert page_feature.text_density > 0
    assert page_feature.image_fraction == 0


def test_sniff_mime_identifies_other_types(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("col1,col2\n1,2\n")
    csv_result = sniff_path(csv_path)
    assert csv_result.kind == "spreadsheet"

    png_path = tmp_path / "image.png"
    png_path.write_bytes(_PIXEL_PNG)
    image_result = sniff_path(png_path)
    assert image_result.kind == "image"


def test_extract_pdf_text_structure_sorted(tmp_path):
    pdf_path = _make_text_pdf(tmp_path)
    extraction = extract_pdf_text(pdf_path, sort=True)

    assert len(extraction.pages) == 1
    page = extraction.pages[0]
    block_texts = [block.text for block in page.blocks]
    assert block_texts == ["Header", "left one\nleft two", "right one\nright two"]
    assert page.word_count == 9
    assert page.glyph_count == len("Headerleft oneleft tworight oneright two".replace(" ", ""))


def test_scanned_pdf_detection(tmp_path):
    pdf_path = _make_scanned_pdf(tmp_path)
    extraction = extract_pdf_text(pdf_path)

    assert extraction.pages[0].image_fraction > 0.8
    assert extraction.likely_scanned is True
    assert extraction.scan_score >= 0.5
