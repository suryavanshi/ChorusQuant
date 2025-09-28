from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("pytesseract")

import fitz
import numpy as np

from agents.ingest.ocr import OCRSettings, extract_pdf_text_with_ocr


def _make_skewed_scan(tmp_path: Path) -> tuple[Path, str]:
    reference_text = "Account Summary\nBalance 1234.56"
    lines = reference_text.splitlines()

    canvas = np.full((420, 600), 255, dtype=np.uint8)
    y = 150
    for line in lines:
        cv2.putText(canvas, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3, cv2.LINE_AA)
        y += 80

    center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
    rotation = cv2.getRotationMatrix2D(center, 8.5, 1.0)
    rotated = cv2.warpAffine(canvas, rotation, (canvas.shape[1], canvas.shape[0]), borderValue=255)
    color = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

    png_path = tmp_path / "scan.png"
    cv2.imwrite(str(png_path), color)

    pdf_path = tmp_path / "scan.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    rect = fitz.Rect(36, 100, 576, 700)
    page.insert_image(rect, filename=str(png_path))
    doc.save(pdf_path)

    return pdf_path, reference_text


def _token_set(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dist = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + cost,
            )
    if not ref_words:
        return 0.0
    return dist[-1][-1] / len(ref_words)


def test_ocr_recall_beats_text_layer(tmp_path):
    pdf_path, _ = _make_skewed_scan(tmp_path)

    extraction = extract_pdf_text_with_ocr(pdf_path, OCRSettings(dpi=300, min_confidence=50))
    page = extraction.pages[0]

    text_tokens = _token_set(page.text_layer)
    merged_tokens = _token_set(page.merged_text)

    assert len(text_tokens) < len(merged_tokens)
    assert abs(page.deskew_angle) > 1.0
    assert abs(page.deskew_angle) < 15.0


def test_ocr_word_error_rate_reasonable(tmp_path):
    pdf_path, reference = _make_skewed_scan(tmp_path)

    extraction = extract_pdf_text_with_ocr(pdf_path, OCRSettings(dpi=300, min_confidence=40))
    page = extraction.pages[0]

    wer = _word_error_rate(reference, page.ocr_text)

    assert wer < 0.4
