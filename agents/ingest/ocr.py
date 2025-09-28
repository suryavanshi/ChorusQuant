"""OCR utilities for scanned and hybrid PDFs.

This module implements a dual-path text extraction strategy:

1. A *text* path that uses PyMuPDF to recover any embedded text layer.
2. An *image* path that rasterizes each page, runs an OpenCV preprocessing
   pipeline (grayscale → threshold → deskew → denoise) and performs OCR with
   Tesseract via :mod:`pytesseract`.

The outputs from both paths are reconciled using a confidence-weighted merge.
When the PDF contains a reliable text layer we prefer it, but we patch gaps
with high-confidence OCR spans.  This approach improves recall on scanned or
partially scanned statements while preserving provenance for downstream
validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import List, Sequence, Tuple

import cv2
import fitz
import numpy as np
import pytesseract
from pytesseract import Output


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OCRSettings:
    """Configuration for the OCR pipeline."""

    dpi: int = 300
    """Rasterization resolution passed to PyMuPDF."""

    language: str = "eng"
    """Tesseract language pack specifier (comma separated for multiple)."""

    psm: int = 3
    """Tesseract page segmentation mode."""

    oem: int = 3
    """Tesseract OCR Engine mode."""

    min_confidence: float = 60.0
    """Minimum confidence (0-100) for OCR spans to be used in reconciliation."""


@dataclass(slots=True)
class OCRSupplement:
    """Text recovered from OCR that was absent in the text layer."""

    text: str
    confidence: float


@dataclass(slots=True)
class PageOCRResult:
    """Combined OCR and text-layer information for a single PDF page."""

    page_number: int
    text_layer: str
    ocr_text: str
    merged_text: str
    deskew_angle: float
    average_confidence: float
    supplements: List[OCRSupplement] = field(default_factory=list)


@dataclass(slots=True)
class OCRExtraction:
    """Dual-path extraction output for a PDF document."""

    pages: List[PageOCRResult]

    def merged_document_text(self) -> str:
        """Return the merged text across all pages."""

        return "\n\f\n".join(page.merged_text for page in self.pages)


def _rasterize_pdf(pdf_path: Path, dpi: int) -> List[np.ndarray]:
    """Rasterize each page of ``pdf_path`` into RGB numpy arrays."""

    doc = fitz.open(pdf_path)
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    images: List[np.ndarray] = []
    for page_number, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        samples = np.frombuffer(pix.samples, dtype=np.uint8)
        image = samples.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:  # pragma: no cover - defensive, alpha disabled above
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        images.append(image)
        LOGGER.debug("Rasterized page %d at %ddpi -> %s", page_number, dpi, image.shape)
    return images


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _threshold(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    combined = cv2.bitwise_or(otsu, adaptive)
    return combined


def _estimate_skew_angle(binary: np.ndarray) -> float:
    """Estimate the skew angle (in degrees) of a binarized page."""

    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angles: List[float] = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = theta * 180.0 / np.pi - 90.0
            if -45.0 < angle < 45.0:
                angles.append(angle)
    if not angles:
        coords = np.column_stack(np.where(binary > 0))
        if coords.size == 0:
            return 0.0
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45.0:
            angle = -(90.0 + angle)
        else:
            angle = -angle
        return float(angle)
    return float(np.median(angles))


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return image
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rotated


def _denoise(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def _preprocess_page(image: np.ndarray) -> Tuple[np.ndarray, float]:
    gray = _to_grayscale(image)
    thresholded = _threshold(gray)
    angle = _estimate_skew_angle(thresholded)
    deskewed = _rotate_image(gray, -angle)
    binary = _threshold(deskewed)
    denoised = _denoise(binary)
    return denoised, angle


def _extract_text_layer(pdf_path: Path) -> List[str]:
    doc = fitz.open(pdf_path)
    text_pages: List[str] = []
    for page in doc:
        text_pages.append(page.get_text("text"))
    return text_pages


def _group_ocr_lines(data: dict) -> Tuple[List[str], List[float]]:
    lines: List[str] = []
    confidences: List[float] = []
    current_id: Tuple[int, int, int] | None = None
    words: List[str] = []
    word_confidences: List[float] = []

    for idx, level in enumerate(data["level"]):
        if level != 5:
            continue
        line_id = (data["block_num"][idx], data["par_num"][idx], data["line_num"][idx])
        if current_id != line_id:
            if words:
                line_text = " ".join(words).strip()
                if line_text:
                    mean_conf = float(np.mean(word_confidences)) if word_confidences else -1.0
                    lines.append(line_text)
                    confidences.append(mean_conf)
            current_id = line_id
            words = []
            word_confidences = []
        text = data["text"][idx].strip()
        conf = float(data["conf"][idx])
        if text:
            words.append(text)
            if conf >= 0:
                word_confidences.append(conf)

    if words:
        line_text = " ".join(words).strip()
        if line_text:
            mean_conf = float(np.mean(word_confidences)) if word_confidences else -1.0
            lines.append(line_text)
            confidences.append(mean_conf)

    return lines, confidences


_SPACE_RE = re.compile(r"\s+")


def _merge_text(
    text_layer: str,
    ocr_lines: Sequence[str],
    ocr_confidences: Sequence[float],
    min_confidence: float,
) -> Tuple[str, List[OCRSupplement]]:
    normalized_text = _SPACE_RE.sub(" ", text_layer.lower()).strip()
    supplements: List[OCRSupplement] = []
    for line, confidence in zip(ocr_lines, ocr_confidences):
        clean = line.strip()
        if not clean:
            continue
        if confidence < min_confidence:
            continue
        normalized_line = _SPACE_RE.sub(" ", clean.lower())
        if not normalized_line:
            continue
        if normalized_line in normalized_text:
            continue
        supplements.append(OCRSupplement(text=clean, confidence=confidence))

    if not supplements:
        return text_layer, []

    merged = text_layer.rstrip()
    if merged:
        merged += "\n"
    merged += "\n".join(f"[OCR {supp.confidence:.1f}] {supp.text}" for supp in supplements)
    return merged, supplements


def extract_pdf_text_with_ocr(pdf_path: Path, settings: OCRSettings | None = None) -> OCRExtraction:
    """Run the dual-path OCR pipeline on ``pdf_path``.

    Parameters
    ----------
    pdf_path:
        Path to the PDF document.
    settings:
        Optional :class:`OCRSettings` instance to override defaults.
    """

    if settings is None:
        settings = OCRSettings()

    text_pages = _extract_text_layer(pdf_path)
    images = _rasterize_pdf(pdf_path, settings.dpi)

    config = f"--psm {settings.psm} --oem {settings.oem}"

    results: List[PageOCRResult] = []
    for index, (image, text_layer) in enumerate(zip(images, text_pages), start=1):
        preprocessed, estimated_angle = _preprocess_page(image)
        ocr_gray = cv2.bitwise_not(preprocessed)
        ocr_input = cv2.cvtColor(ocr_gray, cv2.COLOR_GRAY2RGB)

        ocr_text = pytesseract.image_to_string(
            ocr_input,
            lang=settings.language,
            config=config,
        )
        data = pytesseract.image_to_data(
            ocr_input,
            lang=settings.language,
            config=config,
            output_type=Output.DICT,
        )
        lines, line_confidences = _group_ocr_lines(data)
        merged_text, supplements = _merge_text(
            text_layer=text_layer,
            ocr_lines=lines,
            ocr_confidences=line_confidences,
            min_confidence=settings.min_confidence,
        )
        confidences = [float(conf) for conf in data["conf"] if conf >= 0]
        avg_confidence = float(np.mean(confidences)) if confidences else -1.0

        results.append(
            PageOCRResult(
                page_number=index,
                text_layer=text_layer,
                ocr_text=ocr_text,
                merged_text=merged_text,
                deskew_angle=estimated_angle,
                average_confidence=avg_confidence,
                supplements=supplements,
            )
        )

    return OCRExtraction(pages=results)


__all__ = [
    "OCRExtraction",
    "OCRSettings",
    "OCRSupplement",
    "PageOCRResult",
    "extract_pdf_text_with_ocr",
]
