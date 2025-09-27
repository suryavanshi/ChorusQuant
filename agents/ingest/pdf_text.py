"""Structured PDF text extraction built on top of PyMuPDF."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from .sniff_mime import compute_page_features

try:  # pragma: no cover - import guarded for environments without PyMuPDF
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore[assignment]


BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class WordSpan:
    """A single token extracted from a PDF page."""

    text: str
    bbox: BBox
    block_index: int
    line_index: int
    word_index: int


@dataclass(slots=True)
class LineSpan:
    """A logical line containing one or more words."""

    text: str
    bbox: BBox
    block_index: int
    line_index: int
    words: List[WordSpan] = field(default_factory=list)


@dataclass(slots=True)
class BlockSpan:
    """Higher-level block (paragraph, table cell, etc.)."""

    text: str
    bbox: BBox
    block_index: int
    lines: List[LineSpan] = field(default_factory=list)


@dataclass(slots=True)
class PageExtraction:
    """Per-page representation of text, geometry, and layout features."""

    page_number: int
    text: str
    blocks: List[BlockSpan]
    word_count: int
    glyph_count: int
    text_density: float
    image_fraction: float


@dataclass(slots=True)
class PdfTextExtraction:
    """Container for a PDF's textual content."""

    pages: List[PageExtraction] = field(default_factory=list)
    scan_score: float = 0.0
    likely_scanned: bool = False


def extract_pdf_text(path: Path, *, sort: bool = True) -> PdfTextExtraction:
    """Extract structured text and layout metadata from a PDF."""

    if fitz is None:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required to extract PDF text.")

    document = fitz.open(path)  # type: ignore[call-arg]
    pages: List[PageExtraction] = []
    for page in document:
        page_feature = compute_page_features(page, sort_words=sort)
        blocks = _build_block_structure(page, sort=sort)
        page_text = "\n".join(block.text for block in blocks if block.text).strip()
        pages.append(
            PageExtraction(
                page_number=page.number + 1,
                text=page_text,
                blocks=blocks,
                word_count=page_feature.word_count,
                glyph_count=page_feature.glyph_count,
                text_density=page_feature.text_density,
                image_fraction=page_feature.image_fraction,
            )
        )

    scan_score, likely_scanned = _estimate_scan_likelihood(pages)
    return PdfTextExtraction(pages=pages, scan_score=scan_score, likely_scanned=likely_scanned)


def _build_block_structure(page: "fitz.Page", *, sort: bool = True) -> List[BlockSpan]:
    """Reconstruct the block/line/word hierarchy from PyMuPDF output."""

    words = page.get_text("words", sort=sort)
    block_map: dict[int, dict[int, List[WordSpan]]] = {}
    bbox_map: dict[tuple[int, int], List[float]] = {}

    for x0, y0, x1, y1, text, block_no, line_no, word_no in words:
        bbox = (float(x0), float(y0), float(x1), float(y1))
        block_idx = int(block_no)
        line_idx = int(line_no)
        word_idx = int(word_no)
        span = WordSpan(
            text=str(text),
            bbox=bbox,
            block_index=block_idx,
            line_index=line_idx,
            word_index=word_idx,
        )
        block_map.setdefault(block_idx, {}).setdefault(line_idx, []).append(span)
        bbox_map.setdefault((block_idx, line_idx), []).extend([bbox[0], bbox[1], bbox[2], bbox[3]])

    blocks: List[BlockSpan] = []
    for block_idx in sorted(block_map.keys()):
        line_spans: List[LineSpan] = []
        for line_idx in sorted(block_map[block_idx].keys()):
            word_spans = sorted(block_map[block_idx][line_idx], key=lambda w: w.word_index)
            line_text = " ".join(word.text for word in word_spans).strip()
            coords = bbox_map[(block_idx, line_idx)]
            x_coords = coords[0::4]
            y_coords = coords[1::4]
            x2_coords = coords[2::4]
            y2_coords = coords[3::4]
            line_bbox = (
                min(x_coords) if x_coords else 0.0,
                min(y_coords) if y_coords else 0.0,
                max(x2_coords) if x2_coords else 0.0,
                max(y2_coords) if y2_coords else 0.0,
            )
            line_spans.append(
                LineSpan(
                    text=line_text,
                    bbox=line_bbox,
                    block_index=block_idx,
                    line_index=line_idx,
                    words=word_spans,
                )
            )
        if not line_spans:
            continue
        block_bbox = _union_bbox(span.bbox for span in line_spans)
        block_text = "\n".join(line.text for line in line_spans if line.text)
        blocks.append(
            BlockSpan(
                text=block_text,
                bbox=block_bbox,
                block_index=block_idx,
                lines=line_spans,
            )
        )

    return blocks


def _union_bbox(bboxes: Iterable[BBox]) -> BBox:
    """Return the bounding box that encloses all supplied boxes."""

    xs0: List[float] = []
    ys0: List[float] = []
    xs1: List[float] = []
    ys1: List[float] = []
    for x0, y0, x1, y1 in bboxes:
        xs0.append(float(x0))
        ys0.append(float(y0))
        xs1.append(float(x1))
        ys1.append(float(y1))
    if not xs0:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _estimate_scan_likelihood(pages: List[PageExtraction]) -> tuple[float, bool]:
    """Heuristic scoring for identifying likely scanned documents."""

    if not pages:
        return 0.0, False

    avg_image_fraction = sum(page.image_fraction for page in pages) / len(pages)
    sparse_pages = [page for page in pages if page.glyph_count < 50]
    low_density_pages = [page for page in pages if page.text_density < 0.02]

    sparse_ratio = len(sparse_pages) / len(pages)
    low_density_ratio = len(low_density_pages) / len(pages)

    score = 0.5 * avg_image_fraction + 0.3 * sparse_ratio + 0.2 * low_density_ratio
    score = min(1.0, score)
    likely_scanned = score >= 0.5
    return score, likely_scanned
