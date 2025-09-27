"""Document type sniffing and lightweight PDF layout features."""
from __future__ import annotations

from dataclasses import dataclass, field
import mimetypes
from pathlib import Path
from typing import Literal, Optional

try:  # pragma: no cover - import guarded for environments without PyMuPDF
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore[assignment]


BBox = tuple[float, float, float, float]
DocumentKind = Literal["pdf", "image", "spreadsheet", "text", "unknown"]


@dataclass(slots=True)
class PageFeature:
    """Lightweight statistics computed per PDF page."""

    page_number: int
    word_count: int
    glyph_count: int
    text_area: float
    page_area: float
    image_area: float

    @property
    def text_density(self) -> float:
        """Density of text coverage on the page (0.0 – 1.0)."""

        if self.page_area <= 0:
            return 0.0
        return min(1.0, self.text_area / self.page_area)

    @property
    def image_fraction(self) -> float:
        """Fraction of the page covered by raster images (0.0 – 1.0)."""

        if self.page_area <= 0:
            return 0.0
        return min(1.0, self.image_area / self.page_area)


@dataclass(slots=True)
class SniffResult:
    """Result from sniffing a document path."""

    path: Path
    mime_type: Optional[str]
    kind: DocumentKind
    page_features: list[PageFeature] = field(default_factory=list)

    def is_pdf(self) -> bool:
        return self.kind == "pdf"


_PDF_SIGNATURE = b"%PDF"

_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
}

_SPREADSHEET_EXTS = {
    ".csv",
    ".tsv",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".ods",
}

_TEXT_EXTS = {".txt", ".md", ".rtf"}


def sniff_path(path: Path, *, sort_words: bool = True) -> SniffResult:
    """Identify the document type and compute coarse PDF layout features."""

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    mime_type, _ = mimetypes.guess_type(path.name)

    if _looks_like_pdf(path):
        if fitz is None:  # pragma: no cover
            raise RuntimeError("PyMuPDF is required to analyze PDF documents.")
        document = fitz.open(path)  # type: ignore[call-arg]
        page_features = [
            compute_page_features(page, sort_words=sort_words)
            for page in document
        ]
        return SniffResult(path=path, mime_type=mime_type or "application/pdf", kind="pdf", page_features=page_features)

    if suffix in _IMAGE_EXTS:
        return SniffResult(path=path, mime_type=mime_type or "image/unknown", kind="image")

    if suffix in _SPREADSHEET_EXTS:
        if suffix == ".csv":
            mime_type = mime_type or "text/csv"
        elif suffix == ".tsv":
            mime_type = mime_type or "text/tab-separated-values"
        else:
            mime_type = (
                mime_type
                or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        return SniffResult(path=path, mime_type=mime_type, kind="spreadsheet")

    if suffix in _TEXT_EXTS:
        return SniffResult(path=path, mime_type=mime_type or "text/plain", kind="text")

    if mime_type == "application/pdf":
        if fitz is None:  # pragma: no cover
            raise RuntimeError("PyMuPDF is required to analyze PDF documents.")
        document = fitz.open(path)  # type: ignore[call-arg]
        page_features = [
            compute_page_features(page, sort_words=sort_words)
            for page in document
        ]
        return SniffResult(path=path, mime_type=mime_type, kind="pdf", page_features=page_features)

    return SniffResult(path=path, mime_type=mime_type, kind="unknown")


def compute_page_features(page: "fitz.Page", *, sort_words: bool = True) -> PageFeature:
    """Compute text and image coverage statistics for a PyMuPDF page."""

    page_area = float(page.rect.width * page.rect.height)
    word_count = 0
    glyph_count = 0
    text_area = 0.0
    for word in page.get_text("words", sort=sort_words):
        x0, y0, x1, y1, text, *_ = word
        width = max(0.0, float(x1) - float(x0))
        height = max(0.0, float(y1) - float(y0))
        text_area += width * height
        word_count += 1
        glyph_count += len(text.strip())

    image_area = 0.0
    try:
        blocks = page.get_text("dict", sort=sort_words)["blocks"]
    except RuntimeError:
        blocks = []
    for block in blocks:
        if block.get("type") == 1:
            x0, y0, x1, y1 = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
            width = max(0.0, float(x1) - float(x0))
            height = max(0.0, float(y1) - float(y0))
            image_area += width * height

    return PageFeature(
        page_number=page.number + 1,
        word_count=word_count,
        glyph_count=glyph_count,
        text_area=text_area,
        page_area=page_area,
        image_area=image_area,
    )


def _looks_like_pdf(path: Path) -> bool:
    """Return True if the file extension or header suggests a PDF document."""

    if path.suffix.lower() == ".pdf":
        return True

    with path.open("rb") as handle:
        prefix = handle.read(len(_PDF_SIGNATURE))
    return prefix.startswith(_PDF_SIGNATURE)
