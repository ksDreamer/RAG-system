"""Bounded PDF/UTF-8 extraction and page-preserving, offset-aware chunking."""

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath

from pypdf import PdfReader

MAX_BYTES = 20 * 1024 * 1024
MAX_TEXT = 2_000_000
CHUNKER_VERSION = "section-v2-1200-160"
STOPWORDS = frozenset(
    "a an and are as at be by can do for from how i in is it of on or that the this to was what when where which who why will with you your".split()
)


def tokens(text: str) -> list[str]:
    # CJK characters are separated explicitly instead of relying on unicode61 word boundaries.
    return [
        t
        for t in re.findall(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", text.casefold())
        if t not in STOPWORDS
    ]


@dataclass(frozen=True)
class Chunk:
    page: int
    section: str
    start: int
    end: int
    text: str


def validate_name(name: str) -> str:
    if not name or len(name) > 200 or name != PurePosixPath(name).name or "\\" in name:
        raise ValueError("Use a filename without directory components (up to 200 characters).")
    if any(ord(c) < 32 for c in name):
        raise ValueError("Filename contains control characters.")
    if PurePosixPath(name).suffix.lower() not in {".pdf", ".md", ".txt"}:
        raise ValueError("Supported documents: PDF, Markdown and UTF-8 text.")
    return name


def extract(name: str, data: bytes) -> list[tuple[int, str]]:
    validate_name(name)
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Document must be nonempty and at most 20 MiB.")
    if name.lower().endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported; upload an unlocked copy.")
            if len(reader.pages) > 500:
                raise ValueError("PDF exceeds the 500-page limit.")
            pages = []
            total = 0
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                total += len(text)
                if total > MAX_TEXT:
                    raise ValueError("Extracted document exceeds the 2-million-character limit.")
                pages.append((number, text))
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Could not read this PDF.") from exc
    else:
        try:
            pages = [(1, data.decode("utf-8-sig"))]
        except UnicodeDecodeError as exc:
            raise ValueError("Text documents must use UTF-8 encoding.") from exc
    if sum(len(t) for _, t in pages) > MAX_TEXT:
        raise ValueError("Extracted document exceeds the 2-million-character limit.")
    if not any(t.strip() for _, t in pages):
        raise ValueError("No readable text found. Scanned PDFs need OCR before upload.")
    return [(n, t.replace("\x00", "").replace("\r\n", "\n")) for n, t in pages]


def chunk_pages(pages: list[tuple[int, str]], size: int = 1200, overlap: int = 160) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")
    chunks = []
    for page, text in pages:
        headings = list(re.finditer(r"^#{1,6}\s+(.+)$", text, re.MULTILINE))
        boundaries = sorted({0, len(text), *(m.start() for m in headings)})
        for segment_start, segment_end in zip(boundaries[:-1], boundaries[1:], strict=True):
            start = segment_start
            while start < segment_end:
                end = min(start + size, segment_end)
                if end < segment_end:
                    boundary = max(
                        text.rfind("\n\n", start + size // 2, end),
                        text.rfind(". ", start + size // 2, end),
                    )
                    if boundary > start:
                        end = boundary + 1
                left = start + len(text[start:end]) - len(text[start:end].lstrip())
                right = end - len(text[start:end]) + len(text[start:end].rstrip())
                if left < right:
                    section = next((m[1] for m in reversed(headings) if m.start() <= left), "")
                    chunks.append(Chunk(page, section, left, right, text[left:right]))
                if end == segment_end:
                    break
                start = max(start + 1, end - overlap)
    return chunks
