"""
Knowledge ingestion system for the Agent Builder.

Provides a pluggable loader system that normalizes diverse knowledge sources
(directories, PDFs, webpages) into a lightweight index (structural metadata)
plus raw content for on-demand reading.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse, urljoin

from autocomp.common import logger


@dataclass
class SourceIndex:
    """Output of a loader: structural metadata + content store."""
    source_type: str
    source_id: str
    structural_metadata: str  # Human-readable index (file tree, TOC, etc.)
    content: dict[str, str] = field(default_factory=dict)  # section_id -> text


class SourceLoader(ABC):
    """Base class for knowledge source loaders."""

    @abstractmethod
    def load(self, **kwargs) -> SourceIndex:
        """Load a source and return its SourceIndex."""
        ...


_BINARY_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".tox", ".eggs",
    "build", "dist", ".mypy_cache", ".pytest_cache",
}

_MAX_FILE_SIZE = 512 * 1024  # 512 KB per file


def _is_text_file(path: Path) -> bool:
    """Content-based text detection: reject files containing null bytes (same heuristic as git)."""
    try:
        chunk = path.read_bytes()[:8192]
        return b'\x00' not in chunk
    except OSError:
        return False


def _build_file_tree(root: Path, max_depth: int = 6) -> tuple[str, dict[str, str], list[Path]]:
    """Walk a directory and return (tree_string, content_dict, pdf_paths)."""
    tree_lines: list[str] = []
    content: dict[str, str] = {}
    pdfs: list[Path] = []

    def _walk(directory: Path, prefix: str, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            logger.warning("Skipping directory %s: permission denied", directory)
            return

        dirs = [e for e in entries if e.is_dir() and e.name not in _BINARY_SKIP_DIRS]
        files = [e for e in entries if e.is_file()]

        for d in dirs:
            tree_lines.append(f"{prefix}{d.name}/")
            _walk(d, prefix + "  ", depth + 1)

        for f in files:
            if f.suffix.lower() == ".pdf":
                size = f.stat().st_size
                tree_lines.append(f"{prefix}{f.name} ({size} bytes) [PDF]")
                pdfs.append(f)
            elif _is_text_file(f):
                size = f.stat().st_size
                tree_lines.append(f"{prefix}{f.name} ({size} bytes)")
                if size <= _MAX_FILE_SIZE:
                    rel = str(f.relative_to(root))
                    try:
                        content[rel] = f.read_text(errors="replace")
                    except Exception as e:
                        logger.warning("Failed to read %s: %s", rel, e)
                else:
                    logger.warning(
                        "Skipping %s (%s bytes) -- exceeds %s byte file-size limit",
                        f.relative_to(root), size, _MAX_FILE_SIZE,
                    )

    _walk(root, "", 0)
    tree_str = "\n".join(tree_lines)
    return tree_str, content, pdfs


class DirectoryLoader(SourceLoader):
    """Loads a local directory: file tree + text file contents."""

    def load(self, *, path: str, **kwargs) -> SourceIndex:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")

        logger.info("DirectoryLoader: indexing %s", root)
        tree_str, content, pdfs = _build_file_tree(root)
        total_chars = sum(len(v) for v in content.values())
        logger.info("DirectoryLoader: found %d text files (%d chars), %d PDFs in %s",
                     len(content), total_chars, len(pdfs), root)

        if pdfs:
            pdf_loader = PDFLoader()
            for pdf_path in pdfs:
                try:
                    pdf_index = pdf_loader.load(path=str(pdf_path))
                    rel = str(pdf_path.relative_to(root))
                    for key, text in pdf_index.content.items():
                        content[f"{rel}:{key}"] = text
                    logger.info("DirectoryLoader: extracted %d chunks from %s",
                                len(pdf_index.content), pdf_path.name)
                except Exception as e:
                    logger.warning("DirectoryLoader: failed to read PDF %s: %s",
                                   pdf_path.name, e)

        return SourceIndex(
            source_type="directory",
            source_id=str(root),
            structural_metadata=f"Directory: {root}\n\n{tree_str}",
            content=content,
        )


class FileLoader(SourceLoader):
    """Loads a single file: delegates PDFs to PDFLoader, reads text files directly."""

    def load(self, *, path: str, **kwargs) -> SourceIndex:
        file_path = Path(path).expanduser().resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() == ".pdf":
            return PDFLoader().load(path=str(file_path))

        if not _is_text_file(file_path):
            raise ValueError(f"File does not appear to be text: {file_path}")

        logger.info("FileLoader: reading %s", file_path)
        text = file_path.read_text(errors="replace")
        if len(text) > _MAX_FILE_SIZE:
            logger.warning("FileLoader: %s is large (%d bytes) -- processing may be slow",
                           file_path.name, len(text))
        return SourceIndex(
            source_type="file",
            source_id=str(file_path),
            structural_metadata=f"File: {file_path.name} ({len(text)} chars)",
            content={file_path.name: text},
        )


_PDF_CHUNK_MAX_CHARS = 20_000     # split a chunk if it grows past this char budget
                                  # (keep <= synthesizer._ROUTE_PREVIEW_CHARS so the
                                  # routing LLM sees the full chunk when classifying)
_PDF_CHUNK_MIN_CHARS = 2_000      # merge tiny adjacent sections up to at least this size


def _chunk_markdown(md: str) -> list[tuple[str, str]]:
    """Chunk a markdown string on heading boundaries.

    Splits on the shallowest heading level present (``#`` or ``##``), merges
    tiny adjacent sections up to ``_PDF_CHUNK_MIN_CHARS``, and hard-splits
    any resulting section that exceeds ``_PDF_CHUNK_MAX_CHARS`` (on the
    next heading level, then on blank-line paragraph boundaries).

    Returns a list of ``(key, text)`` tuples keyed by heading title (or
    ``section_N`` when no heading precedes the text).
    """

    def _sections_by_level(text: str, level: int) -> list[tuple[str | None, str]]:
        """Split ``text`` on ``#{level} `` headings. Returns (title, body) pairs."""
        pattern = re.compile(rf"(?m)^(#{{{level}}} .+)$")
        out: list[tuple[str | None, str]] = []
        matches = list(pattern.finditer(text))
        if not matches:
            return [(None, text)]
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                out.append((None, preamble))
        for i, m in enumerate(matches):
            title = m.group(1).lstrip("#").strip().strip("*").strip()
            body_start = m.start()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].rstrip()
            out.append((title, body))
        return out

    def _hard_split(text: str, title: str | None) -> list[tuple[str, str]]:
        """Split ``text`` into chunks <= _PDF_CHUNK_MAX_CHARS on paragraph boundaries."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[tuple[str, str]] = []
        buf: list[str] = []
        buf_len = 0
        part_idx = 1
        base = title or "section"
        for p in paragraphs:
            if buf and buf_len + len(p) > _PDF_CHUNK_MAX_CHARS:
                chunks.append((f"{base} (part {part_idx})", "\n\n".join(buf)))
                buf = []
                buf_len = 0
                part_idx += 1
            buf.append(p)
            buf_len += len(p) + 2
        if buf:
            suffix = f" (part {part_idx})" if part_idx > 1 else ""
            chunks.append((f"{base}{suffix}", "\n\n".join(buf)))
        return chunks

    # Pick the shallowest heading level actually used in the document.
    split_level = None
    for lvl in (1, 2, 3):
        if re.search(rf"(?m)^#{{{lvl}}} ", md):
            split_level = lvl
            break

    if split_level is None:
        raw_sections: list[tuple[str | None, str]] = [(None, md)]
    else:
        raw_sections = _sections_by_level(md, split_level)

    # Merge tiny adjacent sections.
    merged: list[tuple[str | None, str]] = []
    for title, body in raw_sections:
        body = body.strip()
        if not body:
            continue
        if merged and len(body) < _PDF_CHUNK_MIN_CHARS:
            prev_title, prev_body = merged[-1]
            merged[-1] = (prev_title, f"{prev_body}\n\n{body}")
        else:
            merged.append((title, body))

    # Split oversized sections: try next heading level first, then paragraphs.
    chunks: list[tuple[str, str]] = []
    anon_idx = 1
    for title, body in merged:
        if len(body) <= _PDF_CHUNK_MAX_CHARS:
            key = title if title else f"section_{anon_idx}"
            if not title:
                anon_idx += 1
            chunks.append((key, body))
            continue

        # Try splitting on the next heading level down.
        sub_level = (split_level or 0) + 1 if split_level else 2
        subs = _sections_by_level(body, sub_level) if sub_level <= 6 else [(None, body)]
        if len(subs) > 1:
            for sub_title, sub_body in subs:
                sub_body = sub_body.strip()
                if not sub_body:
                    continue
                sub_key = sub_title or title or f"section_{anon_idx}"
                if not sub_title and not title:
                    anon_idx += 1
                if len(sub_body) <= _PDF_CHUNK_MAX_CHARS:
                    chunks.append((sub_key, sub_body))
                else:
                    chunks.extend(_hard_split(sub_body, sub_key))
        else:
            chunks.extend(_hard_split(body, title))

    # De-duplicate keys by appending a counter when needed.
    seen: dict[str, int] = {}
    deduped: list[tuple[str, str]] = []
    for key, text in chunks:
        if key in seen:
            seen[key] += 1
            deduped.append((f"{key} ({seen[key]})", text))
        else:
            seen[key] = 1
            deduped.append((key, text))
    return deduped


class PDFLoader(SourceLoader):
    """Extracts PDFs as markdown via pymupdf4llm, then chunks on heading boundaries.

    pymupdf4llm reconstructs document structure (headings, paragraphs, tables)
    from the PDF, which lets us chunk along true semantic boundaries rather
    than page breaks. Content that spans pages is no longer fragmented.
    """

    def load(self, *, path: str, **kwargs) -> SourceIndex:
        try:
            import pymupdf4llm
        except ImportError:
            raise ImportError(
                "pymupdf4llm is required for PDF loading: pip install pymupdf4llm"
            )

        pdf_path = Path(path).expanduser().resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("PDFLoader: reading %s", pdf_path)
        # use_ocr=False skips the layout/OCR pipeline (which requires tesseract)
        # and uses pymupdf4llm's text-based markdown conversion instead.
        md = pymupdf4llm.to_markdown(str(pdf_path), show_progress=False, use_ocr=False)

        chunks = _chunk_markdown(md)
        content: dict[str, str] = dict(chunks)

        toc_lines = [
            key for key, _ in chunks if not key.startswith("section_")
        ]
        metadata_parts = [
            f"PDF: {pdf_path.name} ({len(md):,} chars, {len(content)} chunks)"
        ]
        if toc_lines:
            metadata_parts.append(
                "\nSections:\n" + "\n".join(f"  {t}" for t in toc_lines[:100])
            )

        return SourceIndex(
            source_type="pdf",
            source_id=str(pdf_path),
            structural_metadata="\n".join(metadata_parts),
            content=content,
        )


def _extract_text(soup) -> str:
    """Convert HTML soup to clean markdown via html2text.

    Uses ignore_emphasis and ignore_links to avoid escaping artifacts
    from Sphinx-generated API docs (e.g. \\* in signatures, [[source]] links).
    """
    import html2text

    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    h2t = html2text.HTML2Text()
    h2t.ignore_links = True
    h2t.ignore_images = True
    h2t.ignore_emphasis = True
    h2t.body_width = 0

    md = h2t.handle(str(soup))

    # Strip Sphinx [source]# permalink suffixes on function signatures
    md = re.sub(r'\[source\]\s*#', '', md)
    # Strip trailing # permalink anchors on headings (e.g. "## Heading#")
    md = re.sub(r'(?m)^(#+\s+.+?)#\s*$', r'\1', md)

    # Collapse runs of 3+ blank lines to 2
    md = re.sub(r'\n{3,}', '\n\n', md)

    return md.strip()


class WebpageLoader(SourceLoader):
    """Fetches webpages, extracts text, optionally follows same-domain links.

    URLs sharing the starting URL's path prefix are crawled first (priority
    queue), so the crawler exhausts the relevant subtree before venturing into
    other same-domain pages.
    """

    def load(self, *, url: str, max_depth: int = 1, max_pages: int = 50,
             **kwargs) -> SourceIndex:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("requests and beautifulsoup4 are required: pip install requests beautifulsoup4")

        logger.info("WebpageLoader: fetching %s (depth=%d, max_pages=%d)", url, max_depth, max_pages)
        parsed_base = urlparse(url)
        base_domain = parsed_base.netloc
        # Derive prefix from the starting URL's directory path
        base_path = parsed_base.path.rsplit("/", 1)[0] + "/"

        visited: set[str] = set()
        content: dict[str, str] = {}
        headings_by_url: dict[str, list[str]] = {}
        queue: list[tuple[str, int]] = [(url, 0)]

        while queue and len(visited) < max_pages:
            current_url, depth = queue.pop(0)
            if current_url in visited:
                continue
            visited.add(current_url)
            if len(visited) % 10 == 0 or len(visited) == 1:
                logger.info("WebpageLoader: fetching page %d/%d (depth=%d) %s",
                            len(visited), max_pages, depth, current_url)
            else:
                logger.debug("WebpageLoader: [%d/%d depth=%d] %s",
                             len(visited), max_pages, depth, current_url)

            try:
                resp = requests.get(current_url, timeout=15, headers={"User-Agent": "autocomp-agent-builder/1.0"})
                resp.raise_for_status()
            except Exception as e:
                logger.warning("WebpageLoader: failed to fetch %s: %s", current_url, e)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Discover links BEFORE stripping nav (sidebar links are
            # the primary navigation on Sphinx/ReadTheDocs sites).
            if depth < max_depth:
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    abs_url = urljoin(current_url, href)
                    parsed = urlparse(abs_url)
                    abs_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    if parsed.netloc != base_domain or abs_url in visited:
                        continue
                    if not parsed.path.startswith(base_path):
                        continue
                    queue.append((abs_url, depth + 1))

            # Remove script/style/nav elements for text extraction
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = _extract_text(soup)
            if text:
                content[current_url] = text

            # Extract headings for the index
            headings = []
            for h in soup.find_all(re.compile(r"^h[1-6]$")):
                level = int(h.name[1])
                indent = "  " * (level - 1)
                headings.append(f"{indent}{h.get_text(strip=True)}")
            headings_by_url[current_url] = headings

        logger.info("WebpageLoader: crawled %d pages (content from %d, %d empty/failed) for %s",
                     len(visited), len(content), len(visited) - len(content), url)
        # Build structural metadata
        meta_lines = [f"Webpage: {url} ({len(content)} pages fetched)"]
        for page_url, headings in headings_by_url.items():
            meta_lines.append(f"\n  {page_url}")
            for h in headings[:20]:
                meta_lines.append(f"    {h}")

        return SourceIndex(
            source_type="webpage",
            source_id=url,
            structural_metadata="\n".join(meta_lines),
            content=content,
        )


# ---------------------------------------------------------------------------
# KnowledgeIngestor: registry + orchestration
# ---------------------------------------------------------------------------

_LOADER_REGISTRY: dict[str, type[SourceLoader]] = {
    "directory": DirectoryLoader,
    "file": FileLoader,
    "webpage": WebpageLoader,
}


class KnowledgeIngestor:
    """
    Ingests multiple knowledge sources into SourceIndex objects.

    Usage:
        ingestor = KnowledgeIngestor()
        ingestor.add_source("directory", path="/path/to/docs")
        ingestor.add_source("pdf", path="/path/to/manual.pdf")
        indices = ingestor.ingest()
    """

    def __init__(self):
        self._sources: list[tuple[str, dict]] = []

    @staticmethod
    def register_loader(source_type: str, loader_cls: type[SourceLoader]):
        """Register a custom loader for a new source type."""
        _LOADER_REGISTRY[source_type] = loader_cls

    def add_source(self, source_type: str, **kwargs):
        """Add a knowledge source to be ingested."""
        if source_type not in _LOADER_REGISTRY:
            raise ValueError(
                f"Unknown source type '{source_type}'. "
                f"Available: {list(_LOADER_REGISTRY.keys())}"
            )
        self._sources.append((source_type, kwargs))

    def ingest(self) -> list[SourceIndex]:
        """Ingest all added sources and return their indices."""
        import time
        t0 = time.time()
        indices = []
        for source_type, kwargs in self._sources:
            loader = _LOADER_REGISTRY[source_type]()
            try:
                index = loader.load(**kwargs)
                indices.append(index)
                total_chars = sum(len(v) for v in index.content.values())
                logger.info(
                    "Ingested %s source '%s': %d content sections, %d chars",
                    source_type, index.source_id, len(index.content), total_chars,
                )
            except Exception as e:
                logger.error("Failed to ingest %s source: %s", source_type, e)
                raise
        elapsed = time.time() - t0
        total_sections = sum(len(idx.content) for idx in indices)
        total_chars = sum(sum(len(v) for v in idx.content.values()) for idx in indices)
        logger.info("Ingestion complete: %d sources, %d sections, %d chars in %.1fs",
                     len(indices), total_sections, total_chars, elapsed)
        return indices
