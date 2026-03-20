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
                    except Exception:
                        pass

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

        if pdfs:
            pdf_loader = PDFLoader()
            for pdf_path in pdfs:
                try:
                    pdf_index = pdf_loader.load(path=str(pdf_path))
                    rel = str(pdf_path.relative_to(root))
                    for key, text in pdf_index.content.items():
                        content[f"{rel}:{key}"] = text
                    logger.info("DirectoryLoader: extracted %d pages from %s",
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
        return SourceIndex(
            source_type="file",
            source_id=str(file_path),
            structural_metadata=f"File: {file_path.name} ({len(text)} chars)",
            content={file_path.name: text},
        )


class PDFLoader(SourceLoader):
    """Extracts text from PDFs, preserving page numbers and TOC structure."""

    def load(self, *, path: str, **kwargs) -> SourceIndex:
        try:
            import pymupdf  # PyMuPDF
        except ImportError:
            raise ImportError("pymupdf is required for PDF loading: pip install pymupdf")

        pdf_path = Path(path).expanduser().resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("PDFLoader: reading %s", pdf_path)
        with pymupdf.open(str(pdf_path)) as doc:
            # Try to extract TOC
            toc = doc.get_toc()
            toc_lines = []
            if toc:
                for level, title, page_num in toc:
                    indent = "  " * (level - 1)
                    toc_lines.append(f"{indent}{title} (p.{page_num})")

            # Extract text per page
            content: dict[str, str] = {}
            page_summaries: list[str] = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    key = f"page_{page_num + 1}"
                    content[key] = text
                    first_line = text.strip().split("\n")[0][:100]
                    page_summaries.append(f"  Page {page_num + 1}: {first_line}")

        metadata_parts = [f"PDF: {pdf_path.name} ({len(content)} pages with text)"]
        if toc_lines:
            metadata_parts.append("\nTable of Contents:\n" + "\n".join(toc_lines))
        else:
            metadata_parts.append("\nPage summaries:\n" + "\n".join(page_summaries[:50]))

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
            logger.debug("WebpageLoader: [%d/%d depth=%d] %s",
                         len(visited), max_pages, depth, current_url)

            try:
                resp = requests.get(current_url, timeout=15, headers={"User-Agent": "autocomp-agent-builder/1.0"})
                resp.raise_for_status()
            except Exception as e:
                logger.warning("WebpageLoader: failed to fetch %s: %s", current_url, e)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove script/style/nav elements
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

            # Follow links under the same path prefix
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

        logger.info("WebpageLoader: crawled %d pages (content from %d) for %s",
                     len(visited), len(content), url)
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
        indices = []
        for source_type, kwargs in self._sources:
            loader = _LOADER_REGISTRY[source_type]()
            try:
                index = loader.load(**kwargs)
                indices.append(index)
                logger.info(
                    "Ingested %s source '%s': %d content sections",
                    source_type, index.source_id, len(index.content),
                )
            except Exception as e:
                logger.error("Failed to ingest %s source: %s", source_type, e)
                raise
        return indices
