"""
Knowledge ingestion system for the Agent Builder.

Provides a pluggable loader system that normalizes diverse knowledge sources
(directories, GitHub repos, PDFs, webpages, Confluence pages) into a lightweight
index (structural metadata) plus raw content for on-demand reading.
"""

import os
import re
import shutil
import tempfile
import subprocess
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


# ---------------------------------------------------------------------------
# Text file extensions we consider readable
# ---------------------------------------------------------------------------
_TEXT_EXTENSIONS = {
    ".py", ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    ".java", ".js", ".ts", ".jsx", ".tsx",
    ".rs", ".go", ".rb", ".pl", ".sh", ".bash",
    ".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    ".html", ".htm", ".xml", ".csv",
    ".r", ".R", ".m", ".f", ".f90",
    ".cu", ".cuh", ".cl",  # CUDA, OpenCL
    ".nki",  # NKI
    ".makefile", ".cmake",
}

_BINARY_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".tox", ".eggs",
    "build", "dist", ".mypy_cache", ".pytest_cache",
}

_MAX_FILE_SIZE = 512 * 1024  # 512 KB per file


def _is_text_file(path: Path) -> bool:
    if path.suffix.lower() in _TEXT_EXTENSIONS:
        return True
    if path.suffix == "" and path.name in ("Makefile", "Dockerfile", "LICENSE", "README"):
        return True
    return False


def _build_file_tree(root: Path, max_depth: int = 6) -> tuple[str, dict[str, str]]:
    """Walk a directory and return (tree_string, content_dict)."""
    tree_lines: list[str] = []
    content: dict[str, str] = {}

    def _walk(directory: Path, prefix: str, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        dirs = [e for e in entries if e.is_dir() and e.name not in _BINARY_SKIP_DIRS]
        files = [e for e in entries if e.is_file() and _is_text_file(e)]

        for d in dirs:
            tree_lines.append(f"{prefix}{d.name}/")
            _walk(d, prefix + "  ", depth + 1)

        for f in files:
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
    return tree_str, content


class DirectoryLoader(SourceLoader):
    """Loads a local directory: file tree + text file contents."""

    def load(self, *, path: str, **kwargs) -> SourceIndex:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")

        logger.info("DirectoryLoader: indexing %s", root)
        tree_str, content = _build_file_tree(root)
        return SourceIndex(
            source_type="directory",
            source_id=str(root),
            structural_metadata=f"Directory: {root}\n\n{tree_str}",
            content=content,
        )


class GitHubRepoLoader(SourceLoader):
    """Shallow-clones a GitHub repo then delegates to DirectoryLoader."""

    def load(self, *, url: str, branch: str = None, **kwargs) -> SourceIndex:
        logger.info("GitHubRepoLoader: cloning %s", url)
        tmp_dir = tempfile.mkdtemp(prefix="autocomp_gh_")
        try:
            cmd = ["git", "clone", "--depth", "1"]
            if branch:
                cmd += ["--branch", branch]
            cmd += [url, tmp_dir]
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)

            loader = DirectoryLoader()
            index = loader.load(path=tmp_dir)
            index.source_type = "github"
            index.source_id = url
            index.structural_metadata = index.structural_metadata.replace(
                f"Directory: {tmp_dir}", f"GitHub repo: {url}"
            )
            return index
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


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
        doc = pymupdf.open(str(pdf_path))

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
                # Extract first non-empty line as a summary hint
                first_line = text.strip().split("\n")[0][:100]
                page_summaries.append(f"  Page {page_num + 1}: {first_line}")

        doc.close()

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


class WebpageLoader(SourceLoader):
    """Fetches webpages, extracts text, optionally follows same-domain links."""

    def load(self, *, url: str, max_depth: int = 1, max_pages: int = 50, **kwargs) -> SourceIndex:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("requests and beautifulsoup4 are required: pip install requests beautifulsoup4")

        logger.info("WebpageLoader: fetching %s (depth=%d)", url, max_depth)
        parsed_base = urlparse(url)
        base_domain = parsed_base.netloc

        visited: set[str] = set()
        content: dict[str, str] = {}
        headings_by_url: dict[str, list[str]] = {}
        queue: list[tuple[str, int]] = [(url, 0)]

        while queue and len(visited) < max_pages:
            current_url, depth = queue.pop(0)
            if current_url in visited:
                continue
            visited.add(current_url)

            try:
                resp = requests.get(current_url, timeout=15, headers={"User-Agent": "autocomp-agent-builder/1.0"})
                resp.raise_for_status()
            except Exception as e:
                logger.warning("WebpageLoader: failed to fetch %s: %s", current_url, e)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove script/style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            if text:
                content[current_url] = text

            # Extract headings for the index
            headings = []
            for h in soup.find_all(re.compile(r"^h[1-6]$")):
                level = int(h.name[1])
                indent = "  " * (level - 1)
                headings.append(f"{indent}{h.get_text(strip=True)}")
            headings_by_url[current_url] = headings

            # Follow same-domain links
            if depth < max_depth:
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    abs_url = urljoin(current_url, href)
                    parsed = urlparse(abs_url)
                    # Strip fragment
                    abs_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    if parsed.netloc == base_domain and abs_url not in visited:
                        queue.append((abs_url, depth + 1))

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


class ConfluenceLoader(SourceLoader):
    """
    Loads Confluence Cloud pages via REST API v2.

    Index phase: recursively fetches page tree (titles only, no body).
    Content phase: fetches full page body on demand when content is accessed.
    """

    def load(self, *, base_url: str, page_id: str = None, space_id: str = None,
             email: str = None, api_token: str = None, max_pages: int = 200, **kwargs) -> SourceIndex:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("requests and beautifulsoup4 are required: pip install requests beautifulsoup4")

        if not email or not api_token:
            raise ValueError("ConfluenceLoader requires email and api_token")
        if not page_id and not space_id:
            raise ValueError("ConfluenceLoader requires either page_id or space_id")

        logger.info("ConfluenceLoader: indexing %s (page_id=%s, space_id=%s)", base_url, page_id, space_id)
        auth = (email, api_token)
        api_base = f"{base_url.rstrip('/')}/wiki/api/v2"

        # Collect page tree: (id, title, parent_id, depth)
        pages: list[dict] = []

        def _fetch_children(parent_id: str, depth: int):
            if len(pages) >= max_pages:
                return
            url = f"{api_base}/pages/{parent_id}/children"
            cursor = None
            while len(pages) < max_pages:
                params = {"limit": 50}
                if cursor:
                    params["cursor"] = cursor
                resp = requests.get(url, auth=auth, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                for child in data.get("results", []):
                    pages.append({
                        "id": child["id"],
                        "title": child.get("title", "Untitled"),
                        "parent_id": parent_id,
                        "depth": depth,
                    })
                    _fetch_children(child["id"], depth + 1)
                # Pagination
                links = data.get("_links", {})
                if "next" in links:
                    cursor = links["next"].split("cursor=")[-1] if "cursor=" in links["next"] else None
                    if not cursor:
                        break
                else:
                    break

        if page_id:
            # Fetch the root page info
            resp = requests.get(f"{api_base}/pages/{page_id}", auth=auth, timeout=15)
            resp.raise_for_status()
            root = resp.json()
            pages.append({
                "id": root["id"],
                "title": root.get("title", "Untitled"),
                "parent_id": None,
                "depth": 0,
            })
            _fetch_children(page_id, 1)
        elif space_id:
            url = f"{api_base}/spaces/{space_id}/pages"
            params = {"limit": 50, "depth": "root"}
            resp = requests.get(url, auth=auth, params=params, timeout=15)
            resp.raise_for_status()
            for p in resp.json().get("results", []):
                pages.append({
                    "id": p["id"],
                    "title": p.get("title", "Untitled"),
                    "parent_id": None,
                    "depth": 0,
                })
                _fetch_children(p["id"], 1)

        # Build structural metadata (page tree)
        tree_lines = [f"Confluence: {base_url} ({len(pages)} pages)"]
        for p in pages:
            indent = "  " * p["depth"]
            tree_lines.append(f"  {indent}{p['title']} [id:{p['id']}]")

        # Build a lazy content store -- fetch bodies on demand
        # For simplicity in the two-pass pipeline, we pre-fetch all content now
        # since the synthesizer will select what to read via Pass 1
        content: dict[str, str] = {}
        for p in pages:
            try:
                resp = requests.get(
                    f"{api_base}/pages/{p['id']}",
                    auth=auth, params={"body-format": "storage"}, timeout=15,
                )
                resp.raise_for_status()
                body = resp.json().get("body", {}).get("storage", {}).get("value", "")
                if body:
                    soup = BeautifulSoup(body, "html.parser")
                    text = soup.get_text(separator="\n", strip=True)
                    if text:
                        content[f"page:{p['id']}:{p['title']}"] = text
            except Exception as e:
                logger.warning("ConfluenceLoader: failed to fetch page %s: %s", p["id"], e)

        return SourceIndex(
            source_type="confluence",
            source_id=f"{base_url}/page/{page_id or space_id}",
            structural_metadata="\n".join(tree_lines),
            content=content,
        )


# ---------------------------------------------------------------------------
# KnowledgeIngestor: registry + orchestration
# ---------------------------------------------------------------------------

_LOADER_REGISTRY: dict[str, type[SourceLoader]] = {
    "directory": DirectoryLoader,
    "github": GitHubRepoLoader,
    "pdf": PDFLoader,
    "webpage": WebpageLoader,
    "confluence": ConfluenceLoader,
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
