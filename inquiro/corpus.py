from collections import defaultdict
from collections.abc import Generator
from pathlib import Path

import pypdf
from chromadb.api.shared_system_client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from inquiro.config import CHROMA_DIR, PAPERS_DIR
from inquiro.resources import build_embeddings

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 300

# Written inside CHROMA_DIR once a corpus is fully ingested. Holds "topic\ncount"
# so the UI can tell setup is done and display what was indexed. Lives inside
# CHROMA_DIR so deleting the index also clears the ready flag.
READY_SENTINEL_NAME = ".ready"


def clear_chroma_cache() -> None:
    """Drop chromadb's process-global client registry.

    chromadb caches one client per persist path for the lifetime of the
    process. Under Streamlit (one long-lived process across reruns) a rebuild
    can otherwise reuse a connection opened on a database file that was since
    deleted or recreated, which surfaces as SQLITE_READONLY_DBMOVED (code 1032).
    Clearing the cache forces the next client to open the current file fresh.
    """
    try:
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def corpus_ready(chroma_dir: Path = CHROMA_DIR) -> bool:
    """True when a usable corpus exists: the ready sentinel, or a CLI-built DB."""
    if (chroma_dir / READY_SENTINEL_NAME).exists():
        return True
    # Fallback for corpora built via scripts/ingest.py before this flag existed:
    # a real index has a populated sqlite plus a collection (uuid) subdirectory.
    sqlite = chroma_dir / "chroma.sqlite3"
    has_collection = any(p.is_dir() for p in chroma_dir.glob("*-*-*-*-*"))
    return sqlite.exists() and sqlite.stat().st_size > 100_000 and has_collection


def read_corpus_info(chroma_dir: Path = CHROMA_DIR) -> tuple[str, str] | None:
    """Return (topic, paper_count) from the sentinel, or None if absent/empty."""
    sentinel = chroma_dir / READY_SENTINEL_NAME
    if not sentinel.exists():
        return None
    parts = sentinel.read_text().strip().split("\n")
    topic = parts[0] if parts else ""
    count = parts[1] if len(parts) > 1 else ""
    return topic, count


def mark_corpus_ready(topic: str, paper_count: int, chroma_dir: Path = CHROMA_DIR) -> None:
    """Record that ingestion finished so future launches skip setup."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    (chroma_dir / READY_SENTINEL_NAME).write_text(f"{topic}\n{paper_count}")


def extract_pdf_text(pdf_path: Path) -> str:
    text_parts: list[str] = []
    with pdf_path.open("rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


def load_papers(papers_dir: Path = PAPERS_DIR) -> list[Document]:
    papers: list[Document] = []
    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        text = extract_pdf_text(pdf_path)
        if text.strip():
            papers.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_path.name},
                )
            )
    return papers


def split_papers(
    papers: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(papers)


def ingest_papers(
    papers_dir: Path = PAPERS_DIR,
    chroma_dir: Path = CHROMA_DIR,
) -> tuple[int, int]:
    papers = load_papers(papers_dir)
    chunks = split_papers(papers)
    if not chunks:
        return 0, len(papers)

    clear_chroma_cache()
    Chroma.from_documents(
        documents=chunks,
        embedding=build_embeddings(),
        persist_directory=str(chroma_dir),
    )
    mark_corpus_ready("structured pruning language models", len(papers), chroma_dir)
    return len(chunks), len(papers)


def iter_load_papers(
    papers_dir: Path = PAPERS_DIR,
) -> Generator[tuple[int, int, str, Document | None], None, None]:
    """Yield (index, total, filename, Document|None) for each PDF in papers_dir."""
    pdf_paths = sorted(papers_dir.glob("*.pdf"))
    total = len(pdf_paths)
    for i, pdf_path in enumerate(pdf_paths, start=1):
        text = extract_pdf_text(pdf_path)
        doc = (
            Document(page_content=text, metadata={"source": pdf_path.name})
            if text.strip()
            else None
        )
        yield i, total, pdf_path.name, doc


def iter_ingest_papers(
    papers: list[Document],
    chroma_dir: Path = CHROMA_DIR,
) -> Generator[tuple[int, int, str], None, None]:
    """Chunk, embed, and store papers in ChromaDB one paper at a time.

    Yields (index, total, source_name) after each paper is indexed so the
    caller can drive a live progress bar. Embedding model is loaded once and
    reused across all papers.
    """
    chunks = split_papers(papers)

    by_paper: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        by_paper[chunk.metadata.get("source", "unknown")].append(chunk)

    paper_items = list(by_paper.items())
    total = len(paper_items)

    # Start from a clean client registry so we never write through a stale,
    # since-deleted database handle left over from an earlier rerun.
    clear_chroma_cache()
    embeddings = build_embeddings()
    vectorstore: Chroma | None = None

    for i, (source, paper_chunks) in enumerate(paper_items, start=1):
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=paper_chunks,
                embedding=embeddings,
                persist_directory=str(chroma_dir),
            )
        else:
            vectorstore.add_documents(paper_chunks)
        yield i, total, source
