"""PDF ingestion: load -> chunk -> embed -> store in Chroma.

Accepts a single PDF or a folder (searched recursively).

Re-ingesting is safe:
- `source` is the PDF's path relative to the folder being ingested, so two files with the
  same name in different subfolders (2024/paper.pdf, 2025/paper.pdf) stay distinct;
- each file REPLACES its previous chunks: re-running on an unchanged folder changes
  nothing, and re-running after editing a PDF leaves no stale chunks from the old version
  mixed into retrieval.
"""
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import settings
from .llm import get_vectorstore


def chunk_id(doc: Document) -> str:
    """Stable id for a chunk: same file + page + offset + text -> same id."""
    m = doc.metadata
    key = f"{m.get('source')}|{m.get('page')}|{m.get('start_index')}|{doc.page_content}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def load_pdf(pdf: Path, source: str | None = None) -> list[Document]:
    """One Document per page. `page` is 1-based — the number a reader sees in the PDF
    viewer — so a citation like "page 9" can be checked by opening page 9."""
    reader = PdfReader(str(pdf))
    src = source or pdf.name
    return [
        Document(page_content=page.extract_text() or "", metadata={"source": src, "page": i})
        for i, page in enumerate(reader.pages, start=1)
    ]


def replace_source(store, source: str, chunks: list[Document]) -> None:
    """Delete every chunk previously stored for `source`, then add `chunks`."""
    old = store.get(where={"source": source}).get("ids", [])
    if old:
        store.delete(ids=old)
    if chunks:
        ids = [chunk_id(c) for c in chunks]
        unique = dict(zip(ids, chunks))  # identical chunk twice in one file -> stored once
        store.add_documents(list(unique.values()), ids=list(unique.keys()))


def ingest(path: str) -> tuple[int, int]:
    """Returns (num_pdfs, num_chunks) stored."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        pdfs = [(p, p.name)]
    else:
        pdfs = [(f, f.relative_to(p).as_posix()) for f in sorted(p.glob("**/*.pdf"))]
    if not pdfs:
        raise SystemExit(f"No PDFs found at: {path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,  # makes identical text at different offsets distinct
    )

    store = get_vectorstore()
    total = 0
    for pdf, source in pdfs:
        pages = [pg for pg in load_pdf(pdf, source) if pg.page_content.strip()]  # skip image-only pages
        chunks = splitter.split_documents(pages)
        replace_source(store, source, chunks)
        total += len(chunks)
    return len(pdfs), total
