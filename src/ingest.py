"""PDF ingestion: load -> chunk -> embed -> store in Chroma.

Accepts a single PDF or a folder (searched recursively).

Idempotent: every chunk gets a deterministic id derived from its file, page, position
and content, and Chroma upserts by id. Re-running `ingest` on the same folder therefore
does not duplicate chunks — duplicates would otherwise crowd the top-k retrieval slots.
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


def load_pdf(pdf: Path) -> list[Document]:
    """One Document per page. `page` is 1-based — the number a reader sees in the PDF
    viewer — so a citation like "page 9" can be checked by opening page 9."""
    reader = PdfReader(str(pdf))
    return [
        Document(page_content=page.extract_text() or "", metadata={"source": pdf.name, "page": i})
        for i, page in enumerate(reader.pages, start=1)
    ]


def ingest(path: str) -> tuple[int, int]:
    """Returns (num_pdfs, num_chunks) upserted into the vector store."""
    p = Path(path)
    pdfs = [p] if p.suffix.lower() == ".pdf" else sorted(p.glob("**/*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found at: {path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,  # makes identical text at different offsets distinct
    )

    chunks = []
    for pdf in pdfs:
        pages = [p for p in load_pdf(pdf) if p.page_content.strip()]  # skip image-only pages
        chunks.extend(splitter.split_documents(pages))

    if chunks:
        get_vectorstore().add_documents(chunks, ids=[chunk_id(c) for c in chunks])
    return len(pdfs), len(chunks)
