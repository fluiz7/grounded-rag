"""PDF ingestion: load -> chunk -> embed -> store in Chroma.

Accepts a single PDF or a folder (searched recursively)."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .llm import get_vectorstore


def ingest(path: str) -> tuple[int, int]:
    """Returns (num_pdfs, num_chunks) added to the vector store."""
    p = Path(path)
    pdfs = [p] if p.suffix.lower() == ".pdf" else sorted(p.glob("**/*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found at: {path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks = []
    for pdf in pdfs:
        pages = PyPDFLoader(str(pdf)).load()  # one Document per page, with a `page` metadata
        for page in pages:
            page.metadata["source"] = pdf.name  # normalize to just the file name
        chunks.extend(splitter.split_documents(pages))

    get_vectorstore().add_documents(chunks)
    return len(pdfs), len(chunks)
