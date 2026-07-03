"""Central configuration. Everything is overridable via environment variables,
so the demo runs out-of-the-box but stays easy to tune."""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    # Ollama (local, no API key needed)
    chat_model: str = os.getenv("RAG_CHAT_MODEL", "llama3.1")
    embed_model: str = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Vector store (Chroma, embedded — persists to disk, no server)
    persist_dir: str = os.getenv("RAG_PERSIST_DIR", ".chroma")
    collection: str = os.getenv("RAG_COLLECTION", "grounded_rag")

    # Chunking / retrieval
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))

    # Agentic critic loop: how many extra retrieve→answer→critic cycles to allow
    max_retries: int = int(os.getenv("RAG_MAX_RETRIES", "2"))


settings = Settings()
