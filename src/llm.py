"""Thin factory around the local Ollama models and the Chroma vector store.
Kept in one place so swapping the backend (e.g. to a remote LLM) is a one-file change."""
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from .config import settings


def get_chat(temperature: float = 0.0) -> ChatOllama:
    """Chat model. temperature=0 keeps the critic and answerer deterministic."""
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)


def get_vectorstore() -> Chroma:
    """Embedded Chroma — persists under settings.persist_dir, no external server."""
    return Chroma(
        collection_name=settings.collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.persist_dir,
    )
