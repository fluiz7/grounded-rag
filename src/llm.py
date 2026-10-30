"""Thin factory around the local Ollama models and the Chroma vector store.
Kept in one place so swapping the backend (e.g. to a remote LLM) is a one-file change."""
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from .config import settings


def get_chat(temperature: float = 0.0, json_mode: bool = False) -> ChatOllama:
    """Chat model. temperature=0 keeps the critic and answerer deterministic.

    json_mode=True is used by the Critic: Ollama constrains decoding to valid JSON and
    the verdict uses `settings.critic_model` — set RAG_CRITIC_MODEL to a different model
    than the answerer to avoid a model grading its own output.
    """
    # reasoning=False: "thinking" models (qwen3.x, deepseek-r1...) otherwise spend the whole
    # token budget on hidden reasoning and return EMPTY content — a silent empty verdict.
    if json_mode:
        return ChatOllama(
            model=settings.critic_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            format="json",
            reasoning=False,
        )
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        reasoning=False,
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
