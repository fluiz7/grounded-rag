"""Test doubles: a scripted fake chat model and an in-memory vector store.

They let the whole LangGraph pipeline run in CI with no Ollama server.
"""
import pytest
from langchain_core.documents import Document


class FakeChat:
    """Returns pre-scripted responses in order; records the prompts it saw."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        content = self.responses.pop(0) if self.responses else ""

        class _Msg:
            pass

        msg = _Msg()
        msg.content = content
        return msg


class FakeVectorStore:
    """similarity_search returns canned documents; records queries."""

    def __init__(self, docs=None):
        self.docs = docs or [
            Document(
                page_content="Signal quality dominates latency under mobility.",
                metadata={"source": "paper.pdf", "page": 9},
            ),
            Document(
                page_content="Distance alone is a weak latency predictor (low R2).",
                metadata={"source": "paper.pdf", "page": 6},
            ),
        ]
        self.queries = []

    def similarity_search(self, query, k=5):
        self.queries.append(query)
        return self.docs[:k]


@pytest.fixture
def fake_store():
    return FakeVectorStore()


@pytest.fixture
def patch_pipeline(monkeypatch, fake_store):
    """Patch graph-level factories; returns a helper to script chat responses."""

    def _install(chat_responses):
        chat = FakeChat(chat_responses)
        monkeypatch.setattr("src.graph.get_chat", lambda *a, **k: chat)
        monkeypatch.setattr("src.graph.get_vectorstore", lambda: fake_store)
        return chat

    return _install
