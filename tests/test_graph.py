"""Unit tests for the multi-agent pipeline (no Ollama required)."""
from langchain_core.documents import Document
from langgraph.graph import END

from src.graph import _format_context, build_graph, critic, route_after_critic


# --- pure helpers -----------------------------------------------------------

def test_format_context_numbers_and_cites_sources():
    docs = [
        Document(page_content="alpha", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="beta", metadata={"source": "b.pdf", "page": 2}),
    ]
    ctx = _format_context(docs)
    assert "[1] (source: a.pdf, page 1)\nalpha" in ctx
    assert "[2] (source: b.pdf, page 2)\nbeta" in ctx


def test_route_stops_when_grounded():
    assert route_after_critic({"grounded": True, "attempts": 1}) == END


def test_route_retries_when_not_grounded_and_budget_left():
    assert route_after_critic({"grounded": False, "attempts": 1}) == "retrieve"


def test_route_stops_when_budget_exhausted():
    # settings.max_retries defaults to 2 -> budget is 3 total attempts
    assert route_after_critic({"grounded": False, "attempts": 3}) == END


# --- critic parsing ---------------------------------------------------------

def test_critic_parses_grounded_yes(patch_pipeline, fake_store):
    patch_pipeline(["GROUNDED: yes\nQUERY: --"])
    out = critic({"question": "q?", "documents": fake_store.docs, "answer": "a [1]"})
    assert out["grounded"] is True
    assert out["query"] == "q?"  # unchanged when grounded


def test_critic_parses_rewritten_query(patch_pipeline, fake_store):
    patch_pipeline(["GROUNDED: no\nQUERY: latency vs signal level"])
    out = critic({"question": "q?", "documents": fake_store.docs, "answer": "a"})
    assert out["grounded"] is False
    assert out["query"] == "latency vs signal level"


# --- full pipeline ----------------------------------------------------------

def test_pipeline_happy_path_single_round(patch_pipeline, fake_store):
    patch_pipeline([
        "Latency is driven by signal quality [1].",   # answerer
        "GROUNDED: yes\nQUERY: --",                   # critic approves
    ])
    state = build_graph().invoke({"question": "what drives latency?"})
    assert state["grounded"] is True
    assert state["attempts"] == 1
    assert "[1]" in state["answer"]
    assert fake_store.queries == ["what drives latency?"]


def test_pipeline_retries_with_rewritten_query(patch_pipeline, fake_store):
    patch_pipeline([
        "Unsupported claim.",                          # answer, round 1
        "GROUNDED: no\nQUERY: signal level impact",    # critic rejects, rewrites
        "Signal quality dominates [1].",               # answer, round 2
        "GROUNDED: yes\nQUERY: --",                    # critic approves
    ])
    state = build_graph().invoke({"question": "what drives latency?"})
    assert state["grounded"] is True
    assert state["attempts"] == 2
    # second retrieval must have used the critic's rewritten query
    assert fake_store.queries == ["what drives latency?", "signal level impact"]


def test_pipeline_gives_up_after_budget(patch_pipeline, fake_store):
    patch_pipeline([
        "bad answer 1", "GROUNDED: no\nQUERY: try 2",
        "bad answer 2", "GROUNDED: no\nQUERY: try 3",
        "bad answer 3", "GROUNDED: no\nQUERY: try 4",
    ])
    state = build_graph().invoke({"question": "q?"})
    assert state["grounded"] is False
    assert state["attempts"] == 3  # 1 initial + max_retries(2)
