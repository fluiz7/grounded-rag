"""The multi-agent pipeline, wired with LangGraph.

    Retriever  ->  Answerer  ->  Critic  --grounded-->  END
        ^                            |
        |________ not grounded ______|  (Critic reformulates the query and we retry)

The Critic is what makes this "agentic" rather than a plain RAG chain: it inspects
whether the answer is actually supported by the retrieved context and, if not,
proposes a sharper search query so the Retriever can try again.
"""
from typing import List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from .config import settings
from .llm import get_chat, get_vectorstore


class RAGState(TypedDict, total=False):
    question: str          # the user's original question (never mutated)
    query: str             # current search query (Critic may rewrite this)
    documents: List[Document]
    answer: str
    grounded: bool
    critique: str
    attempts: int          # number of retrieval rounds so far


def _format_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {d.metadata.get('source', '?')}, "
        f"page {d.metadata.get('page', '?')})\n{d.page_content}"
        for i, d in enumerate(docs)
    )


# --- Node 1: Retriever ------------------------------------------------------
def retrieve(state: RAGState) -> RAGState:
    query = state.get("query") or state["question"]
    docs = get_vectorstore().similarity_search(query, k=settings.top_k)
    return {"documents": docs, "attempts": state.get("attempts", 0) + 1}


# --- Node 2: Answerer -------------------------------------------------------
ANSWER_PROMPT = """You are a careful research assistant. Answer the QUESTION using \
ONLY the CONTEXT below. Cite every claim inline with the bracket numbers, e.g. [1], [2]. \
If the context does not contain the answer, reply exactly: \
"I could not find this in the provided documents."

QUESTION:
{question}

CONTEXT:
{context}

Answer (with [n] citations):"""


def answer(state: RAGState) -> RAGState:
    context = _format_context(state["documents"])
    prompt = ANSWER_PROMPT.format(question=state["question"], context=context)
    return {"answer": get_chat().invoke(prompt).content.strip()}


# --- Node 3: Critic ---------------------------------------------------------
CRITIC_PROMPT = """You are a strict verifier. Decide whether the ANSWER is fully \
supported by the CONTEXT (no unsupported claims). Reply in EXACTLY this format:
GROUNDED: yes OR no
QUERY: <if GROUNDED is no, a better search query to find the missing evidence; otherwise write -->

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}"""


def critic(state: RAGState) -> RAGState:
    context = _format_context(state["documents"])
    out = get_chat().invoke(
        CRITIC_PROMPT.format(question=state["question"], context=context, answer=state["answer"])
    ).content

    grounded = "grounded: yes" in out.lower()
    new_query = state["question"]
    for line in out.splitlines():
        if line.lower().startswith("query:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate and candidate not in ("--", "-"):
                new_query = candidate
    return {"grounded": grounded, "critique": out.strip(), "query": new_query}


# --- Routing ----------------------------------------------------------------
def route_after_critic(state: RAGState) -> str:
    """Stop if grounded or if we've exhausted the retry budget; otherwise retry."""
    if state.get("grounded") or state.get("attempts", 0) >= settings.max_retries + 1:
        return END
    return "retrieve"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("critic", critic)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"retrieve": "retrieve", END: END})
    return g.compile()
