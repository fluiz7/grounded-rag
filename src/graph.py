"""The multi-agent pipeline, wired with LangGraph.

    Retriever  ->  Answerer  ->  Critic  --grounded-->  Finalize -> END
        ^                            |
        |________ not grounded ______|  (Critic reformulates the query and we retry)

The Critic is what makes this "agentic" rather than a plain RAG chain: it inspects
whether the answer is actually supported by the retrieved context and, if not,
proposes a sharper search query so the Retriever can try again.

Safety rule enforced by `finalize`: an answer the Critic did not approve is never
returned as the answer. When the retry budget runs out, the pipeline answers with
the refusal and keeps the unverified text in `draft` for inspection.
"""
import json
import re
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from .config import settings
from .llm import get_chat, get_vectorstore

REFUSAL = "I could not find this in the provided documents."


class RAGState(TypedDict, total=False):
    question: str          # the user's original question (never mutated)
    query: str             # current search query (Critic may rewrite this)
    documents: list[Document]
    answer: str
    draft: str             # last unverified answer, kept when the Critic never approved
    grounded: bool
    critique: str
    unsupported: list[str]  # claims the Critic flagged as unsupported
    critic_error: bool     # the Critic produced no verdict at all (infrastructure, not evidence)
    found: bool            # False when the answer is the refusal (documents lack the answer)
    attempts: int          # number of retrieval rounds so far


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {d.metadata.get('source', '?')}, "
        f"page {d.metadata.get('page', '?')})\n{d.page_content}"
        for i, d in enumerate(docs)
    )


def _doc_key(d: Document) -> tuple:
    return (d.metadata.get("source"), d.metadata.get("page"), d.page_content)


# --- Node 1: Retriever ------------------------------------------------------
def retrieve(state: RAGState) -> RAGState:
    """Search with the current query and MERGE with evidence from earlier rounds.

    Replacing the documents on every retry would throw away evidence the first round
    already found: if round 1 retrieves half of the support and round 2 the other half,
    the answer could never be grounded on both. Fresh hits come first (they target what
    the Critic said was missing); the merged context is capped at 2 x top_k.
    """
    query = state.get("query") or state["question"]
    fresh = get_vectorstore().similarity_search(query, k=settings.top_k)

    merged, seen = [], set()
    for d in [*fresh, *state.get("documents", [])]:
        key = _doc_key(d)
        if key not in seen:
            seen.add(key)
            merged.append(d)
    return {"documents": merged[: 2 * settings.top_k], "attempts": state.get("attempts", 0) + 1}


# --- Node 2: Answerer -------------------------------------------------------
ANSWER_PROMPT = """You are a careful research assistant. Answer the QUESTION using \
ONLY the CONTEXT below, in the same language as the QUESTION. Cite every claim inline \
with the bracket numbers, e.g. [1], [2]. If the context does not contain the answer, \
reply exactly: "{refusal}"

QUESTION:
{question}

CONTEXT:
{context}

Answer (with [n] citations):"""


def answer(state: RAGState) -> RAGState:
    context = _format_context(state["documents"])
    prompt = ANSWER_PROMPT.format(question=state["question"], context=context, refusal=REFUSAL)
    return {"answer": get_chat().invoke(prompt).content.strip()}


# --- Node 3: Critic ---------------------------------------------------------
CRITIC_PROMPT = """You are a strict verifier. Decide whether EVERY claim in the ANSWER \
is supported by the CONTEXT. Partial support counts as NOT grounded.

Reply with a single JSON object and nothing else:
{{"grounded": true or false,
  "unsupported_claims": ["<claim without support>", ...],
  "query": "<if not grounded, a better search query for the missing evidence; else empty>"}}

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}"""

_CITATION = re.compile(r"\[(\d+)\]")
_MARKUP = re.compile(r"[*_`#>]")


def check_citations(answer_text: str, n_docs: int) -> str | None:
    """Deterministic pre-check. Returns a problem description, or None if citations are sane.

    Cheap and LLM-free: an answer with no citations, or citing a source number that was
    never retrieved, cannot be grounded — no need to spend a Critic call to find out.
    """
    if answer_text.strip() == REFUSAL:
        return None
    cited = [int(n) for n in _CITATION.findall(answer_text)]
    if not cited:
        return "the answer cites no sources"
    invalid = sorted({n for n in cited if not 1 <= n <= n_docs})
    if invalid:
        return f"the answer cites sources that were not retrieved: {invalid} (only {n_docs} available)"
    return None


def parse_verdict(text: str) -> tuple[bool, str | None, list[str]]:
    """Parse the Critic's reply into (grounded, rewritten_query, unsupported_claims).

    Fail-closed: anything that is not an unambiguous approval counts as NOT grounded.
    Accepts the JSON format first; falls back to the legacy `GROUNDED:/QUERY:` lines,
    tolerating markdown (`**GROUNDED:** yes`) and missing spaces (`GROUNDED:yes`).
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and "grounded" in data:
            grounded = data["grounded"] is True
            query = str(data.get("query") or "").strip() or None
            claims = [str(c) for c in data.get("unsupported_claims") or [] if str(c).strip()]
            # a verdict that approves while listing unsupported claims contradicts itself
            return (grounded and not claims), query, claims

    grounded, query = False, None
    verdict_seen = False
    for raw in text.splitlines():
        line = _MARKUP.sub("", raw).strip()
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "grounded" and not verdict_seen:
            verdict_seen = True  # only the first verdict line counts; prose later is ignored
            # exact "yes" (optionally followed by punctuation) — "yes, partially" is a no
            grounded = re.fullmatch(r"yes[.!]?", value.lower()) is not None
        elif key == "query" and value not in ("", "-", "--", "—"):
            query = value
    return grounded, query, []


def critic(state: RAGState) -> RAGState:
    docs = state["documents"]
    if state["answer"].strip() == REFUSAL:
        # A refusal asserts nothing, so there is nothing to verify. Asking the Critic anyway
        # made it flag the question's own topic as an "unsupported claim" and retry for nothing.
        return {"grounded": True, "found": False, "unsupported": [],
                "critique": "answer is the refusal — no claims to verify"}
    problem = check_citations(state["answer"], len(docs))
    if problem:
        return {"grounded": False, "critique": f"citation check failed: {problem}",
                "unsupported": [], "query": state.get("query") or state["question"]}

    context = _format_context(docs)
    out = get_chat(json_mode=True).invoke(
        CRITIC_PROMPT.format(question=state["question"], context=context, answer=state["answer"])
    ).content
    if not out.strip():
        # An empty reply is a broken critic, not missing evidence: say so, and don't burn
        # retries — asking the same model again won't fix it.
        return {"grounded": False, "critic_error": True, "unsupported": [],
                "critique": "critic returned no output (e.g. a reasoning model that spent its "
                            "whole token budget thinking) — answer left unverified"}
    grounded, new_query, claims = parse_verdict(out)
    return {"grounded": grounded, "found": True, "critique": out.strip(), "unsupported": claims,
            "query": new_query or state["question"]}


# --- Node 4: Finalize -------------------------------------------------------
def finalize(state: RAGState) -> RAGState:
    """Never hand out an answer the Critic did not approve."""
    if state.get("grounded"):
        return {}
    return {"draft": state.get("answer", ""), "answer": REFUSAL}


# --- Routing ----------------------------------------------------------------
def route_after_critic(state: RAGState) -> str:
    """Stop if grounded or if we've exhausted the retry budget; otherwise retry."""
    if state.get("grounded") or state.get("critic_error") or state.get("attempts", 0) >= settings.max_retries + 1:
        return "finalize"
    return "retrieve"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("critic", critic)
    g.add_node("finalize", finalize)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"retrieve": "retrieve", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()
