"""The multi-agent pipeline, wired with LangGraph.

    Retriever  ->  Answerer  ->  Critic  --grounded-->  Finalize -> END
        ^                            |
        |________ not grounded ______|  (rewritten query + feedback, then retry)

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
# Language-neutral marker the Answerer emits when the context lacks the answer. A fixed
# English sentence conflicted with "answer in the question's language": models translate
# it, the translation no longer matched, and a refusal was mistaken for an uncited answer.
NOT_FOUND = "NOT_FOUND"


class RAGState(TypedDict, total=False):
    question: str          # the user's original question (never mutated)
    query: str             # current search query (Critic may rewrite this)
    queries_tried: list[str]
    rounds: list[list[Document]]  # hits per retrieval round, best first
    documents: list[Document]     # merged context shown to the Answerer
    answer: str
    prev_answer: str       # answer of the previous round, to detect a stuck loop
    answered_docs: list    # evidence the current answer was written from...
    prev_answered_docs: list  # ...and the evidence behind the previous one
    feedback: str          # why the previous answer was rejected, fed back to the Answerer
    draft: str             # last unverified answer, kept when the Critic never approved
    grounded: bool         # True only when the Critic approved a real answer
    found: bool            # False when the documents did not contain the answer
    stalled: bool          # a retry would repeat the same work — stop early
    critique: str
    unsupported: list[str]  # claims the Critic flagged as unsupported
    critic_error: bool     # the Critic produced no verdict at all (infrastructure, not evidence)
    attempts: int          # number of retrieval rounds so far


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {d.metadata.get('source', '?')}, "
        f"page {d.metadata.get('page', '?')})\n{d.page_content}"
        for i, d in enumerate(docs)
    )


def _doc_key(d: Document) -> tuple:
    return (d.metadata.get("source"), d.metadata.get("page"), d.page_content)


def merge_rounds(rounds: list[list[Document]], cap: int) -> list[Document]:
    """Interleave hits by rank across rounds (newest round first within a rank), dedupe, cap.

    "Newest round first, then the rest" let rounds 2 and 3 fill a 2 x top_k cap and evict
    every document from round 1. Interleaving by rank keeps each round's best evidence.
    """
    merged, seen = [], set()
    depth = max((len(r) for r in rounds), default=0)
    for rank in range(depth):
        for hits in reversed(rounds):
            if rank < len(hits):
                key = _doc_key(hits[rank])
                if key not in seen:
                    seen.add(key)
                    merged.append(hits[rank])
    return merged[:cap]


# --- Node 1: Retriever ------------------------------------------------------
def retrieve(state: RAGState) -> RAGState:
    """Search with the current query and merge with the evidence of earlier rounds.

    Replacing the documents on every retry would throw away evidence the first round
    already found: if round 1 retrieves half of the support and round 2 the other half,
    the answer could never be grounded on both.
    """
    query = state.get("query") or state["question"]
    fresh = get_vectorstore().similarity_search(query, k=settings.top_k)
    rounds = [*state.get("rounds", []), fresh]
    return {
        "rounds": rounds,
        "documents": merge_rounds(rounds, 2 * settings.top_k),
        "queries_tried": [*state.get("queries_tried", []), query],
        "attempts": state.get("attempts", 0) + 1,
    }


# --- Node 2: Answerer -------------------------------------------------------
ANSWER_PROMPT = """You are a careful research assistant. Answer the QUESTION using \
ONLY the CONTEXT below, in the same language as the QUESTION. Cite every claim inline \
with the bracket numbers, e.g. [1], [2]. If the context does not contain the answer, \
reply with exactly {not_found} and nothing else — no citations.
The CONTEXT may be in a different language than the QUESTION. That is expected: a fact \
stated in the CONTEXT in any language counts as support. Translate it into the \
QUESTION's language and cite it; do not answer {not_found} because of the language.
{feedback}
QUESTION:
{question}

CONTEXT:
{context}

Write the answer in the language the QUESTION is written in, citing sources as [n]. Only the {not_found} marker itself is never translated.
Answer:"""

FEEDBACK_BLOCK = """
Your PREVIOUS answer to this question was REJECTED: {reason}
Write a corrected answer that fixes this. Every claim needs a [n] citation to the CONTEXT.
"""


def is_not_found(answer_text: str) -> bool:
    """The Answerer declined: the NOT_FOUND marker or the legacy English refusal sentence.

    Quotes, punctuation and bracket citations around the marker are tolerated — told to
    both "reply NOT_FOUND" and "cite sources as [n]", models do write `NOT_FOUND [1-5]`.
    """
    core = re.sub(r"\[[^\]]*\]|[\s\"'`*.!]", "", answer_text)
    return core == NOT_FOUND or answer_text.strip() == REFUSAL


def answer(state: RAGState) -> RAGState:
    context = _format_context(state["documents"])
    reason = state.get("feedback")
    prompt = ANSWER_PROMPT.format(
        question=state["question"], context=context, not_found=NOT_FOUND,
        feedback=FEEDBACK_BLOCK.format(reason=reason) if reason else "",
    )
    return {
        "prev_answer": state.get("answer", ""),
        "prev_answered_docs": state.get("answered_docs", []),
        "answered_docs": [_doc_key(d) for d in state["documents"]],
        "answer": get_chat().invoke(prompt).content.strip(),
    }


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

REWRITE_PROMPT = """The documents retrieved for the QUESTION below did not contain the answer.
Queries already tried: {tried}
Write ONE different search query (keywords, synonyms or a narrower angle) likely to find \
the answer in a collection of technical PDFs. Reply with the query only.

QUESTION:
{question}"""

# A bracket group of numbers, optionally a list or range: [1], [1, 2], [1-3], [1–3].
# No lookbehind: a citation may follow a word (`quality[1]`) or another citation
# (`[1][7]` — the second group must be checked too, or a phantom source slips through).
_CITATION = re.compile(r"\[(\s*\d+(?:\s*[,–-]\s*\d+)*\s*)\]")
# Fenced blocks first (``` or ~~~, unterminated runs to the end), then inline spans:
# otherwise the first two backticks of ```python match as an empty inline span and the
# block's `items[99]` is read as a citation.
_CODE_SPAN = re.compile(r"```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*`")
_MARKUP = re.compile(r"[*_`#>]")


def cited_sources(answer_text: str) -> list[int]:
    """Source numbers cited in the text.

    Whether a bracket group is a citation is decided by its CONTENT, not by what precedes
    it: source numbers start at 1, so a group containing 0 (`buffer[0]`, the interval
    `[0, 1]`) is not a citation. 4-digit numbers are dropped one by one (`[2024]` is a year),
    without hiding the rest of the group: `[99, 2024]` still cites 99. Code is ignored.
    """
    numbers: list[int] = []
    for group in _CITATION.findall(_CODE_SPAN.sub(" ", answer_text)):
        flat: list[int] = []
        for part in group.split(","):
            bounds = [int(n) for n in re.split(r"\s*[–-]\s*", part.strip()) if n]
            if len(bounds) == 2 and 0 < bounds[0] <= bounds[1] and bounds[1] - bounds[0] < 50:
                flat.extend(range(bounds[0], bounds[1] + 1))
            else:
                flat.extend(bounds)
        if flat and 0 not in flat:
            numbers.extend(n for n in flat if n < 1000)
    return numbers


def check_citations(answer_text: str, n_docs: int) -> str | None:
    """Deterministic pre-check. Returns a problem description, or None if citations are sane.

    Cheap and LLM-free: an answer with no citations, or citing a source number that was
    never retrieved, cannot be grounded — no need to spend a Critic call to find out.
    """
    cited = cited_sources(answer_text)
    if not cited:
        return "the answer cites no sources"
    # seen end-to-end from qwen2.5: an "answer" that is only `[4]` — citations, no claim.
    # One word character is enough: "No [1].", "42 [1]." and "5% [1]." are real answers.
    if not re.search(r"\w", _CITATION.sub("", answer_text)):
        return "the answer contains only citations; write the actual answer, citing it"
    invalid = sorted({n for n in cited if not 1 <= n <= n_docs})
    if invalid:
        return f"the answer cites sources that were not retrieved: {invalid} (only {n_docs} available)"
    return None


def _verdict_dicts(obj) -> list[dict]:
    """Dicts carrying a "grounded" key at ANY depth (`{"verdict": {"grounded": ...}}`)."""
    if isinstance(obj, dict):
        own = [obj] if "grounded" in obj else []
        return own + [d for v in obj.values() for d in _verdict_dicts(v)]
    if isinstance(obj, list):
        return [d for v in obj for d in _verdict_dicts(v)]
    return []


def _json_verdicts(text: str) -> list[dict]:
    """Every verdict object in `text`, top-level or nested.

    raw_decode from each '{' instead of one greedy regex: a greedy `\\{.*\\}` swallowed any
    later brace (`(see {ref})`), failed to parse, and silently fell back to prose.
    """
    decoder, found, i = json.JSONDecoder(), [], text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            end = i + 1
        else:
            found += _verdict_dicts(obj)
        i = text.find("{", end)
    return found


def _legacy_verdicts(text: str) -> tuple[list[bool], str | None]:
    """EVERY `GROUNDED:` line (markdown-tolerant) and the `QUERY:` line, if any.

    All of them, not only the first: a critic that judges claim by claim writes one verdict
    per claim, and `GROUNDED: yes` followed by `GROUNDED: no` must not count as approval.
    """
    verdicts: list[bool] = []
    query = None
    for raw in text.splitlines():
        line = _MARKUP.sub("", raw).strip()
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "grounded":
            # exact "yes"/"true" (optionally followed by punctuation) — "yes, partially" is
            # a no. "true" because CRITIC_PROMPT itself asks for `"grounded": true or false`.
            verdicts.append(re.fullmatch(r"(?:yes|true)[.!]?", value.lower()) is not None)
        elif key == "query" and value not in ("", "-", "--", "—"):
            query = value
    return verdicts, query


def parse_verdict(text: str) -> tuple[bool, str | None, list[str]]:
    """Parse the Critic's reply into (grounded, rewritten_query, unsupported_claims).

    Fail-closed:
    - fields are type-checked (Ollama's JSON mode guarantees syntax, not a schema);
      a malformed verdict is a rejection, never a crash;
    - if the reply contains several verdicts (JSON and/or a `GROUNDED:` line), it is an
      approval only if ALL of them approve;
    - approving while listing unsupported claims contradicts itself -> rejection.
    """
    verdicts: list[bool] = []
    query: str | None = None
    claims: list[str] = []

    for data in _json_verdicts(text):
        ok = data["grounded"] is True
        raw_claims = data.get("unsupported_claims", [])
        if isinstance(raw_claims, str):
            raw_claims = [raw_claims]
        if isinstance(raw_claims, list) and all(isinstance(c, str) for c in raw_claims):
            claims += [c.strip() for c in raw_claims if c.strip()]
        elif raw_claims is not None:
            ok = False  # unsupported_claims of a nonsensical type: malformed verdict
        q = data.get("query")
        if isinstance(q, str) and q.strip() and query is None:
            query = q.strip()
        verdicts.append(ok)

    legacy, legacy_query = _legacy_verdicts(text)
    verdicts += legacy
    query = query or legacy_query

    grounded = bool(verdicts) and all(verdicts) and not claims
    return grounded, query, claims


def _rewrite_query(state: RAGState) -> str | None:
    tried = state.get("queries_tried", [])
    out = get_chat().invoke(REWRITE_PROMPT.format(question=state["question"], tried=tried)).content
    query = out.strip().strip("\"'").strip()
    return query if query and query not in tried else None


def critic(state: RAGState) -> RAGState:
    docs = state["documents"]
    budget_left = state.get("attempts", 0) < settings.max_retries + 1

    if is_not_found(state["answer"]):
        # Nothing to verify — but "not found" in round 1 may just mean the first search
        # missed. Rewrite the query and search again instead of ending the corrective loop.
        new_query = _rewrite_query(state) if budget_left else None
        return {"grounded": False, "found": False, "unsupported": [], "feedback": "",
                "stalled": budget_left and new_query is None,
                "query": new_query or state.get("query") or state["question"],
                "critique": "answer was NOT_FOUND"
                            + (f" — retrying with query: {new_query}" if new_query else "")}

    if (state["answer"] == state.get("prev_answer")
            and state.get("answered_docs") == state.get("prev_answered_docs")):
        # Same answer over the SAME evidence: a deterministic model won't change it — stop.
        # (Same answer over NEW evidence is different: the new documents may now support
        # it, so it goes to the Critic again.)
        return {"grounded": False, "found": True, "stalled": True, "unsupported": [],
                "draft": state["answer"],
                "critique": "the answerer repeated a rejected answer over the same evidence — stopping"}

    problem = check_citations(state["answer"], len(docs))
    if problem:
        return {"grounded": False, "found": True, "unsupported": [], "feedback": problem,
                "draft": state["answer"],
                "critique": f"citation check failed: {problem}",
                "query": state.get("query") or state["question"]}

    context = _format_context(docs)
    out = get_chat(json_mode=True).invoke(
        CRITIC_PROMPT.format(question=state["question"], context=context, answer=state["answer"])
    ).content
    if not out.strip():
        # An empty reply is a broken critic, not missing evidence: say so, and don't burn
        # retries — asking the same model again won't fix it.
        return {"grounded": False, "found": True, "critic_error": True, "unsupported": [],
                "draft": state["answer"],
                "critique": "critic returned no output (e.g. a reasoning model that spent its "
                            "whole token budget thinking) — answer left unverified"}
    grounded, new_query, claims = parse_verdict(out)
    rejected: RAGState = {} if grounded else {"draft": state["answer"]}
    feedback = "" if grounded else (
        "these claims are not supported by the context: " + "; ".join(claims) if claims
        else "the verifier found claims not supported by the context")
    return {"grounded": grounded, "found": True, "critique": out.strip(), "unsupported": claims,
            "feedback": feedback, "query": new_query or state["question"], **rejected}


# --- Node 4: Finalize -------------------------------------------------------
def finalize(state: RAGState) -> RAGState:
    """Never hand out an answer the Critic did not approve.

    Every rejected answer is kept in `draft` as soon as it is rejected, so a later round
    that falls back to NOT_FOUND cannot erase it: a run that DID draft an answer ends as
    "unverified" (exit 3), and "not found" (exit 4) is reserved for runs that never did.
    """
    if state.get("grounded"):
        return {"draft": ""}
    if state.get("draft"):
        return {"answer": REFUSAL, "found": True}
    return {"answer": REFUSAL, "found": False}


# --- Routing ----------------------------------------------------------------
def route_after_critic(state: RAGState) -> str:
    """Stop when grounded, when retrying can't help, or when the budget is spent."""
    if (state.get("grounded") or state.get("critic_error") or state.get("stalled")
            or state.get("attempts", 0) >= settings.max_retries + 1):
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
