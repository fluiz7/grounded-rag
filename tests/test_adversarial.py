"""Regression tests from an adversarial review of the Critic loop.

Each test pins a failure that existed before: the verdict parser could invert a
rejection into an approval, unverified answers were handed out as answers, retries
threw away evidence, and re-ingesting duplicated the index.
"""
import pytest
from langchain_core.documents import Document

from src.graph import REFUSAL, build_graph, check_citations, parse_verdict

ANSWER = "Signal quality dominates latency [1]."


# --- verdict parsing: fail-closed ---------------------------------------------

@pytest.mark.parametrize("reply, expected", [
    ('{"grounded": true, "unsupported_claims": [], "query": ""}', True),
    ('{"grounded": false, "unsupported_claims": ["x"], "query": "q"}', False),
    # approving while listing unsupported claims contradicts itself -> not grounded
    ('{"grounded": true, "unsupported_claims": ["claim 2"], "query": ""}', False),
    ('Sure! {"grounded": true, "unsupported_claims": []} hope it helps', True),
    ("GROUNDED: yes\nQUERY: --", True),
    ("**GROUNDED:** yes\n**QUERY:** --", True),        # markdown, common in llama3.1
    ("GROUNDED:yes", True),                            # no space after the colon
    ("GROUNDED: Yes.", True),
    # a rejection that later quotes the approval phrase must stay a rejection
    ("GROUNDED: no\nQUERY: x\nNote: I cannot say grounded: yes here.", False),
    ("GROUNDED: yes, partially - claim 2 is unsupported", False),
    ("The answer looks fine to me.", False),           # no verdict at all
    ("", False),
    ('{"grounded": "yes"}', False),                    # only a real JSON true approves
])
def test_parse_verdict_is_fail_closed(reply, expected):
    grounded, _query, _claims = parse_verdict(reply)
    assert grounded is expected


def test_parse_verdict_reads_markdown_query():
    _, query, _ = parse_verdict("GROUNDED: no\n**QUERY:** signal strength latency")
    assert query == "signal strength latency"


def test_parse_verdict_returns_unsupported_claims():
    _, _, claims = parse_verdict('{"grounded": false, "unsupported_claims": ["a", ""], "query": "q"}')
    assert claims == ["a"]


# --- deterministic citation check ---------------------------------------------

@pytest.mark.parametrize("text, n_docs, ok", [
    ("A [1] and B [2].", 2, True),
    ("No citations here.", 2, False),
    ("Cites a phantom source [7].", 2, False),
    ("Zero is not a source [0].", 2, False),
])
def test_check_citations(text, n_docs, ok):
    assert (check_citations(text, n_docs) is None) is ok


def test_phantom_citation_is_rejected_without_calling_the_critic(patch_pipeline, fake_store):
    chat = patch_pipeline(["Signal quality dominates [7]."] * 3)   # only answerer replies
    state = build_graph().invoke({"question": "q?"})
    assert state["grounded"] is False
    assert all("strict verifier" not in p for p in chat.prompts)    # critic never consulted


# --- pipeline behaviour -------------------------------------------------------

def test_rejected_answer_is_never_returned(patch_pipeline):
    patch_pipeline([ANSWER, '{"grounded": false, "query": "a"}'] * 3)
    state = build_graph().invoke({"question": "q?"})
    assert state["answer"] == REFUSAL
    assert state["draft"] == ANSWER


def test_inverted_verdict_no_longer_approves(patch_pipeline):
    patch_pipeline([ANSWER, "GROUNDED: no\nQUERY: x\nI cannot say grounded: yes."] * 3)
    state = build_graph().invoke({"question": "q?"})
    assert state["grounded"] is False
    assert state["answer"] == REFUSAL


def test_markdown_query_is_actually_used_for_the_retry(patch_pipeline, fake_store):
    patch_pipeline([ANSWER, "GROUNDED: no\n**QUERY:** handover latency", ANSWER, "GROUNDED: yes"])
    build_graph().invoke({"question": "q?"})
    assert fake_store.queries == ["q?", "handover latency"]


def test_evidence_accumulates_across_rounds(patch_pipeline, fake_store):
    page9 = Document(page_content="Signal quality dominates latency.", metadata={"source": "p.pdf", "page": 9})
    page12 = Document(page_content="Low RSRP doubles latency.", metadata={"source": "p.pdf", "page": 12})

    def search(q, k=5):
        fake_store.queries.append(q)
        return [page9] if len(fake_store.queries) == 1 else [page12]

    fake_store.similarity_search = search
    patch_pipeline([ANSWER, '{"grounded": false, "query": "RSRP"}', "Both [1][2].", '{"grounded": true}'])
    state = build_graph().invoke({"question": "q?"})
    pages = [d.metadata["page"] for d in state["documents"]]
    assert pages == [12, 9]          # fresh evidence first, earlier evidence kept
    assert state["grounded"] is True


def test_merged_documents_are_deduplicated(patch_pipeline, fake_store):
    patch_pipeline([ANSWER, '{"grounded": false, "query": "again"}', ANSWER, '{"grounded": true}'])
    state = build_graph().invoke({"question": "q?"})
    assert len(state["documents"]) == len(fake_store.docs)   # same hits twice -> no duplicates


# --- ingestion ----------------------------------------------------------------

def test_reingesting_does_not_duplicate_chunks(tmp_path):
    from langchain_chroma import Chroma
    from langchain_core.embeddings import FakeEmbeddings

    from src.ingest import chunk_id

    def store():
        return Chroma(collection_name="idempotency", embedding_function=FakeEmbeddings(size=8),
                      persist_directory=str(tmp_path))

    chunks = [Document(page_content=f"chunk {i}", metadata={"source": "a.pdf", "page": i, "start_index": 0})
              for i in range(3)]
    for _ in range(2):   # the user runs `ingest` twice on the same folder
        store().add_documents(chunks, ids=[chunk_id(c) for c in chunks])
    assert store()._collection.count() == 3


def test_chunk_id_distinguishes_same_text_at_different_offsets():
    from src.ingest import chunk_id

    a = Document(page_content="same", metadata={"source": "a.pdf", "page": 1, "start_index": 0})
    b = Document(page_content="same", metadata={"source": "a.pdf", "page": 1, "start_index": 900})
    assert chunk_id(a) != chunk_id(b)


def test_pages_are_numbered_like_the_pdf_viewer(tmp_path):
    from pypdf import PdfWriter

    from src.ingest import load_pdf

    pdf = tmp_path / "three.pdf"
    w = PdfWriter()
    for _ in range(3):
        w.add_blank_page(width=72, height=72)
    with open(pdf, "wb") as f:
        w.write(f)
    assert [d.metadata["page"] for d in load_pdf(pdf)] == [1, 2, 3]
    assert {d.metadata["source"] for d in load_pdf(pdf)} == {"three.pdf"}


def test_empty_critic_output_is_reported_as_a_critic_failure(patch_pipeline, fake_store):
    # Found end-to-end with qwen3.5:4b: a reasoning model exhausted its token budget
    # thinking and returned "" — which used to look exactly like "not grounded".
    patch_pipeline([ANSWER, ""])
    state = build_graph().invoke({"question": "q?"})
    assert state["critic_error"] is True
    assert state["attempts"] == 1            # no retries burned on a broken critic
    assert state["answer"] == REFUSAL
    assert "no output" in state["critique"]
