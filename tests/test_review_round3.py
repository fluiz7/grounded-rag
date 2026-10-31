"""Regression tests for the third adversarial review of PR #1 (review of commit d940004).

Every case reproduced against that commit before being fixed.
"""
import pytest
from langchain_core.documents import Document

from src.graph import NOT_FOUND, REFUSAL, build_graph, check_citations, parse_verdict
from tests.conftest import FakeVectorStore

# --- 1. citations are decided by content, not by the preceding character ----------

@pytest.mark.parametrize("text, n_docs, ok", [
    ("Signal quality dominates [1][7].", 2, False),     # adjacent groups: [7] is a phantom
    ("Both claims hold [1][2].", 2, True),
    ("Signal quality[1] dominates.", 2, True),           # glued to the word — still a citation
    ("Normalized to the [0, 1] interval [1].", 2, True),  # [0, 1] is an interval
    ("Normalized to [0, 1] only.", 2, False),            # ...so this cites nothing
    ("buffer[0] overflows [1].", 2, True),               # [0] is indexing
    ("As shown in [2024], latency fell [1].", 2, True),  # a year
    ("See `a[3]` in the code [1].", 2, True),            # inline code is ignored
    ("Ranges [1-2] and lists [1, 2].", 2, True),
])
def test_citation_content_rules(text, n_docs, ok):
    assert (check_citations(text, n_docs) is None) is ok


# --- 2. every verdict counts, at any depth ----------------------------------------

def test_every_legacy_verdict_line_counts():
    grounded, query, _ = parse_verdict("Claim 1:\nGROUNDED: yes\nClaim 2:\nGROUNDED: no\nQUERY: better query")
    assert grounded is False
    assert query == "better query"


def test_all_legacy_yes_still_approves():
    assert parse_verdict("Claim 1:\nGROUNDED: yes\nClaim 2:\nGROUNDED: yes")[0] is True


def test_nested_json_verdict_is_found():
    reply = '{"verdict": {"grounded": false, "unsupported_claims": ["x"], "query": "better"}}'
    assert parse_verdict(reply) == (False, "better", ["x"])


def test_nested_approval_with_a_nested_rejection_is_a_rejection():
    reply = '{"claims": [{"grounded": true}, {"grounded": false, "unsupported_claims": ["b"]}]}'
    assert parse_verdict(reply)[0] is False


# --- 3. a repeated answer over NEW evidence goes back to the critic ---------------

def test_same_answer_with_new_documents_is_rechecked(patch_pipeline, fake_store):
    a = Document(page_content="Claim A holds.", metadata={"source": "p.pdf", "page": 1})
    b = Document(page_content="Claim B holds.", metadata={"source": "p.pdf", "page": 2})

    def search(q, k=5):
        fake_store.queries.append(q)
        return [a] if len(fake_store.queries) == 1 else [b]

    fake_store.similarity_search = search
    answer = "Claim A and Claim B [1][2]."
    patch_pipeline([
        answer, '{"grounded": false, "unsupported_claims": ["Claim B"], "query": "claim b"}',
        answer, '{"grounded": true, "unsupported_claims": [], "query": ""}',
    ])
    state = build_graph().invoke({"question": "q?"})
    assert state["grounded"] is True
    assert not state.get("stalled")


def test_same_answer_over_the_same_documents_still_stops(patch_pipeline):
    patch_pipeline(["No citation here."] * 3)   # FakeVectorStore returns the same docs
    state = build_graph().invoke({"question": "q?"})
    assert state["stalled"] is True
    assert state["attempts"] == 2


# --- 4. a rejected draft survives a later NOT_FOUND --------------------------------

def test_rejected_draft_is_kept_when_a_later_round_says_not_found(patch_pipeline):
    patch_pipeline([
        "Latency is 5ms [1].", '{"grounded": false, "unsupported_claims": ["5ms"], "query": "latency"}',
        NOT_FOUND, "another query", NOT_FOUND,
    ])
    state = build_graph().invoke({"question": "q?"})
    assert state["draft"] == "Latency is 5ms [1]."
    assert state["found"] is True          # an answer WAS drafted -> unverified (exit 3), not "not found"
    assert state["answer"] == REFUSAL


def test_grounded_answer_clears_an_earlier_draft(patch_pipeline):
    patch_pipeline([
        "Wrong [1].", '{"grounded": false, "unsupported_claims": ["w"], "query": "q2"}',
        "Right [1].", '{"grounded": true, "unsupported_claims": [], "query": ""}',
    ])
    state = build_graph().invoke({"question": "q?"})
    assert state["grounded"] is True
    assert state["answer"] == "Right [1]."
    assert not state.get("draft")


# --- 5. a file has one identity, however it is ingested -------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    from langchain_chroma import Chroma
    from langchain_core.embeddings import FakeEmbeddings

    s = Chroma(collection_name="review3", embedding_function=FakeEmbeddings(size=8),
               persist_directory=str(tmp_path / "db"))
    monkeypatch.setattr("src.ingest.get_vectorstore", lambda: s)
    return s


def test_folder_then_single_file_replaces_instead_of_duplicating(tmp_path, store):
    from src.ingest import ingest
    from tests.test_review_round2 import _pdf

    pdf = tmp_path / "papers" / "2024" / "paper.pdf"
    _pdf(pdf, ["version one"])
    ingest(str(tmp_path / "papers"))
    _pdf(pdf, ["version two"])
    ingest(str(pdf))                      # the same file, passed directly
    texts = store.get()["documents"]
    assert len(texts) == 1 and "version two" in texts[0]


def test_single_file_then_folder_also_replaces(tmp_path, store):
    from src.ingest import ingest
    from tests.test_review_round2 import _pdf

    pdf = tmp_path / "papers" / "2024" / "paper.pdf"
    _pdf(pdf, ["version one"])
    ingest(str(pdf))
    _pdf(pdf, ["version two"])
    ingest(str(tmp_path / "papers"))
    assert store._collection.count() == 1


def test_source_stays_a_short_display_label(tmp_path, store):
    from src.ingest import ingest
    from tests.test_review_round2 import _pdf

    _pdf(tmp_path / "papers" / "2024" / "paper.pdf", ["text"])
    ingest(str(tmp_path / "papers"))
    meta = store.get()["metadatas"][0]
    assert meta["source"] == "2024/paper.pdf"              # what citations show
    assert meta["path"].endswith("/papers/2024/paper.pdf")  # what identifies the file


def test_store_fixture_is_isolated():
    # guard: the fake store used by other tests still returns its two canned documents
    assert len(FakeVectorStore().docs) == 2


# --- found end-to-end while verifying this round --------------------------------

@pytest.mark.parametrize("text", ["[4]", "[1][2].", " [1, 2] ", "**[1]**"])
def test_an_answer_made_only_of_citations_is_rejected(text):
    problem = check_citations(text, 5)
    assert problem is not None and "only citations" in problem


def test_a_short_real_answer_is_not_mistaken_for_citations_only():
    assert check_citations("Yes [1].", 2) is None
