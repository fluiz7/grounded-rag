"""Regression tests for the second adversarial review of PR #1 (the author's own review).

Every case reproduced against the PR branch before being fixed.
"""
import pytest
from langchain_core.documents import Document

from src import graph as G
from src.graph import (
    NOT_FOUND,
    REFUSAL,
    build_graph,
    check_citations,
    is_not_found,
    merge_rounds,
    parse_verdict,
)

ANSWER = "Signal quality dominates latency [1]."


# --- 1. greedy regex ----------------------------------------------------------

def test_trailing_braces_do_not_hide_the_json_verdict():
    reply = '{"grounded": false, "unsupported_claims": ["bad"], "query": "better query"} (see {ref})'
    assert parse_verdict(reply) == (False, "better query", ["bad"])


def test_legacy_yes_cannot_override_a_json_rejection():
    reply = 'GROUNDED: yes\nNote: {"grounded": false, "unsupported_claims": ["x"], "query": "real"} in {ctx}'
    grounded, query, _ = parse_verdict(reply)
    assert grounded is False
    assert query == "real"


def test_all_verdicts_must_approve():
    assert parse_verdict('{"grounded": true} {"grounded": false}')[0] is False
    assert parse_verdict('{"grounded": true}\nGROUNDED: no')[0] is False
    assert parse_verdict('{"grounded": true}\nGROUNDED: yes')[0] is True


# --- 2. field types -----------------------------------------------------------

def test_claims_as_a_string_is_one_claim_not_characters():
    _, _, claims = parse_verdict('{"grounded": false, "unsupported_claims": "Claim X", "query": "q"}')
    assert claims == ["Claim X"]


@pytest.mark.parametrize("bad", ["1", "true", '{"a": 1}', '[1, 2]'])
def test_claims_of_a_nonsensical_type_fail_closed_without_crashing(bad):
    grounded, _, _ = parse_verdict(f'{{"grounded": true, "unsupported_claims": {bad}, "query": ""}}')
    assert grounded is False


def test_non_string_query_is_ignored():
    _, query, _ = parse_verdict('{"grounded": false, "unsupported_claims": [], "query": ["a", "b"]}')
    assert query is None


# --- 3. refusal ---------------------------------------------------------------

@pytest.mark.parametrize("text", [NOT_FOUND, f'"{NOT_FOUND}"', f"{NOT_FOUND}.", f"**{NOT_FOUND}**", REFUSAL,
                                  f"{NOT_FOUND} [1-5]", f"{NOT_FOUND} [1, 2]."])  # seen from qwen2.5 end-to-end
def test_not_found_marker_is_recognised(text):
    assert is_not_found(text)


def test_an_answer_merely_mentioning_the_marker_is_not_a_refusal():
    assert not is_not_found(f"The status was {NOT_FOUND} in the logs [1].")


def test_not_found_is_language_neutral():
    # the old fixed English sentence got translated ("Não encontrei...") and was then
    # mistaken for an answer without citations
    assert "NOT_FOUND" in G.ANSWER_PROMPT.format(question="q", context="c", not_found=NOT_FOUND, feedback="")


def test_not_found_in_round_1_rewrites_the_query_and_searches_again(patch_pipeline, fake_store):
    patch_pipeline([
        NOT_FOUND,                     # answerer, round 1: the first search missed
        "handover latency RSRP",       # query rewrite
        ANSWER,                        # answerer, round 2
        '{"grounded": true, "unsupported_claims": [], "query": ""}',
    ])
    state = build_graph().invoke({"question": "q?"})
    assert fake_store.queries == ["q?", "handover latency RSRP"]
    assert state["grounded"] is True
    assert state["answer"] == ANSWER


def test_not_found_everywhere_is_reported_as_not_found_not_as_grounded(patch_pipeline, fake_store):
    patch_pipeline([NOT_FOUND, "query 2", NOT_FOUND, "query 3", NOT_FOUND])
    state = build_graph().invoke({"question": "q?"})
    assert state["grounded"] is False      # API contract: nothing was verified
    assert state["found"] is False
    assert state["answer"] == REFUSAL
    assert "draft" not in state            # a refusal is not an unverified draft
    assert fake_store.queries == ["q?", "query 2", "query 3"]


def test_not_found_stops_when_the_rewrite_repeats_a_query(patch_pipeline, fake_store):
    patch_pipeline([NOT_FOUND, "q?"])      # rewrite returns the query already tried
    state = build_graph().invoke({"question": "q?"})
    assert state["attempts"] == 1
    assert state["found"] is False


# --- 4. deterministic loop ----------------------------------------------------

def test_rejection_reason_is_fed_back_to_the_answerer(patch_pipeline):
    chat = patch_pipeline(["No citation here.", ANSWER, '{"grounded": true}'])
    state = build_graph().invoke({"question": "q?"})
    answer_prompts = [p for p in chat.prompts if "careful research assistant" in p]
    assert "REJECTED" not in answer_prompts[0]
    assert "REJECTED: the answer cites no sources" in answer_prompts[1]
    assert state["grounded"] is True


def test_identical_rejected_answer_stops_the_loop(patch_pipeline, fake_store):
    patch_pipeline(["No citation here."] * 3)
    state = build_graph().invoke({"question": "q?"})
    assert state["attempts"] == 2          # round 2 repeated round 1 verbatim -> stop
    assert state["stalled"] is True
    assert state["answer"] == REFUSAL


# --- 5. citation syntax -------------------------------------------------------

@pytest.mark.parametrize("text, n_docs, ok", [
    ("Latency increases under mobility [1, 2].", 2, True),
    ("Grouped [1,2,3] citations.", 3, True),
    ("A range [1-3] of sources.", 3, True),
    ("An en-dash range [1–2].", 2, True),
    ("As shown in [2024], latency fell [1].", 2, True),   # a year, not a citation
    ("buffer[0] overflows [1].", 2, True),                # indexing, not a citation
    ("Only a year [2024].", 2, False),                    # ...so this cites nothing
    ("Out of range [1, 9].", 2, False),
    ("Range past the end [1-4].", 2, False),
])
def test_citation_syntax(text, n_docs, ok):
    assert (check_citations(text, n_docs) is None) is ok


# --- 6. evidence cap ----------------------------------------------------------

def _docs(start):
    return [Document(page_content=f"p{p}", metadata={"source": "a.pdf", "page": p}) for p in range(start, start + 5)]


def test_round_one_evidence_survives_round_three():
    merged = merge_rounds([_docs(1), _docs(11), _docs(21)], cap=10)
    pages = [d.metadata["page"] for d in merged]
    assert len(pages) == 10
    assert {1, 11, 21} <= set(pages)        # each round's best hit is kept
    assert pages[:3] == [21, 11, 1]          # rank 1 of every round, newest first


# --- 7. ingestion -------------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    from langchain_chroma import Chroma
    from langchain_core.embeddings import FakeEmbeddings

    s = Chroma(collection_name="review2", embedding_function=FakeEmbeddings(size=8),
               persist_directory=str(tmp_path / "db"))
    monkeypatch.setattr("src.ingest.get_vectorstore", lambda: s)
    return s


def _pdf(path, texts):
    """A real PDF whose pages carry the given text (via reportlab-free raw content stream)."""
    from pypdf import PdfWriter
    from pypdf.generic import ContentStream, DecodedStreamObject, DictionaryObject, NameObject

    w = PdfWriter()
    for text in texts:
        page = w.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 150 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = w._add_object(ContentStream(stream, w))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        w.write(f)


def test_same_file_name_in_two_subfolders(tmp_path, store):
    from src.ingest import ingest

    _pdf(tmp_path / "papers" / "2024" / "paper.pdf", ["identical text"])
    _pdf(tmp_path / "papers" / "2025" / "paper.pdf", ["identical text"])
    ingest(str(tmp_path / "papers"))         # used to raise DuplicateIDError
    sources = {m["source"] for m in store.get()["metadatas"]}
    assert sources == {"2024/paper.pdf", "2025/paper.pdf"}


def test_editing_a_pdf_replaces_its_old_chunks(tmp_path, store):
    from src.ingest import ingest

    pdf = tmp_path / "papers" / "x.pdf"
    _pdf(pdf, ["version one"])
    ingest(str(tmp_path / "papers"))
    _pdf(pdf, ["version two"])
    ingest(str(tmp_path / "papers"))
    texts = store.get()["documents"]
    assert len(texts) == 1 and "version two" in texts[0]


def test_reingesting_an_unchanged_folder_is_a_no_op(tmp_path, store):
    from src.ingest import ingest

    _pdf(tmp_path / "papers" / "a.pdf", ["alpha", "beta"])
    ingest(str(tmp_path / "papers"))
    ingest(str(tmp_path / "papers"))
    assert store._collection.count() == 2
