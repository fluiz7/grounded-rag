"""Regression tests for the adversarial review of PR #2 (review of commit 39c04f8).

Every case reproduced end-to-end (qwen2.5:7b-instruct, Vehiclouds_Presentation.pdf) before
being fixed.
"""
import pytest
from langchain_core.documents import Document

from src.graph import (
    ANSWER_PROMPT,
    build_graph,
    check_language,
    detect_language,
    dominant_script,
    strip_echoed_question,
)

# --- 1. the answer stays in the question's language -----------------------------

ZH = "根据提供的内容，距离并不是影响延迟的主要因素 [1]。"
OK = '{"grounded": true, "unsupported_claims": [], "query": ""}'


@pytest.mark.parametrize("text, expected", [
    ("Does distance explain the latency observed under vehicle mobility?", "English"),
    # one exclusive word only ("what"): silence, and the prompt falls back to the generic wording
    ("What factors drive latency in edge computing under mobility?", None),
    ("A distância explica a latência observada com mobilidade veicular?", "Portuguese"),
    ("Quais fatores determinam a latência em computação de borda com mobilidade?", "Portuguese"),
    ("Qual é a relação entre a força do sinal e a latência?", "Portuguese"),
    ("Die Latenz wird hauptsächlich durch die Netzwerkqualität beeinflusst [7].", "German"),
    ("¿Qué factores determinan la latencia?", None),                # "qué" alone: unsure
    ("¿Cuáles son los factores que determinan la latencia de la red?", "Spanish"),
    ("Latency?", None),                                  # too little evidence: no guess
    ("距離は遅延を説明しますか？", None),                  # not Latin: the script check's job
])
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_chinese_answer_to_english_question_is_rejected():
    problem = check_language(ZH, "Does distance explain the latency under vehicle mobility?")
    assert problem is not None and "CJK" in problem and "LATIN" in problem


@pytest.mark.parametrize("answer, question, lang", [
    # both seen end-to-end with the first version of this PR
    ("Distance is not the primary factor affecting latency [2].",
     "A distância explica a latência observada com mobilidade veicular?", "English"),
    (("Die Latenz in der Edge-Computing wird hauptsächlich durch die Netzwerkqualität "
     "beeinflusst [7]. Zudem hat die Beweglichkeit der Endgeräte Einfluss auf die Latenz [4]."),
     "Quais fatores determinam a latência em computação de borda com mobilidade?", "German"),
])
def test_wrong_latin_language_is_rejected(answer, question, lang):
    problem = check_language(answer, question)
    assert problem is not None and f"is in {lang}" in problem and "write it in Portuguese" in problem


@pytest.mark.parametrize("answer, question", [
    ("Distance is not the primary factor [1].", "Does distance explain the latency?"),
    ("A distância não é o fator principal da latência [1].", "A distância explica a latência?"),
    ("A força do sinal influencia a latência, segundo o RSRP [4].", "Qual é a relação entre o sinal e a latência?"),
    ("距離は主な要因ではありません [1]。", "距離は遅延を説明しますか？"),        # kanji + kana
    ("Use `延迟` as the key [1].", "Which key should I use?"),               # code is ignored
    ("RSRP [1].", "A distância explica a latência?"),       # answer too short to judge: allowed
    ("[1]", "Qual?"),
])
def test_same_language_or_undecidable_passes(answer, question):
    assert check_language(answer, question) is None


def test_dominant_script_counts_letters_only():
    assert dominant_script("R2 = 0.0177 [1]") == "LATIN"
    assert dominant_script("123 [1] —") is None


def test_chinese_draft_goes_back_to_the_answerer(patch_pipeline):
    chat = patch_pipeline([ZH, "Distance is not the primary factor [1].", OK])
    state = build_graph().invoke({"question": "Does distance explain the latency?"})
    assert state["grounded"] is True
    assert state["answer"] == "Distance is not the primary factor [1]."
    assert "another script" in chat.prompts[1]                 # fed back to the Answerer
    assert len(chat.prompts) == 3                               # Critic never saw the Chinese


def test_english_draft_to_portuguese_question_goes_back(patch_pipeline):
    chat = patch_pipeline(["Distance is not the primary factor affecting latency [2].",
                           "A distância não é o principal fator que afeta a latência [2].", OK])
    state = build_graph().invoke({"question": "A distância explica a latência observada com mobilidade?"})
    assert state["grounded"] is True and state["answer"].startswith("A distância")
    assert "write it in Portuguese" in chat.prompts[1]


def test_prompt_names_the_language():
    chat_prompt = lambda q: ANSWER_PROMPT.format(
        not_found="NOT_FOUND", feedback="", question=q, context="c",
        language=detect_language(q) or "the language the QUESTION is written in")
    assert "Write the answer in Portuguese" in chat_prompt("A distância explica a latência?")
    assert "Write the answer in the language the QUESTION" in chat_prompt("Latency?")
    assert "Translate it into" not in chat_prompt("Latency?")


# --- 2. an echoed question is not part of the answer -----------------------------

@pytest.mark.parametrize("raw, question, expected", [
    ("Does distance explain the latency under vehicle mobility?\n\n[1] No [1].",
     "Does distance explain the latency under vehicle mobility?", "[1] No [1]."),
    ("Qual é a relação entre a força do sinal e a latência\nA força do sinal influencia [4].",
     "Qual é a relação entre a força do sinal e a latência?", "A força do sinal influencia [4]."),
    ("**Does distance explain latency?**\nNo [2].", "does distance explain latency", "No [2]."),
])
def test_echo_is_stripped(raw, question, expected):
    assert strip_echoed_question(raw, question) == expected


@pytest.mark.parametrize("raw", [
    "Does distance explain latency? No, it does not [2].",    # single line: it IS the answer
    "Distance does not explain latency [2].\nSignal does [1].",
])
def test_non_echo_is_kept(raw):
    assert strip_echoed_question(raw, "Does distance explain latency?") == raw


def test_echo_never_reaches_the_critic(patch_pipeline):
    q = "Does distance explain the latency?"
    chat = patch_pipeline([f"{q}\n\nDistance is not the primary factor [1].",
                           '{"grounded": true, "unsupported_claims": [], "query": ""}'])
    state = build_graph().invoke({"question": q})
    assert state["answer"] == "Distance is not the primary factor [1]."
    assert f"ANSWER:\n{q}" not in chat.prompts[1]


# --- 3. the rewrite is told the collection's language ----------------------------

EN_DOCS = [
    Document(page_content="The signal strength is one of the factors that influence the latency "
                          "of the MEC server, and it is also related to the distance.",
             metadata={"source": "slides.pdf", "page": 31}),
]


def test_rewrite_is_told_to_search_in_the_documents_language(patch_pipeline, fake_store):
    fake_store.docs = EN_DOCS
    chat = patch_pipeline(["NOT_FOUND", "signal strength latency", "NOT_FOUND",
                           "RSRP latency mobility", "NOT_FOUND"])
    build_graph().invoke({"question": "Quais fatores determinam a latência?"})
    assert "The documents are written in English: write the query in English" in chat.prompts[1]


def test_no_hint_when_question_and_documents_share_the_language(patch_pipeline, fake_store):
    fake_store.docs = EN_DOCS
    chat = patch_pipeline(["NOT_FOUND", "signal strength", "NOT_FOUND", "rsrp", "NOT_FOUND"])
    build_graph().invoke({"question": "What factors drive the latency?"})
    assert "The documents are written in" not in chat.prompts[1]


def test_no_hint_without_documents(patch_pipeline, fake_store):
    fake_store.docs = []
    chat = patch_pipeline(["NOT_FOUND", "outra busca", "NOT_FOUND", "mais uma", "NOT_FOUND"])
    build_graph().invoke({"question": "Quais fatores determinam a latência?"})
    assert "The documents are written in" not in chat.prompts[1]
