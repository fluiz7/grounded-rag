"""Regression tests for the second adversarial review of PR #2 (review of commit afe5558).

Every case reproduced against that commit before being fixed.
"""
import pytest
from langchain_core.documents import Document

from src.graph import build_graph, check_language
from src.language import detect_language

# --- 1. no language wins on words it shares with another -------------------------

@pytest.mark.parametrize("question", [
    "Does a handover cause a latency spike?",
    "Can a vehicle maintain a 5G connection?",
    "Why did a packet suffer a delay?",
    "Provide a summary of a drive test.",
])
def test_english_questions_with_a_are_not_portuguese(question):
    assert detect_language(question) in ("English", None)


def test_german_question_is_not_english():
    assert detect_language("Was verursacht Latenz in 5G-Netzen an Basisstationen?") in ("German", None)


@pytest.mark.parametrize("answer", [
    "Yes, a handover causes a 200ms spike [1].",
    "Factors such as a handover or a weak signal increase latency [1].",
    "As a vehicle moves, latency spikes as signal quality drops [1].",
    "No correlation was found, as shown in a regression test [1].",
    "The experiments ran in São Paulo and Fundão [1].",      # names, not Portuguese
])
def test_english_answers_to_english_questions_pass(answer):
    assert check_language(answer, "What causes latency spikes in the network?") is None


@pytest.mark.parametrize("answer, question", [
    ("La latencia se debe a cambios de celda o a mala señal, como se observa [1].",
     "¿Qué factores determinan la latencia?"),
    ("La distance a peu d'impact, mais le signal se dégrade à haute vitesse [1].",
     "Quels facteurs déterminent la latence dans le réseau ?"),
])
def test_spanish_and_french_answers_are_not_portuguese(answer, question):
    assert check_language(answer, question) is None


# the cases that motivated the check must still be caught
@pytest.mark.parametrize("answer, question, lang", [
    ("Distance is not the primary factor affecting latency [2].",
     "A distância explica a latência observada com mobilidade veicular?", "English"),
    (("Die Latenz wird hauptsächlich durch die Netzwerkqualität beeinflusst [7]. Zudem hat "
      "die Beweglichkeit der Endgeräte Einfluss auf die Latenz [4]."),
     "Quais fatores determinam a latência em computação de borda com mobilidade?", "German"),
])
def test_real_wrong_language_answers_still_rejected(answer, question, lang):
    problem = check_language(answer, question)
    assert problem is not None and f"is in {lang}" in problem


@pytest.mark.parametrize("question", [
    "A distância explica a latência observada com mobilidade veicular?",
    "Quais fatores determinam a latência em computação de borda com mobilidade?",
    "Qual slide analisa a distância e a latência?",
])
def test_portuguese_questions_still_named(question):
    assert detect_language(question) == "Portuguese"


def test_rejecting_needs_more_evidence_than_naming():
    # 2 points name a language in the prompt; 3 are needed to reject an answer with it
    assert detect_language("Is it latency?") == "English"
    assert detect_language("Is it latency?", min_score=3) is None


# --- 2. quoted source text is not the answer's language --------------------------

@pytest.mark.parametrize("answer", [
    'O slide "Relationship between Latency and Distance" mostra R² = 0,0177 [1].',
    'O objetivo é "assess the relationship between distance and delay in the context of vehicle mobility" [1].',
    "O slide “Influence of Link Conditions” mostra que o sinal pesa [3].",
])
def test_quoted_english_inside_portuguese_answer_passes(answer):
    assert check_language(answer, "Qual slide analisa a distância e a latência?") is None


def test_apostrophes_are_not_quotes():
    # "d'impact … l'effet" must not be paired up and masked as a quotation
    answer = "La distance a peu d'impact sur la latence, et l'effet du signal est fort [1]."
    assert check_language(answer, "Quels facteurs déterminent la latence dans le réseau ?") is None


# --- 3. the corpus hint needs a question KNOWN to differ -------------------------

EN_DOCS = [Document(page_content="The signal strength is one of the factors that influence the "
                                 "latency of the MEC server, and it is also related to the distance.",
                    metadata={"source": "slides.pdf", "page": 31})]


@pytest.mark.parametrize("question", [
    "Why did packet loss increase?", "Where were 5G tests conducted?", "Latency under mobility?",
])
def test_no_hint_when_question_language_is_unknown(patch_pipeline, fake_store, question):
    fake_store.docs = EN_DOCS
    chat = patch_pipeline(["NOT_FOUND", "q2", "NOT_FOUND", "q3", "NOT_FOUND"])
    build_graph().invoke({"question": question})
    assert "The documents are written in" not in chat.prompts[1]


def test_hint_for_non_latin_question(patch_pipeline, fake_store):
    fake_store.docs = EN_DOCS
    chat = patch_pipeline(["NOT_FOUND", "q2", "NOT_FOUND", "q3", "NOT_FOUND"])
    build_graph().invoke({"question": "距離は遅延を説明しますか？"})
    assert "The documents are written in English" in chat.prompts[1]


def test_language_failure_is_not_labelled_a_citation_failure(patch_pipeline):
    patch_pipeline(["根据提供的内容 [1]。", "距离不是主要因素 [1]。", "信号质量很重要 [1]。"])
    state = build_graph().invoke({"question": "Does distance explain the latency?"})
    assert "citation check" not in state["critique"]
    assert "pre-check failed" in state["critique"]


# --- third review: the "exclusive words" invariant, checked instead of assumed --

def test_function_word_sets_are_pairwise_disjoint():
    from src.language import _WORDS
    langs = list(_WORDS)
    for i, a in enumerate(langs):
        for b in langs[i + 1:]:
            assert not (_WORDS[a] & _WORDS[b]), f"{a} and {b} share {_WORDS[a] & _WORDS[b]}"


def test_repeated_das_no_longer_cancels_portuguese():
    # "das" voted for Portuguese AND German: 5 vs 3 here failed the 2x margin (None)
    assert detect_language("Qual é o efeito das falhas das antenas das redes?") == "Portuguese"
    # the review's example: without "das" only "qual" is left — unsure, and never German
    assert detect_language("Qual o impacto das falhas das redes 5G?") is None
