"""Deterministic language hints — no model call, no dependency.

A 7B model follows "write in Portuguese" far better than "write in the QUESTION's language"
(measured: the latter produced English and even German answers to Portuguese questions).
So the language is detected here and NAMED in the prompts, and the answer is checked
against it. When the evidence is weak the detector says None and nothing is enforced.
"""
import re
import unicodedata

# Japanese mixes kanji and kana: one script for this purpose
_SCRIPT_GROUPS = {"HIRAGANA": "CJK", "KATAKANA": "CJK"}
_SCRIPTS = ("LATIN", "CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "GREEK", "ARABIC",
            "HEBREW", "DEVANAGARI", "THAI")

# Function words EXCLUSIVE to one language: frequent and topic-independent. A word that
# is a function word in two of them ("a", "as", "no", "de", "que", "se", "mais", "was",
# "in", "an", "um", "à", "das"…) is left out everywhere — it would vote for the wrong language
# ("Does a handover…" read as Portuguese). The test suite pins the collisions.
_FUNCTION_WORDS = {
    "Portuguese": "os da dos em na nas uma com não é qual quais sua seu ao pela pelo são "
                  "também foi isso essa esse onde muito há pelos pelas quando",
    "English": "the of and to is are were what which does did how why where who with for by "
               "not that this between it be from also has have can at or its these",
    "Spanish": "el los las del y es con una qué cuál cómo lo su son también fue muy pero "
               "está hay esto cuando",
    "German": "der die und ist nicht mit von zu den dem ein eine wird durch auf für im "
              "sich auch wie bei werden oder zwischen nach",
    "French": "le les des du et est une qui pas pour avec sur dans quel quels aussi été ce "
              "cette sont au aux ces",
}
_WORDS = {lang: set(words.split()) for lang, words in _FUNCTION_WORDS.items()}
# letters that occur in one of these languages only — counted in lowercase words only:
# "São Paulo" and "Fundão" are names an English answer legitimately contains
_LETTERS = {"Portuguese": "ãõ", "Spanish": "ñ", "German": "ßäöü", "French": "èùœë"}
# accented endings no other of these languages writes ("latência" vs latence/latencia/
# latency): technical questions are mostly content words, and these carry the signal
_SUFFIXES = {"Portuguese": ("ência", "ância", "ção", "ções", "ões"), "Spanish": ("ción", "ciones")}


def dominant_script(text: str) -> str | None:
    """The writing system most letters of `text` belong to (LATIN, CJK, CYRILLIC, …)."""
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            script = next((s for s in _SCRIPTS if name.startswith(s)), None)
            if script:
                script = _SCRIPT_GROUPS.get(script, script)
                counts[script] = counts.get(script, 0) + 1
    return max(counts, key=counts.get) if counts else None


def detect_language(text: str, min_score: int = 2) -> str | None:
    """English name of the language of a Latin-script `text`, or None when unsure.

    Unsure = fewer than `min_score` points, or a winner without twice the runner-up's
    points. Naming the language in a prompt can use the default; REJECTING an answer
    should ask for more (`min_score=3`): a wrong guess there blocks a correct answer.
    """
    if dominant_script(text) != "LATIN":
        return None
    tokens = re.findall(r"[^\W\d_]+", text)
    scores = {lang: sum(t.casefold() in words for t in tokens) for lang, words in _WORDS.items()}
    lowercase = "".join(t for t in tokens if not t[0].isupper())
    for lang, letters in _LETTERS.items():
        scores[lang] += sum(lowercase.count(c) for c in letters)
    for lang, endings in _SUFFIXES.items():
        scores[lang] += sum(t.casefold().endswith(endings) for t in tokens)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    return best if top >= min_score and top >= 2 * second else None
