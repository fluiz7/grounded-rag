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

# Function words: frequent, topic-independent. Overlaps (de, la, en) only add noise
# to several languages at once; the decision needs a clear winner.
_FUNCTION_WORDS = {
    "Portuguese": "o a os as de do da dos das em no na nos nas um uma com por para que não é e "
                  "se mais como qual quais entre sua seu ao à pela pelo são também foi",
    "English": "the of and to in is are was what which does do how with for on by not that "
               "this between an it be from also",
    "Spanish": "el la los las de del en y que es con por para un una no qué cuál cómo entre lo "
               "su son también fue",
    "German": "der die das und ist nicht mit von zu den dem ein eine wird durch auf für im "
              "sich auch wie",
    "French": "le la les de des du et est un une que qui pas pour avec sur dans quel quels "
              "aussi été",
}
_WORDS = {lang: set(words.split()) for lang, words in _FUNCTION_WORDS.items()}
# letters that occur in one of these languages only
_LETTERS = {"Portuguese": "ãõ", "Spanish": "ñ¿¡", "German": "ßäöü", "French": "èùœë"}


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


def detect_language(text: str) -> str | None:
    """English name of the language of a Latin-script `text`, or None when unsure.

    Unsure = fewer than 2 points of evidence, or a tie: a wrong guess would make the
    pipeline reject correct answers, so silence is the safe answer.
    """
    if dominant_script(text) != "LATIN":
        return None
    tokens = re.findall(r"[^\W\d_]+", text.casefold())
    scores = {lang: sum(t in words for t in tokens) for lang, words in _WORDS.items()}
    for lang, letters in _LETTERS.items():
        scores[lang] += 2 * sum(text.casefold().count(c) for c in letters)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    return best if top >= 2 and top > second else None
