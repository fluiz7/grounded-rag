"""Regression tests for the fourth adversarial review of PR #1 (review of commit 1606462).

Every case reproduced against that commit before being fixed.
"""
import pytest

from src.graph import check_citations, cited_sources, parse_verdict

# --- 1. short legitimate answers are not "only citations" -------------------------

@pytest.mark.parametrize("text", ["No [1].", "42 [1].", "5% [1].", "R2 [1].", "OK [1]."])
def test_short_answers_pass(text):
    assert check_citations(text, 2) is None


@pytest.mark.parametrize("text", ["[1]", "[1], [2].", " [1] [2] "])
def test_citation_only_answers_still_rejected(text):
    problem = check_citations(text, 2)
    assert problem is not None and "only citations" in problem


# --- 2. fenced code blocks are not read as citations ------------------------------

@pytest.mark.parametrize("text", [
    "Use the first element as shown in [1]:\n```python\nx = items[99]\n```",
    "As in [1]:\n~~~\nitems[99]\n~~~",
    "As in [1]:\n```\nitems[99]",                    # unterminated fence runs to the end
])
def test_fenced_code_is_ignored(text):
    assert cited_sources(text) == [1]
    assert check_citations(text, 2) is None


def test_citation_after_a_fence_still_counts():
    assert cited_sources("A [1].\n```\nx[5]\n```\nB [7].") == [1, 7]


# --- 3. a year does not hide a phantom citation in the same group -----------------

def test_year_does_not_hide_phantom():
    text = "Sales grew 20% [1] and peaked in [99, 2024]."
    assert cited_sources(text) == [1, 99]
    assert "[99]" in check_citations(text, 2)


def test_groups_with_zero_stay_intervals():
    # kept on purpose: [0, 99] is an interval/index, filtering per element would turn
    # "the range [0, 5]" into a citation of a source that does not exist
    assert cited_sources("In the range [0, 5] it holds [1].") == [1]


# --- 4. "true" is a positive legacy verdict, as the prompt itself asks for --------

@pytest.mark.parametrize("text", [
    "GROUNDED: true\nUNSUPPORTED: none\nQUERY:",
    'Grounded: true\n{"grounded": true, "unsupported_claims": [], "query": ""}',
    "**GROUNDED:** True.",
])
def test_true_is_grounded(text):
    assert parse_verdict(text)[0] is True


@pytest.mark.parametrize("text", [
    "GROUNDED: false",
    'Grounded: true\n{"grounded": false, "unsupported_claims": ["x"], "query": ""}',
    "GROUNDED: true, mostly",
])
def test_fail_closed_still_holds(text):
    assert parse_verdict(text)[0] is False
