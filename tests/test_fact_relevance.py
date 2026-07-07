"""
Regression: qualified queries (name + company) must not discard every fact.

Bug: _post_process_facts required the whole target string to appear
contiguously in a fact, so "phil gallagher avnet" matched no real sentence
("Phil Gallagher ... CEO of Avnet") and dropped all facts -> empty report.
Maps to PLAN.md edge cases #4 (query format) and #5 (harder subjects).
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.extraction.extractor import FactExtractor, Fact


def _facts(*contents):
    return [Fact(content=c, category="professional", confidence=1.0) for c in contents]


def _keep(extractor, facts, target):
    return extractor._post_process_facts(facts, search_results=[], target_name=target)


def test_qualified_query_keeps_facts():
    """The exact reported failure: 'phil gallagher avnet' retained facts."""
    ex = FactExtractor(router=None)
    facts = _facts(
        "Phil Gallagher has been the Chief Executive Officer of Avnet since November 2020.",
        "Gallagher holds a bachelor's degree from Drexel University.",
        "Phil Gallagher's total compensation as CEO in 2024 was $8,775,978.",
    )
    kept = _keep(ex, facts, "phil gallagher avnet")
    assert len(kept) == 3  # was 0 before the fix


def test_plain_name_query_unchanged():
    """Single-name queries that already worked still work."""
    ex = FactExtractor(router=None)
    facts = _facts(
        "Jensen Huang co-founded NVIDIA in 1993.",
        "Jensen Huang was born in Taiwan.",
    )
    assert len(_keep(ex, facts, "Jensen Huang")) == 2


def test_cross_entity_noise_still_filtered():
    """A fact mentioning none of the target's tokens is still dropped."""
    ex = FactExtractor(router=None)
    facts = _facts(
        "Phil Gallagher leads Avnet.",                 # keep
        "The weather in Phoenix was sunny that day.",  # drop (no target token)
    )
    kept = _keep(ex, facts, "phil gallagher avnet")
    assert len(kept) == 1
    assert "Gallagher" in kept[0].content


def test_surname_only_fact_kept():
    """Facts that use only the surname are common and must survive."""
    ex = FactExtractor(router=None)
    facts = _facts("Gallagher received the North Star Award from ECIA.")
    assert len(_keep(ex, facts, "phil gallagher avnet")) == 1
