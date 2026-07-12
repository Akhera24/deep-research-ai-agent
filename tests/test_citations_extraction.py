"""
Phase B0 — per-fact provenance at extraction.

Covers PLAN.md Phase B edge cases: numbered source blocks aligned to the
8000-char prompt budget (M5), per-id validation that drops bad ids but keeps
good ones (M4), batch fallback ONLY when no valid id remains, and the
source_reliabilities plumbing for B4 tier badges.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from datetime import datetime

from src.extraction.extractor import (
    FactExtractor,
    Fact,
    EXTRACTION_TEXT_BUDGET,
    MAX_EXTRACTION_SOURCES,
)
from src.search.executor import SearchResult


def _result(i, snippet="snippet", content=None, reliability=0.5):
    return SearchResult(
        query="q",
        url=f"https://example{i}.com/page",
        title=f"Title {i}",
        snippet=snippet,
        rank=i,
        search_engine="brave",
        fetched_at=datetime.now(),
        content=content,
        source_reliability=reliability,
    )


def _extractor():
    return FactExtractor(router=None, enable_verification=False)


# ── _prepare_text_for_extraction: numbering + budget (M5) ──────────────────

def test_sources_are_numbered_sequentially():
    ex = _extractor()
    results = [_result(i) for i in range(1, 4)]
    text, fed = ex._prepare_text_for_extraction(results)
    assert fed == results
    for n in (1, 2, 3):
        assert f"[Source {n}] Title {n}" in text


def test_budget_truncates_at_block_boundary():
    """Oversized batch: numbering stops at the block that no longer fits —
    a source the model never saw is never numbered."""
    ex = _extractor()
    # Each block ≈ 3000 chars (content is capped at 500, so use the snippet)
    # → only 2 fit in the 8000 budget
    results = [_result(i, snippet="x" * 3000) for i in range(1, 6)]
    text, fed = ex._prepare_text_for_extraction(results)
    assert len(text) <= EXTRACTION_TEXT_BUDGET
    assert len(fed) == 2
    assert "[Source 2]" in text
    assert "[Source 3]" not in text


def test_single_oversized_first_block_is_truncated_not_empty():
    ex = _extractor()
    results = [_result(1, snippet="y" * 20000)]
    text, fed = ex._prepare_text_for_extraction(results)
    assert len(fed) == 1
    assert text.startswith("[Source 1]")
    assert len(text) == EXTRACTION_TEXT_BUDGET


def test_max_sources_cap_still_applies():
    ex = _extractor()
    results = [_result(i, snippet="s") for i in range(1, 40)]
    _, fed = ex._prepare_text_for_extraction(results)
    assert len(fed) <= MAX_EXTRACTION_SOURCES


def test_empty_results_yield_empty_feed():
    ex = _extractor()
    text, fed = ex._prepare_text_for_extraction([])
    assert text == ""
    assert fed == []


# ── _validate_source_ids: per-id drop, keep the good ones (M4) ─────────────

def test_valid_subset_kept_when_siblings_invalid():
    valid = FactExtractor._validate_source_ids([1, 99, 0, 3, -2], num_fed=5)
    assert valid == [1, 3]


def test_non_int_ids_dropped_digit_strings_accepted():
    valid = FactExtractor._validate_source_ids(
        [1, "2", "abc", 2.5, None, True, [4]], num_fed=5
    )
    # True is bool (would silently mean Source 1) — dropped; "2" accepted
    assert valid == [1, 2]


def test_duplicate_ids_deduped_order_preserved():
    assert FactExtractor._validate_source_ids([3, 1, 3, 1], num_fed=5) == [3, 1]


def test_missing_empty_or_non_list_yield_empty():
    assert FactExtractor._validate_source_ids(None, 5) == []
    assert FactExtractor._validate_source_ids([], 5) == []
    assert FactExtractor._validate_source_ids("1,2", 5) == []
    assert FactExtractor._validate_source_ids({"a": 1}, 5) == []


# ── _convert_to_fact_objects: resolution + fallback ────────────────────────

def _fact_data(**overrides):
    data = {
        "content": "Sarah Chen is CEO of TechCorp",
        "category": "professional",
        "confidence": 0.9,
        "evidence": "TechCorp website states: Sarah Chen, CEO",
    }
    data.update(overrides)
    return data


def test_valid_ids_resolve_to_per_fact_urls():
    ex = _extractor()
    results = [_result(i, reliability=0.1 * i) for i in range(1, 6)]
    facts = ex._convert_to_fact_objects(
        [_fact_data(source_ids=[2, 4])], results, "Sarah Chen", results
    )
    assert len(facts) == 1
    f = facts[0]
    assert f.source_ids == [2, 4]
    assert f.source_urls == [results[1].url, results[3].url]
    # B4 plumbing: per-source reliability travels with the URLs
    assert f.source_reliabilities == {
        results[1].url: results[1].source_reliability,
        results[3].url: results[3].source_reliability,
    }


def test_partial_invalid_ids_keep_valid_subset_not_batch():
    ex = _extractor()
    results = [_result(i) for i in range(1, 6)]
    facts = ex._convert_to_fact_objects(
        [_fact_data(source_ids=[99, 2])], results, "Sarah Chen", results
    )
    assert facts[0].source_urls == [results[1].url]  # NOT all 5 (M4)


def test_missing_ids_fall_back_to_batch():
    ex = _extractor()
    results = [_result(i) for i in range(1, 6)]
    facts = ex._convert_to_fact_objects(
        [_fact_data()], results, "Sarah Chen", results
    )
    assert facts[0].source_ids == []
    assert facts[0].source_urls == [r.url for r in results]
    assert set(facts[0].source_reliabilities) == {r.url for r in results}


def test_all_invalid_ids_fall_back_to_batch():
    ex = _extractor()
    results = [_result(i) for i in range(1, 4)]
    facts = ex._convert_to_fact_objects(
        [_fact_data(source_ids=[0, 99, "x"])], results, "Sarah Chen", results
    )
    assert facts[0].source_urls == [r.url for r in results]


def test_ids_validate_against_fed_slice_not_full_batch():
    """20 results but only 2 fed: id 3 is out of range even though
    search_results[2] exists (N2 slice alignment)."""
    ex = _extractor()
    results = [_result(i) for i in range(1, 21)]
    fed = results[:2]
    facts = ex._convert_to_fact_objects(
        [_fact_data(source_ids=[3])], results, "Sarah Chen", fed
    )
    # No valid id in the fed range → batch fallback
    assert facts[0].source_urls == [r.url for r in results]


def test_malformed_source_ids_never_crash():
    ex = _extractor()
    results = [_result(1)]
    for bad in (None, "1", 3.7, {"id": 1}, [[1]], [None], [""], [True]):
        facts = ex._convert_to_fact_objects(
            [_fact_data(source_ids=bad)], results, "Sarah Chen", results
        )
        assert len(facts) == 1
        assert facts[0].source_urls == [results[0].url]


def test_duplicate_urls_across_ids_deduped():
    ex = _extractor()
    r1, r2 = _result(1), _result(1)  # same URL twice
    facts = ex._convert_to_fact_objects(
        [_fact_data(source_ids=[1, 2])], [r1, r2], "Sarah Chen", [r1, r2]
    )
    assert facts[0].source_urls == [r1.url]


# ── dedup merge unions provenance ───────────────────────────────────────────

def test_dedup_merge_unions_source_urls_and_reliabilities():
    ex = _extractor()
    a = Fact(content="Sarah Chen has served as the chief executive officer of TechCorp since 2018",
             category="professional", confidence=0.9,
             source_urls=["https://a.com"],
             source_reliabilities={"https://a.com": 0.9})
    b = Fact(content="Sarah Chen has reportedly served as the chief executive officer of TechCorp since 2018",
             category="professional", confidence=0.8,
             source_urls=["https://b.com"],
             source_reliabilities={"https://b.com": 0.4})
    unique = ex._deduplicate_facts([a, b])
    assert len(unique) == 1
    assert set(unique[0].source_urls) == {"https://a.com", "https://b.com"}
    assert unique[0].source_reliabilities == {
        "https://a.com": 0.9, "https://b.com": 0.4
    }


# ── serialization: new fields survive to the report dict ────────────────────

def test_to_dict_carries_provenance_fields():
    f = Fact(content="c", category="professional", confidence=0.9,
             source_urls=["https://a.com"], source_ids=[1],
             source_reliabilities={"https://a.com": 0.8})
    d = f.to_dict()
    assert d["source_ids"] == [1]
    assert d["source_reliabilities"] == {"https://a.com": 0.8}


# ── prompt contract ─────────────────────────────────────────────────────────

def test_prompt_asks_for_source_ids():
    ex = _extractor()
    prompt = ex._build_extraction_prompt("[Source 1] t\ns", "Sarah Chen")
    assert "source_ids" in prompt
    assert "[Source 1]" in prompt
