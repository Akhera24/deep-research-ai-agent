"""
B3.1 — page-verified highlight anchors.

A #:~:text= fragment only highlights if its text exists VERBATIM on the
target page. These tests pin the server-side anchor resolution: verbatim
match (case/whitespace/typographic-punctuation folded, returning the PAGE's
exact text), elided-quote longest-piece recovery, conservative paraphrase
sentence fallback, and the per-URL plumbing through Fact -> render.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from datetime import datetime

from src.extraction.extractor import FactExtractor, Fact
from src.search.executor import SearchResult
from src.reporting.html_report import render_html_report


def _result(url="https://ex.com/a", content=None, snippet="snip", reliability=0.5):
    return SearchResult(
        query="q", url=url, title="T", snippet=snippet, rank=1,
        search_engine="brave", fetched_at=datetime.now(), content=content,
        source_reliability=reliability,
    )


def _anchor(quote, fact_content, result):
    return FactExtractor._resolve_source_anchor(quote, fact_content, result)


# ── (a) verbatim match ──────────────────────────────────────────────────────

def test_exact_quote_found_in_content():
    r = _result(content="Intro text. Huang founded NVIDIA in 1993. More text.")
    assert _anchor("Huang founded NVIDIA in 1993", "fact", r) == \
        "Huang founded NVIDIA in 1993"


def test_match_returns_pages_own_punctuation():
    """LLM says straight quotes; the page uses curly ones — the anchor must
    be the PAGE's text or the browser match fails."""
    r = _result(content="He said “we're all in” at the — keynote event yesterday.")
    anchor = _anchor('He said "we\'re all in" at the - keynote', "fact", r)
    assert anchor == "He said “we're all in” at the — keynote"


def test_match_is_case_and_whitespace_insensitive():
    r = _result(content="NVIDIA   Founder\n\nand CEO   Jensen Huang spoke.")
    anchor = _anchor("nvidia founder and ceo jensen huang", "fact", r)
    assert anchor == "NVIDIA Founder and CEO Jensen Huang"  # collapsed


def test_snippet_used_when_content_missing():
    r = _result(content=None, snippet="Jensen Huang was born in Tainan, Taiwan in 1963.")
    assert _anchor("born in Tainan, Taiwan", "fact", r) == "born in Tainan, Taiwan"


# ── (b) elided quote: longest piece only ────────────────────────────────────

def test_elided_quote_recovers_longest_matching_piece():
    page = ("This securities fraud class action brings claims against NVIDIA. "
            "Other paragraphs. Its Chief Executive Officer made statements.")
    r = _result(content=page)
    anchor = _anchor(
        "This securities fraud class action brings claims against NVIDIA... "
        "and its Chief Executive", "fact", r)
    assert anchor == "This securities fraud class action brings claims against NVIDIA"


def test_elided_unicode_ellipsis_also_split():
    r = _result(content="The quarterly report showed record revenue this year.")
    anchor = _anchor("nothing matching here at all… record revenue this year", "fact", r)
    # shorter piece < 20 chars is skipped, so nothing verbatim; sentence
    # fallback may still fire — accept either the sentence or None, but
    # never the raw elided string
    assert anchor != "nothing matching here at all… record revenue this year"


# ── (c) paraphrase: conservative sentence fallback ──────────────────────────

def test_paraphrase_falls_back_to_best_overlap_sentence():
    page = ("Unrelated opening sentence about the weather. "
            "Jensen Huang and Lisa Su were both born in Tainan, Taiwan. "
            "Closing remarks about semiconductors.")
    r = _result(content=page)
    anchor = _anchor("they were both born in Tainan, Taiwan.",
                     "Jensen Huang was born in Tainan, Taiwan.", r)
    assert anchor == "Jensen Huang and Lisa Su were both born in Tainan, Taiwan."


def test_paraphrase_below_threshold_returns_none():
    r = _result(content="A page entirely about cooking recipes and gardens. "
                        "Nothing else relevant appears here at all today.")
    assert _anchor("quarterly earnings guidance raised", "revenue fact", r) is None


def test_sentence_fallback_never_uses_snippet():
    """Snippets are engine-generated summaries — their sentences may not
    exist on the page, so only (a)/(b) may match them."""
    r = _result(content=None,
                snippet="Jensen Huang and Lisa Su were both born in Tainan, Taiwan.")
    assert _anchor("completely different words", "born Tainan Taiwan Huang", r) is None


def test_empty_quote_and_no_content_returns_none():
    assert _anchor("", "fact", _result(content=None, snippet="")) is None
    assert _anchor(None, "fact", _result(content="text here")) is None


# ── plumbing: extraction -> Fact.anchor_texts ───────────────────────────────

def _extractor():
    return FactExtractor(router=None, enable_verification=False)


def test_convert_populates_per_url_anchors():
    r1 = _result(url="https://a.com/x",
                 content="Filler. Huang adopted a 10b5-1 trading plan in March 2024. End.")
    r2 = _result(url="https://b.com/y", content="Totally unrelated page about llamas and alpacas.")
    facts = _extractor()._convert_to_fact_objects(
        [{"content": "Huang adopted a 10b5-1 trading plan in March 2024.",
          "category": "financial", "confidence": 0.9,
          "evidence": "Huang adopted a 10b5-1 trading plan in March 2024",
          "source_ids": [1, 2]}],
        [r1, r2], "Jensen Huang", [r1, r2],
    )
    f = facts[0]
    assert f.anchor_texts.get("https://a.com/x") == \
        "Huang adopted a 10b5-1 trading plan in March 2024"
    assert "https://b.com/y" not in f.anchor_texts  # nothing verified there
    assert f.to_dict()["anchor_texts"] == f.anchor_texts  # serializes


def test_batch_fallback_gets_no_anchors():
    r = _result(content="Huang founded NVIDIA in 1993.")
    facts = _extractor()._convert_to_fact_objects(
        [{"content": "Huang founded NVIDIA in 1993.", "category": "professional",
          "confidence": 0.9, "evidence": "Huang founded NVIDIA in 1993"}],
        [r], "Jensen Huang", [r],
    )
    assert facts[0].anchor_texts == {}


def test_dedup_merge_unions_anchor_texts():
    a = Fact(content="Sarah Chen has served as the chief executive officer of TechCorp since 2018",
             category="professional", confidence=0.9,
             anchor_texts={"https://a.com": "served as CEO since 2018"})
    b = Fact(content="Sarah Chen has reportedly served as the chief executive officer of TechCorp since 2018",
             category="professional", confidence=0.8,
             anchor_texts={"https://b.com": "chief executive officer of TechCorp"})
    unique = _extractor()._deduplicate_facts([a, b])
    assert len(unique) == 1
    assert unique[0].anchor_texts == {
        "https://a.com": "served as CEO since 2018",
        "https://b.com": "chief executive officer of TechCorp",
    }


# ── render: per-URL anchors drive the fragments ─────────────────────────────

def test_each_chip_gets_its_own_pages_anchor():
    fact = {
        "content": "Subject Person adopted a trading plan in March 2024",
        "category": "financial", "confidence": 0.9,
        "source_urls": ["https://a.com/x", "https://b.com/y"],
        "evidence": ["the generic fact-level quote"],
        "source_reliabilities": {},
        "anchor_texts": {"https://a.com/x": "anchor text from page A",
                         "https://b.com/y": "different anchor from page B"},
        "verified": False, "verification_count": 1, "entities_mentioned": [],
    }
    html = render_html_report(
        {"facts": [fact], "risk_flags": [], "connections": [],
         "metadata": {"coverage": {"average": 0.5}, "iterations": 1}},
        "Subject Person", 1.0)
    assert "anchor%20text%20from%20page%20A" in html
    assert "different%20anchor%20from%20page%20B" in html
    assert "generic%20fact%2Dlevel%20quote" not in html


def test_url_without_anchor_falls_back_to_fact_quote():
    fact = {
        "content": "Subject Person did a notable thing recently",
        "category": "professional", "confidence": 0.9,
        "source_urls": ["https://a.com/x"],
        "evidence": ["the fact level quote"],
        "source_reliabilities": {}, "anchor_texts": {},
        "verified": False, "verification_count": 1, "entities_mentioned": [],
    }
    html = render_html_report(
        {"facts": [fact], "risk_flags": [], "connections": [],
         "metadata": {"coverage": {"average": 0.5}, "iterations": 1}},
        "Subject Person", 1.0)
    assert "#:~:text=the%20fact%20level%20quote" in html
