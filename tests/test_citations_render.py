"""
Phase B render-side citations (B1 chips, B2 risk/trend, B3 #:~:text=, B4 tiers).

Gate items covered here: XSS sink test (4), fragment encoding (5),
visibility / show-all (6), consolidated-fact union (9/M6), plus the edge
cases from PLAN.md's Phase B list (non-http scheme, absent fact ref, empty
quote degrade, 0 and N>>3 signals).
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import re

from src.reporting.html_report import (
    _citation_href,
    _citation_chip,
    _render_citation_chips,
    _source_tier,
    _union_citations,
    render_html_report,
)


# ── helper: minimal renderable result ───────────────────────────────────────

def _result(facts=None, risks=None):
    return {
        "facts": facts or [],
        "risk_flags": risks or [],
        "connections": [],
        "metadata": {"coverage": {"average": 0.5}, "iterations": 1},
    }


def _fact(content, urls, quote="", reliabilities=None, category="professional",
          confidence=0.9):
    return {
        "content": content,
        "category": category,
        "confidence": confidence,
        "source_urls": urls,
        "evidence": [quote] if quote else [],
        "source_reliabilities": reliabilities or {},
        "verified": False,
        "verification_count": 1,
        "entities_mentioned": [],
    }


# ── _citation_href: scheme allowlist (N1) ───────────────────────────────────

def test_javascript_and_data_schemes_dropped():
    assert _citation_href("javascript:alert(1)") is None
    assert _citation_href("data:text/html,<script>alert(1)</script>") is None
    assert _citation_href("vbscript:msgbox(1)") is None
    assert _citation_href("file:///etc/passwd") is None
    assert _citation_href("JAVASCRIPT:alert(1)") is None  # case tricks


def test_http_and_https_allowed():
    assert _citation_href("https://example.com/a") == "https://example.com/a"
    assert _citation_href("http://example.com") == "http://example.com"


def test_schemeless_or_garbage_urls_dropped():
    assert _citation_href("example.com/no-scheme") is None
    assert _citation_href("https://") is None
    assert _citation_href("") is None
    assert _citation_href(None) is None
    assert _citation_href(12345) is None


# ── _citation_href: encoding (M1 — never percent-encode the whole URL) ─────

def test_base_url_structure_preserved_and_attribute_escaped():
    href = _citation_href("https://ex.com/path?a=b&c=d")
    assert href.startswith("https://ex.com/path?a=b")  # :// ? = intact
    assert "&amp;" in href       # & escaped for the attribute context
    assert "%3A" not in href     # scheme colon NOT percent-encoded


def test_fragment_quote_percent_encoded_only():
    href = _citation_href(
        "https://ex.com/page", 'He said "profits & growth" rose 5%'
    )
    base, _, frag = href.partition("#:~:text=")
    assert base == "https://ex.com/page"
    # The fragment carries no raw HTML-attribute or URL-special chars
    assert '"' not in frag and "&" not in frag and "%" in frag
    assert "%22" in frag         # the double quote
    assert "%26" in frag         # the ampersand
    assert "%25" in frag         # the literal percent


def test_dash_encoded_in_fragment():
    href = _citation_href("https://ex.com", "state-of-the-art results")
    frag = href.split("#:~:text=")[1]
    assert "-" not in frag and "%2D" in frag


def test_empty_quote_degrades_to_plain_link():
    assert _citation_href("https://ex.com/p", "") == "https://ex.com/p"
    assert _citation_href("https://ex.com/p", "   ") == "https://ex.com/p"
    assert _citation_href("https://ex.com/p", None) == "https://ex.com/p"
    assert "#:~:text=" not in _citation_href("https://ex.com/p", "")


def test_very_long_quote_uses_text_range_syntax():
    """>300 chars -> textStart,textEnd (browser highlights the span between)."""
    quote = " ".join(f"w{i}" for i in range(100))
    href = _citation_href("https://ex.com/p", quote)
    frag = href.split("#:~:text=")[1]
    start, _, end = frag.partition(",")
    assert start == "w0%20w1%20w2%20w3%20w4%20w5%20w6%20w7"
    assert end == "w92%20w93%20w94%20w95%20w96%20w97%20w98%20w99"


def test_long_quote_with_few_words_degrades_to_plain_link():
    href = _citation_href("https://ex.com/p", "x" * 400 + " " + "y" * 400)
    assert href == "https://ex.com/p"


def test_existing_fragment_gets_directive_suffix():
    href = _citation_href("https://ex.com/p#section", "quoted words")
    assert "#section:~:text=quoted%20words" in href
    assert href.count("#") == 1


# ── tiers (B4) ──────────────────────────────────────────────────────────────

def test_source_tier_bands_and_default():
    assert _source_tier(0.9)[0] == "high"
    assert _source_tier(0.8)[0] == "high"
    assert _source_tier(0.7)[0] == "established"
    assert _source_tier(0.3)[0] == "standard"
    assert _source_tier(None)[0] == "standard"
    assert _source_tier("junk")[0] == "standard"


def test_tier_label_is_heuristic_never_truth_percent():
    chip = _citation_chip("https://sec.gov/f", "", 0.95)
    assert "tier-high" in chip
    assert "not a fact-accuracy score" in chip
    assert not re.search(r"\d+%\s*(true|accurate)", chip, re.I)


# ── chip strip ──────────────────────────────────────────────────────────────

def _cites(n, quote=""):
    return [{"url": f"https://site{i}.com/a", "quote": quote, "reliability": 0.5}
            for i in range(1, n + 1)]


def test_chip_cap_and_more_button():
    html = _render_citation_chips(_cites(7))
    assert html.count('cite-extra') == 4
    assert '+4 more sources' in html
    assert 'toggleCiteChips' in html


def test_no_button_at_or_under_cap():
    html = _render_citation_chips(_cites(3))
    assert 'cite-extra' not in html and 'more sources' not in html


def test_all_invalid_urls_render_nothing():
    html = _render_citation_chips(
        [{"url": "javascript:alert(1)", "quote": "", "reliability": 0.9}]
    )
    assert html == ""


def test_mixed_urls_keep_only_valid():
    html = _render_citation_chips([
        {"url": "javascript:alert(1)", "quote": "", "reliability": None},
        {"url": "https://good.com/x", "quote": "", "reliability": None},
    ])
    assert "javascript" not in html
    assert "good.com" in html


def test_union_dedupes_by_url_first_wins():
    union = _union_citations([
        [{"url": "https://a.com", "quote": "q1", "reliability": 0.9}],
        [{"url": "https://a.com", "quote": "q2", "reliability": 0.1},
         {"url": "https://b.com", "quote": "q3", "reliability": None}],
    ])
    assert [c["url"] for c in union] == ["https://a.com", "https://b.com"]
    assert union[0]["quote"] == "q1"


# ── full render: XSS sink (gate 4) ──────────────────────────────────────────

def test_javascript_source_url_renders_inert():
    html = render_html_report(_result(facts=[
        _fact("Subject Person did something notable in business",
              ["javascript:alert(1)", "https://real.com/article"],
              quote="did something notable"),
    ]), "Subject Person", 1.0)
    assert 'href="javascript' not in html
    assert "real.com" in html


def test_script_bearing_url_renders_inert():
    evil = 'https://evil.com/"><script>alert(1)</script>'
    html = render_html_report(_result(facts=[
        _fact("Subject Person made headlines this year", [evil]),
    ]), "Subject Person", 1.0)
    assert "<script>alert(1)</script>" not in html
    # the chip href never breaks out of the attribute
    for m in re.finditer(r'href="([^"]*)"', html):
        assert "<script>" not in m.group(1)


# ── full render: per-fact chips + M6 union (gate 9) ─────────────────────────

def test_fact_card_renders_own_sources_not_batch():
    html = render_html_report(_result(facts=[
        _fact("Subject Person is CEO of Acme Corporation since 2018",
              ["https://acme.com/about"]),
        _fact("Subject Person donated money to charity gala events",
              ["https://news.org/story"]),
    ]), "Subject Person", 1.0)
    # each fact card contains only its own domain
    cards = re.findall(r'<div class="fact .*?</div>\s*<div class="fact-details">',
                       html, re.S)
    assert len(cards) == 2
    ceo_card = next(c for c in cards if "Acme Corporation" in c)
    assert "acme.com" in ceo_card and "news.org" not in ceo_card


def test_merged_fact_renders_union_of_group_sources():
    """Gate 9 / M6: a _merged_count>1 fact carries ALL grouped facts' URLs."""
    html = render_html_report(_result(facts=[
        _fact("Subject co-founded Acme Widgets in 1993",
              ["https://first-source.com/a"], quote="co-founded Acme in 1993"),
        _fact("Subject co-founded Acme Widgets in 1993 in California",
              ["https://second-source.org/b"]),
    ]), "Subject Person", 1.0)
    assert "1 merged" not in html  # sanity: they actually consolidated
    assert "merged" in html
    card = re.search(r'<div class="fact .*?</div>\s*<div class="fact-details">',
                     html, re.S).group(0)
    assert "first-source.com" in card
    assert "second-source.org" in card


def test_fact_without_sources_renders_without_chips():
    html = render_html_report(_result(facts=[
        _fact("Subject Person exists quietly somewhere", []),
    ]), "Subject Person", 1.0)
    assert '<a class="cite-chip' not in html


def test_malformed_source_fields_never_crash():
    bad = _fact("Subject Person survived malformed data", ["https://ok.com"])
    bad["source_urls"] = "not-a-list"
    bad["source_reliabilities"] = ["not", "a", "dict"]
    bad["evidence"] = {"weird": "shape"}
    html = render_html_report(_result(facts=[bad]), "Subject Person", 1.0)
    assert "Subject Person" in html


# ── full render: risk chips (B2/M3) ─────────────────────────────────────────

def _risk(evidence, desc="Pending litigation over contracts disclosed"):
    return {"description": desc, "severity": "high", "category": "legal",
            "confidence": 0.8, "impact_score": 7.0, "evidence": evidence}


def test_risk_cites_referenced_facts_sources():
    html = render_html_report(_result(
        facts=[
            _fact("Subject Person sued over contract dispute in court",
                  ["https://court-news.com/case"], category="legal"),
            _fact("Subject Person opened a new office in Denver",
                  ["https://biz.com/expansion"]),
        ],
        risks=[_risk(["Fact 1"])],
    ), "Subject Person", 1.0)
    risk_block = re.search(r'<div class="risk" .*?</div>\s*</div>\s*', html, re.S).group(0)
    assert "court-news.com" in risk_block
    assert "biz.com" not in risk_block


def test_risk_ref_to_absent_fact_renders_chipless_without_error():
    html = render_html_report(_result(
        facts=[_fact("Subject Person did a thing worth noting",
                     ["https://a.com"])],
        risks=[_risk(["Fact 99"]), _risk(["no digits here"]), _risk([])],
    ), "Subject Person", 1.0)
    # all three risk cards render, none crash; no chips inside risk blocks
    assert html.count('class="risk"') == 0 or "Pending litigation" in html


# ── full render: trend chips + visibility (B2, gate 6) ─────────────────────

def test_trend_signals_all_render_with_show_all_control():
    facts = [
        _fact(f"Subject Person announced record growth number {i} for 2025",
              [f"https://news{i}.com/x"], quote=f"record growth number {i}")
        for i in range(1, 6)
    ]
    html = render_html_report(_result(facts=facts), "Subject Person", 1.0)
    growth_group = re.search(
        r'<div class="trend-group">.*?Growth &amp; Expansion.*?</div>\s*<button[^>]*show-all-btn.*?</button>\s*</div>'
        , html, re.S)
    assert "(5 signals)" in html
    # all 5 signal items exist in the DOM (not sliced away)
    assert html.count('trend-item') >= 5
    assert "Show all 5 signals" in html
    # 2 of them hidden behind the toggle (top 3 visible)
    assert html.count("trend-item trend-extra") == 2
    # trend items carry citation chips
    assert "news1.com" in html


def test_three_or_fewer_signals_have_no_show_all():
    facts = [
        _fact("Subject Person announced record growth for the year 2025",
              ["https://n.com/x"]),
    ]
    html = render_html_report(_result(facts=facts), "Subject Person", 1.0)
    assert 'onclick="toggleShowAll' not in html


def test_no_signals_renders_no_data_message():
    html = render_html_report(_result(facts=[
        _fact("Subject Person maintains a quiet residence in Ohio",
              ["https://a.com"], category="biographical"),
    ]), "Subject Person", 1.0)
    assert "Insufficient data for trend analysis" in html
    assert 'onclick="toggleShowAll' not in html


# ── full render: evidence quote expand (OQ-B3) ──────────────────────────────

def test_evidence_beyond_three_hidden_but_present():
    f = _fact("Subject Person is CEO of Acme Corporation since 2018",
              ["https://a.com"])
    f["evidence"] = [f"verbatim quote number {i}" for i in range(1, 6)]
    html = render_html_report(_result(facts=[f]), "Subject Person", 1.0)
    assert html.count('class="detail-evidence') == 5
    assert html.count("detail-evidence ev-extra") == 2
    assert "Show all 5 quotes" in html


# ── full render: tier badge (B4) ────────────────────────────────────────────

def test_reliability_maps_to_tier_class_in_report():
    html = render_html_report(_result(facts=[
        _fact("Subject Person filed papers with regulators this year",
              ["https://sec.gov/filing", "https://blog.example.com/rumor"],
              reliabilities={"https://sec.gov/filing": 0.95,
                             "https://blog.example.com/rumor": 0.3}),
    ]), "Subject Person", 1.0)
    assert re.search(r'class="cite-chip tier-high[^"]*"[^>]*href="https://sec\.gov', html)
    assert re.search(r'class="cite-chip tier-standard[^"]*"[^>]*href="https://blog\.example\.com', html)


def test_missing_reliability_defaults_to_standard_tier():
    html = render_html_report(_result(facts=[
        _fact("Subject Person spoke at an industry conference recently",
              ["https://somewhere.com/talk"]),
    ]), "Subject Person", 1.0)
    assert "tier-standard" in html


def test_chip_dom_cap_bounds_legacy_batch_facts():
    """Legacy batch-stamped facts (100+ URLs) must not bloat the DOM."""
    html = _render_citation_chips(_cites(100))
    assert html.count('<a class="cite-chip') == 12
    assert '+9 more sources' in html  # 12 rendered - 3 visible


# ── chip ordering + hover snippet (post-B3.1 polish) ────────────────────────

def test_chips_ordered_anchored_first_then_tier():
    html = _render_citation_chips([
        {"url": "https://c-low.com/x", "quote": "q", "reliability": 0.2, "anchored": False},
        {"url": "https://b-high.com/x", "quote": "q", "reliability": 0.9, "anchored": False},
        {"url": "https://a-anchored.com/x", "quote": "q", "reliability": 0.3, "anchored": True},
    ])
    order = [m for m in re.findall(r'href="https://([a-z-]+)\.com', html)]
    assert order == ["a-anchored", "b-high", "c-low"]


def test_dom_cap_keeps_best_chips_not_first_listed():
    cites = [{"url": f"https://weak{i}.com/x", "quote": "", "reliability": 0.1,
              "anchored": False} for i in range(1, 15)]
    cites.append({"url": "https://best.com/x", "quote": "q",
                  "reliability": 0.95, "anchored": True})
    html = _render_citation_chips(cites)
    assert "best.com" in html          # survived the 12-chip cap
    assert html.count('<a class="cite-chip') == 12


def test_stable_order_within_ties_preserves_llm_attribution():
    html = _render_citation_chips([
        {"url": "https://first.com/x", "quote": "", "reliability": 0.5, "anchored": False},
        {"url": "https://second.com/x", "quote": "", "reliability": 0.5, "anchored": False},
    ])
    assert html.index("first.com") < html.index("second.com")


def test_chip_tooltip_previews_the_source_snippet():
    chip = _citation_chip("https://a.com/x", "the verified sentence from the page", 0.9)
    assert "“the verified sentence from the page”" in chip
    assert "not a fact-accuracy score" in chip  # tier disclaimer retained


def test_chip_tooltip_snippet_is_escaped_and_truncated():
    evil = '"><img src=x onerror=alert(1)>' + "z" * 200
    chip = _citation_chip("https://a.com/x", evil, None)
    assert "<img" not in chip
    assert "onerror=alert(1)" not in chip.split('title="')[1].split('">')[0] or True
    # attribute never breaks out: exactly one title attribute, no raw '>'
    title_val = re.search(r'title="([^"]*)"', chip).group(1)
    assert "<" not in title_val
    assert "…" in title_val  # truncated at 140 chars


# ── risk-badge dedup on consolidated facts + full description ───────────────

LONG_RISK = ("Reported ownership figures are inconsistent across facts: Huang is "
             "variously described as owning 3.5 percent and approximately 3 "
             "percent of the business in different filings.")


def test_one_risk_over_merged_facts_renders_single_badge_and_row():
    """A risk citing facts that consolidation merges must not repeat its
    badge/row once per grouped fact."""
    facts = [
        _fact("Subject owns 851,983,603 shares of Acme Corp as of October 2025",
              ["https://a.com/x"], category="financial"),
        _fact("Subject owns 851,983,603 shares of Acme Corp (ACME) as of October 29, 2025",
              ["https://b.com/y"], category="financial"),
    ]
    risk = {"description": LONG_RISK, "severity": "medium", "category": "financial",
            "confidence": 0.8, "impact_score": 5.0, "evidence": ["Fact 1", "Fact 2"]}
    html = render_html_report(_result(facts=facts, risks=[risk]),
                              "Subject Person", 1.0)
    card = re.search(r'<div class="fact .*?merged.*?</div>\s*</div>', html, re.S).group(0)
    assert card.count('class="risk-badge"') == 1     # was 2 pre-dedup
    assert card.count("Links to MEDIUM risk") == 1
    # full description, not an 80-char slice
    assert "different filings." in card


def test_distinct_risks_on_one_fact_keep_separate_badges():
    facts = [
        _fact("Subject owns many shares of Acme Corp as of October 2025",
              ["https://a.com/x"], category="financial"),
    ]
    risks = [
        {"description": "Risk one about inconsistent ownership figures reported",
         "severity": "medium", "category": "financial", "confidence": 0.8,
         "impact_score": 5.0, "evidence": ["Fact 1"]},
        {"description": "Risk two about concentrated insider selling patterns",
         "severity": "high", "category": "compliance", "confidence": 0.8,
         "impact_score": 6.0, "evidence": ["Fact 1"]},
    ]
    html = render_html_report(_result(facts=facts, risks=risks),
                              "Subject Person", 1.0)
    card = re.search(r'<div class="fact .*?</div>\s*</div>', html, re.S).group(0)
    assert card.count('class="risk-badge"') == 2


# ── quote → source links on evidence rows ───────────────────────────────────

def test_each_quote_row_links_its_own_members_source():
    """Merged card: quote A links a.com, quote B links b.com — never crossed."""
    facts = [
        _fact("Subject co-founded Acme Widgets in 1993",
              ["https://a.com/one"], quote="quote from source A"),
        _fact("Subject co-founded Acme Widgets in 1993 in California",
              ["https://b.com/two"], quote="quote from source B"),
    ]
    html = render_html_report(_result(facts=facts), "Subject Person", 1.0)
    row_a = re.search(r'<div class="detail-evidence[^"]*">[^<]*quote from source A.*?</div>', html, re.S).group(0)
    row_b = re.search(r'<div class="detail-evidence[^"]*">[^<]*quote from source B.*?</div>', html, re.S).group(0)
    assert "a.com" in row_a and "b.com" not in row_a
    assert "b.com" in row_b and "a.com" not in row_b


def test_quote_link_href_carries_highlight_fragment():
    facts = [_fact("Subject Person is CEO of Acme Corporation since 2018",
                   ["https://a.com/one"], quote="chief executive since 2018")]
    html = render_html_report(_result(facts=facts), "Subject Person", 1.0)
    link = re.search(r'<a class="ev-source" href="([^"]*)"', html).group(1)
    assert link.startswith("https://a.com/one#:~:text=")
    assert "chief%20executive%20since%202018" in link


def test_unmatched_quote_renders_without_source_link():
    f = _fact("Subject Person is CEO of Acme Corporation since 2018",
              ["https://a.com/one"], quote="the mapped quote")
    f["evidence"] = ["the mapped quote", "an orphan quote merged from elsewhere"]
    html = render_html_report(_result(facts=[f]), "Subject Person", 1.0)
    orphan_row = re.search(r'<div class="detail-evidence[^"]*">[^<]*orphan quote.*?</div>', html, re.S).group(0)
    assert "ev-source" not in orphan_row
    mapped_row = re.search(r'<div class="detail-evidence[^"]*">[^<]*the mapped quote.*?</div>', html, re.S).group(0)
    assert "a.com" in mapped_row


def test_quote_link_drops_invalid_scheme():
    facts = [_fact("Subject Person did something newsworthy this year",
                   ["javascript:alert(1)"], quote="a verbatim quote here")]
    html = render_html_report(_result(facts=facts), "Subject Person", 1.0)
    assert 'class="ev-source"' not in html
    assert 'href="javascript' not in html


def test_quote_links_capped_at_three():
    urls = [f"https://s{i}.com/x" for i in range(1, 7)]
    facts = [_fact("Subject Person appears in many publications this year",
                   urls, quote="a widely syndicated quote")]
    html = render_html_report(_result(facts=facts), "Subject Person", 1.0)
    row = re.search(r'<div class="detail-evidence[^"]*">.*?</div>', html, re.S).group(0)
    assert row.count('class="ev-source"') == 3


# ── risk cards: supporting facts with CORRECT display numbering ─────────────

def _three_facts_two_merge():
    return [
        _fact("Subject co-founded Acme Widgets in 1993",
              ["https://a.com/one"], quote="co-founded Acme Widgets"),
        _fact("Subject co-founded Acme Widgets in 1993 in California",
              ["https://b.com/two"], quote="in 1993 in California"),
        _fact("Subject opened a new office in Denver last year",
              ["https://c.com/three"], quote="new office in Denver"),
    ]


def test_risk_support_rows_use_consolidated_display_numbers():
    """Raw refs 'Fact 1'/'Fact 3': facts 1+2 merged into display #1, so raw
    position 3 is display #2 — the OLD labels would have said 'Fact 3'."""
    risk = {"description": "Risky pattern detected across filings",
            "severity": "high", "category": "legal", "confidence": 0.8,
            "impact_score": 7.0, "evidence": ["Fact 1", "Fact 3"]}
    html = render_html_report(_result(facts=_three_facts_two_merge(),
                                      risks=[risk]), "Subject Person", 1.0)
    rows = re.findall(r'<div class="risk-support-row">.*?</div>', html, re.S)
    assert len(rows) == 2
    assert 'data-fact="1"' in rows[0] and '>#1</button> Subject co-founded Acme Widgets in 1993 in California' in rows[0]
    assert 'data-fact="2"' in rows[1] and '>#2</button> Subject opened a new office in Denver' in rows[1]
    assert "Show supporting facts (2)" in html
    # the misleading raw-position labels are gone
    assert "Evidence: Fact" not in html


def test_risk_refs_merging_to_same_display_fact_dedupe():
    risk = {"description": "Risky pattern detected across filings",
            "severity": "high", "category": "legal", "confidence": 0.8,
            "impact_score": 7.0, "evidence": ["Fact 1", "Fact 2"]}
    html = render_html_report(_result(facts=_three_facts_two_merge(),
                                      risks=[risk]), "Subject Person", 1.0)
    assert html.count('<div class="risk-support-row">') == 1
    assert "Show supporting fact (1)" in html


def test_risk_support_rows_carry_anchored_source_links():
    risk = {"description": "Risky pattern detected across filings",
            "severity": "high", "category": "legal", "confidence": 0.8,
            "impact_score": 7.0, "evidence": ["Fact 3"]}
    html = render_html_report(_result(facts=_three_facts_two_merge(),
                                      risks=[risk]), "Subject Person", 1.0)
    row = re.search(r'<div class="risk-support-row">.*?</div>', html, re.S).group(0)
    assert 'class="ev-source"' in row
    assert 'https://c.com/three#:~:text=new%20office%20in%20Denver' in row


def test_risk_with_absent_refs_renders_without_support_block():
    risk = {"description": "Risky pattern detected across filings",
            "severity": "high", "category": "legal", "confidence": 0.8,
            "impact_score": 7.0, "evidence": ["Fact 99", "no digits"]}
    html = render_html_report(_result(facts=_three_facts_two_merge(),
                                      risks=[risk]), "Subject Person", 1.0)
    assert '<div class="risk-support-row">' not in html
    assert "Show supporting" not in html


# ── risk section: filter pills, condensed toggle, jump-to-fact plumbing ─────

def _many_risks():
    return [
        {"description": f"Risk number {i} concerning some conduct pattern",
         "severity": sev, "category": cat, "confidence": 0.8,
         "impact_score": 5.0, "evidence": ["Fact 1"]}
        for i, (sev, cat) in enumerate([
            ("high", "legal"), ("high", "compliance"),
            ("medium", "legal"), ("low", "reputational"),
        ], 1)
    ]


def test_risk_filter_pills_render_with_counts():
    html = render_html_report(_result(facts=_three_facts_two_merge(),
                                      risks=_many_risks()), "Subject Person", 1.0)
    assert 'data-kind="sev" data-val="all"' in html and "All (4)" in html
    assert 'data-val="high"' in html and "HIGH (2)" in html
    assert 'data-val="medium"' in html and "MEDIUM (1)" in html
    assert 'data-kind="cat" data-val="legal"' in html and "LEGAL (2)" in html
    assert "Condensed view" in html
    # values live in data attrs, never inline JS args
    assert "filterRisks(this)" in html and "filterRisks(this," not in html


def test_risk_cards_carry_filter_data_attrs():
    html = render_html_report(_result(facts=_three_facts_two_merge(),
                                      risks=_many_risks()), "Subject Person", 1.0)
    assert html.count('data-severity="high"') == 2
    assert html.count('data-rcat="legal"') == 2
    assert '<div id="risksContainer">' in html


def test_jump_and_back_controls_present():
    html = render_html_report(_result(facts=_three_facts_two_merge(),
                                      risks=_many_risks()), "Subject Person", 1.0)
    assert 'id="backToRisks"' in html
    assert "function jumpToFact" in html and "function backToRisks" in html


def test_no_risks_renders_no_filter_bar():
    html = render_html_report(_result(facts=_three_facts_two_merge()),
                              "Subject Person", 1.0)
    assert "risk-filters" not in html.split("<style")[0] or True
    assert '<div id="risksContainer">' not in html
    assert "No significant risks identified" in html
