"""
Phase B acceptance gate — browser/DOM assertions (items 4, 5, 6).

4. XSS sink: a javascript:/<script>-bearing source URL renders inert.
5. Encoding: a #:~:text= fragment from a quote with "/&/% survives as a
   parseable URL whose fragment decodes back to the exact quote.
6. Visibility: >3 trend signals -> top-3 visible + a working "Show all N"
   whose expanded count equals the data count.

Uses page.set_content on the real renderer output — no server needed; the
sink/encoding/toggle properties are independent of transport headers.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import re

import pytest
from playwright.sync_api import expect

from src.reporting.html_report import render_html_report

pytestmark = pytest.mark.e2e


def _fact(content, urls, quote="", category="professional"):
    return {
        "content": content, "category": category, "confidence": 0.9,
        "source_urls": urls, "evidence": [quote] if quote else [],
        "source_reliabilities": {}, "verified": False,
        "verification_count": 1, "entities_mentioned": [],
    }


def _render(facts):
    return render_html_report({
        "facts": facts, "risk_flags": [], "connections": [],
        "metadata": {"coverage": {"average": 0.5}, "iterations": 1},
    }, "Subject Person", 1.0)


def test_gate4_javascript_url_inert_in_dom(page):
    html = _render([
        _fact("Subject Person did something notable in business",
              ["javascript:alert(1)",
               'https://evil.com/"><script>alert(2)</script>',
               "https://real.com/article"]),
    ])
    dialogs = []
    page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
    page.set_content(html)

    # No javascript:-scheme anchor anywhere in the live DOM (case-insensitive)
    assert page.evaluate(
        "[...document.querySelectorAll('a')].filter("
        "a => a.href.trim().toLowerCase().startsWith('javascript:')).length"
    ) == 0
    # No script element smuggled through a href attribute
    assert page.evaluate("document.scripts.length") > 0  # report's own JS runs
    assert dialogs == []  # nothing executed on load
    # javascript: dropped; evil.com IS a valid https host, so its chip
    # renders — but the "><script> payload stayed INSIDE the href attribute
    # (getAttribute returns it as an inert string) instead of becoming markup.
    chips = page.locator("a.cite-chip")
    expect(chips).to_have_count(2)
    # No script element carries the payload — the attribute never broke out
    assert page.evaluate(
        "[...document.scripts].filter(s => s.textContent.includes('alert(2)')).length"
    ) == 0
    assert page.evaluate("document.querySelectorAll('.fact a').length") == 2


def test_gate5_fragment_encoding_roundtrip(page):
    quote = 'He said "profits & growth" rose 5% — 100-fold'
    html = _render([
        _fact("Subject Person made a statement about profits this year",
              ["https://news.com/story?id=7&lang=en"], quote=quote),
    ])
    page.set_content(html)
    chip = page.locator("a.cite-chip").first
    parsed = chip.evaluate("""el => {
        const u = new URL(el.href);
        return {origin: u.origin, path: u.pathname, search: u.search,
                hash: u.hash};
    }""")
    # Base URL structure intact — never percent-encoded wholesale (M1)
    assert parsed["origin"] == "https://news.com"
    assert parsed["path"] == "/story"
    assert parsed["search"] == "?id=7&lang=en"
    # Fragment is a text directive whose payload decodes to the exact quote
    assert parsed["hash"].startswith("#:~:text=")
    decoded = chip.evaluate(
        "el => decodeURIComponent(new URL(el.href).hash.slice('#:~:text='.length))"
    )
    assert decoded == quote


def test_gate6_show_all_signals_expands_to_data_count(page):
    facts = [
        _fact(f"Subject Person announced record growth number {i} for 2025",
              [f"https://news{i}.com/x"])
        for i in range(1, 6)  # 5 expansion signals
    ]
    page.set_content(_render(facts))
    group = page.locator(".trend-group", has_text="Growth & Expansion")
    items = group.locator(".trend-item")
    expect(items).to_have_count(5)          # all in the DOM (no [:3] slice)
    visible_before = [i for i in range(5) if items.nth(i).is_visible()]
    assert len(visible_before) == 3         # top-3 by default (OQ-B3)

    btn = group.locator("button.show-all-btn")
    expect(btn).to_have_text("Show all 5 signals")
    btn.click()
    for i in range(5):
        expect(items.nth(i)).to_be_visible()  # expanded count == data count
    expect(btn).to_have_text("Show fewer")
    btn.click()
    assert sum(items.nth(i).is_visible() for i in range(5)) == 3


def test_cite_more_toggle_reveals_all_sources(page):
    urls = [f"https://source{i}.com/a" for i in range(1, 8)]  # 7 sources
    page.set_content(_render([
        _fact("Subject Person appears in many independent publications", urls),
    ]))
    chips = page.locator("a.cite-chip")
    expect(chips).to_have_count(7)
    assert sum(chips.nth(i).is_visible() for i in range(7)) == 3
    btn = page.locator("button.cite-more-btn")
    expect(btn).to_have_text("+4 more sources")
    btn.click()
    for i in range(7):
        expect(chips.nth(i)).to_be_visible()


def _report_with_risks():
    facts = [
        _fact("Subject co-founded Acme Widgets in 1993", ["https://a.com/1"],
              quote="co-founded Acme Widgets"),
    ] + [
        _fact(f"Subject filler fact number {i} about business dealings and events",
              [f"https://f{i}.com/x"]) for i in range(2, 60)
    ] + [
        _fact("Subject faces a lawsuit over contract disputes in Delaware court",
              ["https://court.com/case"], category="legal"),
    ]
    risks = [
        {"description": "Active litigation exposure in Delaware", "severity": "high",
         "category": "legal", "confidence": 0.8, "impact_score": 7.0,
         "evidence": ["Fact 60"]},
        {"description": "Concentrated governance power", "severity": "medium",
         "category": "governance", "confidence": 0.7, "impact_score": 5.0,
         "evidence": ["Fact 1"]},
    ]
    return render_html_report({
        "facts": facts, "risk_flags": risks, "connections": [],
        "metadata": {"coverage": {"average": 0.5}, "iterations": 1},
    }, "Subject Person", 1.0)


def test_jump_to_fact_crosses_pagination_and_flashes(page):
    """Fact 60 sits on page 3 of the facts list — the jump must page to it,
    scroll it into view, flash it, and offer a way back."""
    page.set_content(_report_with_risks())
    risk = page.locator(".risk", has_text="Delaware").first
    risk.locator("button.show-all-btn").click()
    link = risk.locator("button.risk-fact-link")
    fact_no = link.text_content().lstrip("#")
    link.click()
    target = page.locator(f'.fact[data-index="{fact_no}"]')
    expect(target).to_be_visible()          # unhidden despite pagination
    expect(target).to_have_class(re.compile("fact-flash"))
    expect(target).to_be_in_viewport()
    # the return button parks INSIDE the landed-on fact card
    back = target.locator("#backToRisks")
    expect(back).to_be_visible()
    back.click()
    expect(back).not_to_be_visible()
    expect(page.locator(".risk", has_text="Delaware").first).to_be_in_viewport()


def test_severity_pill_filters_risk_cards(page):
    page.set_content(_report_with_risks())
    cards = page.locator("#risksContainer .risk")
    expect(cards).to_have_count(2)
    page.locator('.risk-pill[data-val="high"]').click()
    expect(page.locator("#risksContainer .risk:visible")).to_have_count(1)
    expect(page.locator("#risksContainer .risk:visible")).to_contain_text("Delaware")
    page.locator('.risk-pill[data-val="high"]').click()  # toggle back to all
    expect(page.locator("#risksContainer .risk:visible")).to_have_count(2)


def test_condensed_view_hides_detail_and_toggles_back(page):
    page.set_content(_report_with_risks())
    toggle = page.locator('button[onclick="toggleRiskView(this)"]')
    toggle.click()
    expect(toggle).to_have_text("Detailed view")
    risk = page.locator("#risksContainer .risk").first
    expect(risk.locator(".risk-meta")).not_to_be_visible()
    expect(risk.locator(".citation-chips")).not_to_be_visible()
    toggle.click()
    expect(toggle).to_have_text("Condensed view")
    expect(risk.locator(".risk-meta")).to_be_visible()


def test_counts_recompute_across_dimensions_and_zero_pills_disable(page):
    """Selecting HIGH must recompute category counts under that selection —
    a pill may never promise results the other dimension filtered away."""
    page.set_content(_report_with_risks())
    page.locator('.risk-pill[data-val="high"]').click()
    legal = page.locator('.risk-pill[data-val="legal"]')
    gov = page.locator('.risk-pill[data-val="governance"]')
    expect(legal).to_have_text("LEGAL (1)")
    expect(gov).to_have_text("GOVERNANCE (0)")
    expect(gov).to_be_disabled()          # unclickable at zero
    # deselect HIGH -> counts restore, governance clickable again
    page.locator('.risk-pill[data-val="high"]').click()
    expect(gov).to_have_text("GOVERNANCE (1)")
    expect(gov).to_be_enabled()


def test_multi_select_severities_and_all_resets(page):
    page.set_content(_report_with_risks())
    cards = page.locator("#risksContainer .risk")
    page.locator('.risk-pill[data-val="high"]').click()
    expect(page.locator("#risksContainer .risk:visible")).to_have_count(1)
    page.locator('.risk-pill[data-val="medium"]').click()   # additive
    expect(page.locator("#risksContainer .risk:visible")).to_have_count(2)
    expect(page.locator('.risk-pill[data-val="high"]')).to_have_class(re.compile("active"))
    expect(page.locator('.risk-pill[data-val="medium"]')).to_have_class(re.compile("active"))
    # All clears BOTH dimensions
    page.locator('.risk-pill[data-val="legal"]').click()
    page.locator('.risk-pill[data-val="all"]').click()
    expect(page.locator("#risksContainer .risk:visible")).to_have_count(2)
    expect(page.locator('.risk-pill[data-val="all"]')).to_have_class(re.compile("active"))
    expect(page.locator('.risk-pill[data-val="legal"]')).not_to_have_class(re.compile("active"))


def test_condensed_card_click_expands_just_that_card(page):
    page.set_content(_report_with_risks())
    page.locator("button.risk-view-toggle", has_text="Condensed view").click()
    cards = page.locator("#risksContainer .risk")
    first, second = cards.nth(0), cards.nth(1)
    expect(first.locator(".risk-meta")).not_to_be_visible()
    first.click()
    expect(first).to_have_class(re.compile("risk-expanded"))
    expect(first.locator(".risk-meta")).to_be_visible()       # this card opened
    expect(second.locator(".risk-meta")).not_to_be_visible()  # neighbor stayed condensed
    first.click()
    expect(first.locator(".risk-meta")).not_to_be_visible()   # re-condensed
    # switching back to detailed clears any per-card expansion state
    first.click()
    page.locator("button.risk-view-toggle", has_text="Detailed view").click()
    expect(first).not_to_have_class(re.compile("risk-expanded"))
    expect(first.locator(".risk-meta")).to_be_visible()


def test_expand_all_facts_master_toggle(page):
    page.set_content(_report_with_risks())
    master = page.locator('button[onclick="toggleAllRiskFacts(this)"]')
    expect(master).to_have_text("Expand all facts")
    rows = page.locator(".risk-support-row")
    assert sum(rows.nth(i).is_visible() for i in range(rows.count())) == 0
    master.click()
    expect(master).to_have_text("Collapse all facts")
    for i in range(rows.count()):
        expect(rows.nth(i)).to_be_visible()
    # per-risk buttons stayed label-synced
    expect(page.locator(".risk-support .show-all-btn").first).to_contain_text("Hide")
    master.click()
    expect(master).to_have_text("Expand all facts")
    assert sum(rows.nth(i).is_visible() for i in range(rows.count())) == 0


def test_expand_all_facts_works_in_condensed_view(page):
    """Master expand must not be defeated by the condensed hide rule —
    cards stay clamped for scanning, but every support block shows."""
    page.set_content(_report_with_risks())
    page.locator('button[onclick="toggleRiskView(this)"]').click()  # condensed FIRST
    rows = page.locator(".risk-support-row")
    assert sum(rows.nth(i).is_visible() for i in range(rows.count())) == 0
    page.locator('button[onclick="toggleAllRiskFacts(this)"]').click()
    for i in range(rows.count()):
        expect(rows.nth(i)).to_be_visible()
    # cards themselves remained condensed (meta still hidden)
    expect(page.locator("#risksContainer .risk").first.locator(".risk-meta")).not_to_be_visible()
    page.locator('button[onclick="toggleAllRiskFacts(this)"]').click()
    assert sum(rows.nth(i).is_visible() for i in range(rows.count())) == 0
