"""
Browser tests for the loading screen (PLAN.md Rev 3.3, Phases A1/A3/A4 + A.2 + A.3).

Runs against the real app + scripted orchestrator (see conftest). Excluded
from the default gate (pytest.ini); run with: pytest -m e2e

A.3 revisions (plan-review-A3 R1, deliberate): the log defaults to NARRATIVE
rows (no engines/models/raw queries) with a "Technical log" toggle; the
skeleton was replaced by the instantly-visible brief + zeroed Overview; the
reconnect test now guards keyed-diff idempotency under group-row semantics;
the XSS test asserts inertness IN THE TECHNICAL VIEW (R2 — that's where the
raw scraped strings render).
"""

import re
import threading

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

HOSTILE_QUERY_TITLE = '<script>alert("xss-feed")</script> exposed'
HOSTILE_FACT = '<img src=x onerror="window.__pwned = true"> is CEO of Acme'


def _p(node, iteration=0, facts=0, max_iterations=10):
    return {"node": node, "iteration": iteration, "facts": facts,
            "max_iterations": max_iterations, "coverage": {"average": 0.3}}


def _search(query="jane doe career", results=8, engine="brave",
            category="professional", iteration=0):
    return {"kind": "search", "engine": engine, "query": query,
            "results": results, "category": category, "iteration": iteration}


def _extract_done(facts=2, samples=("Jane Doe is CEO of Acme",
                                    "Jane Doe founded Acme in 2010"),
                  categories=("professional", "biographical"), iteration=0):
    return {"kind": "extract", "status": "done", "facts": facts,
            "iteration": iteration,
            "samples": list(samples),
            "facts_new": [{"content": s, "category": categories[i % len(categories)],
                           "confidence": 0.9 - i * 0.1}
                          for i, s in enumerate(samples)]}


def _submit(page: Page, server: str, query="Test Subject"):
    page.goto(server)
    page.fill("#query", query)
    page.click("#submit-btn")
    expect(page.locator("#status")).to_be_visible()


def _bar_pct(page: Page) -> float:
    return page.locator("#bar").evaluate("el => parseFloat(el.style.width)")


class TestPhaseWeightedBar:
    def test_bar_advances_monotonically_through_phases(self, server, page,
                                                       set_script):
        g1, g2, g3 = threading.Event(), threading.Event(), threading.Event()
        set_script([
            ("progress", _p("strategy_planning")),
            ("gate", g1),
            ("progress", _p("data_collection")),
            ("activity", _search()),
            ("progress", _p("fact_extraction", facts=5)),
            ("gate", g2),
            ("progress", _p("query_refinement", iteration=1, facts=9)),
            ("progress", _p("verification", iteration=2, facts=12)),
            ("gate", g3),
            ("progress", _p("report_generation", iteration=2, facts=12)),
        ])
        _submit(page, server)

        expect(page.locator("#phase")).to_contain_text(
            "Planning research strategy")
        w1 = _bar_pct(page)
        assert w1 > 0

        g1.set()
        expect(page.locator("#phase")).to_contain_text(
            "Reading & extracting facts")
        w2 = _bar_pct(page)
        assert w2 > w1

        g2.set()
        expect(page.locator("#phase")).to_contain_text("Cross-checking facts")
        w3 = _bar_pct(page)
        assert w3 > w2

        g3.set()
        expect(page.locator("#status-title")).to_contain_text(
            "Research complete")
        assert _bar_pct(page) == 100

    def test_unknown_phase_holds_bucket_never_stalls_at_zero(self, server,
                                                             page, set_script):
        g1, g2 = threading.Event(), threading.Event()
        set_script([
            ("progress", _p("strategy_planning")),
            ("gate", g1),
            ("progress", _p("shiny_new_phase", iteration=1)),
            ("gate", g2),
        ])
        _submit(page, server)

        expect(page.locator("#phase")).to_contain_text(
            "Planning research strategy")
        w1 = _bar_pct(page)

        g1.set()
        expect(page.locator("#phase")).to_contain_text("shiny_new_phase")
        assert _bar_pct(page) == w1          # held, no reset, no stall at 0
        g2.set()


class TestNarrativeLog:
    def test_narrative_default_groups_searches_no_plumbing_visible(
            self, server, page, set_script):
        g1, g2 = threading.Event(), threading.Event()
        searches = [
            _search(query="jane doe topic 0", results=10, category="biographical"),
            _search(query="jane doe topic 1", results=10, category="financial"),
            _search(query="jane doe topic 2", results=10, category="legal"),
        ]
        set_script([
            ("progress", _p("data_collection")),
            *[("activity", s) for s in searches],
            ("gate", g1),
            ("activity", {"kind": "extract", "status": "start", "iteration": 0}),
            ("activity", _extract_done(facts=2)),
            ("gate", g2),
        ])
        _submit(page, server)

        # ONE group row for three searches; intent labels, no plumbing.
        expect(page.locator("#activity-feed li")).to_have_count(1)
        row = page.locator("#activity-feed li").first
        expect(row).to_contain_text("Investigating background")
        expect(row).to_contain_text("financial records")
        expect(row).to_contain_text("3 searches")
        log = page.locator("#activity-feed")
        expect(log).not_to_contain_text("Brave")
        expect(log).not_to_contain_text("jane doe topic")
        # Strip is narrative too (R3).
        expect(page.locator("#activity-line")).to_contain_text("Investigating")
        expect(page.locator("#activity-line")).not_to_contain_text("Brave")

        # Technical-log toggle is disabled pending the polish pass
        # (human decision 2026-07-10) — it must not render.
        expect(page.locator("#log-toggle")).to_have_count(0)

        g1.set()
        # Extraction becomes one transitioning row: sources sum + fact count.
        expect(log).to_contain_text("Read ~30 sources — 2 new facts")
        expect(page.locator("#activity-feed li")).to_have_count(2)
        g2.set()

    def test_llm_rows_render_narrative_only(self, server, page, set_script):
        g1 = threading.Event()
        set_script([
            ("progress", _p("strategy_planning")),
            ("activity", {"kind": "llm", "task": "strategy_planning",
                          "status": "done", "queries": 15}),
            ("activity", {"kind": "llm", "task": "query_refinement",
                          "status": "done", "queries": 12,
                          "sample_queries": ["jane doe lawsuits 2024"]}),
            ("activity", {"kind": "llm", "task": "risk_assessment",
                          "status": "done", "risks": 9, "model": "Claude",
                          "severities": {"high": 2, "medium": 4, "low": 3}}),
            ("gate", g1),
        ])
        _submit(page, server)

        log = page.locator("#activity-feed")
        expect(log).to_contain_text("Planned 15 research angles")
        expect(log).to_contain_text("Refined strategy — 12 new angles")
        expect(log).to_contain_text("Found 9 risk flags")
        # Decision (2): no models, no raw queries in the default view —
        # and with the toggle disabled, in ANY view.
        expect(log).not_to_contain_text("Claude")
        expect(log).not_to_contain_text("next:")
        g1.set()


class TestLiveReport:
    def test_opening_brief_and_zeroed_overview_render_instantly(
            self, server, page, set_script):
        """A3.2: no dead air — the brief + report structure appear before
        ANY backend event arrives."""
        g1, g2 = threading.Event(), threading.Event()
        set_script([
            ("gate", g1),
            ("activity", {"kind": "llm", "task": "strategy_planning",
                          "status": "done", "queries": 15}),
            ("progress", _p("strategy_planning")),
            ("gate", g2),
        ])
        _submit(page, server)

        # Instant, client-only: brief + tabs + zeroed counters + all 6 bars.
        # C1.6b (deliberate revision): on a scoped run the brief's Subject
        # line shows the RESOLVED entity descriptor (here the harness's
        # default auto candidate), not the raw query — brief agrees with
        # the assumption banner. Instant-render intent unchanged.
        expect(page.locator("#brief")).to_contain_text(
            "Subject: Scripted Test Entity")
        expect(page.locator("#brief")).to_contain_text("Coverage:")
        expect(page.locator("#live-report")).to_be_visible()
        expect(page.locator("#cat-bars .cat-row")).to_have_count(6)
        expect(page.locator("#count-facts")).to_have_text("0")
        expect(page.locator("#count-risks")).to_have_text("–")

        g1.set()
        # Planning becomes a visible deliverable.
        expect(page.locator("#plan-note")).to_have_text(
            "15 research angles planned")
        g2.set()

    def test_facts_tab_and_counter_stream_before_completion(self, server,
                                                            page, set_script):
        g1 = threading.Event()
        set_script([
            ("progress", _p("fact_extraction", facts=2)),
            ("activity", _extract_done()),
            ("gate", g1),
        ])
        _submit(page, server)

        expect(page.locator("#live-report")).to_be_visible()
        expect(page.locator("#count-facts")).to_have_text("2")
        page.click("#tab-btn-facts")
        expect(page.locator("#facts-list li")).to_have_count(2)
        expect(page.locator("#facts-list")).to_contain_text(
            "Jane Doe is CEO of Acme")
        expect(page.locator("#facts-list")).to_contain_text("confidence 90%")
        # Still mid-run: the report streamed BEFORE the terminal event.
        expect(page.locator("#status-title")).to_contain_text("Researching")
        g1.set()

    def test_signals_tab_populates_from_late_phase_events(self, server, page,
                                                          set_script):
        g1, g2 = threading.Event(), threading.Event()
        set_script([
            ("progress", _p("fact_extraction", iteration=1, facts=2)),
            ("activity", _extract_done()),        # facts arrive first
            ("gate", g1),
            ("progress", _p("risk_assessment", iteration=3, facts=20)),
            ("activity", {"kind": "llm", "task": "risk_assessment",
                          "status": "done", "risks": 9, "model": "Claude",
                          "severities": {"high": 2, "medium": 4, "low": 3}}),
            ("activity", {"kind": "llm", "task": "connection_mapping",
                          "status": "done", "connections": 21,
                          "sample": ["Apple", "Steve Jobs"], "model": "Claude"}),
            ("gate", g2),
        ])
        _submit(page, server)

        # Facts arrived, late-phase signals haven't: honest "coming up" state.
        page.click("#tab-btn-signals")
        expect(page.locator("#risks-box")).to_contain_text("coming up")

        g1.set()
        expect(page.locator("#count-risks")).to_have_text("9")
        expect(page.locator("#count-connections")).to_have_text("21")
        expect(page.locator("#risks-box .chip.sev-high")).to_have_text("2 high")
        expect(page.locator("#connections-box")).to_contain_text(
            "Apple · Steve Jobs")
        g2.set()

    def test_elapsed_timer_ticks_and_eta_copy_present(self, server, page,
                                                      set_script):
        g1 = threading.Event()
        set_script([("progress", _p("data_collection")), ("gate", g1)])
        _submit(page, server)

        expect(page.locator("#eta")).to_contain_text("2–4 minutes")
        expect(page.locator("#elapsed")).to_have_text(re.compile(r"\d+:\d{2}"))
        first = page.locator("#elapsed").inner_text()
        expect(page.locator("#elapsed")).not_to_have_text(first, timeout=3000)
        g1.set()


class TestHardening:
    def test_hostile_content_renders_inert(self, server, page, set_script):
        """A4(c): with the technical toggle disabled (polish-pass deferral),
        raw scraped log strings must not render ANYWHERE; the fact cards
        still render scraped content and must stay inert, and a hostile
        category falls back to the whitelist class. NOTE: when the toggle
        returns, restore the technical-view inertness assertions
        (plan-review-A3 R2)."""
        dialogs = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        g1 = threading.Event()
        hostile_extract = {
            "kind": "extract", "status": "done", "facts": 1, "iteration": 0,
            "samples": [HOSTILE_FACT],
            "facts_new": [{"content": HOSTILE_FACT,
                           "category": '"><img src=x onerror=alert(2)>',
                           "confidence": 0.9}],
        }
        set_script([
            ("progress", _p("data_collection")),
            ("activity", _search(query=HOSTILE_QUERY_TITLE,
                                 category='"><script>bad</script>')),
            ("activity", hostile_extract),
            ("gate", g1),
        ])
        _submit(page, server)

        log = page.locator("#activity-feed")
        # Narrative-only: hostile raw strings are NOT shown at all; the
        # hostile category falls back to a whitelist label.
        expect(log).to_contain_text("Investigating more angles")
        expect(log).not_to_contain_text("xss-feed")

        # Fact card: literal text; hostile category → whitelist default class.
        page.click("#tab-btn-facts")
        expect(page.locator("#facts-list")).to_contain_text(
            "<img src=x onerror=")
        expect(page.locator("#facts-list li").first).to_have_class(
            re.compile(r"cat-other"))
        assert page.locator("#activity-feed script, #activity-feed img,"
                            "#facts-list script, #facts-list img"
                            ).count() == 0
        assert page.evaluate("window.__pwned === undefined")
        assert dialogs == []
        g1.set()

    def test_reconnect_does_not_duplicate_or_reanimate_group_rows(
            self, server, page, set_script):
        """A4(3) under A.3 group semantics: the snapshot re-send must neither
        duplicate the group row nor lose the post-reconnect update."""
        g1, g2 = threading.Event(), threading.Event()
        set_script([
            ("progress", _p("data_collection")),
            ("activity", _search(query="jane doe alpha", category="biographical")),
            ("activity", _search(query="jane doe beta", category="financial")),
            ("activity", _search(query="jane doe gamma", category="legal")),
            ("gate", g1),
            ("activity", _search(query="jane doe delta", category="connections")),
            ("gate", g2),
        ])
        _submit(page, server)

        expect(page.locator("#activity-feed li")).to_have_count(1)
        expect(page.locator("#activity-feed li").first).to_contain_text(
            "3 searches")

        # Force an EventSource drop + auto-reconnect.
        page.context.set_offline(True)
        page.wait_for_timeout(500)
        page.context.set_offline(False)

        # Release one more event: the SAME group row updates in place —
        # stream live again, nothing duplicated by the snapshot replay.
        g1.set()
        expect(page.locator("#activity-feed li").first).to_contain_text(
            "4 searches", timeout=15000)
        expect(page.locator("#activity-feed li")).to_have_count(1)
        g2.set()

    def test_no_facts_soft_message_then_clears(self, server, page, set_script):
        g1, g2 = threading.Event(), threading.Event()
        set_script([
            ("progress", _p("data_collection", iteration=0, facts=0)),
            ("progress", _p("query_refinement", iteration=1, facts=0)),
            ("progress", _p("data_collection", iteration=2, facts=0)),
            ("gate", g1),
            ("activity", _extract_done(iteration=2)),
            ("progress", _p("fact_extraction", iteration=2, facts=2)),
            ("gate", g2),
        ])
        _submit(page, server)

        expect(page.locator("#no-facts")).to_be_visible()
        expect(page.locator("#no-facts")).to_contain_text("Still searching")

        g1.set()
        expect(page.locator("#no-facts")).not_to_be_visible()
        page.click("#tab-btn-facts")
        expect(page.locator("#facts-list li")).to_have_count(2)
        g2.set()

    def test_reduced_motion_still_renders_everything(self, server, page,
                                                     set_script):
        g1 = threading.Event()
        set_script([
            ("progress", _p("fact_extraction", facts=2)),
            ("activity", _search()),
            ("activity", _extract_done()),
            ("gate", g1),
        ])
        page.emulate_media(reduced_motion="reduce")
        _submit(page, server)

        expect(page.locator("#count-facts")).to_have_text("2")
        expect(page.locator("#activity-feed li")).to_have_count(2)
        page.click("#tab-btn-facts")
        expect(page.locator("#facts-list li")).to_have_count(2)
        g1.set()

    def test_failed_job_renders_terminal_state(self, server, page, set_script):
        set_script([
            ("progress", _p("data_collection")),
            ("fail", "boom"),
        ])
        _submit(page, server)

        expect(page.locator("#status-title")).to_contain_text("Research failed")
        expect(page.locator("#phase")).to_contain_text("research failed")

    def test_budget_exhausted_renders_terminal_state(self, server, page,
                                                     set_script,
                                                     high_cost_orchestrator):
        set_script([("progress", _p("data_collection"))])
        _submit(page, server)

        expect(page.locator("#status-title")).to_contain_text("Research failed")
        expect(page.locator("#phase")).to_contain_text("per-job budget")
