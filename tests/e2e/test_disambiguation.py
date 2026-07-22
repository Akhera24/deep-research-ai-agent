"""
Browser tests for the C1.3/C1.4 disambiguation flow (PLAN.md Rev 3.8).

Real app + scripted pre-flight + scripted orchestrator (conftest). Covers the
acceptance-gate browser items: picker rendering (evidence-ranked, ≤5,
textContent-only, XSS-inert), auto-proceed banner + neutral cancel (R7),
hints → both endpoints, refine + unscoped escape hatches, fail-open.
"""

import re
import threading
import time

import httpx
import pytest
from playwright.sync_api import Page, expect

from src.core.preflight import PreflightCandidate, PreflightResult
from tests.e2e.conftest import ScriptedPreflight

pytestmark = pytest.mark.e2e

HOSTILE_NAME = 'Jane <img src=x onerror="window.__pwned = true"> Doe'
HOSTILE_DESCRIPTOR = 'javascript:alert(1) — attacker, Evil Corp (web)'
HOSTILE_DOMAIN = '<script>window.__pwned = true</script>evil.com'


def _p(node, iteration=0, facts=0):
    return {"node": node, "iteration": iteration, "facts": facts,
            "max_iterations": 10, "coverage": {"average": 0.3}}


def _candidate(name="Jane Doe",
               descriptor="Jane Doe — VP Engineering, Stripe (San Francisco)",
               mass=5, domains=None, disambiguators=("Stripe", "VP Engineering")):
    return PreflightCandidate(
        canonical_name=name, descriptor=descriptor,
        disambiguators=list(disambiguators), domain_mass=mass,
        domains=domains or ["stripe.com", "wikipedia.org", "linkedin.com"],
    )


def _pick_result():
    return PreflightResult(decision="pick", candidates=[
        _candidate(),
        _candidate(name=HOSTILE_NAME, descriptor=HOSTILE_DESCRIPTOR,
                   mass=2, domains=[HOSTILE_DOMAIN, "law.example.co.uk"],
                   disambiguators=()),
    ])


def _submit(page: Page, server: str, query="Jane Doe"):
    page.goto(server)
    page.fill("#query", query)
    page.click("#submit-btn")


class TestPicker:
    def test_picker_renders_candidates_ranked_and_xss_inert(self, server, page):
        ScriptedPreflight.result = _pick_result()
        dialogs = []
        page.on("dialog", lambda d: (dialogs.append(d), d.dismiss()))
        _submit(page, server)

        expect(page.locator("#picker")).to_be_visible()
        cards = page.locator(".picker-card")
        expect(cards).to_have_count(2)
        # evidence-ranked: most-documented first
        expect(cards.nth(0)).to_contain_text("5 independent sources")
        expect(cards.nth(0)).to_contain_text("VP Engineering, Stripe")
        # hostile name/descriptor/domain render as LITERAL text
        expect(cards.nth(1)).to_contain_text('Jane <img src=x onerror=')
        expect(cards.nth(1)).to_contain_text("javascript:alert(1)")
        expect(cards.nth(1)).to_contain_text("<script>")
        # zero injected nodes, zero side effects
        assert page.locator(".picker-card img").count() == 0
        assert page.locator(".picker-card script").count() == 0
        assert page.evaluate("window.__pwned") is None
        assert dialogs == []
        # the unscoped consequence is visible BEFORE any click
        expect(page.locator("#picker-warning")).to_be_visible()
        # submit stays disabled while the picker is open
        expect(page.locator("#submit-btn")).to_be_disabled()
        # keyboard-operable: candidates are real buttons and focusable
        assert page.locator(".picker-card").nth(0).evaluate(
            "el => el.tagName") == "BUTTON"
        page.locator(".picker-card").nth(0).focus()
        assert page.evaluate(
            "document.activeElement.classList.contains('picker-card')")

    def test_pick_candidate_starts_scoped_run_with_banner(self, server, page,
                                                          set_script):
        ScriptedPreflight.result = _pick_result()
        set_script([("progress", _p("data_collection"))])
        _submit(page, server)
        page.locator(".picker-card").nth(0).click()

        expect(page.locator("#picker")).to_be_hidden()
        expect(page.locator("#status")).to_be_visible()
        banner = page.locator("#assumption-banner")
        expect(banner).to_be_visible()
        expect(banner).to_contain_text("Researching: Jane Doe — VP Engineering, Stripe")
        assert "warn" not in (banner.get_attribute("class") or "")
        # run completes; the banner stays as the record of what was researched
        expect(page.locator("#status-title")).to_contain_text(
            "Research complete", timeout=15000)
        expect(page.locator("#cancel-btn")).to_be_hidden()
        expect(page.locator("#banner-cancel")).to_be_hidden()

    def test_refine_reopens_hints_and_reenables_submit(self, server, page):
        ScriptedPreflight.result = _pick_result()
        _submit(page, server)
        expect(page.locator("#picker")).to_be_visible()
        page.click("#picker-refine")
        expect(page.locator("#picker")).to_be_hidden()
        assert page.locator("#hints").get_attribute("open") is not None
        expect(page.locator("#submit-btn")).to_be_enabled()
        assert page.evaluate(
            "document.activeElement.id") == "hint-company"

    def test_just_search_the_name_runs_unscoped_with_warning(self, server,
                                                             page, set_script):
        ScriptedPreflight.result = _pick_result()
        set_script([])
        _submit(page, server)
        page.click("#picker-unscoped")
        expect(page.locator("#status")).to_be_visible()
        banner = page.locator("#assumption-banner")
        expect(banner).to_be_visible()
        expect(banner).to_contain_text("may mix")
        assert "warn" in banner.get_attribute("class")


class TestAutoProceedAndCancel:
    def test_auto_banner_then_cancel_renders_neutral_state(self, server, page,
                                                           set_script):
        ScriptedPreflight.result = PreflightResult(
            decision="auto", note="dominant", candidates=[_candidate()])
        gate = threading.Event()
        set_script([
            ("progress", _p("data_collection")),
            ("gate", gate),
            ("progress", _p("fact_extraction", facts=3)),
        ])
        _submit(page, server)

        # auto-proceed: NO picker — navigated straight to the run page,
        # banner reconstructed there (P0 deliberate revision: the run left
        # the homepage; cancel→refine now crosses pages via dra_refine, R2)
        page.wait_for_url(re.compile(r"/research/"))
        banner = page.locator("#assumption-banner")
        expect(banner).to_contain_text("Researching: Jane Doe — VP Engineering")
        cancel = page.locator("#banner-cancel")
        expect(cancel).to_be_visible()       # owner device → control renders
        expect(cancel).to_contain_text("Not who you meant?")

        cancel.click()
        expect(cancel).to_be_disabled()
        gate.set()      # next node boundary → CancelledByUser

        # live cancel on the OWNER page → homepage with refine re-armed:
        # hints open, query prefilled, submit enabled, one-shot key consumed
        page.wait_for_url(f"{server}/", timeout=15000)
        assert page.locator("#hints").get_attribute("open") is not None
        expect(page.locator("#query")).to_have_value("Jane Doe")
        expect(page.locator("#submit-btn")).to_be_enabled()
        assert page.evaluate("sessionStorage.getItem('dra_refine')") is None

    def test_plain_cancel_button_hands_back_too(self, server, page,
                                                set_script):
        ScriptedPreflight.result = None      # default: auto scripted entity
        gate = threading.Event()
        set_script([
            ("progress", _p("data_collection")),
            ("gate", gate),
            ("progress", _p("fact_extraction")),
        ])
        _submit(page, server)
        page.wait_for_url(re.compile(r"/research/"))
        expect(page.locator("#status")).to_be_visible()
        cancel = page.locator("#cancel-btn")
        expect(cancel).to_be_visible()
        cancel.click()
        gate.set()
        # same cross-page hand-back as the banner cancel (both buttons
        # route through cancelJob → cancelRequestedHere)
        page.wait_for_url(f"{server}/", timeout=15000)
        assert page.locator("#hints").get_attribute("open") is not None
        expect(page.locator("#query")).to_have_value("Jane Doe")

    def test_genuine_failure_still_renders_error_state(self, server, page,
                                                       set_script):
        # regression guard for the R7 mapping: only the cancel string is
        # neutral — a real failure keeps the failed state
        ScriptedPreflight.result = None      # default auto → run starts
        set_script([("progress", _p("data_collection")),
                    ("fail", "provider melted")])
        _submit(page, server)
        expect(page.locator("#status-title")).to_have_text(
            "Research failed", timeout=15000)


class TestFinishEarly:
    def test_two_step_confirm_posts_finish_and_wraps_up(self, server, page,
                                                        set_script):
        gate = threading.Event()
        set_script([
            ("progress", _p("fact_extraction", iteration=1, facts=4)),
            ("gate", gate),
            ("progress", _p("query_refinement", iteration=1, facts=4)),
        ])
        finish_posts = []
        page.on("request", lambda r: finish_posts.append(r.url)
                if r.method == "POST" and r.url.endswith("/finish") else None)
        _submit(page, server)

        btn = page.locator("#finish-btn")
        expect(btn).to_be_visible()          # facts>0 → button appears
        btn.click()                          # step 1: confirm, NO POST yet
        expect(page.locator("#finish-question")).to_contain_text("4 facts")
        assert finish_posts == []

        page.click("#finish-no")             # backs out cleanly
        expect(btn).to_be_visible()
        assert finish_posts == []

        btn.click()
        page.click("#finish-yes")            # step 2: the actual request
        expect(page.locator("#finish-pending")).to_contain_text(
            "Finishing after the current pass")
        assert len(finish_posts) == 1
        # in-process server: the registry really holds the job id
        from src.api import jobs as jobs_mod
        assert len(jobs_mod._finish_early) == 1

        gate.set()                           # scripted run completes
        expect(page.locator("#status-title")).to_contain_text(
            "Research complete", timeout=15000)
        expect(page.locator("#finish-wrap")).to_be_hidden()

    def test_button_hidden_while_zero_facts(self, server, page, set_script):
        gate = threading.Event()
        set_script([
            ("progress", _p("data_collection", iteration=1, facts=0)),
            ("gate", gate),
        ])
        _submit(page, server)
        expect(page.locator("#status")).to_be_visible()
        expect(page.locator("#finish-wrap")).to_be_hidden()
        expect(page.locator("#cancel-btn")).to_be_visible()  # cancel remains
        gate.set()
        expect(page.locator("#status-title")).to_contain_text(
            "Research complete", timeout=15000)


class TestHintsAndFailOpen:
    def test_hints_posted_to_disambiguate(self, server, page, set_script):
        set_script([])
        page.goto(server)
        page.fill("#query", "Jane Doe")
        page.click("#hints summary")
        page.fill("#hint-company", "Stripe")
        page.fill("#hint-role", "VP Engineering")
        page.click("#submit-btn")
        expect(page.locator("#status")).to_be_visible()
        assert ScriptedPreflight.last["query"] == "Jane Doe"
        assert ScriptedPreflight.last["hints"] == {
            "company": "Stripe", "role": "VP Engineering"}

    def test_expired_ticket_403_client_recovers(self, server, page,
                                                monkeypatch):
        # Gate (6): user ponders the picker past the TTL → /api/research 403s
        # → the client shows a friendly retry message and re-enables submit
        # (fail() also resets the Turnstile widget for a fresh challenge).
        from src.api import db as api_db
        monkeypatch.setattr(api_db.settings, "PREFLIGHT_TICKET_TTL_SECONDS", 0)
        ScriptedPreflight.result = _pick_result()
        _submit(page, server)
        expect(page.locator("#picker")).to_be_visible()
        page.locator(".picker-card").nth(0).click()

        err = page.locator("#error")
        expect(err).to_be_visible()
        expect(err).to_contain_text("retry the challenge")
        expect(page.locator("#submit-btn")).to_be_enabled()
        expect(page.locator("#status")).to_be_hidden()

    def test_error_decision_offers_broad_never_auto_posts(self, server, page,
                                                          set_script):
        # D6 (Rev 3.9): fail-open is OFFER-to-proceed. An error decision
        # must land on the decision surface and POST NO research until the
        # user explicitly clicks "Just search the name".
        ScriptedPreflight.result = PreflightResult(decision="error",
                                                   note="pre-flight failed")
        set_script([])
        research_posts = []
        page.on("request", lambda r: research_posts.append(r.url)
                if r.method == "POST" and r.url.endswith("/api/research")
                else None)
        _submit(page, server)

        expect(page.locator("#picker")).to_be_visible()
        expect(page.locator("#picker-title")).to_contain_text(
            "didn’t complete")
        expect(page.locator("#picker-note")).to_contain_text(
            "pre-flight failed")
        expect(page.locator("#picker-warning")).to_be_visible()
        assert research_posts == []          # nothing auto-submitted

        page.click("#picker-unscoped")       # the explicit broad choice
        expect(page.locator("#status")).to_be_visible()
        assert len(research_posts) == 1
        banner = page.locator("#assumption-banner")
        expect(banner).to_contain_text("may mix")
        expect(page.locator("#status-title")).to_contain_text(
            "Research complete", timeout=15000)

    def test_unscoped_decision_shows_surface_with_note(self, server, page):
        ScriptedPreflight.result = PreflightResult(
            decision="unscoped", note="thin evidence — single weakly-documented entity")
        _submit(page, server)
        expect(page.locator("#picker")).to_be_visible()
        expect(page.locator("#picker-title")).to_contain_text(
            "couldn’t confidently identify")
        expect(page.locator("#picker-note")).to_contain_text("thin evidence")
        # refine path re-enables submit for a fresh attempt
        page.click("#picker-refine")
        expect(page.locator("#submit-btn")).to_be_enabled()

    def test_zero_hint_match_single_candidate_pick_shows_note_and_card(
            self, server, page):
        # R6: D7 can return pick with ONE candidate + note — the surface
        # must show BOTH (the old >=2 guard would have gone broad silently)
        ScriptedPreflight.result = PreflightResult(
            decision="pick",
            note="none of the identified profiles matched your details",
            candidates=[_candidate()])
        _submit(page, server)
        expect(page.locator("#picker")).to_be_visible()
        expect(page.locator("#picker-note")).to_contain_text(
            "matched your details")
        expect(page.locator(".picker-card")).to_have_count(1)

    def test_preflight_pending_state_visible(self, server, page, set_script):
        import threading
        ScriptedPreflight.hold = threading.Event()
        set_script([])
        page.goto(server)
        page.fill("#query", "Jane Doe")
        page.click("#submit-btn")
        expect(page.locator("#preflight-pending")).to_be_visible()
        ScriptedPreflight.hold.set()
        expect(page.locator("#preflight-pending")).to_be_hidden()
        expect(page.locator("#status")).to_be_visible()   # default auto runs


class TestSidelineSection:
    """C1.7a gate (1) browser half: the collapsed sideline section arrives
    through the REAL job pipeline (ScriptedOrchestrator.result carries
    sidelined_facts — the scripted harness has no graph/extractor, so
    attribution mechanics stay unit-tested; Rev 3.9 R5 lesson)."""

    def test_sidelined_facts_render_in_collapsed_section(self, server, page,
                                                         set_script):
        from tests.e2e.conftest import ScriptedOrchestrator
        hostile = 'John <img src=x onerror="window.__pwned = true"> HSBC clerk'
        ScriptedOrchestrator.result = {
            "facts": [{
                "content": "Jane Doe is a VP at Stripe",
                "category": "professional", "confidence": 0.9,
                "source_urls": ["https://stripe.com/about"],
                "source_reliabilities": {}, "evidence": ["quote"],
                "verified": True, "verification_count": 2,
            }],
            "sidelined_facts": [
                {"content": "John Smith (explorer) mapped Virginia in 1607",
                 "category": "biographical", "confidence": 0.85,
                 "about_target": False,
                 "source_urls": ["https://history.example/smith?a=1&b=2"],
                 "source_reliabilities": {}, "evidence": ["colonial quote"]},
                {"content": hostile, "category": "professional",
                 "confidence": 0.8, "about_target": False,
                 "source_urls": ["https://rocketreach.example/johns"],
                 "source_reliabilities": {}, "evidence": []},
            ],
            "risk_flags": [], "connections": [],
            "metadata": {"coverage": {"average": 0.5}, "iterations": 1,
                         "duration_seconds": 0.1},
        }
        set_script([("progress", _p("data_collection", facts=1))])
        dialogs = []
        page.on("dialog", lambda d: (dialogs.append(d), d.dismiss()))
        _submit(page, server)
        expect(page.locator("#status-title")).to_contain_text(
            "Research complete", timeout=15000)

        href = page.locator("#report-link").get_attribute("href")
        page.goto(server + href)

        header = page.locator("#sidelineSection .section-header")
        expect(header).to_contain_text(
            "Facts about other people named Jane Doe (2)")
        expect(header).to_contain_text(
            "not included in the analysis or score")
        # collapsed by default: no `open` class → max-height 0 + clipped
        # (Playwright treats a clipped-but-laid-out child as "visible", so
        # assert the collapse mechanics the report's own toggle uses)
        content = page.locator("#sidelineSection .section-content")
        assert "open" not in (content.get_attribute("class") or "")
        assert content.evaluate("e => e.getBoundingClientRect().height") == 0
        header.click()
        expect(content).to_have_class("section-content open")
        assert content.evaluate("e => e.getBoundingClientRect().height") > 0
        expect(content).to_contain_text(
            "John Smith (explorer) mapped Virginia in 1607")
        # hostile sidelined text is literal — no injected nodes, no dialogs
        expect(content).to_contain_text('John <img src=x onerror=')
        assert page.locator("#sidelineSection img").count() == 0
        assert page.evaluate("window.__pwned") is None
        assert dialogs == []
        # R6: citation chip href from the RAW seam — & encoded exactly once
        chip_href = page.locator(
            "#sidelineSection a.cite-chip").first.get_attribute("href")
        assert "a=1&b=2" in chip_href
        assert "%26amp" not in chip_href

    def test_set_aside_note_appears_in_overview(self, server, page,
                                                set_script):
        # Gate the run open so the mid-run snapshot (not just the terminal
        # event) is what the client renders.
        g1 = threading.Event()
        set_script([
            ("progress", _p("data_collection", iteration=1, facts=1)),
            ("activity", {"kind": "extract", "status": "done", "facts": 1,
                          "samples": ["Jane Doe is a VP at Stripe"],
                          "facts_new": [{"content": "Jane Doe is a VP at Stripe",
                                         "category": "professional",
                                         "confidence": 0.9}],
                          "set_aside": 3}),
            ("gate", g1),
        ])
        _submit(page, server)
        note = page.locator("#set-aside-note")
        expect(note).to_be_visible(timeout=15000)
        expect(note).to_contain_text(
            "3 facts about other people with this name set aside")
        g1.set()
        expect(page.locator("#status-title")).to_contain_text(
            "Research complete", timeout=15000)


class TestHintMatchLabel:
    """C1.7c: the 'matches your details' label renders on hint_match=strong
    ONLY (LLM display signal — constant string, fixed class)."""

    def test_label_on_strong_only_and_microcopy(self, server, page):
        strong = _candidate(descriptor="Jane Doe — VP Engineering, Stripe (SF)")
        strong.hint_match = "strong"
        partial = _candidate(name="Jane B. Doe",
                             descriptor="Jane Doe — Barrister, London (UK)",
                             mass=2, domains=["law.example.co.uk"])
        partial.hint_match = "partial"
        ScriptedPreflight.result = PreflightResult(
            decision="pick", candidates=[strong, partial])

        page.goto(server)
        # microcopy visible once the hints form is opened
        page.click("#hints summary")
        expect(page.locator("#hints-tip")).to_be_visible()
        expect(page.locator("#hints-tip")).to_contain_text(
            "Short keywords work best")
        page.fill("#query", "Jane Doe")
        page.click("#submit-btn")

        cards = page.locator(".picker-card")
        expect(cards).to_have_count(2)
        expect(cards.nth(0).locator(".match-strong")).to_have_text(
            "matches your details")
        # partial (and none) carry NO label — diluted signal rejected
        assert cards.nth(1).locator(".match-strong").count() == 0


class TestRejectedEntities:
    """C1.7d gate (7) e2e half (R8): the stub records kwargs AND the
    /api/disambiguate + /api/research request BODIES are asserted via the
    Playwright network log — a stub that swallowed kwargs would let a broken
    wire pass silently."""

    def test_refine_stash_rides_both_request_bodies(self, server, page,
                                                    set_script):
        import json as _json
        ScriptedPreflight.result = _pick_result()
        set_script([("progress", _p("data_collection"))])
        bodies = {}
        page.on("request", lambda r: bodies.setdefault(
            r.url.split(server)[-1], []).append(r.post_data)
            if r.method == "POST" and "/api/" in r.url else None)

        _submit(page, server)
        expect(page.locator("#picker")).to_be_visible()
        # explicit rejection of every displayed candidate
        page.click("#picker-refine")
        expect(page.locator("#submit-btn")).to_be_enabled()

        # second pre-flight: default auto → the run starts by itself
        ScriptedPreflight.result = None
        page.click("#submit-btn")
        expect(page.locator("#status-title")).to_contain_text(
            "Research complete", timeout=15000)

        # network log: the SECOND /api/disambiguate body carries both
        # rejected descriptors; the first carried none
        dis = [_json.loads(b) for b in bodies["/api/disambiguate"]]
        assert "rejected_entities" not in dis[0]
        assert dis[1]["rejected_entities"] == [
            "Jane Doe — VP Engineering, Stripe (San Francisco)",
            HOSTILE_DESCRIPTOR,
        ]
        # ...and the /api/research body carries them too
        res = [_json.loads(b) for b in bodies["/api/research"]]
        assert res[0]["rejected_entities"] == dis[1]["rejected_entities"]
        # R8: the server stub recorded the kwarg (not swallowed)
        assert ScriptedPreflight.last["rejected_entities"] == \
            dis[1]["rejected_entities"]

    def test_cancel_stashes_entity_and_query_change_clears(self, server,
                                                           page, set_script):
        import json as _json
        ScriptedPreflight.result = PreflightResult(
            decision="auto", note="dominant", candidates=[_candidate()])
        gate = threading.Event()
        # the post-gate progress write is where the node-boundary cancel
        # check raises (same shape as the C1.4 cancel test)
        set_script([("progress", _p("data_collection")), ("gate", gate),
                    ("progress", _p("fact_extraction", facts=3))])
        bodies = {}
        page.on("request", lambda r: bodies.setdefault(
            r.url.split(server)[-1], []).append(r.post_data)
            if r.method == "POST" and "/api/" in r.url else None)

        _submit(page, server)
        page.wait_for_url(re.compile(r"/research/"))
        expect(page.locator("#assumption-banner")).to_be_visible()
        page.click("#banner-cancel")
        gate.set()
        # P0 R2 (deliberate revision): the stash now crosses pages — the
        # run page writes dra_refine, the homepage re-arms it on load.
        page.wait_for_url(f"{server}/", timeout=15000)
        assert page.locator("#hints").get_attribute("open") is not None

        # resubmit the SAME query → the cancelled entity rides as rejected
        set_script([("progress", _p("data_collection"))])
        page.click("#submit-btn")
        page.wait_for_url(re.compile(r"/research/"), timeout=15000)
        dis = [_json.loads(b) for b in bodies["/api/disambiguate"]]
        assert dis[1]["rejected_entities"] == [
            "Jane Doe — VP Engineering, Stripe (San Francisco)"]
        res = [_json.loads(b) for b in bodies["/api/research"]]
        assert res[1]["rejected_entities"] == dis[1]["rejected_entities"]

        # a DIFFERENT name from a fresh homepage carries no stale scope
        page.goto(server)
        set_script([("progress", _p("data_collection"))])
        page.fill("#query", "Someone Else")
        page.click("#submit-btn")
        page.wait_for_url(re.compile(r"/research/"), timeout=15000)
        assert "rejected_entities" not in _json.loads(
            bodies["/api/disambiguate"][2])


class TestNewInputRearmsSubmit:
    """Human-reported 2026-07-13: with the picker open (submit disabled),
    typing a NEW name left the button dead — the only way out was a page
    reload. Editing the query while the decision surface is open abandons
    the surface and re-arms submit for a fresh pre-flight."""

    def test_typing_new_name_closes_picker_and_reenables_submit(
            self, server, page, set_script):
        ScriptedPreflight.result = _pick_result()
        _submit(page, server)
        expect(page.locator("#picker")).to_be_visible()
        expect(page.locator("#submit-btn")).to_be_disabled()

        page.fill("#query", "Someone Else")
        expect(page.locator("#picker")).to_be_hidden()
        expect(page.locator("#submit-btn")).to_be_enabled()

        # the re-armed submit runs a fresh pre-flight end-to-end
        ScriptedPreflight.result = None      # default auto
        set_script([("progress", _p("data_collection"))])
        page.click("#submit-btn")
        expect(page.locator("#status-title")).to_contain_text(
            "Research complete", timeout=15000)
        assert ScriptedPreflight.last["query"] == "Someone Else"

    def test_home_nav_from_run_page_never_cancels_the_job(self, server, page,
                                                          set_script):
        # P0 D4 (deliberate revision): the old invariant — typing during an
        # in-page run must not re-arm submit — is obsolete: the run left the
        # homepage entirely. Its successor: leaving the run page via the
        # nav's Home does NOT cancel the job, and the homepage returns
        # fresh (submit enabled, hints closed — no refine re-arm without a
        # cancel).
        gate = threading.Event()
        set_script([("progress", _p("data_collection")), ("gate", gate),
                    ("progress", _p("fact_extraction", facts=1))])
        _submit(page, server)
        page.wait_for_url(re.compile(r"/research/"))
        job_id = page.url.rsplit("/", 1)[-1]

        page.click("#site-nav .nav-brand")
        page.wait_for_url(f"{server}/")
        expect(page.locator("#submit-btn")).to_be_enabled()
        assert page.locator("#hints").get_attribute("open") is None

        r = httpx.get(f"{server}/api/research/{job_id}")
        assert r.json()["status"] == "running"      # navigation ≠ cancel
        gate.set()
        deadline = time.time() + 10
        while time.time() < deadline:
            if httpx.get(f"{server}/api/research/{job_id}"
                         ).json()["status"] == "completed":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("job did not complete after Home nav")
