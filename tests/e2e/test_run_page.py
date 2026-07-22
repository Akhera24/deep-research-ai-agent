"""
P0 Step 2 — the dedicated run page's cold-load state machine + live SSE.

Jobs are created via the REAL POST /api/research (Turnstile faked by the
harness, ScriptedOrchestrator only — $0 spend); the browser then COLD-LOADS
/research/{id} — the exact refresh/bookmark/share path. The D5 table rows
that need no real job (410/422/404-variant/network) are driven by stubbing
the snapshot response with page.route — that exercises the CLIENT state
machine, which is the thing under test (the server's 404/410 behavior is
unit-tested).
"""

import threading
import time

import httpx
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

ENTITY = {"canonical_name": "Jane Doe",
          "descriptor": "Jane Doe — VP Engineering, Stripe",
          "disambiguators": ["Stripe", "VP Engineering"],
          "decision": "picked"}


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
            "facts_new": [{"content": s,
                           "category": categories[i % len(categories)],
                           "confidence": 0.9 - i * 0.1}
                          for i, s in enumerate(samples)]}


def _create_job(server, query="Test Subject", entity=None):
    body = {"query": query, "turnstile_token": ""}
    if entity:
        body["entity"] = entity
    r = httpx.post(f"{server}/api/research", json=body)
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def _wait_status(server, job_id, statuses, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = httpx.get(f"{server}/api/research/{job_id}")
        if r.status_code == 200 and r.json()["status"] in statuses:
            return r.json()
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {statuses}")


def _own(page: Page, server: str, job_id: str, query="Test Subject"):
    """Mark this browser device as the run's owner (what Step 3's submit
    flow will do) — localStorage needs the origin, so touch it first."""
    page.goto(server)
    page.evaluate(
        """([id, q]) => localStorage.setItem('dra_owned_v1',
               JSON.stringify({[id]: {query: q,
                               createdAt: new Date().toISOString()}}))""",
        [job_id, query])


class TestColdLoadRunning:
    def test_running_job_renders_live_panel_banner_and_attaches_sse(
            self, server, page, set_script):
        gate = threading.Event()
        set_script([
            ("progress", _p("data_collection")),
            ("activity", _search()),
            ("activity", _extract_done()),
            ("progress", _p("fact_extraction", facts=2)),
            ("gate", gate),
            ("progress", _p("report_generation", iteration=2, facts=2)),
        ])
        job_id = _create_job(server, entity=ENTITY)
        _wait_status(server, job_id, ("running",))

        page.goto(f"{server}/research/{job_id}")
        # Snapshot render: live panel + R3 banner reconstruction
        expect(page.locator("#status")).to_be_visible()
        expect(page.locator("#assumption-banner")).to_be_visible()
        expect(page.locator("#banner-text")).to_contain_text(
            "Jane Doe — VP Engineering, Stripe")
        expect(page.locator("#brief-subject")).to_contain_text(
            "Jane Doe — VP Engineering, Stripe")
        expect(page.locator("#count-facts")).to_have_text("2")
        expect(page.locator("#elapsed")).to_have_text(
            __import__("re").compile(r"\d+:\d\d"))
        # SSE attached: release the gate → the live terminal state arrives
        gate.set()
        expect(page.locator("#status-title")).to_contain_text(
            "Research complete", timeout=15000)
        expect(page.locator("#report-link")).to_be_visible()
        assert page.locator("#report-link").get_attribute("href") == \
            f"/api/research/{job_id}/report"
        expect(page.locator("#new-research")).to_be_visible()

    def test_midrun_refresh_produces_zero_duplicate_dom(self, server, page,
                                                        set_script):
        gate = threading.Event()
        set_script([
            ("progress", _p("data_collection")),
            ("activity", _search()),
            ("activity", _search(category="financial")),
            ("activity", _extract_done()),
            ("progress", _p("fact_extraction", facts=2)),
            ("gate", gate),
        ])
        job_id = _create_job(server)
        _wait_status(server, job_id, ("running",))

        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#count-facts")).to_have_text("2")
        rows_before = page.locator("#activity-feed li").count()
        facts_before = page.locator("#facts-list li").count()
        assert rows_before > 0 and facts_before == 2

        page.reload()
        expect(page.locator("#count-facts")).to_have_text("2")
        assert page.locator("#activity-feed li").count() == rows_before
        assert page.locator("#facts-list li").count() == facts_before
        gate.set()

    def test_unscoped_run_gets_warn_banner_never_undefined(self, server,
                                                           page, set_script):
        gate = threading.Event()
        set_script([("progress", _p("data_collection")), ("gate", gate)])
        job_id = _create_job(server)   # no entity → {decision: "unscoped"}
        _wait_status(server, job_id, ("running",))

        page.goto(f"{server}/research/{job_id}")
        banner = page.locator("#assumption-banner")
        expect(banner).to_be_visible()
        expect(banner).to_have_class(
            __import__("re").compile(r"\bwarn\b"))
        expect(page.locator("#banner-text")).to_contain_text(
            "Searching the name broadly")
        expect(page.locator("#banner-text")).not_to_contain_text("undefined")
        gate.set()

    def test_banner_arrives_over_sse_when_cold_load_was_queued(
            self, server, page, set_script):
        # Finalization A: cold-load hits a QUEUED snapshot (no
        # resolved_entity yet — jobs.py writes it on the first RUNNING
        # write); the banner must grow from the SSE stream. The queued
        # snapshot is stubbed (the real queued→running window is too narrow
        # to hit deterministically); the SSE stream is the REAL one.
        gate = threading.Event()
        set_script([("progress", _p("data_collection")), ("gate", gate)])
        job_id = _create_job(server, entity=ENTITY)
        _wait_status(server, job_id, ("running",))

        served_fake = {"count": 0}

        def first_snapshot_queued(route):
            if served_fake["count"] == 0:
                served_fake["count"] += 1
                route.fulfill(json={
                    "job_id": job_id, "status": "queued",
                    "progress": {"phase": "queued"},
                    "created_at": "2026-07-13T00:00:00Z",
                })
            else:
                route.fallback()

        page.route(f"**/api/research/{job_id}", first_snapshot_queued)
        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#status")).to_be_visible()
        # The stubbed cold snapshot carried NO resolved_entity, so a visible
        # banner can only have come from the SSE stream's render() hook —
        # the property finalization A adds. (The intermediate hidden state
        # is unobservable at the harness's 0.1s SSE poll.)
        expect(page.locator("#assumption-banner")).to_be_visible(
            timeout=10000)
        expect(page.locator("#banner-text")).to_contain_text("Jane Doe")
        assert served_fake["count"] == 1, "queued stub never served"
        gate.set()

    def test_cold_load_elapsed_seeds_from_started_at(self, server, page,
                                                     set_script):
        # R9: a 3-minute-old run must not restart at 0:00. The snapshot's
        # started_at is rewritten in flight; everything else is real.
        gate = threading.Event()
        set_script([("progress", _p("data_collection")), ("gate", gate)])
        job_id = _create_job(server)
        _wait_status(server, job_id, ("running",))

        def shift_started_at(route):
            r = route.fetch()
            body = r.json()
            body["started_at"] = "2026-07-13T00:00:00+00:00"
            route.fulfill(json=body)

        page.route(f"**/api/research/{job_id}", shift_started_at)
        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#elapsed")).to_have_text(
            __import__("re").compile(r"\d+:\d\d"))
        elapsed = page.locator("#elapsed").inner_text()
        minutes = int(elapsed.split(":")[0])
        assert minutes >= 60, f"elapsed not seeded from started_at: {elapsed}"
        gate.set()


class TestColdLoadTerminal:
    def test_completed_job_renders_grade_summary_and_report_link(
            self, server, page, set_script):
        set_script([("progress", _p("report_generation", facts=3))])
        job_id = _create_job(server, entity=ENTITY)
        _wait_status(server, job_id, ("completed",))

        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#status-title")).to_contain_text(
            "Research complete")
        expect(page.locator("#phase")).to_contain_text("Complete")
        expect(page.locator("#report-link")).to_be_visible()
        assert page.locator("#report-link").get_attribute("href") == \
            f"/api/research/{job_id}/report"
        # R3: banner reconstructed on a terminal cold-load too
        expect(page.locator("#assumption-banner")).to_be_visible()
        expect(page.locator("#banner-cancel")).to_be_hidden()
        expect(page.locator("#new-research")).to_be_visible()
        # No live-run controls on a terminal page
        expect(page.locator("#cancel-btn")).to_be_hidden()
        assert page.locator("#bar").evaluate(
            "el => el.style.width") == "100%"
        # Completed cold-load fidelity: the terminal dict has no preview, so
        # "coming up" copy and zeroed category bars must not render (they
        # would read as "no risks found" / "0 facts per category").
        expect(page.locator("#risks-box .pending-note")).to_have_text(
            "Finalized — see the full report below.")
        expect(page.locator("#connections-box .pending-note")).to_have_text(
            "Finalized — see the full report below.")
        expect(page.locator("#cat-bars")).to_be_hidden()

    def test_nav_new_research_cta_returns_home_without_cancel(
            self, server, page, set_script):
        # D4 affordance (2026-07-21): the nav CTA leaves mid-run, the job
        # keeps running — same invariant as the brand link.
        gate = threading.Event()
        set_script([("progress", _p("data_collection", facts=1)),
                    ("gate", gate),
                    ("progress", _p("report_generation", iteration=2,
                                    facts=1))])
        job_id = _create_job(server)
        _wait_status(server, job_id, ("running",))
        page.goto(f"{server}/research/{job_id}")
        cta = page.locator("#site-nav .nav-cta")
        expect(cta).to_be_visible()
        expect(cta).to_have_attribute(
            "title", __import__("re").compile("keeps running"))
        cta.click()
        page.wait_for_url(f"{server}/")
        assert httpx.get(
            f"{server}/api/research/{job_id}").json()["status"] == "running"
        gate.set()
        _wait_status(server, job_id, ("completed",))

    def test_cancelled_job_cold_load(self, server, page, set_script):
        g1, g2 = threading.Event(), threading.Event()
        set_script([
            ("progress", _p("data_collection")),
            ("gate", g1),
            ("progress", _p("fact_extraction", facts=1)),
            ("gate", g2),
        ])
        job_id = _create_job(server, entity=ENTITY)
        _wait_status(server, job_id, ("running",))
        httpx.post(f"{server}/api/research/{job_id}/cancel")
        g1.set()   # next progress write trips the cancel check
        snap = _wait_status(server, job_id, ("failed",))
        assert snap["error"] == "cancelled by user"
        g2.set()

        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#status-title")).to_contain_text(
            "Research cancelled")
        expect(page.locator("#status-title")).not_to_contain_text("failed")
        expect(page.locator("#assumption-banner")).to_be_hidden()
        expect(page.locator("#new-research")).to_be_visible()
        # Visitor (no dra_owned): no refine affordance either
        expect(page.locator("#refine-again")).to_be_hidden()

    def test_failed_job_renders_error_state(self, server, page, set_script):
        set_script([("progress", _p("data_collection")), ("fail", "boom")])
        job_id = _create_job(server)
        _wait_status(server, job_id, ("failed",))

        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#status-title")).to_contain_text(
            "Research failed")
        expect(page.locator("#new-research")).to_be_visible()
        expect(page.locator("#cancel-btn")).to_be_hidden()


class TestColdLoadErrorStates:
    def test_unknown_uuid_renders_not_found(self, server, page):
        page.goto(f"{server}/research/00000000-0000-4000-8000-000000000000")
        expect(page.locator("#page-note")).to_be_visible()
        expect(page.locator("#note-title")).to_have_text("Research not found")
        expect(page.locator("#page-note a")).to_have_attribute("href", "/")
        expect(page.locator("#status")).to_be_hidden()

    def test_malformed_id_renders_not_found_via_422(self, server, page):
        # R8: the page route (str) serves the shell; the snapshot's
        # uuid.UUID param answers 422 — same friendly render as 404.
        page.goto(f"{server}/research/not-a-uuid")
        expect(page.locator("#page-note")).to_be_visible()
        expect(page.locator("#note-title")).to_have_text("Research not found")
        expect(page.locator("#status")).to_be_hidden()

    def test_expired_job_renders_expired_note(self, server, page):
        job_id = "11111111-1111-4111-8111-111111111111"
        page.route(f"**/api/research/{job_id}",
                   lambda route: route.fulfill(
                       status=410, json={"detail": "report expired"}))
        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#note-title")).to_have_text(
            "This research has expired")
        expect(page.locator("#note-text")).to_contain_text("7 days")

    def test_network_failure_renders_retry_note(self, server, page):
        job_id = "22222222-2222-4222-8222-222222222222"
        page.route(f"**/api/research/{job_id}", lambda route: route.abort())
        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#note-title")).to_have_text("Connection problem")
        expect(page.locator("#note-text")).to_contain_text("refresh")

    def test_no_console_errors_on_any_cold_load_state(self, server, page,
                                                      set_script):
        # R1/R8 guard: no ReferenceError on any state. Collect console
        # errors across every state visited above in one pass.
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        set_script([("progress", _p("report_generation", facts=1))])
        job_id = _create_job(server)
        _wait_status(server, job_id, ("completed",))
        for path in (f"/research/{job_id}", "/research/not-a-uuid",
                     "/research/00000000-0000-4000-8000-000000000000"):
            page.goto(f"{server}{path}")
            page.wait_for_timeout(300)
        assert errors == [], f"page errors: {errors}"


class TestNavigationAndTabs:
    def test_back_button_returns_to_live_run_and_resumes(self, server, page,
                                                         set_script):
        # Edge 7: Home mid-run, return via browser history → the run page
        # cold-loads again, SSE re-attaches, the run finishes on screen.
        import re as _re
        gate = threading.Event()
        set_script([("progress", _p("data_collection", facts=1)),
                    ("gate", gate),
                    ("progress", _p("report_generation", iteration=2,
                                    facts=1))])
        page.goto(server)
        page.fill("#query", "Jane Doe")
        page.click("#submit-btn")
        page.wait_for_url(_re.compile(r"/research/"))
        run_url = page.url

        page.click("#site-nav .nav-brand")
        page.wait_for_url(f"{server}/")
        page.go_back()
        page.wait_for_url(run_url)
        expect(page.locator("#status")).to_be_visible()
        expect(page.locator("#phase")).to_contain_text("Searching the web")
        gate.set()
        expect(page.locator("#status-title")).to_contain_text(
            "Research complete", timeout=15000)

    def test_two_tabs_same_run_both_render_and_finish(self, server, page,
                                                      context, set_script):
        # Edge 8: two tabs watching one run — both read-only, both
        # idempotent, both reach the terminal state.
        gate = threading.Event()
        set_script([
            ("progress", _p("data_collection")),
            ("activity", _search()),
            ("activity", _extract_done()),
            ("progress", _p("fact_extraction", facts=2)),
            ("gate", gate),
        ])
        job_id = _create_job(server)
        _wait_status(server, job_id, ("running",))

        page.goto(f"{server}/research/{job_id}")
        page2 = context.new_page()
        page2.goto(f"{server}/research/{job_id}")
        for pg in (page, page2):
            expect(pg.locator("#count-facts")).to_have_text("2")
            assert pg.locator("#facts-list li").count() == 2
        gate.set()
        for pg in (page, page2):
            expect(pg.locator("#status-title")).to_contain_text(
                "Research complete", timeout=15000)
        page2.close()


class TestHostileEntityDescriptor:
    def test_banner_and_brief_render_hostile_descriptor_inert(
            self, server, page, set_script):
        # Edge 12 (run-page half): the descriptor is display-tier text
        # (length+control-strip validation only — it legitimately carries
        # <, >, punctuation) and is the run page's brief/banner source.
        hostile = {"canonical_name": "Jane Doe",
                   "descriptor":
                       '<img src=x onerror="window.__pwned = true"> — Corp',
                   "disambiguators": [], "decision": "picked"}
        gate = threading.Event()
        set_script([("progress", _p("data_collection")), ("gate", gate)])
        job_id = _create_job(server, entity=hostile)
        _wait_status(server, job_id, ("running",))

        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#banner-text")).to_contain_text(
            '<img src=x onerror=')
        expect(page.locator("#brief-subject")).to_contain_text(
            '<img src=x onerror=')
        assert page.locator("#assumption-banner img").count() == 0
        assert page.locator("#brief img").count() == 0
        assert page.evaluate("window.__pwned") is None
        gate.set()


class TestOwnerControls:
    def test_owner_sees_cancel_and_finish_visitor_does_not(self, server,
                                                           page, set_script):
        gate = threading.Event()
        set_script([
            ("progress", _p("data_collection", facts=2)),
            ("gate", gate),
        ])
        job_id = _create_job(server, entity=ENTITY)
        _wait_status(server, job_id, ("running",))

        # Visitor first: fresh origin storage → watch-only page (R5)
        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#status")).to_be_visible()
        expect(page.locator("#cancel-btn")).to_be_hidden()
        expect(page.locator("#finish-wrap")).to_be_hidden()
        expect(page.locator("#banner-cancel")).to_be_hidden()
        # ...but the watch surface stays: banner text + elapsed visible
        expect(page.locator("#assumption-banner")).to_be_visible()
        expect(page.locator("#elapsed")).to_be_visible()

        # Owner: dra_owned carries the job → controls render
        _own(page, server, job_id)
        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#cancel-btn")).to_be_visible()
        expect(page.locator("#banner-cancel")).to_be_visible()
        expect(page.locator("#finish-wrap")).to_be_visible()  # facts>0
        gate.set()

    def test_owner_cancel_hands_back_to_homepage(self, server, page,
                                                 set_script):
        # R2 live path: cancel on the run page → dra_refine written →
        # homepage. (The homepage re-arm assertions land with Step 3.)
        g1, g2 = threading.Event(), threading.Event()
        set_script([
            ("progress", _p("data_collection")),
            ("gate", g1),
            ("progress", _p("fact_extraction", facts=1)),
            ("gate", g2),
        ])
        job_id = _create_job(server, query="Jane Doe", entity=ENTITY)
        _wait_status(server, job_id, ("running",))
        _own(page, server, job_id, query="Jane Doe")

        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#cancel-btn")).to_be_visible()
        page.click("#cancel-btn")
        expect(page.locator("#cancel-btn")).to_be_disabled()
        g1.set()   # deliver the cancel at the next node boundary
        page.wait_for_url(f"{server}/", timeout=15000)
        g2.set()
        # Step 3's homepage re-arm consumed the one-shot key and re-armed
        # the refine flow from it.
        assert page.evaluate(
            "sessionStorage.getItem('dra_refine')") is None
        assert page.locator("#hints").get_attribute("open") is not None
        expect(page.locator("#query")).to_have_value("Jane Doe")

    def test_visitor_watching_cancel_is_not_yanked_to_homepage(
            self, server, page, context, set_script):
        # Checkpoint ruling 2: only the page that ASKED navigates.
        g1, g2 = threading.Event(), threading.Event()
        set_script([
            ("progress", _p("data_collection")),
            ("gate", g1),
            ("progress", _p("fact_extraction", facts=1)),
            ("gate", g2),
        ])
        job_id = _create_job(server, entity=ENTITY)
        _wait_status(server, job_id, ("running",))

        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#status")).to_be_visible()
        # The owner cancels from elsewhere (direct API here)
        httpx.post(f"{server}/api/research/{job_id}/cancel")
        g1.set()
        _wait_status(server, job_id, ("failed",))
        g2.set()
        expect(page.locator("#status-title")).to_contain_text(
            "Research cancelled", timeout=15000)
        # Still on the run page — no yank
        assert page.url == f"{server}/research/{job_id}"
        expect(page.locator("#refine-again")).to_be_hidden()
