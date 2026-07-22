"""
P0 Step 4 — "Your recent runs" (localStorage history) on the homepage.

Edge-case list (testing-edge-cases): empty store → section hidden; hostile
label planted DIRECTLY in the store (the server's query charset can't be
relied on — another tab or a stale schema could hold anything) → inert;
corrupt/wrong-shape JSON → reset-not-crash; hydration 404 → prune while a
5xx keeps the entry; Clear wipes display but NEVER ownership (R5/gate 15);
show-5/expand-all boundary; localStorage disabled → homepage + submit work.
"""

import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

HOSTILE_LABEL = '<img src=x onerror="window.__pwned = true"> & <script>x</script>'


def _p(node, iteration=0, facts=0):
    return {"node": node, "iteration": iteration, "facts": facts,
            "max_iterations": 10, "coverage": {"average": 0.3}}


def _plant_history(page: Page, server: str, entries):
    """Write a dra_history_v1 store, then reload so the page renders it."""
    page.goto(server)
    page.evaluate(
        "e => localStorage.setItem('dra_history_v1', JSON.stringify(e))",
        entries)
    page.reload()


def _entry(job_id, label="Test Subject", status="completed", **extra):
    # createdAt is RELATIVE to now — a hardcoded date silently ages past
    # the 7-day client TTL and the store's own prune deletes the fixture
    # before the assertion runs (bit us 2026-07-22).
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)
                 ).strftime("%Y-%m-%dT%H:%M:%SZ")
    e = {"jobId": job_id, "label": label,
         "createdAt": yesterday, "status": status}
    e.update(extra)
    return e


class TestHistoryFlow:
    def test_submit_records_entry_and_hydration_shows_grade(
            self, server, page, set_script):
        set_script([("progress", _p("report_generation", facts=2))])
        page.goto(server)
        page.fill("#query", "Jane Doe")
        page.click("#submit-btn")
        page.wait_for_url(re.compile(r"/research/"))
        job_id = page.url.rsplit("/", 1)[-1]
        # wait server-side terminal, then return home
        deadline = time.time() + 10
        while time.time() < deadline:
            if httpx.get(f"{server}/api/research/{job_id}"
                         ).json()["status"] == "completed":
                break
            time.sleep(0.05)

        page.goto(server)
        expect(page.locator("#history")).to_be_visible()
        row = page.locator("#history-list li a")
        expect(row).to_have_count(1)
        expect(row.locator(".hist-label")).to_have_text("Jane Doe")
        assert row.get_attribute("href") == f"/research/{job_id}"
        # hydration flipped queued → completed and fetched the grade
        expect(row.locator(".chip")).to_contain_text("Grade")
        # ...and persisted it back to the store
        stored = json.loads(page.evaluate(
            "localStorage.getItem('dra_history_v1')"))
        assert stored[0]["status"] == "completed"
        assert stored[0]["grade"]

    def test_empty_store_keeps_section_hidden(self, server, page):
        page.goto(server)
        expect(page.locator("#history")).to_be_hidden()

    def test_hostile_label_in_store_renders_inert(self, server, page):
        # stub the snapshot so hydration can't prune the planted entry
        page.route("**/api/research/33333333-*",
                   lambda route: route.fulfill(status=503, body=""))
        _plant_history(page, server, [
            _entry("33333333-3333-4333-8333-333333333333",
                   label=HOSTILE_LABEL, status="running")])
        expect(page.locator(".hist-label")).to_contain_text(
            '<img src=x onerror=')
        assert page.locator("#history-list img").count() == 0
        assert page.locator("#history-list script").count() == 0
        assert page.evaluate("window.__pwned") is None

    def test_corrupt_and_wrong_shape_stores_never_crash(self, server, page):
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server)
        for bad in ("{corrupt", '"a string"', '{"an": "object"}', "[null]",
                    '[{"jobId": ""}]'):
            page.evaluate(
                "v => localStorage.setItem('dra_history_v1', v)", bad)
            page.reload()
            expect(page.locator("#history")).to_be_hidden()
        ours = [e for e in errors if "Turnstile" not in e]
        assert ours == [], f"page errors: {ours}"

    def test_hydration_prunes_purged_jobs_but_keeps_on_5xx(self, server,
                                                           page):
        gone = "44444444-4444-4444-8444-444444444444"   # real 404 from API
        flaky = "55555555-5555-4555-8555-555555555555"  # stubbed 500
        page.goto(server)
        page.route(f"**/api/research/{flaky}",
                   lambda route: route.fulfill(status=500, body="boom"))
        page.evaluate(
            "e => localStorage.setItem('dra_history_v1', JSON.stringify(e))",
            [_entry(gone, label="purged one", status="running"),
             _entry(flaky, label="flaky one", status="running")])
        page.reload()
        # hydration settles: purged entry pruned, flaky entry retained
        expect(page.locator("#history-list li")).to_have_count(1)
        expect(page.locator(".hist-label")).to_have_text("flaky one")
        stored = json.loads(page.evaluate(
            "localStorage.getItem('dra_history_v1')"))
        assert [e["jobId"] for e in stored] == [flaky]

    def test_clear_history_keeps_ownership_of_live_run(self, server, page,
                                                       set_script):
        # Gate item 15: Clear wipes the DISPLAY list only — an in-flight
        # owned run keeps its controls.
        gate = threading.Event()
        set_script([("progress", _p("data_collection", facts=1)),
                    ("gate", gate)])
        page.goto(server)
        page.fill("#query", "Jane Doe")
        page.click("#submit-btn")
        page.wait_for_url(re.compile(r"/research/"))
        job_id = page.url.rsplit("/", 1)[-1]

        page.goto(server)
        expect(page.locator("#history")).to_be_visible()
        page.click("#history-clear")
        expect(page.locator("#history")).to_be_hidden()
        assert page.evaluate(
            "localStorage.getItem('dra_history_v1')") is None
        owned = json.loads(page.evaluate(
            "localStorage.getItem('dra_owned_v1')"))
        assert job_id in owned          # ownership survives

        page.goto(f"{server}/research/{job_id}")
        expect(page.locator("#cancel-btn")).to_be_visible()   # still owner
        gate.set()

    def test_future_created_at_survives_prune(self, server, page):
        # Edge 13 (clock skew): a future createdAt is kept, never pruned.
        page.route("**/api/research/88888888-*",
                   lambda route: route.fulfill(status=503, body=""))
        _plant_history(page, server, [
            _entry("88888888-8888-4888-8888-888888888888",
                   label="clock skew", status="running",
                   createdAt="2027-01-01T00:00:00Z")])
        expect(page.locator(".hist-label")).to_have_text("clock skew")

    def test_show5_collapse_and_expand_all(self, server, page):
        entries = [_entry(f"66666666-0000-4000-8000-{i:012d}",
                          label=f"Subject {i}", status="running")
                   for i in range(8)]
        page.goto(server)
        # stub every snapshot GET so hydration keeps all 8 untouched
        page.route(re.compile(r".*/api/research/66666666-.*"),
                   lambda route: route.fulfill(status=503, body=""))
        page.evaluate(
            "e => localStorage.setItem('dra_history_v1', JSON.stringify(e))",
            entries)
        page.reload()
        expect(page.locator("#history-list li")).to_have_count(8)
        visible = page.locator("#history-list li:not([hidden])")
        expect(visible).to_have_count(5)
        more = page.locator("#history-more")
        expect(more).to_be_visible()
        expect(more).to_have_text("Show all 8")
        more.click()
        expect(page.locator("#history-list li:not([hidden])")).to_have_count(8)
        expect(more).to_be_hidden()

    def test_localstorage_disabled_homepage_and_submit_still_work(
            self, server, page, set_script):
        # Safari-private-mode class of failure: ANY storage touch throws.
        set_script([("progress", _p("data_collection", facts=1))])
        page.add_init_script("""
            Object.defineProperty(window, 'localStorage', {
                get() { throw new Error('denied'); }});
            Object.defineProperty(window, 'sessionStorage', {
                get() { throw new Error('denied'); }});
        """)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server)
        expect(page.locator("#history")).to_be_hidden()
        page.fill("#query", "Jane Doe")
        page.click("#submit-btn")
        # navigation still happens; the run page loads watch-only
        page.wait_for_url(re.compile(r"/research/"), timeout=15000)
        expect(page.locator("#status")).to_be_visible()
        expect(page.locator("#cancel-btn")).to_be_hidden()  # not "owner"
        ours = [e for e in errors if "Turnstile" not in e]
        assert ours == [], f"page errors: {ours}"
