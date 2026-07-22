"""
P0 Step 1 — shared Jinja partials (research/phase-runpage-plan-2026-07-13.md).

_styles.html / _nav.html / _shared_js.html are the single source for the
design tokens, the persistent nav, and page-agnostic storage helpers. The
homepage includes all three; run.html (Step 2) includes the SAME files —
one shared-partial source is the divergence guard (acceptance gate item 9).
R7: only styles/nav/utils are shared — each page keeps its own <head>, so
the Turnstile <script> and marketing OG stay homepage-only.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pathlib import Path

from tests.test_api_endpoints import client  # noqa: F401  (app fixture)

TEMPLATES = Path(__file__).parent.parent / "src" / "api" / "templates"
PARTIALS = ("_styles.html", "_nav.html", "_shared_js.html")


class TestPartialSource:
    def test_partials_exist(self):
        for name in PARTIALS:
            assert (TEMPLATES / name).exists(), f"{name} missing"

    def test_index_includes_partials_and_owns_no_style_copy(self):
        index = (TEMPLATES / "index.html").read_text()
        for name in PARTIALS:
            assert f'{{% include "{name}" %}}' in index, f"{name} not included"
        # Divergence guard: the design tokens live ONLY in _styles.html.
        # (.cat-biographical is a token every page needs — if it re-appears
        # in index.html, someone re-inlined the styles.)
        assert "cat-biographical" not in index

    def test_nav_partial_is_static_markup_only(self):
        nav = (TEMPLATES / "_nav.html").read_text()
        assert "{{" not in nav, "nav must carry no data (no injection surface)"
        assert 'href="/"' in nav

    def test_shared_js_is_page_agnostic(self):
        js = (TEMPLATES / "_shared_js.html").read_text()
        # Storage keys + reduced-motion only — no render code, no terminal
        # handlers, no Turnstile (R1/R7 boundary).
        for key in ("dra_history_v1", "dra_owned_v1", "dra_refine",
                    "REDUCED_MOTION"):
            assert key in js
        for banned in ("turnstile", "EventSource", "showBanner", "watch("):
            assert banned not in js, f"page-specific symbol '{banned}' leaked"


class TestRunCodeSingleHomed:
    def test_index_owns_no_run_experience_after_step3(self):
        # P0 Step 3: the run render JS + #status markup are single-homed on
        # run.html — any of these re-appearing in index.html is divergence.
        index = (TEMPLATES / "index.html").read_text()
        for banned in ('id="status"', "EventSource", "PHASE_LABEL",
                       "function watch", "function resetPanel",
                       "techText", "showBanner", "cancelJob"):
            assert banned not in index, f"run-page symbol '{banned}' in index"
        # The navigation handoff + both store writes are present.
        assert "location.assign('/research/'" in index
        assert "ownJob(job.job_id" in index
        assert "recordHistoryEntry(job.job_id" in index
        # R2 re-arm hook reads-and-clears the one-shot key.
        assert "ssTakeJSON(REFINE_KEY)" in index


class TestHomepageRendersPartials:
    def test_index_renders_all_three_partials(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "#assumption-banner[hidden]" in r.text   # _styles.html token
        assert 'id="site-nav"' in r.text                # _nav.html
        assert "dra_owned_v1" in r.text                 # _shared_js.html
        # R7: homepage keeps its own head — Turnstile + marketing OG stay.
        assert "challenges.cloudflare.com/turnstile" in r.text
        assert 'property="og:url" content="https://tryvettr.com/"' in r.text
