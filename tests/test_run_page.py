"""
P0 Step 2 — GET /research/{job_id}: the run page's server contract.

New public HTML endpoint → the full header/indexability contract is stated
and asserted (REVIEW-LEARNINGS rule): X-Robots-Tag noindex + nosniff +
Cache-Control no-store + the strict run-page CSP INCLUDING frame-ancestors
'none' (R6). The route param is a plain str: any value serves the shell and
the CLIENT resolves validity via the snapshot's 404/422 (R8) — no raw 422
HTML from the page route. The page is READ-ONLY (no writer to Job.progress).
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import uuid

from src.api.routes import RUN_PAGE_CSP
from tests.test_api_endpoints import client  # noqa: F401  (app fixture)

VALID_ID = str(uuid.uuid4())


class TestRunPageHeaders:
    def test_200_with_full_header_contract(self, client):
        r = client.get(f"/research/{VALID_ID}")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert r.headers["x-robots-tag"] == "noindex"
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["cache-control"] == "no-store"
        assert r.headers["content-security-policy"] == RUN_PAGE_CSP

    def test_csp_has_frame_ancestors_and_connect_self(self):
        # R6: frame-ancestors never inherits from default-src; the page has
        # framable cancel/finish buttons, so this directive is load-bearing.
        assert "frame-ancestors 'none'" in RUN_PAGE_CSP
        # SSE (EventSource) + the snapshot fetch are governed by connect-src.
        assert "connect-src 'self'" in RUN_PAGE_CSP

    def test_homepage_headers_unchanged(self, client):
        # The homepage stays indexable + CSP-less (out of P0 scope).
        r = client.get("/")
        assert "content-security-policy" not in r.headers
        assert "x-robots-tag" not in r.headers


class TestRunPageShell:
    def test_malformed_id_still_serves_the_shell(self, client):
        # str param (R8): the client renders "not found" from the API's
        # 404/422 — the page route never emits FastAPI's raw 422 HTML.
        for bad in ("not-a-uuid", "123", "a" * 500):
            r = client.get(f"/research/{bad}")
            assert r.status_code == 200, bad
            assert "data-job-id" in r.text

    def test_job_id_is_autoescaped_in_the_data_attribute(self, client):
        # Percent-encoded attack (what a browser actually sends) — FastAPI
        # decodes it into the str param; Jinja autoescape must neuter it in
        # the attribute context. Slash-free payload: Starlette decodes %2F
        # BEFORE routing, so an id containing / never reaches this route.
        r = client.get("/research/%22%3E%3Cimg%20src%3Dx%20onerror%3Dalert(1)%3E")
        assert r.status_code == 200
        assert '"><img' not in r.text
        assert "<img src=x onerror" not in r.text
        assert "&lt;img" in r.text or "&#34;" in r.text

    def test_valid_id_lands_in_data_attribute(self, client):
        r = client.get(f"/research/{VALID_ID}")
        assert f'data-job-id="{VALID_ID}"' in r.text

    def test_nav_new_research_cta_is_run_page_only(self, client):
        # D4's visible Home affordance (human 2026-07-21): task-named CTA
        # with the keeps-running reassurance tooltip — run page only (the
        # homepage IS the new-research surface).
        run = client.get(f"/research/{VALID_ID}").text
        assert 'class="nav-cta"' in run
        assert "New research" in run
        assert "keeps running" in run          # the title tooltip
        home = client.get("/").text
        assert 'class="nav-cta"' not in home
        # brand stays on both — the CTA augments, never replaces
        assert 'class="nav-brand"' in run and 'class="nav-brand"' in home

    def test_head_is_generic_and_turnstile_free(self, client):
        # R7: Turnstile stays homepage-only (the run-page CSP would block
        # it); OG is generic — the subject name never rides in URL or meta.
        r = client.get(f"/research/{VALID_ID}")
        assert "challenges.cloudflare.com" not in r.text
        assert 'property="og:url"' not in r.text
        assert "report_header" not in r.text  # no marketing og:image

    def test_run_page_is_read_only_get(self, client):
        # No mutating method on the page route.
        assert client.post(f"/research/{VALID_ID}").status_code == 405


class TestRunPageNeverRateLimited:
    def test_run_page_gets_unlimited_while_post_bucket_exhausted(self, client):
        # Gate 8: the 3/hour bucket applies to the POST only — the shell and
        # snapshot GETs (read-only, UUID-gated, $0) must keep answering long
        # past it, or a shared link would die with the sharer's quota.
        job_ids = []
        for i in range(3):
            r = client.post("/api/research",
                            json={"query": f"Person {i}",
                                  "turnstile_token": "t"})
            assert r.status_code == 202, r.text
            job_ids.append(r.json()["job_id"])
        r = client.post("/api/research",
                        json={"query": "Person 3", "turnstile_token": "t"})
        assert r.status_code == 429
        for _ in range(10):   # 30 page + 30 snapshot GETs ≫ 3/hour
            for jid in job_ids:
                assert client.get(f"/research/{jid}").status_code == 200
                assert client.get(f"/api/research/{jid}").status_code == 200


class TestSharedPartialDivergenceGuard:
    def test_both_pages_render_the_same_shared_partials(self, client):
        # Acceptance gate item 9: one shared-partial source. Both pages must
        # carry the _styles.html token set and the _nav.html header.
        home = client.get("/").text
        run = client.get(f"/research/{VALID_ID}").text
        for token in ("cat-biographical", 'id="site-nav"', "dra_owned_v1",
                      "#assumption-banner[hidden]"):
            assert token in home, f"homepage lost shared token {token}"
            assert token in run, f"run page lost shared token {token}"

    def test_run_page_keeps_the_technical_log_seam(self, client):
        # D6: the commented techText/techMode machinery is the P8 seam —
        # present, NOT enabled (no live #log-toggle button markup).
        r = client.get(f"/research/{VALID_ID}").text
        assert "techText" in r
        assert "TECHNICAL-LOG" in r
