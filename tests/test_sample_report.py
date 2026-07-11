"""
Phase D4 — /sample-report header + content contract (review R1, D5 gate).

Public HTML endpoints with inline scripts: they MUST inherit the job_report
header contract (REPORT_CSP + nosniff) and add X-Robots-Tag: noindex — the
report HTML carries no robots meta, so the header IS the noindex mechanism.
Content must be REGENERATED artifacts (the Feb 2026 sample predated the
markupsafe escaping chokepoint and was a stored-XSS artifact — never served).
Two samples ship (human decision 2026-07-10): person (Jensen Huang) and
company (Stripe); /sample-report redirects to the person report.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pathlib import Path

import pytest

from src.api import routes as routes_mod
from src.api.routes import REPORT_CSP
from tests.test_api_endpoints import client  # noqa: F401  (app fixture)

KINDS = {"person": "Jensen Huang", "company": "Stripe"}

# The homepage screenshot deep-links land on these (D3 click-through).
SECTION_ANCHORS = ['id="report-top"', 'id="factsSection"', 'id="risksSection"',
                   'id="connectionsSection"', 'id="trendsSection"']


class TestSampleReport:
    @pytest.mark.parametrize("kind", list(KINDS))
    def test_200_with_full_header_contract(self, client, kind):
        r = client.get(f"/sample-report/{kind}")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert r.headers["content-security-policy"] == REPORT_CSP
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-robots-tag"] == "noindex"

    @pytest.mark.parametrize("kind,subject", KINDS.items())
    def test_content_subject_and_navigation(self, client, kind, subject):
        r = client.get(f"/sample-report/{kind}")
        text = r.text
        assert text.lstrip().lower().startswith("<!doctype html")
        assert subject in text
        assert 'class="home-link"' in text          # back-to-search link
        for anchor in SECTION_ANCHORS:              # screenshot deep-links
            assert anchor in text, f"missing {anchor} in {kind} sample"

    @pytest.mark.parametrize("kind", list(KINDS))
    def test_content_is_post_escaping_vintage(self, client, kind):
        """The Feb artifact carried onerror/javascript: in scraped content."""
        body = client.get(f"/sample-report/{kind}").text.lower()
        assert "onerror" not in body
        assert "javascript:" not in body

    def test_bare_path_redirects_to_person(self, client):
        r = client.get("/sample-report", follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"] == "/sample-report/person"

    def test_unknown_kind_404(self, client):
        assert client.get("/sample-report/nope").status_code == 404

    def test_missing_file_is_graceful_404(self, client, monkeypatch):
        monkeypatch.setitem(routes_mod.SAMPLE_REPORTS, "person",
                            Path("/nonexistent/sample_report.html"))
        assert client.get("/sample-report/person").status_code == 404
