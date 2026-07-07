"""
XSS regression tests for the shared report renderer (edge case #10).

Facts are scraped from the adversarial open web; a planted <script> in any
fact/risk/connection/metadata string (or the query itself) must arrive in
the report HTML escaped, never executable.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest

from src.reporting.html_report import (
    _escape_deep,
    calculate_quality_score,
    render_html_report,
)

PAYLOAD = '<script>alert("xss")</script>'
IMG_PAYLOAD = '<img src=x onerror=alert(1)>'


def _result_with_payloads():
    return {
        "facts": [
            {"content": f"He said {PAYLOAD} on stage", "category": "professional",
             "confidence": 0.9, "source": f"https://evil.example/{IMG_PAYLOAD}"},
            {"content": "Normal fact", "category": "biographical", "confidence": 0.8,
             "source": "https://ok.example"},
        ],
        "risk_flags": [
            {"description": f"Risky {PAYLOAD}", "severity": "high",
             "category": "legal", "confidence": 0.7, "evidence": ["fact_0"]},
        ],
        "connections": [
            {"entity": IMG_PAYLOAD, "relationship_type": "employer", "strength": 0.5},
        ],
        "metadata": {
            "coverage": {"average": 0.5},
            "iterations": 1,
            "duration_seconds": 10.0,
            "summary": f"Summary with {PAYLOAD}",
        },
    }


class TestEscapeDeep:
    def test_strings_escaped_everywhere(self):
        out = _escape_deep({"a": [PAYLOAD, {"b": IMG_PAYLOAD}], "n": 3, "f": 0.5, "none": None})
        assert "<script>" not in str(out)
        assert out["a"][0] == "&lt;script&gt;alert(&#34;xss&#34;)&lt;/script&gt;"
        assert out["n"] == 3 and out["f"] == 0.5 and out["none"] is None

    def test_original_not_mutated(self):
        original = {"facts": [{"content": PAYLOAD}]}
        _escape_deep(original)
        assert original["facts"][0]["content"] == PAYLOAD


class TestRenderedReport:
    def test_planted_scripts_arrive_escaped(self):
        html = render_html_report(_result_with_payloads(), "Test Subject", 12.3)
        assert PAYLOAD not in html               # raw script never survives
        assert IMG_PAYLOAD not in html           # raw img/onerror never survives
        assert "onerror=alert" not in html       # no executable remnant anywhere
        assert "&lt;script&gt;" in html          # fact text present, but inert

    def test_query_is_escaped(self):
        html = render_html_report(
            _result_with_payloads(), f'Evil "{PAYLOAD}" Corp', 1.0
        )
        assert PAYLOAD not in html

    def test_report_still_renders_structure(self):
        html = render_html_report(_result_with_payloads(), "Test Subject", 12.3)
        assert html.lstrip().lower().startswith("<!doctype")
        assert html.rstrip().endswith("</html>")

    def test_quality_score_unaffected_by_extraction(self):
        r = _result_with_payloads()
        score = calculate_quality_score(
            facts=r["facts"], risk_flags=r["risk_flags"],
            connections=r["connections"], coverage=r["metadata"]["coverage"],
        )
        assert 0 <= score["score"] <= 100
        assert "grade" in score
