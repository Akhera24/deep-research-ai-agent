"""
Static XSS-sink checks for the loading-screen template (PLAN.md Step A4(b)).

The feed/ticker render UNTRUSTED scraped web content client-side; the DOM
boundary must be textContent-only. This bans the whole HTML-injection sink
family — not just innerHTML — so a different sink can't sneak in under a
green test (reviewer instruction, 2026-07-09). Runtime inertness is proven
by the Playwright test (tests/e2e, A4(c)); this is the fast-gate backstop.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pathlib import Path

TEMPLATE = Path(__file__).parent.parent / "src" / "api" / "templates" / "index.html"

HTML_INJECTION_SINKS = [
    "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write",
    "createContextualFragment", "srcdoc",
]


def test_template_bans_html_injection_sink_family():
    src = TEMPLATE.read_text()
    for sink in HTML_INJECTION_SINKS:
        assert sink not in src, f"HTML-injection sink '{sink}' found in index.html"


def test_untrusted_text_rendered_via_textcontent():
    src = TEMPLATE.read_text()
    assert "textContent" in src
    # Feed/ticker rows are rebuilt from the snapshot (reconnect-idempotent).
    assert "replaceChildren" in src
