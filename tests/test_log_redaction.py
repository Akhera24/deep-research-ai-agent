"""
Regression: production logs must not carry the raw research subject (§12.S3).

Railway retains logs, which would otherwise outlive the 7-day PII wipe.
Redaction is production-only by design (dev keeps the query for debugging).
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import contextlib
import io

from config import logging_config


def _emit(env, monkeypatch, **kw):
    monkeypatch.setattr(logging_config.settings, "ENVIRONMENT", env)
    logging_config.configure_structlog()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        logging_config.get_logger("t").info("Starting research", **kw)
    return buf.getvalue()


def test_pii_keys_redacted_in_production(monkeypatch):
    out = _emit("production", monkeypatch, target="Jane SECRET",
                extra={"query": "Jane SECRET", "count": 5})
    assert "Jane SECRET" not in out
    assert "<redacted>" in out
    assert "5" in out  # non-PII field survives
    monkeypatch.undo()
    logging_config.configure_structlog()  # restore dev config for other tests


def test_development_keeps_query_for_debugging(monkeypatch):
    out = _emit("development", monkeypatch, target="Jane DEBUG")
    assert "Jane DEBUG" in out
    monkeypatch.undo()
    logging_config.configure_structlog()
