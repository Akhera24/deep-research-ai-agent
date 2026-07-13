"""
C1.1 — /api/disambiguate endpoint + pre-flight ticket (PLAN.md Rev 3.8).

Ticket unit tests run against security.py directly (mint/consume, TTL,
IP + canonical-query binding, single-use nonce, tampering, malformed input).
Endpoint tests run the real app with the pre-flight module faked: gate order
(ratelimit → budget → turnstile → spend), ledger row with job_id NULL,
fail-open ticket delivery, and the /api/research ticket path (replay → 403,
byte-identical resubmit → accepted).
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import uuid

import pytest
from sqlalchemy import select

from src.api import db as api_db
from src.api import routes as routes_mod
from src.api import security as security_mod
from src.api.db import SpendLedger
from src.core import preflight as preflight_mod
from src.core.preflight import PreflightCandidate, PreflightResult
from tests.test_api_endpoints import (  # noqa: F401 — shared app fixture
    ADMIN, FakeOrchestrator, _turnstile_fail, _turnstile_ok, _wait_terminal,
    client,
)

XSS_NAME = 'Jane <img src=x onerror=alert(1)>'


# ---------------------------------------------------------------------------
# Ticket unit tests (no app)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fresh_ticket_state():
    security_mod.reset_ticket_state()
    yield
    security_mod.reset_ticket_state()


class TestTicket:
    def test_mint_consume_roundtrip(self):
        t = security_mod.mint_preflight_ticket("iphash", "Jane Doe")
        ok, reason = security_mod.consume_preflight_ticket(t, "iphash", "Jane Doe")
        assert ok, reason

    def test_single_use_replay_rejected(self):
        t = security_mod.mint_preflight_ticket("iphash", "Jane Doe")
        assert security_mod.consume_preflight_ticket(t, "iphash", "Jane Doe")[0]
        ok, reason = security_mod.consume_preflight_ticket(t, "iphash", "Jane Doe")
        assert not ok
        assert "used" in reason

    def test_wrong_ip_rejected(self):
        t = security_mod.mint_preflight_ticket("iphash-a", "Jane Doe")
        ok, _ = security_mod.consume_preflight_ticket(t, "iphash-b", "Jane Doe")
        assert not ok

    def test_query_binding_rejects_different_query(self):
        t = security_mod.mint_preflight_ticket("iphash", "Jane Doe")
        ok, _ = security_mod.consume_preflight_ticket(t, "iphash", "John Doe")
        assert not ok

    def test_canonical_query_match_is_exact_bytes(self):
        # both endpoints hash the POST-validation canonical query (R6):
        # the strings compared here are both post-strip values
        t = security_mod.mint_preflight_ticket("iphash", "Jane Doe")
        ok, _ = security_mod.consume_preflight_ticket(t, "iphash", "Jane Doe")
        assert ok

    def test_expired_rejected(self, monkeypatch):
        real_time = security_mod.time.time
        t = security_mod.mint_preflight_ticket("iphash", "Jane Doe")
        monkeypatch.setattr(security_mod.time, "time",
                            lambda: real_time() + 301)
        ok, reason = security_mod.consume_preflight_ticket(t, "iphash", "Jane Doe")
        assert not ok
        assert "expired" in reason

    def test_future_dated_rejected(self, monkeypatch):
        real_time = security_mod.time.time
        monkeypatch.setattr(security_mod.time, "time",
                            lambda: real_time() + 3600)
        t = security_mod.mint_preflight_ticket("iphash", "Jane Doe")
        monkeypatch.setattr(security_mod.time, "time", real_time)
        ok, _ = security_mod.consume_preflight_ticket(t, "iphash", "Jane Doe")
        assert not ok

    @pytest.mark.parametrize("bad", [
        "", "a", "a.b", "a.b.c.d", "notanint.deadbeef.00", "1.2",
        "99999999999.nonce." + "0" * 64,
    ])
    def test_malformed_rejected_without_crash(self, bad):
        ok, _ = security_mod.consume_preflight_ticket(bad, "iphash", "Jane Doe")
        assert not ok

    def test_tampered_mac_rejected(self):
        t = security_mod.mint_preflight_ticket("iphash", "Jane Doe")
        issued, nonce, mac = t.split(".")
        flipped = ("0" if mac[0] != "0" else "1") + mac[1:]
        ok, _ = security_mod.consume_preflight_ticket(
            f"{issued}.{nonce}.{flipped}", "iphash", "Jane Doe")
        assert not ok

    def test_secret_rotation_invalidates_old_tickets(self, monkeypatch):
        monkeypatch.setattr(api_db.settings, "PREFLIGHT_TICKET_SECRET", "key-one")
        t = security_mod.mint_preflight_ticket("iphash", "Jane Doe")
        monkeypatch.setattr(api_db.settings, "PREFLIGHT_TICKET_SECRET", "key-two")
        ok, _ = security_mod.consume_preflight_ticket(t, "iphash", "Jane Doe")
        assert not ok

    def test_none_ip_hash_binds_consistently(self):
        t = security_mod.mint_preflight_ticket(None, "Jane Doe")
        assert security_mod.consume_preflight_ticket(t, None, "Jane Doe")[0]
        t2 = security_mod.mint_preflight_ticket(None, "Jane Doe")
        assert not security_mod.consume_preflight_ticket(t2, "iphash", "Jane Doe")[0]


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------

def _fake_preflight(result):
    calls = []

    async def fake(query, hints=None, **kwargs):
        calls.append({"query": query, "hints": hints})
        return result

    fake.calls = calls
    return fake


PICK_RESULT = PreflightResult(
    decision="pick",
    candidates=[
        PreflightCandidate(canonical_name=XSS_NAME,
                           descriptor="javascript:alert(1) — attacker, Evil (web)",
                           disambiguators=["Evil Corp"],
                           domain_mass=3, domains=["a.com", "b.com", "c.org"]),
        PreflightCandidate(canonical_name="Jane Doe",
                           descriptor="Jane Doe — Barrister, London (UK)",
                           domain_mass=2, domains=["x.co.uk", "y.com"]),
    ],
    cost=0.0042,
)


class TestDisambiguateEndpoint:
    def test_happy_path_returns_candidates_raw_plus_ticket_and_nosniff(
            self, client, monkeypatch):
        fake = _fake_preflight(PICK_RESULT)
        monkeypatch.setattr(preflight_mod, "discover_candidates", fake)
        r = client.post("/api/disambiguate",
                        json={"query": "Jane Doe", "turnstile_token": "tok"})
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] == "pick"
        assert body["ticket"]
        assert r.headers["x-content-type-options"] == "nosniff"
        # scraped/LLM text arrives RAW in JSON — the client renders
        # textContent-only (server-side HTML-escaping would double-escape)
        assert body["candidates"][0]["canonical_name"] == XSS_NAME
        assert body["candidates"][0]["domain_mass"] == 3
        # hints threaded through
        assert fake.calls[0]["query"] == "Jane Doe"

    def test_hints_forwarded_and_validated(self, client, monkeypatch):
        fake = _fake_preflight(PICK_RESULT)
        monkeypatch.setattr(preflight_mod, "discover_candidates", fake)
        r = client.post("/api/disambiguate", json={
            "query": "Jane Doe", "turnstile_token": "tok",
            "hints": {"company": "Stripe", "role": "VP Engineering"},
        })
        assert r.status_code == 200
        assert fake.calls[0]["hints"] == {"company": "Stripe",
                                          "role": "VP Engineering"}

    def test_hint_with_forbidden_chars_422(self, client):
        r = client.post("/api/disambiguate", json={
            "query": "Jane Doe", "turnstile_token": "tok",
            "hints": {"company": "<script>evil</script>"},
        })
        assert r.status_code == 422

    def test_query_validation_mirrors_research(self, client):
        r = client.post("/api/disambiguate", json={
            "query": "https://evil.example/x", "turnstile_token": "tok"})
        assert r.status_code == 422

    def test_ledger_row_written_with_null_job_id(self, client, monkeypatch):
        fake = _fake_preflight(PICK_RESULT)
        monkeypatch.setattr(preflight_mod, "discover_candidates", fake)
        r = client.post("/api/disambiguate",
                        json={"query": "Jane Doe", "turnstile_token": "tok"})
        assert r.status_code == 200

        async def rows():
            async with api_db.get_sessionmaker()() as session:
                return (await session.execute(select(SpendLedger))).scalars().all()

        ledger_rows = client.portal.call(rows)
        assert len(ledger_rows) == 1
        assert ledger_rows[0].job_id is None
        assert float(ledger_rows[0].amount_usd) == pytest.approx(0.0042)

    def test_budget_exhausted_503_with_zero_spend(self, client, monkeypatch):
        monkeypatch.setattr(api_db.settings, "MONTHLY_BUDGET_USD", 0.0)
        fake = _fake_preflight(PICK_RESULT)
        monkeypatch.setattr(preflight_mod, "discover_candidates", fake)
        r = client.post("/api/disambiguate",
                        json={"query": "Jane Doe", "turnstile_token": "tok"})
        assert r.status_code == 503
        assert fake.calls == []          # gate ran BEFORE any spend

    def test_turnstile_failure_403_no_spend(self, client, monkeypatch):
        monkeypatch.setattr(routes_mod, "verify_turnstile", _turnstile_fail)
        fake = _fake_preflight(PICK_RESULT)
        monkeypatch.setattr(preflight_mod, "discover_candidates", fake)
        r = client.post("/api/disambiguate",
                        json={"query": "Jane Doe", "turnstile_token": "bad"})
        assert r.status_code == 403
        assert fake.calls == []

    def test_admin_skips_turnstile(self, client, monkeypatch):
        monkeypatch.setattr(routes_mod, "verify_turnstile", _turnstile_fail)
        fake = _fake_preflight(PICK_RESULT)
        monkeypatch.setattr(preflight_mod, "discover_candidates", fake)
        r = client.post("/api/disambiguate", json={"query": "Jane Doe"},
                        headers=ADMIN)
        assert r.status_code == 200

    def test_rate_limit_429_past_3x_report_limit_admin_exempt(
            self, client, monkeypatch):
        fake = _fake_preflight(PICK_RESULT)
        monkeypatch.setattr(preflight_mod, "discover_candidates", fake)
        limit = api_db.settings.RATE_LIMIT_REPORTS_PER_HOUR * 3
        for _ in range(limit):
            assert client.post(
                "/api/disambiguate",
                json={"query": "Jane Doe", "turnstile_token": "tok"},
            ).status_code == 200
        r = client.post("/api/disambiguate",
                        json={"query": "Jane Doe", "turnstile_token": "tok"})
        assert r.status_code == 429
        # admin exempt
        r = client.post("/api/disambiguate", json={"query": "Jane Doe"},
                        headers=ADMIN)
        assert r.status_code == 200

    def test_fail_open_error_decision_still_ships_ticket(
            self, client, monkeypatch):
        fake = _fake_preflight(PreflightResult(decision="error", cost=0.0))
        monkeypatch.setattr(preflight_mod, "discover_candidates", fake)
        r = client.post("/api/disambiguate",
                        json={"query": "Jane Doe", "turnstile_token": "tok"})
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] == "error"
        assert body["candidates"] == []
        assert body["ticket"]            # client proceeds unscoped WITH it


class TestResearchTicketPath:
    def _get_ticket(self, client, monkeypatch, query="Jane Doe"):
        fake = _fake_preflight(PICK_RESULT)
        monkeypatch.setattr(preflight_mod, "discover_candidates", fake)
        r = client.post("/api/disambiguate",
                        json={"query": query, "turnstile_token": "tok"})
        assert r.status_code == 200
        return r.json()["ticket"]

    def test_valid_ticket_accepted_without_turnstile(self, client, monkeypatch):
        ticket = self._get_ticket(client, monkeypatch)
        # turnstile now FAILS — only the ticket path can admit this request
        monkeypatch.setattr(routes_mod, "verify_turnstile", _turnstile_fail)
        r = client.post("/api/research",
                        json={"query": "Jane Doe", "preflight_ticket": ticket})
        assert r.status_code == 202
        _wait_terminal(client, r.json()["job_id"])

    def test_replayed_ticket_403(self, client, monkeypatch):
        ticket = self._get_ticket(client, monkeypatch)
        monkeypatch.setattr(routes_mod, "verify_turnstile", _turnstile_fail)
        first = client.post("/api/research",
                            json={"query": "Jane Doe", "preflight_ticket": ticket})
        assert first.status_code == 202
        replay = client.post("/api/research",
                             json={"query": "Jane Doe", "preflight_ticket": ticket})
        assert replay.status_code == 403
        _wait_terminal(client, first.json()["job_id"])

    def test_query_mismatch_403(self, client, monkeypatch):
        ticket = self._get_ticket(client, monkeypatch)
        monkeypatch.setattr(routes_mod, "verify_turnstile", _turnstile_fail)
        r = client.post("/api/research",
                        json={"query": "John Smith", "preflight_ticket": ticket})
        assert r.status_code == 403

    def test_whitespace_variant_query_still_accepted_via_canonicalization(
            self, client, monkeypatch):
        # R6: the ticket binds the POST-validation canonical query, so a
        # client resending with padding must still verify (validator strips).
        ticket = self._get_ticket(client, monkeypatch)
        monkeypatch.setattr(routes_mod, "verify_turnstile", _turnstile_fail)
        r = client.post("/api/research",
                        json={"query": "  Jane Doe  ", "preflight_ticket": ticket})
        assert r.status_code == 202
        _wait_terminal(client, r.json()["job_id"])

    def test_expired_ticket_403(self, client, monkeypatch):
        ticket = self._get_ticket(client, monkeypatch)
        real_time = security_mod.time.time
        monkeypatch.setattr(security_mod.time, "time",
                            lambda: real_time() + 301)
        monkeypatch.setattr(routes_mod, "verify_turnstile", _turnstile_fail)
        r = client.post("/api/research",
                        json={"query": "Jane Doe", "preflight_ticket": ticket})
        assert r.status_code == 403

    def test_garbage_ticket_403(self, client, monkeypatch):
        monkeypatch.setattr(routes_mod, "verify_turnstile", _turnstile_fail)
        r = client.post("/api/research",
                        json={"query": "Jane Doe",
                              "preflight_ticket": "not.a.ticket"})
        assert r.status_code == 403

    def test_no_ticket_falls_back_to_turnstile(self, client):
        # regression: the existing Turnstile path is untouched
        r = client.post("/api/research",
                        json={"query": "Jane Doe", "turnstile_token": "tok"})
        assert r.status_code == 202
        _wait_terminal(client, r.json()["job_id"])
