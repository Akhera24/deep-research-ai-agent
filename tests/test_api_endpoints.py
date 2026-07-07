"""
End-to-end tests for the job API against a temp sqlite DB.

External boundaries are faked (Turnstile verification, the research
orchestrator) — everything else (routes, validation, budget gates, jobs
service, ledger, reaper, SSE, escaping) runs for real.

Covers PLAN.md edge cases #4 (garbage query), #8 (zombie jobs), #10 (stored
XSS), #11 (cost runaway) and PHASE3_DESIGN §11.R1/R2 and §12.S1/S2.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import time
import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from src.api import db as api_db
from src.api import jobs as jobs_mod
from src.api import routes as routes_mod
from src.api.db import (
    COMPLETED, EXPIRED, FAILED, QUEUED, RUNNING, Job, utcnow,
)
from src.api.reaper import reap_once

XSS = '<script>alert("xss")</script>'


class FakeOrchestrator:
    """Stands in for ResearchOrchestrator: instant, no network, no spend."""

    total_cost = 0.05          # overridden per-test via class attribute
    fail_with: Exception | None = None

    def __init__(self, max_iterations=10, enable_checkpoints=False):
        class _Cfg:
            total_cost = 0.0
        class _Client:
            config = _Cfg()
        self._client = _Client()
        self._client.config.total_cost = type(self).total_cost
        class _Router:
            pass
        self.router = _Router()
        self.router.clients = {"fake": self._client}

    async def research(self, query, context=None, progress_callback=None):
        if progress_callback is not None:
            await progress_callback({
                "node": "extracting_facts", "iteration": 1, "max_iterations": 10,
                "facts": 2, "coverage": {"average": 0.4},
            })
        if type(self).fail_with is not None:
            raise type(self).fail_with
        return {
            "facts": [
                {"content": f"Planted {XSS} fact", "category": "professional",
                 "confidence": 0.9, "source": "https://evil.example"},
                {"content": "Normal fact", "category": "biographical",
                 "confidence": 0.8, "source": "https://ok.example"},
            ],
            "risk_flags": [],
            "connections": [],
            "metadata": {"coverage": {"average": 0.4}, "iterations": 1,
                         "duration_seconds": 0.1},
        }


async def _turnstile_ok(token, ip):
    return True, "ok"


async def _turnstile_fail(token, ip):
    return False, "challenge verification failed"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """App on a fresh sqlite DB with faked externals; admin token set."""
    import asyncio

    monkeypatch.setattr(api_db.settings, "DATABASE_URL", f"sqlite:///{tmp_path}/api.db")
    monkeypatch.setattr(api_db.settings, "ADMIN_BYPASS_TOKEN", "test-admin-token")
    monkeypatch.setattr(api_db.settings, "ENVIRONMENT", "development")
    # Pin Cloudflare's official TEST sitekey so CI never depends on the real
    # keys a developer may have in their local .env (reviewer instruction).
    monkeypatch.setattr(api_db.settings, "TURNSTILE_SITE_KEY", "1x00000000000000000000AA")
    monkeypatch.setattr(api_db.settings, "TURNSTILE_SECRET_KEY", "1x0000000000000000000000000000000AA")
    monkeypatch.setattr(routes_mod, "verify_turnstile", _turnstile_ok)
    monkeypatch.setattr(jobs_mod, "ResearchOrchestrator", FakeOrchestrator)
    FakeOrchestrator.total_cost = 0.05
    FakeOrchestrator.fail_with = None
    jobs_mod.reset_state()
    try:
        routes_mod.limiter.reset()
    except Exception:
        pass

    # Engine may be bound to a previous test's loop — drop it.
    asyncio.get_event_loop_policy().new_event_loop()
    import src.api.main as main_mod
    from src.api.db import dispose_engine
    with TestClient(main_mod.app) as c:
        yield c
    # TestClient shutdown ran lifespan cleanup (dispose_engine) already.


ADMIN = {"X-Admin-Token": "test-admin-token"}


def _wait_terminal(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/research/{job_id}", headers=ADMIN)
        if r.status_code == 200 and r.json()["status"] in (COMPLETED, FAILED):
            return r.json()
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached a terminal state")


class TestBasics:
    def test_healthz(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "db": "ok"}

    def test_index_page_renders_widget_and_tos(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "cf-turnstile" in r.text
        assert "1x00000000000000000000AA" in r.text     # test sitekey until real keys land
        assert "takedown" in r.text.lower() or "Removal" in r.text


class TestQueryValidation:
    """Edge case #4 + §12.S1 (business names are first-class)."""

    @pytest.mark.parametrize("q", [
        "AT&T", "Berkshire Hathaway Inc.", "O'Brien-Smith, Jr. (CEO) + Co",
        "Jensen Huang", "3M", "Ernst & Young",
    ])
    def test_business_names_accepted(self, client, q):
        r = client.post("/api/research", json={"query": q}, headers=ADMIN)
        assert r.status_code == 202, (q, r.text)

    @pytest.mark.parametrize("q", [
        "", "   ", "x" * 201, "https://evil.example/payload",
        "see http://x.co", XSS, "run `rm -rf`", "a{b}", "back\\slash",
    ])
    def test_garbage_rejected_422(self, client, q):
        r = client.post("/api/research", json={"query": q}, headers=ADMIN)
        assert r.status_code == 422, (q, r.status_code)


class TestJobLifecycle:
    def test_full_happy_path_and_xss_escaping(self, client):
        r = client.post("/api/research", json={"query": "Test Subject"}, headers=ADMIN)
        assert r.status_code == 202
        body = r.json()
        job_id = body["job_id"]
        assert body["events_url"].endswith("/events")

        status = _wait_terminal(client, job_id)
        assert status["status"] == COMPLETED
        assert status["cost_usd"] == pytest.approx(0.05)         # admin sees cost
        assert status["progress"]["quality_score"] is not None

        # Non-admin status: no cost leak
        r = client.get(f"/api/research/{job_id}")
        assert "cost_usd" not in r.json()

        # Report: renders, escaped, CSP header (edge case #10)
        r = client.get(f"/api/research/{job_id}/report")
        assert r.status_code == 200
        assert XSS not in r.text
        assert "&lt;script&gt;" in r.text
        assert "Content-Security-Policy" in r.headers
        # S3: no report file leaked to disk
        assert not any(f.startswith("research_report_Test_Subject") for f in os.listdir("."))

    def test_unknown_job_404(self, client):
        r = client.get(f"/api/research/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_report_on_unfinished_job_409(self, client, monkeypatch):
        # Freeze the job in queued state by making the runner never start:
        # occupy the semaphore fully.
        import asyncio

        async def _hog():
            sem = jobs_mod._get_semaphore()
            for _ in range(api_db.settings.MAX_CONCURRENT_JOBS):
                await sem.acquire()

        client.portal.call(_hog)
        r = client.post("/api/research", json={"query": "Queued Person"}, headers=ADMIN)
        job_id = r.json()["job_id"]
        r = client.get(f"/api/research/{job_id}/report")
        assert r.status_code == 409
        jobs_mod.reset_state()

    def test_expired_job_410_everywhere(self, client):
        job_id = uuid.uuid4()

        async def _insert():
            async with api_db.get_sessionmaker()() as s:
                s.add(Job(id=job_id, query=None, status=EXPIRED,
                          created_at=utcnow(), expires_at=utcnow(), progress={}))
                await s.commit()

        client.portal.call(_insert)
        assert client.get(f"/api/research/{job_id}").status_code == 410
        assert client.get(f"/api/research/{job_id}/report").status_code == 410
        assert client.get(f"/api/research/{job_id}/events").status_code == 410


class TestBudgets:
    """Edge case #11 + §11.R2 double gate."""

    def test_post_gate_503_when_monthly_cap_zero(self, client, monkeypatch):
        monkeypatch.setattr(api_db.settings, "MONTHLY_BUDGET_USD", 0.0)
        r = client.post("/api/research", json={"query": "Anyone"}, headers=ADMIN)
        assert r.status_code == 503
        assert "budget" in r.text

    def test_dequeue_gate_fails_job_terminally(self, client, monkeypatch):
        # POST gate passes (0 inflight at check time); dequeue gate then sees
        # the job's own reservation and rejects → terminal failed (§11.R2).
        monkeypatch.setattr(api_db.settings, "MONTHLY_BUDGET_USD", 1.0)
        monkeypatch.setattr(api_db.settings, "PER_JOB_BUDGET_USD", 1.0)
        r = client.post("/api/research", json={"query": "Anyone"}, headers=ADMIN)
        assert r.status_code == 202
        status = _wait_terminal(client, r.json()["job_id"])
        assert status["status"] == FAILED
        assert status["error"] == "budget exhausted"

    def test_per_job_abort(self, client):
        FakeOrchestrator.total_cost = 2.0   # > $1 default cap
        r = client.post("/api/research", json={"query": "Expensive Person"}, headers=ADMIN)
        status = _wait_terminal(client, r.json()["job_id"])
        assert status["status"] == FAILED
        assert "per-job budget" in status["error"]

    def test_research_crash_lands_in_row(self, client):
        FakeOrchestrator.fail_with = RuntimeError("provider melted")
        r = client.post("/api/research", json={"query": "Crash Person"}, headers=ADMIN)
        status = _wait_terminal(client, r.json()["job_id"])
        assert status["status"] == FAILED
        assert "research failed" in status["error"]


class TestAuthAndRateLimit:
    def test_turnstile_failure_403(self, client, monkeypatch):
        monkeypatch.setattr(routes_mod, "verify_turnstile", _turnstile_fail)
        r = client.post("/api/research", json={"query": "Someone", "turnstile_token": "bad"})
        assert r.status_code == 403

    def test_rate_limit_429_on_4th_and_admin_exempt(self, client):
        for i in range(3):
            r = client.post("/api/research", json={"query": f"Person {i}", "turnstile_token": "t"})
            assert r.status_code == 202, r.text
        r = client.post("/api/research", json={"query": "Person 3", "turnstile_token": "t"})
        assert r.status_code == 429
        # Admin bypass still works after the bucket is exhausted
        r = client.post("/api/research", json={"query": "Admin Person"}, headers=ADMIN)
        assert r.status_code == 202


class TestSSE:
    def test_terminal_event_immediate_for_completed_job(self, client):
        r = client.post("/api/research", json={"query": "SSE Person"}, headers=ADMIN)
        job_id = r.json()["job_id"]
        _wait_terminal(client, job_id)

        with client.stream("GET", f"/api/research/{job_id}/events") as resp:
            assert resp.status_code == 200
            got = ""
            for chunk in resp.iter_text():
                got += chunk
                if "event: completed" in got:
                    break
        assert "event: completed" in got
        assert f"/api/research/{job_id}/report" in got


class TestReaper:
    def test_stale_running_reaped_and_expiry_purges_pii(self, client):
        stale_id, expire_id = uuid.uuid4(), uuid.uuid4()

        async def _seed():
            async with api_db.get_sessionmaker()() as s:
                s.add(Job(id=stale_id, query="Stale Person", status=RUNNING,
                          created_at=utcnow(), expires_at=utcnow() + timedelta(days=7),
                          heartbeat_at=utcnow() - timedelta(minutes=10), progress={}))
                s.add(Job(id=expire_id, query="Old Person", status=COMPLETED,
                          created_at=utcnow() - timedelta(days=8),
                          expires_at=utcnow() - timedelta(days=1),
                          report_html="<html>old</html>", report_json={"a": 1},
                          client_ip_hash="deadbeef", progress={}))
                await s.commit()

        async def _reap():
            return await reap_once()

        client.portal.call(_seed)
        counts = client.portal.call(_reap)
        assert counts["stale_failed"] >= 1
        assert counts["expired"] >= 1

        async def _fetch():
            async with api_db.get_sessionmaker()() as s:
                stale = (await s.execute(select(Job).where(Job.id == stale_id))).scalar_one()
                expired = (await s.execute(select(Job).where(Job.id == expire_id))).scalar_one()
                return stale, expired

        stale, expired = client.portal.call(_fetch)
        assert stale.status == FAILED and "stale heartbeat" in stale.error
        # §12.S2 PII purge: html, json, ip hash AND query all NULL
        assert expired.status == EXPIRED
        assert expired.report_html is None
        assert expired.report_json is None
        assert expired.client_ip_hash is None
        assert expired.query is None
