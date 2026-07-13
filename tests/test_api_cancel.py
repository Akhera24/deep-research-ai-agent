"""
C1.4 — cancel endpoint + in-memory registry (PLAN.md Rev 3.8, review R1/R7).

Cancel mirrors BudgetExceeded exactly: raised in the node-boundary progress
callback, cost recorded, terminal FAILED with the canonical error string
("cancelled by user" — the frontend maps it to a neutral Cancelled state),
ledger row written. Registry is in-process (NO DB column — create_all never
ALTERs the prod jobs table), bound to the --workers 1 pin.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import asyncio
import json
import threading
import uuid

import pytest
from sqlalchemy import select

from src.api import db as api_db
from src.api import jobs as jobs_mod
from src.api.db import COMPLETED, FAILED, SpendLedger
from tests.test_api_endpoints import (  # noqa: F401 — shared app fixture
    ADMIN, FakeOrchestrator, _wait_terminal, client,
)


class GatedOrchestrator(FakeOrchestrator):
    """Runs one progress write, then holds at a gate the test controls,
    then writes progress again (where a pending cancel fires)."""

    gate: threading.Event

    async def research(self, query, context=None, progress_callback=None,
                       activity_callback=None, rejected_entities=None):
        await progress_callback({"node": "data_collection", "iteration": 1,
                                 "max_iterations": 10, "facts": 0,
                                 "coverage": {}})
        while not type(self).gate.is_set():
            await asyncio.sleep(0.02)
        await progress_callback({"node": "fact_extraction", "iteration": 1,
                                 "max_iterations": 10, "facts": 2,
                                 "coverage": {}})
        return type(self).result if hasattr(type(self), "result") else {
            "facts": [], "risk_flags": [], "connections": [],
            "metadata": {"coverage": {}, "iterations": 1,
                         "duration_seconds": 0.1},
        }


def _wait_status(client, job_id, wanted, timeout=5.0):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/research/{job_id}", headers=ADMIN)
        if r.status_code == 200 and r.json()["status"] == wanted:
            return r.json()
        time.sleep(0.02)
    raise AssertionError(f"job never reached {wanted}")


def _ledger_rows(client):
    async def rows():
        async with api_db.get_sessionmaker()() as session:
            return (await session.execute(select(SpendLedger))).scalars().all()
    return client.portal.call(rows)


class TestCancel:
    def test_cancel_unknown_job_404(self, client):
        r = client.post(f"/api/research/{uuid.uuid4()}/cancel")
        assert r.status_code == 404

    def test_cancel_running_job_terminal_cost_ledger(self, client, monkeypatch):
        monkeypatch.setattr(jobs_mod, "ResearchOrchestrator", GatedOrchestrator)
        GatedOrchestrator.gate = threading.Event()
        GatedOrchestrator.total_cost = 0.07
        r = client.post("/api/research", json={"query": "Jane Doe"}, headers=ADMIN)
        job_id = r.json()["job_id"]
        _wait_status(client, job_id, "running")

        rc = client.post(f"/api/research/{job_id}/cancel")
        assert rc.status_code == 200
        GatedOrchestrator.gate.set()          # next node boundary → raises

        final = _wait_terminal(client, job_id)
        assert final["status"] == FAILED
        assert final["error"] == "cancelled by user"    # R7 exact string
        assert final["cost_usd"] == pytest.approx(0.07)  # cost recorded
        rows = _ledger_rows(client)
        assert len(rows) == 1 and float(rows[0].amount_usd) == pytest.approx(0.07)
        # registry cleaned up
        assert uuid.UUID(job_id) not in jobs_mod._cancel_requested

    def test_cancel_is_idempotent(self, client, monkeypatch):
        monkeypatch.setattr(jobs_mod, "ResearchOrchestrator", GatedOrchestrator)
        GatedOrchestrator.gate = threading.Event()
        GatedOrchestrator.total_cost = 0.01
        r = client.post("/api/research", json={"query": "Jane Doe"}, headers=ADMIN)
        job_id = r.json()["job_id"]
        _wait_status(client, job_id, "running")
        assert client.post(f"/api/research/{job_id}/cancel").status_code == 200
        assert client.post(f"/api/research/{job_id}/cancel").status_code == 200
        GatedOrchestrator.gate.set()
        final = _wait_terminal(client, job_id)
        assert final["status"] == FAILED
        assert final["error"] == "cancelled by user"

    def test_cancel_after_completion_noop_200(self, client):
        r = client.post("/api/research", json={"query": "Jane Doe"}, headers=ADMIN)
        job_id = r.json()["job_id"]
        final = _wait_terminal(client, job_id)
        assert final["status"] == COMPLETED
        rc = client.post(f"/api/research/{job_id}/cancel")
        assert rc.status_code == 200
        # still completed — cancel-vs-completion race is a no-op
        assert client.get(f"/api/research/{job_id}",
                          headers=ADMIN).json()["status"] == COMPLETED
        assert uuid.UUID(job_id) not in jobs_mod._cancel_requested

    def test_cancel_queued_job_terminates_with_zero_spend(self, client):
        async def _hog():
            sem = jobs_mod._get_semaphore()
            for _ in range(api_db.settings.MAX_CONCURRENT_JOBS):
                await sem.acquire()

        async def _release():
            for _ in range(api_db.settings.MAX_CONCURRENT_JOBS):
                jobs_mod._get_semaphore().release()

        client.portal.call(_hog)
        r = client.post("/api/research", json={"query": "Queued Person"},
                        headers=ADMIN)
        job_id = r.json()["job_id"]
        assert client.get(f"/api/research/{job_id}",
                          headers=ADMIN).json()["status"] == "queued"
        assert client.post(f"/api/research/{job_id}/cancel").status_code == 200
        client.portal.call(_release)
        final = _wait_terminal(client, job_id)
        assert final["status"] == FAILED
        assert final["error"] == "cancelled by user"
        assert not final.get("cost_usd")        # $0 — cancelled pre-orchestrator
        assert _ledger_rows(client) == []

    def test_sse_terminal_event_carries_cancel_string(self, client, monkeypatch):
        monkeypatch.setattr(jobs_mod, "ResearchOrchestrator", GatedOrchestrator)
        GatedOrchestrator.gate = threading.Event()
        GatedOrchestrator.total_cost = 0.01
        r = client.post("/api/research", json={"query": "Jane Doe"}, headers=ADMIN)
        job_id = r.json()["job_id"]
        _wait_status(client, job_id, "running")
        client.post(f"/api/research/{job_id}/cancel")
        GatedOrchestrator.gate.set()
        _wait_terminal(client, job_id)

        # already-terminal jobs stream the terminal event immediately
        with client.stream("GET", f"/api/research/{job_id}/events") as resp:
            event, data = None, None
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                if line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    break
        assert event == "failed"
        # R7: the test distinguishes a cancel from a genuine failure
        assert data["error"] == "cancelled by user"
