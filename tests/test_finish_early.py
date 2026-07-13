"""
C1.5 — finish-early ("Generate report now"), PLAN.md Rev 3.9 D8/D9 + review
R1/R3/R5/R7/R8.

Endpoint tests run the real app (fixture from test_api_endpoints); the
orchestrator fake ignores the flag — jobs-layer semantics only. The
REAL-workflow behavior (branch short-circuit, tail runs, honest stamping)
is covered by graph-level tests on ResearchOrchestrator with mocked engines
(review R5a: ScriptedOrchestrator has no graph, so browser tests cannot
assert this).
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import asyncio
import threading
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.api import db as api_db
from src.api import jobs as jobs_mod
from src.api.db import COMPLETED, FAILED, RUNNING, Job, utcnow
from tests.test_api_endpoints import (  # noqa: F401 — shared app fixture
    ADMIN, FakeOrchestrator, _wait_terminal, client,
)


class FinishableOrchestrator(FakeOrchestrator):
    """First boundary reports facts>0, then holds at a gate."""

    gate: threading.Event

    async def research(self, query, context=None, progress_callback=None,
                       activity_callback=None, rejected_entities=None):
        await progress_callback({"node": "fact_extraction", "iteration": 1,
                                 "max_iterations": 10, "facts": 3,
                                 "coverage": {}})
        while not type(self).gate.is_set():
            await asyncio.sleep(0.02)
        await progress_callback({"node": "query_refinement", "iteration": 1,
                                 "max_iterations": 10, "facts": 3,
                                 "coverage": {}})
        return {
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


# ---------------------------------------------------------------------------
# Endpoint semantics
# ---------------------------------------------------------------------------

class TestFinishEndpoint:
    def test_unknown_job_404(self, client):
        assert client.post(f"/api/research/{uuid.uuid4()}/finish").status_code == 404

    def test_terminal_job_noop_200(self, client):
        r = client.post("/api/research", json={"query": "Jane Doe"}, headers=ADMIN)
        job_id = r.json()["job_id"]
        _wait_terminal(client, job_id)
        rf = client.post(f"/api/research/{job_id}/finish")
        assert rf.status_code == 200
        assert rf.json()["status"] == COMPLETED
        assert uuid.UUID(job_id) not in jobs_mod._finish_early

    @pytest.mark.parametrize("progress,expected", [
        (None, 409),                     # R1 precedence case: progress NULL
        ({"phase": "queued"}, 409),      # queued row — no facts key
        ({"facts": 0}, 409),             # explicit zero
        ({"facts": 5}, 200),             # the R1 bug returned 409 here
    ])
    def test_zero_facts_guard_precedence_pinned(self, client, progress,
                                                expected):
        # Review R1: `x or 0 == 0` is truthy for EVERY input — the
        # parenthesized guard must 409 only when facts are truly absent.
        job_id = uuid.uuid4()

        async def _insert():
            async with api_db.get_sessionmaker()() as s:
                s.add(Job(id=job_id, query="X", status=RUNNING,
                          created_at=utcnow(),
                          expires_at=utcnow() + timedelta(days=1),
                          progress=progress))
                await s.commit()

        client.portal.call(_insert)
        r = client.post(f"/api/research/{job_id}/finish")
        assert r.status_code == expected
        if expected == 200:
            assert r.json()["status"] == "finishing"
            assert job_id in jobs_mod._finish_early
        else:
            assert job_id not in jobs_mod._finish_early

    def test_finish_running_job_registers_and_cleans_up(self, client,
                                                        monkeypatch):
        monkeypatch.setattr(jobs_mod, "ResearchOrchestrator",
                            FinishableOrchestrator)
        FinishableOrchestrator.gate = threading.Event()
        r = client.post("/api/research", json={"query": "Jane Doe"}, headers=ADMIN)
        job_id = r.json()["job_id"]
        _wait_status(client, job_id, "running")
        import time
        deadline = time.time() + 5
        while time.time() < deadline:      # wait for the facts>0 boundary
            p = client.get(f"/api/research/{job_id}", headers=ADMIN).json()["progress"]
            if (p.get("facts") or 0) > 0:
                break
            time.sleep(0.02)

        rf = client.post(f"/api/research/{job_id}/finish")
        assert rf.status_code == 200 and rf.json()["status"] == "finishing"
        assert uuid.UUID(job_id) in jobs_mod._finish_early
        # idempotent
        assert client.post(f"/api/research/{job_id}/finish").status_code == 200

        FinishableOrchestrator.gate.set()
        final = _wait_terminal(client, job_id)
        assert final["status"] == COMPLETED    # fake ignores the flag
        assert uuid.UUID(job_id) not in jobs_mod._finish_early  # finally-clean

    def test_cancel_beats_finish(self, client, monkeypatch):
        monkeypatch.setattr(jobs_mod, "ResearchOrchestrator",
                            FinishableOrchestrator)
        FinishableOrchestrator.gate = threading.Event()
        r = client.post("/api/research", json={"query": "Jane Doe"}, headers=ADMIN)
        job_id = r.json()["job_id"]
        _wait_status(client, job_id, "running")
        client.post(f"/api/research/{job_id}/finish")
        client.post(f"/api/research/{job_id}/cancel")
        FinishableOrchestrator.gate.set()
        final = _wait_terminal(client, job_id)
        assert final["status"] == FAILED
        assert final["error"] == "cancelled by user"
        assert uuid.UUID(job_id) not in jobs_mod._finish_early
        assert uuid.UUID(job_id) not in jobs_mod._cancel_requested


# ---------------------------------------------------------------------------
# Real-workflow behavior (R5a)
# ---------------------------------------------------------------------------

def _orchestrator(max_iterations=3):
    from src.core.workflow import ResearchOrchestrator
    return ResearchOrchestrator(max_iterations=max_iterations,
                                enable_checkpoints=False)


def _branch_state(queries=True):
    return {"iteration": 1, "max_iterations": 5, "facts": [],
            "queries": [object()] if queries else [],
            "facts_per_iteration": [5], "coverage": {}}


class TestBranchShortCircuit:
    def test_finish_early_routes_to_verify_and_records_applied(self):
        orch = _orchestrator()
        orch.strategy_engine = MagicMock()
        orch.strategy_engine.is_coverage_adequate.return_value = False
        orch.finish_early = True
        assert orch._decide_continue_or_finish(_branch_state()) == "verify"
        assert orch._finished_early_applied is True

    def test_natural_path_unchanged_and_not_applied(self):
        orch = _orchestrator()
        orch.strategy_engine = MagicMock()
        orch.strategy_engine.is_coverage_adequate.return_value = False
        assert orch._decide_continue_or_finish(_branch_state()) == "continue"
        assert orch._finished_early_applied is False

    def test_format_results_stamps_only_when_applied(self):
        orch = _orchestrator(max_iterations=7)
        orch.strategy_engine = MagicMock()
        orch.strategy_engine.executed_queries = []
        orch.strategy_engine.coverage.to_dict.return_value = {}
        now = datetime.now()
        state = {"target_name": "T", "facts": [], "risk_flags": [],
                 "connections": [], "summary": {}, "iteration": 2,
                 "start_time": now, "completed_at": now}
        meta = orch._format_results(state)["metadata"]
        assert "finished_early" not in meta      # R3: natural exit stamps NOTHING
        assert meta["max_iterations"] == 7       # R8: available for "N of M"

        orch._finished_early_applied = True
        meta = orch._format_results(state)["metadata"]
        assert meta["finished_early"] is True


def _mocked_engines(orch):
    """Fake every external component; the REAL compiled graph still runs."""
    from src.extraction.extractor import Fact
    from src.search.executor import SearchResult
    from src.search.strategy import SearchCategory, SearchDepth, SearchQuery

    queries = [SearchQuery(text=f"query {i}", purpose="p",
                           category=SearchCategory.PROFESSIONAL,
                           depth=SearchDepth.SURFACE)
               for i in range(30)]
    engine = MagicMock()
    engine.generate_initial_queries.return_value = queries
    engine.refine_based_on_findings.return_value = []
    engine.is_coverage_adequate.return_value = False
    engine.coverage.to_dict.return_value = {}
    engine.coverage.get_average.return_value = 0.1
    engine.executed_queries = []
    orch.strategy_engine = engine

    async def fake_search(text, max_results=10, activity_callback=None,
                          engine=None):
        return [SearchResult(query=text, url=f"https://x.example/{text}",
                             title=text, snippet="s", rank=1,
                             search_engine="fake", fetched_at=datetime.now())]

    orch.search_executor = SimpleNamespace(search=fake_search)

    counter = {"n": 0}

    async def fake_extract(search_results, target_name, max_facts=50,
                           activity_callback=None, target_context=None,
                           rejected_entities=None):
        counter["n"] += 1
        return [Fact(content=f"fact {counter['n']}-{i}",
                     category="professional", confidence=0.9)
                for i in range(4)]     # ≥3/iter → stagnation never fires

    orch.fact_extractor = SimpleNamespace(extract=fake_extract)
    orch.router = SimpleNamespace(
        route_and_call=lambda **kw: SimpleNamespace(content="[]", cost=0.0))


class TestFullGraphFinishEarly:
    @pytest.mark.asyncio
    async def test_finish_early_runs_tail_and_stamps(self):
        orch = _orchestrator(max_iterations=3)
        _mocked_engines(orch)
        stages = []

        async def cb(p):
            stages.append(p["node"])
            if p["node"] == "fact_extraction":
                orch.finish_early = True      # what jobs.py does on /finish

        result = await orch.research("Test Person", progress_callback=cb)
        # loop stopped after ONE iteration; the FULL tail still ran
        assert result["metadata"]["iterations"] == 1
        for stage in ("verification", "risk_assessment",
                      "connection_mapping", "report_generation"):
            assert stage in stages, f"tail node {stage} did not run"
        assert result["metadata"]["finished_early"] is True
        assert result["metadata"]["max_iterations"] == 3

    @pytest.mark.asyncio
    async def test_natural_run_stamps_nothing(self):
        orch = _orchestrator(max_iterations=2)
        _mocked_engines(orch)
        result = await orch.research("Test Person")
        assert result["metadata"]["iterations"] == 2   # ran to max-iter
        assert "finished_early" not in result["metadata"]


# ---------------------------------------------------------------------------
# Report header honesty line
# ---------------------------------------------------------------------------

def _minimal_result(metadata):
    return {"facts": [{"content": "A fact", "category": "professional",
                       "confidence": 0.9, "source": "https://ok.example"}],
            "risk_flags": [], "connections": [], "metadata": metadata}


class TestReportHeaderLine:
    def test_early_note_rendered_with_n_of_m(self):
        from src.reporting.html_report import render_html_report
        html = render_html_report(_minimal_result(
            {"coverage": {"average": 0.5}, "iterations": 2,
             "max_iterations": 7, "finished_early": True,
             "duration_seconds": 10.0}), "Test Subject", 12.3)
        assert "Generated early" in html
        assert "after 2 of 7 iterations" in html

    def test_no_note_on_natural_run(self):
        from src.reporting.html_report import render_html_report
        html = render_html_report(_minimal_result(
            {"coverage": {"average": 0.5}, "iterations": 2,
             "duration_seconds": 10.0}), "Test Subject", 12.3)
        assert "Generated early" not in html
