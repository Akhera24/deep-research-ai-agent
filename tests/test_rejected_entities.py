"""
Phase C1.7d — rejected-entity negative scoping (D13/R9).

The client holds what the user explicitly rejected (picker "None of these",
banner cancel) and sends it as `rejected_entities` on BOTH endpoints. Server
side it is PROMPT-TRANSIENT (R9): one delimited "NOT these" data line in the
clustering / initial-query / AI-refinement / extraction prompts — never in
the context echo, progress/banner, or resolved_entity. Validation is the
EntitySelection display-tier pattern: ≤5 items (6 → 422), each control-
stripped and capped at 200 chars; they never enter search-engine queries.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from src.api.models import DisambiguateRequest, ResearchRequest
from tests.test_api_endpoints import (  # noqa: F401 — shared app fixture
    ADMIN, FakeOrchestrator, _wait_terminal, client,
)
from tests.test_context_threading import _submit, _get_job

REJECTED = ["John Smith — Explorer, Jamestown colony (17th century)",
            "John Smith — mortgage advisor, HSBC (Leeds)"]


# ---------------------------------------------------------------------------
# Validation (display-tier: control-strip + caps, NOT the query charset)
# ---------------------------------------------------------------------------

class TestValidation:
    @pytest.mark.parametrize("model", [ResearchRequest, DisambiguateRequest])
    def test_six_items_rejected(self, model):
        with pytest.raises(ValidationError):
            model(query="John Smith", rejected_entities=[f"p{i}" for i in range(6)])

    @pytest.mark.parametrize("model", [ResearchRequest, DisambiguateRequest])
    def test_five_items_accepted(self, model):
        m = model(query="John Smith", rejected_entities=[f"p{i}" for i in range(5)])
        assert len(m.rejected_entities) == 5

    def test_absent_defaults_empty(self):
        assert ResearchRequest(query="x").rejected_entities == []

    def test_control_chars_stripped_text_kept_raw(self):
        m = ResearchRequest(query="x", rejected_entities=["bad\x00\x1fperson\x7f"])
        assert m.rejected_entities == ["badperson"]

    def test_em_dash_descriptor_accepted(self):
        # display tier — the D5 descriptor format carries `—`, which the
        # query charset would reject
        m = ResearchRequest(query="x", rejected_entities=[REJECTED[0]])
        assert m.rejected_entities == [REJECTED[0]]

    def test_overlong_item_truncated_to_200(self):
        m = ResearchRequest(query="x", rejected_entities=["a" * 500])
        assert len(m.rejected_entities[0]) == 200

    def test_empty_items_dropped(self):
        m = ResearchRequest(query="x", rejected_entities=["  ", "", "real one"])
        assert m.rejected_entities == ["real one"]


# ---------------------------------------------------------------------------
# Prompt lines — one delimited "NOT these" data line in all four prompts
# ---------------------------------------------------------------------------

class FakeRouter:
    def __init__(self, content="[]"):
        self.content = content
        self.prompts = []

    def route_and_call(self, task_type=None, prompt="", **kwargs):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.content, cost=0.0,
                               provider=SimpleNamespace(value="google"),
                               model_name="fake")


FIVE_FINDINGS = [
    {"content": f"Fact number {i} about the subject's career",
     "category": "professional", "entities": ["NSSF"]}
    for i in range(5)
]


class TestPromptLines:
    def test_extraction_prompt_carries_not_line(self):
        from src.extraction.extractor import FactExtractor
        fe = FactExtractor(FakeRouter())
        p = fe._build_extraction_prompt(
            "[Source 1] text", "John Smith",
            context={"company": "NSSF"}, rejected_entities=REJECTED)
        assert "is NOT" in p
        for desc in REJECTED:
            assert desc in p

    def test_extraction_prompt_no_line_when_absent(self):
        from src.extraction.extractor import FactExtractor
        fe = FactExtractor(FakeRouter())
        p = fe._build_extraction_prompt("[Source 1] text", "John Smith",
                                        context={"company": "NSSF"})
        assert "is NOT" not in p

    def test_initial_query_prompt_carries_not_line(self):
        from src.search.strategy import SearchStrategyEngine
        router = FakeRouter()
        engine = SearchStrategyEngine(router)
        engine.generate_initial_queries("John Smith",
                                        context={"company": "NSSF"},
                                        rejected_entities=REJECTED)
        assert router.prompts
        assert "is NOT" in router.prompts[0]
        assert REJECTED[0] in router.prompts[0]

    def test_refinement_prompt_carries_not_line(self):
        from src.search.strategy import SearchStrategyEngine
        router = FakeRouter()
        engine = SearchStrategyEngine(router)
        engine.refine_based_on_findings(
            "John Smith", FIVE_FINDINGS, max_follow_ups=5,
            context={"company": "NSSF"}, rejected_entities=REJECTED)
        ai_prompts = [p for p in router.prompts
                      if "due diligence investigator" in p]
        assert ai_prompts, "AI refinement was not invoked"
        assert "is NOT" in ai_prompts[0]
        assert REJECTED[1] in ai_prompts[0]

    def test_clustering_prompt_carries_not_line(self):
        from src.core.preflight import _build_clustering_prompt
        from tests.test_preflight import RESULTS
        p = _build_clustering_prompt("John Smith", RESULTS[:3],
                                     {"company": "NSSF"},
                                     rejected_entities=REJECTED)
        assert "is NOT" in p
        assert REJECTED[0] in p
        bare = _build_clustering_prompt("John Smith", RESULTS[:3],
                                        {"company": "NSSF"})
        assert "is NOT" not in bare

    def test_not_line_never_enters_search_queries(self):
        """They shape prompt GENERATION only — a rejected descriptor must
        never appear verbatim in a generated search-query string."""
        from src.search.strategy import SearchStrategyEngine
        engine = SearchStrategyEngine(FakeRouter())
        queries = engine.refine_based_on_findings(
            "John Smith",
            [{"content": "John Smith is a VP", "category": "professional",
              "entities": []}],
            max_follow_ups=15, context={"company": "NSSF"},
            rejected_entities=REJECTED)
        for q in queries:
            for desc in REJECTED:
                assert desc not in q.text


# ---------------------------------------------------------------------------
# Workflow threading: research() kwarg → state → all three callsites
# ---------------------------------------------------------------------------

def _orchestrator():
    from src.core.workflow import ResearchOrchestrator
    return ResearchOrchestrator(max_iterations=2, enable_checkpoints=False)


class TestWorkflowThreading:
    @pytest.mark.asyncio
    async def test_plan_node_passes_rejected(self):
        orch = _orchestrator()
        orch.strategy_engine = MagicMock()
        orch.strategy_engine.generate_initial_queries.return_value = []
        state = {"target_name": "John Smith", "context": {"company": "NSSF"},
                 "rejected_entities": REJECTED}
        await orch._node_plan_strategy(state)
        kwargs = orch.strategy_engine.generate_initial_queries.call_args.kwargs
        assert kwargs["rejected_entities"] == REJECTED

    @pytest.mark.asyncio
    async def test_refine_node_passes_rejected(self):
        orch = _orchestrator()
        orch.strategy_engine = MagicMock()
        orch.strategy_engine.refine_based_on_findings.return_value = []
        fact = SimpleNamespace(content="c", category="professional",
                               entities_mentioned=[])
        state = {"target_name": "John Smith", "iteration": 0, "queries": [],
                 "facts": [fact], "context": {"company": "NSSF"},
                 "rejected_entities": REJECTED}
        await orch._node_refine_queries(state)
        kwargs = orch.strategy_engine.refine_based_on_findings.call_args.kwargs
        assert kwargs["rejected_entities"] == REJECTED

    @pytest.mark.asyncio
    async def test_extract_node_passes_rejected(self):
        orch = _orchestrator()
        orch.fact_extractor = MagicMock()
        orch.fact_extractor.extract = AsyncMock(return_value=[])
        state = {"target_name": "John Smith", "iteration": 1,
                 "context": {"company": "NSSF"},
                 "rejected_entities": REJECTED,
                 "search_results": [SimpleNamespace(url="https://x.com")],
                 "search_results_processed_index": 0,
                 "facts": [], "coverage": {}}
        await orch._node_extract_facts(state)
        kwargs = orch.fact_extractor.extract.call_args.kwargs
        assert kwargs["rejected_entities"] == REJECTED

    @pytest.mark.asyncio
    async def test_empty_rejected_passes_none(self):
        orch = _orchestrator()
        orch.strategy_engine = MagicMock()
        orch.strategy_engine.generate_initial_queries.return_value = []
        state = {"target_name": "John Smith", "context": {},
                 "rejected_entities": []}
        await orch._node_plan_strategy(state)
        kwargs = orch.strategy_engine.generate_initial_queries.call_args.kwargs
        assert kwargs["rejected_entities"] is None


# ---------------------------------------------------------------------------
# Endpoint threading + R9 lifecycle (prompt-transient — never persisted)
# ---------------------------------------------------------------------------

class TestEndpointLifecycle:
    def test_research_threads_rejected_to_orchestrator(self, client):
        _submit(client, {"query": "John Smith",
                         "context": {"company": "NSSF"},
                         "rejected_entities": REJECTED})
        assert FakeOrchestrator.last_rejected == REJECTED

    def test_r9_rejected_never_in_progress_or_report(self, client):
        """Prompt-transient: hostile descriptor text can never reach a
        render sink — not progress/banner, not resolved_entity, not
        report_json."""
        marker = "UNIQUE-REJECTED-MARKER-XYZ"
        job_id = _submit(client, {"query": "John Smith",
                                  "rejected_entities": [f"Evil — {marker}"]})
        r = client.get(f"/api/research/{job_id}", headers=ADMIN)
        assert marker not in r.text
        job = _get_job(client, job_id)
        import json as _json
        assert marker not in _json.dumps(job.report_json)
        assert marker not in _json.dumps(job.progress)

    def test_six_rejected_entities_422_endpoint(self, client):
        r = client.post("/api/research",
                        json={"query": "John Smith",
                              "rejected_entities": [f"p{i}" for i in range(6)]},
                        headers=ADMIN)
        assert r.status_code == 422

    def test_disambiguate_threads_rejected_to_preflight(self, client,
                                                        monkeypatch):
        from src.api import routes as routes_mod
        from src.core import preflight as preflight_mod
        from src.core.preflight import PreflightResult
        calls = []

        async def fake(query, hints=None, rejected_entities=None, **kwargs):
            calls.append({"query": query, "hints": hints,
                          "rejected_entities": rejected_entities})
            return PreflightResult(decision="unscoped", note="x")

        monkeypatch.setattr(preflight_mod, "discover_candidates", fake)
        r = client.post("/api/disambiguate",
                        json={"query": "John Smith",
                              "rejected_entities": REJECTED},
                        headers=ADMIN)
        assert r.status_code == 200
        assert calls[0]["rejected_entities"] == REJECTED
