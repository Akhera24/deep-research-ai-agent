"""
C1.2 — context/entity threading (PLAN.md Rev 3.8, D3/D5, review R5).

Covers: the D5 entity_id formula (server-recomputed, deterministic, client id
ignored); two-tier entity validation (length + control-char strip — NOT the
query charset, descriptors carry `—`); context threading API → create_job →
orchestrator.research; resolved_entity in report_json.metadata + progress;
the refine_based_on_findings context prompt line (the "NEW gap": follow-up
drift); and the extraction-prompt context line.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api import db as api_db
from src.core.preflight import compute_entity_id
from src.api.models import EntitySelection, ResearchHints, ResearchRequest
from tests.test_api_endpoints import (  # noqa: F401 — shared app fixture
    ADMIN, FakeOrchestrator, _wait_terminal, client,
)


# ---------------------------------------------------------------------------
# D5 entity_id
# ---------------------------------------------------------------------------

class TestEntityId:
    def test_deterministic_and_formatted(self):
        a = compute_entity_id("Jane Doe", "VP Engineering, Stripe")
        b = compute_entity_id("Jane Doe", "VP Engineering, Stripe")
        assert a == b
        assert a.startswith("ent_") and len(a) == 4 + 16
        int(a[4:], 16)   # hex

    def test_nfkc_case_insensitive(self):
        # case + full-width/compatibility forms normalize to the same id
        assert (compute_entity_id("JANE DOE", "STRIPE")
                == compute_entity_id("jane doe", "stripe"))
        assert (compute_entity_id("Ｊａｎｅ Ｄｏｅ", "Stripe")
                == compute_entity_id("jane doe", "stripe"))

    def test_disambiguator_changes_id(self):
        assert (compute_entity_id("Jane Doe", "Stripe")
                != compute_entity_id("Jane Doe", "Boeing"))


# ---------------------------------------------------------------------------
# R5 two-tier validation
# ---------------------------------------------------------------------------

class TestEntityValidation:
    def test_descriptor_with_em_dash_accepted(self):
        # the D5 descriptor format contains `—` — the query charset would
        # reject it; entity display fields must NOT use that charset (R5)
        e = EntitySelection(canonical_name="Jane Doe",
                            descriptor="Jane Doe — VP Engineering, Stripe (SF)")
        assert "—" in e.descriptor

    def test_hint_charset_rejects_em_dash(self):
        with pytest.raises(ValueError):
            ResearchHints(company="Stripe — the payments company")

    def test_control_chars_stripped(self):
        e = EntitySelection(canonical_name="Jane\x00 Doe\x1b",
                            descriptor="d\x07esc")
        assert e.canonical_name == "Jane Doe"
        assert e.descriptor == "desc"

    def test_empty_name_after_strip_rejected(self):
        with pytest.raises(ValueError):
            EntitySelection(canonical_name="\x00\x01")

    def test_length_caps(self):
        with pytest.raises(ValueError):
            EntitySelection(canonical_name="x" * 121)
        with pytest.raises(ValueError):
            EntitySelection(canonical_name="ok", descriptor="x" * 201)
        with pytest.raises(ValueError):
            EntitySelection(canonical_name="ok",
                            disambiguators=[f"d{i}" for i in range(9)])

    def test_disambiguator_items_cleaned(self):
        e = EntitySelection(canonical_name="ok",
                            disambiguators=["  Stripe  ", "", "x" * 200])
        assert e.disambiguators[0] == "Stripe"
        assert "" not in e.disambiguators
        assert len(e.disambiguators[1]) <= 80

    def test_decision_allowlisted(self):
        with pytest.raises(ValueError):
            EntitySelection(canonical_name="ok", decision="evil")

    def test_client_entity_id_ignored_by_model(self):
        e = EntitySelection.model_validate(
            {"canonical_name": "Jane Doe", "entity_id": "ent_evil"})
        assert not hasattr(e, "entity_id") or getattr(e, "entity_id", None) is None

    def test_research_request_accepts_context_and_entity(self):
        r = ResearchRequest.model_validate({
            "query": "Jane Doe",
            "context": {"company": "Stripe"},
            "entity": {"canonical_name": "Jane Doe",
                       "descriptor": "Jane Doe — VP, Stripe"},
        })
        assert r.context.company == "Stripe"
        assert r.entity.canonical_name == "Jane Doe"


# ---------------------------------------------------------------------------
# API → jobs → orchestrator threading
# ---------------------------------------------------------------------------

ENTITY_PAYLOAD = {
    "canonical_name": "Jane Doe",
    "descriptor": "Jane Doe — VP Engineering, Stripe (San Francisco)",
    "disambiguators": ["Stripe", "VP Engineering"],
    "decision": "picked",
    "entity_id": "ent_client_forged",     # must be IGNORED (R5)
}


def _submit(client, payload):
    r = client.post("/api/research", json=payload, headers=ADMIN)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    _wait_terminal(client, job_id)
    return job_id


def _get_job(client, job_id):
    import uuid as _uuid
    from src.api import jobs as jobs_mod

    async def fetch():
        return await jobs_mod.get_job(_uuid.UUID(job_id))

    return client.portal.call(fetch)


class TestThreading:
    def test_context_and_entity_reach_orchestrator_merged(self, client):
        _submit(client, {"query": "Jane Doe",
                         "context": {"company": "Stripe", "role": "VP"},
                         "entity": ENTITY_PAYLOAD})
        ctx = FakeOrchestrator.last_context
        assert ctx["company"] == "Stripe"
        assert ctx["role"] == "VP"
        # entity contribution present (entity wins / rides alongside)
        joined = " ".join(str(v) for v in ctx.values())
        assert "VP Engineering, Stripe" in joined

    def test_no_context_stays_none(self, client):
        _submit(client, {"query": "Jane Doe"})
        assert FakeOrchestrator.last_context is None

    def test_resolved_entity_in_report_json_with_server_id(self, client):
        job_id = _submit(client, {"query": "Jane Doe", "entity": ENTITY_PAYLOAD})
        job = _get_job(client, job_id)
        resolved = job.report_json["metadata"]["resolved_entity"]
        expected = compute_entity_id("Jane Doe", "Stripe")   # first disambiguator
        assert resolved["entity_id"] == expected
        assert resolved["entity_id"] != "ent_client_forged"
        assert resolved["decision"] == "picked"
        assert resolved["canonical_name"] == "Jane Doe"
        # stable across runs: same input twice → same id
        job2 = _get_job(client, _submit(
            client, {"query": "Jane Doe", "entity": ENTITY_PAYLOAD}))
        assert (job2.report_json["metadata"]["resolved_entity"]["entity_id"]
                == expected)

    def test_unscoped_run_records_unscoped_decision(self, client):
        job_id = _submit(client, {"query": "Jane Doe"})
        job = _get_job(client, job_id)
        resolved = job.report_json["metadata"]["resolved_entity"]
        assert resolved == {"decision": "unscoped"}

    def test_progress_carries_resolved_entity_for_banner(self, client):
        job_id = _submit(client, {"query": "Jane Doe", "entity": ENTITY_PAYLOAD})
        r = client.get(f"/api/research/{job_id}", headers=ADMIN)
        progress = r.json()["progress"]
        assert progress["resolved_entity"]["canonical_name"] == "Jane Doe"
        assert progress["resolved_entity"]["decision"] == "picked"

    def test_hints_only_run_threads_hints_as_context(self, client):
        _submit(client, {"query": "Jane Doe",
                         "context": {"company": "Stripe"}})
        assert FakeOrchestrator.last_context == {"company": "Stripe"}


# ---------------------------------------------------------------------------
# Workflow node callsites (the "NEW gap": follow-ups must stay scoped)
# ---------------------------------------------------------------------------

def _orchestrator():
    from src.core.workflow import ResearchOrchestrator
    return ResearchOrchestrator(max_iterations=2, enable_checkpoints=False)


class TestNodeCallsites:
    @pytest.mark.asyncio
    async def test_refine_node_passes_context(self):
        orch = _orchestrator()
        orch.strategy_engine = MagicMock()
        orch.strategy_engine.refine_based_on_findings.return_value = []
        fact = SimpleNamespace(content="c", category="professional",
                               entities_mentioned=[])
        state = {"target_name": "Jane Doe", "iteration": 0, "queries": [],
                 "facts": [fact], "context": {"company": "Stripe"}}
        await orch._node_refine_queries(state)
        kwargs = orch.strategy_engine.refine_based_on_findings.call_args.kwargs
        assert kwargs["context"] == {"company": "Stripe"}

    @pytest.mark.asyncio
    async def test_refine_node_empty_context_passes_none(self):
        orch = _orchestrator()
        orch.strategy_engine = MagicMock()
        orch.strategy_engine.refine_based_on_findings.return_value = []
        fact = SimpleNamespace(content="c", category="professional",
                               entities_mentioned=[])
        state = {"target_name": "Jane Doe", "iteration": 0, "queries": [],
                 "facts": [fact], "context": {}}
        await orch._node_refine_queries(state)
        kwargs = orch.strategy_engine.refine_based_on_findings.call_args.kwargs
        assert kwargs["context"] is None

    @pytest.mark.asyncio
    async def test_extract_node_passes_target_context(self):
        orch = _orchestrator()
        orch.fact_extractor = MagicMock()
        orch.fact_extractor.extract = AsyncMock(return_value=[])
        state = {"target_name": "Jane Doe", "iteration": 1,
                 "context": {"company": "Stripe"},
                 "search_results": [SimpleNamespace(url="https://x.com")],
                 "search_results_processed_index": 0,
                 "facts": [], "coverage": {}}
        await orch._node_extract_facts(state)
        kwargs = orch.fact_extractor.extract.call_args.kwargs
        assert kwargs["target_context"] == {"company": "Stripe"}


# ---------------------------------------------------------------------------
# Prompt lines (strategy + extraction)
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
     "category": "professional", "entities": ["Stripe"]}
    for i in range(5)
]


class TestPromptLines:
    def test_refinement_prompt_carries_context_line(self):
        from src.search.strategy import SearchStrategyEngine
        router = FakeRouter()
        engine = SearchStrategyEngine(router)
        engine.refine_based_on_findings(
            "Jane Doe", FIVE_FINDINGS, max_follow_ups=5,
            context={"company": "Stripe", "role": "VP"})
        ai_prompts = [p for p in router.prompts
                      if "due diligence investigator" in p]
        assert ai_prompts, "AI refinement was not invoked"
        assert "RESEARCH TARGET CONTEXT" in ai_prompts[0]
        assert "company: Stripe" in ai_prompts[0]

    def test_refinement_prompt_unchanged_without_context(self):
        from src.search.strategy import SearchStrategyEngine
        router = FakeRouter()
        engine = SearchStrategyEngine(router)
        engine.refine_based_on_findings("Jane Doe", FIVE_FINDINGS,
                                        max_follow_ups=5)
        ai_prompts = [p for p in router.prompts
                      if "due diligence investigator" in p]
        assert ai_prompts
        assert "RESEARCH TARGET CONTEXT" not in ai_prompts[0]

    def test_extraction_prompt_carries_context_line(self):
        from src.extraction.extractor import FactExtractor
        fe = FactExtractor(FakeRouter())
        p = fe._build_extraction_prompt(
            "[Source 1] text", "Jane Doe",
            context={"company": "Stripe"})
        assert "Research target context" in p
        assert "company: Stripe" in p

    def test_extraction_prompt_unchanged_without_context(self):
        from src.extraction.extractor import FactExtractor
        fe = FactExtractor(FakeRouter())
        p = fe._build_extraction_prompt("[Source 1] text", "Jane Doe")
        assert "Research target context" not in p

    @pytest.mark.asyncio
    async def test_extract_threads_target_context_to_prompt_builder(self):
        from src.extraction.extractor import FactExtractor
        from datetime import datetime
        from src.search.executor import SearchResult
        router = FakeRouter(content="[]")
        fe = FactExtractor(router, enable_verification=False)
        result = SearchResult(
            query="q", url="https://x.com", title="Jane Doe VP",
            snippet="Jane Doe is a VP at Stripe with a long career",
            rank=1, search_engine="fake", fetched_at=datetime.now(),
        )
        await fe.extract([result], "Jane Doe",
                         target_context={"company": "Stripe"})
        assert router.prompts, "extraction LLM call did not happen"
        assert "Research target context" in router.prompts[0]
        assert "company: Stripe" in router.prompts[0]
