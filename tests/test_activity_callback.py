"""
Contract tests for the Step A2 activity_callback (PLAN.md Rev 2.1).

Mirrors tests/test_workflow_progress.py: all externals are mocked — no
network, no spend. The two non-negotiables under test:

- No-callback equivalence: activity_callback=None leaves the CLI path
  behaviorally unchanged.
- Raising-callback isolation: the activity channel is fire-and-forget; a
  failing UI write must never cost search results or extracted facts
  (an unguarded raise would land in the callers' broad excepts and
  silently discard the batch).

Plus the jobs.py merge rules for the ONE shared progress dict
(REVIEW-LEARNINGS: a second writer to an overwrite-persisted column must
mutate and persist the same full object — bounds, truncation, clobber).
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.jobs import (
    ACTIVITY_MAX, SAMPLE_FACTS_MAX, SAMPLE_FACT_CHARS, _apply_activity,
)
from src.core.workflow import ResearchOrchestrator
from src.extraction.extractor import Fact, FactExtractor
from src.search.executor import SearchExecutor, SearchResult

HOSTILE_TITLE = '<script>alert("xss")</script> Jane Doe exposed'
HOSTILE_FACT = '<img src=x onerror=alert(1)> Jane Doe is CEO of Acme'


# ============================================================================
# Helpers
# ============================================================================

def _results(query="Jane Doe", engine="brave", n=3, title="Jane Doe profile"):
    return [
        SearchResult(
            query=query, url=f"https://example.com/{i}", title=title,
            snippet="Jane Doe snippet", rank=i + 1, search_engine=engine,
            fetched_at=datetime.now(),
        )
        for i in range(n)
    ]


def _executor(results, enable_cache=False):
    """Executor with a faked Brave engine — no keys, no network."""
    ex = SearchExecutor(
        brave_api_key="fake-key", serper_api_key="unused",
        enable_cache=enable_cache,
    )
    ex.serper_api_key = None      # never fall through to a real engine
    calls = []

    async def fake_brave(query, max_results):
        calls.append(query)
        return list(results)

    ex._search_brave = fake_brave

    async def fake_fallback(query, max_results):
        return []

    ex._search_fallback = fake_fallback
    ex._fake_brave_calls = calls
    return ex


def _extractor(contents):
    """Extractor whose AI step returns Facts with the given contents."""
    fx = FactExtractor(MagicMock(), enable_verification=False)

    async def fake_ai(text, target_name, search_results, fed_results):
        return [
            Fact(content=c, category="professional", confidence=0.9)
            for c in contents
        ]

    fx._extract_facts_with_ai = fake_ai
    return fx


def _collector():
    events = []

    async def cb(event):
        events.append(event)

    return events, cb


async def _raising_cb(event):
    raise RuntimeError("simulated DB write failure")


# ============================================================================
# SearchExecutor.search() contract
# ============================================================================

@pytest.mark.asyncio
async def test_search_no_callback_matches_old_behavior():
    ex = _executor(_results(n=3))
    results = await ex.search("Jane Doe")
    assert len(results) == 3
    # Same executor, callback supplied: identical results.
    events, cb = _collector()
    results_cb = await ex.search("Jane Doe", activity_callback=cb)
    assert [r.url for r in results_cb] == [r.url for r in results]


@pytest.mark.asyncio
async def test_search_emits_one_event_with_contract_fields():
    ex = _executor(_results(n=3))
    events, cb = _collector()
    await ex.search("Jane Doe", activity_callback=cb)
    assert events == [
        {"kind": "search", "engine": "brave", "query": "Jane Doe", "results": 3}
    ]


@pytest.mark.asyncio
async def test_search_zero_results_event():
    ex = _executor(_results(n=0))
    events, cb = _collector()
    results = await ex.search("Nobody At All", activity_callback=cb)
    assert results == []
    assert events == [
        {"kind": "search", "engine": "none", "query": "Nobody At All", "results": 0}
    ]


@pytest.mark.asyncio
async def test_search_cache_hit_emits_event():
    ex = _executor(_results(n=2), enable_cache=True)
    events, cb = _collector()
    await ex.search("Jane Doe", activity_callback=cb)
    await ex.search("Jane Doe", activity_callback=cb)
    assert len(ex._fake_brave_calls) == 1          # second hit served from cache
    assert len(events) == 2                        # ...but still visible in the feed
    assert all(e["kind"] == "search" and e["results"] == 2 for e in events)


@pytest.mark.asyncio
async def test_search_raising_callback_does_not_lose_results():
    ex = _executor(_results(n=3))
    results = await ex.search("Jane Doe", activity_callback=_raising_cb)
    assert len(results) == 3


# ============================================================================
# FactExtractor.extract() contract
# ============================================================================

@pytest.mark.asyncio
async def test_extract_no_callback_equivalence():
    contents = ["Jane Doe is CEO of Acme", "Jane Doe founded Acme in 2010"]
    facts = await _extractor(contents).extract(_results(), "Jane Doe")
    events, cb = _collector()
    facts_cb = await _extractor(contents).extract(
        _results(), "Jane Doe", activity_callback=cb
    )
    assert [f.content for f in facts_cb] == [f.content for f in facts]
    assert len(facts) == 2


@pytest.mark.asyncio
async def test_extract_emits_start_and_done_with_samples():
    contents = [
        "Jane Doe is CEO of Acme",
        "Jane Doe founded Acme in 2010",
        "Jane Doe lives in Denver",
        "Jane Doe holds an MBA",
    ]
    fx = _extractor(contents)
    events, cb = _collector()
    facts = await fx.extract(_results(), "Jane Doe", activity_callback=cb)
    assert events[0] == {"kind": "extract", "status": "start"}
    done = events[1]
    assert done["kind"] == "extract" and done["status"] == "done"
    assert done["facts"] == len(facts)
    assert done["samples"] == [f.content for f in facts[:3]]
    assert len(done["samples"]) == 3


@pytest.mark.asyncio
async def test_extract_empty_results_no_events():
    fx = _extractor(["Jane Doe is CEO of Acme"])
    events, cb = _collector()
    facts = await fx.extract([], "Jane Doe", activity_callback=cb)
    assert facts == []
    assert events == []


@pytest.mark.asyncio
async def test_extract_failure_emits_done_zero_no_dangling_start():
    fx = FactExtractor(MagicMock(), enable_verification=False)

    async def broken_ai(text, target_name, search_results):
        raise RuntimeError("provider down")

    fx._extract_facts_with_ai = broken_ai
    events, cb = _collector()
    facts = await fx.extract(_results(), "Jane Doe", activity_callback=cb)
    assert facts == []
    assert events[0] == {"kind": "extract", "status": "start"}
    assert events[1] == {"kind": "extract", "status": "done", "facts": 0,
                         "samples": [], "facts_new": []}


@pytest.mark.asyncio
async def test_extract_raising_callback_does_not_lose_facts():
    fx = _extractor(["Jane Doe is CEO of Acme"])
    facts = await fx.extract(_results(), "Jane Doe",
                             activity_callback=_raising_cb)
    assert len(facts) == 1


# ============================================================================
# Workflow plumbing: callback rides the INSTANCE, never LangGraph state
# ============================================================================

def _orch_with_fake_graph(snapshots):
    orch = ResearchOrchestrator(max_iterations=3, enable_checkpoints=False)

    class FakeGraph:
        def astream(self, initial_state, stream_mode):
            async def gen():
                for s in snapshots:
                    yield s
            return gen()

    orch.workflow = FakeGraph()
    orch._format_results = MagicMock(side_effect=lambda state: {
        "final": state, "facts": state.get("facts", []),
        "metadata": {"iterations": 0, "duration_seconds": 0.0},
    })
    return orch


@pytest.mark.asyncio
async def test_research_stores_activity_callback_on_instance():
    orch = _orch_with_fake_graph([{"stage": "complete", "iteration": 0}])
    assert orch._activity_callback is None          # default: attribute exists

    async def cb(event):
        pass

    await orch.research("Test Person", activity_callback=cb)
    assert orch._activity_callback is cb


@pytest.mark.asyncio
async def test_node_execute_searches_enriches_events_with_category_iteration():
    """A.3/R4-R5: nodes wrap the instance callback so search events carry the
    query's category and the server-side iteration (never on LangGraph state)."""
    orch = _orch_with_fake_graph([])
    events, cb = _collector()
    orch._activity_callback = cb

    async def fake_search(text, max_results, activity_callback):
        await activity_callback({"kind": "search", "engine": "brave",
                                 "query": text, "results": 1})
        return []

    orch.search_executor = SimpleNamespace(search=fake_search)
    orch.strategy_engine = MagicMock()
    state = {"target_name": "T", "iteration": 2, "errors": [],
             "queries": [SimpleNamespace(text="q1",
                                         category=SimpleNamespace(value="legal"))],
             "search_results": []}
    await orch._node_execute_searches(state)
    assert events == [{"kind": "search", "engine": "brave", "query": "q1",
                       "results": 1, "category": "legal", "iteration": 2}]
    # The callback must never be routed through LangGraph state
    # (must stay serializable — checkpoints are a V2 concern).
    assert not any(callable(v) for v in state.values())


@pytest.mark.asyncio
async def test_node_execute_searches_none_callback_stays_none():
    orch = _orch_with_fake_graph([])
    orch.search_executor = MagicMock()
    orch.search_executor.search = AsyncMock(return_value=[])
    orch.strategy_engine = MagicMock()
    state = {"target_name": "T", "iteration": 0, "errors": [],
             "queries": [SimpleNamespace(text="q1",
                                         category=SimpleNamespace(value="legal"))],
             "search_results": []}
    await orch._node_execute_searches(state)
    kwargs = orch.search_executor.search.await_args.kwargs
    assert kwargs["activity_callback"] is None      # no-callback fast path


@pytest.mark.asyncio
async def test_node_extract_facts_enriches_events_with_iteration():
    orch = _orch_with_fake_graph([])
    events, cb = _collector()
    orch._activity_callback = cb

    async def fake_extract(search_results, target_name, max_facts,
                           activity_callback):
        await activity_callback({"kind": "extract", "status": "start"})
        return []

    orch.fact_extractor = SimpleNamespace(extract=fake_extract)
    state = {"target_name": "T", "iteration": 3,
             "search_results": [object()],
             "search_results_processed_index": 0, "facts": [],
             "facts_per_iteration": []}
    await orch._node_extract_facts(state)
    assert events == [{"kind": "extract", "status": "start", "iteration": 3}]
    assert not any(callable(v) for v in state.values())


# ============================================================================
# jobs._apply_activity: merge rules for the ONE shared progress dict
# ============================================================================

def _search_event(i=0, query="jane doe ceo"):
    return {"kind": "search", "engine": "brave", "query": f"{query} {i}",
            "results": 5}


def test_apply_activity_bounds_feed_at_last_8():
    ps = {"phase": "data_collection"}
    for i in range(ACTIVITY_MAX + 2):
        _apply_activity(ps, _search_event(i))
    assert len(ps["activity"]) == ACTIVITY_MAX
    assert ps["activity"][0]["query"].endswith(" 2")     # oldest two dropped
    assert ps["activity"][-1]["query"].endswith(f" {ACTIVITY_MAX + 1}")
    assert ps["phase"] == "data_collection"              # never clobbered


def test_apply_activity_sample_facts_bounded_and_truncated():
    ps = {}
    long_fact = "Jane Doe " + "x" * 200
    _apply_activity(ps, {"kind": "extract", "status": "done", "facts": 4,
                         "samples": ["Jane Doe fact 1", "Jane Doe fact 2",
                                     "Jane Doe fact 3", long_fact]})
    assert len(ps["sample_facts"]) == SAMPLE_FACTS_MAX   # last 3 kept
    assert all(len(s) <= SAMPLE_FACT_CHARS for s in ps["sample_facts"])
    assert ps["sample_facts"][-1].endswith("…")


def test_apply_activity_strips_samples_from_feed_entry():
    ps = {}
    _apply_activity(ps, {"kind": "extract", "status": "done", "facts": 1,
                         "samples": ["Jane Doe fact"]})
    assert "samples" not in ps["activity"][0]
    assert ps["activity"][0]["facts"] == 1


def test_apply_activity_unknown_kind_and_garbage_are_safe():
    ps = {"phase": "x"}
    _apply_activity(ps, {"kind": "mystery", "note": "future event type"})
    assert ps["activity"][0]["kind"] == "mystery"
    _apply_activity(ps, "not a dict")                    # must not raise
    _apply_activity(ps, None)
    assert len(ps["activity"]) == 1


def test_apply_activity_hostile_content_flows_through_unmodified():
    """A4(a): the server does NOT sanitize scraped text — the client DOM
    boundary (textContent) is load-bearing. Prove the payload arrives raw."""
    ps = {}
    _apply_activity(ps, {"kind": "search", "engine": "brave",
                         "query": HOSTILE_TITLE, "results": 1})
    _apply_activity(ps, {"kind": "extract", "status": "done", "facts": 1,
                         "samples": [HOSTILE_FACT]})
    assert ps["activity"][0]["query"] == HOSTILE_TITLE
    assert ps["sample_facts"][0] == HOSTILE_FACT


# ============================================================================
# A.2/UX1: node-level LLM + refine events (PLAN.md Rev 3, plan-review-A2)
# ============================================================================

def _fact_ns(content, category="professional", confidence=0.9):
    return SimpleNamespace(content=content, category=category,
                           confidence=confidence, entities_mentioned=[],
                           verified=False, verification_count=1, source_urls=[])


@pytest.mark.asyncio
async def test_node_plan_strategy_emits_start_and_done():
    orch = _orch_with_fake_graph([])
    events, cb = _collector()
    orch._activity_callback = cb
    orch.strategy_engine = MagicMock()
    orch.strategy_engine.generate_initial_queries.return_value = [
        SimpleNamespace(text=f"q{i}") for i in range(15)
    ]
    await orch._node_plan_strategy({"target_name": "T", "context": {}})
    assert events[0] == {"kind": "llm", "task": "strategy_planning",
                         "status": "start"}
    assert events[1] == {"kind": "llm", "task": "strategy_planning",
                         "status": "done", "queries": 15}


@pytest.mark.asyncio
async def test_node_refine_emits_zero_queries_on_no_findings():
    """MI3: the refine event fires even on a zero-fact iteration."""
    orch = _orch_with_fake_graph([])
    events, cb = _collector()
    orch._activity_callback = cb
    orch.strategy_engine = MagicMock()
    state = {"target_name": "T", "iteration": 0, "facts": [], "queries": []}
    await orch._node_refine_queries(state)
    assert events == [{"kind": "llm", "task": "query_refinement",
                       "status": "done", "queries": 0, "sample_queries": []}]
    orch.strategy_engine.refine_based_on_findings.assert_not_called()


@pytest.mark.asyncio
async def test_node_refine_emits_count_and_sample_queries():
    orch = _orch_with_fake_graph([])
    events, cb = _collector()
    orch._activity_callback = cb
    orch.strategy_engine = MagicMock()
    orch.strategy_engine.refine_based_on_findings.return_value = [
        SimpleNamespace(text=f"jane doe follow-up {i}") for i in range(12)
    ]
    state = {"target_name": "T", "iteration": 0, "queries": [],
             "facts": [_fact_ns("Jane Doe is CEO of Acme")]}
    await orch._node_refine_queries(state)
    done = events[-1]
    assert done["queries"] == 12
    assert done["sample_queries"] == ["jane doe follow-up 0",
                                      "jane doe follow-up 1"]


@pytest.mark.asyncio
async def test_node_assess_risks_emits_model_on_success():
    orch = _orch_with_fake_graph([])
    events, cb = _collector()
    orch._activity_callback = cb
    orch.router = MagicMock()
    orch.router.route.return_value = SimpleNamespace(
        content="[]", provider=SimpleNamespace(value="anthropic"))
    state = {"target_name": "T", "connections": [],
             "facts": [_fact_ns(f"Jane Doe fact {i}") for i in range(6)]}
    await orch._node_assess_risks(state)
    assert events[0] == {"kind": "llm", "task": "risk_assessment",
                         "status": "start"}
    done = events[1]
    assert done["status"] == "done" and done["model"] == "Claude"
    assert done["risks"] == 0 and "severities" in done


@pytest.mark.asyncio
async def test_node_assess_risks_fallback_omits_model():
    """MI2: no ModelResponse in hand on the fallback path -> no model key.
    MA1 placement: the done event still fires after the fallback."""
    orch = _orch_with_fake_graph([])
    events, cb = _collector()
    orch._activity_callback = cb
    orch.router = MagicMock()
    orch.router.route.side_effect = RuntimeError("provider down")
    state = {"target_name": "T", "connections": [],
             "facts": [_fact_ns(f"Jane Doe fact {i}") for i in range(6)]}
    await orch._node_assess_risks(state)
    done = events[-1]
    assert done["task"] == "risk_assessment" and done["status"] == "done"
    assert "model" not in done
    assert done["risks"] == len(state["risk_flags"])


@pytest.mark.asyncio
async def test_node_map_connections_emits_count_sample_model():
    orch = _orch_with_fake_graph([])
    events, cb = _collector()
    orch._activity_callback = cb
    orch.router = MagicMock()
    orch.router.route.return_value = SimpleNamespace(
        content='[{"entity_1": "T", "entity_2": "Acme Corp",'
                ' "relationship_type": "employer"}]',
        provider=SimpleNamespace(value="anthropic"))
    state = {"target_name": "T",
             "facts": [_fact_ns("Jane Doe worked at Acme Corp") for _ in range(4)]}
    await orch._node_map_connections(state)
    done = events[-1]
    assert done == {"kind": "llm", "task": "connection_mapping",
                    "status": "done", "connections": 1,
                    "sample": ["Acme Corp"], "model": "Claude"}


@pytest.mark.asyncio
async def test_node_verify_emits_verified_and_deduped():
    orch = _orch_with_fake_graph([])
    events, cb = _collector()
    orch._activity_callback = cb
    dup = "Jane Doe is CEO of Acme Corp since 2010"
    state = {"target_name": "T",
             "facts": [_fact_ns(dup, confidence=0.9),
                       _fact_ns(dup, confidence=0.8)]}
    await orch._node_verify_facts(state)
    done = events[-1]
    assert done["task"] == "verification" and done["status"] == "done"
    assert done["deduped"] == 1
    assert done["verified"] >= 1


@pytest.mark.asyncio
async def test_node_emit_raising_callback_does_not_fail_node():
    """MA1: a raising callback must never fail a node or trip a fallback."""
    orch = _orch_with_fake_graph([])
    orch._activity_callback = _raising_cb
    orch.strategy_engine = MagicMock()
    orch.strategy_engine.generate_initial_queries.return_value = [
        SimpleNamespace(text="q")
    ]
    state = {"target_name": "T", "context": {}}
    await orch._node_plan_strategy(state)          # must not raise
    assert len(state["queries"]) == 1


# ============================================================================
# A.2/UX2: report_preview merge in jobs.py
# ============================================================================

def _extract_event(facts_new, n=None):
    return {"kind": "extract", "status": "done",
            "facts": n if n is not None else len(facts_new),
            "samples": [], "facts_new": facts_new}


def test_preview_merge_assigns_seq_ids_truncates_and_counts():
    ps = {}
    _apply_activity(ps, _extract_event([
        {"content": "Jane Doe " + "y" * 200, "category": "professional",
         "confidence": 0.9},
        {"content": "Jane Doe 2", "category": "biographical",
         "confidence": 0.8},
    ]))
    pv = ps["report_preview"]
    assert [f["id"] for f in pv["facts"]] == ["f1", "f2"]
    assert all(len(f["content"]) <= 160 for f in pv["facts"])
    assert pv["by_category"] == {"professional": 1, "biographical": 1}
    assert pv["facts_found"] == 2
    _apply_activity(ps, _extract_event(
        [{"content": "Jane Doe 3", "category": "legal", "confidence": 0.7}]))
    assert [f["id"] for f in ps["report_preview"]["facts"]] == ["f1", "f2", "f3"]
    assert ps["report_preview"]["facts_found"] == 3


def test_preview_caps_at_max_keeps_highest_confidence_in_arrival_order():
    from src.api.jobs import REPORT_FACTS_MAX
    ps = {}
    _apply_activity(ps, _extract_event([
        {"content": f"Jane Doe fact {i}", "category": "professional",
         "confidence": (i + 1) / 100}
        for i in range(REPORT_FACTS_MAX + 5)
    ]))
    facts = ps["report_preview"]["facts"]
    assert len(facts) == REPORT_FACTS_MAX
    # Lowest-confidence 5 (arrived first) dropped; order preserved.
    assert facts[0]["id"] == "f6"
    assert ps["report_preview"]["facts_found"] == REPORT_FACTS_MAX + 5


def test_preview_risks_connections_verification_and_hostile_passthrough():
    ps = {}
    _apply_activity(ps, {"kind": "llm", "task": "risk_assessment",
                         "status": "done", "risks": 9,
                         "severities": {"high": 2, "medium": 4, "low": 3},
                         "model": "Claude"})
    _apply_activity(ps, {"kind": "llm", "task": "connection_mapping",
                         "status": "done", "connections": 21,
                         "sample": [HOSTILE_TITLE, "Acme", "Beta", "extra"]})
    _apply_activity(ps, {"kind": "llm", "task": "verification",
                         "status": "done", "verified": 31, "deduped": 9})
    pv = ps["report_preview"]
    assert pv["risks"] == {"count": 9,
                           "severities": {"high": 2, "medium": 4, "low": 3}}
    assert pv["connections"]["count"] == 21
    assert pv["connections"]["sample"] == [HOSTILE_TITLE, "Acme", "Beta"]
    assert pv["verification"] == {"verified": 31, "deduped": 9}


def test_activity_entries_get_monotonic_seq_and_never_carry_facts_new():
    ps = {}
    _apply_activity(ps, _extract_event(
        [{"content": "Jane Doe x", "category": "legal", "confidence": 0.5}]))
    _apply_activity(ps, _search_event(0))
    assert [e["seq"] for e in ps["activity"]] == [1, 2]
    assert "facts_new" not in ps["activity"][0]


# ============================================================================
# A.3/A3.3: refine-query hygiene (plan-review-A3 R8/R9)
# ============================================================================

from src.core.workflow import _filter_refined_queries


def _q(text):
    return SimpleNamespace(text=text)


def test_filter_rejects_the_live_run_degenerate_form():
    refined = [_q('"Tim Cook" AND "Tim Cook"'), _q('"Tim Cook" biography')]
    kept = _filter_refined_queries(refined, [], "Tim Cook")
    assert [q.text for q in kept] == ['"Tim Cook" biography']   # R9: survives


def test_filter_degenerate_case_variance_and_company_target():
    assert _filter_refined_queries([_q("TIM COOK and tim cook")], [], "Tim Cook") == []
    assert _filter_refined_queries([_q('"Stripe" AND "Stripe"')], [], "Stripe") == []
    kept = _filter_refined_queries([_q('"Stripe" lawsuits')], [], "Stripe")
    assert len(kept) == 1


def test_filter_rejects_pending_queue_duplicates_and_intra_batch():
    pending = ["Tim Cook legal court lawsuit litigation regulatory"]
    refined = [
        _q("tim cook LEGAL court lawsuit litigation regulatory"),  # pending dup
        _q("Tim Cook net worth 2026"),
        _q("tim cook net worth 2026"),                             # intra-batch dup
    ]
    kept = _filter_refined_queries(refined, pending, "Tim Cook")
    assert [q.text for q in kept] == ["Tim Cook net worth 2026"]


def test_filter_survivors_apostrophes_and_boolean_augments():
    kept = _filter_refined_queries(
        [_q('"O\'Brien" lawsuits'), _q('"Tim Cook" AND "Apple"')],
        [], "O'Brien")
    assert len(kept) == 2                       # no false rejects (R9)
    assert _filter_refined_queries([_q("O'Brien"), _q("o brien")], [], "O'Brien") == []
