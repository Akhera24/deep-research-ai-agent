"""
Phase C1.7a — entity-scoped facts: about_target attribution + sideline pipeline.

Covers PLAN.md Rev 4.0b D10/D14 + review fixes:
- R1 (MAJOR): about_target is a hard merge boundary — the extractor's own
  dedup/cross-reference must never merge, dedup, or corroborate across it
  (including the exact-content seen_content path, and the all_facts
  accumulation cross-reference corroborates against).
- R4: coverage updates + facts_per_iteration count the TARGET partition only;
  high-sideline iterations log a counts-only warning.
- R5: the facts_new preview event is filtered to target at the emit site and
  carries a set_aside count (incl. the failure path); jobs' preview handles
  set_aside explicitly and facts_found never counts sidelined facts.
- R6: the report's sideline section captures its OWN raw-citation seam before
  _escape_deep (no double-encoded hrefs) and is DOM-bounded (30-fact cap).
- R7: the meta-fact drop is ANCHORED to statements about the extractor's own
  inputs; generic negative findings survive at any confidence.
- Fail-open: about_target missing/garbage -> True; bare runs (no context)
  keep an attribution-free prompt and an empty sideline pool.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.extraction.extractor import Fact, FactExtractor
from src.reporting.html_report import render_html_report
from src.search.executor import SearchResult


def _result(i, snippet="snippet", content=None, reliability=0.5, url=None):
    return SearchResult(
        query="q",
        url=url or f"https://example{i}.com/page",
        title=f"Title {i}",
        snippet=snippet,
        rank=i,
        search_engine="brave",
        fetched_at=datetime.now(),
        content=content,
        source_reliability=reliability,
    )


def _extractor():
    return FactExtractor(router=None, enable_verification=False)


def _fact(content, category="professional", confidence=0.8,
          about_target=True, urls=None):
    return Fact(
        content=content,
        category=category,
        confidence=confidence,
        about_target=about_target,
        source_urls=list(urls or []),
        evidence=[f"evidence for: {content[:40]}"],
    )


# ── Fact dataclass: the field exists, defaults True, serializes ────────────

class TestFactField:
    def test_default_true(self):
        assert Fact(content="x").about_target is True

    def test_to_dict_carries_field(self):
        assert _fact("x", about_target=False).to_dict()["about_target"] is False
        assert _fact("x").to_dict()["about_target"] is True


# ── _convert_to_fact_objects: strict-boolean allowlist, fail-open True ─────

def _convert(ex, data):
    return ex._convert_to_fact_objects(
        [data], [_result(1)], "John Smith", [_result(1)]
    )


BASE = {"content": "John Smith is a VP at NSSF", "category": "professional",
        "confidence": 0.9, "evidence": "quote", "source_ids": [1]}


class TestAboutTargetValidation:
    def test_literal_false_accepted(self):
        facts = _convert(_extractor(), {**BASE, "about_target": False})
        assert facts[0].about_target is False

    def test_literal_true_accepted(self):
        facts = _convert(_extractor(), {**BASE, "about_target": True})
        assert facts[0].about_target is True

    def test_missing_defaults_true(self):
        facts = _convert(_extractor(), dict(BASE))
        assert facts[0].about_target is True

    @pytest.mark.parametrize("garbage", ["false", "no", 0, 1, [], {}, None])
    def test_garbage_fails_open_true(self, garbage):
        facts = _convert(_extractor(), {**BASE, "about_target": garbage})
        assert facts[0].about_target is True

    def test_per_fact_independence(self):
        """One garbage sibling never poisons a valid classification."""
        ex = _extractor()
        facts = ex._convert_to_fact_objects(
            [{**BASE, "about_target": False},
             {**BASE, "content": "John Smith won an award", "about_target": "??"}],
            [_result(1)], "John Smith", [_result(1)],
        )
        assert [f.about_target for f in facts] == [False, True]


# ── D14/R7: anchored meta-fact drop — never a fact, at any confidence ──────

class TestMetaFactDrop:
    META = [
        "The provided text sources do not contain factual information "
        "regarding John Smith's financial records.",
        "These sources do not mention any litigation involving John Smith.",
        "The given source does not include information about John Smith's role.",
        "The source text lacks details about John Smith's early life.",
        "The sources provided contain no information about John Smith.",
    ]
    LEGIT = [
        "No litigation was found against John Smith.",
        "The SEC filing does not contain related-party transactions "
        "involving John Smith.",
        "John Smith stated the company does not include contractors "
        "in its headcount.",
        "These sources mention John Smith's award for community service.",
    ]

    @pytest.mark.parametrize("content", META)
    def test_meta_statement_dropped_even_at_high_confidence(self, content):
        facts = _convert(_extractor(), {**BASE, "content": content,
                                        "confidence": 0.98})
        assert facts == []

    @pytest.mark.parametrize("content", LEGIT)
    def test_negative_findings_survive(self, content):
        facts = _convert(_extractor(), {**BASE, "content": content})
        assert len(facts) == 1

    def test_drop_is_unconditional_no_context_needed(self):
        """The drop is defensive validation (like _validate_source_ids) —
        it fires on bare runs too (approved assumption 1)."""
        ex = _extractor()
        facts = ex._convert_to_fact_objects(
            [{**BASE, "content": self.META[0], "confidence": 0.98}],
            [_result(1)], "John Smith", [_result(1)],
        )
        assert facts == []


# ── Prompt gating: attribution requested ONLY when context exists ──────────

class TestPromptGating:
    def test_prompt_with_context_carries_attribution_and_meta_rule(self):
        ex = _extractor()
        p = ex._build_extraction_prompt(
            "[Source 1] text", "John Smith",
            context={"research_target": "John Smith — VP, NSSF"})
        assert "About target" in p
        assert '"about_target": true' in p
        assert "NEVER emit statements about the sources themselves" in p

    def test_bare_prompt_has_no_attribution(self):
        """CLI bare runs: the prompt is exactly the pre-C1.7 prompt —
        no attribution field, no meta rule, no example field."""
        ex = _extractor()
        p = ex._build_extraction_prompt("[Source 1] text", "John Smith")
        assert "about_target" not in p
        assert "About target" not in p
        assert "sources themselves" not in p


# ── R1: about_target is a hard merge boundary ──────────────────────────────

TARGET_TEXT = "John Smith is the vice president of finance at NSSF"
NEAR_DUPE_TEXT = "John Smith is the vice president of finance at NSSF today"


class TestMergeBoundary:
    def test_near_duplicate_across_boundary_survives_unmerged(self):
        """The R1 pin: a target x off-target near-duplicate pair survives
        dedup UNMERGED with provenance un-unioned."""
        ex = _extractor()
        target = _fact(TARGET_TEXT, confidence=0.7,
                       urls=["https://nssf.org/about"])
        off = _fact(NEAR_DUPE_TEXT, confidence=0.95, about_target=False,
                    urls=["https://rocketreach.co/johns"])
        unique = ex._deduplicate_facts([target, off])
        assert len(unique) == 2
        by_flag = {f.about_target: f for f in unique}
        assert by_flag[True].source_urls == ["https://nssf.org/about"]
        assert by_flag[False].source_urls == ["https://rocketreach.co/johns"]
        assert by_flag[True].verification_count == 1
        assert by_flag[False].verification_count == 1

    def test_exact_content_across_boundary_both_survive(self):
        """seen_content keyed by (content, about_target): an exact-text pair
        straddling the boundary must not silently drop one side."""
        ex = _extractor()
        pair = [_fact(TARGET_TEXT, confidence=0.9),
                _fact(TARGET_TEXT, confidence=0.8, about_target=False)]
        unique = ex._deduplicate_facts(pair)
        assert len(unique) == 2
        assert {f.about_target for f in unique} == {True, False}

    def test_same_class_near_duplicates_still_merge(self):
        """Regression: the boundary gate must not break in-class merging."""
        ex = _extractor()
        a = _fact(TARGET_TEXT, confidence=0.9, urls=["https://a.com/1"])
        b = _fact(NEAR_DUPE_TEXT, confidence=0.7, urls=["https://b.com/2"])
        unique = ex._deduplicate_facts([a, b])
        assert len(unique) == 1
        assert set(unique[0].source_urls) == {"https://a.com/1", "https://b.com/2"}
        assert unique[0].verification_count == 2

    def test_cross_reference_never_corroborates_across_boundary(self):
        """all_facts accumulates BOTH classes across iterations; a target
        fact must not gain confidence from an off-target sibling."""
        ex = FactExtractor(router=None, enable_verification=True)
        ex.all_facts = [_fact(TARGET_TEXT, about_target=False)]
        fresh = _fact(NEAR_DUPE_TEXT, confidence=0.8)
        [out] = ex._cross_reference_facts([fresh])
        assert out.verified is False
        assert out.verification_count == 1
        assert out.confidence == 0.8

    def test_cross_reference_same_class_still_corroborates(self):
        ex = FactExtractor(router=None, enable_verification=True)
        ex.all_facts = [_fact(TARGET_TEXT)]
        fresh = _fact(NEAR_DUPE_TEXT, confidence=0.8)
        [out] = ex._cross_reference_facts([fresh])
        assert out.verified is True
        assert out.verification_count == 2


# ── R5: facts_new filtered at the emit site + set_aside count ──────────────

class EmitRouter:
    """Returns a mixed target/off-target extraction batch."""
    def __init__(self):
        self.prompts = []

    def route_and_call(self, task_type=None, prompt="", **kwargs):
        self.prompts.append(prompt)
        import json
        content = json.dumps([
            {"content": "John Smith is a VP at NSSF",
             "category": "professional", "confidence": 0.9,
             "evidence": "q1", "source_ids": [1], "about_target": True},
            {"content": "John Smith explored Virginia in 1607",
             "category": "biographical", "confidence": 0.85,
             "evidence": "q2", "source_ids": [1], "about_target": False},
            {"content": "John Smith works at HSBC on mortgages",
             "category": "professional", "confidence": 0.8,
             "evidence": "q3", "source_ids": [1], "about_target": False},
        ])
        return SimpleNamespace(content=content, cost=0.0,
                               provider=SimpleNamespace(value="google"),
                               model_name="fake")


class TestEmitSiteFilter:
    @pytest.mark.asyncio
    async def test_facts_new_target_only_with_set_aside_count(self):
        ex = FactExtractor(EmitRouter(), enable_verification=False)
        events = []

        async def cb(event):
            events.append(event)

        returned = await ex.extract(
            [_result(1)], "John Smith", activity_callback=cb,
            target_context={"research_target": "John Smith — VP, NSSF"},
        )
        # The RETURN stays the mixed pool — the workflow partitions it.
        assert len(returned) == 3
        done = [e for e in events if e.get("status") == "done"][0]
        assert done["facts"] == 1
        assert done["set_aside"] == 2
        assert [f["content"] for f in done["facts_new"]] == [
            "John Smith is a VP at NSSF"]
        assert all("explored" not in s and "HSBC" not in s
                   for s in done["samples"])

    @pytest.mark.asyncio
    async def test_failure_path_event_carries_set_aside_zero(self):
        class BoomRouter:
            def route_and_call(self, **kwargs):
                raise RuntimeError("boom")

        ex = FactExtractor(BoomRouter(), enable_verification=False)
        # Force total failure past the regex fallback by breaking prepare
        ex._prepare_text_for_extraction = MagicMock(side_effect=RuntimeError)
        events = []

        async def cb(event):
            events.append(event)

        out = await ex.extract([_result(1)], "John Smith",
                               activity_callback=cb)
        assert out == []
        done = [e for e in events if e.get("status") == "done"][0]
        assert done["facts"] == 0
        assert done["set_aside"] == 0
        assert done["facts_new"] == []


# ── Workflow partition (R4): target-only coverage + stagnation ─────────────

def _orchestrator():
    from src.core.workflow import ResearchOrchestrator
    return ResearchOrchestrator(max_iterations=2, enable_checkpoints=False)


def _node_state(context=None, facts=None):
    return {
        "target_name": "John Smith", "iteration": 1,
        "context": context,
        "search_results": [SimpleNamespace(url="https://x.com")],
        "search_results_processed_index": 0,
        "facts": facts if facts is not None else [],
        "coverage": {},
    }


MIXED_BATCH = [
    _fact("John Smith is a VP at NSSF", category="professional"),
    _fact("John Smith explored Virginia in 1607", category="biographical",
          about_target=False),
    _fact("John Smith works at HSBC", category="professional",
          about_target=False),
    _fact("John Smith manages healthcare sales", category="professional",
          about_target=False),
]


class TestWorkflowPartition:
    @pytest.mark.asyncio
    async def test_partition_and_target_only_counters(self):
        orch = _orchestrator()
        orch.fact_extractor = MagicMock()
        orch.fact_extractor.extract = AsyncMock(return_value=list(MIXED_BATCH))
        orch.strategy_engine = MagicMock()
        state = _node_state(context={"research_target": "John Smith — VP, NSSF"})
        await orch._node_extract_facts(state)

        assert [f.content for f in state["facts"]] == [
            "John Smith is a VP at NSSF"]
        assert [f.content for f in state["sidelined_facts"]] == [
            "John Smith explored Virginia in 1607",
            "John Smith works at HSBC",
            "John Smith manages healthcare sales",
        ]
        # R4: coverage bumped ONLY for the target fact's category
        cats = [c.args[0].value for c in
                orch.strategy_engine.update_coverage.call_args_list]
        assert cats == ["professional"]
        # R4: stagnation counts the target partition only
        assert state["facts_per_iteration"] == [1]

    @pytest.mark.asyncio
    async def test_high_sideline_warning_counts_only(self, capsys):
        """structlog here uses PrintLoggerFactory (stdout), so the warning
        is asserted on captured stdout, not caplog."""
        orch = _orchestrator()
        orch.fact_extractor = MagicMock()
        orch.fact_extractor.extract = AsyncMock(return_value=list(MIXED_BATCH))
        orch.strategy_engine = MagicMock()
        state = _node_state(context={"research_target": "John Smith — VP, NSSF"})
        await orch._node_extract_facts(state)
        out = capsys.readouterr().out
        warn_lines = [ln for ln in out.splitlines()
                      if "high-sideline" in ln.lower()]
        assert warn_lines, "high-sideline warning not logged"
        assert "John Smith" not in warn_lines[0]

    @pytest.mark.asyncio
    async def test_no_warning_when_target_dominates(self, capsys):
        orch = _orchestrator()
        orch.fact_extractor = MagicMock()
        orch.fact_extractor.extract = AsyncMock(return_value=[
            _fact("John Smith is a VP at NSSF"),
            _fact("John Smith joined NSSF in 2015"),
            _fact("John Smith explored Virginia", about_target=False),
        ])
        orch.strategy_engine = MagicMock()
        state = _node_state(context={"research_target": "John Smith — VP, NSSF"})
        await orch._node_extract_facts(state)
        out = capsys.readouterr().out
        assert "high-sideline" not in out.lower()

    @pytest.mark.asyncio
    async def test_sidelined_accumulate_across_iterations(self):
        orch = _orchestrator()
        orch.fact_extractor = MagicMock()
        orch.fact_extractor.extract = AsyncMock(return_value=[
            _fact("John Smith works at HSBC", about_target=False)])
        orch.strategy_engine = MagicMock()
        state = _node_state(context={"research_target": "x"})
        state["sidelined_facts"] = [
            _fact("John Smith explored Virginia", about_target=False)]
        await orch._node_extract_facts(state)
        assert len(state["sidelined_facts"]) == 2

    @pytest.mark.asyncio
    async def test_bare_run_sidelines_nothing(self):
        """No context -> attribution never requested -> defaults keep every
        fact in the main pool (CLI bare runs unchanged)."""
        orch = _orchestrator()
        orch.fact_extractor = MagicMock()
        orch.fact_extractor.extract = AsyncMock(return_value=[
            _fact("John Smith is a VP at NSSF"),
            _fact("John Smith explored Virginia"),
        ])
        orch.strategy_engine = MagicMock()
        state = _node_state(context=None)
        await orch._node_extract_facts(state)
        assert len(state["facts"]) == 2
        assert state["sidelined_facts"] == []

    @pytest.mark.asyncio
    async def test_refine_findings_exclude_sidelined(self):
        """The refinement drift-amplifier is dead: findings build from the
        target partition only."""
        orch = _orchestrator()
        orch.strategy_engine = MagicMock()
        orch.strategy_engine.refine_based_on_findings.return_value = []
        state = {
            "target_name": "John Smith", "iteration": 0, "queries": [],
            "facts": [_fact("John Smith is a VP at NSSF")],
            "sidelined_facts": [_fact("John Smith explored Virginia",
                                      about_target=False)],
            "context": {"research_target": "x"},
        }
        await orch._node_refine_queries(state)
        findings = orch.strategy_engine.refine_based_on_findings.call_args \
            .kwargs["findings"]
        assert len(findings) == 1
        assert "explored" not in findings[0]["content"]


# ── _format_results: sidelined ride INSIDE the result dict ─────────────────

class TestFormatResults:
    def _state(self, sidelined):
        return {
            "target_name": "John Smith", "iteration": 2,
            "facts": [_fact("John Smith is a VP at NSSF")],
            "sidelined_facts": sidelined,
            "risk_flags": [], "connections": [], "summary": {},
            "start_time": datetime.now(), "completed_at": datetime.now(),
        }

    def test_sidelined_facts_and_count_in_result(self):
        orch = _orchestrator()
        out = orch._format_results(self._state(
            [_fact("John Smith explored Virginia", about_target=False)]))
        assert len(out["sidelined_facts"]) == 1
        assert out["sidelined_facts"][0]["about_target"] is False
        assert out["metadata"]["sidelined_count"] == 1
        assert len(out["facts"]) == 1

    def test_empty_sideline_pool(self):
        orch = _orchestrator()
        out = orch._format_results(self._state([]))
        assert out["sidelined_facts"] == []
        assert out["metadata"]["sidelined_count"] == 0


# ── R6: report sideline section — own raw seam, 30-cap, collapsed ──────────

def _render_result(sidelined=None, facts=None):
    result = {
        "target_name": "John Smith",
        "facts": facts if facts is not None else [
            {"content": "John Smith is a VP at NSSF",
             "category": "professional", "confidence": 0.9,
             "source_urls": ["https://nssf.org/about"],
             "source_reliabilities": {"https://nssf.org/about": 0.8},
             "evidence": ["John Smith serves as VP"], "verified": True,
             "verification_count": 2},
        ],
        "risk_flags": [], "connections": [],
        "metadata": {"coverage": {"average": 0.5}, "iterations": 2,
                     "duration_seconds": 10.0},
    }
    if sidelined is not None:
        result["sidelined_facts"] = sidelined
    return result


def _sidelined_fact(i=1, content=None, url=None):
    return {
        "content": content or f"John Smith (explorer) mapped Virginia #{i}",
        "category": "biographical", "confidence": 0.8,
        "about_target": False,
        "source_urls": [url or f"https://history{i}.example/smith?a=1&b=2"],
        "source_reliabilities": {},
        "evidence": [f"colonial records quote {i}"],
    }


class TestSidelineSection:
    def test_section_renders_collapsed_with_count_and_disclaimer(self):
        html = render_html_report(
            _render_result(sidelined=[_sidelined_fact()]), "John Smith", 10.0)
        assert "Facts about other people named John Smith (1)" in html
        assert "not included in the analysis or score" in html
        assert "John Smith (explorer) mapped Virginia #1" in html

    def test_own_raw_seam_no_double_encoded_hrefs(self):
        """R6: chips built from the RAW seam — an & in the URL must encode
        exactly once (&amp;), never &amp;amp; / %26amp%3B."""
        html = render_html_report(
            _render_result(sidelined=[_sidelined_fact()]), "John Smith", 10.0)
        assert 'href="https://history1.example/smith?a=1&amp;b=2' in html
        assert "%26amp" not in html
        assert "&amp;amp;" not in html

    def test_dom_cap_30_with_overflow_line(self):
        sidelined = [_sidelined_fact(i) for i in range(1, 38)]
        html = render_html_report(
            _render_result(sidelined=sidelined), "John Smith", 10.0)
        assert html.count("John Smith (explorer) mapped Virginia #") == 30
        assert "and 7 more set aside" in html

    def test_exactly_30_no_overflow_line(self):
        sidelined = [_sidelined_fact(i) for i in range(1, 31)]
        html = render_html_report(
            _render_result(sidelined=sidelined), "John Smith", 10.0)
        assert "more set aside" not in html

    def test_no_section_when_absent_or_empty(self):
        for result in (_render_result(), _render_result(sidelined=[])):
            html = render_html_report(result, "John Smith", 10.0)
            assert "Facts about other people named" not in html

    def test_hostile_sidelined_content_escaped(self):
        payload = '<script>window.__pwned=1</script>'
        html = render_html_report(
            _render_result(sidelined=[_sidelined_fact(content=payload)]),
            "John Smith", 10.0)
        assert payload not in html
        assert "&lt;script&gt;window.__pwned=1&lt;/script&gt;" in html

    def test_javascript_url_in_sidelined_fact_not_linked(self):
        html = render_html_report(
            _render_result(sidelined=[_sidelined_fact(
                url="javascript:alert(1)")]), "John Smith", 10.0)
        assert 'href="javascript:' not in html

    def test_all_sidelined_zero_fact_main_list_renders(self):
        """Hopeless mixed pool: 0-fact main list + populated sideline."""
        html = render_html_report(
            _render_result(facts=[], sidelined=[_sidelined_fact()]),
            "John Smith", 10.0)
        assert "Facts about other people named John Smith (1)" in html

    def test_score_ignores_sidelined_facts(self):
        """Gate (1): the score computes on target facts only — identical
        with and without the sideline pool."""
        import re as _re
        with_side = render_html_report(
            _render_result(sidelined=[_sidelined_fact(i) for i in range(9)]),
            "John Smith", 10.0)
        without = render_html_report(_render_result(), "John Smith", 10.0)
        pat = _re.compile(r"Quality Score (\d+(?:\.\d+)?)/100")
        assert pat.search(with_side).group(1) == pat.search(without).group(1)


# ── F3: jobs preview — set_aside handled explicitly, never in facts_found ──

class TestPreviewSetAside:
    def _apply(self, state, event):
        from src.api.jobs import _apply_report_preview
        _apply_report_preview(state, event)

    def test_set_aside_accumulates_even_with_empty_facts_new(self):
        """An ALL-sidelined batch has facts_new=[] but a set_aside count —
        the early return must not swallow it."""
        state = {}
        self._apply(state, {"kind": "extract", "status": "done",
                            "facts": 0, "facts_new": [], "set_aside": 4})
        self._apply(state, {"kind": "extract", "status": "done",
                            "facts": 1, "set_aside": 2,
                            "facts_new": [{"content": "John Smith is a VP",
                                           "category": "professional",
                                           "confidence": 0.9}]})
        preview = state["report_preview"]
        assert preview["set_aside"] == 6
        assert preview["facts_found"] == 1

    @pytest.mark.parametrize("garbage", ["4", True, -1, None, {}])
    def test_garbage_set_aside_ignored(self, garbage):
        state = {}
        self._apply(state, {"kind": "extract", "status": "done",
                            "facts": 0, "facts_new": [],
                            "set_aside": garbage})
        assert state.get("report_preview", {}).get("set_aside", 0) == 0
