"""
C1.0 pre-flight module tests (PLAN.md Rev 3.8, Phase C).

Everything external is faked (search executor, Gemini client) — the module's
own logic runs for real: search fan-out, clustering-response parsing with
truncation repair, per-candidate validation with per-item drop, server-side
domain-mass computation (eTLD+1), the D2 dominance gate, and the hints
hard pre-filter. Edge-case list from PLAN.md C1+C2 is the checklist.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.core.preflight import (
    PreflightResult,
    discover_candidates,
    registrable_domain,
    _build_clustering_prompt,
    _validate_candidates,
)
from src.search.executor import SearchResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _result(url, title="t", snippet="s", rank=1):
    return SearchResult(
        query="q", url=url, title=title, snippet=snippet, rank=rank,
        search_engine="fake", fetched_at=datetime.now(),
    )


class FakeExecutor:
    """Deterministic search results; records queries; can be told to fail."""

    def __init__(self, results=None, fail=False):
        self.results = results if results is not None else []
        self.fail = fail
        self.queries = []

    async def search(self, query, max_results=10, engine=None,
                     activity_callback=None):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("all engines down")
        return self.results[:max_results]


class FakeLLM:
    """Returns a canned content string as a ModelResponse-shaped object."""

    def __init__(self, content, cost=0.002, fail=False):
        self.content = content
        self.cost = cost
        self.fail = fail
        self.prompts = []

    def call(self, prompt, system_prompt=None, task_type=None, **kwargs):
        self.prompts.append((system_prompt or "") + "\n" + prompt)
        if self.fail:
            raise RuntimeError("model unavailable")
        return SimpleNamespace(content=self.content, cost=self.cost)


def _run(coro):
    return asyncio.run(coro)


def _candidates_json(candidates):
    return json.dumps(candidates)


# Ten results across clearly distinct domains: indices 1..10 (1-based in the
# prompt). Two entities: results 1-5 (5 distinct domains), results 6-7
# (2 distinct domains), plus 8-10 spare.
RESULTS = [
    _result("https://www.stripe.com/about", "Jane Doe — Stripe VP", rank=1),
    _result("https://en.wikipedia.org/wiki/Jane_Doe", "Jane Doe (executive)", rank=2),
    _result("https://www.linkedin.com/in/janedoe", "Jane Doe - VP Eng at Stripe", rank=3),
    _result("https://techcrunch.com/janedoe", "Stripe's Jane Doe on payments", rank=4),
    _result("https://forbes.com/janedoe", "Jane Doe, Stripe", rank=5),
    _result("https://uklaw.example.co.uk/janedoe", "Jane Doe QC, London barrister", rank=6),
    _result("https://chambers.example.com/jane-doe", "Jane Doe — barrister", rank=7),
    _result("https://random1.example.org/jd", "Jane Doe mention", rank=8),
    _result("https://random2.example.net/jd", "Jane Doe mention", rank=9),
    _result("https://random3.example.io/jd", "Jane Doe mention", rank=10),
]

TWO_CLUSTERS = [
    {"canonical_name": "Jane Doe", "descriptor": "Jane Doe — VP Engineering, Stripe (San Francisco)",
     "disambiguators": ["Stripe", "VP Engineering"], "supporting_results": [1, 2, 3, 4, 5]},
    {"canonical_name": "Jane Doe", "descriptor": "Jane Doe — Barrister, London (UK)",
     "disambiguators": ["barrister", "London"], "supporting_results": [6, 7]},
]


# ---------------------------------------------------------------------------
# registrable_domain (eTLD+1 approximation)
# ---------------------------------------------------------------------------

class TestRegistrableDomain:
    def test_strips_subdomains(self):
        assert registrable_domain("https://www.stripe.com/about") == "stripe.com"
        assert registrable_domain("https://blog.deep.stripe.com/x") == "stripe.com"

    def test_two_part_suffixes(self):
        assert registrable_domain("https://uklaw.example.co.uk/x") == "example.co.uk"
        assert registrable_domain("https://www.acme.com.au/") == "acme.com.au"

    def test_bare_and_invalid(self):
        assert registrable_domain("not a url") is None
        assert registrable_domain("") is None
        assert registrable_domain("https://localhost/x") == "localhost"


# ---------------------------------------------------------------------------
# Candidate validation (per-item drop, mirrors _validate_source_ids)
# ---------------------------------------------------------------------------

class TestValidateCandidates:
    def test_per_item_drop_keeps_valid(self):
        raw = [
            {"canonical_name": "A", "descriptor": "A — x", "disambiguators": ["d"],
             "supporting_results": [1, 2]},
            {"descriptor": "no name"},                     # dropped: no name
            "not a dict",                                  # dropped
            {"canonical_name": "", "supporting_results": [1]},   # dropped: empty name
            {"canonical_name": "B", "descriptor": 5, "disambiguators": "oops",
             "supporting_results": [2]},                   # kept, fields coerced
        ]
        out = _validate_candidates(raw, num_results=5)
        assert [c["canonical_name"] for c in out] == ["A", "B"]
        assert out[1]["descriptor"] == ""          # non-str descriptor dropped
        assert out[1]["disambiguators"] == []      # non-list coerced to empty

    def test_supporting_indices_per_id_drop(self):
        raw = [{"canonical_name": "A",
                "supporting_results": [1, 99, 0, 3, -2, True, "2", 3]}]
        out = _validate_candidates(raw, num_results=5)
        # bool is not an index; strings dropped; out-of-range dropped; deduped
        assert out[0]["supporting_results"] == [1, 3]

    def test_non_list_input_returns_empty(self):
        assert _validate_candidates("nope", 5) == []
        assert _validate_candidates({"a": 1}, 5) == []

    def test_descriptor_truncated_to_120(self):
        raw = [{"canonical_name": "A", "descriptor": "x" * 300,
                "supporting_results": [1]}]
        out = _validate_candidates(raw, num_results=3)
        assert len(out[0]["descriptor"]) <= 120

    def test_control_chars_stripped_but_text_raw(self):
        # XSS-bearing text flows through RAW (the client boundary is
        # load-bearing, A4a) — only control chars are stripped.
        raw = [{"canonical_name": "A\x00B <script>alert(1)</script>",
                "descriptor": "d\x1b[31m — e", "supporting_results": [1]}]
        out = _validate_candidates(raw, num_results=2)
        assert "\x00" not in out[0]["canonical_name"]
        assert "\x1b" not in out[0]["descriptor"]
        assert "<script>alert(1)</script>" in out[0]["canonical_name"]


# ---------------------------------------------------------------------------
# D2 gate through discover_candidates
# ---------------------------------------------------------------------------

class TestGate:
    def test_ambiguous_two_clusters_pick(self):
        # top mass 5, runner-up 2 → runner-up > 1 → picker
        llm = FakeLLM(_candidates_json(TWO_CLUSTERS))
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision == "pick"
        assert len(res.candidates) == 2
        # most-documented first
        assert res.candidates[0].domain_mass == 5
        assert res.candidates[1].domain_mass == 2
        assert res.cost > 0

    def test_dominant_cluster_auto(self):
        clusters = [
            dict(TWO_CLUSTERS[0]),
            {"canonical_name": "Jane Doe", "descriptor": "Jane Doe — mention",
             "disambiguators": [], "supporting_results": [6]},
        ]
        llm = FakeLLM(_candidates_json(clusters))
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(RESULTS), llm_client=llm))
        # top 5 ≥ 3, runner-up 1 ≤ 1, 5 ≥ 0.7·6? No — 0.7·6 = 4.2, 5 ≥ 4.2 ✓
        assert res.decision == "auto"
        assert res.note == "dominant"
        assert res.candidates[0].domain_mass == 5

    def test_share_clause_blocks_auto(self):
        # top 3, runner-up 1, but another cluster also mass 3 → share fails
        clusters = [
            {"canonical_name": "A", "supporting_results": [1, 2, 4]},
            {"canonical_name": "B", "supporting_results": [6, 7, 8]},
            {"canonical_name": "C", "supporting_results": [9]},
        ]
        llm = FakeLLM(_candidates_json(clusters))
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision == "pick"

    def test_single_strong_cluster_auto(self):
        clusters = [dict(TWO_CLUSTERS[0])]
        llm = FakeLLM(_candidates_json(clusters))
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision == "auto"
        assert res.note == "single"

    def test_single_thin_cluster_unscoped(self):
        clusters = [{"canonical_name": "Jane Doe", "supporting_results": [1]}]
        llm = FakeLLM(_candidates_json(clusters))
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision == "unscoped"
        assert "thin" in (res.note or "")

    def test_zero_clusters_unscoped(self):
        llm = FakeLLM("[]")
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision == "unscoped"

    def test_mass_is_distinct_domains_not_result_count(self):
        # 4 results but only 2 distinct registrable domains → mass 2
        results = [
            _result("https://a.example.com/1"),
            _result("https://a.example.com/2"),
            _result("https://www.example.com/3"),
            _result("https://b.other.org/4"),
        ]
        clusters = [{"canonical_name": "A", "supporting_results": [1, 2, 3, 4]}]
        llm = FakeLLM(_candidates_json(clusters))
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(results), llm_client=llm))
        assert res.candidates[0].domain_mass == 2

    def test_fabricated_indices_do_not_inflate_mass(self):
        clusters = [{"canonical_name": "A",
                     "supporting_results": [1, 77, 88, 99]}]
        llm = FakeLLM(_candidates_json(clusters))
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.candidates[0].domain_mass == 1


# ---------------------------------------------------------------------------
# Hints hard pre-filter (D3)
# ---------------------------------------------------------------------------

class TestHints:
    def test_exactly_one_consistent_auto_hinted_beats_fame(self):
        # Famous cluster (mass 5) does NOT match the hint; obscure one does.
        llm = FakeLLM(_candidates_json(TWO_CLUSTERS))
        res = _run(discover_candidates(
            "Jane Doe", hints={"role": "barrister"},
            executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision == "auto"
        assert res.note == "hinted"
        assert "Barrister" in res.candidates[0].descriptor

    def test_zero_consistent_returns_pick_with_note_and_unfiltered(self):
        # D7 (Rev 3.9): the user expressed SPECIFIC intent — going broad
        # contradicts it. Zero hint-consistent clusters → the full picker
        # with a note, never a silent unscoped run.
        llm = FakeLLM(_candidates_json(TWO_CLUSTERS))
        res = _run(discover_candidates(
            "Jane Doe", hints={"company": "NASA"},
            executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision == "pick"
        assert "match" in (res.note or "")
        # unfiltered set, still mass-ranked
        assert len(res.candidates) == 2
        assert res.candidates[0].domain_mass >= res.candidates[1].domain_mass

    def test_zero_consistent_single_cluster_still_pick_with_note(self):
        # R6: a pick can carry ONE candidate — the client must still show
        # the decision surface, so the server must not degrade to unscoped.
        clusters = [dict(TWO_CLUSTERS[0])]
        llm = FakeLLM(_candidates_json(clusters))
        res = _run(discover_candidates(
            "Jane Doe", hints={"company": "NASA"},
            executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision == "pick"
        assert len(res.candidates) == 1
        assert res.note

    def test_multiple_consistent_normal_gate(self):
        # Both clusters mention "Jane" trivially; use a hint both satisfy
        clusters = [
            {"canonical_name": "Jane Doe",
             "descriptor": "Jane Doe — VP, Stripe (SF)",
             "disambiguators": ["Stripe", "engineer"],
             "supporting_results": [1, 2, 3, 4, 5]},
            {"canonical_name": "Jane Doe",
             "descriptor": "Jane Doe — engineer, Boeing (Seattle)",
             "disambiguators": ["Boeing", "engineer"],
             "supporting_results": [6, 7]},
        ]
        llm = FakeLLM(_candidates_json(clusters))
        res = _run(discover_candidates(
            "Jane Doe", hints={"role": "engineer"},
            executor=FakeExecutor(RESULTS), llm_client=llm))
        # both consistent → gate applies: top 5, runner 2 → pick
        assert res.decision == "pick"
        assert len(res.candidates) == 2

    def test_hint_search_query_included(self):
        ex = FakeExecutor(RESULTS)
        llm = FakeLLM(_candidates_json(TWO_CLUSTERS))
        _run(discover_candidates(
            "Jane Doe", hints={"company": "Stripe"},
            executor=ex, llm_client=llm))
        assert any("Stripe" in q for q in ex.queries)


# ---------------------------------------------------------------------------
# Robustness: truncation, malformed output, failures (fail-open = "error")
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_truncated_json_array_repaired(self):
        full = _candidates_json(TWO_CLUSTERS)
        cut = full[:full.rindex('{') - 2]     # cut mid-array, second obj lost
        llm = FakeLLM("```json\n" + cut)      # opening fence, no close
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision in ("auto", "pick", "unscoped")
        assert len(res.candidates) >= 1
        assert res.candidates[0].canonical_name == "Jane Doe"

    def test_unparseable_llm_output_error(self):
        llm = FakeLLM("I could not find any entities, sorry!")
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision == "error"
        assert res.candidates == []

    def test_llm_raises_error(self):
        llm = FakeLLM("", fail=True)
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision == "error"

    def test_all_searches_fail_error(self):
        llm = FakeLLM(_candidates_json(TWO_CLUSTERS))
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(fail=True), llm_client=llm))
        assert res.decision == "error"

    def test_searches_run_concurrently(self):
        # C1.6a: the 2-3 pre-flight searches must overlap (gather), not run
        # sequentially — ~2-3s of avoidable latency per submit.
        class OverlapExecutor(FakeExecutor):
            active = 0
            max_active = 0

            async def search(self, query, max_results=10, engine=None,
                             activity_callback=None):
                cls = type(self)
                cls.active += 1
                cls.max_active = max(cls.max_active, cls.active)
                await asyncio.sleep(0.05)
                cls.active -= 1
                return RESULTS[:3]

        OverlapExecutor.active = OverlapExecutor.max_active = 0
        llm = FakeLLM(_candidates_json(TWO_CLUSTERS))
        _run(discover_candidates("Jane Doe", hints={"company": "Stripe"},
                                 executor=OverlapExecutor(), llm_client=llm))
        assert OverlapExecutor.max_active >= 2

    def test_partial_search_failure_uses_surviving_results(self):
        class FlakyExecutor(FakeExecutor):
            async def search(self, query, max_results=10, engine=None,
                             activity_callback=None):
                if query.startswith('"'):
                    raise RuntimeError("engine down")
                return RESULTS[:5]

        llm = FakeLLM(_candidates_json([dict(TWO_CLUSTERS[0])]))
        res = _run(discover_candidates(
            "Jane Doe", executor=FlakyExecutor(), llm_client=llm))
        assert res.decision != "error"          # one search survived
        assert len(res.candidates) == 1

    def test_dedupe_keeps_first_query_order(self):
        # URL seen by search #1 must not be re-added (or re-ordered) by
        # search #2 — order semantics identical to the old sequential loop.
        class OrderedExecutor(FakeExecutor):
            async def search(self, query, max_results=10, engine=None,
                             activity_callback=None):
                if query.startswith('"'):
                    return [RESULTS[1], RESULTS[5]]   # 1 overlap + 1 new
                return RESULTS[:3]

        llm = FakeLLM(_candidates_json(TWO_CLUSTERS))
        _run(discover_candidates(
            "Jane Doe", executor=OrderedExecutor(), llm_client=llm))
        prompt = llm.prompts[0]
        # wikipedia (the overlap) numbered exactly once; the new co.uk URL
        # lands AFTER search #1's block (position 4)
        assert prompt.count(RESULTS[1].url) == 1
        assert f"[4] {RESULTS[5].title}" in prompt

    def test_failure_logs_never_carry_the_raw_query(self, caplog):
        # §12.S3: executor exceptions embed the query in their message —
        # pre-flight logging must record the exception TYPE only.
        class LeakyExecutor(FakeExecutor):
            async def search(self, query, max_results=10, engine=None,
                             activity_callback=None):
                raise RuntimeError(
                    f"All search methods failed for query: {query}")

        llm = FakeLLM(_candidates_json(TWO_CLUSTERS))
        with caplog.at_level("DEBUG"):
            res = _run(discover_candidates(
                "Confidential Person Name",
                executor=LeakyExecutor(), llm_client=llm))
        assert res.decision == "error"
        assert "Confidential Person Name" not in caplog.text

    def test_empty_search_results_unscoped_no_llm_call(self):
        llm = FakeLLM(_candidates_json(TWO_CLUSTERS))
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor([]), llm_client=llm))
        assert res.decision == "unscoped"
        assert llm.prompts == []          # no spend on an empty pool
        assert res.cost == 0.0

    def test_more_than_five_candidates_capped_by_mass(self):
        clusters = [
            {"canonical_name": f"C{i}", "supporting_results": [i]}
            for i in range(1, 8)
        ]
        clusters[5]["supporting_results"] = [1, 2, 3]   # C6: mass 3
        llm = FakeLLM(_candidates_json(clusters))
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(RESULTS), llm_client=llm))
        assert len(res.candidates) == 5
        assert res.candidates[0].canonical_name == "C6"

    def test_result_returns_raw_llm_text_for_client(self):
        # Hostile descriptor arrives RAW in the result — escaping is the
        # client's job (textContent), not the module's.
        clusters = [{"canonical_name": "Jane <img onerror=alert(1) src=x>",
                     "descriptor": "javascript:alert(1)",
                     "supporting_results": [1, 2, 3]}]
        llm = FakeLLM(_candidates_json(clusters))
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(RESULTS), llm_client=llm))
        assert "<img onerror" in res.candidates[0].canonical_name
        assert res.candidates[0].descriptor == "javascript:alert(1)"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

class TestPrompt:
    def test_prompt_numbers_results_and_carries_split_bias(self):
        prompt = _build_clustering_prompt("Jane Doe", RESULTS[:3], None)
        assert "[1]" in prompt and "[3]" in prompt
        assert "KEEP THEM SEPARATE" in prompt
        assert "Jane Doe" in prompt

    def test_prompt_delimits_hints_as_data(self):
        prompt = _build_clustering_prompt(
            "Jane Doe", RESULTS[:3], {"company": "Stripe", "role": "VP"})
        assert "company: Stripe" in prompt
        assert "role: VP" in prompt


# ---------------------------------------------------------------------------
# C1.7c — semantic hint matcher v2 (D11/R2)
# ---------------------------------------------------------------------------

def _cand(descriptor, disambiguators=(), supporting=(), name="John Smith"):
    return {"canonical_name": name, "descriptor": descriptor,
            "disambiguators": list(disambiguators),
            "supporting_results": list(supporting)}


def _consistent(candidate, hints, results=()):
    from src.core.preflight import _hint_consistent
    return _hint_consistent(candidate, hints, list(results))


class TestMatcherV2:
    """The R2 table (gate 5), re-derived: exact word / prefix-inflection
    (min(len)>=5, common_prefix >= max(5, len(shorter)-3)) / edit-distance-1
    typo (len(T)>=5). Mechanisms do NOT compose."""

    NSSF_VP = _cand("John Smith — Vice President of Finance, NSSF (US)",
                    ["NSSF", "finance"])

    def test_typo_fianance_matches_finance(self):
        assert _consistent(self.NSSF_VP, {"role": "Fianance"})

    def test_inflection_finance_matches_financial_only_descriptor(self):
        cand = _cand("John Smith — director, financial services firm (UK)")
        assert _consistent(cand, {"role": "finance"})

    def test_filler_skip_being_a_vp(self):
        cand = _cand("John Smith — VP, NSSF (US)")
        assert _consistent(cand, {"known_for": "Being a VP"})

    def test_short_token_vp_exact_word_boundary(self):
        assert _consistent(_cand("John Smith — VP, NSSF"), {"role": "VP"})
        # word-boundary: "vp" inside "mvp" is NOT a match (the old substring
        # matcher would have matched — this falsifies v2's boundary rule)
        assert not _consistent(
            _cand("John Smith — MVP award winner, basketball"), {"role": "VP"})

    def test_all_filler_hint_skipped_never_fails(self):
        cand = _cand("John Smith — VP, NSSF (US)")
        assert _consistent(cand, {"known_for": "known for being at"})

    def test_non_composing_bound_typo_vs_inflected_only(self):
        # fianance vs a descriptor that says ONLY "financial": edit distance
        # >1 AND common prefix 2 — no match; D7 pick+note is the documented
        # floor (single-mechanism tolerance is the spec).
        cand = _cand("John Smith — analyst, financial services (UK)")
        assert not _consistent(cand, {"role": "fianance"})

    def test_garbage_hint_no_match(self):
        assert not _consistent(self.NSSF_VP, {"role": "xyzzyq"})

    def test_acronym_ai_exact_not_substring(self):
        assert _consistent(_cand("Jane Roe — AI researcher, MIT"),
                           {"known_for": "AI"})
        assert not _consistent(_cand("Jane Roe — air quality engineer"),
                               {"known_for": "AI"})

    def test_case_and_nfkc_normalization(self):
        assert _consistent(_cand("john smith — vp, nssf"),
                           {"company": "NSSF"})

    def test_and_semantics_across_hints_preserved(self):
        # one matching hint + one non-matching hint → inconsistent
        assert not _consistent(self.NSSF_VP,
                               {"company": "NSSF", "role": "surgeon"})

    def test_supporting_results_still_searched(self):
        cand = _cand("John Smith — executive", supporting=[1])
        results = [_result("https://nssf.org/x", title="NSSF leadership",
                           snippet="John Smith VP finance")]
        assert _consistent(cand, {"company": "NSSF"}, results)

    def test_multiword_hint_all_tokens_must_match(self):
        cand = _cand("John Smith — VP Engineering, Stripe (SF)")
        assert _consistent(cand, {"role": "VP Engineering"})
        assert not _consistent(cand, {"role": "VP Surgery"})


# ---------------------------------------------------------------------------
# C1.7c — hint_match: validation, gate order (R3), pick re-ordering
# ---------------------------------------------------------------------------

DOMINANT_BOTH_ENGINEER = [
    {"canonical_name": "Jane Doe",
     "descriptor": "Jane Doe — engineer, Stripe (SF)",
     "disambiguators": ["Stripe", "engineer"],
     "supporting_results": [1, 2, 3, 4, 5], "hint_match": "none"},
    {"canonical_name": "Jane Doe",
     "descriptor": "Jane Doe — engineer, Boeing (Seattle)",
     "disambiguators": ["Boeing", "engineer"],
     "supporting_results": [6], "hint_match": "strong"},
]


class TestHintMatch:
    def test_validation_allowlist_default_none(self):
        raw = [
            dict(_cand("a"), supporting_results=[1], hint_match="strong"),
            dict(_cand("b"), supporting_results=[2], hint_match="partial"),
            dict(_cand("c"), supporting_results=[3], hint_match="definitely!"),
            dict(_cand("d"), supporting_results=[4], hint_match=3),
            dict(_cand("e"), supporting_results=[5]),
        ]
        out = _validate_candidates(raw, num_results=10)
        assert [c["hint_match"] for c in out] == [
            "strong", "partial", "none", "none", "none"]

    def test_gate6_mass_dominant_auto_ignores_lowmass_strong(self):
        """R3 (the REAL property): the D2 gate evaluates on MASS order — a
        mass-dominant cluster still autos on the mass leader even when a
        lower-mass sibling carries hint_match=strong (auto neither
        suppressed into pick nor retargeted by an LLM claim)."""
        llm = FakeLLM(_candidates_json(DOMINANT_BOTH_ENGINEER))
        res = _run(discover_candidates(
            "Jane Doe", hints={"role": "engineer"},   # consistent with BOTH
            executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision == "auto"
        assert res.note == "dominant"
        assert res.candidates[0].domain_mass == 5
        assert res.candidates[0].hint_match == "none"

    def test_pick_response_reordered_by_hint_match(self):
        clusters = [
            dict(DOMINANT_BOTH_ENGINEER[0]),
            dict(DOMINANT_BOTH_ENGINEER[1], supporting_results=[6, 7]),
        ]
        llm = FakeLLM(_candidates_json(clusters))
        res = _run(discover_candidates(
            "Jane Doe", hints={"role": "engineer"},
            executor=FakeExecutor(RESULTS), llm_client=llm))
        # masses 5 vs 2 → gate says pick; display order leads with strong
        assert res.decision == "pick"
        assert res.candidates[0].hint_match == "strong"
        assert res.candidates[0].domain_mass == 2
        assert res.candidates[1].domain_mass == 5
        # serialization carries the field for the card label
        assert res.candidates[0].to_dict()["hint_match"] == "strong"

    def test_no_hints_pick_order_stays_mass(self):
        llm = FakeLLM(_candidates_json(TWO_CLUSTERS))
        res = _run(discover_candidates(
            "Jane Doe", executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision == "pick"
        assert [c.domain_mass for c in res.candidates] == sorted(
            [c.domain_mass for c in res.candidates], reverse=True)

    def test_d7_zero_consistent_pick_also_reordered(self):
        # The D7 unfiltered picker is a pick response too — a strong label
        # from the LLM may lead its display order (server filter found
        # nothing; ordering is the only thing hint_match ever drives).
        clusters = [
            dict(TWO_CLUSTERS[0], hint_match="none"),
            dict(TWO_CLUSTERS[1], hint_match="strong"),
        ]
        llm = FakeLLM(_candidates_json(clusters))
        res = _run(discover_candidates(
            "Jane Doe", hints={"company": "NASA"},
            executor=FakeExecutor(RESULTS), llm_client=llm))
        assert res.decision == "pick"
        assert "match" in (res.note or "")
        assert res.candidates[0].hint_match == "strong"

    def test_prompt_asks_for_hint_match_only_with_hints(self):
        with_hints = _build_clustering_prompt(
            "Jane Doe", RESULTS[:3], {"role": "VP"})
        without = _build_clustering_prompt("Jane Doe", RESULTS[:3], None)
        assert "hint_match" in with_hints
        assert "hint_match" not in without
