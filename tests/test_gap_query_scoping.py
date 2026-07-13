"""
Phase C1.7b — context-aware gap-filler template queries.

The bare `{name} + category` templates in _create_gap_query were the dominant
context-blind contamination vector (PLAN.md C1.7 root cause 1). With context
present the query becomes `"{name}" {scope} {terms}`; without context it is
byte-identical to the pre-C1.7 template. Scope-term chain (approved F1):
context["company"] → first segment of the "; "-joined context["disambiguators"]
STRING (routes.py builds it that way) → template unchanged.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from types import SimpleNamespace

from src.search.strategy import SearchStrategyEngine


class FakeRouter:
    def route_and_call(self, task_type=None, prompt="", **kwargs):
        return SimpleNamespace(content="[]", cost=0.0,
                               provider=SimpleNamespace(value="google"),
                               model_name="fake")


def _engine():
    return SearchStrategyEngine(FakeRouter())


ENTITY_CONTEXT = {
    "research_target": "John Smith — VP of Finance, NSSF (US)",
    "disambiguators": "NSSF; VP of Finance; firearms industry",
}


class TestScopeTerm:
    def test_company_hint_wins(self):
        assert SearchStrategyEngine._gap_scope_term(
            {"company": "NSSF", "disambiguators": "other; thing"}) == "NSSF"

    def test_first_disambiguator_segment_when_no_company(self):
        # F1: disambiguators is a "; "-joined STRING, not a list
        assert SearchStrategyEngine._gap_scope_term(ENTITY_CONTEXT) == "NSSF"

    def test_no_usable_term(self):
        assert SearchStrategyEngine._gap_scope_term(None) is None
        assert SearchStrategyEngine._gap_scope_term({}) is None
        assert SearchStrategyEngine._gap_scope_term(
            {"research_target": "John Smith — VP, NSSF"}) is None

    def test_empty_strings_fall_through(self):
        assert SearchStrategyEngine._gap_scope_term(
            {"company": "  ", "disambiguators": "NSSF; x"}) == "NSSF"
        assert SearchStrategyEngine._gap_scope_term(
            {"company": "", "disambiguators": "   "}) is None


class TestGapQueryScoping:
    def test_scoped_query_carries_quoted_name_and_disambiguator(self):
        q = _engine()._create_gap_query("John Smith", "financial",
                                        context=ENTITY_CONTEXT)
        assert q.text == ('"John Smith" NSSF wealth assets property '
                          'investments financial')
        assert q.category.value == "financial"

    def test_company_hint_scopes_when_present(self):
        q = _engine()._create_gap_query(
            "John Smith", "legal",
            context={"company": "Acme Corp",
                     "disambiguators": "ignored; segments"})
        assert q.text.startswith('"John Smith" Acme Corp legal')

    def test_no_context_byte_identical_template(self):
        """Gate (4): the disambiguator appears IFF context exists."""
        q = _engine()._create_gap_query("John Smith", "financial")
        assert q.text == ("John Smith wealth assets property investments "
                          "financial")
        assert '"' not in q.text

    def test_context_without_usable_term_keeps_bare_template(self):
        q = _engine()._create_gap_query(
            "John Smith", "biographical",
            context={"research_target": "John Smith — VP, NSSF"})
        assert q.text == "John Smith personal background family early life"

    def test_unknown_category_still_none(self):
        assert _engine()._create_gap_query("John Smith", "astrology",
                                           context=ENTITY_CONTEXT) is None

    def test_all_known_categories_scope(self):
        for cat in ("biographical", "professional", "financial", "legal",
                    "connections", "behavioral"):
            q = _engine()._create_gap_query("John Smith", cat,
                                            context=ENTITY_CONTEXT)
            assert q.text.startswith('"John Smith" NSSF '), cat


class TestEndToEndThreading:
    def test_refine_threads_context_into_gap_queries(self):
        """refine_based_on_findings → _generate_gap_filling_queries →
        _create_gap_query is a 3-function chain (reviewer correction) —
        prove the context arrives at the far end."""
        engine = _engine()
        findings = [{"content": "John Smith is a VP", "category":
                     "professional", "entities": []}]  # <5 → no AI call
        queries = engine.refine_based_on_findings(
            "John Smith", findings, max_follow_ups=15,
            context=ENTITY_CONTEXT)
        gap = [q for q in queries if q.purpose.startswith("Fill ")]
        assert gap, "no gap-filler queries generated"
        assert all(q.text.startswith('"John Smith" NSSF ') for q in gap)

    def test_refine_without_context_keeps_bare_gap_queries(self):
        engine = _engine()
        findings = [{"content": "John Smith is a VP", "category":
                     "professional", "entities": []}]
        queries = engine.refine_based_on_findings(
            "John Smith", findings, max_follow_ups=15)
        gap = [q for q in queries if q.purpose.startswith("Fill ")]
        assert gap
        assert all(q.text.startswith("John Smith ") for q in gap)
        assert all('"' not in q.text for q in gap)
