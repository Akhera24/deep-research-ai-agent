"""
Thin-subject honesty framing (PLAN.md C3-minimal, human-directed 2026-07-13).

The score measures REPORT DEPTH, not the subject — but a low grade on a
low-footprint person reads as "this person is risky/failing". Render-side
framing only (scoring math untouched — that is future_updates P1):
- an always-present caption under the grade naming what the score measures;
- a "limited public footprint" note when few facts were found, carrying the
  searches/iterations counts (the agent exhausted its searches — scarcity,
  not failure) and the sidelined count when same-name people were set aside;
- the low-tier quality WORDS describe the report, never the person
  ("Failing" → "Minimal Public Data").
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.reporting.html_report import calculate_quality_score, render_html_report


def _fact(i):
    return {
        "content": f"John Smith fact number {i} about his NSSF role",
        "category": "professional", "confidence": 0.9,
        "source_urls": [f"https://source{i}.example/page"],
        "source_reliabilities": {}, "evidence": [f"quote {i}"],
        "verified": True, "verification_count": 2,
    }


def _render(n_facts, sidelined_count=0, sidelined=None):
    result = {
        "target_name": "John Smith",
        "facts": [_fact(i) for i in range(n_facts)],
        "risk_flags": [], "connections": [],
        "metadata": {"coverage": {"average": 0.3}, "iterations": 4,
                     "max_iterations": 10, "queries_executed": 37,
                     "duration_seconds": 60.0,
                     "sidelined_count": sidelined_count},
    }
    if sidelined is not None:
        result["sidelined_facts"] = sidelined
    return render_html_report(result, "John Smith", 60.0)


class TestScoreCaption:
    def test_caption_always_present(self):
        for n in (3, 40):
            html = _render(n)
            assert "not the subject" in html
            assert "research depth" in html.lower()


class TestThinSubjectNote:
    def test_note_present_below_threshold_with_counts(self):
        html = _render(5)
        assert "Limited public footprint" in html
        assert "37 searches" in html
        assert "4 iterations" in html
        assert "5 verified facts" in html
        assert "not elevated risk" in html

    def test_note_absent_at_threshold_and_above(self):
        for n in (15, 40):
            assert "Limited public footprint" not in _render(n)

    def test_note_mentions_sidelined_when_present(self):
        html = _render(5, sidelined_count=35)
        assert "35 facts about other people sharing the name" in html

    def test_note_no_sideline_clause_when_zero(self):
        html = _render(5, sidelined_count=0)
        assert "other people sharing the name" not in html

    def test_zero_fact_report_gets_note(self):
        html = _render(0)
        assert "Limited public footprint" in html
        assert "0 verified facts" in html

    def test_coexists_with_early_note(self):
        result = {
            "target_name": "X",
            "facts": [_fact(1)],
            "risk_flags": [], "connections": [],
            "metadata": {"coverage": {}, "iterations": 3, "max_iterations": 10,
                         "queries_executed": 12, "duration_seconds": 10.0,
                         "finished_early": True, "sidelined_count": 0},
        }
        html = render_html_report(result, "X", 10.0)
        assert "Generated early at the user" in html
        assert "Limited public footprint" in html


class TestDepthNeutralQualityWords:
    """Low-tier words must describe the REPORT, never the person."""

    def test_f_tier_word(self):
        out = calculate_quality_score(facts=[], risk_flags=[],
                                      connections=[], coverage={})
        assert out["grade"] == "F"
        assert out["quality"] == "Minimal Public Data"
        assert "Failing" not in str(out)

    def test_no_person_implicating_words_in_any_tier(self):
        # sweep the scorer's own label table via the module source
        import inspect
        import src.reporting.html_report as hr
        src = inspect.getsource(hr.calculate_quality_score)
        for banned in ('"Failing"', '"Poor"', '"Needs Improvement"'):
            assert banned not in src
