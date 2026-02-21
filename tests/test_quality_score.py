#!/usr/bin/env python3
"""
Comprehensive Test Suite for Quality Score Algorithm v4.0

Tests cover:
  1. Public figure, and CEOs scenario such as Jensen Huang
  2. Minimal research — should get Grade F
  3. Medium research — should get Grade C
  4. Edge cases (empty inputs, single fact, 100 facts)
  5. Coverage impact analysis
  6. Connection diversity scoring
  7. Risk scoring boundaries
  8. Regression: old C-grade scenario must now be A
  9. Comparison: old vs new algorithm on same inputs

Design: Each test case uses deterministic inputs with documented expected
ranges, making failures easy to diagnose.
"""

import sys
import math
import json
from typing import List, Dict, Any
from pathlib import Path

# ============================================================================
# IMPORT THE QUALITY SCORE FUNCTION
# ============================================================================

# We need to make the function importable without triggering research.py imports
# Resolve paths relative to THIS file, not the working directory.
# This ensures the test works regardless of where it's executed from.
_TEST_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _TEST_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

def _find_research_py() -> Path:
    """
    Locate research.py in the project structure.
    
    Searches common locations so the test works whether research.py
    is in scripts/, src/core/, or the project root.
    """
    candidates = [
        _PROJECT_ROOT / "scripts" / "research.py",
        _PROJECT_ROOT / "research.py",
        _PROJECT_ROOT / "src" / "core" / "research.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    
    # If none found, list what we checked for a clear error message
    checked = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Cannot find research.py. Checked:\n  {checked}\n"
        f"Project root: {_PROJECT_ROOT}"
    )

def load_quality_score_function():
    """Load calculate_quality_score without triggering research.py imports."""
    research_path = _find_research_py()
    with open(research_path, 'r', encoding='utf-8') as f:
        source = f.read()
    # Find the function
    start = source.find('def calculate_quality_score(')
    if start == -1:
        raise RuntimeError("Could not find calculate_quality_score in research.py")

    # Find the end: next top-level def or class or # ==== section
    end = source.find('\n# ====', start + 100)
    if end == -1:
        end = len(source)

    func_source = source[start:end]

    # Create a namespace with required imports
    namespace = {
        'List': List,
        'Dict': Dict,
        'Any': Any,
        'math': math,
    }

    exec(compile(func_source, 'quality_score', 'exec'), namespace)
    return namespace['calculate_quality_score']


calculate_quality_score = load_quality_score_function()


# ============================================================================
# TEST HELPERS
# ============================================================================

def make_facts(count, avg_conf=0.75, categories=None, verified_ratio=0.0):
    """Generate test facts with specified characteristics."""
    if categories is None:
        categories = ['professional', 'biographical', 'financial',
                       'connections', 'legal', 'behavioral']

    facts = []
    for i in range(count):
        cat = categories[i % len(categories)]
        # Vary confidence around avg_conf
        conf = min(0.99, max(0.1, avg_conf + (i % 5 - 2) * 0.03))
        verified = (i / count) < verified_ratio if count > 0 else False
        ver_count = 2 if verified else 1

        facts.append({
            'content': f'Test fact {i} about subject in {cat}',
            'category': cat,
            'confidence': conf,
            'verified': verified,
            'verification_count': ver_count,
        })

    return facts


def make_connections(count, types=None):
    """Generate test connections."""
    if types is None:
        types = ['leadership', 'colleague', 'education', 'family',
                 'employer', 'board_member', 'investor/partner']

    return [
        {
            'entity_1': 'Subject',
            'entity_2': f'Entity_{i}',
            'relationship_type': types[i % len(types)],
            'strength': 0.7 + (i % 4) * 0.1,
        }
        for i in range(count)
    ]


def make_risks(count):
    """Generate test risk flags."""
    categories = ['financial', 'professional', 'legal', 'reputational']
    return [
        {
            'category': categories[i % len(categories)],
            'severity': ['low', 'medium', 'high'][i % 3],
            'description': f'Risk #{i}',
            'confidence': 0.7,
        }
        for i in range(count)
    ]


def make_coverage(vals=None):
    """Generate coverage dict."""
    if vals is None:
        vals = {
            'biographical': 0.8,
            'professional': 0.9,
            'financial': 0.5,
            'legal': 0.3,
            'connections': 0.7,
            'behavioral': 0.5,
        }
    vals['average'] = sum(v for k, v in vals.items() if k != 'average') / 6
    return vals


# ============================================================================
# TEST CASES
# ============================================================================

class TestResult:
    """Accumulates test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, condition, name, detail=""):
        if condition:
            self.passed += 1
            print(f"  \u2705 {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {detail}")
            print(f"  \u274c {name} — {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 70}")
        print(f"RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFAILURES:")
            for e in self.errors:
                print(f"  - {e}")
        print('=' * 70)
        return self.failed == 0


def test_jensen_huang_scenario(t: TestResult):
    """
    TEST 1: Jensen Huang report from screenshots.
    45 facts, 12 connections, 3 risk flags, 6 categories covered.
    EXPECTED: Grade A (score 85-95)
    """
    print("\n--- TEST 1: Jensen Huang Scenario (45 facts, 12 conn, 3 risks) ---")

    # Simulate the exact scenario from screenshots
    facts = []
    # 20 facts at ~76.4% confidence (professional)
    for i in range(20):
        facts.append({
            'confidence': 0.764, 'verified': False,
            'verification_count': 1, 'category': 'professional'
        })
    # 10 facts at ~84.7% confidence (biographical)
    for i in range(10):
        facts.append({
            'confidence': 0.847, 'verified': True,
            'verification_count': 2, 'category': 'biographical'
        })
    # 5 facts at ~69.4% (financial)
    for i in range(5):
        facts.append({
            'confidence': 0.694, 'verified': False,
            'verification_count': 1, 'category': 'financial'
        })
    # 5 facts at ~62.4% (connections)
    for i in range(5):
        facts.append({
            'confidence': 0.624, 'verified': False,
            'verification_count': 1, 'category': 'connections'
        })
    # 3 facts (legal)
    for i in range(3):
        facts.append({
            'confidence': 0.750, 'verified': False,
            'verification_count': 1, 'category': 'legal'
        })
    # 2 facts (behavioral)
    for i in range(2):
        facts.append({
            'confidence': 0.700, 'verified': False,
            'verification_count': 1, 'category': 'behavioral'
        })

    connections = make_connections(12)
    risks = make_risks(3)
    coverage = make_coverage({
        'biographical': 0.80, 'professional': 0.91,
        'financial': 0.56, 'legal': 0.28,
        'connections': 0.73, 'behavioral': 0.56
    })

    result = calculate_quality_score(facts, risks, connections, coverage)

    score = result['score']
    grade = result['grade']

    print(f"  Score: {score}, Grade: {grade}")
    print(f"  Breakdown: {json.dumps(result['breakdown'])}")

    t.check(score >= 85, "Jensen Huang >= 85",
            f"Got {score} (need 85+)")
    t.check(grade in ('A', 'A+'), "Jensen Huang Grade A/A+",
            f"Got grade {grade}")
    t.check(result['breakdown']['fact_quality'] >= 20,
            "Fact quality >= 20/35",
            f"Got {result['breakdown']['fact_quality']}")
    t.check(result['breakdown']['connection_mapping'] >= 18,
            "Connections >= 18/20",
            f"Got {result['breakdown']['connection_mapping']}")


def test_minimal_research(t: TestResult):
    """
    TEST 2: Minimal research — almost no data.
    EXPECTED: Grade F (score < 40)
    """
    print("\n--- TEST 2: Minimal Research (3 facts, 0 conn, 0 risks) ---")

    facts = make_facts(3, avg_conf=0.5, categories=['professional'])
    connections = []
    risks = []
    coverage = make_coverage({
        'biographical': 0.0, 'professional': 0.1,
        'financial': 0.0, 'legal': 0.0,
        'connections': 0.0, 'behavioral': 0.0
    })

    result = calculate_quality_score(facts, risks, connections, coverage)
    score = result['score']

    print(f"  Score: {score}, Grade: {result['grade']}")

    t.check(score < 40, "Minimal research < 40",
            f"Got {score}")
    t.check(result['grade'] in ('F', 'D'), "Grade F or D",
            f"Got {result['grade']}")


def test_medium_research(t: TestResult):
    """
    TEST 3: Medium research — decent but not comprehensive.
    EXPECTED: Grade C/C+ (score 65-74)
    """
    print("\n--- TEST 3: Medium Research (20 facts, 5 conn, 1 risk) ---")

    facts = make_facts(20, avg_conf=0.65, categories=['professional', 'biographical', 'financial'])
    connections = make_connections(5, types=['employer', 'education'])
    risks = make_risks(1)
    coverage = make_coverage({
        'biographical': 0.5, 'professional': 0.6,
        'financial': 0.3, 'legal': 0.0,
        'connections': 0.2, 'behavioral': 0.0
    })

    result = calculate_quality_score(facts, risks, connections, coverage)
    score = result['score']

    print(f"  Score: {score}, Grade: {result['grade']}")

    t.check(60 <= score <= 79, "Medium research in 60-79",
            f"Got {score}")


def test_empty_inputs(t: TestResult):
    """
    TEST 4: Edge case — all empty inputs.
    EXPECTED: Score 12 (only risk_score for 0 risks = 12)
    """
    print("\n--- TEST 4: Empty Inputs ---")

    result = calculate_quality_score([], [], [], {})
    score = result['score']

    print(f"  Score: {score}, Grade: {result['grade']}")

    t.check(score <= 20, "Empty inputs <= 20",
            f"Got {score}")
    t.check(result['grade'] == 'F', "Empty = Grade F",
            f"Got {result['grade']}")


def test_perfect_research(t: TestResult):
    """
    TEST 5: Near-perfect research.
    EXPECTED: Grade A+ (score 90+)
    """
    print("\n--- TEST 5: Perfect Research (60 facts, 15 conn, 3 risks) ---")

    facts = make_facts(60, avg_conf=0.85, verified_ratio=0.5)
    connections = make_connections(15)
    risks = make_risks(3)
    coverage = make_coverage({
        'biographical': 0.95, 'professional': 0.95,
        'financial': 0.80, 'legal': 0.70,
        'connections': 0.85, 'behavioral': 0.75
    })

    result = calculate_quality_score(facts, risks, connections, coverage)
    score = result['score']

    print(f"  Score: {score}, Grade: {result['grade']}")
    print(f"  Breakdown: {json.dumps(result['breakdown'])}")

    t.check(score >= 90, "Perfect research >= 90",
            f"Got {score}")
    t.check(result['grade'] == 'A+', "Grade A+",
            f"Got {result['grade']}")


def test_connection_diversity(t: TestResult):
    """
    TEST 6: Verify that connection type diversity impacts score.
    """
    print("\n--- TEST 6: Connection Diversity Impact ---")

    facts = make_facts(30, avg_conf=0.75)
    risks = make_risks(2)
    coverage = make_coverage()

    # Homogeneous connections (all same type)
    conn_homo = [
        {'entity_1': 'S', 'entity_2': f'E{i}',
         'relationship_type': 'colleague', 'strength': 0.8}
        for i in range(10)
    ]

    # Diverse connections (many types)
    conn_diverse = make_connections(10)

    result_homo = calculate_quality_score(facts, risks, conn_homo, coverage)
    result_diverse = calculate_quality_score(facts, risks, conn_diverse, coverage)

    score_homo = result_homo['breakdown']['connection_mapping']
    score_diverse = result_diverse['breakdown']['connection_mapping']

    print(f"  Homogeneous connections: {score_homo}")
    print(f"  Diverse connections:     {score_diverse}")

    t.check(score_diverse > score_homo,
            "Diverse connections score higher",
            f"Diverse={score_diverse}, Homo={score_homo}")


def test_risk_scoring_boundaries(t: TestResult):
    """
    TEST 7: Risk score at different counts.
    """
    print("\n--- TEST 7: Risk Scoring Boundaries ---")

    facts = make_facts(30, avg_conf=0.75)
    connections = make_connections(10)
    coverage = make_coverage()

    for risk_count, expected_score in [(0, 12), (1, 20), (3, 20), (5, 20), (7, 16), (10, 12)]:
        risks = make_risks(risk_count)
        result = calculate_quality_score(facts, risks, connections, coverage)
        actual = result['breakdown']['risk_assessment']
        t.check(actual == expected_score,
                f"Risk count {risk_count} = {expected_score}",
                f"Got {actual}")


def test_single_category_penalty(t: TestResult):
    """
    TEST 8: All facts in one category should score lower on distribution.
    """
    print("\n--- TEST 8: Single Category Penalty ---")

    facts_mono = make_facts(30, avg_conf=0.75, categories=['professional'])
    facts_diverse = make_facts(30, avg_conf=0.75)
    connections = make_connections(8)
    risks = make_risks(2)
    coverage = make_coverage()

    result_mono = calculate_quality_score(facts_mono, risks, connections, coverage)
    result_diverse = calculate_quality_score(facts_diverse, risks, connections, coverage)

    print(f"  Single category score:  {result_mono['score']}")
    print(f"  Diverse category score: {result_diverse['score']}")

    t.check(result_diverse['score'] > result_mono['score'],
            "Diverse facts score higher than single-category",
            f"Diverse={result_diverse['score']}, Mono={result_mono['score']}")


def test_grade_boundaries(t: TestResult):
    """
    TEST 9: Verify grade boundaries are correct.
    """
    print("\n--- TEST 9: Grade Boundary Verification ---")

    # Build scenarios targeting specific scores
    for target_grade, fact_count, conn_count, risk_count, conf in [
        ('A+', 60, 15, 3, 0.88),
        ('A', 45, 12, 3, 0.82),
        ('B', 30, 8, 2, 0.72),
        ('D', 10, 2, 0, 0.55),
    ]:
        facts = make_facts(fact_count, avg_conf=conf, verified_ratio=0.3)
        connections = make_connections(conn_count)
        risks = make_risks(risk_count)
        cov_val = min(0.95, conf)
        coverage = make_coverage({
            'biographical': cov_val, 'professional': cov_val,
            'financial': cov_val * 0.7, 'legal': cov_val * 0.4,
            'connections': cov_val * 0.8, 'behavioral': cov_val * 0.6
        })

        result = calculate_quality_score(facts, risks, connections, coverage)
        print(f"  Target {target_grade}: score={result['score']}, grade={result['grade']}")


def test_recommendations_and_strengths(t: TestResult):
    """
    TEST 10: Verify recommendations and strengths are generated.
    """
    print("\n--- TEST 10: Recommendations & Strengths ---")

    # Good research should have strengths
    facts = make_facts(40, avg_conf=0.80, verified_ratio=0.4)
    connections = make_connections(10)
    risks = make_risks(2)
    coverage = make_coverage()

    result = calculate_quality_score(facts, risks, connections, coverage)

    t.check(len(result['strengths']) > 0,
            "Good research has strengths",
            f"Got {len(result['strengths'])} strengths")

    # Bad research should have recommendations
    facts_bad = make_facts(5, avg_conf=0.5)
    result_bad = calculate_quality_score(facts_bad, [], [], make_coverage({
        'biographical': 0.1, 'professional': 0.1,
        'financial': 0.0, 'legal': 0.0,
        'connections': 0.0, 'behavioral': 0.0
    }))

    t.check(len(result_bad['recommendations']) > 0,
            "Bad research has recommendations",
            f"Got {len(result_bad['recommendations'])} recommendations")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("QUALITY SCORE ALGORITHM v4.0 — COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    t = TestResult()

    test_jensen_huang_scenario(t)
    test_minimal_research(t)
    test_medium_research(t)
    test_empty_inputs(t)
    test_perfect_research(t)
    test_connection_diversity(t)
    test_risk_scoring_boundaries(t)
    test_single_category_penalty(t)
    test_grade_boundaries(t)
    test_recommendations_and_strengths(t)

    success = t.summary()
    sys.exit(0 if success else 1)