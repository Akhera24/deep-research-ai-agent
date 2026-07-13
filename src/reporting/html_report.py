"""
Shared report path: quality scoring + interactive HTML report rendering.

Extracted from scripts/research.py (PLAN.md Step 10 / PHASE3_DESIGN §3) so
the CLI and the job API share ONE report implementation.

Security: render_html_report() deep-escapes every string in the research
result (and the query) via markupsafe BEFORE interpolation — facts are
scraped from the open web and were previously interpolated raw (stored-XSS,
edge case #10).
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from urllib.parse import quote as _pct_quote, urlsplit as _urlsplit

from markupsafe import escape as _html_escape

from config.logging_config import get_logger

logger = get_logger(__name__)


def _escape_deep(value):
    """Recursively HTML-escape every string in a result structure.

    Returns escaped copies; never mutates the caller's data (the CLI also
    prints the same result object to the console).
    """
    if isinstance(value, str):
        return str(_html_escape(value))
    if isinstance(value, dict):
        return {k: _escape_deep(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_escape_deep(v) for v in value]
    return value


# ============================================================================
# CITATION CHIPS (Phase B1/B3/B4)
# ============================================================================
# This helper owns ALL citation encoding, fed from RAW (pre-_escape_deep)
# values — building a href or #:~:text= fragment from the escaped copy
# double-encodes and breaks the link, and HTML-escaping is NOT scheme
# validation (a javascript: URL survives it as a live XSS href). See
# REVIEW-LEARNINGS "Escaping chokepoints vs. features that need the raw value".

_CITATION_SCHEMES = ('http', 'https')

# A #:~:text= fragment built from a very long quote makes an unwieldy URL;
# past this length degrade to a plain link (browsers ignore missing
# fragments, so shorter-but-wrong would be strictly worse than none).
_MAX_FRAGMENT_QUOTE_CHARS = 300

# DOM cap per chip strip. Post-B0 facts cite 1-3 sources; only legacy
# batch-stamped data (one fact "citing" the whole result batch — the exact
# non-provenance B0 fixes) can exceed this, and rendering hundreds of
# anchors per fact triples the page weight for links that aren't real
# per-fact attribution anyway.
_MAX_CHIPS_PER_ITEM = 12


def _source_tier(reliability):
    """Map a source_reliability score to a (css_class, label) display tier.

    Deliberately a domain-based SOURCE-TIER heuristic — never rendered as a
    "% true" / fact-accuracy claim (PLAN.md B4 trust/liability rule).
    Missing/invalid reliability falls back to the default tier.
    """
    try:
        r = float(reliability)
    except (TypeError, ValueError):
        return ('standard', 'Standard source')
    if r >= 0.8:
        return ('high', 'High-credibility source')
    if r >= 0.6:
        return ('established', 'Established source')
    return ('standard', 'Standard source')


def _citation_href(raw_url, raw_quote=''):
    """Build an attribute-safe citation href from RAW url + quote.

    - Allowlists http/https (drop javascript:/data:/anything else).
    - Appends a #:~:text= fragment for the verbatim quote, percent-encoding
      ONLY the fragment text (quote(text, safe='')); the whole URL is never
      percent-encoded (that would break :// ? &).
    - HTML-escapes the final URL for the href attribute context.

    Returns None when the URL is not linkable (caller drops the chip).
    """
    if not isinstance(raw_url, str):
        return None
    url = raw_url.strip()
    try:
        parts = _urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in _CITATION_SCHEMES or not parts.hostname:
        return None

    quote_text = raw_quote.strip() if isinstance(raw_quote, str) else ''
    fragment = None
    if quote_text:
        # '-' is meaningful in text-fragment syntax (prefix-/suffix-
        # separators), so encode it beyond what quote() does.
        def _enc(t):
            return _pct_quote(t, safe='').replace('-', '%2D')

        if len(quote_text) <= _MAX_FRAGMENT_QUOTE_CHARS:
            fragment = _enc(quote_text)
        else:
            # Long quote: textStart,textEnd range instead of degrading —
            # the browser highlights everything between the two matches.
            words = quote_text.split()
            if len(words) >= 16:
                fragment = _enc(' '.join(words[:8])) + ',' + _enc(' '.join(words[-8:]))
    if fragment:
        url += (':~:text=' if '#' in url else '#:~:text=') + fragment

    return str(_html_escape(url))


def _citation_chip(raw_url, raw_quote='', reliability=None):
    """One citation chip <a> from RAW values; '' when not linkable.

    The native title tooltip previews what the source says (hover = see
    the snippet, click = land on it highlighted) — deliberately not a
    custom hover card: title needs no JS, adds no DOM injection surface,
    and degrades silently on touch.
    """
    href = _citation_href(raw_url, raw_quote)
    if not href:
        return ''
    domain = _urlsplit(raw_url.strip()).hostname or ''
    if domain.startswith('www.'):
        domain = domain[4:]
    tier_class, tier_label = _source_tier(reliability)
    domain_esc = str(_html_escape(domain))
    title = f'{tier_label} — source-tier heuristic (by domain), not a fact-accuracy score'
    quote_text = raw_quote.strip() if isinstance(raw_quote, str) else ''
    if quote_text:
        snippet = quote_text[:140] + ('…' if len(quote_text) > 140 else '')
        title = f'“{snippet}”\n{title}'
    title_esc = str(_html_escape(title))
    return (
        f'<a class="cite-chip tier-{tier_class}" href="{href}" target="_blank" '
        f'rel="noopener noreferrer nofollow" title="{title_esc}">'
        f'<span class="cite-tier-dot"></span>{domain_esc}</a>'
    )


def _render_citation_chips(citations, visible=3):
    """Chip strip for one fact/risk/trend item.

    citations: list of {'url', 'quote', 'reliability', 'anchored'} dicts
    holding RAW (pre-escape) values. Chips beyond `visible` are hidden
    behind a class-based "+N more sources" toggle (OQ-B3 pattern).

    Order: best source first — page-verified anchor (click provably lands
    on the highlighted sentence), then source tier; stable within ties so
    the LLM's attribution order still breaks them. Sorting happens BEFORE
    the DOM cap so the cap keeps the best chips, not the first-listed.
    """
    ranked = sorted(
        citations or [],
        key=lambda c: (
            not c.get('anchored'),
            -(c['reliability'] if isinstance(c.get('reliability'), (int, float))
              else 0.5),
        ),
    )
    chips = []
    for c in ranked:
        chip = _citation_chip(
            c.get('url'), c.get('quote', ''), c.get('reliability')
        )
        if chip:
            chips.append(chip)
        if len(chips) >= _MAX_CHIPS_PER_ITEM:
            break
    if not chips:
        return ''

    parts = []
    for i, chip in enumerate(chips):
        if i >= visible:
            chip = chip.replace('class="cite-chip', 'class="cite-chip cite-extra', 1)
        parts.append(chip)

    toggle = ''
    hidden_count = len(chips) - visible
    if hidden_count > 0:
        toggle = (
            f'<button type="button" class="cite-more-btn" '
            f'data-more="+{hidden_count} more sources" data-less="Show fewer" '
            f'onclick="toggleCiteChips(this)">+{hidden_count} more sources</button>'
        )
    return f'<div class="citation-chips">{"".join(parts)}{toggle}</div>'


def _first_evidence_quote(evidence):
    """First non-empty verbatim quote from a fact's raw evidence field."""
    if isinstance(evidence, str):
        return evidence.strip()
    if isinstance(evidence, list):
        for ev in evidence:
            if isinstance(ev, str) and ev.strip():
                return ev.strip()
    return ''


def _render_quote_source_links(citations, limit=3):
    """Small trailing source links for ONE evidence-quote row.

    citations hold RAW values; hrefs go through the same allowlist +
    anchored-fragment helper as chips, so clicking the link opens the
    source with this quote's sentence highlighted.
    """
    links = []
    for c in citations or []:
        if len(links) >= limit:
            break
        href = _citation_href(c.get('url'), c.get('quote', ''))
        if not href:
            continue
        domain = _urlsplit(c['url'].strip()).hostname or ''
        if domain.startswith('www.'):
            domain = domain[4:]
        links.append(
            f'<a class="ev-source" href="{href}" target="_blank" '
            f'rel="noopener noreferrer nofollow">{_html_escape(domain)} ↗</a>'
        )
    return ' '.join(links)


def _raw_fact_citations(raw_facts):
    """Per-fact RAW citation triples, captured BEFORE _escape_deep (M2/N1).

    Returns a list parallel to result['facts']: element i is the citation
    list for fact i — [{'url', 'quote', 'reliability'}, ...] with the
    fact's own first verbatim evidence quote attached to each of its URLs.
    """
    per_fact = []
    for rf in raw_facts:
        if not isinstance(rf, dict):
            per_fact.append([])
            continue
        urls = rf.get('source_urls', [])
        if not isinstance(urls, list):
            urls = []
        rels = rf.get('source_reliabilities', {})
        if not isinstance(rels, dict):
            rels = {}
        anchors = rf.get('anchor_texts', {})
        if not isinstance(anchors, dict):
            anchors = {}
        quote = _first_evidence_quote(rf.get('evidence', []))
        citations = []
        seen = set()
        for url in urls:
            if not isinstance(url, str) or url in seen:
                continue
            seen.add(url)
            # B3.1: prefer the page-VERIFIED anchor for this URL; the
            # fact-level quote is the legacy fallback (matches at most one
            # of a multi-source fact's pages)
            anchor = anchors.get(url)
            has_anchor = isinstance(anchor, str) and bool(anchor)
            citations.append({
                'url': url,
                'quote': anchor if has_anchor else quote,
                'reliability': rels.get(url),
                'anchored': has_anchor,
            })
        per_fact.append(citations)
    return per_fact


def _union_citations(citation_lists):
    """Dedupe-by-URL union across grouped facts (C3), preserving order and
    each URL's own quote/tier (first occurrence wins)."""
    union = []
    seen = set()
    for citations in citation_lists:
        for c in citations:
            if c['url'] in seen:
                continue
            seen.add(c['url'])
            union.append(c)
    return union


# ============================================================================
# QUALITY SCORE CALCULATION
# ============================================================================

def calculate_quality_score(
    facts: List[Dict],
    risk_flags: List[Dict],
    connections: List[Dict],
    coverage: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculate comprehensive research quality score with detailed breakdown.

    Scoring Components (100 points total):

    1. Fact Discovery & Quality (35 points):
       - Quantity:        How many facts were found            (12 pts)
       - Avg Confidence:  Weighted-mean confidence score       (10 pts)
       - Corroboration:   % of facts seen in multiple sources   (8 pts)
       - Uniqueness:      Diversity of information              (5 pts)

    2. Research Coverage (25 points):
       - Category Breadth:  # of categories with meaningful data (10 pts)
       - Category Depth:    Avg depth of covered categories      (10 pts)
       - Distribution:      Evenness of facts across categories   (5 pts)

    3. Risk Assessment (20 points):
       - Thoroughness:    Were risks identified and analyzed?
       - Optimal range:   1-5 risks = thorough investigation

    4. Connection Mapping (20 points):
       - Network Size:      Number of unique connections found
       - Network Diversity: Variety of relationship types

    Args:
        facts: List of fact dictionaries
        risk_flags: List of identified risk flags
        connections: List of entity connections
        coverage: Dict mapping category -> float (0.0-1.0)

    Returns:
        Dict with score, grade, components, breakdown, recommendations,
        strengths, and metadata.
    """
    import math

    # ---- Initialization for main variables ----
    facts_count = len(facts) if facts else 0
    risk_flags_count = len(risk_flags) if risk_flags else 0
    connections_count = len(connections) if connections else 0

    components = {}
    total_score = 0.0

    # ================================================================
    # COMPONENT 1: FACT DISCOVERY & QUALITY
    # ================================================================

    # 1a. Quantity - Generous scaling, 30+ = near-full marks
    if facts_count >= 40:
        fact_quantity_score = 12.0
    elif facts_count >= 25:
        fact_quantity_score = 9.0 + (facts_count - 25) * (3.0 / 15.0)
    elif facts_count >= 10:
        fact_quantity_score = 4.0 + (facts_count - 10) * (5.0 / 15.0)
    elif facts_count >= 1:
        fact_quantity_score = 0.5 + (facts_count - 1) * (3.5 / 9.0)
    else:
        fact_quantity_score = 0.0

    components['fact_quantity'] = {
        'score': round(fact_quantity_score, 2),
        'max': 12,
        'value': facts_count,
        'target': 40
    }

    # 1b. Average Confidence - Weighted mean instead of binary threshold
    if facts_count > 0:
        avg_confidence = sum(f.get('confidence', 0.5) for f in facts) / facts_count
        # Rescale: 0.5 avg = 0 pts, 0.9+ = 10 pts
        confidence_score = min(10.0, max(0.0, (avg_confidence - 0.5) / 0.4 * 10.0))
    else:
        avg_confidence = 0.0
        confidence_score = 0.0

    high_confidence_facts = len(
        [f for f in facts if f.get('confidence', 0) >= 0.75]
    ) if facts else 0

    components['fact_confidence'] = {
        'score': round(confidence_score, 2),
        'max': 10,
        'avg_confidence': round(avg_confidence, 3),
        'high_confidence_count': high_confidence_facts,
        'total': facts_count
    }

    # 1c. Corroboration cross-referencing sources
    #
    # Design Decision: Graduated scale instead of linear
    # ──────────────────────────────────────────────────
    # With 110 facts, achieving 53% corroboration means 58 facts were
    # independently cross-verified — that's exceptional. Linear scaling
    # (ratio × 8.0) gave only 4.24/8 for 53%, which unfairly penalized
    # excellent research. The graduated scale rewards quality thresholds:
    #
    #   50%+ corroboration = 8.0/8  (exceptional cross-verification)
    #   40%+ corroboration = 7.0-8.0/8  (excellent)
    #   30%+ corroboration = 5.5-7.0/8  (good)
    #   20%+ corroboration = 4.0-5.5/8  (adequate)
    #   10%+ corroboration = 2.0-4.0/8  (minimal)
    #   <10% corroboration = 0-2.0/8  (insufficient)
    
    if facts_count > 0:
        corroborated_facts = len([
            f for f in facts
            if f.get('verified', False) or f.get('verification_count', 1) > 1
        ])
        corroboration_ratio = corroborated_facts / facts_count
        
        # Graduated scale — diminishing penalty as ratio increases
        if corroboration_ratio >= 0.50:
            corroboration_score = 8.0
        elif corroboration_ratio >= 0.40:
            corroboration_score = 7.0 + (corroboration_ratio - 0.40) * (1.0 / 0.10)
        elif corroboration_ratio >= 0.30:
            corroboration_score = 5.5 + (corroboration_ratio - 0.30) * (1.5 / 0.10)
        elif corroboration_ratio >= 0.20:
            corroboration_score = 4.0 + (corroboration_ratio - 0.20) * (1.5 / 0.10)
        elif corroboration_ratio >= 0.10:
            corroboration_score = 2.0 + (corroboration_ratio - 0.10) * (2.0 / 0.10)
        else:
            corroboration_score = corroboration_ratio * 20.0  # Linear below 10%
    else:
        corroborated_facts = 0
        corroboration_ratio = 0.0
        corroboration_score = 0.0

    components['fact_corroboration'] = {
        'score': round(corroboration_score, 2),
        'max': 8,
        'corroborated': corroborated_facts,
        'total': facts_count,
        'ratio': round(corroboration_ratio, 3)
    }

    # 1d. Uniqueness - Category diversity within facts
    if facts_count > 0:
        fact_categories = set(f.get('category', 'unknown') for f in facts)
        unique_categories_in_facts = len(fact_categories - {'unknown', ''})
        uniqueness_score = min(5.0, unique_categories_in_facts * 1.0)
    else:
        unique_categories_in_facts = 0
        uniqueness_score = 0.0

    components['fact_uniqueness'] = {
        'score': round(uniqueness_score, 2),
        'max': 5,
        'unique_categories': unique_categories_in_facts
    }

    fact_quality_total = (
        fact_quantity_score + confidence_score +
        corroboration_score + uniqueness_score
    )
    total_score += fact_quality_total

    # ================================================================
    # COMPONENT 2: RESEARCH COVERAGE
    # ================================================================

    # 2a. Category Breadth 
    categories_with_data = 0
    category_depths = []
    total_categories = 6

    for key, value in (coverage or {}).items():
        if key != 'average' and isinstance(value, (int, float)):
            if value > 0.1:
                categories_with_data += 1
                category_depths.append(value)

    if categories_with_data >= 6:
        breadth_score = 10.0
    elif categories_with_data >= 5:
        breadth_score = 9.0
    elif categories_with_data >= 4:
        breadth_score = 7.5
    elif categories_with_data >= 3:
        breadth_score = 5.5
    elif categories_with_data >= 2:
        breadth_score = 3.5
    elif categories_with_data >= 1:
        breadth_score = 1.5
    else:
        breadth_score = 0.0

    components['category_breadth'] = {
        'score': round(breadth_score, 2),
        'max': 10,
        'categories_covered': categories_with_data,
        'total_categories': total_categories
    }

    # 2b. Category Depth - Square-root boost to counteract
    #     aggressive diminishing returns in coverage tracker
    if category_depths:
        avg_depth = sum(category_depths) / len(category_depths)
        boosted_depth = min(1.0, avg_depth ** 0.75)
        depth_score = boosted_depth * 10.0
    else:
        avg_depth = 0.0
        boosted_depth = 0.0
        depth_score = 0.0

    components['category_depth'] = {
        'score': round(depth_score, 2),
        'max': 10,
        'raw_avg_depth': round(avg_depth, 3),
        'boosted_depth': round(boosted_depth, 3)
    }

    # 2c. Distribution Evenness - Entropy-based
    if facts_count > 0 and unique_categories_in_facts > 1:
        cat_counts = {}
        for f in facts:
            cat = f.get('category', 'unknown')
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        cat_counts.pop('unknown', None)
        cat_counts.pop('', None)

        if cat_counts:
            total_categorized = sum(cat_counts.values())
            entropy = 0.0
            for count in cat_counts.values():
                if count > 0:
                    p = count / total_categorized
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(len(cat_counts)) if len(cat_counts) > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            distribution_score = normalized_entropy * 5.0
        else:
            normalized_entropy = 0.0
            distribution_score = 0.0
    else:
        normalized_entropy = 0.0
        distribution_score = 0.0

    components['distribution_evenness'] = {
        'score': round(distribution_score, 2),
        'max': 5,
        'entropy': round(normalized_entropy, 3) if facts_count > 0 else 0.0
    }

    coverage_total = breadth_score + depth_score + distribution_score
    total_score += coverage_total

    # ================================================================
    # COMPONENT 3: RISK ASSESSMENT 
    # ================================================================
    #
    # Design Decision: Why more risks = higher score
    # ────────────────────────────────────────────────
    # The previous formula penalized finding 6+ risks (16/20 for 7 risks).
    # This is incorrect — a thorough investigation uncovering 7 genuine risks
    # with mixed severities (3 HIGH, 2 MEDIUM, 2 LOW) demonstrates BETTER
    # due diligence than finding only 3. The new scale:
    #
    #   0 risks    → 10/20  (likely missed something — almost everyone has risk)
    #   1-3 risks  → 16/20  (basic investigation found some issues)
    #   4-10 risks → 20/20  (thorough, comprehensive due diligence)
    #   11+ risks  → 18/20  (comprehensive but may include marginal flags)

    if 4 <= risk_flags_count <= 10:
        risk_score = 20.0   # Thorough: found substantial, varied risks
    elif 1 <= risk_flags_count <= 3:
        risk_score = 16.0   # Basic: found some risks, may have missed others
    elif risk_flags_count == 0:
        risk_score = 10.0   # Suspicious: almost everyone has SOME risk
    else:  # 11+
        risk_score = 18.0   # Comprehensive but may include marginal flags

    components['risk_assessment'] = {
        'score': round(risk_score, 2),
        'max': 20,
        'risks_found': risk_flags_count
    }
    total_score += risk_score

    # ================================================================
    # COMPONENT 4: CONNECTION MAPPING
    # ================================================================

    # 4a. Network size (14 pts)
    if connections_count >= 12:
        network_size_score = 14.0
    elif connections_count >= 8:
        network_size_score = 10.0 + (connections_count - 8) * 1.0
    elif connections_count >= 5:
        network_size_score = 6.0 + (connections_count - 5) * (4.0 / 3.0)
    elif connections_count >= 1:
        network_size_score = connections_count * 1.2
    else:
        network_size_score = 0.0

    # 4b. Relationship diversity
    if connections:
        rel_types = set(
            c.get('relationship_type', 'unknown')
            for c in connections
            if c.get('relationship_type') not in (None, '', 'unknown')
        )
        diversity_count = len(rel_types)
        network_diversity_score = min(6.0, diversity_count * 1.5)
    else:
        diversity_count = 0
        network_diversity_score = 0.0

    connections_score = min(20.0, network_size_score + network_diversity_score)

    components['connection_mapping'] = {
        'score': round(connections_score, 2),
        'max': 20,
        'connections_found': connections_count,
        'network_size_score': round(network_size_score, 2),
        'relationship_types': diversity_count,
        'diversity_score': round(network_diversity_score, 2)
    }
    total_score += connections_score

    # ================================================================
    # FINAL SCORE for quality of the report
    # ================================================================
    final_score = min(100.0, max(0.0, total_score))

# ── Standard College GPA Grading Scale ──────────────────────────
    # A+ = 97-100, A = 93-96, A- = 90-92
    # B+ = 87-89,  B = 83-86, B- = 80-82
    # C+ = 77-79,  C = 73-76, C- = 70-72
    # D+ = 67-69,  D = 65-66
    # F  = Below 65
    if final_score >= 97:
        grade, quality, indicator = "A+", "Outstanding", "\U0001f3c6"
    elif final_score >= 93:
        grade, quality, indicator = "A", "Excellent", "\U0001f3c6"
    elif final_score >= 90:
        grade, quality, indicator = "A-", "Very Good", "\U0001f3c6"
    elif final_score >= 87:
        grade, quality, indicator = "B+", "Good", "\u2705"
    elif final_score >= 83:
        grade, quality, indicator = "B", "Above Average", "\u2705"
    elif final_score >= 80:
        grade, quality, indicator = "B-", "Satisfactory", "\u2705"
    elif final_score >= 77:
        grade, quality, indicator = "C+", "Fair", "\U0001f44d"
    elif final_score >= 73:
        grade, quality, indicator = "C", "Average", "\U0001f44d"
    elif final_score >= 70:
        grade, quality, indicator = "C-", "Below Average", "\U0001f44d"
    elif final_score >= 67:
        # Low-tier words describe the REPORT's depth, never the person \u2014
        # "Failing/Poor" next to a name reads as a judgment of the subject
        # (human-directed 2026-07-13; letter grades + math untouched, P1).
        grade, quality, indicator = "D+", "Shallow Coverage", "\u26a0\ufe0f"
    elif final_score >= 65:
        grade, quality, indicator = "D", "Thin Findings", "\u26a0\ufe0f"
    else:
        grade, quality, indicator = "F", "Minimal Public Data", "\u274c"

        
    # ---- RECOMMENDATIONS & STRENGTHS ----
    recommendations = []
    strengths = []

    if facts_count < 20:
        recommendations.append(
            f"Increase search iterations (only {facts_count} facts, target 50+)"
        )
    elif facts_count >= 50:
        strengths.append(f"Excellent fact discovery ({facts_count} facts)")
    elif facts_count >= 30:
        strengths.append(f"Good fact discovery ({facts_count} facts)")

    if avg_confidence >= 0.8:
        strengths.append(
            f"High-quality sources (avg confidence {avg_confidence:.0%})"
        )
    elif avg_confidence >= 0.7:
        strengths.append(
            f"Good source quality (avg confidence {avg_confidence:.0%})"
        )
    elif avg_confidence < 0.55:
        recommendations.append(
            f"Improve source quality (avg confidence {avg_confidence:.0%})"
        )

    if categories_with_data < 3:
        missing_cats = [
            k for k, v in (coverage or {}).items()
            if k != 'average' and isinstance(v, (int, float)) and v < 0.1
        ]
        recommendations.append(
            f"Add searches for: {', '.join(missing_cats[:3])}"
        )
    elif categories_with_data >= 5:
        strengths.append(
            f"Comprehensive coverage ({categories_with_data}/{total_categories} categories)"
        )

    if connections_count < 5:
        recommendations.append(
            f"Increase connections (currently {connections_count}, target 10+)"
        )
    elif connections_count >= 12:
        strengths.append(
            f"Rich connection mapping ({connections_count} connections, "
            f"{diversity_count} types)"
        )
    elif connections_count >= 8:
        strengths.append(
            f"Good connection mapping ({connections_count} connections, "
            f"{diversity_count} types)"
        )

    if risk_flags_count >= 4:
        strengths.append(
            f"Comprehensive risk assessment ({risk_flags_count} risks identified across multiple categories)"
        )
    elif 1 <= risk_flags_count <= 3:
        strengths.append(
            f"Thorough risk assessment ({risk_flags_count} risks identified)"
        )
    elif risk_flags_count == 0:
        recommendations.append(
            "Consider deeper investigation for potential risks"
        )

    if corroboration_ratio >= 0.3:
        strengths.append(
            f"Strong fact corroboration ({corroboration_ratio:.0%} cross-verified)"
        )
    elif corroboration_ratio >= 0.15:
        strengths.append(
            f"Moderate fact corroboration ({corroboration_ratio:.0%} cross-verified)"
        )

    return {
        'score': round(final_score, 1),
        'grade': grade,
        'quality': quality,
        'indicator': indicator,
        'components': components,
        'breakdown': {
            'fact_quality': round(fact_quality_total, 1),
            'coverage': round(coverage_total, 1),
            'risk_assessment': round(risk_score, 1),
            'connection_mapping': round(connections_score, 1)
        },
        'recommendations': recommendations,
        'strengths': strengths,
        'metadata': {
            'facts_count': facts_count,
            'avg_confidence': round(avg_confidence, 3),
            'high_confidence_facts': high_confidence_facts,
            'corroborated_facts': corroborated_facts,
            'categories_covered': categories_with_data,
            'risk_flags_count': risk_flags_count,
            'connections_count': connections_count,
            'relationship_types': diversity_count,
            'distribution_entropy': round(normalized_entropy, 3) if facts_count > 0 else 0,
            'avg_depth_of_covered': round(avg_depth, 3) if category_depths else 0
        }
    }


# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

def render_html_report(
    result: Dict[str, Any],
    query: str,
    duration: float
) -> str:
    """
    Generate a professional, interactive HTML due diligence report.
    
    Design Decisions:
    - Facts paginated at 25 per page to prevent scroll fatigue and the 
      CSS max-height overflow bug that previously clipped facts beyond ~38
    - Facts grouped by category with filter tabs for quick navigation
    - Executive summary at top with score breakdown visualization
    - Collapsible sections open by default (all content visible on load)
    - Risk flags color-coded by severity with prominent display
    - Print-friendly CSS media query for PDF export
    - Responsive design for mobile/tablet LinkedIn viewing
    
    Args:
        result: Research results dictionary from orchestrator
        query: Target entity name
        duration: Research duration in seconds
        output_path: Optional output file path
        
    Returns:
        Absolute path to generated HTML file
    """
    
    # ── Raw citation seam (Phase B — capture BEFORE the escape chokepoint) ──
    # Citation hrefs / #:~:text= fragments must be built from RAW values;
    # every downstream union (consolidation C3, risk/trend resolution) reads
    # from THIS list, never from the escaped facts (M2). Parallel to
    # result['facts'] by index — the escaped copy preserves list order.
    raw_citations = _raw_fact_citations(result.get('facts', []) or [])
    # Each fact's own raw quote, parallel to raw_citations — pairs every
    # evidence row with the sources of the group member it came from
    raw_quotes = [
        _first_evidence_quote(rf.get('evidence', [])) if isinstance(rf, dict) else ''
        for rf in (result.get('facts', []) or [])
    ]
    # C1.7a (R6): the sideline section captures its OWN raw seam — the seam
    # is per-list, not inherited; chips built from the escaped copy would
    # double-encode every href (&→&amp;→%26amp%3B).
    raw_sidelined_citations = _raw_fact_citations(
        result.get('sidelined_facts', []) or [])

    # ── XSS guard (PHASE3_DESIGN §3/§11.R6, edge case #10) ──────────────────
    # Everything below interpolates scraped-web text into HTML f-strings.
    # Escape ALL strings at this single chokepoint, on a deep copy, before
    # any interpolation. The research result contains no legitimate markup,
    # so full escaping (markupsafe) is strictly safer than an allowlist
    # sanitizer here; bleach remains available if markup is ever permitted.
    result = _escape_deep(result)
    query = str(_html_escape(query))

    # Extract data
    facts = result.get('facts', [])
    connections = result.get('connections', [])
    risks = result.get('risk_flags', [])
    coverage = result.get('metadata', {}).get('coverage', {})
    metadata = result.get('metadata', {})
    
    # Calculate quality score
    score_result = calculate_quality_score(
        facts=facts,
        risk_flags=risks,
        connections=connections,
        coverage=coverage
    )
    
    score = score_result['score']
    grade = score_result['grade']
    quality = score_result['quality']
    breakdown = score_result.get('breakdown', {})
    strengths = score_result.get('strengths', [])
    recommendations = score_result.get('recommendations', [])
    
    # ── Group facts by category for tabbed display ──
    facts_by_category = {}
    for fact in facts:
        cat = fact.get('category', 'unknown')
        if cat not in facts_by_category:
            facts_by_category[cat] = []
        facts_by_category[cat].append(fact)
    
    # Category display order and labels
    category_order = ['biographical', 'professional', 'financial', 'legal', 
                      'behavioral', 'connections', 'unknown']
    category_labels = {
        'biographical': '👤 Biographical',
        'professional': '💼 Professional', 
        'financial': '💰 Financial',
        'legal': '⚖️ Legal',
        'behavioral': '🗣️ Behavioral',
        'connections': '🔗 Connections',
        'unknown': '📋 Other'
    }
    category_colors = {
        'biographical': '#3b82f6',
        'professional': '#8b5cf6', 
        'financial': '#10b981',
        'legal': '#ef4444',
        'behavioral': '#f59e0b',
        'connections': '#06b6d4',
        'unknown': '#6b7280'
    }
    
    # ── Build risk-to-fact mapping for risk badges ──
    # Reverse index: fact_index → [risk_info] so we can tag facts that support risk flags
    risk_fact_map = {}
    for risk in risks:
        evidence_refs = risk.get('evidence', [])
        severity = risk.get('severity', 'low')
        # Full description — truncating here cut every risk row/tooltip to
        # 80 chars mid-sentence ("…Huang is variously des")
        risk_desc = risk.get('description', '')
        for ref in evidence_refs:
            try:
                idx = int(''.join(c for c in str(ref) if c.isdigit()))
                if idx not in risk_fact_map:
                    risk_fact_map[idx] = []
                risk_fact_map[idx].append({'severity': severity, 'desc': risk_desc})
            except (ValueError, IndexError):
                pass
    
    # ── Intelligent Fact Consolidation ──
    # Groups near-duplicate facts (e.g., "born in Taiwan" / "born Feb 17, 1963, in Tainan, Taiwan")
    # into single consolidated entries. Preserves the most detailed version.
    # 
    # Design: Uses Jaccard similarity (symmetric) + containment (asymmetric) to catch
    # both near-duplicates and strict subset relationships. Requires minimum 2 shared
    # words to prevent false merges on very short facts.
    import re as _re_dedup
    
    def _word_set(text):
        """Extract meaningful words for similarity comparison.
        
        Removes stop words AND the research subject's name/company since
        every fact contains these, making them noise for deduplication.
        """
        stop = {'is', 'the', 'a', 'an', 'of', 'and', 'in', 'to', 'for', 'as', 'was', 'has', 'had',
                'been', 'that', 'its', 'his', 'her', 'by', 'at', 'on', 'with', 'from', 'he', 'she',
                'are', 'not', 'also', 'or', 'who', 'which', 'their', 'this', 'have', 'were', 'being'}
        # Also remove subject name tokens (appears in every fact — noise for dedup)
        subject_tokens = set(_re_dedup.findall(r'[a-z]+', query.lower()))
        words = set(_re_dedup.findall(r'[a-z]+', text.lower())) - stop - subject_tokens
        return words
    
    def _jaccard(a, b):
        """Jaccard similarity: |A ∩ B| / |A ∪ B|. Symmetric."""
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)
    
    def _containment(a, b):
        """Containment: fraction of smaller set found in larger set. Asymmetric.
        
        Catches cases where one fact is a strict summary of another:
        e.g., "born in Taiwan" ⊂ "born Feb 17, 1963, in Tainan, Taiwan"
        """
        if not a or not b:
            return 0.0
        smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
        if len(smaller) == 0:
            return 0.0
        overlap = len(smaller & larger)
        return overlap / len(smaller)
    
    def _should_consolidate(words_a, words_b):
        """Determine if two facts should be merged.
        
        Rules:
        1. Jaccard ≥ 0.45 AND at least 2 shared words → merge
        2. Containment ≥ 0.80 AND at least 2 shared words → merge
        3. Otherwise → keep separate
        
        The min-2-shared-words guard prevents false merges on very short facts
        where single-word overlap creates misleadingly high scores.
        """
        shared = len(words_a & words_b)
        if shared < 2:
            return False  # Guard: require meaningful overlap
        
        if _jaccard(words_a, words_b) >= 0.45:
            return True
        if _containment(words_a, words_b) >= 0.80:
            return True
        return False
    
    # Group by category first, then consolidate within each group
    consolidated_facts = []
    seen_indices = set()
    raw_pos_to_display = {}
    
    # Build word sets once for all facts
    fact_words = [_word_set(f.get('content', '')) for f in facts]
    
    for i, fact in enumerate(facts):
        if i in seen_indices:
            continue
        
        # Find all similar facts (same category, passes similarity threshold)
        group = [i]
        for j in range(i + 1, len(facts)):
            if j in seen_indices:
                continue
            if facts[j].get('category') != fact.get('category'):
                continue
            if _should_consolidate(fact_words[i], fact_words[j]):
                group.append(j)
                seen_indices.add(j)
        
        # Pick the longest (most detailed) fact as the primary representative
        best_idx = max(group, key=lambda idx: len(facts[idx].get('content', '')))
        best_fact = dict(facts[best_idx])  # Copy to avoid mutating original

        # C3: union RAW citations across the whole group — the representative
        # alone would silently drop the other grouped facts' sources. Kept
        # independent of the evidence[:5] cap below, and read from the raw
        # seam (M2), not the escaped facts this loop iterates.
        best_fact['_citations'] = _union_citations(
            [raw_citations[idx] for idx in group if idx < len(raw_citations)]
        )

        # Quote → its OWN member's sources: the merged evidence list loses
        # which grouped fact each quote came from; key by the ESCAPED quote
        # (what the evidence rows render) so each row can link its source.
        quote_sources = {}
        for idx in group:
            rq = raw_quotes[idx] if idx < len(raw_quotes) else ''
            if not rq:
                continue
            key = str(_html_escape(rq))
            bucket = quote_sources.setdefault(key, [])
            seen_urls = {c['url'] for c in bucket}
            bucket.extend(
                c for c in (raw_citations[idx] if idx < len(raw_citations) else [])
                if c['url'] not in seen_urls
            )
        best_fact['_quote_sources'] = quote_sources
        
        # Merge metadata from all grouped facts
        if len(group) > 1:
            all_evidence = []
            all_entities = set()
            max_confidence = best_fact.get('confidence', 0)
            any_verified = best_fact.get('verified', False)
            max_verification_count = best_fact.get('verification_count', 1)
            
            for idx in group:
                f = facts[idx]
                ev = f.get('evidence', [])
                if isinstance(ev, list):
                    all_evidence.extend(ev)
                elif ev:
                    all_evidence.append(str(ev))
                ent = f.get('entities_mentioned', [])
                if isinstance(ent, list):
                    all_entities.update(ent)
                max_confidence = max(max_confidence, f.get('confidence', 0))
                if f.get('verified'):
                    any_verified = True
                max_verification_count = max(max_verification_count, f.get('verification_count', 1))
            
            best_fact['evidence'] = list(dict.fromkeys(all_evidence))[:5]  # Dedupe, keep top 5
            best_fact['entities_mentioned'] = list(all_entities)
            best_fact['confidence'] = max_confidence
            best_fact['verified'] = any_verified
            best_fact['verification_count'] = max_verification_count
            best_fact['_merged_count'] = len(group)
            best_fact['_original_index'] = best_idx + 1  # 1-based
        else:
            best_fact['_merged_count'] = 1
            best_fact['_original_index'] = i + 1
        
        # Carry over risk mapping from any original indices in the group.
        # Dedupe: ONE risk often cites several facts that consolidation
        # merges into one card — without this it renders N identical
        # badges + "Links to risk" rows on that card.
        merged_risks = []
        seen_risk_keys = set()
        for idx in group:
            orig_i = idx + 1  # 1-based index for risk_fact_map lookup
            for rm in risk_fact_map.get(orig_i, []):
                key = (rm['severity'], rm['desc'])
                if key not in seen_risk_keys:
                    seen_risk_keys.add(key)
                    merged_risks.append(rm)
        if merged_risks:
            best_fact['_risk_links'] = merged_risks
        
        consolidated_facts.append(best_fact)
        # Raw 1-based position -> consolidated display number (#N as shown).
        # Risk evidence refs point at RAW positions; rendering those numbers
        # verbatim mislabels facts once consolidation renumbers them.
        display_no = len(consolidated_facts)
        for idx in group:
            raw_pos_to_display[idx + 1] = display_no
    
    # Use consolidated facts for display
    display_facts = consolidated_facts
    
    # ── Build facts HTML with consolidation, trend badges, sort, multi-select, pagination ──
    FACTS_PER_PAGE = 25
    
    # Pre-compute trend membership for each consolidated fact
    trend_keywords_lookup = {
        'expansion': ['growth', 'expanding', 'partnership', 'new', 'launch', 'record', 'billion', 'trillion', 'largest'],
        'risk_escalation': ['lawsuit', 'investigation', 'fraud', 'violation', 'concern', 'restriction', 'ban', 'sanctions'],
        'leadership': ['strategy', 'vision', 'warns', 'calls for', 'announces', 'says', 'believes', 'predicts'],
        'geopolitical': ['china', 'export', 'trade', 'trump', 'government', 'policy', 'regulation', 'national security'],
    }
    trend_labels = {
        'expansion': ('📈', 'Growth'),
        'risk_escalation': ('⚠️', 'Risk'),
        'leadership': ('🎯', 'Strategy'),
        'geopolitical': ('🌐', 'Geopolitical'),
    }
    trend_badge_colors = {
        'expansion': '#10b981',
        'risk_escalation': '#ef4444',
        'leadership': '#3b82f6',
        'geopolitical': '#8b5cf6',
    }
    
    for df in display_facts:
        content_lower = df.get('content', '').lower()
        fact_trends = []
        for t_type, kws in trend_keywords_lookup.items():
            if any(kw in content_lower for kw in kws):
                fact_trends.append(t_type)
        df['_trends'] = fact_trends
    
    # Count facts per category in consolidated view
    consol_by_category = {}
    for df in display_facts:
        cat = df.get('category', 'unknown')
        consol_by_category[cat] = consol_by_category.get(cat, 0) + 1
    
    facts_html = ""
    
    # Sort controls — now includes trend sort
    facts_html += """
    <div class="sort-controls">
        <span class="sort-label">Sort by:</span>
        <button class="sort-btn active" onclick="sortFacts('default')">Default</button>
        <button class="sort-btn" onclick="sortFacts('confidence-desc')">Highest Confidence</button>
        <button class="sort-btn" onclick="sortFacts('confidence-asc')">Lowest Confidence</button>
        <button class="sort-btn" onclick="sortFacts('risk')">⚠️ Risk-Linked</button>
        <button class="sort-btn" onclick="sortFacts('trend')">📊 Trend-Linked</button>
    </div>
    """
    
    # Category filter tabs — multi-select enabled
    total_consol = len(display_facts)
    facts_html += '<div class="category-tabs" id="categoryTabs">'
    facts_html += f'<button class="cat-tab active" data-cat="all" onclick="toggleCategory(this, \'all\')">All ({total_consol})</button>'
    for cat in category_order:
        if cat in consol_by_category:
            count = consol_by_category[cat]
            label = category_labels.get(cat, cat.title())
            facts_html += f'<button class="cat-tab" data-cat="{cat}" onclick="toggleCategory(this, \'{cat}\')">{label} ({count})</button>'
    facts_html += '</div>'
    
    # Consolidated fact count banner
    if total_consol < len(facts):
        facts_html += f'<div class="consolidation-banner">🧠 {len(facts)} raw facts intelligently consolidated into {total_consol} unique findings — duplicates merged, detail preserved</div>'
    
    # All facts with expandable detail panels
    facts_html += '<div id="factsContainer">'
    for i, fact in enumerate(display_facts, 1):
        content = fact.get('content', 'No content')
        category = fact.get('category', 'unknown')
        confidence = fact.get('confidence', 0) * 100
        verified = fact.get('verified', False)
        verification_count = fact.get('verification_count', 1)
        evidence_list = fact.get('evidence', [])
        entities = fact.get('entities_mentioned', [])
        merged_count = fact.get('_merged_count', 1)
        fact_trends = fact.get('_trends', [])
        risk_links = fact.get('_risk_links', [])
        
        conf_class = "fact-high" if confidence >= 80 else "fact-med" if confidence >= 60 else "fact-low"
        color = category_colors.get(category, '#6b7280')
        hidden = ' style="display:none;"' if i > FACTS_PER_PAGE else ''
        verified_badge = ' <span class="verified-badge">✓ Verified</span>' if verified else ''
        
        # Merged count badge
        merged_badge = f'<span class="merged-badge" title="{merged_count} similar facts consolidated">🔗 {merged_count} merged</span> ' if merged_count > 1 else ''
        
        # Risk badge from consolidated risk links
        risk_badge = ''
        has_risk = 0
        for rm in risk_links:
            sev = rm['severity']
            sev_colors_map = {'critical': '#dc2626', 'high': '#ea580c', 'medium': '#d97706', 'low': '#059669'}
            sev_color = sev_colors_map.get(sev, '#6b7280')
            risk_badge += f'<span class="risk-badge" style="background:{sev_color}" title="{rm["desc"]}">⚠ {sev.upper()}</span> '
            has_risk = 1
        
        # Trend badges
        trend_badge_html = ''
        has_trend = 0
        trend_data = ','.join(fact_trends) if fact_trends else ''
        for t in fact_trends:
            t_icon, t_label = trend_labels.get(t, ('', ''))
            t_color = trend_badge_colors.get(t, '#6b7280')
            trend_badge_html += f'<span class="trend-badge-fact" style="background:{t_color}15;color:{t_color};border:1px solid {t_color}33">{t_icon} {t_label}</span> '
            has_trend = 1
        
        # Per-fact citation chips (B1) — from the raw seam via the C3 union
        citations_html = _render_citation_chips(fact.get('_citations', []))

        # Build expandable details
        details_parts = []
        if evidence_list:
            ev_items = evidence_list if isinstance(evidence_list, list) else [str(evidence_list)]
            quote_sources = fact.get('_quote_sources', {})
            # Show 3 by default (OQ-B3), rest behind a class-based toggle —
            # never silently discarded
            for ev_i, ev in enumerate(ev_items):
                extra_cls = ' ev-extra' if ev_i >= 3 else ''
                # Quote → source connection: link the row to the source(s)
                # of the group member this quote came from, anchored so the
                # click lands on the highlighted sentence
                ev_key = ev.strip() if isinstance(ev, str) else ev
                src_links = _render_quote_source_links(quote_sources.get(ev_key))
                src_html = f' <span class="ev-sources">— {src_links}</span>' if src_links else ''
                details_parts.append(f'<div class="detail-evidence{extra_cls}">📝 "{ev}"{src_html}</div>')
            if len(ev_items) > 3:
                details_parts.append(
                    f'<button type="button" class="show-all-btn" '
                    f'data-more="Show all {len(ev_items)} quotes" data-less="Show fewer" '
                    f'onclick="toggleShowAll(this)">Show all {len(ev_items)} quotes</button>'
                )
        if verified:
            s_suffix = "s" if verification_count > 1 else ""
            details_parts.append(f'<div class="detail-verified">✅ Cross-verified across {verification_count} source{s_suffix}</div>')
        else:
            details_parts.append(f'<div class="detail-unverified">📄 Single source (not yet cross-verified)</div>')
        if entities:
            ent_str = ", ".join(entities[:6])
            details_parts.append(f'<div class="detail-entities">🏷️ Entities: {ent_str}</div>')
        if risk_links:
            for rm in risk_links:
                details_parts.append(f'<div class="detail-risk">⚠️ Links to {rm["severity"].upper()} risk: {rm["desc"]}</div>')
        if merged_count > 1:
            details_parts.append(f'<div class="detail-merged">🔗 Consolidated from {merged_count} similar facts — highest confidence and all evidence retained</div>')
        if fact_trends:
            trend_names = [trend_labels.get(t, ('', t))[1] for t in fact_trends]
            trend_str = ", ".join(trend_names)
            details_parts.append(f'<div class="detail-trend">📊 Trend signals: {trend_str}</div>')
        
        details_html = '\n'.join(details_parts)
        
        facts_html += f"""
        <div class="fact {conf_class}" data-category="{category}" data-index="{i}" data-confidence="{confidence}" data-risk="{has_risk}" data-trend="{trend_data}" data-has-trend="{has_trend}"{hidden} onclick="toggleFactDetail(this, event)">
            <div class="fact-header">
                <span class="fact-number">#{i}</span>
                <span class="fact-category" style="background:{color}22;color:{color};border:1px solid {color}44">{category.upper()}</span>
                {merged_badge}{risk_badge}{trend_badge_html}
                <span class="confidence">Confidence: {confidence:.0f}%{verified_badge}</span>
                <span class="expand-icon">▸</span>
            </div>
            <div class="fact-content">{content}</div>
            {citations_html}
            <div class="fact-details">{details_html}</div>
        </div>
        """
    facts_html += '</div>'
    
    # Pagination controls
    total_pages = (total_consol + FACTS_PER_PAGE - 1) // FACTS_PER_PAGE
    if total_pages > 1:
        facts_html += f"""
        <div class="pagination">
            <button class="page-btn" onclick="changePage(-1)" id="prevBtn" disabled>← Previous</button>
            <span class="page-info">Page <span id="currentPage">1</span> of {total_pages} ({total_consol} facts)</span>
            <button class="page-btn" onclick="changePage(1)" id="nextBtn">Next →</button>
        </div>
        """
    
    # ── Build connections HTML ──
    conn_type_colors = {
        'co-founder': '#7c3aed',
        'family': '#ec4899', 
        'leadership': '#2563eb',
        'board_member': '#0891b2',
        'employer': '#6366f1',
        'education': '#059669',
        'colleague': '#8b5cf6',
        'friend': '#f472b6',
        'investor/partner': '#d97706',
        'recognition': '#eab308',
        'political': '#dc2626',
        'philanthropy': '#10b981',
        'regulatory': '#ea580c',
        'mentor': '#0ea5e9',
        'predecessor': '#64748b',
        'successor': '#64748b',
    }
    
    connections_html = ""
    if connections:
        for conn in connections:
            entity_1 = conn.get('entity_1', query)
            entity_2 = conn.get('entity_2', 'Unknown')
            rel_type = conn.get('relationship_type', 'unknown')
            strength = conn.get('strength', 0.5)
            time_period = conn.get('time_period', '')
            evidence = conn.get('evidence', '')
            
            # Strength bar visualization
            bar_width = int(strength * 100)
            bar_color = '#10b981' if strength >= 0.8 else '#f59e0b' if strength >= 0.5 else '#6b7280'
            type_color = conn_type_colors.get(rel_type, '#6b7280')
            
            connections_html += f"""
            <div class="connection">
                <div class="conn-entities">{entity_1} ↔ {entity_2}</div>
                <div class="conn-meta">
                    <span class="conn-type" style="background:{type_color}22;color:{type_color};border:1px solid {type_color}44">{rel_type}</span>
                    {f'<span class="conn-period">{time_period}</span>' if time_period else ''}
                </div>
                <div class="strength-bar-container">
                    <div class="strength-bar" style="width:{bar_width}%;background:{bar_color}"></div>
                    <span class="strength-label">{strength:.0%}</span>
                </div>
                {f'<div class="conn-evidence">{evidence}</div>' if evidence else ''}
            </div>
            """
    else:
        connections_html = "<p class='no-data'>No connections mapped</p>"
    
    # ── Build risks HTML with severity color coding ──
    severity_colors = {
        'critical': {'bg': '#fef2f2', 'border': '#dc2626', 'badge': '#dc2626'},
        'high': {'bg': '#fff7ed', 'border': '#ea580c', 'badge': '#ea580c'},
        'medium': {'bg': '#fffbeb', 'border': '#f59e0b', 'badge': '#d97706'},
        'low': {'bg': '#f0fdf4', 'border': '#10b981', 'badge': '#059669'}
    }
    
    risks_html = ""
    if risks:
        # Sort risks by severity (critical > high > medium > low)
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        sorted_risks = sorted(risks, key=lambda r: severity_order.get(r.get('severity', 'low'), 3))

        # ── Filter pills (severity = the triage lens, then category) +
        #    condensed/detailed toggle. Same pattern the Facts section
        #    teaches; values live in data attrs (never inline JS args — an
        #    escaped quote inside an onclick string would still break it).
        sev_counts = {}
        rcat_counts = {}
        for r in sorted_risks:
            s = r.get('severity', 'low').lower()
            sev_counts[s] = sev_counts.get(s, 0) + 1
            c = r.get('category', 'unknown').lower()
            rcat_counts[c] = rcat_counts.get(c, 0) + 1

        sev_badge_colors = {'critical': '#dc2626', 'high': '#ea580c',
                            'medium': '#d97706', 'low': '#059669'}
        pills = [f'<button type="button" class="risk-pill active" data-kind="sev" '
                 f'data-val="all" data-label="All" '
                 f'onclick="filterRisks(this)">All ({len(sorted_risks)})</button>']
        for s in ('critical', 'high', 'medium', 'low'):
            if s in sev_counts:
                color = sev_badge_colors.get(s, '#6b7280')
                pills.append(
                    f'<button type="button" class="risk-pill" data-kind="sev" data-val="{s}" '
                    f'data-label="{s.upper()}" style="color:{color};border-color:{color}66" '
                    f'onclick="filterRisks(this)">{s.upper()} ({sev_counts[s]})</button>'
                )
        pills.append('<span class="risk-pill-divider"></span>')
        for c, count in sorted(rcat_counts.items(), key=lambda kv: -kv[1]):
            pills.append(
                f'<button type="button" class="risk-pill" data-kind="cat" data-val="{c}" '
                f'data-label="{c.upper()}" '
                f'onclick="filterRisks(this)">{c.upper()} ({count})</button>'
            )
        pills.append(
            '<span class="risk-view-controls">'
            '<button type="button" class="risk-pill risk-view-toggle" '
            'onclick="toggleAllRiskFacts(this)">Expand all facts</button>'
            '<button type="button" class="risk-pill risk-view-toggle" '
            'onclick="toggleRiskView(this)">Condensed view</button>'
            '</span>'
        )
        risks_html += f'<div class="risk-filters">{"".join(pills)}</div>'
        risks_html += '<div id="risksContainer">'
        
        # Risk category colors for styled labels
        risk_cat_colors = {
            'legal': '#ef4444', 'compliance': '#f97316', 'financial': '#10b981',
            'reputational': '#8b5cf6', 'integrity': '#6366f1', 'tax & asset': '#d97706',
            'regulatory': '#ea580c', 'operational': '#0891b2', 'governance': '#7c3aed',
            'political': '#dc2626', 'cyber': '#06b6d4', 'environmental': '#059669',
        }
        
        for risk in sorted_risks:
            category = risk.get('category', 'unknown')
            category_upper = category.upper()
            severity = risk.get('severity', 'low').lower()
            desc = risk.get('description', 'No description')
            confidence = risk.get('confidence', 0.5) * 100
            impact = risk.get('impact_score', 5.0)
            trend = risk.get('trend', 'isolated').title()
            evidence = risk.get('evidence', [])
            
            colors = severity_colors.get(severity, severity_colors['low'])
            cat_color = risk_cat_colors.get(category.lower(), '#6b7280')

            # Risk citations (B2): resolve this risk's OWN "Fact N" evidence
            # refs positionally (same digit-strip as risk_fact_map — which is
            # a reverse fact→risk index and can NOT do this lookup, M3) and
            # union the referenced facts' RAW citations. A ref to an absent
            # position simply contributes nothing — card renders chip-less.
            risk_citation_lists = []
            support_positions = []
            for ref in evidence:
                try:
                    pos = int(''.join(c for c in str(ref) if c.isdigit()))
                except (ValueError, IndexError):
                    continue
                if 1 <= pos <= len(raw_citations):
                    risk_citation_lists.append(raw_citations[pos - 1])
                    support_positions.append(pos)
            risk_chips_html = _render_citation_chips(
                _union_citations(risk_citation_lists)
            )

            # Supporting facts: the raw "Evidence: Fact 32" labels were
            # positions in the PRE-consolidation list and mislabeled facts
            # once consolidation renumbered them. Show the actual facts
            # instead — display number (#N as shown in the Facts section),
            # consolidated text, and anchored source links — behind the
            # standard class-based expand.
            support_rows = []
            seen_display_nos = set()
            for pos in support_positions:
                display_no = raw_pos_to_display.get(pos)
                if not display_no or display_no in seen_display_nos:
                    continue
                seen_display_nos.add(display_no)
                display_fact = display_facts[display_no - 1]
                links = _render_quote_source_links(display_fact.get('_citations'))
                links_html = f' <span class="ev-sources">— {links}</span>' if links else ''
                support_rows.append(
                    f'<div class="risk-support-row">'
                    f'<button type="button" class="risk-fact-link" data-fact="{display_no}" '
                    f'onclick="jumpToFact(this)" title="Jump to this fact">#{display_no}</button> '
                    f'{display_fact.get("content", "")}{links_html}</div>'
                )
            risk_support_html = ''
            if support_rows:
                n = len(support_rows)
                s_suffix = 's' if n != 1 else ''
                risk_support_html = (
                    f'<div class="risk-support">'
                    f'<button type="button" class="show-all-btn" '
                    f'data-more="Show supporting fact{s_suffix} ({n})" '
                    f'data-less="Hide supporting fact{s_suffix}" '
                    f'onclick="toggleShowAll(this)">Show supporting fact{s_suffix} ({n})</button>'
                    f'<div class="risk-support-rows">{"".join(support_rows)}</div>'
                    f'</div>'
                )

            risks_html += f"""
            <div class="risk" data-severity="{severity}" data-rcat="{category.lower()}" style="background:{colors['bg']};border-left:4px solid {colors['border']}" onclick="toggleRiskDetail(this, event)">
                <div class="risk-header">
                    <span class="risk-category-label" style="background:{cat_color}18;color:{cat_color};border:1px solid {cat_color}40">{category_upper}</span>
                    <span class="risk-header-right"><span class="severity" style="background:{colors['badge']}">{severity.upper()}</span><span class="risk-expand-icon">▸</span></span>
                </div>
                <div class="risk-desc">{desc}</div>
                {risk_chips_html}
                <div class="risk-meta">
                    <span>Confidence: {confidence:.0f}%</span>
                    <span>Impact: {impact:.0f}/10</span>
                    <span>Trend: {trend}</span>
                </div>
                {risk_support_html}
            </div>
            """
        risks_html += '</div>'
        risks_html += ('<div id="riskEmptyState" class="no-data" '
                       'style="display:none">No risks match the selected filters.</div>')
    else:
        risks_html = "<p class='no-data'>✅ No significant risks identified</p>"
    
    # ── Build coverage bars ──
    coverage_html = ""
    for cat in category_order[:-1]:  # Skip 'unknown'
        value = coverage.get(cat, 0)
        pct = value * 100
        color = category_colors.get(cat, '#6b7280')
        label = cat.title()
        coverage_html += f"""
        <div class="coverage-row">
            <span class="coverage-label">{label}</span>
            <div class="coverage-bar-bg">
                <div class="coverage-bar-fill" style="width:{pct:.0f}%;background:{color}"></div>
            </div>
            <span class="coverage-pct">{pct:.1f}%</span>
        </div>
        """
    
    # ── Build score breakdown ──
    score_breakdown_html = ""
    breakdown_items = [
        ('Fact Quality', breakdown.get('fact_quality', 0), 35, '#3b82f6'),
        ('Coverage', breakdown.get('coverage', 0), 25, '#10b981'),
        ('Risk Assessment', breakdown.get('risk_assessment', 0), 20, '#ef4444'),
        ('Connections', breakdown.get('connection_mapping', 0), 20, '#8b5cf6')
    ]
    for label, val, max_val, color in breakdown_items:
        pct = (val / max_val * 100) if max_val > 0 else 0
        score_breakdown_html += f"""
        <div class="score-component">
            <div class="score-comp-header">
                <span>{label}</span>
                <span>{val:.1f}/{max_val}</span>
            </div>
            <div class="score-comp-bar-bg">
                <div class="score-comp-bar" style="width:{pct:.0f}%;background:{color}"></div>
            </div>
        </div>
        """
    
    # ── Executive Intelligence Summary (replaces plain strengths/recommendations) ──
    # Build rich metric cards instead of plain text list
    #
    # Extract scoring metrics from calculate_quality_score output.
    # These were computed in the scoring function and returned in score_result['metadata'].
    score_meta = score_result.get('metadata', {})
    facts_count = score_meta.get('facts_count', len(facts))
    high_confidence_facts = score_meta.get('high_confidence_facts', 0)
    corroborated_facts = score_meta.get('corroborated_facts', 0)
    corroboration_ratio = corroborated_facts / facts_count if facts_count > 0 else 0
    categories_with_data = score_meta.get('categories_covered', 0)
    risk_flags_count = score_meta.get('risk_flags_count', len(risks))
    connections_count = score_meta.get('connections_count', len(connections))
    diversity_count = score_meta.get('relationship_types', 0)
    avg_confidence = score_meta.get('avg_confidence', 0.0)
    total_categories = 6  # biographical, professional, financial, legal, behavioral, connections
    coverage_score = coverage.get('average', 0) * 100  # Convert to percentage
    strength_cards = []
    
    # Fact discovery card
    if facts_count >= 50:
        strength_cards.append({
            'icon': '🔬', 'title': 'Deep Fact Discovery',
            'metric': str(facts_count), 'unit': 'facts',
            'detail': f'{facts_count} unique facts extracted across {categories_with_data} research categories',
            'color': '#10b981'
        })
    elif facts_count >= 30:
        strength_cards.append({
            'icon': '🔬', 'title': 'Good Fact Discovery',
            'metric': str(facts_count), 'unit': 'facts',
            'detail': f'{facts_count} facts extracted across {categories_with_data} categories',
            'color': '#3b82f6'
        })
    
    # Corroboration card
    if corroboration_ratio >= 0.3:
        strength_cards.append({
            'icon': '✅', 'title': 'Cross-Verified Intelligence',
            'metric': f'{corroboration_ratio:.0%}', 'unit': 'verified',
            'detail': f'{corroborated_facts} of {facts_count} facts independently confirmed across multiple sources',
            'color': '#10b981'
        })
    elif corroboration_ratio >= 0.15:
        strength_cards.append({
            'icon': '✅', 'title': 'Moderate Corroboration',
            'metric': f'{corroboration_ratio:.0%}', 'unit': 'verified',
            'detail': f'{corroborated_facts} of {facts_count} facts cross-verified',
            'color': '#f59e0b'
        })
    
    # Source quality card
    if avg_confidence >= 0.8:
        strength_cards.append({
            'icon': '🎯', 'title': 'High-Quality Sources',
            'metric': f'{avg_confidence:.0%}', 'unit': 'avg confidence',
            'detail': f'{high_confidence_facts} of {facts_count} facts exceed 70% confidence threshold',
            'color': '#3b82f6'
        })
    
    # Risk assessment card
    if risk_flags_count >= 4:
        severity_breakdown = {}
        for r in risks:
            sev = r.get('severity', 'low')
            severity_breakdown[sev] = severity_breakdown.get(sev, 0) + 1
        sev_parts = [f'{v} {k.upper()}' for k, v in sorted(severity_breakdown.items())]
        strength_cards.append({
            'icon': '🛡️', 'title': 'Comprehensive Risk Scan',
            'metric': str(risk_flags_count), 'unit': 'risks identified',
            'detail': f'Multi-category assessment: {", ".join(sev_parts)}',
            'color': '#ef4444'
        })
    elif 1 <= risk_flags_count <= 3:
        strength_cards.append({
            'icon': '🛡️', 'title': 'Risk Assessment',
            'metric': str(risk_flags_count), 'unit': 'risks identified',
            'detail': f'{risk_flags_count} risk flags detected and documented',
            'color': '#f59e0b'
        })
    
    # Coverage card
    if categories_with_data >= 5:
        strength_cards.append({
            'icon': '📊', 'title': 'Full Research Coverage',
            'metric': f'{categories_with_data}/{total_categories}', 'unit': 'categories',
            'detail': f'{coverage_score:.1f}% average coverage across all research dimensions',
            'color': '#8b5cf6'
        })
    
    # Connections card
    if connections_count >= 8:
        strength_cards.append({
            'icon': '🕸️', 'title': 'Rich Network Mapping',
            'metric': str(connections_count), 'unit': 'connections',
            'detail': f'{connections_count} relationships mapped across {diversity_count} relationship types',
            'color': '#06b6d4'
        })
    
    # Build HTML for strength cards
    strengths_html = ""
    for card in strength_cards:
        strengths_html += f"""
        <div class="strength-card">
            <div class="strength-icon" style="color:{card['color']}">{card['icon']}</div>
            <div class="strength-body">
                <div class="strength-title">{card['title']}</div>
                <div class="strength-metric" style="color:{card['color']}">{card['metric']} <span class="strength-unit">{card['unit']}</span></div>
                <div class="strength-detail">{card['detail']}</div>
            </div>
        </div>"""
    if not strengths_html:
        strengths_html = '<p class="no-data">N/A</p>'
    
    # Recommendations (keep simple — only show if actionable)
    recommendations_html = "".join(f"<li>{r}</li>" for r in recommendations) if recommendations else "<li>No improvements needed</li>"
    
    # ── Trend Analysis ──
    # Extract directional signals from recent behavioral/professional/legal facts
    # Provides forward-looking intelligence about the subject's trajectory
    import re as _re
    
    trend_signals = {
        'expansion': [],      # Growth, new markets, acquisitions
        'risk_escalation': [], # Increasing legal/regulatory pressure
        'leadership': [],     # Leadership signals, strategic direction
        'geopolitical': [],   # International/political dynamics
    }
    
    trend_keywords = {
        'expansion': ['growth', 'expanding', 'partnership', 'new', 'launch', 'record', 'billion', 'trillion', 'largest'],
        'risk_escalation': ['lawsuit', 'investigation', 'fraud', 'violation', 'concern', 'restriction', 'ban', 'sanctions'],
        'leadership': ['strategy', 'vision', 'warns', 'calls for', 'announces', 'says', 'believes', 'predicts'],
        'geopolitical': ['china', 'export', 'trade', 'trump', 'government', 'policy', 'regulation', 'national security'],
    }
    
    for fact_idx, fact in enumerate(facts):
        content_lower = fact.get('content', '').lower()
        category = fact.get('category', '')
        if category not in ('behavioral', 'professional', 'legal', 'financial'):
            continue
        has_recent_year = bool(_re.search(r'202[4-6]', fact.get('content', '')))
        for signal_type, keywords in trend_keywords.items():
            if any(kw in content_lower for kw in keywords):
                trend_signals[signal_type].append({
                    'content': fact.get('content', ''),
                    'recent': has_recent_year,
                    'category': category,
                    # B2: the originating fact's RAW citations (chips render
                    # through the helper, never interpolated directly)
                    'citations': raw_citations[fact_idx] if fact_idx < len(raw_citations) else [],
                })
    
    trend_html = ""
    trend_icons = {
        'expansion': ('📈', 'Growth & Expansion', '#10b981'),
        'risk_escalation': ('⚠️', 'Risk Trajectory', '#ef4444'),
        'leadership': ('🎯', 'Strategic Direction', '#3b82f6'),
        'geopolitical': ('🌐', 'Geopolitical Dynamics', '#8b5cf6'),
    }
    
    total_trends = sum(len(v) for v in trend_signals.values())
    
    for signal_type, items in trend_signals.items():
        if not items:
            continue
        icon, label, color = trend_icons[signal_type]
        # Visibility (B2/OQ-B3): top-3 by recency visible, the REST hidden
        # behind a class toggle — the old [:3] slice hid e.g. 14 of 17
        # signals with no way to see them.
        sorted_items = sorted(items, key=lambda x: x['recent'], reverse=True)
        trend_html += f'<div class="trend-group">'
        trend_html += f'<div class="trend-header" style="color:{color}">{icon} {label} ({len(items)} signals)</div>'
        for item_i, item in enumerate(sorted_items):
            recent_tag = ' <span class="trend-recent">RECENT</span>' if item['recent'] else ''
            chips = _render_citation_chips(item.get('citations', []))
            extra_cls = ' trend-extra' if item_i >= 3 else ''
            trend_html += f'<div class="trend-item{extra_cls}">• {item["content"][:150]}{recent_tag}{chips}</div>'
        if len(sorted_items) > 3:
            trend_html += (
                f'<button type="button" class="show-all-btn" '
                f'data-more="Show all {len(sorted_items)} signals" data-less="Show fewer" '
                f'onclick="toggleShowAll(this)">Show all {len(sorted_items)} signals</button>'
            )
        trend_html += '</div>'
    
    if not trend_html:
        trend_html = "<p class='no-data'>Insufficient data for trend analysis</p>"
    
    # ── Sidelined facts (C1.7a D10): collapsed, visible, never deleted ──
    # Text renders from the ESCAPED copy (result already passed the
    # chokepoint); hrefs come from the section's OWN raw seam (R6).
    # DOM-bounded: the sideline pool is NOT counted by the target-only
    # hard cap and can be large on exactly the contaminated runs this
    # section exists to display.
    SIDELINE_RENDER_CAP = 30
    sidelined_facts = result.get('sidelined_facts', []) or []
    sideline_section_html = ''
    if sidelined_facts:
        sideline_cards = []
        for sf_i, sf in enumerate(sidelined_facts[:SIDELINE_RENDER_CAP]):
            if not isinstance(sf, dict):
                continue
            sf_chips = _render_citation_chips(
                raw_sidelined_citations[sf_i]
                if sf_i < len(raw_sidelined_citations) else []
            )
            sf_conf = sf.get('confidence', 0)
            sf_conf_txt = (f' · confidence {sf_conf * 100:.0f}%'
                           if isinstance(sf_conf, (int, float)) else '')
            sideline_cards.append(
                f'<div class="sideline-fact">'
                f'<div class="sideline-fact-text">{sf.get("content", "")}</div>'
                f'<div class="sideline-fact-meta">'
                f'{str(sf.get("category", "")).upper()}{sf_conf_txt}</div>'
                f'{sf_chips}</div>'
            )
        sideline_more_html = ''
        overflow = len(sidelined_facts) - SIDELINE_RENDER_CAP
        if overflow > 0:
            sideline_more_html = (
                f'<div class="sideline-more">and {overflow} more set aside '
                f'— the full list is kept in the report data</div>'
            )
        sideline_section_html = f'''
        <!-- ═══ SIDELINED FACTS (C1.7a — other people sharing the name) ═══ -->
        <div class="section" id="sidelineSection">
            <div class="section-header" onclick="toggleSection(this)">
                🧍 Facts about other people named {query} ({len(sidelined_facts)}) — not included in the analysis or score
                <span class="arrow">▼</span>
            </div>
            <div class="section-content">
                <p class="sideline-note">Set aside during extraction because the
                source’s subject conflicted with the research target
                (different person, employer, era, or role). Kept visible so
                nothing is silently discarded — if one of these is actually
                your subject, refine the details and run again.</p>
                {''.join(sideline_cards)}
                {sideline_more_html}
            </div>
        </div>'''

    # ── Metadata ──
    iterations = metadata.get('iterations', 0)
    queries_executed = metadata.get('queries_executed', 0)
    # C1.5 honesty marker: present ONLY when the continue/verify branch
    # actually short-circuited on a user request (workflow stamps it — R3).
    finished_early_html = ""
    if metadata.get('finished_early'):
        max_iterations = metadata.get('max_iterations', 0)
        of_max = f" of {max_iterations}" if max_iterations else ""
        finished_early_html = (
            f'<div class="early-note">⏹ Generated early at the user\'s '
            f'request after {iterations}{of_max} iterations — coverage may '
            f'be thinner than a full run.</div>'
        )

    # ── Thin-subject honesty note (C3-minimal, 2026-07-13) ──────────────
    # A low score on a low-footprint subject reads as "this person is
    # risky/failing" when it actually means "little public information
    # exists". Say so explicitly. Render-side framing only — the scoring
    # math is untouched (grade semantics redesign lives in future P1).
    THIN_SUBJECT_FACTS = 15
    thin_note_html = ""
    if len(facts) < THIN_SUBJECT_FACTS:
        sidelined_clause = ""
        sidelined_n = metadata.get('sidelined_count', 0)
        if isinstance(sidelined_n, int) and sidelined_n > 0:
            sidelined_clause = (
                f" {sidelined_n} facts about other people sharing the name "
                f"were set aside (see the section at the bottom)."
            )
        thin_note_html = (
            f'<div class="thin-note">ℹ️ Limited public footprint — '
            f'{queries_executed} searches over {iterations} iterations were '
            f'exhausted and {len(facts)} verified facts were found. The '
            f'depth score reflects how little public information exists '
            f'about this subject — scarcity is not elevated risk.'
            f'{sidelined_clause}</div>'
        )
    
    # ================================================================
    # FULL HTML TEMPLATE
    # ================================================================
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Due Diligence Report: {query}</title>
    <style>
        /* ── Reset & Base ── */
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f1f5f9;
            padding: 20px;
            line-height: 1.6;
            color: #1e293b;
        }}

        /* ── Home link (back to the search page; plain navigation — allowed
              under the report CSP) ── */
        .home-link {{
            display: block;
            max-width: 1100px;
            margin: 0 auto 10px;
            color: #64748b;
            font-size: 0.88em;
            text-decoration: none;
        }}
        .home-link:hover {{ color: #1e293b; text-decoration: underline; }}
        @media print {{ .home-link {{ display: none; }} }}

        /* ── Container ── */
        .container {{ 
            max-width: 1100px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 16px; 
            box-shadow: 0 4px 24px rgba(0,0,0,0.08); 
            overflow: hidden; 
        }}
        
        /* ── Header ── */
        .header {{ 
            background: linear-gradient(135deg, #1e3a5f 0%, #2d5a8e 50%, #3b82f6 100%); 
            color: white; 
            padding: 48px 40px 36px; 
        }}
        .header h1 {{ font-size: 1.1em; font-weight: 400; opacity: 0.9; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; }}
        .header h2 {{ font-size: 2.4em; font-weight: 700; margin-bottom: 16px; }}
        .header-meta {{ display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.95em; opacity: 0.85; }}
        .thin-note {{ margin-top: 12px; padding: 8px 14px; border-radius: 8px;
            background: rgba(255,255,255,0.14); font-size: 0.85em; }}
        .score-caption {{ font-size: 0.72em; opacity: 0.75; margin-top: 6px;
            max-width: 380px; line-height: 1.45; }}
        .early-note {{ margin-top: 12px; padding: 8px 14px; border-radius: 8px;
            background: rgba(255,255,255,0.14); font-size: 0.9em; }}
        .score-hero {{
            display: flex; align-items: center; gap: 16px; 
            margin-top: 24px; padding: 16px 24px; 
            background: rgba(255,255,255,0.12); border-radius: 12px; 
            backdrop-filter: blur(10px);
        }}
        .score-number {{ font-size: 3em; font-weight: 800; line-height: 1; }}
        .score-details {{ font-size: 1.1em; }}
        .score-grade {{ font-size: 1.4em; font-weight: 700; }}
        
        /* ── Executive Summary ── */
        .exec-summary {{ 
            display: grid; grid-template-columns: 1fr 1fr; gap: 24px; 
            padding: 32px 40px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; 
        }}
        .summary-card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
        .summary-card h3 {{ font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; color: #64748b; margin-bottom: 12px; }}
        
        /* ── Score Breakdown ── */
        .score-component {{ margin-bottom: 12px; }}
        .score-comp-header {{ display: flex; justify-content: space-between; font-size: 0.9em; margin-bottom: 4px; }}
        .score-comp-bar-bg {{ height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }}
        .score-comp-bar {{ height: 100%; border-radius: 4px; transition: width 0.5s ease; }}
        
        /* ── Coverage Bars ── */
        .coverage-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
        .coverage-label {{ width: 100px; font-size: 0.85em; color: #475569; }}
        .coverage-bar-bg {{ flex: 1; height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }}
        .coverage-bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s ease; }}
        .coverage-pct {{ width: 50px; font-size: 0.85em; color: #475569; text-align: right; }}
        
        /* ── Strengths/Recommendations ── */
        .sr-list {{ list-style: none; padding: 0; }}
        .sr-list li {{ padding: 6px 0; font-size: 0.9em; border-bottom: 1px solid #f1f5f9; }}
        .sr-list li:last-child {{ border: none; }}
        .summary-card-wide {{ grid-column: 1 / -1; }}
        .strength-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
        .strength-card {{
            display: flex; gap: 12px; padding: 14px 16px;
            background: #f8fafc; border-radius: 10px; border: 1px solid #e2e8f0;
            transition: box-shadow 0.2s;
        }}
        .strength-card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        .strength-icon {{ font-size: 1.6em; flex-shrink: 0; }}
        .strength-body {{ flex: 1; min-width: 0; }}
        .strength-title {{ font-weight: 700; font-size: 0.85em; color: #334155; }}
        .strength-metric {{ font-size: 1.4em; font-weight: 800; line-height: 1.2; }}
        .strength-unit {{ font-size: 0.45em; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
        .strength-detail {{ font-size: 0.78em; color: #64748b; margin-top: 2px; line-height: 1.4; }}
        
        /* ── Search Box ── */
        .search-container {{ padding: 16px 40px; border-bottom: 1px solid #e2e8f0; }}
        .search-box {{ 
            width: 100%; padding: 12px 20px; border: 2px solid #e2e8f0; border-radius: 10px; 
            font-size: 15px; outline: none; transition: border 0.2s;
        }}
        .search-box:focus {{ border-color: #3b82f6; }}
        
        /* ── Sidelined facts (C1.7a — muted: excluded from the analysis) ── */
        .sideline-note {{ font-size: 0.85em; color: #64748b; margin-bottom: 14px; }}
        .sideline-fact {{
            padding: 10px 14px; margin-bottom: 8px; border-radius: 8px;
            background: #f8fafc; border: 1px dashed #cbd5e1;
        }}
        .sideline-fact-text {{ font-size: 0.9em; color: #475569; }}
        .sideline-fact-meta {{
            font-size: 0.7em; color: #94a3b8; margin-top: 3px;
            letter-spacing: 0.5px;
        }}
        .sideline-more {{ font-size: 0.82em; color: #94a3b8; margin-top: 10px; }}

        /* ── Sections ── */
        .section {{ border-bottom: 1px solid #e2e8f0; }}
        .section-header {{ 
            padding: 20px 40px; background: #f8fafc; cursor: pointer; 
            font-size: 1.1em; font-weight: 600; color: #1e293b; 
            display: flex; justify-content: space-between; align-items: center;
            user-select: none; transition: background 0.2s;
        }}
        .section-header:hover {{ background: #f1f5f9; }}
        .section-header .arrow {{ transition: transform 0.3s; font-size: 0.8em; }}
        .section-header.open .arrow {{ transform: rotate(180deg); }}
        .section-content {{ 
            padding: 0 40px; 
            max-height: 0; overflow: hidden; 
            transition: max-height 0.4s ease, padding 0.3s ease; 
        }}
        .section-content.open {{ 
            max-height: none;  /* NO HEIGHT LIMIT — was 5000px, caused truncation */
            padding: 24px 40px; 
            overflow: visible;
        }}
        
        /* ── Category Tabs ── */
        .category-tabs {{ 
            display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; 
            padding-bottom: 16px; border-bottom: 1px solid #e2e8f0; 
        }}
        .cat-tab {{ 
            padding: 6px 14px; border: 1px solid #e2e8f0; border-radius: 20px; 
            background: white; font-size: 0.85em; cursor: pointer; transition: all 0.2s;
        }}
        .cat-tab:hover {{ border-color: #3b82f6; color: #3b82f6; }}
        .cat-tab.active {{ background: #3b82f6; color: white; border-color: #3b82f6; }}
        
        /* ── Facts ── */
        .fact {{ 
            background: white; padding: 16px 20px; margin: 8px 0; 
            border: 1px solid #e2e8f0; border-radius: 10px; 
            border-left: 4px solid #3b82f6; transition: box-shadow 0.2s;
        }}
        .fact:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.06); cursor: pointer; }}
        .fact-high {{ border-left-color: #10b981; }}
        .fact-med {{ border-left-color: #f59e0b; }}
        .fact-low {{ border-left-color: #ef4444; }}
        .fact-header {{ 
            display: flex; align-items: center; gap: 10px; 
            margin-bottom: 8px; flex-wrap: wrap;
        }}
        .fact-number {{ font-weight: 700; color: #94a3b8; font-size: 0.85em; }}
        .fact-category {{ 
            padding: 2px 10px; border-radius: 12px; 
            font-size: 0.75em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
        }}
        .confidence {{ font-size: 0.8em; color: #64748b; margin-left: auto; }}
        .verified-badge {{ color: #10b981; font-weight: 600; }}
        .fact-content {{ font-size: 0.95em; line-height: 1.7; }}
        
        /* ── Merged Fact Badge ── */
        .merged-badge {{
            padding: 2px 8px; border-radius: 10px; font-size: 0.7em; font-weight: 600;
            background: #dbeafe; color: #1d4ed8; white-space: nowrap;
        }}
        
        /* ── Trend Badge on Facts ── */
        .trend-badge-fact {{
            padding: 1px 8px; border-radius: 10px; font-size: 0.7em; font-weight: 600;
            white-space: nowrap;
        }}
        
        /* ── Consolidation Banner ── */
        .consolidation-banner {{
            padding: 10px 16px; margin-bottom: 16px; border-radius: 8px;
            background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af;
            font-size: 0.85em; font-weight: 500;
        }}
        
        /* ── Risk Category Label (styled pill) ── */
        .risk-category-label {{
            padding: 3px 12px; border-radius: 12px; font-size: 0.78em;
            font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
        }}
        
        /* ── Multi-select Category Tabs ── */
        .cat-tab.selected {{ background: #dbeafe; color: #1d4ed8; border-color: #93c5fd; }}
        
        /* ── Detail rows for merged/trend info ── */
        .detail-merged {{ font-size: 0.85em; color: #1d4ed8; padding: 4px 0; }}
        .detail-trend {{ font-size: 0.85em; color: #7c3aed; padding: 4px 0; }}
        
        /* ── Expandable Fact Details ── */
        .fact-details {{ 
            max-height: 0; overflow: hidden; transition: max-height 0.3s ease;
            padding: 0; margin-top: 0; border-top: none;
        }}
        .fact.expanded .fact-details {{
            max-height: 500px; padding: 12px 0; margin-top: 10px;
            border-top: 1px solid #e2e8f0;
        }}
        .expand-icon {{ 
            font-size: 0.8em; color: #94a3b8; transition: transform 0.3s;
            margin-left: 8px;
        }}
        .fact.expanded .expand-icon {{ transform: rotate(90deg); }}
        .detail-evidence {{ 
            font-size: 0.85em; color: #475569; font-style: italic; 
            padding: 6px 12px; margin: 4px 0; background: #f8fafc; 
            border-left: 3px solid #cbd5e1; border-radius: 0 6px 6px 0;
        }}
        .ev-sources {{ font-style: normal; white-space: nowrap; }}
        .ev-source {{
            font-weight: 600; font-size: 0.92em; color: #3b82f6;
            text-decoration: none;
        }}
        .ev-source:hover {{ text-decoration: underline; }}
        .detail-verified {{ font-size: 0.85em; color: #059669; padding: 4px 0; }}
        .detail-unverified {{ font-size: 0.85em; color: #94a3b8; padding: 4px 0; }}
        .detail-entities {{ font-size: 0.85em; color: #6366f1; padding: 4px 0; }}
        .detail-risk {{ 
            font-size: 0.85em; color: #ea580c; padding: 6px 10px; 
            margin: 4px 0; background: #fff7ed; border-radius: 6px;
        }}
        
        /* ── Risk Badges on Facts ── */
        .risk-badge {{
            padding: 2px 8px; border-radius: 10px; font-size: 0.7em;
            font-weight: 700; color: white; text-transform: uppercase;
            letter-spacing: 0.3px; vertical-align: middle;
        }}

        /* ── Citation Chips (Phase B) ── */
        .citation-chips {{
            display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
            margin-top: 8px;
        }}
        .cite-chip {{
            display: inline-flex; align-items: center; gap: 5px;
            padding: 2px 10px; border-radius: 12px; font-size: 0.75em;
            font-weight: 600; text-decoration: none; color: #334155;
            background: #f1f5f9; border: 1px solid #e2e8f0;
            transition: all 0.15s;
        }}
        .cite-chip:hover {{ border-color: #3b82f6; color: #1d4ed8; background: #eff6ff; }}
        .cite-tier-dot {{
            width: 7px; height: 7px; border-radius: 50%; flex: 0 0 auto;
            background: #94a3b8;
        }}
        .cite-chip.tier-high .cite-tier-dot {{ background: #059669; }}
        .cite-chip.tier-established .cite-tier-dot {{ background: #3b82f6; }}
        .citation-chips .cite-extra {{ display: none; }}
        .citation-chips.cites-expanded .cite-extra {{ display: inline-flex; }}
        .cite-more-btn, .show-all-btn {{
            padding: 2px 10px; border-radius: 12px; font-size: 0.75em;
            font-weight: 600; cursor: pointer; color: #64748b;
            background: white; border: 1px dashed #cbd5e1;
        }}
        .cite-more-btn:hover, .show-all-btn:hover {{ border-color: #3b82f6; color: #3b82f6; }}
        .show-all-btn {{ display: block; margin-top: 8px; }}
        .ev-extra, .trend-extra {{ display: none; }}
        .expanded-all .ev-extra, .expanded-all .trend-extra {{ display: block; }}

        /* ── Risk filters + condensed view ── */
        .risk-filters {{
            display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
            margin-bottom: 14px;
        }}
        .risk-pill {{
            padding: 3px 12px; border: 1px solid #e2e8f0; border-radius: 14px;
            background: white; font-size: 0.78em; font-weight: 600;
            color: #64748b; cursor: pointer; transition: all 0.15s;
        }}
        .risk-pill:hover:not(:disabled) {{ border-color: #94a3b8; }}
        .risk-pill.active {{ background: #1e293b; color: white !important; border-color: #1e293b; }}
        .risk-pill:disabled {{ opacity: 0.35; cursor: default; }}
        .risk-pill-divider {{
            width: 1px; height: 18px; background: #e2e8f0; margin: 0 4px;
        }}
        .risk-view-controls {{ margin-left: auto; display: flex; gap: 6px; }}
        .risk-view-toggle {{ border-style: dashed; }}
        /* Condensed = 2-line scan; clicking a card expands JUST that card
           (same click-to-expand the fact cards teach) */
        .risks-condensed .risk:not(.risk-expanded) .citation-chips,
        .risks-condensed .risk:not(.risk-expanded) .risk-meta,
        .risks-condensed .risk:not(.risk-expanded) .risk-support:not(.expanded-all) {{ display: none; }}
        .risks-condensed .risk:not(.risk-expanded) .risk-desc {{
            display: -webkit-box; -webkit-line-clamp: 2;
            -webkit-box-orient: vertical; overflow: hidden;
        }}
        .risks-condensed .risk {{ padding: 10px 16px; margin-bottom: 8px; cursor: pointer; }}
        .risk-header-right {{ display: flex; align-items: center; gap: 8px; }}
        .risk-expand-icon {{
            display: none; font-size: 0.85em; color: #94a3b8;
            transition: transform 0.25s;
        }}
        .risks-condensed .risk-expand-icon {{ display: inline; }}
        .risks-condensed .risk.risk-expanded .risk-expand-icon {{ transform: rotate(90deg); }}

        /* ── Jump-to-fact from risk cards ── */
        .risk-fact-link {{
            font-weight: 700; color: #1d4ed8; background: #eff6ff;
            border: 1px solid #bfdbfe; border-radius: 10px;
            padding: 1px 8px; cursor: pointer; font-size: 0.95em;
        }}
        .risk-fact-link:hover {{ background: #dbeafe; }}
        .fact-flash {{ animation: factFlash 1.8s ease-out; }}
        @keyframes factFlash {{
            0% {{ box-shadow: 0 0 0 3px #3b82f6aa; }}
            100% {{ box-shadow: 0 0 0 3px transparent; }}
        }}
        /* Lives INSIDE the fact card the user jumped to (jumpToFact moves
           the one static node there) — right where their eye already is */
        .back-to-risks {{
            display: none; margin-top: 10px;
            padding: 5px 14px; border-radius: 16px; border: none;
            background: #1e293b; color: white; font-weight: 600;
            font-size: 0.8em; cursor: pointer;
        }}
        .back-to-risks.visible {{ display: inline-block; }}
        .back-to-risks:hover {{ background: #334155; }}
        @media print {{ .back-to-risks {{ display: none !important; }} }}

        /* ── Supporting facts on risk cards ── */
        .risk-support {{ margin-top: 10px; }}
        .risk-support .show-all-btn {{ margin-top: 0; }}
        .risk-support-rows {{ display: none; }}
        .risk-support.expanded-all .risk-support-rows {{ display: block; }}
        .risk-support-row {{
            font-size: 0.88em; color: #334155; padding: 6px 10px;
            margin: 6px 0; background: rgba(255,255,255,0.65);
            border-left: 3px solid #cbd5e1; border-radius: 0 6px 6px 0;
            line-height: 1.5;
        }}
        .risk-support-no {{ font-weight: 700; color: #64748b; margin-right: 4px; }}
        
        /* ── Sort Controls ── */
        .sort-controls {{
            display: flex; align-items: center; gap: 8px; 
            margin-bottom: 12px; flex-wrap: wrap;
        }}
        .sort-label {{ font-size: 0.85em; color: #64748b; font-weight: 600; }}
        .sort-btn {{
            padding: 4px 12px; border: 1px solid #e2e8f0; border-radius: 16px;
            background: white; font-size: 0.8em; cursor: pointer; transition: all 0.2s;
        }}
        .sort-btn:hover {{ border-color: #3b82f6; color: #3b82f6; }}
        .sort-btn.active {{ background: #1e293b; color: white; border-color: #1e293b; }}
        
        /* ── Trend Analysis ── */
        .trend-group {{
            margin-bottom: 20px; padding: 16px 20px;
            background: #f8fafc; border-radius: 10px;
            border-left: 4px solid #e2e8f0;
        }}
        .trend-header {{
            font-size: 1.05em; font-weight: 700; margin-bottom: 10px;
        }}
        .trend-item {{
            font-size: 0.9em; color: #334155; padding: 4px 0;
            line-height: 1.6;
        }}
        .trend-recent {{
            display: inline-block; padding: 1px 8px; border-radius: 10px;
            font-size: 0.7em; font-weight: 700; background: #dbeafe;
            color: #1d4ed8; vertical-align: middle; margin-left: 6px;
        }}
        .no-data {{ color: #94a3b8; font-style: italic; }}
        
        /* ── Connection Type Colors (expanded palette) ── */
        .conn-type {{ 
            padding: 3px 10px; border-radius: 12px; font-size: 0.75em; 
            font-weight: 600; text-transform: lowercase; 
        }}
        .conn-type-leadership {{ background: #dbeafe; color: #1e40af; }}
        .conn-type-family {{ background: #fce7f3; color: #be185d; }}
        .conn-type-colleague {{ background: #e0e7ff; color: #4338ca; }}
        .conn-type-employer {{ background: #d1fae5; color: #065f46; }}
        .conn-type-education {{ background: #fef3c7; color: #92400e; }}
        .conn-type-board_member {{ background: #e0e7ff; color: #3730a3; }}
        .conn-type-investor_partner {{ background: #dcfce7; color: #166534; }}
        .conn-type-predecessor_successor {{ background: #f1f5f9; color: #475569; }}
        .conn-type-recognition {{ background: #fef9c3; color: #a16207; }}
        .conn-type-political {{ background: #fee2e2; color: #991b1b; }}
        .conn-type-philanthropy {{ background: #f3e8ff; color: #7e22ce; }}
        .conn-type-co-founder {{ background: #cffafe; color: #0e7490; }}
        .conn-type-friend {{ background: #fce7f3; color: #9d174d; }}
        .conn-type-mentor {{ background: #e0f2fe; color: #0369a1; }}
        .conn-type-regulatory {{ background: #fef2f2; color: #b91c1c; }}
        
        /* ── Trend Analysis ── */
        .trend-group {{ margin-bottom: 16px; }}
        .trend-header {{ font-weight: 700; font-size: 1em; margin-bottom: 6px; }}
        .trend-item {{ font-size: 0.9em; color: #334155; padding: 4px 0 4px 8px; line-height: 1.5; }}
        .trend-recent {{ 
            display: inline-block; padding: 1px 6px; border-radius: 8px;
            background: #dbeafe; color: #1e40af; font-size: 0.75em; font-weight: 700;
            vertical-align: middle; margin-left: 6px;
        }}
        .no-data {{ color: #94a3b8; font-style: italic; }}
        
        /* ── Pagination ── */
        .pagination {{ 
            display: flex; justify-content: center; align-items: center; gap: 16px; 
            padding: 20px 0; margin-top: 16px; border-top: 1px solid #e2e8f0; 
        }}
        .page-btn {{ 
            padding: 8px 20px; border: 1px solid #3b82f6; border-radius: 8px; 
            background: white; color: #3b82f6; font-weight: 600; cursor: pointer; 
            transition: all 0.2s;
        }}
        .page-btn:hover:not(:disabled) {{ background: #3b82f6; color: white; }}
        .page-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
        .page-info {{ font-size: 0.9em; color: #64748b; }}
        
        /* ── Connections ── */
        .connection {{ 
            background: white; padding: 16px 20px; margin: 8px 0; 
            border: 1px solid #e2e8f0; border-radius: 10px; border-left: 4px solid #8b5cf6;
        }}
        .conn-entities {{ font-weight: 700; font-size: 1.05em; margin-bottom: 6px; color: #1e293b; }}
        .conn-meta {{ display: flex; gap: 12px; margin-bottom: 8px; }}
        .conn-type {{ 
            padding: 2px 10px; border-radius: 12px; 
            font-size: 0.8em; font-weight: 600; 
        }}
        .conn-period {{ font-size: 0.85em; color: #64748b; }}
        .strength-bar-container {{ 
            display: flex; align-items: center; gap: 8px; margin-top: 4px; 
        }}
        .strength-bar-container > div:first-child {{ 
            flex: 1; height: 6px; border-radius: 3px; background: #e2e8f0; overflow: hidden; position: relative; 
        }}
        .strength-bar {{ height: 100%; border-radius: 3px; }}
        .strength-label {{ font-size: 0.8em; color: #64748b; min-width: 36px; }}
        .conn-evidence {{ font-size: 0.8em; color: #94a3b8; margin-top: 6px; font-style: italic; }}
        
        /* ── Risks ── */
        .risk {{ 
            padding: 18px 22px; margin: 10px 0; border-radius: 10px; 
        }}
        .risk-header {{ 
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; 
        }}
        .risk-header strong {{ font-size: 0.9em; }}
        .severity {{ 
            padding: 3px 12px; border-radius: 12px; font-size: 0.75em; 
            font-weight: 700; color: white; text-transform: uppercase; letter-spacing: 0.5px;
        }}
        .risk-desc {{ font-size: 0.95em; line-height: 1.7; margin-bottom: 10px; }}
        .risk-meta {{ 
            display: flex; flex-wrap: wrap; gap: 16px; 
            font-size: 0.8em; color: #64748b; 
        }}
        .no-data {{ text-align: center; padding: 40px; color: #94a3b8; font-size: 1.1em; }}
        
        /* ── Footer ── */
        .footer {{ 
            padding: 24px 40px; background: #f8fafc; 
            font-size: 0.8em; color: #94a3b8; text-align: center; 
        }}
        
        /* ── Print ── */
        @media print {{
            body {{ padding: 0; background: white; }}
            .container {{ box-shadow: none; }}
            .search-container, .pagination, .cat-tab {{ display: none !important; }}
            .section-content {{ max-height: none !important; padding: 20px 40px !important; overflow: visible !important; }}
            .fact {{ break-inside: avoid; }}
        }}
        
        /* ── Mobile ── */
        @media (max-width: 768px) {{
            .exec-summary {{ grid-template-columns: 1fr; }}
            .header {{ padding: 28px 20px; }}
            .section-header, .section-content, .section-content.open {{ padding-left: 20px; padding-right: 20px; }}
            .search-container {{ padding: 12px 20px; }}
            .header h2 {{ font-size: 1.8em; }}
            .score-number {{ font-size: 2.2em; }}
        }}
    </style>
</head>
<body>
    <a class="home-link" href="/">← Run another report</a>
    <div class="container" id="report-top">
        <!-- ═══ HEADER ═══ -->
        <div class="header">
            <h1>🔍 Deep Research — Due Diligence Report</h1>
            <h2>{query}</h2>
            <div class="header-meta">
                <span>📅 {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</span>
                <span>⏱️ {duration:.0f}s ({duration/60:.1f} min)</span>
                <span>🔄 {iterations} iterations</span>
                <span>🔎 {queries_executed} searches</span>
            </div>
            {finished_early_html}
            {thin_note_html}
            <div class="score-hero">
                <div class="score-number">{score}</div>
                <div class="score-details">
                    <div class="score-grade">Grade {grade} — {quality}</div>
                    <div>out of 100 points</div>
                    <div class="score-caption">Measures research depth —
                    not the subject's credibility or risk.</div>
                </div>
            </div>
        </div>
        
        <!-- ═══ EXECUTIVE SUMMARY ═══ -->
        <div class="exec-summary">
            <div class="summary-card">
                <h3>📊 Score Breakdown</h3>
                {score_breakdown_html}
            </div>
            <div class="summary-card">
                <h3>📈 Research Coverage</h3>
                {coverage_html}
            </div>
            <div class="summary-card summary-card-wide">
                <h3>💪 Research Intelligence</h3>
                <div class="strength-grid">{strengths_html}</div>
            </div>
            <div class="summary-card">
                <h3>📋 Recommendations</h3>
                <ul class="sr-list">{recommendations_html}</ul>
            </div>
        </div>
        
        <!-- ═══ SEARCH ═══ -->
        <div class="search-container">
            <input type="text" class="search-box" id="searchInput" 
                   placeholder="🔍 Search across all facts, connections, and risks..." 
                   onkeyup="searchContent()">
        </div>
        
        <!-- ═══ RISK FLAGS ═══ -->
        <div class="section" id="risksSection">
            <div class="section-header open" onclick="toggleSection(this)">
                ⚠️ Risk Flags ({len(risks)})
                <span class="arrow">▼</span>
            </div>
            <div class="section-content open">
                {risks_html}
            </div>
        </div>
        
        <!-- ═══ ALL FACTS ═══ -->
        <div class="section" id="factsSection">
            <div class="section-header open" onclick="toggleSection(this)">
                📋 Research Facts ({total_consol} unique findings{f' from {len(facts)} raw' if total_consol < len(facts) else ''})
                <span class="arrow">▼</span>
            </div>
            <div class="section-content open">
                {facts_html}
            </div>
        </div>
        
        <!-- ═══ CONNECTIONS ═══ -->
        <div class="section" id="connectionsSection">
            <div class="section-header open" onclick="toggleSection(this)">
                🕸️ Connections & Relationships ({len(connections)})
                <span class="arrow">▼</span>
            </div>
            <div class="section-content open">
                {connections_html}
            </div>
        </div>
        
        <!-- ═══ TREND ANALYSIS ═══ -->
        <div class="section" id="trendsSection">
            <div class="section-header open" onclick="toggleSection(this)">
                📊 Trend Analysis ({total_trends} signals)
                <span class="arrow">▼</span>
            </div>
            <div class="section-content open">
                {trend_html}
            </div>
        </div>
        {sideline_section_html}
        <!-- ═══ FOOTER ═══ -->
        <div class="footer">
            Generated by Deep Research AI Agent · {total_consol} facts ({len(facts)} raw) · {len(connections)} connections · {len(risks)} risk flags · Quality Score {score}/100
        </div>
    </div>
    
    <button type="button" id="backToRisks" class="back-to-risks"
            onclick="backToRisks(event)">↑ Back to Risk Flags</button>

    <script>
        // ── Risk section: multi-select severity/category filters with live
        //    cross-dimension counts (a pill must never promise results the
        //    other dimension's selection has filtered away) ──
        const riskSevSel = new Set(), riskCatSel = new Set();
        function filterRisks(btn) {{
            const kind = btn.dataset.kind, val = btn.dataset.val;
            if (kind === 'sev' && val === 'all') {{
                riskSevSel.clear(); riskCatSel.clear();
            }} else {{
                const sel = (kind === 'sev') ? riskSevSel : riskCatSel;
                if (sel.has(val)) sel.delete(val); else sel.add(val);
            }}
            applyRiskFilters();
        }}
        function applyRiskFilters() {{
            const cards = [...document.querySelectorAll('#risksContainer .risk')];
            let shown = 0;
            cards.forEach(r => {{
                const ok = (riskSevSel.size === 0 || riskSevSel.has(r.dataset.severity))
                        && (riskCatSel.size === 0 || riskCatSel.has(r.dataset.rcat));
                r.style.display = ok ? '' : 'none';
                if (ok) shown++;
            }});
            document.querySelectorAll('.risk-pill[data-kind]').forEach(p => {{
                const kind = p.dataset.kind, val = p.dataset.val;
                let n, selected;
                if (kind === 'sev' && val === 'all') {{
                    n = cards.length;
                    selected = riskSevSel.size === 0 && riskCatSel.size === 0;
                }} else if (kind === 'sev') {{
                    n = cards.filter(r => r.dataset.severity === val
                        && (riskCatSel.size === 0 || riskCatSel.has(r.dataset.rcat))).length;
                    selected = riskSevSel.has(val);
                }} else {{
                    n = cards.filter(r => r.dataset.rcat === val
                        && (riskSevSel.size === 0 || riskSevSel.has(r.dataset.severity))).length;
                    selected = riskCatSel.has(val);
                }}
                p.textContent = p.dataset.label + ' (' + n + ')';
                p.classList.toggle('active', selected);
                // zero-result pills are unclickable UNLESS selected (so a
                // selection whose count collapsed can still be cleared)
                p.disabled = (n === 0 && !selected);
            }});
            const empty = document.getElementById('riskEmptyState');
            if (empty) empty.style.display = shown === 0 ? '' : 'none';
        }}
        function toggleRiskView(btn) {{
            const c = document.getElementById('risksContainer');
            if (!c) return;
            const condensed = c.classList.toggle('risks-condensed');
            btn.textContent = condensed ? 'Detailed view' : 'Condensed view';
            // fresh scan state on every view switch
            c.querySelectorAll('.risk.risk-expanded')
                .forEach(r => r.classList.remove('risk-expanded'));
        }}

        // Condensed view: click a card to expand JUST that card (links and
        // buttons inside keep their own behavior — same guard as facts)
        function toggleRiskDetail(riskEl, event) {{
            const c = document.getElementById('risksContainer');
            if (!c || !c.classList.contains('risks-condensed')) return;
            if (event && (event.target.tagName === 'A' || event.target.tagName === 'BUTTON')) return;
            riskEl.classList.toggle('risk-expanded');
        }}

        // Master toggle for every risk's supporting facts (deep-dive/print)
        function toggleAllRiskFacts(btn) {{
            const expand = btn.dataset.state !== 'open';
            document.querySelectorAll('#risksContainer .risk-support').forEach(w => {{
                w.classList.toggle('expanded-all', expand);
                const b = w.querySelector('.show-all-btn');
                if (b) b.textContent = expand ? b.dataset.less : b.dataset.more;
            }});
            btn.dataset.state = expand ? 'open' : '';
            btn.textContent = expand ? 'Collapse all facts' : 'Expand all facts';
        }}

        // ── Jump from a risk's supporting fact to the fact card ──
        let riskReturnEl = null;
        function jumpToFact(btn) {{
            const n = btn.dataset.fact;
            const target = document.querySelector('.fact[data-index="' + n + '"]');
            if (!target) return;
            riskReturnEl = btn.closest('.risk');
            // Clear any category filter so the fact can't be filter-hidden
            const allBtn = document.querySelector('.cat-tab[data-cat="all"]');
            if (allBtn && !activeCategories.has('all')) toggleCategory(allBtn, 'all');
            // Page to wherever the fact sits under the CURRENT sort order
            const pos = getVisibleFacts().indexOf(target);
            if (pos >= 0) {{
                currentPage = Math.floor(pos / factsPerPage) + 1;
                showPage(currentPage);
            }}
            target.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            target.classList.remove('fact-flash');
            void target.offsetWidth;  // restart the flash animation
            target.classList.add('fact-flash');
            // Park the return button INSIDE the landed-on fact card (moving
            // the one static node — no markup is ever built from strings)
            const back = document.getElementById('backToRisks');
            if (back) {{
                target.appendChild(back);
                back.classList.add('visible');
            }}
        }}
        function backToRisks(event) {{
            if (event) event.stopPropagation();  // don't toggle the fact card
            const back = document.getElementById('backToRisks');
            if (back) back.classList.remove('visible');
            const dest = riskReturnEl || document.getElementById('risksSection');
            if (dest) dest.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }}

        // ── Section Toggle ──
        function toggleSection(header) {{
            header.classList.toggle('open');
            const content = header.nextElementSibling;
            content.classList.toggle('open');
        }}
        
        // ── Expandable Fact Details (click anywhere on card) ──
        function toggleFactDetail(factEl, event) {{
            // Don't toggle if user clicked a link or button inside the card
            if (event && (event.target.tagName === 'A' || event.target.tagName === 'BUTTON')) return;
            factEl.classList.toggle('expanded');
        }}

        // ── Citation chips: "+N more sources" (class/textContent only — no
        //    innerHTML; chip hrefs are built server-side) ──
        function toggleCiteChips(btn) {{
            const wrap = btn.closest('.citation-chips');
            if (!wrap) return;
            const on = wrap.classList.toggle('cites-expanded');
            btn.textContent = on ? btn.dataset.less : btn.dataset.more;
        }}

        // ── "Show all N" for evidence quotes / trend signals ──
        function toggleShowAll(btn) {{
            const wrap = btn.parentElement;
            if (!wrap) return;
            const on = wrap.classList.toggle('expanded-all');
            btn.textContent = on ? btn.dataset.less : btn.dataset.more;
        }}
        
        // ── Fact Sorting (includes trend sort) ──
        function sortFacts(mode) {{
            document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
            if (event && event.target) event.target.classList.add('active');
            
            const container = document.getElementById('factsContainer');
            const facts = Array.from(container.querySelectorAll('.fact'));
            
            facts.sort((a, b) => {{
                if (mode === 'confidence-desc') {{
                    return parseFloat(b.dataset.confidence) - parseFloat(a.dataset.confidence);
                }} else if (mode === 'confidence-asc') {{
                    return parseFloat(a.dataset.confidence) - parseFloat(b.dataset.confidence);
                }} else if (mode === 'risk') {{
                    return parseInt(b.dataset.risk) - parseInt(a.dataset.risk);
                }} else if (mode === 'trend') {{
                    return parseInt(b.dataset.hasTrend || 0) - parseInt(a.dataset.hasTrend || 0);
                }} else {{
                    return parseInt(a.dataset.index) - parseInt(b.dataset.index);
                }}
            }});
            
            facts.forEach(f => container.appendChild(f));
            currentPage = 1;
            showPage(1);
        }}
        
        // ── Search ──
        function searchContent() {{
            const input = document.getElementById('searchInput').value.toLowerCase();
            const items = document.querySelectorAll('.fact, .connection, .risk');
            items.forEach(item => {{
                const match = item.textContent.toLowerCase().includes(input);
                item.style.display = match ? '' : 'none';
            }});
            if (!input) {{
                currentPage = 1;
                showPage(1);
            }}
        }}
        
        // ── Multi-Select Category Filter ──
        let activeCategories = new Set(['all']);
        
        function toggleCategory(btn, cat) {{
            if (cat === 'all') {{
                // "All" resets — deselect everything else, select All
                activeCategories.clear();
                activeCategories.add('all');
                document.querySelectorAll('.cat-tab').forEach(t => t.classList.remove('active'));
                btn.classList.add('active');
            }} else {{
                // Toggle specific category
                // First, deselect "All" if it was active
                const allBtn = document.querySelector('.cat-tab[data-cat="all"]');
                if (activeCategories.has('all')) {{
                    activeCategories.delete('all');
                    if (allBtn) allBtn.classList.remove('active');
                }}
                
                if (activeCategories.has(cat)) {{
                    // Deselect this category
                    activeCategories.delete(cat);
                    btn.classList.remove('active');
                    // If nothing selected, revert to All
                    if (activeCategories.size === 0) {{
                        activeCategories.add('all');
                        if (allBtn) allBtn.classList.add('active');
                    }}
                }} else {{
                    // Select this category (additive)
                    activeCategories.add(cat);
                    btn.classList.add('active');
                }}
            }}
            
            applyFilters();
        }}
        
        function applyFilters() {{
            const facts = document.querySelectorAll('.fact');
            let visibleIndex = 0;
            facts.forEach(f => {{
                const factCat = f.getAttribute('data-category');
                const matches = activeCategories.has('all') || activeCategories.has(factCat);
                if (matches) {{
                    visibleIndex++;
                    f.style.display = visibleIndex <= factsPerPage ? '' : 'none';
                }} else {{
                    f.style.display = 'none';
                }}
            }});
            currentPage = 1;
            updatePaginationForCategory();
        }}
        
        // ── Pagination ──
        let currentPage = 1;
        const factsPerPage = {FACTS_PER_PAGE};
        const totalFacts = {total_consol};
        const totalPages = {total_pages};
        
        function changePage(delta) {{
            const newPage = currentPage + delta;
            const maxPages = getMaxPages();
            if (newPage < 1 || newPage > maxPages) return;
            currentPage = newPage;
            showPage(currentPage);
        }}
        
        function getVisibleFacts() {{
            const facts = document.querySelectorAll('.fact');
            let visible = [];
            facts.forEach(f => {{
                const cat = f.getAttribute('data-category');
                if (activeCategories.has('all') || activeCategories.has(cat)) {{
                    visible.push(f);
                }}
            }});
            return visible;
        }}
        
        function getMaxPages() {{
            return Math.ceil(getVisibleFacts().length / factsPerPage);
        }}
        
        function showPage(page) {{
            const visibleFacts = getVisibleFacts();
            const start = (page - 1) * factsPerPage;
            const end = start + factsPerPage;
            
            // Hide all facts first
            document.querySelectorAll('.fact').forEach(f => f.style.display = 'none');
            
            // Show only visible facts in page range
            visibleFacts.forEach((f, i) => {{
                f.style.display = (i >= start && i < end) ? '' : 'none';
            }});
            
            const pageSpan = document.getElementById('currentPage');
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');
            const maxPages = Math.ceil(visibleFacts.length / factsPerPage);
            
            if (pageSpan) pageSpan.textContent = page;
            if (prevBtn) prevBtn.disabled = page <= 1;
            if (nextBtn) nextBtn.disabled = page >= maxPages;
            
            // Scroll to top of facts section on page change
            const factsSection = document.getElementById('factsSection');
            if (factsSection) {{
                factsSection.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }}
        }}
        
        function updatePaginationForCategory() {{
            const count = getVisibleFacts().length;
            const maxPages = Math.ceil(count / factsPerPage);
            const pageInfo = document.querySelector('.page-info');
            if (pageInfo) pageInfo.textContent = 'Page 1 of ' + maxPages + ' (' + count + ' facts)';
            const pageSpan = document.getElementById('currentPage');
            if (pageSpan) pageSpan.textContent = '1';
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');
            if (prevBtn) prevBtn.disabled = true;
            if (nextBtn) nextBtn.disabled = maxPages <= 1;
        }}
    </script>
</body>
</html>"""
    
    return html


def generate_html_report(
    result: Dict[str, Any],
    query: str,
    duration: float,
    output_path: Optional[str] = None
) -> str:
    """Render the report and write it to disk (CLI path).

    The API path uses render_html_report() directly and persists to the DB
    only (PHASE3_DESIGN §12.S3 — no report files from API jobs).

    Returns:
        Absolute path to the generated HTML file.
    """
    html = render_html_report(result, query, duration)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query)
        safe_query = safe_query.replace(' ', '_').strip('_')
        output_path = f"research_report_{safe_query}_{timestamp}.html"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    abs_path = os.path.abspath(output_path)
    logger.info(f"HTML report generated: {abs_path}")

    return abs_path
