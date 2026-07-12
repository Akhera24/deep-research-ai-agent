# Future updates — deferred by explicit decision

Everything here was discussed and consciously deferred, with the reason and
the trigger for picking it up. Ordered by priority. Phase-level sequencing
(D → B → E1 → C1+C2 → E2) lives in PLAN.md; this list is the quality/UX
backlog that slots between phases.

## P1 — Scoring redesign bundle (one pass, needs plan review first)
**Why bundled:** every item below shifts confidence numbers or coverage math,
which shifts quality scores and grades globally. Doing them separately would
invalidate the quality-parity baseline (95.8–96.5 Grade A band) repeatedly;
one pass = one re-baseline. **Why deferred:** grade changes are
product-facing; the operating contract requires an independent plan review
before building (same treatment as PLAN.md Rev 3.5 gave Phase B).
- **Per-fact confidence weighting.** Today: `confidence = LLM × 0.8 +
  batch-AVERAGE reliability × 0.2` (`extractor.py::_post_process_facts`) —
  pre-B0 legacy. Post-B0 each fact carries its own `source_reliabilities`;
  weight by the fact's OWN cited sources (SEC-filing fact ≠ blog fact).
- **Entity-type-aware scoring** (D7 audit verdict: companies are NOT scored
  correctly — person-shaped 6-category breadth structurally penalizes them;
  Stripe biographical stays 0.0 even at 10 iterations). Subject
  classification at strategy-planning → per-type category schemas.
- **"Searched-but-empty counts as covered"** (D7 audit: clean subjects are
  penalized — coverage derives from facts found, so a searched-and-clean
  category ≡ never-searched).
- **"Verified" semantics upgrade:** require distinct DOMAINS, not distinct
  extractions (today two pages syndicating one wire story count as
  cross-verification). Decided 2026-07-11 while reviewing verified mechanics.
- **Re-baseline the parity band** on fixed fixtures after the above.

## P2 — Source freshness / publish-date extraction
**What:** extract publish dates from page metadata (og:published_time,
JSON-LD, URL date patterns); surface per-source age and feed "freshness"
into the scoring pass; flag stale sources on chips. **Why deferred:** new
extraction surface with real failure modes (missing/lying metadata);
today's only recency signal is a `202[4-6]` regex on fact text (trend
badges) and `fetched_at` is crawl time, not publish time. **Trigger:** build
INTO or immediately after P1 so freshness enters scoring once.

## P3 — Evidence-panel / hover-card redesign
**What:** styled instant hover-cards (replacing the subtle native `title`
tooltips — human found them non-intuitive, 2026-07-11), richer quote rows,
possible per-quote snippets of surrounding page context, and a
"jump to fact #N" cross-link from risk cards (needs pagination-aware
scroll + flash). **Why deferred:** hover excludes touch (mobile/LinkedIn
audience), custom tooltip DOM is a new injection surface to harden, and the
current visible quote→source links already deliver the proof interaction.
**Trigger:** dedicated UI-polish pass once the feature set stabilizes.

## P4 — Predictive / forward-looking risk analysis
**What:** "potential future risks" — trends that are leading toward a risk
(e.g. escalating litigation trajectory), rendered as a distinct
"emerging risk signals" treatment. **Why deferred:** (1) it's a new LLM
prompt surface (the risk prompt is deliberately frozen — OQ-B2 DEFERRED);
(2) forward-looking claims on a paid due-diligence product are a
trust/liability surface (same class as the banned "truth %" label) — any
predictive output must be framed as *cited trajectory signals*, never
predictions. The existing "Risk Trajectory" trend group + per-risk
`Trend: Established/Isolated` field already cover the honest render-side
version. **Trigger:** revisit alongside the risk-prompt Fact.id upgrade
(OQ-B2) so the prompt is touched once.

## P5 — B5: connections citations (knowledge-graph treatment)
From PLAN.md (human direction 2026-07-11): connections stay uncited in
Phase B; cite them via the Phase E1 radial-graph treatment (edge hover =
evidence quote), not flat chips. **Trigger:** Phase E1.

## P6 — Anchor coverage boost (scrape more sources)
**What:** 43% of chip URLs have no page-verified highlight anchor, mostly
snippet-only results (no scraped content to verify against). Scraping
content for more results raises highlight coverage. **Why deferred:** real
latency/network cost per run — the only lever with a genuine runtime
tradeoff. **Trigger:** measure user impact of missing highlights first;
bundle with any scraping-pipeline work.

## P7 — Small operational items
- **Raise `max_tokens` on the connections call** (`workflow.py`, currently
  4000): it truncated on 3/3 gate runs; the JSON repair recovers, but
  headroom is cheaper than repair. Cheap; do with the next workflow touch.
- **Fallback debug log** includes `str(source_ids)[:80]` of model output
  (security-owasp LOW, tracked): switch to type+count only.
- **Chip tooltip copy/UX**: superseded by P3 if hover-cards land.

## End-of-featureset checklist (not future work — the agreed final sync)
1. Regenerate BOTH sample reports (person + company) from fresh runs.
2. Sync homepage entity-card chips + hero demo numbers to the served
   samples; apply the honest-copy upgrade ("every fact cited").
3. Update `docs/example_report.html`; re-shoot homepage screenshots if the
   report visuals changed.
4. Full gate re-run: suites, sink checks, header contracts, quality band.
