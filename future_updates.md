# Future updates — deferred by explicit decision

Everything here was discussed and consciously deferred, with the reason and
the trigger for picking it up. Ordered by priority. Phase-level sequencing
(D → B → E1 → C1+C2 → E2) lives in PLAN.md; this list is the quality/UX
backlog that slots between phases.

## P0 — Dedicated run page (TOP PRIORITY — NEXT SCOPE, human decision 2026-07-12)
**What:** move an active run to its own URL (`/research/{job_id}` page
reusing the same template/JS family) — homepage stays marketing; a run
becomes bookmarkable/refresh-safe (today a refresh loses the panel while the
job keeps running server-side); the picker→run transition gets a clean
stage. Vanilla JS remains the right call at this view count; **the
React/SPA trigger stays "the product becomes genuinely multi-view"**
(dashboard/history/KG explorer, Phase E) — rewriting the tested safe-DOM
layer for two views buys no correctness. **Trigger:** FIRST scope after
C1.5/C1.6 (finish-early + HITL hardening) ships — explicitly promoted ahead
of P1 by the human.

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
- **Thin-subject GRADE semantics** (human 2026-07-13): today a low-footprint
  subject gets an F that reads as a judgment of the person. The C3-minimal
  framing shipped 2026-07-13 (depth caption + "limited public footprint"
  note + depth-neutral tier words) is render-side only; the P1-scope
  question is whether an exhausted-search thin subject should get a
  DIFFERENT grade treatment entirely (e.g. "Limited data" badge in place of
  a letter, or an exhaustiveness-adjusted score) — changes grade semantics
  globally, so it belongs in this bundle.
- **Public-figure fact curation** (human 2026-07-13): high-footprint
  subjects yield 100+ facts; today display = confidence-sorted + consolidated
  + paginated, which is inclusion-by-volume, not editorial selection. P1
  scope: a materiality/recency/theme-coverage ranking for which facts LEAD
  the report (nothing hidden — ordering and grouping, not omission; omission
  would be silent editorializing on a due-diligence product).
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

## P8 — Research-log / progress-messaging overhaul (human-directed 2026-07-12)
**What:** a substantially better way of narrating what the agent is doing and
what it has found while the report generates — today's narrative log rows +
strip line are serviceable but terse; candidates: richer per-iteration
"chapter" summaries, discovered-so-far digests at phase boundaries, clearer
signposting of what remains, and surfacing WHY a query was run (the purpose
field already exists on SearchQuery). **Why deferred:** pure presentation —
no correctness stake; touches the A.2/A.3 narrative layer which is stable and
test-covered, so it deserves its own pass. **Trigger:** UI-polish pass after
the Phase C HITL flow settles (bundle with P3).

## P9 — Mid-run conflation oracle + "refocus" flow (D4 successor)
**What:** detect DURING the run that facts are clustering into multiple
distinct identities (the risk assessor already catches this at the END — a
real John Smith run produced a CRITICAL conflation flag). On detection,
surface the identities discovered so far as profile cards mid-run; picking
one CANCELS the current job and immediately starts a fresh SCOPED run
(compose the existing cancel + entity-scoped submit — NO awaiting-selection
job state, no pause machinery). **Why deferred:** detection needs a cheap
reliable mid-run signal (D4 citations-as-oracle was already deferred);
pre-flight hardening (Phase C1.6) removes most of the need by never starting
a broad run without explicit consent. **Trigger:** evidence that scoped runs
still drift, or D4 gets built for post-hoc QA anyway.

## P10 — "Search everyone" grouped multi-entity report
**What:** when a user explicitly chooses to research ALL same-name entities,
produce a report grouped per identity (facts/risks/connections sectioned by
entity) instead of one mixed pool. **Why deferred:** report-format overhaul +
N× spend/cap math ("research all" was CUT from C1 v1); the honest interim is
the explicit broad-search confirmation + the conflation risk flag the scorer
already emits. **Trigger:** real demand for multi-entity reports; requires P1
(entity-type scoring) and the C1 entity_id primitive (done).

## P12 — Warm-start refinement (fact/search carryover between runs)
**What:** when a user refines and re-runs, seed run B with run A's
still-relevant material — the user's proposal: give the agent (1) the
original input, (2) the refined details, (3) which previous findings match
the new details as a seed, (4) negative scope for the rest. Item (4) ships
NOW as C1.7d (`rejected_entities` → "NOT these" prompt line); items (1)–(3)
are this deferred item. **Why deferred (decision 2026-07-12):** seeding run
B with run A's facts risks carrying the exact cross-identity contamination
C1.7 removes; provenance, re-verification semantics, score honesty, and
cross-job PII lifecycle (run A's purged data flowing into run B) all get
materially harder — to save <$0.35 and ~5 minutes per re-run. **Triggers:**
paid/billed re-runs, or P9's refocus flow where run A's SIDELINED pool is
provably run B's target (the one clean seed: already entity-attributed,
citations intact). Related: the post-finalization "add this sidelined fact
back" action needs a RE-FINALIZATION pipeline (re-run verify/risks/trends
on the amended pool) — same machinery, bundle together.

## P13 — "Continue research" / deeper-iterations control (human 2026-07-13)
**What:** the inverse of finish-early — on a completed or stagnation-stopped
run, a control to push DEEPER: more iterations, and for thin subjects a
deeper-search mode (scrape more result pages per query, alternate source
classes — registries, filings, local news archives — beyond the popularity-
ranked first page). Pairs with honest stopping-point transparency: the
report/run page should say WHY the run stopped (stagnation vs coverage vs
max-iterations vs user finish). **Why deferred:** today's stagnation stop is
the honest exhaustion point for the CURRENT search strategy (C1.7 gate-9:
iteration 4 yielded 0 target facts from 50 results — more iterations of the
same queries find nothing new); going deeper requires new search-strategy
work (deeper scraping = P6's latency tradeoff; new source classes = new
executor surface), and a continue control needs the run page (P0) and
re-finalization semantics (P12's re-run tail). **Trigger:** P0 run page
ships; bundle the deeper-search mode with P6.

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

## P14 — Pre-flight same-person merge guard (human 2026-07-21)
**What:** deterministic post-clustering merge pass in `preflight.py`: after
the LLM returns candidates, merge two clusters when the NAME is a variant
(token subset + small nickname map: Tim→Timothy, Bill→William) AND the
descriptors name the same organization and/or the domain evidence overlaps —
union domains, keep the richer descriptor. **Why:** observed live (Tim Cook
run, 2026-07-21): "Timothy Donald Cook — CEO, Apple" and "Tim Cook —
Executive Chairman, Apple" rendered as two picker cards — the clustering
prompt treats ROLE as a disambiguating attribute, but role is time-varying
(CEO→Executive Chairman is the classic same-person case); org + name-variant
is the stable signal. Nondeterministic (a retry merged them). Severity low —
either card scopes to the right person; cost is one confused click + diluted
domain-mass share (can force an unnecessary picker instead of auto-proceed).
**Why deferred:** preflight (Phase C) code, not P0; the conservative split
is the designed safe direction (false merge contaminates a report — keep the
merge pass conservative: no evidence overlap → no merge). **Trigger:** next
preflight-quality pass; small standalone PR with table tests over name-variant
× org × domain-overlap combinations.

## P15 — Run-page completed-state fidelity (terminal summary counts)
**What:** `jobs.py` terminal `final_progress` gains cheap aggregate keys —
`risks` (count + severities), `connections` (count), `by_category`,
`verification` counts — so a completed run page (cold-load OR the live
completed handler's snapshot re-fetch) renders real counters/bars instead of
"–" and zeroed bars. **Why:** observed live 2026-07-21: the terminal dict
REPLACES the running progress (jobs.py:411-414, the documented D5 gotcha),
and the SSE poll races completion (the loop checks status FIRST, so progress
written in the last poll interval before terminal is never streamed) — a
completed page keeps only {facts, quality_score, grade}. The full data is
persisted (report_json/report_html) — this is a transport-summary gap, not
data loss. Counts are low-sensitivity aggregates; the column is §12.S2-purged
at expiry either way. **Why deferred from P0:** P0 was UI-only; this touches
the jobs.py writer. **Trigger:** FIRST follow-up PR after P0 merges (pairs
with the client already rendering these keys when present).
