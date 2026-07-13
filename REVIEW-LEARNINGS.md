# REVIEW-LEARNINGS

Recurring gaps the plan/code reviewers keep catching. Read this before finalizing a
plan or a diff, and address every applicable item so it never reaches review.

## Persistence / shared state
- **A second writer to an overwrite-persisted field must share the full object.**
  When a field is persisted by full-object / full-column overwrite — e.g.
  `Job.progress` via `update(Job).values(progress=...)` (`src/api/jobs.py:183-186`) —
  any *additional* writer MUST mutate and persist the **same full object**, never a
  partial dict. Two writers each persisting their own partial view silently clobber
  each other's keys (here: an `activity_callback` would wipe `phase`/`iteration`, and
  vice-versa). Fix pattern: one shared mutable state dict owned by the job runner;
  all callbacks mutate it and persist it whole.
  *Origin: 2026-07-09 plan review, loading-screen Step A2 dual-callback hazard.*

## Data lifecycle / retention
- **When a change puts a new CLASS of data (PII, scraped content, secrets) into an
  existing field, audit every lifecycle and exposure handler of that field — not
  just the read/write path being added.** Concretely here: retention/purge paths
  (`reaper.py` expiry purge, startup sweep), terminal-state writes, API responses
  that echo the field, SSE payloads, and logs. A purge list is an allowlist frozen
  at design time; it goes silently stale the moment the field's data class changes.
  Verify the fix on the PRODUCTION dialect, not just sqlite: `Job.progress` is
  `nullable=False`, so `progress=None` only works because SQLAlchemy JSON stores
  Python None as JSON `null`, not SQL NULL — proven against local Postgres.
  *Origin: 2026-07-09 Phase A security-owasp pass — activity/sample_facts made
  `Job.progress` PII-bearing; the §12.S2 expiry purge didn't cover it (MED, fixed).*

## New public endpoints
- **Every NEW public HTML endpoint must state its full header/indexability contract
  at design time**: CSP, `X-Content-Type-Options: nosniff`, and robots policy
  (`X-Robots-Tag` header — do not assume a meta tag exists in the served HTML).
  "No new security surface" claims must enumerate new routes explicitly; a static
  page served publicly IS a surface. Also verify the *content's* vintage: HTML
  generated before an escaping fix is a stored-XSS artifact, not an asset.
  *Origin: 2026-07-10 Phase D review — `/sample-report` had no header spec and the
  Feb sample predated the escaping chokepoint (144 unescaped handler occurrences).*

## Escaping chokepoints vs. features that need the raw value
- **When a render pipeline escapes ALL data at one chokepoint, any feature that
  needs the raw value (URL/percent-encoding, `#:~:text=` text-fragment anchors,
  hashing, signature checks) MUST read from a raw copy captured BEFORE the
  chokepoint — never re-derive from the escaped copy.** Concretely here:
  `html_report.py:550 result = _escape_deep(result)` HTML-escapes every string;
  building a citation `href` or a text-fragment from the post-escape value
  double-encodes (`&`→`&amp;`→`%26amp%3B`) and silently breaks the link. Two
  corollaries: (a) HTML-escaping is NOT scheme validation — a `javascript:` URL
  survives it as a live XSS `href`, so allowlist http/https separately; (b)
  percent-encode ONLY the fragment/query part that needs it (`quote(text,
  safe='')`), never the whole URL (that encodes `://?&` and breaks every link) —
  HTML-escape the base URL for attribute context instead.
  *Origin: 2026-07-10 Phase B citations review (M1/M2/N1) — `_escape_deep` chokepoint.*

## Schema changes / migrations
- **A plan that adds a column/field to an existing ORM model must state the
  migration mechanism.** This repo initializes its schema with
  `Base.metadata.create_all` (`src/api/db.py` — "additive greenfield schema — no
  alembic"), which creates missing TABLES but silently never ALTERs an existing
  one: a new mapped column is absent on the already-created production table, and
  the first ORM query on that model fails ("column does not exist") → app-wide
  outage. Either avoid the column (in-memory state is valid under the pinned
  `--workers 1`, `Dockerfile:22` — say so and bind it to that invariant with a
  code comment) or write the explicit `ALTER TABLE … ADD COLUMN` step into the
  deploy runbook.
  *Origin: 2026-07-11 Phase C plan review R1 — `Job.cancel_requested` column had
  no migration path; fixed as an in-memory cancel registry.*

## Classification fields vs pre-split merges
- **When a plan adds a per-item classification field (relevance, about_target,
  spam, language) that gates downstream inclusion, check every MERGE/DEDUP/
  CROSS-REFERENCE step that runs BEFORE the split** — a similarity merge that
  unions provenance and drops a duplicate will silently move an item across
  the classification boundary (absorbing a keep into the discard pool, or
  contaminating a keep's provenance with a discard's sources). Pin the rule:
  never merge/corroborate across the classifying field, or partition before
  the first merge. The field is only "automatic scope" downstream of the
  split, never upstream of it.
  *Origin: 2026-07-12 Phase C1.7 plan review R1 — `about_target` set at
  extraction, but the extractor's `_deduplicate_facts`/`_find_similar_facts`
  run pre-partition and are boundary-blind.*

## Transport dicts vs model schemas
- **When a consumer reads a field from a merged/transport dict (the
  orchestrator `context`, `progress_state`, `report_preview`), verify the
  shape against the PRODUCER's construction site, not the request-model
  schema** — the same field name can exist in both shapes:
  `EntitySelection.disambiguators` is `List[str]` in the model, but routes.py
  stores `context["disambiguators"] = "; ".join(...)` — a STRING (deliberate:
  context values feed `"; ".join(f"{k}: {v}")` prompt lines). Indexing or
  iterating the schema shape on the transport shape fails SILENTLY —
  `s[0]` is a character, iteration yields characters — a wrong-but-running
  bug (e.g. gap queries like `"John Smith" N wealth assets…`), never an error.
  *Origin: 2026-07-12 C1.7 builder checkpoint, human catch F1 — the gap-query
  scope term assumed the model's list shape.*

## Hostnames / origins
- **When a plan changes an origin/hostname, grep for ALL server-side exact-match
  host/origin checks — not just the client-facing one.** A hostname change (custom
  domain, new frontend origin) must update every server-side pin, e.g. the Turnstile
  widget allowlist AND the single-string `TURNSTILE_EXPECTED_HOSTNAME` in
  `src/api/security.py`, AND `CORS_ORIGINS` (`config/settings.py` → `main.py`). Miss
  the server-side pin and every submission 4xx's server-side even with the client
  fixed.
  *Origin: 2026-07-09 custom-domain research review.*
</content>
