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
