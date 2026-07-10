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
