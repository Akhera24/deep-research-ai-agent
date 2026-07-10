# Plan: Restore Deep-Research Agent to Functional State, Modernize Dependencies, and Productionize Hosting

Date: 2026-07-06 · **Rev 2** (all findings from the independent plan review folded in; MAJOR #1 re-verified live — see changelog at bottom)
Research files: `research/live-verification-2026-07-06.md`, `research/model-migration-2026-07.md`, `research/deployment-options-2026-07.md`

---

## Goal / question

1. Identify every deprecated/retired internal dependency (LLMs, LangGraph, SDKs, Brave/Serper) — DONE, verified live.
2. Restore the agent to its prior functional state (CLI generates reports again).
3. Decide and plan the best production hosting so anyone can submit a person/company and get a full iterated-search report.
4. Defer UI/UX polish, but design so a text-box frontend, rate limiting, and backoff slot in cleanly later.

## Context (what exists today — all VERIFIED live on 2026-07-06 unless marked)

- **2 of 3 configured models are dead at the API.** Live 1-token probes with the project's own keys:
  - `claude-opus-4-20250514` → **404** (retired 2026-06-15). Strategy planning, query refinement, risk assessment, connection mapping all lose their primary model.
  - `gpt-4-turbo-preview` → **404 on this account today** (docs say shutdown 2026-10-23; live result wins). Structured-output/verification fallback dead.
  - `gemini-2.5-flash` → **200 OK**, but scheduled shutdown **2026-10-16**. Extraction works today, dies in 3 months.
  - Precise failure mode (review finding #7): `router.py:304-363` falls back on any exception, so reasoning tasks degrade Claude(404) → OpenAI(404) → **Gemini-only**, with doubled latency and degraded analysis quality — silent degradation, not a clean hard failure. Same urgency, worse observability.
- **No new API keys are needed.** Every replacement model (`claude-opus-4-8`, `claude-sonnet-5`, `gpt-5.4-mini`, `gemini-3.1-flash-lite`) was successfully called live with the project's *existing* keys — model access is account-level, not key-level.
- **Search layer is healthy**: Brave 200 OK, Serper 200 OK. Endpoints unchanged. No Brave/Serper work needed.
- **Installed libraries are already current and compatible**: langgraph 1.0.9 (workflow imports cleanly — executed the import), anthropic 0.83.0, openai 2.21.0, google-genai 1.64.0, pydantic 2.12.5, Python 3.12. LangGraph needs **no migration work**.
- **Architecture is migration-friendly**: `router.py` routes by provider enum; model IDs come from `config/settings.py` → the fix is settings + 2 small client patches, not a rework.
- Dead weight: `src/database/` imported by nothing; `DATABASE_URL` is a *required* setting anyway; `src/api/` is empty despite FastAPI dep; legacy `google-generativeai` SDK is EOL (support ended 2025-11-30) but still in requirements.
- **Three live-verified gotchas that a plain model-ID swap would miss:**
  1. `claude-opus-4-8` and `claude-sonnet-5` **reject `temperature` with 400** ("temperature is deprecated for this model") — and `claude_client.py:156-171` always sends it (default 0.3).
  2. `gpt-5.4-mini` **rejects `max_tokens` with 400** ("use `max_completion_tokens` instead") — and `openai_client.py:161` sends `max_tokens`. Verified live; also verified: `temperature=0.3` IS accepted by `gpt-5.4-mini` and `gemini-3.1-flash-lite` (with `max_completion_tokens`), so only the param *name* changes on the OpenAI path.
  3. `tiktoken.encoding_for_model()` **KeyErrors on `gpt-5.4-*`** IDs; `openai_client.py:91-94` would silently fall back to the wrong encoding (`cl100k_base`); gpt-5.x uses `o200k_base`.
- Bonus finding (review #14): settings' Gemini pricing ($0.15/$0.60, lines 120-127) never matched even the *current* market price ($0.30/$2.50) — cost tracking has been under-reporting Gemini spend all along. Step 1 fixes this as a side effect.
- Constraint (repo git state): working tree clean, no builder work in flight — no conflict risk.

## Research summary (approaches considered, with evidence)

**Model replacements** (full pricing/citations in `research/model-migration-2026-07.md`):

| Role | Old (dead/dying) | Recommended | Rejected alternative |
|---|---|---|---|
| Reasoning (strategy, risk, connections, analysis) | claude-opus-4-20250514 ($15/$75) | **claude-opus-4-8** ($5/$25) — Anthropic's official replacement, 3× cheaper | `claude-sonnet-5` ($2/$10 intro) — keep as documented cost lever via env var; not default so report quality is preserved at parity first |
| Extraction / document processing | gemini-2.5-flash ($0.30/$2.50, dies 10-16) | **gemini-3.1-flash-lite** ($0.25/$1.50) — GA, cheaper than today | `gemini-3.5-flash` ($1.50/$9) — 6× the input cost; only needed if flash-lite context proves too small (verify in Step 6) |
| Structured output / verification / fallback | gpt-4-turbo-preview ($10/$30) | **gpt-5.4-mini** ($0.75/$4.50, 400K ctx, JSON mode) | `gpt-5.4-nano` ($0.20/$1.25) — a fallback path should favor reliability over ~$0.55/1M savings |

Net effect: per-report model cost **drops** (~$0.56 → est. $0.15–0.25/report). ASSUMPTION on exact figure; verify with the cost tracker in Step 7.

**Hosting** (full matrix in `research/deployment-options-2026-07.md`):
- **Vercel**: Hobby hard-caps functions at 300s → a 5–10 min job cannot run there. Pro's 800s works but a 10-min synchronous HTTP request is fragile, and Vercel's durable Python Workflows SDK is still beta. Verdict: fine for a future frontend, wrong for the job.
- **AWS**: App Runner is closed to new customers (2026-04-30). Lambda fits the 15-min cap at near-$0 but costs the most ops plumbing. Fargate has no limits but highest complexity.
- **Railway / Fly / Render**: all handle long-running FastAPI + SSE natively at $7–21/mo.
- **LangSmith Deployment Plus ($39/mo)**: purpose-built for LangGraph background runs with checkpointing/streaming out of the box, but 3–4× the cost and lock-in.
- **Recommended: Railway** — single FastAPI service + managed Postgres (~$10–15/mo), research jobs in-process via `asyncio` + a Postgres job table, SSE progress with heartbeats (Railway caps HTTP/SSE at 15 min — a 10-min job with heartbeats fits), slowapi rate limiting.

## Recommendation — numbered, executable steps

### Phase 1 — Restore functional state (blocks everything; ~half a day)

**Step 1. Update model IDs + pricing in `config/settings.py`.**
- WHAT: `CLAUDE_MODEL` → `claude-opus-4-8` (line 80); `GEMINI_MODEL` → `gemini-3.1-flash-lite` (107); `OPENAI_MODEL` → `gpt-5.4-mini` (133). Pricing fields: Claude $5/$25 (94-101), Gemini $0.25/$1.50 (120-127), OpenAI $0.75/$4.50 (146-153). Update docstring examples (lines 25, 465) and the "as of Jan 2026" comment. Fix `.env.example` while there (review #10): its override names are **wrong** (`ANTHROPIC_MODEL`/`GOOGLE_MODEL` at lines 18-19 vs settings' actual `CLAUDE_MODEL`/`GEMINI_MODEL`) and it omits the required `DATABASE_URL` — a fresh clone fails before models even matter. Check local `.env` for stale overrides.
- WHY: 2 of 3 configured models 404 at the API; this is the root cause of the project being non-functional.
- ALTERNATIVE rejected: pinning date-suffixed snapshots — Anthropic 4.6+ IDs are dateless pinned snapshots; appending dates breaks them.
- Skills: none required (config edit). **Gap flagged:** no installed skill covers LLM-migration checklists.

**Step 2. Stop sending `temperature` from the Claude client.**
- WHAT: In `src/models/claude_client.py` (~lines 156-171), drop `temperature` (and any `top_p`/`top_k`) from the `messages.create()` kwargs. Keep `CLAUDE_TEMPERATURE` in settings as deprecated/no-op or delete it and its base-config plumbing (`base_client.py:84,299` — check Gemini/OpenAI still get theirs; only Claude must stop).
- WHY: VERIFIED live — `claude-opus-4-8` and `claude-sonnet-5` return **400** when temperature is set. Without this, Step 1 swaps one hard failure for another.
- ALTERNATIVE rejected: catch-400-and-retry-without-temperature — hides misconfiguration, adds a failed round-trip to every call.
- Skills: none.

**Step 3. Fix the OpenAI client: request params AND tokenizer (`src/models/openai_client.py`).**
- WHAT: (a) Line 161 — change `max_tokens` → `max_completion_tokens` in `chat.completions.create()`. (b) Lines 91-94 — for `gpt-5`/`gpt-4.1`/`gpt-4o` family names use `tiktoken.get_encoding("o200k_base")`; keep `cl100k_base` fallback for unknowns. (c) Leave `temperature=0.3` as-is — VERIFIED live that `gpt-5.4-mini` accepts it (and so does `gemini-3.1-flash-lite`); only Claude needs the Step 2 removal.
- WHY: VERIFIED live — `gpt-5.4-mini` returns **400 "Unsupported parameter: 'max_tokens'... Use 'max_completion_tokens'"**. Same class of silent breakage as the Claude temperature gotcha; without (a), the fallback path stays broken after the model swap. (b): `encoding_for_model("gpt-5.4-mini")` raises KeyError (tiktoken prefix table has `gpt-5-`, not `gpt-5.`) — VERIFIED from tiktoken source; current code silently uses the wrong encoding, skewing token counts and cost tracking.
- ALTERNATIVE rejected: sending both params or try/catch-retry — `max_completion_tokens` is backward-compatible across the current lineup and one clean param beats a failed round-trip per call. For (b): relying on the `cl100k_base` fallback — doesn't crash, but every count and cost estimate is wrong with no signal.
- Skills: none.

**Step 4. Update router cost comments and `scripts/check_requirements.py` heuristics.**
- WHAT: `src/models/router.py:150-224` `expected_cost_per_1k` values + "Claude Opus 4 / GPT-4 / Gemini 2.5" reasoning strings; `scripts/check_requirements.py:110-112` model-name checks.
- WHY: These drive routing expectations and the self-check script; stale values misreport costs and the checker would flag the new (correct) models as wrong.
- ALTERNATIVE rejected: leaving comments stale — cheap now, misleads every future debugging session.
- Skills: none.

**Step 5. Remove the EOL `google-generativeai` SDK.**
- WHAT: Delete from `requirements.txt` and `pip uninstall` from venv; remove the legacy fallback branch in `src/models/gemini_client.py:44-56` (and the `GENAI_AVAILABLE` dual-path) so only `google-genai` remains.
- WHY: Repo archived; all support ended 2025-11-30 (VERIFIED). The dual-SDK path is dead code that can silently select an unsupported SDK.
- ALTERNATIVE rejected: keeping the fallback "for safety" — the fallback target is the unsupported thing; that's inverted safety.
- Skills: none.
- **DEVIATION D1 (executed 2026-07-07):** requirements.txt pinned ONLY the dead SDK — `google-genai` (the one the client actually uses) was never listed; a fresh install would have had no working Gemini path. Fixed by adding `google-genai>=1.64` alongside the removal.

**Step 6. Verify functional state (evidence, not assertion).**
- WHAT: (a) 1-token probes for all three new model IDs through each client class (not raw SDKs) to prove the temperature/tokenizer fixes; (b) run `python scripts/research.py "Jensen Huang" --save --html --iterations 3` end-to-end; (c) confirm HTML report renders and quality score computes; (d) run `pytest tests/`; (e) send one large extraction payload through `gemini-3.1-flash-lite` to validate its context window (its limit is an ASSUMPTION — if it truncates, flip `GEMINI_MODEL` to `gemini-3.5-flash`, a pure env change).
- WHY: The global harness requires demonstrated runs; also the only way to catch the next silent deprecation.
- ALTERNATIVE rejected: unit tests only — the breakage was at the live-API boundary, which mocks cannot catch.
- Skills: **Gap flagged** — no `testing-edge-cases` skill is installed (referenced by this workflow's conventions); builder should manually cover the edge-case list below. *(Resolved: skill installed 2026-07-06 and applied.)*
- **DEVIATION D2 (executed 2026-07-07):** `pytest tests/` had never worked — `simple_test.py`/`test_imports.py` called `sys.exit()` at module import, aborting collection. Fixed (main-guard / pytest rewrite). Also surfaced psycopg2 pinned-but-not-installed venv drift (resolved by Step 9 rebuild).
- **DEVIATION D3 (2026-07-07, open for human):** `test_quality_score.py` has 8/21 pre-existing scoring-curve expectation failures in code untouched by the migration; runs script-only (excluded from pytest). Decide later: recalibrate weights vs expectations vs defer to Phase 4.
- **6(e) RESULT (2026-07-07):** gemini-3.1-flash-lite handled a ~299K-token payload with exact needle retrieval (plan assumed ~1M ctx; ~300K verified usable) — NO flip to gemini-3.5-flash needed.

**Step 7. Re-baseline cost + README/docs.**
- WHAT: Capture actual per-report cost from the run in Step 6; update README model tables, architecture diagram labels, cost breakdown, and `docs/` references to old models.
- WHY: README currently advertises retired models; anyone cloning gets a broken quickstart.
- ALTERNATIVE rejected: doc updates "later" — docs are the product's front door and drift is the failure mode that caused this task.
- Skills: none.

### Phase 2 — Dependency & structure hygiene (fast follow; ~2 hours)

**Step 8. Default `DATABASE_URL` to `sqlite:///research.db`.**
- WHAT: `config/settings.py:73` — give it the sqlite default. **Also** remove `DATABASE_URL` from the hard-required list in `validate_settings()` (`config/settings.py:487-493`) — review finding #3: that validator independently raises on import, so changing the field alone does NOT fix startup.
- WHY: Nothing in the runtime path imports `src/database/` (only `tests/test_imports.py:32-46` touches it), yet a missing `DATABASE_URL` fails startup — this breaks every deployment config for no benefit. Keep the DB package: Phase 3 wires it for job state.
- ALTERNATIVE rejected: `Optional[str]` — incomplete (validator still raises) and pushes None-handling onto future DB code; a working sqlite default is strictly better. Deleting `src/database/` — Phase 3 needs a Postgres job table within a week.

**Step 9. Prune, pin, and add to `requirements.txt`.**
- WHAT: Remove unused deps: `playwright`, `redis`, `alembic`, `ratelimit` (line 70), and — review finding #5 — the four dead LangChain pins `langchain`, `langchain-anthropic`, `langchain-openai`, `langchain-google-genai` (lines 24-28): **nothing imports them; only `langgraph` is used** (`langchain-core` survives as langgraph's transitive dep). Optional-ize `sentry-sdk`. Keep `psycopg2-binary` for Phase 3. **Add** the Phase 3 deps here so the file is complete before the deploy work starts (review #6): `slowapi` (rate limiting), `bleach` (HTML sanitization), `sse-starlette` (SSE), and promote `markupsafe` to a direct pin (it becomes load-bearing in Step 13). Repin direct deps from the working venv.
- WHY: Container image size, cold-start, audit surface; the file's own comments admit drift ("you have 0.75.0 — works fine"); and a prune step that misses five dead pins fails its own purpose.
- ALTERNATIVE rejected: full `pip freeze` dump — pins 100+ transitive deps and makes upgrades noisy; pin direct deps only. Keeping langchain-* "since langgraph is related" — verified independent; dead weight.
- **DEVIATION (executed 2026-07-07):** two additions to the lists — **`tiktoken>=0.12` added as a direct pin** (unguarded import in claude_client.py/openai_client.py; its only installer was `langchain-openai`, which this step removes — without the pin the fresh-venv import fails) and **`lxml` removed** (verified dead: not installed in the working venv, imported nowhere). Venv was fully rebuilt from the new file, not uninstall-patched.

### Phase 3 — Production hosting on Railway (the deploy; ~2–3 days)

**Step 10. Build the FastAPI job API in the empty `src/api/`.**
- WHAT: `POST /api/research` (validates query, creates row in a Postgres `jobs` table, launches the LangGraph run via `asyncio.create_task`, returns `job_id` immediately) · `GET /api/research/{id}` (status + progress) · `GET /api/research/{id}/events` (SSE progress stream, heartbeat every ≤4 min — Railway requires ≤5) · `GET /api/research/{id}/report` (final HTML from DB) · `GET /healthz`. Reuse `ResearchOrchestrator` (`src/core/workflow.py:57`) and refactor `scripts/research.py::generate_html_report` into an importable module (it's currently trapped in a 2,258-line CLI script) so CLI and API share one report path.
  **Hard requirements added after review (MAJOR #2 + finding #8):**
  - **Job IDs MUST be unguessable** — UUIDv4 (or 128-bit URL-safe tokens), never serial integers. This service generates dossiers on real people; enumerable IDs turn the report store into a browsable database of everyone ever searched.
  - **Report retention/expiry**: reports expire after a configured window (env var, e.g. 7 days); expired reports 410. Access is by possession of the unguessable ID only until/unless auth is added in Phase 4.
  - **Run with exactly ONE uvicorn worker (`workers=1`)** — in-process `asyncio` jobs, the global concurrency semaphore, and SSE affinity all break silently under a default multi-worker CMD. Document this constraint in the Dockerfile next to the CMD; revisit only when moving to an external worker (or LangSmith Deployment).
- WHY: The workload is async I/O — one process with an in-process task + Postgres-backed job state is the standard minimal pattern; no Celery/Redis needed at this scale.
- ALTERNATIVE rejected: Celery/Redis worker tier — real infra cost and complexity for a job that is 99% `await`; revisit only if CPU-bound work or horizontal scale appears.
- Skills: **Gap flagged** — no `security-owasp` skill installed; apply the Security-surface section below manually.

**Step 11. Rate limiting, backoff, cost guardrails, and CAPTCHA gate.**
- WHAT (budget decision resolved 2026-07-06 — $40/mo all-in, **no hard daily cap**):
  - `slowapi` per-IP limits: 3 reports/hour/IP, burst 1.
  - Global concurrent-jobs cap (3) via semaphore.
  - Per-job budget abort at $1.00 using the existing cost tracker.
  - **Monthly hard cap: $40** (env var, tracked in a Postgres spend ledger) — when reached, `POST /api/research` returns 503 with a friendly "budget exhausted" message.
  - **Daily spend: soft alert only, NO hard cap** (human decision: usage legitimately spikes on test/demo days). Log a warning + optional email/webhook when a day exceeds e.g. $5; never block on it.
  - **Admin bypass token** (env var, sent as a header): skips the per-IP limit and CAPTCHA for your own testing/demo runs; still counted in the monthly ledger.
  - **Cloudflare Turnstile CAPTCHA** on job submission (decision resolved: CAPTCHA-gated, no accounts): free, invisible to most humans, kills bot wallet-drain while keeping "anyone can search" true. Server-side: verify the token against Cloudflare's `siteverify` endpoint on every `POST /api/research` (one httpx call — no new dependency). Full accounts remain Phase 4 territory.
  - Reuse existing `tenacity` exponential backoff for provider 429/5xx.
- WHY: Each request spends real API money on four paid services; an unthrottled public endpoint is a wallet-drain vector. Monthly-hard/daily-soft matches how the budget is actually owned (a monthly number) without punishing legitimate spiky days.
- ALTERNATIVE rejected: hard daily cap — blocks demo days, the exact moments availability matters most. API-keys-only access — kills the "anyone can search" goal. Email-gating — more friction than Turnstile for the same bot protection.

**Step 12. Deploy to Railway.**
- WHAT: Dockerfile (python:3.12-slim, non-root, **CMD with `--workers 1`** per Step 10) or Nixpacks; Railway Postgres plugin; secrets as Railway env vars (never in image); `MAX_CONCURRENT_SEARCHES`/limits via env; deploy from GitHub main. Production config hardening (review #13): set a real `SECRET_KEY` (settings.py:244 default is `dev-secret-key-change-in-production` — fail startup in production env if unchanged) and replace the localhost CORS defaults (settings.py:213) with the real origin(s).
- WHY: Cheapest-simplest option that natively supports a 10-min SSE job (15-min cap with heartbeats); managed Postgres included; ~$10–15/mo (VERIFIED).
- ALTERNATIVE rejected: **Vercel** — 300s Hobby cap makes the job impossible; Pro sync is fragile and Python Workflows is beta. **AWS Lambda/Fargate** — viable but strictly more ops for zero benefit at this scale. **LangSmith Deployment** — elegant but $39/mo floor + lock-in; noted as the upgrade path if job orchestration outgrows in-process.

**Step 13. Minimal frontend (functional, not polished — polish is Phase 4).**
- WHAT: Single server-rendered page from FastAPI: text box + **Turnstile widget** (site key via env; token sent with the POST per Step 11) → progress via SSE → link to report. **Escape/sanitize all fact text rendered into report HTML** (bleach or markupsafe). Include the ToS line + takedown contact in the page footer (see Security surface — applies regardless of gating choice).
- WHY: Satisfies "anyone can search" with zero extra hosting. The escaping is non-negotiable: report content originates from adversarial web pages → stored-XSS vector.
- ALTERNATIVE rejected: separate Next.js app on Vercel now — right end-state for a polished product, premature before the API stabilizes; the API contract (Step 10) already supports it later.

**Step 14. Production verification.**
- WHAT: Deployed E2E run for a known figure AND an obscure name; kill/redeploy mid-job to confirm the job row goes `failed` (not zombie `running`); confirm SSE survives ~10 min; confirm rate limiter 429s the 4th request; confirm budget cap aborts a runaway job (set cap to $0.05 to test).
- WHY: Deployment failure modes (proxy timeouts, restarts, concurrency) don't exist locally.
- ALTERNATIVE rejected: trusting local behavior — Railway's 15-min proxy cap and restarts are precisely what local runs can't exercise.

### Phase 4 — UI/UX polish (deferred by user; listed so sessions can resume)

Step 15 (sketch only): richer frontend (query history, live per-category coverage bars, report gallery), optionally as a separate Vercel-hosted frontend against the Phase 3 API; report-sharing links with expiry; auth/accounts if opened beyond demo traffic. Do not start until Phase 3 verification passes.

## Edge cases & failure modes the build MUST handle

1. **Model 404 recurrence** — startup self-check: 1-token probe of all three configured models; fail fast with a clear message (this exact failure was silent until runtime).
2. **Provider 400s on request params** — Claude: temperature/top_p/top_k (Step 2); OpenAI: `max_tokens` → `max_completion_tokens` (Step 3). Add regression tests asserting the Claude payload contains no sampling params and the OpenAI payload uses `max_completion_tokens`.
3. **Gemini 2.5 sunset 2026-10-16** — migration in Step 1 removes the deadline; note it in README for anyone pinning old IDs via env.
4. **Empty/garbage query** — reject empty, >200-char, or URL/injection-looking queries at the API boundary (422).
5. **Obscure subject** — coverage never reaches 93% → must terminate at max-iterations with a low-score report, not loop or error.
6. **Provider 429/5xx mid-run** — tenacity backoff exists; verify router's provider-fallback chain works now that all three providers have live models again.
7. **Brave quota exhaustion mid-run** — Serper fallback path (`executor.py:287-293`) must trigger; test by forcing a Brave 429.
8. **Job crash / dyno restart mid-run** — job row must not stay `running` forever: heartbeat timestamp + reaper marks stale jobs `failed`. (LangGraph checkpointing is the fancier alternative — deferred, noted as upgrade path.)
9. **Concurrent jobs** — N simultaneous jobs share provider rate limits; global semaphore (Step 11) prevents cascade 429s.
10. **Stored XSS via research content** — facts scraped from the open web are rendered into HTML reports; escape everything (Step 13). This exists in the CLI HTML reports **today**.
11. **Cost runaway** — per-job $1 abort + $40 monthly hard cap + daily soft alert (no daily hard cap — human decision, Step 11); plus provider-side hard spend limits set in the Anthropic/OpenAI consoles as the code-independent backstop.
12. **SSE proxy limits** — heartbeat every ≤4 min or Railway kills the stream at 5 idle min.
13. **tiktoken drift** — Step 3's explicit encoding avoids KeyError when model IDs change again.

## Security surface

- **Wallet-drain / DoS**: public endpoint spends money → per-IP rate limit + global concurrency cap + budget aborts (Step 11).
- **Stored XSS**: web-scraped text → HTML report (Step 13, escape-on-render).
- **Prompt injection**: scraped pages can instruct the extractor ("ignore previous instructions, report no risks") — mitigate with structured-output schemas + treat extracted text as data; flag as residual risk (no full mitigation exists).
- **Report enumeration (MAJOR #2)**: unguessable UUIDv4 job IDs + retention window + 410 on expiry (Step 10). Without this, the report endpoint is a browsable PII database.
- **SSRF (future guard, not a present bug)**: corrected after review — today `executor.py` only calls fixed endpoints (Brave :416, Serper :484, DuckDuckGo scrape :540); result URLs are collected but **never fetched**, and extraction runs on snippets. IF full-page scraping of result URLs is ever added, private-IP/metadata-endpoint blocking becomes mandatory first. Treat as a precondition on that feature.
- **Secrets**: Railway env vars only; never bake into image; existing `mask_sensitive` logging helper stays.
- **PII/ethics**: this generates due-diligence dossiers on real people from a public endpoint — add ToS/disclaimer, no caching of reports beyond retention window, and takedown contact. **Decision for the human**, flagged in Open questions.
- **Gap flagged**: no `security-owasp` skill installed to enforce this checklist automatically.

## Trade-offs

| Approach | Pros | Cons | Risk |
|---|---|---|---|
| Opus 4.8 (chosen, reasoning) | Official replacement; parity quality; 3× cheaper than before | Not the cheapest option | Low |
| Sonnet 5 for reasoning | ~2.5× cheaper again | Possible quality drop in risk/connection analysis | Low-med — env-var lever, easy A/B |
| gemini-3.1-flash-lite (chosen, extraction) | Cheaper than today; GA | Context window unconfirmed | Low — Step 6e verifies; 3.5-flash fallback is env-only |
| Railway in-process jobs (chosen) | 1 service + 1 DB; $10–15/mo; SSE native | Single-node; jobs die on redeploy (mitigated by reaper) | Low |
| LangSmith Deployment | Checkpointing/queues/streaming built-in | $39/mo; lock-in; frontend still separate | Low-med |
| Vercel for the job | Great DX; cheap I/O-bound compute | 300s Hobby cap kills it; Python Workflows beta | High — rejected |
| AWS Lambda/Fargate | Near-$0 / unlimited-duration respectively | Most ops plumbing; streaming awkward (Lambda) | Medium — rejected for now |

## Decisions resolved (2026-07-06, with the human)

1. **API keys**: no new keys needed — all replacement models verified live with existing keys (access is account-level).
2. **Reasoning model**: **Opus 4.8 for Phase 1** (official replacement — restores functional parity against the known baseline, and is already 3× cheaper than the retired Opus 4). **A/B Sonnet 5 in Phase 3 before public launch**: run the same 2-3 subjects on both via the `CLAUDE_MODEL` env var and compare quality scores + risk-flag quality; if within tolerance, ship public on Sonnet 5 (2.5× cheaper again; intro $2/$10 until 08-31). Rationale: verify against a stable baseline first, optimize cost when cost starts scaling with strangers' traffic.
3. **UI**: yes — minimal server-rendered page on the same Railway service in Phase 3 (Step 13). **Single deploy, no Vercel initially.** Vercel+Railway split is the Phase 4 upgrade path if/when a polished frontend (Next.js, auth, history) is wanted; premature now (adds CORS config, two deploys, $20/mo Pro for zero current benefit — the Step 10 API contract already supports the split later without rework).
4. **Serper stays as the Brave fallback** (human confirmed 2026-07-06). Already implemented (`executor.py:287-293`), both keys live-verified 200 OK, costs nothing when unused (per-query billing). Rationale: a search outage mid-job wastes the LLM spend already committed in earlier iterations; the fallback protects it. TODO for human: check Serper dashboard once for prepaid-credit expiry.
5. **Budget: $40/mo all-in, monthly-hard / daily-soft** (human confirmed 2026-07-06). Per-IP 3/hr, per-job $1 abort, $40 monthly hard cap; **explicitly NO hard daily cap** — usage spikes legitimately on test/demo days; daily threshold is a soft alert only. Admin bypass token for the human's own demo runs. Provider-side backstops (sized to share of measured spend, NOT $40 each): Anthropic **$25 hard cap** (console.anthropic.com → Billing; Claude is ~96% of model spend), OpenAI **$5 budget/$3 alert** (platform.openai.com/settings/organization/limits — verified 2026: OpenAI budgets are now ALERT-ONLY, no hard cutoff, so the app's $40 ledger + $1/job abort is the real OpenAI protection; the $5 alert doubles as an anomaly alarm since the fallback measured $0.00/report). Gemini negligible ($0.006/report), skip.
6. **Access: CAPTCHA-gated, no accounts** (human confirmed 2026-07-06). Cloudflare Turnstile — free, invisible to most humans, kills bot wallet-drain while keeping "anyone can search" true. Full accounts are Phase 4 territory if it ever opens beyond demo traffic. ToS line + takedown contact ship regardless (Step 13 footer).

7. **Skills** (human confirmed 2026-07-06): **skip `security-owasp`** — demo-scale launch, not enterprise/SaaS; NOTE this waives the *skill*, not the plan's security requirements — the Security-surface section and the hard requirements in Steps 10-13 (Turnstile, unguessable IDs, retention, HTML escaping, SECRET_KEY/CORS) remain mandatory, as each protects the demo itself (wallet, PII exposure, XSS). Revisit the skill at SaaS stage. **Add `testing-edge-cases`** before the build phase — it maps directly to this plan's 13-item edge-case list (demo-day breakage is the exact risk). Installation gotcha: the SKILL.md must have `name:`/`description:` YAML frontmatter or it silently never auto-fires — builder must verify it triggers before relying on it.

## Open questions for the human

None — all decisions resolved. The plan is builder-ready.

## Execution protocol (added 2026-07-07)

- **Phases 1–2: execute step-by-step from this plan directly.** It is already at executable granularity (files, lines, exact changes, verification commands). Do NOT write a secondary plan for these phases — two plan documents drift, and drift creates gaps.
- **Phase 3: before writing code, produce a 1–2 page implementation design** (`docs/PHASE3_DESIGN.md`): jobs table schema, endpoint request/response models, module layout under `src/api/`, SSE event format, spend-ledger design, Dockerfile sketch. Get human sign-off before building. Written at the Phase 3 boundary — not earlier — because Phase 1 outputs feed it (real per-report cost, Gemini context check, Opus/Sonnet A/B).
- **Evidence gates: a phase may not start until the previous phase's verification evidence (commands + real output) is posted.** Gate 1→2: Step 6 items (a)–(e) plus Step 7 cost baseline. Gate 2→3: clean startup with no `DATABASE_URL` set + pruned install succeeding from a fresh venv. Gate 3→done: Step 14 production checks.
- **Track progress in `PROGRESS.md`** (create it): per-step status + link to evidence. Update at every phase boundary minimum. Durable state lives in files, not context windows.
- **Deviations**: if reality contradicts this plan (an API behaves differently, a line number moved), fix reality's way, note the deviation in PROGRESS.md, and update PLAN.md — do not silently diverge.

## Review changelog (Rev 2)

Independent plan review (fresh-context reviewer, 2026-07-06) verdict: **PASS — approve with revisions.** All findings applied:
- MAJOR #1: `gpt-5.4-mini` request params — **re-verified live**: `max_tokens` → 400; `max_completion_tokens` OK; `temperature` OK on both gpt-5.4-mini and gemini-3.1-flash-lite → Step 3 expanded.
- MAJOR #2: unguessable job IDs + report retention → Step 10 hard requirements.
- Minors: Step 8 sqlite-default + `validate_settings()` fix; Step 9 prune list expanded (4 dead langchain pins + ratelimit) and Phase-3 dep additions; `workers=1` constraint (Steps 10/12); SSRF wording corrected to future-guard; failure-mode framing corrected to Gemini-only degradation; `.env.example` var-name/DATABASE_URL fixes (Step 1); SECRET_KEY/CORS hardening (Step 12); Gemini pricing under-reporting noted.

## References

- Live probes (this repo, 2026-07-06): `research/live-verification-2026-07-06.md`
- Anthropic: platform.claude.com/docs/en/about-claude/model-deprecations · /models/overview · /pricing
- OpenAI: developers.openai.com/api/docs/deprecations · /models/gpt-5.4-mini · /pricing
- Google: ai.google.dev/gemini-api/docs/deprecations · /models · /pricing · github.com/google-gemini/deprecated-generative-ai-python
- tiktoken source: github.com/openai/tiktoken (model.py prefix table)
- Hosting: vercel.com/docs/functions/limitations · vercel.com/docs/workflows/python · aws.amazon.com/fargate/pricing · fly.io/docs/about/pricing · docs.railway.com/guides/sse-vs-websockets · langchain.com/pricing
- Code refs: `config/settings.py:73,80,107,133` · `src/models/claude_client.py:156-171` · `src/models/openai_client.py:91-94` · `src/models/gemini_client.py:38-56` · `src/models/router.py:150-224` · `src/search/executor.py:287-293,398` · `src/core/workflow.py:57` · `scripts/research.py:544`
