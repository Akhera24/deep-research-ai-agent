# LEARNINGS

Repo-specific rules from mistakes and observations made once. Read before
starting work; obey every rule (per the global harness contract).

- **A long-running local dev server keeps stale Python imports.** After any
  backend change, restart the uvicorn instance before manual testing, or you
  are testing old code with a new template (Jinja templates reload from disk
  per request; imported Python modules do not — `--reload` is not used here).
  Symptom: the UI renders new elements but no new data arrives. Check with
  `pgrep -fl uvicorn` when behavior doesn't match the diff.
  *Origin: bit twice on 2026-07-10 — a stray job ran on pre-A.2 code, and the
  A.3 manual test would have run on pre-A.3 code.*

- **Every LLM JSON-array response WILL eventually hit max_tokens and truncate
  mid-array — a parser without truncation repair silently returns `[]`, and the
  miss cascades (0 connections → −20 quality points).** Repair by cutting back
  to the last complete object and closing the array (the extractor already did
  this; the risks/connections parser didn't until 2026-07-11). Symptom: a
  "call successful" log with valid-looking JSON content followed by
  "Mapped 0 …" / "No JSON array found".
  *Origin: 2026-07-11 Phase B gate runs — the connections call (max_tokens=4000)
  truncated on 3 of 3 runs; fix in `workflow.py::_repair_truncated_json_array`.*

- **Playwright treats a clipped-but-laid-out child as visible.** An element
  inside a `max-height: 0; overflow: hidden` parent (this repo's collapsed
  `.section-content` pattern) still has a non-empty layout box, so
  `expect(child).not_to_be_visible()` FAILS on a correctly-collapsed section —
  and a `to_be_visible()` would pass without the section ever being opened.
  Assert collapse the way the report's own toggle works: the `open` class is
  absent AND the container's `getBoundingClientRect().height` is 0; after the
  click, both flip. Symptom: "Locator expected not to be visible / Actual:
  visible" on an element the browser plainly clips.
  *Origin: 2026-07-13 C1.7a — sideline-section browser test.*

- **`config/logging_config.get_logger` is structlog with `PrintLoggerFactory`
  — log lines go to STDOUT and never through stdlib logging, so pytest's
  `caplog` captures nothing from them.** Assert log output via `capsys`, and
  treat any negative caplog assertion (`assert X not in caplog.text`) as
  VACUOUS: it passes even while X is being logged (proven by probe: a planted
  secret in a warning → stdout saw it, `caplog.records == 0`, the negative
  assertion passed). Known instance to fix when touched:
  `tests/test_preflight.py::test_failure_logs_never_carry_the_raw_query`.
  *Origin: 2026-07-13 C1.7a — high-sideline warning test red under caplog
  despite the warning visibly firing.*

- **A Playwright assertion on a TRANSIENT UI state (hidden-before-SSE,
  pre-hydration text) races the harness's 0.1s SSE poll / instant hydration
  and flakes.** The state is real but lives ~100ms — the stream/hydration
  rewrites the DOM before the assertion's first check. Either stub the input
  that creates the state (page.route the snapshot GET and serve the fixture
  that freezes it) or assert the causal END-STATE property instead (e.g.
  "banner visible although the cold snapshot carried no entity ⇒ it came
  from SSE"). Symptom: `to_be_hidden`/text asserts that pass alone but fail
  under load, with the "wrong" value being the NEXT state.
  *Origin: 2026-07-14 P0 Step 2 — the finalization-A banner test and the
  elapsed-seed test both failed this way before rewrite.*

- **Planted store fixtures (dra_history/dra_owned) get hydration-pruned by
  the real 404 before the assertion runs.** The homepage's hydration
  refreshes every planted entry against `GET /api/research/{id}`; a made-up
  jobId 404s and the prune-on-404 feature (working as designed) deletes the
  fixture mid-test. Stub the snapshot route with a 503 (the blip-keep branch)
  whenever a test plants entries for jobs that don't exist. Symptom:
  "element(s) not found" on a row the test just planted.
  *Origin: 2026-07-14 P0 Step 4 — the hostile-label test lost its row to
  hydration mid-assertion.*

- **Test fixtures with absolute dates in TTL-pruned stores are time bombs.**
  A fixture hardcoding `createdAt: "2026-07-13"` in `dra_history_v1` passes
  until the calendar moves 7 days past it — then the store's OWN prune
  (feature working correctly) deletes the fixture at read time and the
  assertion fails with "element(s) not found" on a row the test just
  planted. Always compute fixture dates relative to now
  (`datetime.now(timezone.utc) - timedelta(days=1)`); reserve absolute
  dates for the cases that TEST the boundary (a far-future date for the
  clock-skew keep, a stale date for the prune itself).
  *Origin: 2026-07-22 — three green history e2e tests failed on calendar
  rollover alone; no code had changed.*

- **Phase labels lag reality by one node.** `state["stage"]` is set inside each
  node but only streamed at node boundaries (`astream(stream_mode="values")`
  yields AFTER a node returns), so during a long node the UI's phase label still
  shows the PREVIOUS stage while activity events show what is actually happening.
  This is inherent to the stream design, not a bug: any "current status" UI must
  lead with the activity stream (the loading screen's strip does exactly this).
  Do NOT "fix" the lag by routing stage/callbacks through LangGraph state (state
  must stay serializable) or by changing stream semantics — if earlier labels are
  ever needed, emit a stage hint through the activity channel instead.
  *Origin: 2026-07-10 Phase A live E2E observation (Lisa Su run).*
