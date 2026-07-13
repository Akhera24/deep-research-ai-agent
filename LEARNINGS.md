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
