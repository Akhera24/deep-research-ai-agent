"""
Job service: create → queue → run → persist (PHASE3_DESIGN §3, §11.R2-R4, §10.E).

Invariants:
- Tasks are held in a module-level set (asyncio.create_task keeps only a weak
  ref; §10.E) and every job body is wrapped so failures land in the DB row.
- ONE ResearchOrchestrator (and thus one ModelRouter) per job — cost tracking
  is instance-global, sharing would cross-contaminate per-job cost (§11.R4).
- The budget gate runs at POST time and AGAIN at dequeue (§11.R2).
- PII discipline (§12.S3): API jobs never write research_results/ files and
  production logs carry job_id, never the raw query.
"""

import asyncio
import json
import uuid
from datetime import timedelta
from typing import Optional

from sqlalchemy import select, update

from config.settings import settings
from config.logging_config import get_logger
from src.api import ledger
from src.api.db import (
    COMPLETED, FAILED, QUEUED, RUNNING,
    Job, get_sessionmaker, utcnow,
)
from src.core.workflow import ResearchOrchestrator
from src.reporting.html_report import calculate_quality_score, render_html_report

logger = get_logger(__name__)

# Strong references so the GC can't collect running jobs (§10.E)
_tasks: set[asyncio.Task] = set()
_semaphore: Optional[asyncio.Semaphore] = None

API_JOB_MAX_ITERATIONS = 10

# Bounds for the activity feed / ticker / live-report keys inside
# Job.progress (PLAN.md A2 + A.2/UX2): the whole dict is rewritten on every
# update and streamed over SSE, so the payload and the DB row stay small.
ACTIVITY_MAX = 30
SAMPLE_FACTS_MAX = 3
SAMPLE_FACT_CHARS = 140
REPORT_FACTS_MAX = 40
FACT_CONTENT_CHARS = 160


def _truncate(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit - 1] + "…"


class BudgetExceeded(Exception):
    """Raised inside the progress callback when per-job spend passes the cap."""


def _apply_activity(progress_state: dict, event) -> None:
    """Merge one executor/extractor activity event into the shared dict.

    Job.progress is persisted by FULL-COLUMN overwrite, so both writers —
    the node-boundary progress callback and the activity callback — mutate
    this ONE dict and persist it whole; a partial write would clobber the
    other writer's keys (REVIEW-LEARNINGS, shared-writer rule).
    """
    if not isinstance(event, dict):
        return
    # Monotonic per-job seq = the client's stable row key: survives reconnect
    # snapshot re-sends AND legitimately-repeated events (e.g. cache hits).
    seq = progress_state.get("activity_seq", 0) + 1
    progress_state["activity_seq"] = seq
    entry = {k: v for k, v in event.items() if k not in ("samples", "facts_new")}
    entry["seq"] = seq
    feed = progress_state.setdefault("activity", [])
    feed.append(entry)
    del feed[:-ACTIVITY_MAX]

    samples = [s for s in (event.get("samples") or []) if isinstance(s, str)]
    if samples:
        ticker = progress_state.setdefault("sample_facts", [])
        ticker.extend(_truncate(s, SAMPLE_FACT_CHARS) for s in samples)
        del ticker[:-SAMPLE_FACTS_MAX]

    _apply_report_preview(progress_state, event)


def _apply_report_preview(progress_state: dict, event: dict) -> None:
    """Fold an event into the bounded live-report preview (PLAN.md A.2/UX2)."""
    kind, status = event.get("kind"), event.get("status")

    if kind == "extract" and status == "done":
        facts_new = [f for f in (event.get("facts_new") or [])
                     if isinstance(f, dict) and isinstance(f.get("content"), str)]
        if not facts_new:
            return
        preview = progress_state.setdefault("report_preview", {})
        facts = preview.setdefault("facts", [])
        fact_seq = preview.get("fact_seq", 0)
        for f in facts_new:
            fact_seq += 1
            facts.append({
                "id": f"f{fact_seq}",   # collision-safe key (review MA3)
                "content": _truncate(f["content"], FACT_CONTENT_CHARS),
                "category": f.get("category") or "other",
                "confidence": f.get("confidence"),
            })
        preview["fact_seq"] = fact_seq
        preview["facts_found"] = fact_seq  # running discovery count (MI4)
        if len(facts) > REPORT_FACTS_MAX:
            keep = sorted(facts, key=lambda x: x.get("confidence") or 0,
                          reverse=True)[:REPORT_FACTS_MAX]
            keep_ids = {k["id"] for k in keep}
            facts[:] = [f for f in facts if f["id"] in keep_ids]
        by_cat: dict = {}
        for f in facts:
            by_cat[f["category"]] = by_cat.get(f["category"], 0) + 1
        preview["by_category"] = by_cat

    elif kind == "llm" and status == "done":
        task = event.get("task")
        if task == "risk_assessment":
            progress_state.setdefault("report_preview", {})["risks"] = {
                "count": event.get("risks", 0),
                "severities": event.get("severities") or {},
            }
        elif task == "connection_mapping":
            progress_state.setdefault("report_preview", {})["connections"] = {
                "count": event.get("connections", 0),
                "sample": [s for s in (event.get("sample") or [])
                           if isinstance(s, str)][:3],
            }
        elif task == "verification":
            progress_state.setdefault("report_preview", {})["verification"] = {
                "verified": event.get("verified", 0),
                "deduped": event.get("deduped", 0),
            }


def _progress_snapshot(progress_state: dict) -> dict:
    """Whole-dict copy for persistence — lists copied too, so in-place
    mutation after the write can't alias the row payload."""
    snap = dict(progress_state)
    for key in ("activity", "sample_facts"):
        if isinstance(snap.get(key), list):
            snap[key] = list(snap[key])
    preview = snap.get("report_preview")
    if isinstance(preview, dict):
        preview = dict(preview)
        if isinstance(preview.get("facts"), list):
            preview["facts"] = list(preview["facts"])
        snap["report_preview"] = preview
    return snap


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS)
    return _semaphore


def reset_state() -> None:
    """Test hook: fresh semaphore/task set (e.g. after settings change)."""
    global _semaphore
    _semaphore = None
    _tasks.clear()


def _router_cost(orchestrator: ResearchOrchestrator) -> float:
    """Total spend accumulated on this job's private router instance."""
    return float(
        sum(c.config.total_cost for c in orchestrator.router.clients.values())
    )


async def create_job(query: str, client_ip_hash: Optional[str], admin: bool) -> uuid.UUID:
    """Insert the queued row and spawn the runner task. Returns the job id."""
    job_id = uuid.uuid4()
    now = utcnow()
    async with get_sessionmaker()() as session:
        session.add(Job(
            id=job_id,
            query=query,
            status=QUEUED,
            created_at=now,
            expires_at=now + timedelta(days=settings.REPORT_RETENTION_DAYS),
            progress={"phase": "queued"},
            client_ip_hash=client_ip_hash,
            admin=admin,
        ))
        await session.commit()

    task = asyncio.create_task(_run_job_safe(job_id, query))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    logger.info("Job created", extra={"job_id": str(job_id), "admin": admin})
    return job_id


async def _run_job_safe(job_id: uuid.UUID, query: str) -> None:
    """Wrapper so no exception is ever swallowed by fire-and-forget (§10.E)."""
    try:
        await _run_job(job_id, query)
    except Exception as e:  # noqa: BLE001 — terminal safety net
        logger.error(
            "Job crashed", extra={"job_id": str(job_id), "error": str(e)}, exc_info=True
        )
        try:
            await _finish(job_id, FAILED, error=f"internal error: {type(e).__name__}")
        except Exception:  # pragma: no cover — DB down; nothing left to do
            logger.error("Failed to record job failure", extra={"job_id": str(job_id)})


async def _run_job(job_id: uuid.UUID, query: str) -> None:
    async with _get_semaphore():
        # Dequeue budget gate (§11.R2) — re-checked under the semaphore,
        # BEFORE flipping queued → running.
        async with get_sessionmaker()() as session:
            if await ledger.budget_exhausted(session):
                await _finish(job_id, FAILED, error="budget exhausted")
                logger.warning("Job rejected at dequeue: budget", extra={"job_id": str(job_id)})
                return

        # ONE shared mutable progress dict, owned here (PLAN.md Step A2 /
        # REVIEW-LEARNINGS): both callbacks below mutate it and persist it
        # WHOLE. All callbacks run on this job's coroutine, so writes never
        # interleave mid-mutation.
        progress_state: dict = {"phase": "starting", "activity": [],
                                "sample_facts": []}
        await _update(job_id, status=RUNNING, started_at=utcnow(), heartbeat_at=utcnow(),
                      progress=_progress_snapshot(progress_state))

        # ONE orchestrator per job (§11.R4) — private router, private cost.
        orchestrator = ResearchOrchestrator(
            max_iterations=API_JOB_MAX_ITERATIONS, enable_checkpoints=False
        )
        started = utcnow()

        async def progress_callback(p: dict) -> None:
            cost = _router_cost(orchestrator)
            progress_state.update({
                "phase": p.get("node"),
                "iteration": p.get("iteration"),
                "max_iterations": p.get("max_iterations"),
                "facts": p.get("facts"),
                "coverage": (p.get("coverage") or {}).get("average"),
            })
            await _update(
                job_id,
                heartbeat_at=utcnow(),
                progress=_progress_snapshot(progress_state),
                cost_usd=cost,
            )
            if cost >= settings.PER_JOB_BUDGET_USD:
                raise BudgetExceeded(
                    f"per-job budget ${settings.PER_JOB_BUDGET_USD:.2f} exceeded"
                )

        async def activity_callback(event: dict) -> None:
            # UI-only writer: no cost update, no budget check — cost/abort
            # granularity stays at node boundaries (plan-review, "budget
            # gate unchanged"). Heartbeats during long nodes help the reaper.
            _apply_activity(progress_state, event)
            await _update(
                job_id,
                heartbeat_at=utcnow(),
                progress=_progress_snapshot(progress_state),
            )

        try:
            result = await orchestrator.research(
                query,
                progress_callback=progress_callback,
                activity_callback=activity_callback,
            )
        except BudgetExceeded as e:
            cost = _router_cost(orchestrator)
            await _finish(job_id, FAILED, error=str(e), cost_usd=cost)
            await ledger.record(job_id, cost)
            logger.warning("Job aborted: budget", extra={"job_id": str(job_id), "cost_usd": cost})
            return
        except Exception as e:
            cost = _router_cost(orchestrator)
            await _finish(job_id, FAILED, error=f"research failed: {type(e).__name__}", cost_usd=cost)
            await ledger.record(job_id, cost)
            logger.error("Job failed", extra={"job_id": str(job_id), "error": str(e)}, exc_info=True)
            return

        duration = (utcnow() - started).total_seconds()
        # Escaping happens inside render_html_report (edge case #10).
        html = render_html_report(result, query, duration)
        report_json = json.loads(json.dumps(result, default=str))
        cost = _router_cost(orchestrator)
        score = calculate_quality_score(
            facts=result.get("facts", []),
            risk_flags=result.get("risk_flags", []),
            connections=result.get("connections", []),
            coverage=result.get("metadata", {}).get("coverage", {}),
        )

        await _finish(
            job_id, COMPLETED,
            report_html=html, report_json=report_json, cost_usd=cost,
            progress={"phase": "complete", "facts": len(result.get("facts", [])),
                      "quality_score": score.get("score"), "grade": score.get("grade")},
        )
        await ledger.record(job_id, cost)
        logger.info(
            "Job completed",
            extra={"job_id": str(job_id), "cost_usd": round(cost, 4),
                   "duration_seconds": round(duration, 1)},
        )


async def _update(job_id: uuid.UUID, **values) -> None:
    async with get_sessionmaker()() as session:
        await session.execute(update(Job).where(Job.id == job_id).values(**values))
        await session.commit()


async def _finish(job_id: uuid.UUID, status: str, **values) -> None:
    await _update(job_id, status=status, finished_at=utcnow(), **values)


async def get_job(job_id: uuid.UUID) -> Optional[Job]:
    async with get_sessionmaker()() as session:
        return (
            await session.execute(select(Job).where(Job.id == job_id))
        ).scalar_one_or_none()
