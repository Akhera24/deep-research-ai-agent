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


class BudgetExceeded(Exception):
    """Raised inside the progress callback when per-job spend passes the cap."""


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

        await _update(job_id, status=RUNNING, started_at=utcnow(), heartbeat_at=utcnow(),
                      progress={"phase": "starting"})

        # ONE orchestrator per job (§11.R4) — private router, private cost.
        orchestrator = ResearchOrchestrator(
            max_iterations=API_JOB_MAX_ITERATIONS, enable_checkpoints=False
        )
        started = utcnow()

        async def progress_callback(p: dict) -> None:
            cost = _router_cost(orchestrator)
            await _update(
                job_id,
                heartbeat_at=utcnow(),
                progress={
                    "phase": p.get("node"),
                    "iteration": p.get("iteration"),
                    "max_iterations": p.get("max_iterations"),
                    "facts": p.get("facts"),
                    "coverage": (p.get("coverage") or {}).get("average"),
                },
                cost_usd=cost,
            )
            if cost >= settings.PER_JOB_BUDGET_USD:
                raise BudgetExceeded(
                    f"per-job budget ${settings.PER_JOB_BUDGET_USD:.2f} exceeded"
                )

        try:
            result = await orchestrator.research(query, progress_callback=progress_callback)
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
