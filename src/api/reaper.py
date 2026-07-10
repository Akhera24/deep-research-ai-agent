"""
Background reaper (PHASE3_DESIGN §3, §12.S2).

Every 60s:
- running jobs whose heartbeat is older than 5 minutes → failed (edge case #8;
  second layer behind the startup sweep)
- rows past expires_at → expired, with the PII purge: report_html,
  report_json, client_ip_hash, query AND progress NULLed (§12.S2). progress
  joined the purge in Phase A: its activity/sample_facts keys carry search
  queries derived from the researched name plus scraped snippets. The row
  skeleton (id, status, timestamps, cost) is kept for audit.

Datetime comparisons are done in Python via utc_naive() — sqlite returns
naive datetimes, Postgres aware ones (§11.R5).
"""

import asyncio
from datetime import timedelta

from sqlalchemy import select, update

from config.logging_config import get_logger
from src.api.db import (
    EXPIRED, FAILED, RUNNING,
    Job, get_sessionmaker, utc_naive, utcnow,
)

logger = get_logger(__name__)

REAP_INTERVAL_SECONDS = 60
STALE_HEARTBEAT = timedelta(minutes=5)


async def reap_once() -> dict:
    """One reaper pass. Returns counts for logging/tests."""
    now = utc_naive(utcnow())
    stale_ids = []
    expired_ids = []

    async with get_sessionmaker()() as session:
        # Stale running jobs (fetch-then-compare in Python for portability)
        running = (
            await session.execute(select(Job).where(Job.status == RUNNING))
        ).scalars().all()
        for job in running:
            beat = job.heartbeat_at or job.started_at or job.created_at
            if now - utc_naive(beat) > STALE_HEARTBEAT:
                stale_ids.append(job.id)

        if stale_ids:
            await session.execute(
                update(Job)
                .where(Job.id.in_(stale_ids))
                .values(status=FAILED, error="stale heartbeat (reaped)", finished_at=utcnow())
            )

        # Expiry + PII purge (§12.S2) — any non-expired row past expires_at
        candidates = (
            await session.execute(select(Job).where(Job.status != EXPIRED))
        ).scalars().all()
        for job in candidates:
            if now >= utc_naive(job.expires_at):
                expired_ids.append(job.id)

        if expired_ids:
            await session.execute(
                update(Job)
                .where(Job.id.in_(expired_ids))
                .values(
                    status=EXPIRED,
                    report_html=None,
                    report_json=None,
                    client_ip_hash=None,
                    query=None,
                    progress=None,
                )
            )

        await session.commit()

    counts = {"stale_failed": len(stale_ids), "expired": len(expired_ids)}
    if stale_ids or expired_ids:
        logger.info("Reaper pass", extra=counts)
    return counts


async def reaper_loop() -> None:
    """Lifespan background task; cancelled on shutdown."""
    while True:
        try:
            await reap_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — the reaper must survive
            logger.error("Reaper pass failed", extra={"error": str(e)}, exc_info=True)
        await asyncio.sleep(REAP_INTERVAL_SECONDS)
