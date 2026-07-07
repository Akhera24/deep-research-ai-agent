"""Spend ledger: monthly hard cap, daily soft alert (PHASE3_DESIGN §5, §11.R2)."""

import asyncio
import uuid
from typing import Optional

import httpx
from sqlalchemy import func, select

from config.settings import settings
from config.logging_config import get_logger
from src.api.db import Job, SpendLedger, QUEUED, RUNNING, get_sessionmaker, utcnow

logger = get_logger(__name__)


def _today() -> str:
    return utcnow().strftime("%Y-%m-%d")


def _this_month() -> str:
    return utcnow().strftime("%Y-%m")


async def month_total(session) -> float:
    result = await session.execute(
        select(func.coalesce(func.sum(SpendLedger.amount_usd), 0)).where(
            SpendLedger.month == _this_month()
        )
    )
    return float(result.scalar_one())


async def day_total(session) -> float:
    result = await session.execute(
        select(func.coalesce(func.sum(SpendLedger.amount_usd), 0)).where(
            SpendLedger.day == _today()
        )
    )
    return float(result.scalar_one())


async def budget_exhausted(session) -> bool:
    """The §11.R2 gate formula, used at POST time AND at dequeue time.

    Reserves worst-case spend (PER_JOB_BUDGET_USD) for every job that is
    already running or waiting, so queued jobs can't sail past the monthly
    cap between acceptance and execution.
    """
    ledgered = await month_total(session)
    inflight = (
        await session.execute(
            select(func.count()).select_from(Job).where(Job.status.in_([RUNNING, QUEUED]))
        )
    ).scalar_one()
    reserved = ledgered + settings.PER_JOB_BUDGET_USD * inflight
    return reserved >= settings.MONTHLY_BUDGET_USD


async def record(job_id: uuid.UUID, amount_usd: float) -> None:
    """Ledger a job's spend; fire the daily soft alert if the day crossed
    the threshold (log + optional webhook — NEVER blocks, §5)."""
    if amount_usd <= 0:
        return
    async with get_sessionmaker()() as session:
        session.add(
            SpendLedger(
                job_id=job_id,
                day=_today(),
                month=_this_month(),
                amount_usd=amount_usd,
            )
        )
        await session.commit()
        today_spend = await day_total(session)

    if today_spend > settings.DAILY_SOFT_ALERT_USD:
        logger.warning(
            "Daily spend soft alert",
            extra={"day": _today(), "spend_usd": round(today_spend, 4),
                   "threshold_usd": settings.DAILY_SOFT_ALERT_USD},
        )
        if settings.ALERT_WEBHOOK_URL:
            # Fire-and-forget: alerting must never block or fail a job.
            asyncio.create_task(_post_alert(today_spend))


async def _post_alert(today_spend: float) -> None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                settings.ALERT_WEBHOOK_URL,
                json={"type": "daily_spend_soft_alert", "day": _today(),
                      "spend_usd": round(today_spend, 4),
                      "threshold_usd": settings.DAILY_SOFT_ALERT_USD},
            )
    except httpx.HTTPError as e:
        logger.error("Soft-alert webhook failed", extra={"error": str(e)})
