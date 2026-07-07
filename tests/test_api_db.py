"""
Tests for the async API persistence layer (src/api/db.py).

Includes the sqlite tzinfo regression test required by PHASE3_DESIGN §11.R5:
sqlite strips tzinfo on read, so naive/aware datetime comparisons in the
reaper/expiry path must go through utc_naive().
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import uuid
from datetime import timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select

from src.api import db as api_db
from src.api.db import (
    Job, SpendLedger, QUEUED, RUNNING, COMPLETED, FAILED,
    utcnow, utc_naive, _async_url,
)


@pytest_asyncio.fixture
async def sqlite_db(tmp_path, monkeypatch):
    """Fresh sqlite-backed API database per test."""
    url = f"sqlite:///{tmp_path}/api_test.db"
    await api_db.dispose_engine()
    monkeypatch.setattr(api_db.settings, "DATABASE_URL", url)
    await api_db.init_db()
    yield api_db
    await api_db.dispose_engine()


def _job(**kw):
    defaults = dict(
        id=uuid.uuid4(),
        query="Test Subject",
        status=QUEUED,
        created_at=utcnow(),
        expires_at=utcnow() + timedelta(days=7),
        progress={},
    )
    defaults.update(kw)
    return Job(**defaults)


class TestUrlRewrite:
    def test_sqlite_scheme(self):
        assert _async_url("sqlite:///research.db") == "sqlite+aiosqlite:///research.db"

    def test_postgresql_scheme(self):
        assert _async_url("postgresql://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"

    def test_railway_legacy_postgres_scheme(self):
        assert _async_url("postgres://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"

    def test_already_async_untouched(self):
        assert _async_url("sqlite+aiosqlite:///x.db") == "sqlite+aiosqlite:///x.db"


@pytest.mark.asyncio
async def test_init_creates_tables_and_roundtrip(sqlite_db):
    async with sqlite_db.get_sessionmaker()() as session:
        job = _job()
        session.add(job)
        session.add(SpendLedger(job_id=job.id, day="2026-07-07", month="2026-07", amount_usd=0.19))
        await session.commit()

        rows = (await session.execute(select(Job))).scalars().all()
        assert len(rows) == 1
        assert rows[0].query == "Test Subject"
        ledger = (await session.execute(select(SpendLedger))).scalars().one()
        assert float(ledger.amount_usd) == pytest.approx(0.19)


@pytest.mark.asyncio
async def test_sqlite_tzinfo_regression_reaper_comparison(sqlite_db):
    """REGRESSION (§11.R5): sqlite returns naive datetimes for
    DateTime(timezone=True) columns. The reaper computes now - heartbeat_at;
    without normalization that's aware-minus-naive → TypeError."""
    stale = utcnow() - timedelta(minutes=10)
    async with sqlite_db.get_sessionmaker()() as session:
        session.add(_job(status=RUNNING, heartbeat_at=stale))
        await session.commit()

    async with sqlite_db.get_sessionmaker()() as session:
        job = (await session.execute(select(Job))).scalars().one()
        # Prove the hazard is real on sqlite: tzinfo was stripped on read.
        assert job.heartbeat_at.tzinfo is None
        # Raw aware-minus-naive raises — this is the bug utc_naive prevents.
        with pytest.raises(TypeError):
            _ = utcnow() - job.heartbeat_at
        # Normalized comparison works and is correct to within test runtime.
        age = utc_naive(utcnow()) - utc_naive(job.heartbeat_at)
        assert timedelta(minutes=9) < age < timedelta(minutes=11)


def test_utc_naive_handles_both_forms():
    aware = utcnow()
    naive = aware.replace(tzinfo=None)
    assert utc_naive(aware) == naive
    # A non-UTC aware datetime converts to UTC before stripping.
    plus2 = aware.astimezone(timezone(timedelta(hours=2)))
    assert utc_naive(plus2) == naive


@pytest.mark.asyncio
async def test_startup_sweep_fails_running_and_queued_only(sqlite_db):
    async with sqlite_db.get_sessionmaker()() as session:
        session.add(_job(status=RUNNING))
        session.add(_job(status=QUEUED))
        session.add(_job(status=COMPLETED))
        await session.commit()

    swept = await sqlite_db.startup_sweep()
    assert swept == 2

    async with sqlite_db.get_sessionmaker()() as session:
        jobs = (await session.execute(select(Job))).scalars().all()
        by_status = sorted(j.status for j in jobs)
        assert by_status == [COMPLETED, FAILED, FAILED]
        for j in jobs:
            if j.status == FAILED:
                assert j.error == "orphaned by restart"
                assert j.finished_at is not None
