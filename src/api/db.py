"""
Async persistence layer for the job API.

Self-contained per PHASE3_DESIGN §11.R5: its own Base and async engine,
independent of the legacy sync src/database/ module. Tables are created at
lifespan startup via create_all (additive greenfield schema — no alembic).

Portability notes (§11.R5):
- Uuid / DateTime(timezone=True) / JSON.with_variant(JSONB) map cleanly to
  both Postgres (Railway) and sqlite (local dev).
- sqlite strips tzinfo on read (naive datetimes) — every comparison in this
  package normalizes via utc_naive(); regression-tested in
  tests/test_api_db.py.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
    JSON,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

# Job lifecycle states
QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
EXPIRED = "expired"

PortableJSON = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    """Timezone-aware UTC now (stored tz-aware; sqlite reads back naive)."""
    return datetime.now(timezone.utc)


def utc_naive(dt: datetime) -> datetime:
    """Normalize a datetime to naive-UTC for cross-dialect comparison.

    sqlite returns naive datetimes even for DateTime(timezone=True) columns;
    Postgres returns aware ones. Comparing mixed forms raises TypeError, so
    all reaper/expiry math goes through this helper.
    """
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    query: Mapped[str] = mapped_column(String(200), nullable=True)  # NULLed at expiry (§12.S2)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=QUEUED, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    progress: Mapped[dict] = mapped_column(PortableJSON, nullable=False, default=dict)
    error: Mapped[str] = mapped_column(Text, nullable=True)
    cost_usd: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False, default=0)
    report_html: Mapped[str] = mapped_column(Text, nullable=True)
    report_json: Mapped[dict] = mapped_column(PortableJSON, nullable=True)
    client_ip_hash: Mapped[str] = mapped_column(String(64), nullable=True)
    admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class SpendLedger(Base):
    __tablename__ = "spend_ledger"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("jobs.id"), nullable=True)
    day: Mapped[str] = mapped_column(String(10), nullable=False)    # 'YYYY-MM-DD' (UTC)
    month: Mapped[str] = mapped_column(String(7), nullable=False)   # 'YYYY-MM'
    amount_usd: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


Index("idx_ledger_month", SpendLedger.month)
Index("idx_ledger_day", SpendLedger.day)


def _async_url(url: str) -> str:
    """Rewrite the configured DATABASE_URL scheme for the async drivers.

    settings.DATABASE_URL stays driver-agnostic (§11.R5 — no second env var):
    sqlite:///x → sqlite+aiosqlite:///x; postgresql:// (and Railway's legacy
    postgres://) → postgresql+asyncpg://.
    """
    if url.startswith("sqlite+aiosqlite://") or url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker | None = None


def get_engine() -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        _engine = create_async_engine(_async_url(settings.DATABASE_URL), pool_pre_ping=True)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_sessionmaker() -> async_sessionmaker:
    if _sessionmaker is None:
        get_engine()
    return _sessionmaker


async def init_db() -> None:
    """Create tables (lifespan startup)."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("API database initialized", extra={"dialect": engine.dialect.name})


async def startup_sweep() -> int:
    """Mark jobs orphaned by a restart as failed (§11.R1).

    Covers BOTH running and queued: under workers=1 a restart killed every
    in-flight asyncio task, including semaphore-waiters that owned queued rows.
    Returns the number of rows swept.
    """
    async with get_sessionmaker()() as session:
        result = await session.execute(
            update(Job)
            .where(Job.status.in_([RUNNING, QUEUED]))
            .values(status=FAILED, error="orphaned by restart", finished_at=utcnow())
        )
        await session.commit()
    swept = result.rowcount or 0
    if swept:
        logger.warning("Startup sweep failed orphaned jobs", extra={"count": swept})
    return swept


async def dispose_engine() -> None:
    """Dispose the engine (lifespan shutdown / test teardown)."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None
