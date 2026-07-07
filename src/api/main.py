"""
FastAPI app factory (PHASE3_DESIGN §3).

Run with EXACTLY ONE worker (see Dockerfile): in-process asyncio jobs, the
global concurrency semaphore, and SSE affinity all break silently under
multi-worker.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config.settings import settings
from config.logging_config import get_logger
from src.api.db import dispose_engine, init_db, startup_sweep
from src.api.reaper import reaper_loop
from src.api.routes import limiter, router

logger = get_logger(__name__)

DEV_SECRET_DEFAULT = "dev-secret-key-change-in-production"

# Turnstile default = Cloudflare's always-pass TEST key; real deploys override it.
_TURNSTILE_TEST_SITE_KEY = "1x00000000000000000000AA"


def _log_config_presence() -> None:
    """Log PRESENT/MISSING for deploy-critical config — NAMES ONLY, no values."""
    checks = {
        "SECRET_KEY_is_real": settings.SECRET_KEY != DEV_SECRET_DEFAULT,
        "ADMIN_BYPASS_TOKEN": bool(settings.ADMIN_BYPASS_TOKEN),
        "TURNSTILE_SITE_KEY_is_real": settings.TURNSTILE_SITE_KEY != _TURNSTILE_TEST_SITE_KEY,
        "TURNSTILE_EXPECTED_HOSTNAME": bool(settings.TURNSTILE_EXPECTED_HOSTNAME),
        "ANTHROPIC_API_KEY": bool(settings.ANTHROPIC_API_KEY),
        "GOOGLE_API_KEY": bool(settings.GOOGLE_API_KEY),
        "OPENAI_API_KEY": bool(settings.OPENAI_API_KEY),
        "BRAVE_API_KEY": bool(settings.BRAVE_API_KEY),
        "SERPER_API_KEY": bool(settings.SERPER_API_KEY),
        "DATABASE_URL_is_postgres": settings.DATABASE_URL.startswith("postgres"),
    }
    present = sorted(k for k, ok in checks.items() if ok)
    missing = sorted(k for k, ok in checks.items() if not ok)
    logger.info("Config presence check", extra={"present": present, "missing": missing})


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Production hardening (§7): refuse to boot with the dev secret —
    # SECRET_KEY salts IP hashes and must be real in production.
    if settings.ENVIRONMENT == "production" and settings.SECRET_KEY == DEV_SECRET_DEFAULT:
        raise RuntimeError(
            "SECRET_KEY is the development default; set a real SECRET_KEY in production"
        )

    # Config visibility (never values): a deploy log line listing which
    # deploy-critical vars are PRESENT vs MISSING makes a Railway deploy
    # verifiable without exposing secrets. /healthz doesn't touch Turnstile,
    # so this is the only signal that the submit path is actually configured.
    _log_config_presence()

    await init_db()
    swept = await startup_sweep()  # §11.R1: running AND queued → failed
    if swept:
        logger.warning("Orphaned jobs failed at startup", extra={"count": swept})

    reaper_task = asyncio.create_task(reaper_loop())
    logger.info("API started", extra={"environment": settings.ENVIRONMENT})
    try:
        yield
    finally:
        reaper_task.cancel()
        try:
            await reaper_task
        except asyncio.CancelledError:
            pass
        await dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Deep Research Agent API",
        version="1.0.0",
        lifespan=lifespan,
        # No public schema browsing for a demo that spends money per request
        docs_url=None, redoc_url=None, openapi_url=None,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    app.include_router(router)
    return app


app = create_app()
