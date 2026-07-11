"""API routes (PHASE3_DESIGN §2, §4, §11.R6)."""

import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text
from sse_starlette.sse import EventSourceResponse

from config.settings import settings
from config.logging_config import get_logger
from src.api import jobs, ledger
from src.api.db import COMPLETED, EXPIRED, FAILED, get_sessionmaker
from src.api.models import JobCreated, JobStatus, ResearchRequest
from src.api.security import hash_ip, is_admin, verify_turnstile

logger = get_logger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# slowapi: keys on request.client.host, which uvicorn resolves from
# X-Forwarded-For ONLY under --proxy-headers + FORWARDED_ALLOW_IPS (§10.A).
# NEVER swap for get_ipaddr (raw XFF header — spoofable).
limiter = Limiter(key_func=get_remote_address)

# Report responses: strict CSP as defense-in-depth behind escaping (§11.R6).
# The report uses inline style/script (pagination) and no external resources.
REPORT_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src data:; base-uri 'none'; form-action 'none'"
)

SSE_POLL_SECONDS = 1.5

# Phase D4: regenerated reports served as public samples (person + company —
# human decision 2026-07-10, person justified per review R2 option (b): ultra-
# public figure, noindex header, takedown contact in ToS). MUST be
# post-escaping-chokepoint artifacts (never the Feb example_report.html) and
# MUST NOT live under src/api/static/ — the static mount doesn't send the
# report header contract below.
SAMPLE_REPORTS = {
    "person": Path(__file__).parent / "sample_report_person.html",
    "company": Path(__file__).parent / "sample_report_company.html",
}


@router.get("/sample-report")
async def sample_report_default():
    return RedirectResponse(url="/sample-report/person", status_code=307)


@router.get("/sample-report/{kind}")
async def sample_report(kind: str):
    path = SAMPLE_REPORTS.get(kind)
    if path is None:
        raise HTTPException(status_code=404, detail="unknown sample report")
    try:
        html = path.read_text(encoding="utf-8")
    except OSError:
        raise HTTPException(status_code=404, detail="sample report unavailable")
    # job_report contract + noindex: the report HTML has no robots meta, so
    # this header is the only thing keeping a marketing asset out of Google.
    return HTMLResponse(
        content=html,
        headers={"Content-Security-Policy": REPORT_CSP,
                 "X-Content-Type-Options": "nosniff",
                 "X-Robots-Tag": "noindex"},
    )


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"turnstile_site_key": settings.TURNSTILE_SITE_KEY}
    )


@router.get("/healthz")
async def healthz():
    async with get_sessionmaker()() as session:
        await session.execute(text("SELECT 1"))
    return {"status": "ok", "db": "ok"}


@router.post("/api/research", status_code=202, response_model=JobCreated)
@limiter.limit(f"{settings.RATE_LIMIT_REPORTS_PER_HOUR}/hour", exempt_when=is_admin)
async def create_research(request: Request, body: ResearchRequest):
    admin = is_admin(request)
    client_ip = request.client.host if request.client else None

    # POST-time budget gate (§11.R2; re-checked at dequeue)
    async with get_sessionmaker()() as session:
        if await ledger.budget_exhausted(session):
            return JSONResponse(
                status_code=503,
                content={"error": "budget exhausted",
                         "detail": "The monthly research budget is used up. Try again next month."},
            )

    if not admin:
        ok, reason = await verify_turnstile(body.turnstile_token, client_ip)
        if not ok:
            raise HTTPException(status_code=403, detail=reason)

    job_id = await jobs.create_job(
        query=body.query, client_ip_hash=hash_ip(client_ip) if not admin else None,
        admin=admin,
    )
    return JobCreated(
        job_id=job_id,
        status="queued",
        events_url=f"/api/research/{job_id}/events",
        report_url=f"/api/research/{job_id}/report",
    )


async def _load_job_or_404(job_id: uuid.UUID):
    job = await jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job


@router.get("/api/research/{job_id}", response_model=JobStatus, response_model_exclude_none=True)
async def job_status(request: Request, job_id: uuid.UUID):
    job = await _load_job_or_404(job_id)
    if job.status == EXPIRED:
        raise HTTPException(status_code=410, detail="report expired")
    return JobStatus(
        job_id=job.id,
        status=job.status,
        progress=job.progress or {},
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        cost_usd=float(job.cost_usd) if is_admin(request) else None,
    )


def _terminal_event(job) -> dict:
    if job.status == COMPLETED:
        payload = {
            "job_id": str(job.id),
            "report_url": f"/api/research/{job.id}/report",
        }
        score = (job.progress or {}).get("quality_score")
        if score is not None:
            payload["quality_score"] = score
        return {"event": "completed", "data": json.dumps(payload)}
    return {"event": "failed", "data": json.dumps({"error": job.error or "failed"})}


@router.get("/api/research/{job_id}/events")
async def job_events(request: Request, job_id: uuid.UUID):
    job = await _load_job_or_404(job_id)
    if job.status == EXPIRED:
        raise HTTPException(status_code=410, detail="report expired")

    async def stream():
        # Already-terminal jobs: send the terminal event immediately, close (§11.R6)
        current = await jobs.get_job(job_id)
        if current is None:
            return
        if current.status in (COMPLETED, FAILED):
            yield _terminal_event(current)
            return

        last_progress = None
        while True:
            current = await jobs.get_job(job_id)
            if current is None or current.status == EXPIRED:
                yield {"event": "failed", "data": json.dumps({"error": "expired"})}
                return
            if current.status in (COMPLETED, FAILED):
                yield _terminal_event(current)
                return
            progress = current.progress or {}
            if progress != last_progress:
                last_progress = progress
                yield {"event": "progress", "data": json.dumps(progress)}
            await asyncio.sleep(SSE_POLL_SECONDS)

    # sse-starlette's built-in ping (15s default) defeats Railway's 5-min idle
    # timeout; send_timeout reaps half-dead connections (§10.G).
    return EventSourceResponse(stream(), send_timeout=30)


@router.get("/api/research/{job_id}/report")
async def job_report(job_id: uuid.UUID):
    job = await _load_job_or_404(job_id)
    if job.status == EXPIRED:
        raise HTTPException(status_code=410, detail="report expired")
    if job.status != COMPLETED:
        raise HTTPException(status_code=409, detail=f"job is {job.status}, not completed")
    return HTMLResponse(
        content=job.report_html,
        headers={"Content-Security-Policy": REPORT_CSP,
                 "X-Content-Type-Options": "nosniff"},
    )
