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
from src.api.models import (
    DisambiguateRequest, JobCreated, JobStatus, ResearchRequest,
)
from src.api.security import (
    consume_preflight_ticket, hash_ip, is_admin, mint_preflight_ticket,
    verify_turnstile,
)
from src.core import preflight

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


# C1.1 (PLAN.md Rev 3.8): pre-flight disambiguation. Gate order matches
# /api/research (review R3): ratelimit (decorator, own 3× bucket) → budget →
# Turnstile → spend. Candidates are NEVER persisted server-side — they go to
# the client and come back as the chosen entity (stateless; zero retention
# surface). JSON-only POST endpoint: CSP/X-Robots-Tag n/a; nosniff added for
# defence-in-depth (no global security-header middleware exists, review R4).
@router.post("/api/disambiguate")
@limiter.limit(f"{settings.RATE_LIMIT_REPORTS_PER_HOUR * 3}/hour", exempt_when=is_admin)
async def disambiguate(request: Request, body: DisambiguateRequest):
    admin = is_admin(request)
    client_ip = request.client.host if request.client else None

    # Budget gate BEFORE any spend — pre-flight draws from the same $40 cap
    async with get_sessionmaker()() as session:
        if await ledger.budget_exhausted(session):
            return JSONResponse(
                status_code=503,
                content={"error": "budget exhausted",
                         "detail": "The monthly research budget is used up. Try again next month."},
                headers={"X-Content-Type-Options": "nosniff"},
            )

    if not admin:
        ok, reason = await verify_turnstile(body.turnstile_token, client_ip)
        if not ok:
            raise HTTPException(status_code=403, detail=reason)

    hints = body.hints.as_dict() if body.hints else None
    # C1.7d: rejected descriptors reach the clustering prompt as a
    # delimited data line (a refine re-run should not re-lead with them);
    # PROMPT-TRANSIENT per R9 — never persisted, never in search queries.
    result = await preflight.discover_candidates(
        body.query, hints=hints,
        rejected_entities=body.rejected_entities or None)
    await ledger.record(None, result.cost)   # job_id NULL (db.py: nullable)

    # The ticket ships on EVERY outcome (incl. decision "error" — fail-open:
    # the client proceeds unscoped). Bound to ip_hash + the post-validation
    # canonical query (R6); single-use, consumed by /api/research (R2).
    ticket = mint_preflight_ticket(hash_ip(client_ip), body.query)
    return JSONResponse(
        content={
            "decision": result.decision,
            "note": result.note,
            "candidates": [c.to_dict() for c in result.candidates],
            "ticket": ticket,
        },
        headers={"X-Content-Type-Options": "nosniff"},
    )


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
        if body.preflight_ticket:
            # C1.1: ticket minted by /api/disambiguate replaces Turnstile
            # (tokens are single-use — the pre-flight consumed this user's).
            # Verifies HMAC + TTL + ip binding + canonical-query binding,
            # then consumes the nonce (single-use, R2).
            ok, reason = consume_preflight_ticket(
                body.preflight_ticket, hash_ip(client_ip), body.query)
        else:
            ok, reason = await verify_turnstile(body.turnstile_token, client_ip)
        if not ok:
            raise HTTPException(status_code=403, detail=reason)

    # C1.2 (D3/D5): merge hints ∪ entity into the orchestrator context
    # (entity applied last = entity wins) and record the resolved entity.
    # The context travels IN-MEMORY to the runner; a copy rides in
    # `progress` for the banner and in report_json.metadata — all three
    # are already §12.S2-purged at expiry.
    context: dict = {}
    if body.context:
        context.update(body.context.as_dict())
    resolved_entity: dict = {"decision": "unscoped"}
    if body.entity:
        ent = body.entity
        context["research_target"] = ent.descriptor or ent.canonical_name
        if ent.disambiguators:
            context["disambiguators"] = "; ".join(ent.disambiguators)
        primary = (ent.disambiguators[0] if ent.disambiguators
                   else ent.descriptor)
        resolved_entity = {
            # SERVER-computed (R5) — any client-sent entity_id was dropped
            # at model validation and is never read.
            "entity_id": preflight.compute_entity_id(ent.canonical_name, primary),
            "canonical_name": ent.canonical_name,
            "descriptor": ent.descriptor,
            "disambiguators": ent.disambiguators,
            "decision": ent.decision,
        }

    job_id = await jobs.create_job(
        query=body.query, client_ip_hash=hash_ip(client_ip) if not admin else None,
        admin=admin, context=context or None, resolved_entity=resolved_entity,
        # C1.7d (R9): prompt-transient — deliberately NOT merged into
        # `context` (the context detail-join would echo it into the
        # orientation lines) and never into progress/resolved_entity.
        rejected_entities=body.rejected_entities or None,
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


# C1.4: the D2 safety valve. Auth = knowledge of the unguessable job UUID —
# the same capability model as status/report reads (documented tradeoff,
# accepted 2026-07-11). Idempotent; cancel-after-terminal is a no-op 200.
@router.post("/api/research/{job_id}/cancel")
async def cancel_research(job_id: uuid.UUID):
    job = await _load_job_or_404(job_id)
    if job.status in (COMPLETED, FAILED, EXPIRED):
        return {"status": job.status}      # race with completion → no-op
    jobs.request_cancel(job_id)
    logger.info("Cancel requested", extra={"job_id": str(job_id)})
    return {"status": "cancelling"}


# C1.5 (Rev 3.9): "Generate report now" — stop iterating, run the FULL
# finalization tail (D8). Same capability model as cancel; strictly less
# destructive. Takes effect at the next continue/verify branch — up to one
# more full iteration may run first (R2).
@router.post("/api/research/{job_id}/finish")
async def finish_research(job_id: uuid.UUID):
    job = await _load_job_or_404(job_id)
    if job.status in (COMPLETED, FAILED, EXPIRED):
        return {"status": job.status}      # race with completion → no-op
    # R1: parentheses are load-bearing — `x or 0 == 0` is truthy for every
    # input. Node-boundary count; lags mid-extraction (UX floor, not a
    # race-free invariant).
    if ((job.progress or {}).get("facts") or 0) == 0:
        raise HTTPException(status_code=409,
                            detail="nothing found yet — cancel instead")
    jobs.request_finish_early(job_id)
    logger.info("Finish-early requested", extra={"job_id": str(job_id)})
    return {"status": "finishing"}


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
