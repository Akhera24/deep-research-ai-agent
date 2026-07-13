"""Turnstile verification, admin bypass, IP hashing, pre-flight tickets
(PHASE3_DESIGN §6, §10.H, §11.R6; PLAN.md Rev 3.8 Phase C1)."""

import hashlib
import hmac
import secrets
import time
from typing import Optional

import httpx
from fastapi import Request

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


def is_admin(request: Request) -> bool:
    """True when a configured ADMIN_BYPASS_TOKEN matches X-Admin-Token."""
    token = settings.ADMIN_BYPASS_TOKEN
    if not token:
        return False
    provided = request.headers.get("X-Admin-Token", "")
    return bool(provided) and secrets.compare_digest(provided, token)


def hash_ip(ip: Optional[str]) -> Optional[str]:
    """SECRET_KEY-salted sha256 of the client IP (§1 — forensics, not raw PII)."""
    if not ip:
        return None
    return hashlib.sha256(f"{ip}{settings.SECRET_KEY}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Pre-flight ticket (Phase C1.1 — PLAN.md Rev 3.8, reviews R2/R6)
#
# Turnstile tokens are single-use, so /api/disambiguate (which verifies one)
# hands the client an HMAC ticket that /api/research accepts in its place:
#   ticket = "{issued_at}.{nonce}.{HMAC-SHA256(key, ip_hash‖sha256(canonical
#   query)‖issued_at‖nonce)}",  TTL = PREFLIGHT_TICKET_TTL_SECONDS.
# BOTH endpoints hash the POST-VALIDATION canonical query (models.py strips) —
# the frontend resends the query byte-for-byte, else legitimate submits 403.
#
# SINGLE-WORKER INVARIANT (review R1/R2 — do not relax silently): the
# consumed-nonce set below AND jobs._cancel_requested are in-process memory,
# sound ONLY under the pinned `--workers 1` (Dockerfile:22) where mint,
# verify, consume, and the job task share one process. Relaxing that pin
# requires real DB columns + an explicit ALTER TABLE migration
# (REVIEW-LEARNINGS "Schema changes / migrations"). A restart clears the set
# and (when PREFLIGHT_TICKET_SECRET is unset) rotates the boot key — in-flight
# tickets then fail verification and the client re-challenges: fail-safe.
# ---------------------------------------------------------------------------

_BOOT_TICKET_SECRET = secrets.token_bytes(32)
_MAX_CLOCK_SKEW_SECONDS = 60

# mac hex → expiry epoch; entries live at most TTL seconds
_consumed_tickets: dict[str, float] = {}


def reset_ticket_state() -> None:
    """Test hook: forget consumed nonces."""
    _consumed_tickets.clear()


def _ticket_key() -> bytes:
    if settings.PREFLIGHT_TICKET_SECRET:
        return settings.PREFLIGHT_TICKET_SECRET.encode()
    return _BOOT_TICKET_SECRET


def _ticket_mac(ip_hash: Optional[str], canonical_query: str,
                issued_at: int, nonce: str) -> str:
    query_hash = hashlib.sha256(canonical_query.encode()).hexdigest()
    payload = f"{ip_hash or ''}|{query_hash}|{issued_at}|{nonce}".encode()
    return hmac.new(_ticket_key(), payload, hashlib.sha256).hexdigest()


def mint_preflight_ticket(ip_hash: Optional[str], canonical_query: str) -> str:
    issued_at = int(time.time())
    nonce = secrets.token_hex(8)
    mac = _ticket_mac(ip_hash, canonical_query, issued_at, nonce)
    return f"{issued_at}.{nonce}.{mac}"


def consume_preflight_ticket(ticket: str, ip_hash: Optional[str],
                             canonical_query: str) -> tuple[bool, str]:
    """Verify AND consume (single-use). Returns (ok, reason)."""
    parts = ticket.split(".")
    if len(parts) != 3:
        return False, "invalid pre-flight ticket"
    issued_str, nonce, mac = parts
    try:
        issued_at = int(issued_str)
    except ValueError:
        return False, "invalid pre-flight ticket"

    now = time.time()
    if issued_at > now + _MAX_CLOCK_SKEW_SECONDS:
        return False, "invalid pre-flight ticket"
    if now > issued_at + settings.PREFLIGHT_TICKET_TTL_SECONDS:
        return False, "pre-flight ticket expired, please retry the challenge"

    expected = _ticket_mac(ip_hash, canonical_query, issued_at, nonce)
    if not hmac.compare_digest(expected, mac):
        return False, "invalid pre-flight ticket"

    # Opportunistic purge keeps the set bounded (entries outlive their TTL
    # by at most one consume call).
    for stale in [m for m, exp in _consumed_tickets.items() if exp <= now]:
        del _consumed_tickets[stale]
    if mac in _consumed_tickets:
        return False, "pre-flight ticket already used, please retry the challenge"
    _consumed_tickets[mac] = issued_at + settings.PREFLIGHT_TICKET_TTL_SECONDS
    return True, "ok"


async def verify_turnstile(token: str, remote_ip: Optional[str]) -> tuple[bool, str]:
    """Server-side Turnstile verification (§6, §10.H).

    Returns (ok, reason). Tokens are single-use with a 300s lifetime —
    'timeout-or-duplicate' means the client must re-render the widget.
    Validates the response hostname when TURNSTILE_EXPECTED_HOSTNAME is set
    (anti token-farming, §11.R6).
    """
    if not token:
        return False, "missing turnstile token"
    payload = {"secret": settings.TURNSTILE_SECRET_KEY, "response": token}
    if remote_ip:
        payload["remoteip"] = remote_ip
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(SITEVERIFY_URL, data=payload)
            data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        # Fail closed: an unverifiable challenge is a failed challenge.
        logger.error("Turnstile siteverify unreachable", extra={"error": str(e)})
        return False, "verification service unavailable, please retry"

    if not data.get("success"):
        codes = data.get("error-codes", [])
        if "timeout-or-duplicate" in codes:
            return False, "challenge expired, please retry the challenge"
        return False, "challenge verification failed"

    expected = settings.TURNSTILE_EXPECTED_HOSTNAME
    if expected and data.get("hostname") != expected:
        logger.warning(
            "Turnstile hostname mismatch",
            extra={"got": data.get("hostname"), "expected": expected},
        )
        return False, "challenge verification failed"

    return True, "ok"
