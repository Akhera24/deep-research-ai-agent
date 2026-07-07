"""Turnstile verification, admin bypass, IP hashing (PHASE3_DESIGN §6, §10.H, §11.R6)."""

import hashlib
import secrets
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
