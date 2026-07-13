"""Pydantic request/response schemas for the job API (PHASE3_DESIGN §2)."""

import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Query validation (edge case #4 + §12.S1):
# - permit business-name characters: & . ' - , ( ) + digits, unicode letters
# - reject URL-shaped, markup/injection-shaped, and control characters
_ALLOWED_QUERY = re.compile(r"^[\w\s&.,'’\-()+]+$", re.UNICODE)
_FORBIDDEN_SUBSTRINGS = ("://", "<", ">", "{", "}", "`", "$(", "\\")


def _validate_query_charset(v: str, what: str = "query") -> str:
    """Shared allowlist for strings that enter SEARCH QUERIES and prompts
    (the research query and the C2 hints — NOT entity display fields, which
    carry `—` and validate by length + control-char strip instead, R5).
    Returns the canonical (stripped) value — the SAME value both endpoints
    hash into the pre-flight ticket (R6)."""
    v = v.strip()
    if not v:
        raise ValueError(f"{what} must not be empty")
    if len(v) > 200:
        raise ValueError(f"{what} must be at most 200 characters")
    lowered = v.lower()
    for bad in _FORBIDDEN_SUBSTRINGS:
        if bad in lowered:
            raise ValueError(f"{what} contains forbidden characters")
    if not _ALLOWED_QUERY.match(v):
        raise ValueError(
            f"{what} may contain letters, digits, spaces and & . , ' - ( ) + only"
        )
    return v


class ResearchHints(BaseModel):
    """C2 optional details — these strings enter search queries and prompts,
    so they use the SAME charset allowlist as the query (PLAN.md C1.2/R5)."""
    company: Optional[str] = Field(default=None, max_length=100)
    role: Optional[str] = Field(default=None, max_length=100)
    location: Optional[str] = Field(default=None, max_length=100)
    known_for: Optional[str] = Field(default=None, max_length=100)

    @field_validator("company", "role", "location", "known_for")
    @classmethod
    def validate_hint(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return None
        return _validate_query_charset(v, what="hint")

    def as_dict(self) -> Dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v}


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_ENTITY_DISAMBIGUATOR_MAX_CHARS = 80
_REJECTED_ENTITY_MAX_CHARS = 200


def _clean_rejected_entities(v: List[str]) -> List[str]:
    """C1.7d (D13/R5): descriptors the user explicitly rejected (picker
    "None of these" / banner cancel). DISPLAY-tier validation — the D5
    descriptor format carries `—`, so control-strip + length cap only, NOT
    the query charset. They enter prompts delimited-as-data and are
    PROMPT-TRANSIENT (R9): never search queries, progress, or
    resolved_entity."""
    cleaned = []
    for d in v:
        d = _CONTROL_CHARS.sub("", d).strip()[:_REJECTED_ENTITY_MAX_CHARS]
        if d:
            cleaned.append(d)
    return cleaned


class EntitySelection(BaseModel):
    """The candidate the user picked (or auto-proceeded on) at pre-flight.

    R5 two-tier validation: these are DISPLAY fields — rendered
    textContent-only and entering prompts delimited-as-data — so they
    validate by LENGTH CAP + CONTROL-CHAR STRIP ONLY, NOT the query charset
    (the D5 descriptor format carries `—`/`/`/`:`, which _ALLOWED_QUERY
    rejects). The server RECOMPUTES entity_id from the D5 formula; any
    client-supplied entity_id is silently ignored (pydantic drops extras) —
    a trusted client id could bind a report's stable KG identity to a
    mismatched entity."""
    canonical_name: str = Field(..., min_length=1, max_length=120)
    descriptor: str = Field(default="", max_length=200)
    disambiguators: List[str] = Field(default_factory=list, max_length=8)
    decision: Literal["auto", "picked", "hinted"] = "picked"

    @field_validator("canonical_name", "descriptor")
    @classmethod
    def strip_controls(cls, v: str) -> str:
        return _CONTROL_CHARS.sub("", v).strip()

    @field_validator("canonical_name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("canonical_name must not be empty")
        return v

    @field_validator("disambiguators")
    @classmethod
    def clean_disambiguators(cls, v: List[str]) -> List[str]:
        cleaned = []
        for d in v:
            d = _CONTROL_CHARS.sub("", d).strip()[:_ENTITY_DISAMBIGUATOR_MAX_CHARS]
            if d:
                cleaned.append(d)
        return cleaned


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    turnstile_token: str = Field(default="", max_length=4096)
    # C1.1: Turnstile alternative minted by /api/disambiguate (single-use)
    preflight_ticket: str = Field(default="", max_length=256)
    # C1.2: optional C2 hints (query charset — they enter search queries)
    context: Optional[ResearchHints] = None
    # C1.2: the resolved entity from the pre-flight picker/banner
    entity: Optional[EntitySelection] = None
    # C1.7d: negative scope from the refine loop (≤5, display-tier)
    rejected_entities: List[str] = Field(default_factory=list, max_length=5)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        return _validate_query_charset(v)

    @field_validator("rejected_entities")
    @classmethod
    def clean_rejected(cls, v: List[str]) -> List[str]:
        return _clean_rejected_entities(v)


class DisambiguateRequest(BaseModel):
    """Pre-flight request (C1.1) — same query contract as ResearchRequest so
    both endpoints hash the identical canonical query into the ticket (R6)."""
    query: str = Field(..., min_length=1, max_length=200)
    turnstile_token: str = Field(default="", max_length=4096)
    hints: Optional[ResearchHints] = None
    # C1.7d: negative scope from the refine loop (≤5, display-tier)
    rejected_entities: List[str] = Field(default_factory=list, max_length=5)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        return _validate_query_charset(v)

    @field_validator("rejected_entities")
    @classmethod
    def clean_rejected(cls, v: List[str]) -> List[str]:
        return _clean_rejected_entities(v)


class JobCreated(BaseModel):
    job_id: uuid.UUID
    status: str
    events_url: str
    report_url: str


class JobStatus(BaseModel):
    job_id: uuid.UUID
    status: str
    progress: Dict[str, Any] = {}
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    cost_usd: Optional[float] = None  # admin only (§2)
