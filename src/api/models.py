"""Pydantic request/response schemas for the job API (PHASE3_DESIGN §2)."""

import re
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

# Query validation (edge case #4 + §12.S1):
# - permit business-name characters: & . ' - , ( ) + digits, unicode letters
# - reject URL-shaped, markup/injection-shaped, and control characters
_ALLOWED_QUERY = re.compile(r"^[\w\s&.,'’\-()+]+$", re.UNICODE)
_FORBIDDEN_SUBSTRINGS = ("://", "<", ">", "{", "}", "`", "$(", "\\")


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    turnstile_token: str = Field(default="", max_length=4096)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty")
        if len(v) > 200:
            raise ValueError("query must be at most 200 characters")
        lowered = v.lower()
        for bad in _FORBIDDEN_SUBSTRINGS:
            if bad in lowered:
                raise ValueError("query contains forbidden characters")
        if not _ALLOWED_QUERY.match(v):
            raise ValueError(
                "query may contain letters, digits, spaces and & . , ' - ( ) + only"
            )
        return v


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
