"""Pydantic schemas and fixed taxonomies for ticket triage."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT_ACCESS = "account_access"
    FEATURE_REQUEST = "feature_request"


class UrgencyLevel(str, Enum):
    """P1 is highest urgency; P4 is lowest urgency."""

    P1 = "P1"  # Critical outage or severe business block
    P2 = "P2"  # Major degradation or high-priority issue
    P3 = "P3"  # Minor issue, standard query, or workaround available
    P4 = "P4"  # Low-impact request or general feedback


class TriageAction(str, Enum):
    AUTO_RESOLVE = "auto_resolve"
    ESCALATE = "escalate"
    REQUEST_CLARIFICATION = "request_clarification"


class TicketInput(BaseModel):
    """An incoming ticket to triage. Blank body text is allowed."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ticket_id: str = Field(min_length=1)
    subject: str
    body: str


class HistoricalTicket(BaseModel):
    """A resolved historical ticket used to build the retrieval index."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ticket_id: str = Field(min_length=1)
    subject: str
    body: str
    resolution: str = Field(min_length=1)
    category: Optional[TicketCategory] = None
    urgency: Optional[UrgencyLevel] = None


class TriageOutput(BaseModel):
    """The exact structured output returned for each ticket."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ticket_id: str = Field(min_length=1)
    category: TicketCategory
    urgency: UrgencyLevel
    action: TriageAction
    reasoning: str = Field(
        description="Concise factual rationale, no more than 50 words."
    )
    retrieved_ticket_ids: List[str] = Field(
        description="IDs returned by historical-ticket retrieval."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model-reported confidence estimate; not necessarily calibrated.",
    )

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning_word_limit(cls, value: str) -> str:
        words = value.split()
        if len(words) > 50:
            raise ValueError(
                f"Reasoning must not exceed 50 words; got {len(words)}."
            )
        return value