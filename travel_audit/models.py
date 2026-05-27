from pydantic import BaseModel
from typing import Literal


class DocumentInput(BaseModel):
    contract: str = ""
    itinerary: str = ""
    chat: str = ""


class AnalysisRequest(BaseModel):
    documents: DocumentInput
    model: str = "claude-opus-4-7"


class Finding(BaseModel):
    id: str
    dimension: Literal[
        "promise_gap",
        "biased_clause",
        "clarity",
        "cancellation",
        "needs_written_confirmation",
        "evidence_to_keep",
    ]
    risk_level: Literal["red", "yellow", "green"]
    risk_score: int
    title: str
    description: str
    citation: str
    recommendation: str


class AnalysisResponse(BaseModel):
    overall_risk_score: int
    summary: str
    findings: list[Finding]


class MessageRequest(BaseModel):
    findings: list[Finding]
    tone: Literal["polite", "firm", "formal"] = "polite"
    finding_ids: list[str] | None = None
    model: str = "claude-opus-4-7"


class GeneratedMessage(BaseModel):
    finding_id: str
    tone: str
    subject: str
    message: str


class MessageResponse(BaseModel):
    messages: list[GeneratedMessage]
