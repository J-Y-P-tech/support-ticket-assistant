"""Validated output of the triage node (SPEC §4.3 / §5.3)."""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.enums import Category, Sentiment, Urgency


class TriageResult(BaseModel):
    """Classification of a ticket: topic, urgency, and customer mood.

    All three fields are required so no unvalidated LLM output reaches the
    workflow; the triage node retries when the model returns an invalid value.
    """

    category: Category
    urgency: Urgency
    sentiment: Sentiment
