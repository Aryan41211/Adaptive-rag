"""
Grade model for relevance scoring.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Grade(BaseModel):
    """Relevance grade for retrieved context."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the context is relevant to the question, else 'no'."
    )
