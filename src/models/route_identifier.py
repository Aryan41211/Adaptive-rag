"""
Route identifier model.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RouteIdentifier(BaseModel):
    """Routing decision produced by the query classifier."""

    route: Literal["index", "general", "search"] = Field(
        description=(
            "'index' to answer from the user's uploaded documents, "
            "'general' to answer from general knowledge, "
            "'search' to look the answer up on the web."
        )
    )
