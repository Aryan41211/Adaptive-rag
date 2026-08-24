"""
Shared FastAPI dependencies.
"""

from dataclasses import dataclass

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.exceptions import AuthenticationError
from src.core.security import decode_access_token

# auto_error=False so a missing header raises our own 401 with a consistent
# body, rather than FastAPI's 403 default.
_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    """The authenticated caller."""

    user_id: str
    username: str


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> CurrentUser:
    """
    Resolve the authenticated user from the Authorization header.

    Args:
        credentials: Bearer credentials extracted from the request.

    Returns:
        The authenticated user.

    Raises:
        AuthenticationError: If the token is absent, malformed or expired.
    """
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Authentication required.")

    claims = decode_access_token(credentials.credentials)
    return CurrentUser(
        user_id=str(claims["sub"]),
        username=str(claims.get("username", "")),
    )
