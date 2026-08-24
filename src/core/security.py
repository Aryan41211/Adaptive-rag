"""
Authentication primitives: password hashing and JWT access tokens.

Passwords are hashed with bcrypt. Tokens are short-lived HS256 JWTs signed
with ``JWT_SECRET_KEY``; the token subject is the user id, which every
downstream layer uses to scope data to its owner.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import bcrypt
import jwt

from src.core.config import settings
from src.core.exceptions import AuthenticationError

# bcrypt silently truncates input beyond 72 bytes; reject rather than truncate
# so two different long passwords can never collide into the same hash.
MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    """
    Hash a plaintext password with bcrypt.

    Args:
        password: The plaintext password.

    Returns:
        The bcrypt hash, as a UTF-8 string.

    Raises:
        ValueError: If the password exceeds bcrypt's 72-byte limit.
    """
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes."
        )
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """
    Check a plaintext password against a stored bcrypt hash.

    Args:
        password: The plaintext password to check.
        password_hash: The stored bcrypt hash.

    Returns:
        True if the password matches.
    """
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(encoded, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed or legacy hash in storage: treat as a failed login.
        return False


def create_access_token(
    user_id: str,
    username: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Issue a signed JWT access token.

    Args:
        user_id: The user's stable identifier (becomes the token subject).
        username: The user's display name.
        expires_delta: Optional custom lifetime.

    Returns:
        The encoded JWT.
    """
    now = datetime.now(timezone.utc)
    expire = now + (
        expires_delta
        or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload: dict[str, Any] = {
        "sub": user_id,
        "username": username,
        "iat": now,
        "exp": expire,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(
        payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Verify and decode a JWT access token.

    Args:
        token: The encoded JWT.

    Returns:
        The decoded claims.

    Raises:
        AuthenticationError: If the token is expired, malformed or unsigned
            by this application.
    """
    try:
        claims = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Access token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Invalid access token.") from exc

    if not claims.get("sub"):
        raise AuthenticationError("Invalid access token.")
    return claims
