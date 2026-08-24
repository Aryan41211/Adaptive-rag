"""
Authentication endpoints.

Replaces the external Rust auth service the Streamlit frontend used to call,
which does not exist in this repository and made login unachievable.
"""

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status

from src.api.deps import CurrentUser, get_current_user
from src.api.ratelimit import auth_rate_limit
from src.core.config import settings
from src.core.exceptions import AuthenticationError, UserAlreadyExistsError
from src.core.logger import get_logger
from src.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from src.db import revoked_tokens, users
from src.models.query_request import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(auth_rate_limit())],
)
async def register(payload: RegisterRequest) -> TokenResponse:
    """
    Create an account and return an access token.

    Args:
        payload: The requested username and password.

    Returns:
        A freshly issued access token.

    Raises:
        UserAlreadyExistsError: If the username is taken.
    """
    # bcrypt is deliberately slow; keep it off the event loop.
    password_hash = await asyncio.to_thread(hash_password, payload.password)

    try:
        user = await users.create_user(payload.username, password_hash)
    except ValueError as exc:
        raise UserAlreadyExistsError(str(exc)) from exc

    logger.info("Registered new user '%s'", user["username"])
    return TokenResponse(
        access_token=create_access_token(user["user_id"], user["username"]),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        username=user["username"],
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(auth_rate_limit())],
)
async def login(payload: LoginRequest) -> TokenResponse:
    """
    Exchange credentials for an access token.

    Args:
        payload: The submitted username and password.

    Returns:
        A freshly issued access token.

    Raises:
        AuthenticationError: If the credentials are not valid.
    """
    user = await users.get_user_by_username(payload.username)

    # Verify even when the user is unknown so the response time does not
    # reveal which usernames exist.
    stored_hash = user["password_hash"] if user else ""
    matched = await asyncio.to_thread(verify_password, payload.password, stored_hash)

    if not user or not matched:
        logger.info("Failed login attempt for '%s'", payload.username)
        raise AuthenticationError("Incorrect username or password.")

    logger.info("User '%s' logged in", user["username"])
    return TokenResponse(
        access_token=create_access_token(user["user_id"], user["username"]),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        username=user["username"],
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(user: CurrentUser = Depends(get_current_user)) -> None:
    """
    Revoke the caller's access token.

    A JWT is self-contained, so discarding it client-side leaves a copied
    token usable until it expires. Recording its identifier here makes the
    token unusable immediately, on every worker.

    Args:
        user: The authenticated caller.
    """
    if user.jti:
        await revoked_tokens.revoke(
            user.jti,
            datetime.fromtimestamp(user.expires_at, tz=timezone.utc),
        )
        logger.info("Revoked session for '%s'", user.username)
