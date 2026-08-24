"""
Tests for password hashing and JWT access tokens.
"""

from datetime import timedelta

import pytest

from src.core.exceptions import AuthenticationError
from src.core.security import (
    MAX_PASSWORD_BYTES,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_round_trip():
    hashed = hash_password("correct-horse-battery")
    assert hashed != "correct-horse-battery"
    assert verify_password("correct-horse-battery", hashed)


def test_wrong_password_rejected():
    hashed = hash_password("correct-horse-battery")
    assert not verify_password("wrong-password", hashed)


def test_same_password_hashes_differently():
    """Distinct salts must produce distinct hashes."""
    assert hash_password("same-password") != hash_password("same-password")


def test_overlong_password_rejected_not_truncated():
    """bcrypt truncates past 72 bytes; we must reject instead of colliding."""
    with pytest.raises(ValueError):
        hash_password("a" * (MAX_PASSWORD_BYTES + 1))


def test_malformed_hash_is_a_failed_login_not_a_crash():
    assert verify_password("anything", "not-a-bcrypt-hash") is False


def test_token_round_trip():
    token = create_access_token("user-123", "alice")
    claims = decode_access_token(token)
    assert claims["sub"] == "user-123"
    assert claims["username"] == "alice"


def test_expired_token_rejected():
    token = create_access_token(
        "user-123", "alice", expires_delta=timedelta(seconds=-10)
    )
    with pytest.raises(AuthenticationError):
        decode_access_token(token)


def test_tampered_token_rejected():
    token = create_access_token("user-123", "alice")
    tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    with pytest.raises(AuthenticationError):
        decode_access_token(tampered)


def test_token_signed_with_another_secret_rejected():
    import jwt

    forged = jwt.encode(
        {"sub": "attacker", "exp": 9999999999}, "a-different-secret", "HS256"
    )
    with pytest.raises(AuthenticationError):
        decode_access_token(forged)


def test_unsigned_token_rejected():
    """The 'alg: none' downgrade attack must not succeed."""
    import jwt

    forged = jwt.encode({"sub": "attacker", "exp": 9999999999}, key="", algorithm="none")
    with pytest.raises(AuthenticationError):
        decode_access_token(forged)
