"""
Shared fixtures and helpers for the optional live integration suite.

These tests talk to real external services (the Gemini API here) and are not
collected by a plain ``pytest`` run: ``pytest.ini`` constrains ``testpaths``
to ``tests``. Run them deliberately with ``pytest tests_live``.

Credentials come from ``.env`` (untracked, gitignored) via python-dotenv,
mirroring the runtime path, and are never printed. When the Gemini key is
absent every test is skipped - the suite stays runnable on a developer machine
that only wants the offline tests.

Free-tier honesty
-----------------

The Gemini free tier answers a burst of requests with ``RESOURCE_EXHAUSTED``
(HTTP 429) while the per-minute quota resets. These live probes therefore run
through :func:`retry_gemini_quota`, which retries the exact same real call a
bounded number of times with exponential backoff. When the quota has genuinely
reset the test proceeds; if it is still hard-exhausted after the retries the
test is *skipped* with that explicit reason. A test is never silently marked
passed without a real provider call completing.
"""

import time
from collections.abc import Callable
from typing import TypeVar

import pytest
from dotenv import load_dotenv

load_dotenv()

_T = TypeVar("_T")


def retry_gemini_quota(
    call: Callable[[], _T], *, attempts: int = 3, backoff: float = 4.0
) -> _T:
    """
    Run ``call``, absorb free-tier quota bursts with backoff, then fail honestly.

    Every attempt is a genuine provider call. The free tier intermittently
    answers ``RESOURCE_EXHAUSTED`` (429) while the per-minute budget refills,
    so a single attempt produces a flaky red that blames the code for a quota
    pause. A bounded backoff re-runs the same real call until either it
    succeeds or the quota stays hard-exhausted, in which case the test is
    skipped with the reason spelled out.

    Args:
        call: The live probe (one real provider request each invocation).
        attempts: How many real tries before giving up.
        backoff: Base seconds for the exponential backoff between tries.

    Returns:
        The provider's real result.

    Raises:
        pytest.skip.Exception: When the quota stays exhausted after all tries.
        GoogleAPICallError: Any non-quota provider failure, surfaced as-is so
            a real outage is never masked as a quota skip.
    """
    from google.api_core.exceptions import (
        GoogleAPICallError,
        ResourceExhausted,
        TooManyRequests,
    )

    last_error: GoogleAPICallError | None = None
    for attempt in range(attempts):
        try:
            return call()
        except (ResourceExhausted, TooManyRequests) as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(backoff * (2**attempt))
        except GoogleAPICallError:
            raise
    pytest.skip(
        f"Gemini free-tier quota stayed exhausted after {attempts} tries: {last_error}"
    )
