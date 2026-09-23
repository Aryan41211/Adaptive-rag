"""
Shared fixtures for the optional live integration suite.

These tests talk to real external services (the Gemini API here) and are not
collected by a plain ``pytest`` run: ``pytest.ini`` constrains ``testpaths``
to ``tests``. Run them deliberately with ``pytest tests_live``.

Credentials come from ``.env`` (untracked, gitignored) via python-dotenv,
mirroring the runtime path, and are never printed. When the Gemini key is
absent every test is skipped - the suite stays runnable on a developer machine
that only wants the offline tests.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY is not set; skipping live Gemini tests",
)


@pytest.fixture(scope="session")
def gemini_api_key() -> str:
    """The real Gemini key from the environment, never logged."""
    key = os.environ["GEMINI_API_KEY"]
    assert key not in {"", "your-gemini-api-key-here"}, "placeholder key found"
    return key
