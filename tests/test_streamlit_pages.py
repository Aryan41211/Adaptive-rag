"""
Runtime tests for the Streamlit pages.

These actually execute each page through Streamlit's own test harness, so a
misordered ``set_page_config`` call or a bad import fails here rather than in
front of a user.
"""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "streamlit_app"

# Streamlit puts the entrypoint's directory on sys.path at runtime; AppTest
# does not, so replicate it for `from utils.api_client import ...`.
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@pytest.fixture
def offline_api(monkeypatch):
    """Pretend the backend is unreachable (no network in tests)."""
    from utils import api_client

    monkeypatch.setattr(api_client, "api_available", lambda: False)
    return api_client


def _run(page: str, session_state: dict | None = None) -> AppTest:
    app = AppTest.from_file(str(APP_DIR / page), default_timeout=30)
    for key, value in (session_state or {}).items():
        app.session_state[key] = value
    return app.run()


def test_home_page_runs_without_exceptions(offline_api):
    """Covers the set_page_config ordering error that broke this page."""
    app = _run("home.py")
    assert not app.exception, app.exception


def test_home_page_reports_an_unreachable_backend(offline_api):
    app = _run("home.py")
    assert any("not reachable" in error.value for error in app.error)


def test_home_page_shows_the_sign_in_form_when_api_is_up(monkeypatch):
    from utils import api_client

    monkeypatch.setattr(api_client, "api_available", lambda: True)

    app = _run("home.py")
    assert not app.exception, app.exception
    assert len(app.text_input) == 2  # username + password
    assert app.radio


def test_chat_page_runs_and_blocks_anonymous_access():
    """Without a token the chat page must refuse rather than crash."""
    app = _run("pages/chat.py")
    assert not app.exception, app.exception
    assert any("sign in" in warning.value.lower() for warning in app.warning)


def test_chat_page_renders_for_an_authenticated_user():
    app = _run(
        "pages/chat.py",
        {
            "access_token": "a-token",
            "username": "alice",
            "session_id": "abc123",
            "chat_history": [
                ("user", "hello", []),
                ("assistant", "hi there", [{"source": "a.pdf", "snippet": "x"}]),
            ],
        },
    )
    assert not app.exception, app.exception
    assert any("Adaptive RAG Chat" in title.value for title in app.title)
    assert any("alice" in caption.value for caption in app.caption)


def test_chat_page_does_not_expose_server_logs():
    app = _run(
        "pages/chat.py",
        {"access_token": "a-token", "username": "alice", "session_id": "abc"},
    )
    assert not app.exception, app.exception
    assert "app.log" not in str(app.get("expander"))
