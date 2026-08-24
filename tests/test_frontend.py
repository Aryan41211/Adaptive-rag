"""
Tests for the Streamlit frontend and its API client.

The page tests are static (AST-based) because Streamlit pages execute their
whole body on import. They guard the three defects that made the UI unusable:
a ``set_page_config`` call that was not first, ``switch_page`` targets that
did not match the real filenames, and inconsistent import paths between the
two pages.
"""

import ast
import json
from pathlib import Path

import pytest
import requests

from streamlit_app.utils import api_client

APP_DIR = Path(__file__).resolve().parents[1] / "streamlit_app"
PAGES = [APP_DIR / "home.py", APP_DIR / "pages" / "chat.py"]


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _streamlit_calls(tree: ast.Module) -> list[str]:
    """Return the ``st.<name>`` calls in source order."""
    names = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
        ):
            names.append((node.lineno, node.func.attr))
    return [name for _line, name in sorted(names)]


# --- page structure --------------------------------------------------------
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_pages_are_syntactically_valid(page):
    _tree(page)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_set_page_config_is_the_first_streamlit_call(page):
    """Streamlit raises if any other st.* call precedes set_page_config."""
    calls = _streamlit_calls(_tree(page))
    assert calls, f"{page.name} makes no Streamlit calls"
    assert calls[0] == "set_page_config", (
        f"{page.name} calls st.{calls[0]} before st.set_page_config"
    )


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_switch_page_targets_exist(page):
    """switch_page paths are resolved relative to the entrypoint directory."""
    tree = _tree(page)

    # Pages reference their targets through module-level constants.
    constants = {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }

    targets = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "switch_page"
            and node.args
        ):
            arg = node.args[0]
            if isinstance(arg, ast.Constant):
                targets.append(arg.value)
            elif isinstance(arg, ast.Name) and arg.id in constants:
                targets.append(constants[arg.id])
            else:  # pragma: no cover - guards against untestable indirection
                pytest.fail(f"{page.name}: switch_page target is not resolvable")
    assert targets, f"{page.name} has no switch_page calls"
    for target in targets:
        resolved = APP_DIR / target
        assert resolved.is_file(), (
            f"{page.name}: switch_page('{target}') does not exist"
        )
        # Filenames are case-sensitive on Linux containers.
        assert resolved.name in {p.name for p in resolved.parent.iterdir()}


def test_both_pages_import_the_api_client_the_same_way():
    """Mismatched import roots made one of the two pages fail on import."""
    roots = set()
    for page in PAGES:
        for node in ast.walk(_tree(page)):
            if isinstance(node, ast.ImportFrom) and "api_client" in (node.module or ""):
                roots.add(node.module)
    assert len(roots) == 1, f"inconsistent api_client import paths: {roots}"


def test_pages_do_not_render_server_logs():
    """The old Debug Logs expander exposed app.log to any visitor."""
    for page in PAGES:
        source = page.read_text(encoding="utf-8")
        assert "app.log" not in source


# --- API client ------------------------------------------------------------
class _Response:
    def __init__(self, status=200, payload=None, body=b"{}"):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload if payload is not None else {}
        self.text = str(self._payload)
        self.content = body

    def json(self):
        return self._payload


def test_every_request_sets_a_timeout(monkeypatch):
    """Without a timeout the UI hangs forever on a stalled backend."""
    seen = []

    def fake_post(url, **kwargs):
        seen.append(kwargs.get("timeout"))
        return _Response(200, {"access_token": "t", "username": "u", "answer": "a"})

    monkeypatch.setattr(requests, "post", fake_post)

    api_client.login("alice", "password")
    api_client.register("alice", "password")
    api_client.query_backend("q", "s1", "token")

    assert seen and all(isinstance(t, int) and t > 0 for t in seen)


def test_health_check_sets_a_timeout(monkeypatch):
    seen = {}

    def fake_get(url, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return _Response(200, {"status": "ok"})

    monkeypatch.setattr(requests, "get", fake_get)
    assert api_client.api_available() is True
    assert seen["timeout"] > 0


def test_connection_failure_becomes_a_readable_error(monkeypatch):
    def fake_post(url, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(api_client.ApiError) as exc:
        api_client.login("alice", "password")
    assert "Could not reach the API" in str(exc.value)


def test_timeout_becomes_a_readable_error(monkeypatch):
    def fake_post(url, **kwargs):
        raise requests.Timeout()

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(api_client.ApiError) as exc:
        api_client.query_backend("q", "s1", "token")
    assert "too long" in str(exc.value)


def test_error_detail_is_surfaced(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **kwargs: _Response(
            401, {"detail": "Incorrect username or password."}
        ),
    )

    with pytest.raises(api_client.ApiError) as exc:
        api_client.login("alice", "wrong")
    assert str(exc.value) == "Incorrect username or password."


def test_validation_errors_are_summarised(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **kwargs: _Response(
            422,
            {
                "detail": "Request validation failed.",
                "errors": [{"field": "password", "message": "too short"}],
            },
        ),
    )

    with pytest.raises(api_client.ApiError) as exc:
        api_client.register("alice", "x")
    assert "password" in str(exc.value)


def test_query_sends_the_bearer_token(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return _Response(200, {"answer": "the answer", "citations": []})

    monkeypatch.setattr(requests, "post", fake_post)

    answer, citations, usage = api_client.query_backend("q", "s1", "my-token")
    assert answer == "the answer"
    assert citations == []
    assert usage == {}
    assert captured["headers"]["Authorization"] == "Bearer my-token"
    assert captured["json"] == {"query": "q", "session_id": "s1"}


def test_upload_sends_description_and_token(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return _Response(200, {"filename": "a.txt", "chunks_indexed": 1})

    monkeypatch.setattr(requests, "post", fake_post)

    class _File:
        name = "a.txt"
        type = "text/plain"

        def getvalue(self):
            return b"content"

    api_client.upload_document(_File(), "my notes", "my-token")

    assert captured["headers"]["X-Description"] == "my notes"
    assert captured["headers"]["Authorization"] == "Bearer my-token"
    assert captured["files"]["file"][0] == "a.txt"


def test_client_targets_the_python_api_not_a_missing_rust_service():
    """The old client called a Rust service that does not exist here."""
    assert "8080" not in api_client.API_BASE_URL
    source = (APP_DIR / "utils" / "api_client.py").read_text(encoding="utf-8")
    assert "RUST_BASE_URL" not in source


def test_query_returns_citations(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **kwargs: _Response(
            200,
            {
                "answer": "grounded answer",
                "citations": [{"source": "a.pdf", "snippet": "x", "page": 2}],
            },
        ),
    )

    answer, citations, _usage = api_client.query_backend("q", "s1", "token")
    assert answer == "grounded answer"
    assert citations[0]["source"] == "a.pdf"


def test_logout_posts_the_token(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response(204, {}, body=b"")

    monkeypatch.setattr(requests, "post", fake_post)

    api_client.logout("my-token")
    assert captured["url"].endswith("/auth/logout")
    assert captured["headers"]["Authorization"] == "Bearer my-token"


def test_empty_body_responses_are_handled(monkeypatch):
    """A 204 carries no JSON; parsing it as JSON would raise."""
    monkeypatch.setattr(
        requests, "post", lambda url, **kwargs: _Response(204, {}, body=b"")
    )
    api_client.logout("my-token")  # must not raise


def test_list_documents_sends_the_token(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response(200, {"documents": [{"filename": "a.txt", "chunks": 2}]})

    monkeypatch.setattr(requests, "get", fake_get)

    documents = api_client.list_documents("my-token")
    assert documents[0]["filename"] == "a.txt"
    assert captured["headers"]["Authorization"] == "Bearer my-token"
    assert captured["timeout"] > 0


def test_delete_document_url_encodes_the_filename(monkeypatch):
    """A filename with spaces or slashes must not corrupt the request path."""
    captured = {}

    def fake_delete(url, **kwargs):
        captured["url"] = url
        return _Response(200, {"chunks_deleted": 3})

    monkeypatch.setattr(requests, "delete", fake_delete)

    assert api_client.delete_document("my report.pdf", "my-token") == 3
    assert "my%20report.pdf" in captured["url"]


def test_delete_document_surfaces_errors(monkeypatch):
    monkeypatch.setattr(
        requests,
        "delete",
        lambda url, **kwargs: _Response(404, {"detail": "No such document."}),
    )

    with pytest.raises(api_client.ApiError) as exc:
        api_client.delete_document("nope.txt", "my-token")
    assert "No such document" in str(exc.value)


def test_stream_query_decodes_sse_events(monkeypatch):
    class _Stream:
        status_code = 200
        ok = True
        content = b""

        def iter_lines(self, decode_unicode=False):
            yield "data: " + json.dumps({"type": "token", "text": "Hel"})
            yield ""
            yield "data: " + json.dumps({"type": "token", "text": "lo"})
            yield ""
            yield "data: " + json.dumps({"type": "done", "answer": "Hello"})

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(requests, "post", lambda url, **kwargs: _Stream())

    events = list(api_client.stream_query("q", "s1", "token"))
    assert [e["type"] for e in events] == ["token", "token", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Hello"


def test_stream_query_skips_malformed_frames(monkeypatch):
    """One bad frame must not abort an answer already in flight."""

    class _Stream:
        status_code = 200
        ok = True
        content = b""

        def iter_lines(self, decode_unicode=False):
            yield "data: {not valid json"
            yield "data: " + json.dumps({"type": "done", "answer": "ok"})

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(requests, "post", lambda url, **kwargs: _Stream())

    events = list(api_client.stream_query("q", "s1", "token"))
    assert [e["type"] for e in events] == ["done"]


def test_stream_query_requests_a_streaming_response(monkeypatch):
    """Without stream=True the client buffers and streaming is pointless."""
    captured = {}

    class _Stream:
        status_code = 200
        ok = True
        content = b""

        def iter_lines(self, decode_unicode=False):
            return iter(())

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return _Stream()

    monkeypatch.setattr(requests, "post", fake_post)
    list(api_client.stream_query("q", "s1", "token"))

    assert captured["stream"] is True
    assert captured["timeout"] > 0
    assert captured["headers"]["Authorization"] == "Bearer token"


def test_stream_query_surfaces_a_rejected_request(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **kwargs: _Response(429, {"detail": "Rate limit exceeded."}),
    )

    with pytest.raises(api_client.ApiError) as exc:
        list(api_client.stream_query("q", "s1", "token"))
    assert "Rate limit" in str(exc.value)
