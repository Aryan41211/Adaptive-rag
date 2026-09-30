"""
Deployment scripts.

A backup script that fails is worse than none, because it fails silently until
the day it is needed. These checks are static - the full backup, destroy and
restore cycle is exercised manually against a live stack - but they catch the
failure that would otherwise go unnoticed until a recovery.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
SCRIPTS = [DEPLOY / "backup.sh", DEPLOY / "restore.sh"]
COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"


def compose():
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_exists(script):
    assert script.is_file()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_is_syntactically_valid(script):
    """A syntax error here surfaces during a recovery, at the worst moment."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")

    result = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_fails_fast(script):
    """Without `set -e` a failed step is skipped and the backup looks fine."""
    source = script.read_text(encoding="utf-8")
    assert "set -euo pipefail" in source


def test_restore_requires_confirmation():
    """Restoring replaces live data; it must not be a single keystroke."""
    source = (DEPLOY / "restore.sh").read_text(encoding="utf-8")
    assert "FORCE" in source
    assert "Continue?" in source


def test_restore_replaces_rather_than_merges():
    """A merge would leave deleted records resurrected alongside restored ones."""
    source = (DEPLOY / "restore.sh").read_text(encoding="utf-8")
    assert "--drop" in source
    assert "priority=snapshot" in source


def test_backup_removes_the_snapshot_it_created():
    """Snapshots left inside the container would fill the volume over time."""
    source = (DEPLOY / "backup.sh").read_text(encoding="utf-8")
    assert "DELETE" in source


def test_caddyfile_is_present_for_the_tls_profile():
    assert (DEPLOY / "Caddyfile").is_file()


# --- authenticated data stores ---------------------------------------------
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_authenticates_to_mongodb(script):
    """
    An unauthenticated mongodump fails once the database requires credentials,
    which is exactly when a backup matters most.
    """
    source = script.read_text(encoding="utf-8")
    assert "MONGO_ROOT_PASSWORD" in source
    assert "--authenticationDatabase admin" in source


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_reads_credentials_without_sourcing_the_env_file(script):
    """`source .env` executes whatever is in it; these scripts must parse it."""
    source = script.read_text(encoding="utf-8")
    assert "env_file_value" in source
    assert ". ${ENV_FILE}" not in source
    assert f"source {'${ENV_FILE}'}" not in source


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_credentials_are_passed_as_an_array(script):
    """
    Expanded as a bare string, an empty credential set becomes empty-string
    arguments that mongodump rejects, breaking the unauthenticated case.
    """
    source = script.read_text(encoding="utf-8")
    assert "mongo_auth_args=()" in source
    assert '"${mongo_auth_args[@]}"' in source


def test_backup_verifies_the_archive_is_not_empty():
    """A zero-byte archive restores nothing and must not be reported as success."""
    source = (DEPLOY / "backup.sh").read_text(encoding="utf-8")
    assert "empty archive" in source


# --- retention --------------------------------------------------------------
def test_retention_is_opt_in():
    """Deleting backups by default is the wrong way round for this to be wrong."""
    source = (DEPLOY / "backup.sh").read_text(encoding="utf-8")
    assert 'RETAIN="${RETAIN:-0}"' in source
    assert '[ "${RETAIN}" -gt 0 ]' in source


def test_retention_only_matches_this_scripts_own_directory_layout():
    """Anything else the operator keeps under BACKUP_ROOT must survive."""
    source = (DEPLOY / "backup.sh").read_text(encoding="utf-8")
    assert "-name '????????T??????Z'" in source


def test_retention_never_removes_the_backup_just_written():
    source = (DEPLOY / "backup.sh").read_text(encoding="utf-8")
    assert '[ "${old}" = "${destination}" ] && continue' in source


def test_retention_runs_after_the_backup_completes():
    """A failed run must never be the reason an old backup was deleted."""
    source = (DEPLOY / "backup.sh").read_text(encoding="utf-8")
    assert source.index("manifest.txt") < source.index("Retention:")


# --- Caddy edge -------------------------------------------------------------
@pytest.mark.parametrize("path", ["/docs*", "/redoc*", "/openapi.json"])
def test_schema_is_not_served_at_the_public_edge(path):
    """The OpenAPI schema enumerates the whole attack surface."""
    caddyfile = (DEPLOY / "Caddyfile").read_text(encoding="utf-8")
    block = caddyfile.split(f"handle {path} {{", 1)
    assert len(block) == 2, f"no handler for {path}"
    body = block[1].split("}", 1)[0]
    assert "respond 404" in body
    assert "reverse_proxy" not in body


def test_prometheus_endpoint_is_not_served_at_the_public_edge():
    """
    /metrics/prometheus exposes internals and is unauthenticated; the API's
    /metrics handler is an exact match in Caddy, so without its own block this
    path would fall through to the UI handler.
    """
    caddyfile = (DEPLOY / "Caddyfile").read_text(encoding="utf-8")
    block = caddyfile.split("handle /metrics/prometheus* {", 1)
    assert len(block) == 2, "no handler for /metrics/prometheus*"
    body = block[1].split("}", 1)[0]
    assert "respond 404" in body
    assert "reverse_proxy" not in body


# --- provider configuration -------------------------------------------------
@pytest.mark.parametrize(
    "key",
    ["GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_EMBEDDING_MODEL"],
)
def test_api_receives_every_gemini_setting(key):
    """
    The provider is selected in the environment, so the key that makes that
    selection valid has to reach the container with it. LLM_PROVIDER=gemini
    with no GEMINI_API_KEY aborts at import (config validation) rather than
    degrading, so the documented production .env cannot start the stack.
    """
    assert key in compose()["services"]["api"]["environment"]


def test_compose_providers_come_from_the_env_file():
    """A value hardcoded in compose silently overrides the operator's .env."""
    env = compose()["services"]["api"]["environment"]
    for key in ("LLM_PROVIDER", "EMBEDDING_PROVIDER", "GEMINI_MODEL"):
        assert env[key].startswith("${"), f"{key} is pinned to {env[key]}"


# --- interpolation ----------------------------------------------------------
def test_no_variable_is_required_outside_an_opt_in_profile():
    """
    Compose interpolates the whole file before starting anything and before it
    looks at profiles, so `${VAR:?}` in an opt-in service aborts the default
    quickstart for an operator who never asked for that service.
    GRAFANA_ADMIN_PASSWORD had exactly this shape.
    """
    for name, service in compose()["services"].items():
        if not service.get("profiles"):
            continue
        for key, value in (service.get("environment") or {}).items():
            assert ":?" not in str(value), (
                f"{name} is opt-in but requires {key}={value}, which aborts the "
                f"default profile too"
            )


def test_the_default_profile_still_serves_the_application():
    """Gating observability behind a profile must not take the app with it."""
    services = compose()["services"]
    for name in ("api", "ui", "qdrant", "mongo"):
        assert not services[name].get("profiles"), f"{name} is now opt-in"


def test_observability_services_share_one_opt_in_profile():
    """Three separate profiles would mean three ways to start the same stack."""
    profiles = {
        tuple(compose()["services"][name].get("profiles", []))
        for name in ("grafana", "prometheus", "node-exporter")
    }
    assert profiles == {("observability",)}, profiles


def test_ollama_service_is_not_started_by_default():
    """Pulls a multi-gigabyte model image; local mode is opt-in via `ollama`."""
    assert compose()["services"]["ollama"].get("profiles") == ["ollama"]


# --- git index modes --------------------------------------------------------
@pytest.mark.parametrize(
    "script",
    [DEPLOY / "backup.sh", DEPLOY / "restore.sh", DEPLOY / "install-backup.sh"],
    ids=lambda p: p.name,
)
def test_script_is_executable_in_the_git_index(script):
    """
    systemd's ExecStart and the documented `./deploy/backup.sh` both need the
    executable bit. Git only records it if the file is staged as 100755, and
    nothing on a fresh clone sets it otherwise.
    """
    result = subprocess.run(
        ["git", "ls-files", "-s", "--", script.relative_to(DEPLOY.parents[0])],
        capture_output=True,
        text=True,
        cwd=DEPLOY.parents[0],
    )
    assert result.returncode == 0, result.stderr
    mode = result.stdout.split()[0] if result.stdout.split() else ""
    assert mode == "100755", f"{script.name} is staged {mode}, not 100755"


# --- Render blueprint -------------------------------------------------------
RENDER = Path(__file__).resolve().parents[1] / "render.yaml"


def render_service(name):
    for service in yaml.safe_load(RENDER.read_text(encoding="utf-8"))["services"]:
        if service["name"] == name:
            return service
    raise AssertionError(f"no {name} service in render.yaml")


def test_render_ui_pins_the_python_the_lock_file_was_resolved_for():
    """
    The UI installs requirements.lock.txt, which pip-compile resolved for a
    single interpreter. With no pin, Render picks its own default and the
    resolution does not hold.
    """
    lock_header = (
        (DEPLOY.parents[0] / "requirements.lock.txt")
        .read_text(encoding="utf-8")
        .splitlines()[:4]
    )
    locked = next(line for line in lock_header if "Python" in line)
    major_minor = locked.split("Python")[1].strip().split()[0]

    version = {
        v["key"]: v.get("value") for v in render_service("adaptive-rag-ui")["envVars"]
    }
    assert version.get("PYTHON_VERSION", "").startswith(major_minor), (
        f"lock is {locked.strip()!r} but PYTHON_VERSION is "
        f"{version.get('PYTHON_VERSION')!r}"
    )


def test_render_ui_pins_the_python_version_even_when_empty():
    """Render's `runtime: python` honours the variable, not a repo file."""
    assert "PYTHON_VERSION" in {
        v["key"] for v in render_service("adaptive-rag-ui")["envVars"]
    }


def test_render_api_pins_the_python_version_it_ships_in_the_image():
    """The image tag sets the API's interpreter; the blueprint must agree."""
    dockerfile = (DEPLOY.parents[0] / "Dockerfile").read_text(encoding="utf-8")
    tag = next(line for line in dockerfile.splitlines() if line.startswith("FROM "))
    image_python = tag.split("python:")[1].split("-")[0]

    version = {
        v["key"]: v.get("value") for v in render_service("adaptive-rag-api")["envVars"]
    }
    assert version.get("PYTHON_VERSION", "").startswith(image_python), (
        f"image is {tag.strip()!r} but PYTHON_VERSION is "
        f"{version.get('PYTHON_VERSION')!r}"
    )


def render_env(service_name):
    """Env vars of a service, keyed by name, with generateValue as the value."""
    return {
        v["key"]: v.get("value") or v.get("generateValue")
        for v in render_service(service_name)["envVars"]
    }


def test_render_api_restricts_accepted_hosts():
    """
    ALLOWED_HOSTS defaults to "*", which skips TrustedHostMiddleware entirely.
    Deferring it to a post-deploy manual step means the blueprint is created in
    the vulnerable state, and the README's "harden once it is up" is a step
    that has to be remembered.
    """
    assert render_env("adaptive-rag-api").get("ALLOWED_HOSTS", "*") != "*"


def test_render_metrics_endpoint_is_not_publicly_readable():
    """
    Render terminates TLS itself, so the Caddy block that 404s
    /metrics/prometheus does not exist on this path. Without a token the
    endpoint serves route and latency internals to anyone who finds the host.
    """
    assert render_env("adaptive-rag-api").get(
        "METRICS_TOKEN"
    ), "METRICS_TOKEN is unset on the public API"


# --- exposure boundary ------------------------------------------------------
@pytest.mark.parametrize(
    "service",
    [
        "api",
        "ui",
        "qdrant",
        "mongo",
        "prometheus",
        "grafana",
        "node-exporter",
    ],
)
def test_service_is_published_on_loopback_only(service):
    """
    A port published on 0.0.0.0 is reachable from anywhere that can route to
    the host, which on a cloud VM means the internet. Container-to-container
    traffic uses the compose network and ignores these mappings entirely; they
    exist for host tooling, so every one of them must be loopback-bound.

    The UI is here because it is the one service carrying a login form, and
    without the `tls` profile it was the only route to the app that bypassed
    Caddy's security headers entirely.
    """
    ports = compose()["services"][service].get("ports", [])
    assert ports, f"{service} publishes no ports"
    for port in ports:
        assert port.startswith("127.0.0.1:"), f"{service} publishes {port}"


def test_api_is_not_published_on_all_interfaces():
    """
    The API carries the unauthenticated /metrics/prometheus endpoint, the
    OpenAPI schema (ENABLE_API_DOCS defaults to true), and a uvicorn running
    with --forwarded-allow-ips, which trusts a spoofed X-Forwarded-For and
    defeats the rate limiter. Published on 0.0.0.0 it must therefore be fixed
    to the loopback interface.
    """
    ports = compose()["services"]["api"]["ports"]
    assert "127.0.0.1:8000:8000" in ports
