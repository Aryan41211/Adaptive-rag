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
from prometheus_client.metrics import MetricWrapperBase

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
SCRIPTS = [DEPLOY / "backup.sh", DEPLOY / "restore.sh"]
COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"


def compose():
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_exists(script):
    assert script.is_file()


def _working_bash():
    """
    A bash that actually runs, or None.

    On Windows `shutil.which("bash")` finds System32\bash.EXE, which is the
    WSL launcher rather than a shell. It exists, so an `is None` check passes,
    and then every invocation fails with "execvpe(/bin/bash) failed" - a
    red test that says nothing about the scripts under test.
    """
    bash = shutil.which("bash")
    if bash is None:
        return None
    probe = subprocess.run(
        [bash, "-c", "exit 0"], capture_output=True, text=True, timeout=30
    )
    return bash if probe.returncode == 0 else None


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_is_syntactically_valid(script):
    """A syntax error here surfaces during a recovery, at the worst moment."""
    bash = _working_bash()
    if bash is None:
        pytest.skip("no working bash on this host")

    result = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_bash_detection_reports_the_windows_wsl_shim_as_unusable(monkeypatch):
    """
    Guards the skip above. Without this, the suite cannot tell a real syntax
    error from a host with no shell.
    """
    shim = Path(r"C:\Windows\System32\bash.EXE")

    def _fake_which(_name):
        return str(shim) if shim.exists() else None

    monkeypatch.setattr(shutil, "which", _fake_which)

    if not shim.exists():
        pytest.skip("not on Windows")
    assert _working_bash() is None, "the WSL launcher was treated as a shell"


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


def test_render_declares_the_web_search_key_the_graph_routes_to():
    """
    The agent graph routes questions to a web_search node. With no
    TAVILY_API_KEY that node returns "web search is not configured on this
    deployment" to every user who asks anything current, and the graph's
    web_search -> generate edge is dead code on the one deployment this
    repository documents as production.

    Declared with sync:false, so Render prompts for it at deploy time and the
    choice is visible. Left blank it degrades exactly as it does now.
    """
    assert "TAVILY_API_KEY" in render_env(
        "adaptive-rag-api"
    ), "the web_search branch is unreachable on the Render blueprint"


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
    The API carries the unauthenticated /metrics/prometheus endpoint and the
    OpenAPI schema (ENABLE_API_DOCS defaults to true). Published on 0.0.0.0 it
    would be reachable directly, bypassing Caddy entirely.
    """
    ports = compose()["services"]["api"]["ports"]
    assert "127.0.0.1:8000:8000" in ports


def test_uvicorn_is_not_started_with_a_wildcard_forwarded_allow_list():
    """
    The single most important line in the image.

    --forwarded-allow-ips governs request.client.host. With "*", uvicorn's
    ProxyHeadersMiddleware rewrites that field from the client-supplied
    X-Forwarded-For *before* the application sees the request, so src.api.
    ratelimit._client_ip() ends up validating the attacker's own value against
    TRUSTED_PROXIES and then returning it. Every caller could pick a fresh key
    per request and never be rate limited, which is the only thing standing
    between /auth/login and credential guessing.

    The trust decision must be made by the application from the raw socket
    peer, so uvicorn must not touch the forwarded headers at all.
    """
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "--forwarded-allow-ips" not in dockerfile, (
        "uvicorn is trusting client-supplied forwarded headers; the rate limiter "
        "reads request.client.host, which uvicorn has already overwritten"
    )
    assert '--forwarded-allow-ips "*"' not in dockerfile


@pytest.mark.parametrize(
    "variable",
    ["TRUSTED_PROXIES", "METRICS_TOKEN"],
    ids=["rate-limit proxy trust", "metrics token"],
)
def test_compose_forwards_the_settings_the_application_actually_reads(variable):
    """
    A variable documented in .env.example but absent from the api service's
    environment is silently ignored: Compose only passes what it lists, so the
    documented value never reaches the container and the setting keeps its
    default. TRUSTED_PROXIES defaulting to empty means every request looks
    like it came from one caller and shares a single rate-limit bucket.
    """
    environment = compose()["services"]["api"].get("environment", {})
    assert (
        variable in environment
    ), f"{variable} is documented but never passed to the api container"


def test_compose_mounts_the_secrets_directory_the_application_reads():
    """
    src.core.config points pydantic-settings at /run/secrets, and the README
    and .env.production.example both tell operators to put credentials in
    Docker secrets. Nothing ever mounted that directory, so a file placed
    there was invisible and the stack silently fell back to .env values -
    worse than not offering file-based secrets at all, because the operator
    believes the rotation took effect.
    """
    volumes = compose()["services"]["api"].get("volumes", [])
    mounts = [volume for volume in volumes if "/run/secrets" in str(volume)]
    assert mounts, "/run/secrets is read by src.core.config but never mounted"
    for mount in mounts:
        # Read-only: the container has no business rewriting a credential.
        assert str(mount).rstrip().endswith(":ro"), f"{mount} is writable"


def test_the_secrets_mount_is_optional_rather_than_required():
    """
    Long syntax with `required: true` would fail startup when the directory is
    absent, breaking the documented out-of-the-box first run. The short
    bind-mount form creates an empty root-owned directory instead, which is
    the correct "no secrets configured" default.
    """
    volumes = compose()["services"]["api"].get("volumes", [])
    secret_mounts = [volume for volume in volumes if "/run/secrets" in str(volume)]
    assert secret_mounts
    for mount in secret_mounts:
        assert not isinstance(
            mount, dict
        ), f"{mount} uses long syntax, which would make the directory required"


def test_credentials_written_to_the_secrets_directory_cannot_be_committed():
    """
    The README tells operators to write real credentials into
    deploy/secrets/. If that directory is not ignored, following the
    documentation is enough to put a live API key into git history, where
    removing it later does not remove it.

    Asserted through `git check-ignore` rather than by reading .gitignore,
    because what matters is the effective result after all the rules are
    applied in order - not what any single line appears to say.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not available")

    probe = "deploy/secrets/probe_credential_check_only"
    completed = subprocess.run(
        ["git", "check-ignore", "-q", probe],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        timeout=60,
    )
    assert completed.returncode == 0, (
        f"{probe} is not gitignored; following the documented Secrets section "
        "would put a live credential into the index"
    )


def test_the_secrets_directory_survives_a_fresh_clone():
    """
    The compose bind mount targets ./deploy/secrets. If the directory is
    ignored wholesale - rather than ignored except for a placeholder - a fresh
    clone has no such directory, and Compose creates it root-owned, which then
    cannot be written to by the operator who needs to add a credential.
    """
    assert (DEPLOY / "secrets" / ".gitkeep").is_file(), (
        "deploy/secrets/.gitkeep must be tracked so the directory exists in a "
        "fresh clone and the bind mount resolves to a writable path"
    )


def test_worker_count_matches_the_metrics_processing_mode():
    """
    The multiprocess Prometheus registry only engages when
    PROMETHEUS_MULTIPROC_DIR names a real directory. With one worker the
    default registry is correct and the variable is left empty, which is what
    this pins: raising the worker count without also setting that variable
    makes /metrics report only whichever worker happened to answer the scrape,
    and the numbers look plausible while being wrong.
    """
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "--workers" in dockerfile, "no --workers setting in the Dockerfile CMD"
    count = dockerfile.split("--workers")[1].split()[0]
    assert count == "1", (
        f"the image runs {count} workers; PROMETHEUS_MULTIPROC_DIR must be set "
        "before start or per-process metrics will be incomplete"
    )

    environment = compose()["services"]["api"].get("environment", {})
    assert "PROMETHEUS_MULTIPROC_DIR" in environment, (
        "the multiprocess directory is not forwarded, so scaling past one "
        "worker cannot be configured without editing compose"
    )
    # Empty by default, which is correct while the image runs a single worker.
    assert str(environment["PROMETHEUS_MULTIPROC_DIR"]).endswith(":-}"), (
        "PROMETHEUS_MULTIPROC_DIR must default to empty; the single-worker "
        "image should use the default registry"
    )


def test_stale_multiprocess_metric_files_are_cleared_on_start():
    """
    prometheus_client writes per-process counter shards into
    PROMETHEUS_MULTIPROC_DIR and never removes them. A directory that survives
    a restart makes every run re-add the previous run's totals, so counters
    climb forever while still looking like real data.
    """
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert (
        "PROMETHEUS_MULTIPROC_DIR" in dockerfile
    ), "the Dockerfile never references the multiprocess directory"
    cmd = dockerfile.split("CMD [", 1)[1]
    assert "-delete" in cmd or "-exec rm" in cmd, (
        "the multiprocess directory is not cleared before uvicorn starts, so "
        "counters accumulate across restarts"
    )


def test_prometheus_scrapes_the_endpoint_the_stack_exposes():
    """
    Alert rules are only useful if the target they evaluate actually has data.
    """
    prometheus = (DEPLOY / "prometheus" / "prometheus.yml").read_text(encoding="utf-8")
    assert "api:8000" in prometheus, "prometheus does not scrape the api service"
    assert "metrics/prometheus" in prometheus


def test_alert_rules_are_loaded_and_cover_the_failing_signals():
    """
    An alerting rule file that exists but is not referenced by prometheus.yml is
    never evaluated, and Prometheus starts up perfectly happily - the failure
    mode is silence. These checks make the wiring explicit rather than trusting
    the mount to line up.
    """
    prometheus = (DEPLOY / "prometheus" / "prometheus.yml").read_text(encoding="utf-8")
    rules = yaml.safe_load((DEPLOY / "prometheus" / "alerts.yml").read_text("utf-8"))

    # Parse rather than substring-match: "rule_files" appearing anywhere in the
    # file says nothing about whether the shipped rules are the ones loaded.
    rule_files = yaml.safe_load(prometheus).get("rule_files") or []
    assert rule_files, "prometheus.yml has no rule_files section"
    assert any(
        "alerts.yml" in str(path) for path in rule_files
    ), f"rule_files {rule_files} never loads the shipped alerts.yml"

    alerts = [rule for group in rules["groups"] for rule in group["rules"]]
    assert alerts, "no alert rules defined"
    for rule in alerts:
        assert rule.get("expr", "").strip(), f"{rule.get('alert')} has no expression"
        assert rule.get("for", "").strip(), (
            f"{rule.get('alert')} fires instantly; alert fatigue trains people "
            "to ignore the channel"
        )
        assert {"alert", "expr"} <= set(rule), f"{rule} is missing required fields"

    names = {rule["alert"] for rule in alerts}
    # The signals that matter here: the service is down, it is erroring, it is
    # slow, it is rejecting uploads, or nobody is using it.
    assert any(
        "upload" in name.lower() for name in names
    ), "no alert covers document upload, the feature the service exists for"
    assert any("error" in name.lower() for name in names)
    assert any("down" in name.lower() for name in names)
    assert any(
        "memory" in name.lower() or "disk" in name.lower() for name in names
    ), "no host saturation alert; MongoDB and Qdrant fail before the OOM killer"


def test_rule_groups_only_use_keys_prometheus_accepts():
    """
    Prometheus unmarshals a rule group strictly. An unrecognised key fails
    `promtool check rules` and takes the entire file with it, so one stray key
    silently disables every rule in it rather than just the broken one.

    `files` belongs to prometheus.yml's rule_files, never to a group. Written
    inside a group it looks plausible enough to survive review and is inert
    until the container is restarted with the rules mounted.
    """
    rules = yaml.safe_load((DEPLOY / "prometheus" / "alerts.yml").read_text("utf-8"))
    allowed = {"name", "interval", "limit", "labels", "rules"}
    for group in rules["groups"]:
        unexpected = set(group) - allowed
        assert not unexpected, (
            f"rule group {group.get('name')!r} uses keys Prometheus rejects: "
            f"{sorted(unexpected)}"
        )


def test_every_rag_metric_an_alert_reads_is_actually_exported():
    """
    A PromQL expression naming a series the application never exports is
    syntactically valid and permanently empty, so the rule looks configured
    and silently never fires. That is the worst possible alerting failure:
    the channel stays quiet because nothing was ever compared to a threshold.

    A typo is all it takes - `rag_request_latency_seconds` against the real
    `rag_request_duration_seconds` - which is exactly the mistake this check
    was written for.
    """
    import re

    from src.api import metrics as metrics_module

    exported = set()
    for name in dir(metrics_module):
        metric = getattr(metrics_module, name, None)
        if isinstance(metric, MetricWrapperBase):
            # A Counter is declared as `rag_requests_total` but stores its stem
            # internally, while a Gauge keeps its name verbatim - so both
            # spellings are added. `_bucket`/`_sum`/`_count` are generated by
            # the client rather than declared, and PromQL uses them directly.
            exported.add(metric._name)
            exported.add(f"{metric._name}_total")
            exported.add(f"{metric._name}_bucket")

    rules = yaml.safe_load((DEPLOY / "prometheus" / "alerts.yml").read_text("utf-8"))
    referenced = set()
    for group in rules["groups"]:
        for rule in group["rules"]:
            referenced |= set(re.findall(r"rag_[a-z_]+", rule["expr"]))

    unknown = {name for name in referenced if name not in exported}
    assert not unknown, f"alerts reference metrics the app never exports: {unknown}"
