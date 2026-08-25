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

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
SCRIPTS = [DEPLOY / "backup.sh", DEPLOY / "restore.sh"]


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
