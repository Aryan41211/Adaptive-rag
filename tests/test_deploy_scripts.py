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
