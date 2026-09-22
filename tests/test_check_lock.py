"""Tests for the lock-file/requirements drift gate in scripts/check_lock.py.

The gate is what CI runs to stop the deployed image (installed from
requirements.lock.txt) from drifting away from the declared dependencies
(requirements.txt). These tests cover the name normalisation the gate relies
on, plus the whole `main()` gate against the repository's own files.
"""

import importlib.util
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_gate() -> types.ModuleType:
    """Import scripts/check_lock.py without requiring it on sys.path."""
    spec = importlib.util.spec_from_file_location(
        "check_lock", ROOT / "scripts" / "check_lock.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_lock = _load_gate()


# --- canonical() normalisation ---------------------------------------------


def test_canonical_strips_a_single_extra():
    """An extras* name and the bare name are the same package."""
    assert check_lock.canonical("uvicorn[standard]") == check_lock.canonical("uvicorn")


def test_canonical_strips_multiple_extras():
    assert check_lock.canonical("pkg[extra-one,extra-two]") == check_lock.canonical(
        "pkg"
    )


def test_canonical_normalises_hyphens_and_underscores():
    """PEP 503 treats - and _ as equivalent; that must keep working."""
    assert check_lock.canonical("my_pkg") == check_lock.canonical("my-pkg")


def test_canonical_strips_surrounding_whitespace():
    assert check_lock.canonical("  uvicorn  ") == check_lock.canonical("uvicorn")


def test_canonical_lowercases():
    assert check_lock.canonical("FastAPI") == check_lock.canonical("fastapi")


# --- read_lock() parsing ---------------------------------------------------


def test_read_lock_handles_extras_markers_and_padding(tmp_path):
    """Lock lines may carry extras*, an env marker, and loose spacing."""
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text(
        "uvicorn[standard]==0.39.0\n"
        'coverage==7.6.1 ; python_version >= "3.8"\n'
        "  pytest==  8.3.2  \n",
        encoding="utf-8",
    )
    pinned = check_lock.read_lock(lock)

    assert check_lock.canonical("uvicorn") in pinned
    assert check_lock.canonical("coverage") in pinned
    assert check_lock.canonical("pytest") in pinned
    assert str(pinned[check_lock.canonical("uvicorn")]) == "0.39.0"


# --- the CI gate itself ----------------------------------------------------


def test_main_gate_passes_for_the_repository():
    """The repository's own lock file must satisfy its requirements."""
    assert check_lock.main() == 0
