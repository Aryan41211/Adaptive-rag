"""
Packaging.

The wheel is what every other consumer sees: a deploy platform building from
source, a colleague running `pip install -e .`, or a CI job importing the
package. When discovery is left to setuptools defaults it treats the `src`
directory as the package root, so the wheel ships `api`, `core`, `models` and
`tools` as top-level names, shadows unrelated packages on PyPI, and does not
even contain the application - `import src.main` fails after installing it.

These are static checks for speed; `tests/test_packaging.py::test_the_wheel
_actually_contains_the_application` builds a real wheel and is the backstop.
"""

import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_build_backend_is_declared():
    """
    Without [build-system], pip falls back to a legacy default that ignores
    the rest of pyproject.toml. It works by accident until it does not, and
    the failure appears on someone else's machine.
    """
    build_system = pyproject().get("build-system")
    assert build_system, "pyproject.toml has no [build-system] table"
    assert build_system.get("requires"), "[build-system] declares no requirements"
    assert build_system.get("build-backend")


def _discovery():
    """
    Normalise the two ways to be explicit about package layout. A literal
    `packages = [...]` list and a `[tool.setuptools.packages.find]` table both
    state the intent; what must never happen is stating nothing and letting
    setuptools guess.
    """
    tool = pyproject().get("tool", {}).get("setuptools", {})
    packages = tool.get("packages")
    if isinstance(packages, list):
        return "list", packages
    if isinstance(packages, dict) and isinstance(packages.get("find"), dict):
        # [tool.setuptools.packages.find] nests the table under "find".
        return "find", packages["find"]
    return "find", None


def test_package_discovery_is_explicit_rather_than_autodetected():
    """
    Autodetection is the actual bug: with src/__init__.py present, setuptools
    resolves `src` as the package root and ships its children at the top level.
    """
    kind, value = _discovery()
    assert value, (
        "packages are autodetected; setuptools will treat src/ as the package "
        "root and ship api/, core/, tools/ as top-level names"
    )
    if kind == "find":
        assert value.get("include"), "packages.find has no include pattern"
        assert value.get("where") is not None, "packages.find has no where root"


def test_the_wheel_ships_the_application_under_its_real_package_path():
    """
    Imports are written as `from src.api...` throughout, so the installed
    layout has to keep the `src` prefix. Flattening it produces a wheel that
    installs cleanly and then fails on first import.
    """
    kind, value = _discovery()
    if kind == "find":
        include = value.get("include") or []
        assert any(pattern.startswith("src") for pattern in include), (
            f"packages.find includes {include}, so no package keeps the 'src' "
            "prefix; imports would break after install"
        )
        # Discovery is only safe if it cannot sweep up non-library directories.
        assert not any(pattern in ("*", "src.*") for pattern in include), (
            f"packages.find includes {include}, which would ship streamlit_app "
            "and tests into the wheel"
        )
        return

    assert any(package.startswith("src") for package in value), (
        f"no package starts with 'src'; imports would break after install: {value}"
    )
    for package in value:
        assert package == "src" or package.startswith("src."), (
            f"{package} would be installed as a top-level name"
        )


def test_runtime_dependencies_come_from_the_locked_requirements_file():
    """
    A wheel whose metadata declares no dependencies installs into an
    environment that cannot import FastAPI. requirements.txt stays the single
    source of truth so the pin in the comment above it - and the LangChain
    family in ADR-001 - cannot drift away from the package metadata.
    """
    dynamic = pyproject()["project"].get("dynamic", [])
    assert "dependencies" in dynamic, (
        "dependencies are not declared; `pip install adaptive-rag` would "
        "succeed and then fail on the first import"
    )
    dependencies = pyproject()["tool"]["setuptools"]["dynamic"]["dependencies"]
    assert dependencies["file"] == ["requirements.txt"], (
        "dependencies must be read from requirements.txt, not restated here, "
        "or the pinned versions drift apart silently"
    )


def test_the_streamlit_ui_is_not_installed_as_a_library():
    """
    The UI is a Streamlit entrypoint, not an importable library. Shipping it
    would pull Streamlit into every install of the package and put a second
    copy of the app on sys.path.
    """
    packages = pyproject()["tool"]["setuptools"]["packages"]
    assert not any(package.startswith("streamlit_app") for package in packages), (
        "the Streamlit UI should not be packaged as a library"
    )


def test_the_wheel_actually_contains_the_application(tmp_path):
    """
    The backstop for everything above. Every other check here reads the
    configuration and trusts it; this one builds a real wheel and looks at
    what is inside, which is the only way to know the declared packages and
    the shipped layout actually agree.

    Skipped, not failed, when `build` is not installed - otherwise a developer
    running only the runtime requirements sees a failure for something they
    were never asked to install. CI installs requirements-dev.txt, so this
    always runs there.
    """
    pytest.importorskip("build", reason="the build package is not installed")

    completed = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        top_level = (
            archive.read(next(n for n in names if n.endswith("top_level.txt")))
            .decode()
            .split()
        )
        metadata = archive.read(
            next(n for n in names if n.endswith("METADATA"))
        ).decode()

    assert top_level == ["src"], f"top-level packages are {top_level}, expected ['src']"
    assert "src/main.py" in names, "the application entrypoint is missing"
    assert not any(n.startswith("streamlit_app") for n in names)
    assert not any(n.startswith("tests") for n in names)

    # The entrypoint existing is not enough: an explicit `packages = [...]` list
    # once omitted `src.rag.backends`, the wheel still contained src/main.py,
    # and every other assertion above still passed. Compare the full module set
    # against the source tree so a missing subpackage cannot hide again.
    on_disk = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    missing = sorted(on_disk - set(names))
    assert not missing, (
        "modules present in src/ but absent from the wheel; an installed "
        f"distribution would fail to import them: {missing}"
    )

    # Metadata must match the pinned requirements, not a stale copy.
    assert "Requires-Dist: langchain==0.3.27" in metadata, (
        "the wheel no longer carries the pinned LangChain version"
    )


# Distributions that exist only on Windows. They reach a lock file as transitive
# dependencies of perfectly portable packages - pywin32 arrives via
# portalocker <- streamlit - so nothing about requirements.txt looks wrong. An
# unqualified pin then breaks `docker build`, because the Linux image tries to
# install a distribution that was never published for it.
WINDOWS_ONLY_DISTRIBUTIONS = {
    "pywin32",
    "pywin32-ctypes",
    "pywinpty",
    "nt-time",
    "pywin32-com",
}


def test_no_windows_only_pin_is_unqualified_in_the_lock():
    """
    `pip-compile` records whatever platform it ran on. Run on Windows it emits
    pywin32 with no marker, and the image build fails at the pip step with
    "Could not find a version that satisfies the requirement pywin32" - which
    reads like a network or index problem, not a platform problem, and costs a
    lot of time to diagnose from the error alone.
    """
    offenders = []
    lock = (ROOT / "requirements.lock.txt").read_text(encoding="utf-8")
    for raw in lock.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ";" in line:
            continue
        package = re.split(r"[=<>!\[]", line)[0].strip().lower()
        if package in WINDOWS_ONLY_DISTRIBUTIONS:
            offenders.append(line)
    assert not offenders, (
        "Windows-only distributions pinned without a sys_platform marker; the "
        f"Linux image cannot install these: {offenders}"
    )


def test_secrets_are_excluded_from_the_docker_build_context():
    """
    Not being copied into the final image is not the same as not being sent.
    Without an ignore rule the whole directory is uploaded to the Docker daemon
    on every build and persists in its build cache, which is a different (and
    much larger) blast radius than one image. Verified by dropping a canary in
    deploy/secrets/ and confirming `COPY . /ctx` no longer picks it up.
    """
    ignored = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for path in ("deploy/secrets/", "secrets/", ".env"):
        assert path in ignored, (
            f".dockerignore does not exclude {path}; credentials placed there "
            "are uploaded to the Docker daemon on every build"
        )


def test_secrets_are_ignored_by_git():
    """
    `git add -A` from a clean checkout should never be able to stage a
    credential, including one written into a directory nobody remembered to
    gitignore.
    """
    patterns = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "deploy/secrets/*" in patterns, ".gitignore no longer hides deploy/secrets/"
    assert "\nsecrets/" in patterns, ".gitignore no longer ignores a stray secrets/"
