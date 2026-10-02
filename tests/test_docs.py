"""
Documentation links.

A documentation link that 404s is worse than no link: it reads as authoritative
and sends the reader to a dead end, so it quietly erodes trust in everything
else the docs say. These are checked in CI because the referenced files move and
get renamed as the project changes, and nothing else notices.

Only repository-relative links are checked. External URLs are deliberately not
fetched: that would make CI depend on the network and on third parties staying
up, turning a documentation fix into a flaky build.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# [text](target) plus the reference-style [text]: target form.
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)|^\[[^\]]+\]:\s*(\S+)", re.MULTILINE)


def markdown_files() -> list[Path]:
    """
    Tracked markdown files only.

    Globbing the filesystem picks up .venv, build output and any other
    worktree checked out beside the project, whose links resolve against that
    copy rather than this one. Asking git for the tracked set keeps the check
    scoped to what the repository actually ships.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not available, cannot list tracked files")

    completed = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        pytest.skip(f"git ls-files failed: {completed.stderr.strip()[:200]}")

    files = [ROOT / line for line in completed.stdout.splitlines() if line.strip()]
    return sorted(path for path in files if path.is_file())


def relative_targets(markdown: Path) -> list[str]:
    """Every repository-relative link target in one file."""
    targets = []
    for match in MARKDOWN_LINK.finditer(markdown.read_text(encoding="utf-8")):
        target = match.group(1) or match.group(2)
        if not target:
            continue
        # Skip http(s), mailto, in-page anchors, and badges.
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        # Strip the fragment and any title after a space.
        target = target.split("#", 1)[0].split(" ", 1)[0].strip()
        if target:
            targets.append(target)
    return targets


def test_the_repository_has_documentation_to_check():
    """Guards the guard: an empty glob would make every other test vacuous."""
    assert len(markdown_files()) > 3


@pytest.mark.parametrize("markdown", markdown_files(), ids=lambda p: p.name)
def test_every_relative_link_resolves(markdown):
    broken = []
    for target in relative_targets(markdown):
        # A link is relative to the file that contains it.
        resolved = (markdown.parent / target).resolve()
        if not resolved.exists():
            broken.append(f"{target} (from {markdown.relative_to(ROOT).as_posix()})")
    assert not broken, "broken documentation links:\n" + "\n".join(broken)


@pytest.mark.parametrize("markdown", markdown_files(), ids=lambda p: p.name)
def test_referenced_paths_in_prose_exist(markdown):
    """
    Bare paths mentioned in backticks - the style used throughout this
    repository's docs for "the file you need to edit" - rot the same way links
    do, and are easier to miss when reading a diff.
    """
    referenced = re.findall(
        r"`((?:docs|deploy|scripts|src|tests|streamlit_app|\.github)/[\w./-]+)`",
        markdown.read_text(encoding="utf-8"),
    )
    missing = []
    for target in sorted(set(referenced)):
        if any(character in target for character in "*"):
            continue
        if not (ROOT / target).exists():
            missing.append(f"{target} (from {markdown.relative_to(ROOT).as_posix()})")
    assert not missing, (
        "documentation references paths that do not exist:\n" + "\n".join(missing)
    )


def test_root_level_markdown_filenames_mentioned_anywhere_exist():
    """
    An index that lists documents which do not exist is worse than no index:
    every entry is a dead end, and a reader concludes the rest of the
    documentation is unreliable too.

    This catches the bolded-table form specifically. DOCUMENTATION_INDEX.md
    listed three files that were never created, and a link check missed them
    because `**NAME.md**` is not a link.
    """
    phantom = []
    for markdown in markdown_files():
        for name in re.findall(
            r"\*\*([A-Z0-9_-]+\.md)\*\*", markdown.read_text("utf-8")
        ):
            if not (ROOT / name).exists():
                phantom.append(
                    f"{name} (claimed in {markdown.relative_to(ROOT).as_posix()})"
                )

    assert not phantom, "documentation lists files that do not exist:\n" + "\n".join(
        phantom
    )


@pytest.mark.parametrize(
    "source",
    ["requirements.txt", "requirements.lock.txt", "requirements-dev.txt", "Dockerfile"],
    ids=lambda p: p,
)
def test_paths_referenced_outside_the_markage_exist(source):
    """
    The version-pinning comment in requirements.txt points at an ADR that was
    never written, so the one document explaining why those pins exist pointed
    nowhere. The same class of rot affects the Dockerfile and the lock files,
    and none of it is covered by a markdown link check.
    """
    referenced = re.findall(
        r"(?:#|//)\s*(?:See\s+)?((?:docs|deploy|scripts|src|tests|streamlit_app|\.github)/[\w./-]+\.md)",
        (ROOT / source).read_text(encoding="utf-8"),
    )
    missing = [
        target for target in sorted(set(referenced)) if not (ROOT / target).exists()
    ]
    assert not missing, f"{source} references docs that do not exist: {missing}"


# The tools this project actually uses. A documentation file naming anything
# else is describing a setup nobody can reproduce.
TOOLS_IN_USE = {
    "pytest",
    "ruff",
    "pip",
    "python",
    "docker",
    "docker compose",
    "git",
    "curl",
    "uvicorn",
    "streamlit",
    "mongosh",
    "mongodump",
    "mongorestore",
    "promtool",
    # External binaries this project expects an operator to have. They are not
    # pip dependencies, so nothing in requirements*.txt would vouch for them,
    # and the docs legitimately invoke them.
    "ollama",
    "sudo",
}

# Shell builtins and anything that is not really a program invocation.
SHELL_BUILTINS = {
    "cd",
    "source",
    "export",
    "set",
    "echo",
    "printf",
    "mkdir",
    "chmod",
    "chown",
    "cp",
    "mv",
    "rm",
    "cat",
    "grep",
    "sed",
    "awk",
    "tar",
    "ls",
    "ps",
    "kill",
    "docker-compose",
    "openssl",
    "sha256sum",
    "systemctl",
    "sudo",
    "tee",
    "xargs",
    "jq",
    "env",
    "exit",
    "test",
    "sleep",
    "date",
}


def test_docs_do_not_recommend_tools_the_project_does_not_use():
    """
    The README told contributors to run `flake8 src/`. flake8 is not in any
    requirements file and the project uses ruff, so the documented workflow
    fails immediately on a clean checkout - and the first person to follow it
    concludes the contributing guide is broken.

    Only the first word of a line inside a fenced shell block is treated as a
    command. Scanning every backticked token instead would flag ordinary
    identifiers like `gpt-4o` or `username`, which are documentation, not
    instructions.

    Membership of the allowlists above is the ONLY exemption. An earlier
    version also skipped any command that `shutil.which` could resolve, which
    quietly disarmed the whole check: on a developer machine with black,
    flake8, isort, mypy and ollama installed globally it passed, and on
    ubuntu-latest, where none of them exist, it failed on exactly those five.
    The same commit was green locally and red in CI. Do not reintroduce
    "installed on this machine" as a synonym for "part of this project" - add
    external binaries to TOOLS_IN_USE explicitly instead.
    """
    offenders = []
    for markdown in markdown_files():
        for block in re.findall(
            r"```(?:bash|sh|shell|console)\n(.*?)```",
            markdown.read_text(encoding="utf-8"),
            re.DOTALL,
        ):
            for line in block.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "$")):
                    continue
                command = stripped.split()[0]
                # A continuation line (`-H "x: y"`) is an argument to the
                # command above it, not a command in its own right.
                if command.startswith("-"):
                    continue
                # An assignment or a JSON body line is data, not a command.
                if "=" in command or command.startswith(
                    ("./", "/", '"', "{", "}", "]", "[")
                ):
                    continue
                if command in TOOLS_IN_USE or command in SHELL_BUILTINS:
                    continue
                offenders.append(
                    f"{command} (in {markdown.relative_to(ROOT).as_posix()})"
                )

    assert not offenders, (
        "documentation recommends commands that are not part of this "
        f"project's toolchain: {sorted(set(offenders))}"
    )


def test_user_facing_docs_do_not_hardcode_a_test_count():
    """
    "454 tests" in a README is wrong the moment a test is added, and nothing
    fails when it goes stale - it just quietly misinforms. The number is one
    command away (`pytest --collect-only -q`), so the docs point at the
    command instead.

    Session reports and plans under .superpowers/ are excluded: they are
    point-in-time records of what was true then, and rewriting history in them
    would be wrong.
    """
    offenders = []
    pattern = re.compile(r"\b\d{3,}\s+(?:tests?|passed)\b")
    for markdown in markdown_files():
        # Session reports, plans and specs are point-in-time records of what
        # was true when they were written. Rewriting history in them would be
        # wrong, so their counts are left alone.
        if any(part in markdown.parts for part in (".superpowers", "superpowers")):
            continue
        for match in pattern.finditer(markdown.read_text(encoding="utf-8")):
            offenders.append(
                f"{match.group(0)!r} in {markdown.relative_to(ROOT).as_posix()}"
            )

    assert not offenders, (
        "documentation hardcodes a test count that will go stale: "
        f"{offenders}. Point at `pytest --collect-only -q` instead."
    )


def test_the_readme_does_not_hardcode_a_test_count():
    """The same rot in the one file every contributor reads first."""
    pattern = re.compile(r"\b\d{3,}\s+(?:tests?|passed)\b")
    matches = pattern.findall((ROOT / "README.md").read_text(encoding="utf-8"))
    assert not matches, (
        f"README.md hardcodes {matches}. A count is wrong as soon as a test "
        "is added, and nothing fails when it is. Point at "
        "`pytest --collect-only -q` instead."
    )
