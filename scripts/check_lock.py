#!/usr/bin/env python3
"""
Verify requirements.lock.txt satisfies every constraint in requirements.txt.

The lock file is what the container image installs; requirements.txt is what
the project claims to depend on. Nothing keeps the two in step, and a drift
between them is invisible until a deployed image behaves differently from a
developer's virtualenv. This check makes that drift a build failure.

It deliberately does not resolve dependencies or reach the network: it only
answers "is every declared requirement pinned, and is each pin inside its
declared range?".

    python scripts/check_lock.py

Exits non-zero and names every mismatch when they disagree.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
LOCK = ROOT / "requirements.lock.txt"


def canonical(name: str) -> str:
    """Normalise a distribution name for comparison (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def read_lock(path: Path) -> dict[str, Version]:
    """
    Parse the pinned versions out of a lock file.

    Args:
        path: The lock file to read.

    Returns:
        Canonical package name to pinned version.

    Raises:
        SystemExit: If a line pins a version that cannot be parsed.
    """
    pinned: dict[str, Version] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # Drop any environment marker: markers narrow where a pin applies,
        # they do not change what it pins.
        spec = line.split(";", 1)[0].strip()
        if "==" not in spec:
            continue
        name, _, version = spec.partition("==")
        try:
            pinned[canonical(name)] = Version(version.strip())
        except InvalidVersion as exc:
            raise SystemExit(
                f"{path.name}:{lineno}: cannot parse version in {line!r}"
            ) from exc
    return pinned


def read_requirements(path: Path) -> list[Requirement]:
    """
    Parse the direct requirements, skipping comments and `-r` includes.

    Args:
        path: The requirements file to read.

    Returns:
        The declared requirements.
    """
    requirements = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        requirements.append(Requirement(line))
    return requirements


def main() -> int:
    """Compare the two files and report every disagreement."""
    pinned = read_lock(LOCK)
    declared = read_requirements(REQUIREMENTS)

    failures: list[str] = []
    for requirement in declared:
        key = canonical(requirement.name)
        version = pinned.get(key)
        if version is None:
            failures.append(
                f"{requirement.name}: declared in requirements.txt but not "
                f"pinned in {LOCK.name}"
            )
            continue
        # prereleases=True so a pinned release candidate is judged against the
        # declared range rather than silently excluded.
        if not requirement.specifier.contains(version, prereleases=True):
            failures.append(
                f"{requirement.name}: {LOCK.name} pins {version}, which is "
                f"outside the declared range '{requirement.specifier}'"
            )

    print(
        f"Checked {len(declared)} direct requirements "
        f"against {len(pinned)} pinned packages."
    )
    if failures:
        print(f"\n{len(failures)} mismatch(es):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nRegenerate the lock file, or correct the range in "
            "requirements.txt, so the two agree.",
            file=sys.stderr,
        )
        return 1

    print("requirements.lock.txt satisfies requirements.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
