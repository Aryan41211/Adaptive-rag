"""
Golden dataset loading.

A case pairs a question with what a correct system should do with it: which
route it belongs on, which document should be retrieved, and which facts the
answer must contain. Keeping expectations declarative means a prompt change is
measured rather than argued about.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DATASET = DATA_DIR / "golden.yaml"


@dataclass(frozen=True)
class Document:
    """A source document indexed before the cases run."""

    filename: str
    description: str
    content: str


@dataclass(frozen=True)
class Case:
    """One question and what a correct answer looks like."""

    id: str
    question: str
    # Where the classifier should send it: index, general or search.
    expected_route: str
    # The document the answer should be drawn from, when route == "index".
    expected_source: str | None = None
    # Substrings a correct answer must contain, matched case-insensitively.
    must_include: list[str] = field(default_factory=list)
    # Substrings a correct answer must NOT contain. Used for questions the
    # documents cannot answer, where the failure mode is confident invention.
    must_not_include: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class Dataset:
    """Documents to index and the cases to run against them."""

    documents: list[Document]
    cases: list[Case]

    def __len__(self) -> int:
        return len(self.cases)


def load(path: Path | str = DEFAULT_DATASET) -> Dataset:
    """
    Load a golden dataset from YAML.

    Args:
        path: Dataset file.

    Returns:
        The parsed dataset.

    Raises:
        ValueError: If the file is malformed or a case is inconsistent.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    documents = [Document(**entry) for entry in raw.get("documents", [])]
    cases = [Case(**entry) for entry in raw.get("cases", [])]

    if not cases:
        raise ValueError(f"{path} defines no cases.")

    known = {document.filename for document in documents}
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"Duplicate case id: {case.id}")
        seen.add(case.id)

        if case.expected_route not in {"index", "general", "search"}:
            raise ValueError(f"Case {case.id}: unknown route '{case.expected_route}'.")
        if case.expected_source and case.expected_source not in known:
            raise ValueError(
                f"Case {case.id}: expected_source '{case.expected_source}' "
                "is not one of the dataset's documents."
            )

    return Dataset(documents=documents, cases=cases)
