"""
Evaluation CLI.

    python -m evals                     # run the default dataset
    python -m evals --json out.json     # also write machine-readable output
    python -m evals --fail-under 0.8    # non-zero exit below that pass rate

This calls the real model provider and costs money. It is deliberately not
part of the test suite.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from evals import dataset as dataset_module
from evals import report, runner
from evals.metrics import CaseResult


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Score the RAG pipeline against a golden dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=dataset_module.DEFAULT_DATASET,
        help="Path to the dataset YAML.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the full run to this JSON file.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="RATE",
        help="Exit non-zero if the pass rate falls below this (0-1).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case progress.",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    data = dataset_module.load(args.dataset)

    print(
        f"Running {len(data)} cases over {len(data.documents)} documents.\n"
        "This calls the model provider and will incur cost.\n"
    )

    def progress(result: CaseResult) -> None:
        if not args.quiet:
            mark = "." if result.passed else ("!" if result.error else "x")
            print(mark, end="", flush=True)

    results, summary = await runner.run(data, on_result=progress)
    if not args.quiet:
        print()

    print(report.render_text(results, summary))

    if args.json:
        written = report.write_json(results, summary, args.json)
        print(f"  Wrote {written}\n")

    if args.fail_under is not None and summary.pass_rate < args.fail_under:
        print(
            f"  Pass rate {summary.pass_rate:.0%} is below the required "
            f"{args.fail_under:.0%}.\n"
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Entry point.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv``.

    Returns:
        The process exit code.
    """
    args = _parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 - a CLI should not traceback
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
