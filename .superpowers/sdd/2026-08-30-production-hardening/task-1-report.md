# Task 1: Add prometheus-client dependency

## Status: DONE

## What was implemented
Added `prometheus-client>=0.20,<1` to `requirements.txt` under a new `# --- Observability ---` section after the existing `# --- Config & utilities ---` section.

Regenerated `requirements.lock.txt` via `pip-compile` which resolved `prometheus-client==0.26.0`.

## Verification results
- **Import test:** `from prometheus_client import Counter; print('ok')` — passed
- **Tests:** `pytest tests/ -x -q` — failed with pre-existing `ModuleNotFoundError: No module named 'src.api'` (confirmed identical failure without this change via `git stash`)

## Files changed
| File | Change |
|------|--------|
| `requirements.txt` | Added `# --- Observability ---` section with `prometheus-client>=0.20,<1` |
| `requirements.lock.txt` | Regenerated via `pip-compile` (added `prometheus-client==0.26.0` + dependency tree annotations) |

## Commit
- `a3ef08b` — `deps: add prometheus-client for metrics exposition`

## Self-review
- Lock file format changed from flat `pip freeze` style to `pip-compile` annotated style — this is expected per the task instructions.
- Dependency bounds (`>=0.20,<1`) are appropriate for the 0.x line of prometheus-client.
- No issues found.
