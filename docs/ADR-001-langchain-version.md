# ADR-001: Exact pins for the LangChain / LangGraph family

- **Status:** Accepted
- **Date:** 2026-08-30
- **Applies to:** `requirements.txt`, `requirements.lock.txt`

## Context

`requirements.txt` pins the LangChain and LangGraph packages with `==` rather
than `>=`. This is deliberate and unusual; most projects relax their upper
bounds, and a reviewer looking at this file will reasonably ask why.

The reason is that this family ships breaking changes in **patch** and
**minor** releases, not just major ones. The specific case that forced the
decision:

- `langchain-google-genai` 2.1+ requires `langchain-core>=0.3.75`. The 3.x and
  4.x lines require `langchain-core` 1.x.
- The agent API used throughout `src/rag/` — `create_react_agent`, the state
  graph, and the callback interfaces — is the 0.3.x-era API.

So the packages are not independently upgradable. Bumping
`langchain-google-genai` alone forces `langchain-core` forward, and that in
turn drags the agent API out from under `src/rag/`. The pins have to move
together or not at all.

There is a second, quieter reason. `langgraph` 0.5.x changed the checkpoint
serializer behaviour that this project's conversation history relies on. A
patch bump there silently altered how messages round-tripped through MongoDB.

## Decision

Pin the whole family exactly in `requirements.txt`, and record the resolution
of the full transitive closure in `requirements.lock.txt`.

The current pins are:

| Package | Pin | Constraint it satisfies |
| --- | --- | --- |
| `langchain` | `0.3.27` | 0.3.x line |
| `langchain-core` | `0.3.72` | Must stay `<0.3.75` for `langchain-google-genai` 2.0.11 |
| `langchain-community` | `0.3.27` | Must match the `langchain` 0.3.x line |
| `langchain-openai` | `0.3.28` | Must match `langchain-core` 0.3.x |
| `langchain-ollama` | `0.3.5` | Must match `langchain-core` 0.3.x |
| `langchain-google-genai` | `2.0.11` | Newest 2.x supporting `langchain-core` 0.3.x |
| `langchain-text-splitters` | `0.3.9` | Must match `langchain-core` 0.3.x |
| `langgraph` | `0.5.4` | Checkpoint serializer behaviour relied on by `src/memory/` |

`langchain-google-genai` is the tightest constraint in the set: 2.1+ is
incompatible, so it is what pins `langchain-core` below 0.3.75, which in turn
holds the rest of the family in place.

`scripts/check_lock.py` runs in CI and fails the build when `requirements.txt`
and `requirements.lock.txt` disagree. That is the guard on this decision: the
lock file is what the Docker image installs, so drift between the two means a
deployed image behaves differently from a developer's virtualenv.

## Consequences

- Upgrading any package in the family is a deliberate, tested change to all of
  them at once — not a routine dependency bump. Expect to run the full suite
  plus `tests_live` before moving.
- `pip install -r requirements.txt` will not pick up security or bug fixes in
  this family automatically. Read release notes for these packages before
  assuming an upgrade is routine.
- Everything *outside* this family is deliberately unpinned or loosely bounded.
  This ADR covers the LangChain/LangGraph family only; other dependencies
  follow ordinary semver practice.

## How to revisit

Treat an upgrade as an ADR amendment, not a version edit. The bar:

1. `langchain-core` stays within a line whose agent API matches `src/rag/`.
2. The full suite passes, including the checkpoint round-trip tests in
   `src/memory/`.
3. `tests_live` passes against a real provider, because the callback and usage
   paths are exactly what the live probes cover.
4. `scripts/check_lock.py` passes with a regenerated lock file.