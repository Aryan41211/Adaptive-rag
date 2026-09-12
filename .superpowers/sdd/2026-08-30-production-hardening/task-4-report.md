# Task 4: Add Rate Limit Headers to Responses

## Status: DONE

## Changes Made

### `src/api/ratelimit.py`
- Added `_get_counter()` async function that reads the current rate limit counter value without incrementing it (supports both MongoDB and in-memory backends).
- Added `rate_limit_headers()` async function that computes `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers for a given user and endpoint.
- Modified `UserRateLimit.__call__()` to store the authenticated user on `request.state.user` so downstream middleware can access it.

### `src/api/routes.py`
- Cleaned up: removed `Response` import and `rate_limit_headers` import (headers are now set by middleware, not in route handlers).
- Route handler signatures restored to original form (no `response: Response` parameter needed).

### `src/main.py`
- Added `_RATE_LIMIT_ROUTES` mapping that maps paths to endpoint types ("query" or "upload").
- Added `add_rate_limit_headers` middleware that runs after the response is generated. It reads the user from `request.state.user` (set by the rate limit dependency) and attaches rate limit headers to the response. This works for both successful responses AND error responses (502 from OpenAI failures) because the middleware wraps the full request lifecycle including exception handlers.

### `tests/test_ratelimit_headers.py` (new)
- `test_rate_limit_headers_on_query`: Verifies that query responses include all three `x-ratelimit-*` headers, regardless of whether the upstream call succeeds (200) or fails (502).
- `test_rate_limit_headers_are_integers`: Verifies that header values are valid integers.

## Architecture Decision

The rate limit headers are set via an HTTP middleware rather than in individual route handlers. This was necessary because exception handlers (which handle 502/OpenAI errors) create new response objects that bypass route-level header setting. The middleware approach ensures headers are present on ALL responses from rate-limited endpoints, including error responses.

## Tests
- 2 new tests: both pass
- Full suite: 504 passed (1 pre-existing skip in `test_deploy_scripts.py` due to WSL unavailability on Windows)

## Commit
- Commit pending (user to run `git add` + `git commit`)
