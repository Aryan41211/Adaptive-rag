"""
FastAPI application entry point.

Wires logging, request correlation, error translation, health probes and the
API routers. Configuration is validated at import time, so a misconfigured
deployment fails during startup rather than on the first user request.
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.api import ratelimit
from src.api.auth_routes import router as auth_router
from src.api.routes import router as rag_router
from src.core.config import settings
from src.core.exceptions import AdaptiveRagError
from src.core.logger import configure_logging, get_logger, request_id_var
from src.db import mongo_client, revoked_tokens
from src.rag import vector_store

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of external resources."""
    logger.info(
        "Starting Adaptive RAG API v%s (vector_store=%s, persistence=%s, "
        "web_search=%s)",
        settings.APP_VERSION,
        settings.vector_backend,
        "mongodb" if settings.persistence_enabled else "in-memory",
        "enabled" if settings.web_search_enabled else "disabled",
    )

    if settings.RATE_LIMIT_ENABLED and not settings.persistence_enabled:
        logger.warning(
            "Rate limit counters are per-process: without MONGODB_URL the "
            "effective limit is multiplied by the number of workers."
        )

    if not settings.qdrant_enabled:
        logger.warning(
            "QDRANT_URL is not set: documents are held in this process only. "
            "They are lost on restart and the service must run a single "
            "worker."
        )

    if settings.persistence_enabled:
        if await mongo_client.ping():
            await mongo_client.ensure_indexes()
            # TTL indexes expire spent rate-limit counters and revocation
            # entries; without them both collections grow without bound.
            await ratelimit.ensure_indexes()
            await revoked_tokens.ensure_indexes()
        else:
            # Not fatal: the in-memory fallback keeps the service usable, but
            # the operator needs to know history will not survive a restart.
            logger.error(
                "MONGODB_URL is configured but unreachable. Chat history "
                "writes will fail until connectivity is restored."
            )

    yield

    await mongo_client.close_client()
    logger.info("Adaptive RAG API stopped")


app = FastAPI(
    title="Adaptive RAG API",
    version=settings.APP_VERSION,
    description=(
        "Agentic RAG service. All /rag endpoints require a bearer token "
        "obtained from /auth/login."
    ),
    lifespan=lifespan,
)


if settings.allowed_hosts != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

if settings.cors_origins:
    # Only added when origins are configured: the default of no cross-origin
    # access is correct for the server-rendered Streamlit UI.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Description"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach a correlation id to each request and its log records."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(AdaptiveRagError)
async def handle_domain_error(request: Request, exc: AdaptiveRagError) -> JSONResponse:
    """Translate domain errors into their declared HTTP status."""
    logger.info("%s: %s", type(exc).__name__, exc.message)

    headers: dict[str, str] | None = None
    if exc.status_code == 401:
        headers = {"WWW-Authenticate": "Bearer"}
    elif exc.status_code == 429:
        headers = {"Retry-After": str(getattr(exc, "retry_after", 60))}
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error": type(exc).__name__},
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a compact, non-leaking validation error body."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Request validation failed.",
            "errors": [
                {
                    "field": ".".join(str(p) for p in err["loc"][1:]),
                    "message": err["msg"],
                }
                for err in exc.errors()
            ],
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Log the failure in full, return a generic message to the client."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


app.include_router(auth_router)
app.include_router(rag_router)


@app.get("/", tags=["health"])
async def root() -> dict:
    """Report basic service identity."""
    return {
        "service": "Adaptive RAG API",
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/healthz", tags=["health"])
async def healthz() -> dict:
    """Liveness probe: the process is up and serving."""
    return {"status": "ok"}


@app.get("/readyz", tags=["health"])
async def readyz() -> JSONResponse:
    """
    Readiness probe reporting dependency status.

    Returns 200 when the service can serve traffic. MongoDB being unreachable
    is reported as degraded, not unready, because the in-memory fallback keeps
    the API functional.
    """
    persistence = "not-configured"
    if settings.persistence_enabled:
        persistence = "ok" if await mongo_client.ping() else "unreachable"

    # The vector store is the one dependency the service cannot work without,
    # so an unreachable one makes the instance genuinely not ready.
    vector_healthy, vector_detail = await run_in_threadpool(vector_store.health)

    body = {
        "status": "ok" if vector_healthy else "degraded",
        "vector_store": vector_detail,
        "persistence": persistence,
        "web_search": "enabled" if settings.web_search_enabled else "disabled",
        "version": settings.APP_VERSION,
    }
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if vector_healthy
            else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content=body,
    )
