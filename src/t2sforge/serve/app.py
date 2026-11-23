"""FastAPI gateway in front of the model server.

The model server (vLLM) only turns tokens into tokens. Everything that makes a text-to-SQL
*product* safe and observable lives here: schema prompting, SQL guarding, sandboxed execution,
self-correction, input validation and Prometheus metrics.

    uvicorn t2sforge.serve.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from t2sforge import __version__
from t2sforge.serve.backend import BackendError, ChatBackend
from t2sforge.serve.pipeline import DatabaseRegistry, TextToSQL, UnknownDatabaseError


class Settings(BaseModel):
    db_root: str = Field(default_factory=lambda: os.environ.get("T2S_DB_ROOT", "data/databases"))
    backend_url: str = Field(
        default_factory=lambda: os.environ.get("T2S_BACKEND_URL", "http://localhost:8000/v1")
    )
    model: str = Field(default_factory=lambda: os.environ.get("T2S_MODEL", "sql"))
    api_key: str | None = Field(default_factory=lambda: os.environ.get("T2S_BACKEND_API_KEY"))
    max_repairs: int = Field(default_factory=lambda: int(os.environ.get("T2S_MAX_REPAIRS", "1")))
    max_rows: int = Field(default_factory=lambda: int(os.environ.get("T2S_MAX_ROWS", "200")))


class QueryRequest(BaseModel):
    db_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=3, max_length=2000)
    evidence: str | None = Field(default=None, max_length=2000)
    execute: bool = True
    max_rows: int | None = Field(default=None, ge=1, le=10_000)


class AttemptOut(BaseModel):
    sql: str
    error: str | None
    stage: str


class QueryResponse(BaseModel):
    sql: str | None
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    error: str | None
    repaired: bool
    attempts: list[AttemptOut]
    latency_ms: dict[str, float]


class Metrics:
    """Own registry so tests (and multiple apps in one process) don't collide."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "t2s_requests_total", "Queries by outcome", ["outcome"], registry=self.registry
        )
        self.latency = Histogram(
            "t2s_latency_seconds",
            "Latency by stage",
            ["stage"],
            registry=self.registry,
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
        )
        self.repairs = Counter(
            "t2s_repairs_total", "Self-corrections attempted", ["trigger"], registry=self.registry
        )
        self.tokens = Counter("t2s_tokens_total", "Model tokens", ["kind"], registry=self.registry)


def create_app(settings: Settings | None = None, backend: ChatBackend | None = None) -> FastAPI:
    settings = settings or Settings()
    metrics = Metrics()
    backend = backend or ChatBackend(settings.backend_url, settings.model, api_key=settings.api_key)
    registry = DatabaseRegistry(settings.db_root)
    engine = TextToSQL(
        backend, registry, max_repairs=settings.max_repairs, max_rows=settings.max_rows
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await backend.aclose()

    app = FastAPI(title="text2sql-forge", version=__version__, lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(response: Response) -> dict[str, Any]:
        model_ready = await backend.ready()
        dbs = registry.list()
        ready = model_ready and bool(dbs)
        response.status_code = 200 if ready else 503
        return {"model_server": model_ready, "databases": len(dbs), "ready": ready}

    @app.get("/v1/databases")
    async def databases() -> dict[str, list[str]]:
        return {"databases": registry.list()}

    @app.get("/v1/databases/{db_id}/schema")
    async def schema(db_id: str) -> dict[str, str]:
        try:
            return {"db_id": db_id, "schema": registry.schema(db_id)}
        except UnknownDatabaseError:
            raise HTTPException(404, f"unknown database {db_id!r}") from None

    @app.post("/v1/query", response_model=QueryResponse)
    async def query(req: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        try:
            out = await engine.answer(
                req.db_id,
                req.question,
                evidence=req.evidence,
                execute=req.execute,
                max_rows=req.max_rows,
            )
        except UnknownDatabaseError:
            metrics.requests.labels("unknown_db").inc()
            raise HTTPException(404, f"unknown database {req.db_id!r}") from None
        except BackendError as exc:
            metrics.requests.labels("backend_error").inc()
            raise HTTPException(502, str(exc)) from None

        total = time.perf_counter() - started
        metrics.latency.labels("total").observe(total)
        metrics.latency.labels("generation").observe(out.generation_ms / 1000)
        metrics.latency.labels("execution").observe(out.execution_ms / 1000)
        metrics.tokens.labels("prompt").inc(out.prompt_tokens)
        metrics.tokens.labels("completion").inc(out.completion_tokens)
        for failed in out.attempts[:-1] if out.ok else out.attempts[: engine.max_repairs]:
            metrics.repairs.labels(failed.stage).inc()
        if out.ok:
            metrics.requests.labels("repaired" if out.repaired else "ok").inc()
        else:
            metrics.requests.labels(f"failed_{out.attempts[-1].stage}").inc()

        return QueryResponse(
            sql=out.sql,
            columns=out.columns,
            rows=out.rows,
            truncated=out.truncated,
            error=out.error,
            repaired=out.repaired,
            attempts=[AttemptOut(sql=a.sql, error=a.error, stage=a.stage) for a in out.attempts],
            latency_ms={
                "total": round(total * 1000, 2),
                "generation": round(out.generation_ms, 2),
                "execution": round(out.execution_ms, 2),
            },
        )

    @app.get("/metrics")
    async def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def __getattr__(name: str) -> Any:
    # `uvicorn t2sforge.serve.app:app` - build lazily so importing the module has no side effects
    if name == "app":
        return create_app()
    raise AttributeError(name)
