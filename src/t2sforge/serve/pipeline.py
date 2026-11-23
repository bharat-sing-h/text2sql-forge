"""Question -> validated SQL -> rows, with one round of self-correction.

Steps:
1. render the schema prompt (cached per database);
2. generate SQL with the fine-tuned model (greedy);
3. **guard**: parse and reject anything that isn't a single read-only SELECT;
4. cap the result size with a LIMIT, then execute in the read-only sandbox;
5. on a guard rejection or SQLite error, send the error back to the model once and retry.

Step 5 recovers typical near-misses (a wrong column name, a missing GROUP BY) for the price of a
second generation only on failures; the repair rate is exported as a metric because a rising
repair rate is an early warning of schema drift or a bad deploy.
"""

from __future__ import annotations

import functools
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from t2sforge.data.examples import db_path
from t2sforge.data.prompt import build_messages, repair_messages
from t2sforge.data.schema import schema_prompt
from t2sforge.serve.backend import ChatBackend
from t2sforge.sql import enforce_limit, extract_sql, run_query, validate_sql

_DB_ID = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


class UnknownDatabaseError(KeyError):
    pass


class DatabaseRegistry:
    """Maps ``db_id`` to a file under ``db_root``. Ids are validated to block path traversal."""

    def __init__(self, db_root: str | Path) -> None:
        self.root = Path(db_root).resolve()

    def path(self, db_id: str) -> Path:
        if not _DB_ID.match(db_id):
            raise UnknownDatabaseError(db_id)
        path = db_path(self.root, db_id).resolve()
        if self.root not in path.parents or not path.exists():
            raise UnknownDatabaseError(db_id)
        return path

    def list(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / f"{p.name}.sqlite").exists())

    def schema(self, db_id: str) -> str:
        return _schema_cached(str(self.path(db_id)))


@lru_cache(maxsize=256)
def _schema_cached(path: str) -> str:
    return schema_prompt(path)


@dataclass
class Attempt:
    sql: str
    error: str | None
    stage: str  # "guard" | "execute" | "ok"


@dataclass
class QueryOutcome:
    sql: str | None
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None
    attempts: list[Attempt] = field(default_factory=list)
    generation_ms: float = 0.0
    execution_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def repaired(self) -> bool:
        return self.ok and len(self.attempts) > 1


class TextToSQL:
    def __init__(
        self,
        backend: ChatBackend,
        registry: DatabaseRegistry,
        *,
        max_repairs: int = 1,
        max_rows: int = 200,
        exec_timeout_s: float = 5.0,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.max_repairs = max_repairs
        self.max_rows = max_rows
        self.exec_timeout_s = exec_timeout_s

    async def answer(
        self,
        db_id: str,
        question: str,
        *,
        evidence: str | None = None,
        execute: bool = True,
        max_rows: int | None = None,
    ) -> QueryOutcome:
        import anyio

        path = self.registry.path(db_id)
        schema = self.registry.schema(db_id)
        limit = min(max_rows or self.max_rows, self.max_rows)
        messages = build_messages(question, schema, evidence)
        outcome = QueryOutcome(sql=None)

        for attempt_no in range(self.max_repairs + 1):
            started = time.perf_counter()
            completion = await self.backend.complete(messages)
            outcome.generation_ms += (time.perf_counter() - started) * 1000
            outcome.prompt_tokens += completion.prompt_tokens
            outcome.completion_tokens += completion.completion_tokens
            sql = extract_sql(completion.text)
            outcome.sql = sql

            verdict = validate_sql(sql)
            error: str | None
            if not verdict:
                error, stage = f"rejected by SQL guard: {verdict.reason}", "guard"
            elif not execute:
                outcome.attempts.append(Attempt(sql, None, "ok"))
                outcome.error = None
                return outcome
            else:
                # fetch one extra row: with LIMIT n alone we could never tell the caller the
                # result was cut off
                limited = enforce_limit(sql, limit + 1)
                result = await anyio.to_thread.run_sync(
                    functools.partial(
                        run_query, path, limited, timeout_s=self.exec_timeout_s, max_rows=limit
                    )
                )
                outcome.execution_ms += result.elapsed_ms
                if result.ok:
                    outcome.attempts.append(Attempt(sql, None, "ok"))
                    outcome.columns = result.columns
                    outcome.rows = [list(r) for r in result.rows]
                    outcome.truncated = result.truncated
                    outcome.error = None
                    return outcome
                error, stage = result.error or "unknown error", "execute"

            outcome.attempts.append(Attempt(sql, error, stage))
            outcome.error = error
            if attempt_no < self.max_repairs:
                messages = repair_messages(question, schema, sql, error, evidence)
        return outcome
