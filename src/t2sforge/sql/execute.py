"""Sandboxed SQLite execution.

Layer 2 of defence in depth (layer 1 is :mod:`t2sforge.sql.guard`):

* the database is opened **read-only** through a ``mode=ro`` URI;
* an **authorizer** callback whitelists read operations only, so even SQL that slipped past the
  parser cannot write, attach other files or load extensions;
* a **progress handler** aborts queries that exceed a wall-clock budget (model-written SQL loves
  accidental cartesian products);
* results are capped at ``max_rows`` so one query can't exhaust memory.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
_FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit"}


@dataclass
class QueryResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False
    elapsed_ms: float = 0.0


def _authorizer(
    action: int, arg1: str | None, arg2: str | None, db: str | None, trigger: str | None
) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _FORBIDDEN_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def run_query(
    db_path: str | Path, sql: str, *, timeout_s: float = 5.0, max_rows: int = 10_000
) -> QueryResult:
    started = time.perf_counter()
    deadline = started + timeout_s
    try:
        conn = connect_readonly(db_path)
    except (FileNotFoundError, sqlite3.Error) as exc:
        return QueryResult(ok=False, error=f"cannot open database: {exc}")
    try:
        conn.set_authorizer(_authorizer)
        # called every N VM instructions; a non-zero return aborts the statement
        conn.set_progress_handler(lambda: int(time.perf_counter() > deadline), 10_000)
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description or []]
        truncated = len(rows) > max_rows
        return QueryResult(
            ok=True,
            columns=columns,
            rows=[tuple(r) for r in rows[:max_rows]],
            truncated=truncated,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
    except sqlite3.Error as exc:
        message = str(exc)
        if "interrupted" in message:
            message = f"query exceeded {timeout_s}s time limit"
        elif "not authorized" in message:
            message = "operation not permitted (read-only sandbox)"
        return QueryResult(
            ok=False, error=message, elapsed_ms=(time.perf_counter() - started) * 1000
        )
    finally:
        conn.close()
