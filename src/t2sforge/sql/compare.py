"""Execution accuracy: does the predicted query return the same answer as the gold query?

String/AST match under-counts correct SQL (there are many ways to write the same query) so the
field standard (Spider, BIRD) is to *execute* both and compare results. Rules implemented here,
following the Spider test-suite evaluation:

* rows are compared as a multiset, or as a list when the gold query has a top-level ORDER BY;
* column order doesn't matter (``SELECT name, age`` == ``SELECT age, name``), so column
  permutations are searched, pruned by per-column value multisets;
* numbers are normalised (``3 == 3.0``, floats rounded) so formatting doesn't flip a verdict.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from t2sforge.sql.execute import run_query
from t2sforge.sql.guard import has_top_level_order_by

Row = tuple[Any, ...]


def _norm(value: Any, places: int) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        rounded = round(value, places)
        return int(rounded) if rounded.is_integer() else rounded
    if isinstance(value, bytes):
        return value.hex()
    return value


def normalize_rows(rows: Sequence[Row], places: int = 6) -> list[Row]:
    return [tuple(_norm(v, places) for v in row) for row in rows]


def _key(value: Any) -> tuple[str, str]:
    # rows can mix types (NULL, int, str); sort by (type, repr) to get a total order
    return (type(value).__name__, repr(value))


def _column_mappings(gold: list[Row], pred: list[Row], ncols: int) -> Iterator[list[int]]:
    """Yield pred-column orders whose per-column value multisets match gold's."""
    gold_cols = [Counter(r[i] for r in gold) for i in range(ncols)]
    pred_cols = [Counter(r[j] for r in pred) for j in range(ncols)]
    candidates = [[j for j in range(ncols) if pred_cols[j] == gold_cols[i]] for i in range(ncols)]

    def backtrack(i: int, used: set[int], acc: list[int]) -> Iterator[list[int]]:
        if i == ncols:
            yield list(acc)
            return
        for j in candidates[i]:
            if j not in used:
                used.add(j)
                acc.append(j)
                yield from backtrack(i + 1, used, acc)
                acc.pop()
                used.remove(j)

    yield from backtrack(0, set(), [])


def results_match(
    gold: Sequence[Row], pred: Sequence[Row], *, ordered: bool, places: int = 6
) -> bool:
    g, p = normalize_rows(gold, places), normalize_rows(pred, places)
    if len(g) != len(p):
        return False
    if not g:
        return True
    ncols = len(g[0])
    if any(len(r) != ncols for r in p):
        return False
    for mapping in _column_mappings(g, p, ncols):
        permuted = [tuple(row[j] for j in mapping) for row in p]
        if ordered:
            if permuted == g:
                return True
        elif sorted(permuted, key=lambda r: [_key(v) for v in r]) == sorted(
            g, key=lambda r: [_key(v) for v in r]
        ):
            return True
    return False


@dataclass(frozen=True)
class ExecutionVerdict:
    match: bool | None  # None = gold query itself failed (bad label, not the model's fault)
    reason: str
    pred_error: str | None = None


def execution_match(
    db_path: str | Path, gold_sql: str, pred_sql: str, *, timeout_s: float = 10.0
) -> ExecutionVerdict:
    gold = run_query(db_path, gold_sql, timeout_s=timeout_s, max_rows=100_000)
    if not gold.ok:
        return ExecutionVerdict(None, f"gold query failed: {gold.error}")
    pred = run_query(db_path, pred_sql, timeout_s=timeout_s, max_rows=100_000)
    if not pred.ok:
        return ExecutionVerdict(False, "predicted query failed", pred_error=pred.error)
    ordered = has_top_level_order_by(gold_sql)
    if results_match(gold.rows, pred.rows, ordered=ordered):
        return ExecutionVerdict(True, "results match")
    return ExecutionVerdict(
        False, f"results differ (gold {len(gold.rows)} rows, predicted {len(pred.rows)} rows)"
    )
