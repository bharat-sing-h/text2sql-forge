"""llm-verdict plugin: execution-accuracy scoring for text-to-SQL.

Enable it in an llm-verdict run config::

    plugins: [t2sforge.eval.llmverdict_plugin]
    tasks:
      - path: data/eval/spider_dev.jsonl        # written by `t2s data build`
        scorers:
          - {type: execution_accuracy, options: {db_root: data/spider/database}}
          - sql_valid

This gives the fine-tuned model everything llm-verdict offers - confidence intervals, paired
significance tests against the base model, per-difficulty breakdowns, CI gating - with
execution accuracy as the metric.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from llmverdict.scorers import Scorer, register_scorer
from llmverdict.types import Sample, Score

from t2sforge.data.examples import db_path
from t2sforge.sql import execution_match, extract_sql, validate_sql

_ERROR_TYPES = [
    (re.compile(r"no such column", re.I), "unknown_column"),
    (re.compile(r"no such table", re.I), "unknown_table"),
    (re.compile(r"ambiguous column", re.I), "ambiguous_column"),
    (re.compile(r"syntax error|parse error|incomplete input", re.I), "syntax"),
    (re.compile(r"time limit", re.I), "timeout"),
    (re.compile(r"misuse of aggregate|GROUP BY", re.I), "aggregation"),
]


def classify_error(message: str) -> str:
    """Bucket SQL failures so reports can say *why* a model fails, not just how often."""
    for pattern, label in _ERROR_TYPES:
        if pattern.search(message):
            return label
    return "other_error"


def _reference(sample: Sample) -> tuple[str, str] | None:
    ref: Any = sample.reference
    if isinstance(ref, dict) and ref.get("db_id") and ref.get("gold_sql"):
        return str(ref["db_id"]), str(ref["gold_sql"])
    return None


@register_scorer
class ExecutionAccuracy(Scorer):
    """1.0 if the predicted SQL returns the same result as the gold SQL (Spider/BIRD "EX")."""

    type_name = "execution_accuracy"

    def _db_root(self) -> Path:
        return Path(str(self.options.get("db_root") or os.environ.get("T2S_DB_ROOT", "")))

    async def score(self, sample: Sample, output: str) -> Score:
        import anyio

        ref = _reference(sample)
        if ref is None:
            return self._na("reference needs db_id and gold_sql")
        db_id, gold = ref
        sql = extract_sql(output)
        guard = validate_sql(sql)
        if not guard:
            return self._bool(
                False, f"invalid SQL: {guard.reason}", error_type="invalid_sql", predicted_sql=sql
            )
        path = db_path(self._db_root(), db_id)
        timeout = float(self.options.get("timeout_s", 10.0))
        verdict = await anyio.to_thread.run_sync(
            lambda: execution_match(path, gold, sql, timeout_s=timeout)
        )
        if verdict.match is None:
            return self._na(verdict.reason)  # broken gold label: don't blame the model
        if verdict.match:
            return self._bool(True, predicted_sql=sql)
        error_type = classify_error(verdict.pred_error) if verdict.pred_error else "wrong_result"
        explanation = verdict.pred_error or verdict.reason
        return self._bool(False, explanation, error_type=error_type, predicted_sql=sql)


@register_scorer
class SQLValid(Scorer):
    """1.0 if the output contains exactly one parseable, read-only SELECT statement."""

    type_name = "sql_valid"

    async def score(self, sample: Sample, output: str) -> Score:
        guard = validate_sql(extract_sql(output))
        return self._bool(guard.ok, guard.reason)
