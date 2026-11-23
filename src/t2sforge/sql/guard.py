"""Static safety checks for model-generated SQL.

Layer 1 of a defence-in-depth design (layer 2 is the read-only, authorizer-restricted connection
in :mod:`t2sforge.sql.execute`). The guard parses the SQL with a real parser rather than regexes:
keyword blocklists are trivially bypassed with comments, casing or string tricks.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

READ_ROOTS = (exp.Select, exp.Union, exp.Except, exp.Intersect)

FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,  # anything sqlglot can't model, e.g. ATTACH / PRAGMA / VACUUM
    exp.Merge,
    exp.Pragma,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    sql: str
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.ok


def validate_sql(sql: str, *, dialect: str = "sqlite") -> GuardResult:
    """Accept exactly one read-only query; reject everything else with a reason."""
    if not sql.strip():
        return GuardResult(False, sql, "empty query")
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except ParseError as exc:
        return GuardResult(False, sql, f"parse error: {str(exc).splitlines()[0]}")
    if len(statements) != 1:
        return GuardResult(False, sql, f"expected 1 statement, got {len(statements)}")
    tree = statements[0]
    if not isinstance(tree, READ_ROOTS):
        return GuardResult(False, sql, f"only SELECT queries are allowed (got {tree.key})")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return GuardResult(False, sql, f"forbidden operation: {node.key}")
        if isinstance(node, exp.Anonymous) and str(node.this).lower() in FORBIDDEN_FUNCTIONS:
            return GuardResult(False, sql, f"forbidden function: {node.this}")
    return GuardResult(True, sql)


def enforce_limit(sql: str, max_rows: int, *, dialect: str = "sqlite") -> str:
    """Add ``LIMIT max_rows`` to a query that has none (keeps a smaller existing limit)."""
    tree = sqlglot.parse_one(sql, read=dialect)
    limit = tree.args.get("limit")
    if limit is not None:
        value = limit.expression if isinstance(limit, exp.Limit) else None
        if isinstance(value, exp.Literal) and value.is_int and int(value.this) <= max_rows:
            return sql
    if isinstance(tree, exp.Select):
        return tree.limit(max_rows).sql(dialect=dialect)
    # set operations: wrap so the limit applies to the combined result
    return f"SELECT * FROM ({sql}) AS _q LIMIT {max_rows}"


def has_top_level_order_by(sql: str, *, dialect: str = "sqlite") -> bool:
    """Whether row order is part of the query's meaning (decides ordered vs set comparison)."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except ParseError:
        return "order by" in sql.lower()
    return tree.args.get("order") is not None
