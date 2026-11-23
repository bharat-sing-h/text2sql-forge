from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from t2sforge.sql import (
    enforce_limit,
    execution_match,
    extract_sql,
    first_statement,
    has_top_level_order_by,
    results_match,
    run_query,
    validate_sql,
)

# --- extraction ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SELECT 1;", "SELECT 1"),
        ("```sql\nSELECT name FROM t;\n```\nThis returns names.", "SELECT name FROM t"),
        (
            "Here you go:\nSELECT a FROM b WHERE c = 'x;y'; DROP TABLE b;",
            "SELECT a FROM b WHERE c = 'x;y'",
        ),
        ("```\nWITH x AS (SELECT 1) SELECT * FROM x\n```", "WITH x AS (SELECT 1) SELECT * FROM x"),
        ("no sql here", "no sql here"),
    ],
)
def test_extract_sql(text: str, expected: str) -> None:
    assert extract_sql(text) == expected


def test_first_statement_handles_quotes_and_comments() -> None:
    assert first_statement("SELECT 'it''s; fine' -- trailing; comment\n; SELECT 2") == (
        "SELECT 'it''s; fine' -- trailing; comment"
    )
    assert first_statement('SELECT "a;b", [c;d] FROM t; x') == 'SELECT "a;b", [c;d] FROM t'


# --- guard --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM customers",
        "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
        "SELECT name FROM customers UNION SELECT status FROM orders",
        "select count(*) from orders where status = 'drop table'",
    ],
)
def test_guard_accepts_read_queries(sql: str) -> None:
    assert validate_sql(sql)


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DELETE FROM customers", "only SELECT"),
        ("DROP TABLE customers", "only SELECT"),
        ("SELECT 1; DROP TABLE customers", "expected 1 statement"),
        ("UPDATE customers SET name = 'x'", "only SELECT"),
        ("ATTACH DATABASE '/tmp/x.db' AS x", "only SELECT"),
        ("PRAGMA writable_schema = 1", "only SELECT"),
        ("SELECT load_extension('evil.so')", "forbidden function"),
        ("SELEC name FRM customers", "parse error"),
        ("   ", "empty"),
    ],
)
def test_guard_rejects_unsafe_or_invalid(sql: str, reason: str) -> None:
    result = validate_sql(sql)
    assert not result
    assert reason in (result.reason or "")


def test_enforce_limit() -> None:
    assert "LIMIT 100" in enforce_limit("SELECT * FROM t", 100)
    assert enforce_limit("SELECT * FROM t LIMIT 5", 100) == "SELECT * FROM t LIMIT 5"
    assert "LIMIT 100" in enforce_limit("SELECT * FROM t LIMIT 500", 100)
    wrapped = enforce_limit("SELECT a FROM t UNION SELECT b FROM u", 10)
    assert wrapped.startswith("SELECT * FROM (")
    assert wrapped.endswith("LIMIT 10")


def test_order_by_detection() -> None:
    assert has_top_level_order_by("SELECT a FROM t ORDER BY a DESC")
    assert not has_top_level_order_by("SELECT a FROM (SELECT a FROM t ORDER BY a)")
    assert has_top_level_order_by("not valid sql ORDER BY")  # parse fallback


# --- sandboxed execution ------------------------------------------------------------------


def test_run_query_success_and_truncation(shop_db: Path) -> None:
    res = run_query(shop_db, "SELECT name, city FROM customers ORDER BY id")
    assert res.ok
    assert res.columns == ["name", "city"]
    assert res.rows[0] == ("Asha", "Delhi")
    capped = run_query(shop_db, "SELECT * FROM customers", max_rows=2)
    assert capped.truncated
    assert len(capped.rows) == 2


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "INSERT INTO customers VALUES (9, 'x', 'y', 2020)",
        "CREATE TABLE evil (x)",
        "ATTACH DATABASE ':memory:' AS other",
        "PRAGMA journal_mode = DELETE",
    ],
)
def test_sandbox_blocks_writes_even_without_guard(shop_db: Path, sql: str) -> None:
    res = run_query(shop_db, sql)
    assert not res.ok
    # the data is untouched
    assert run_query(shop_db, "SELECT COUNT(*) FROM customers").rows == [(4,)]


def test_sandbox_timeout(shop_db: Path) -> None:
    runaway = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT MAX(x) FROM c"
    res = run_query(shop_db, runaway, timeout_s=0.2)
    assert not res.ok
    assert "time limit" in (res.error or "")


def test_run_query_errors(shop_db: Path, tmp_path: Path) -> None:
    assert "no such column" in (run_query(shop_db, "SELECT nope FROM customers").error or "")
    missing = run_query(tmp_path / "missing.sqlite", "SELECT 1")
    assert not missing.ok
    assert "cannot open" in (missing.error or "")


def test_readonly_connection_cannot_write_via_sqlite_api(shop_db: Path) -> None:
    from t2sforge.sql.execute import connect_readonly

    conn = connect_readonly(shop_db)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM customers")
    conn.close()


# --- execution accuracy -------------------------------------------------------------------


def test_results_match_rules() -> None:
    gold = [("Asha", 2), ("Ben", 1)]
    assert results_match(gold, [("Ben", 1), ("Asha", 2)], ordered=False)
    assert not results_match(gold, [("Ben", 1), ("Asha", 2)], ordered=True)
    assert results_match(gold, [(2, "Asha"), (1, "Ben")], ordered=True)  # column permutation
    assert results_match([(3,)], [(3.0,)], ordered=False)  # numeric normalisation
    assert results_match([(0.1 + 0.2,)], [(0.3,)], ordered=False)
    assert not results_match(gold, gold[:1], ordered=False)
    assert not results_match(gold, [("Asha", 2, "x"), ("Ben", 1, "y")], ordered=False)
    assert results_match([], [], ordered=False)
    # duplicates matter (multiset, not set)
    assert not results_match([(1,), (1,)], [(1,), (2,)], ordered=False)
    assert results_match([(None, 1)], [(None, 1)], ordered=False)
    assert results_match([(b"\x01", True)], [("01", 1)], ordered=False)


def test_execution_match(shop_db: Path) -> None:
    gold = "SELECT name FROM customers WHERE city = 'Delhi'"
    assert execution_match(
        shop_db, gold, "SELECT name FROM customers WHERE city LIKE 'delhi'"
    ).match
    wrong = execution_match(shop_db, gold, "SELECT name FROM customers")
    assert wrong.match is False
    assert "differ" in wrong.reason
    broken = execution_match(shop_db, gold, "SELECT nme FROM customers")
    assert broken.match is False
    assert broken.pred_error
    bad_gold = execution_match(shop_db, "SELECT oops", "SELECT 1")
    assert bad_gold.match is None
    # ORDER BY in gold makes row order significant
    ordered_gold = "SELECT name FROM customers ORDER BY joined_year DESC, name"
    assert not execution_match(
        shop_db, ordered_gold, "SELECT name FROM customers ORDER BY name"
    ).match
