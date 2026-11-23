"""Turn a SQLite database into the schema text the model sees.

Format: ``CREATE TABLE`` statements (types, primary/foreign keys) with a few distinct example
values per column as comments. Chang & Fosler-Lussier (2023, "How to Prompt LLMs for
Text-to-SQL") found CREATE TABLE plus sample content to be among the strongest schema
representations; example values matter most for filters on categorical columns
("status = 'paid'" vs "'Paid'").

The *same* function renders schemas for training and inference: train/serve skew in the prompt
format is one of the easiest ways to silently lose accuracy after fine-tuning.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from t2sforge.sql.execute import connect_readonly


@dataclass
class Column:
    name: str
    type: str
    primary_key: bool = False
    references: tuple[str, str] | None = None
    examples: list[str] = field(default_factory=list)


@dataclass
class Table:
    name: str
    columns: list[Column]


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _ident(identifier: str) -> str:
    """Quote only when needed, so the common case stays readable for the model."""
    return identifier if identifier.isidentifier() else _quote(identifier)


def _fmt_value(value: object, max_len: int = 40) -> str:
    if isinstance(value, str):
        text = value if len(value) <= max_len else value[: max_len - 1] + "…"
        return "'" + text.replace("'", "''") + "'"
    return repr(value)


def describe_database(path: str | Path, *, sample_values: int = 3) -> list[Table]:
    conn = connect_readonly(path)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        tables = []
        for table in names:
            fks = {
                row[3]: (row[2], row[4])
                for row in conn.execute(f"PRAGMA foreign_key_list({_quote(table)})")
            }
            columns = []
            for _cid, name, col_type, _notnull, _default, pk in conn.execute(
                f"PRAGMA table_info({_quote(table)})"
            ):
                examples: list[str] = []
                if sample_values:
                    try:
                        examples = [
                            _fmt_value(r[0])
                            for r in conn.execute(
                                f"SELECT DISTINCT {_quote(name)} FROM {_quote(table)} "
                                f"WHERE {_quote(name)} IS NOT NULL LIMIT {sample_values}"
                            )
                        ]
                    except sqlite3.Error:
                        examples = []
                columns.append(
                    Column(name, col_type or "", bool(pk), fks.get(name), examples)
                )
            tables.append(Table(table, columns))
        return tables
    finally:
        conn.close()


def render_schema(tables: list[Table], *, with_examples: bool = True) -> str:
    blocks = []
    for table in tables:
        lines = []
        for col in table.columns:
            line = f"  {_ident(col.name)} {col.type}".rstrip()
            if col.primary_key:
                line += " PRIMARY KEY"
            if col.references:
                ref_table, ref_col = col.references
                line += f" REFERENCES {_ident(ref_table)}({_ident(ref_col)})"
            lines.append((line, col.examples if with_examples else []))
        body = []
        for i, (line, examples) in enumerate(lines):
            sep = "," if i < len(lines) - 1 else ""
            comment = f"  -- e.g. {', '.join(examples)}" if examples else ""
            body.append(f"{line}{sep}{comment}")
        blocks.append(f"CREATE TABLE {_ident(table.name)} (\n" + "\n".join(body) + "\n);")
    return "\n\n".join(blocks)


@lru_cache(maxsize=512)
def schema_prompt(path: str, sample_values: int = 3, max_chars: int = 12_000) -> str:
    """Cached schema text for a database. Falls back to dropping example values when the
    schema would exceed ``max_chars`` (large BIRD databases), keeping structure intact."""
    tables = describe_database(path, sample_values=sample_values)
    text = render_schema(tables)
    if len(text) > max_chars:
        text = render_schema(tables, with_examples=False)
    return text
