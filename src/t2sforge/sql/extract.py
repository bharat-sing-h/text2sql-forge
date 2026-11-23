"""Pull a single SQL statement out of free-form model output."""

from __future__ import annotations

import re

_FENCE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.S | re.I)
_START = re.compile(r"\b(SELECT|WITH)\b", re.I)


def first_statement(sql: str) -> str:
    """Text up to the first top-level ``;`` (semicolons inside quotes/identifiers don't count)."""
    quote: str | None = None
    i = 0
    while i < len(sql):
        ch = sql[i]
        if quote:
            if ch == quote:
                # doubled quote ('' or "") is an escape, not a terminator
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    i += 1
                else:
                    quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch == "[":
            quote = "]"
        elif ch == ";":
            return sql[:i].strip()
        elif ch == "-" and sql.startswith("--", i):
            end = sql.find("\n", i)
            i = len(sql) if end == -1 else end
            continue
        i += 1
    return sql.strip()


def extract_sql(text: str) -> str:
    """Best-effort SQL extraction.

    Order: first fenced code block -> text starting at the first SELECT/WITH keyword -> the raw
    text. Only the first statement is kept, so trailing chatter or a second statement never
    reaches the database.
    """
    match = _FENCE.search(text)
    if match:
        candidate = match.group(1)
    else:
        start = _START.search(text)
        candidate = text[start.start() :] if start else text
    return first_statement(candidate)
