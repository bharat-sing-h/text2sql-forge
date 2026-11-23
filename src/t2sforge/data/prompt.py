"""The single prompt format shared by training, evaluation and serving."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert SQLite analyst. Given a database schema and a question, write a single "
    "SQLite SELECT query that answers the question. Use only tables and columns from the "
    "schema. Return only the SQL query, with no explanation."
)


def user_prompt(question: str, schema: str, evidence: str | None = None) -> str:
    parts = [f"### Database schema\n{schema}", f"### Question\n{question.strip()}"]
    if evidence:
        parts.append(f"### Hint\n{evidence.strip()}")
    return "\n\n".join(parts)


def build_messages(question: str, schema: str, evidence: str | None = None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(question, schema, evidence)},
    ]


def repair_messages(
    question: str, schema: str, bad_sql: str, error: str, evidence: str | None = None
) -> list[dict[str, str]]:
    """Follow-up turn for self-correction: show the model its failed query and the DB error."""
    return [
        *build_messages(question, schema, evidence),
        {"role": "assistant", "content": bad_sql},
        {
            "role": "user",
            "content": (
                f"That query failed with this SQLite error:\n{error}\n"
                "Write a corrected query. Return only the SQL."
            ),
        },
    ]


def normalize_target(sql: str) -> str:
    """Training target: single line-normalised statement without a trailing semicolon."""
    return " ".join(sql.strip().rstrip(";").split())
