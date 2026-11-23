from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Example(BaseModel):
    """One text-to-SQL pair. Databases follow the Spider/BIRD layout
    ``<db_root>/<db_id>/<db_id>.sqlite`` so every source resolves paths the same way."""

    id: str
    db_id: str
    question: str
    sql: str
    evidence: str | None = None  # BIRD "external knowledge" hint
    difficulty: str | None = None
    source: str = "custom"

    def db_path(self, db_root: str | Path) -> Path:
        return db_path(db_root, self.db_id)


def db_path(db_root: str | Path, db_id: str) -> Path:
    return Path(db_root) / db_id / f"{db_id}.sqlite"
