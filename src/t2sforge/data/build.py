"""Turn examples into training / evaluation files, with leakage checks."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from t2sforge.data.examples import Example
from t2sforge.data.prompt import build_messages, normalize_target
from t2sforge.data.schema import schema_prompt


def split_by_database(
    examples: Sequence[Example], *, val_fraction: float = 0.15, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Hold out whole databases, not random rows.

    A random row split puts questions about the *same* schema on both sides, so validation
    accuracy over-states how the model will do on a new customer's database.
    """
    db_ids = sorted({e.db_id for e in examples})
    rng = random.Random(seed)
    rng.shuffle(db_ids)
    n_val = max(1, round(len(db_ids) * val_fraction)) if len(db_ids) > 1 else 0
    val_dbs = set(db_ids[:n_val])
    train = [e for e in examples if e.db_id not in val_dbs]
    val = [e for e in examples if e.db_id in val_dbs]
    return train, val


@dataclass
class LeakageReport:
    shared_databases: list[str]
    duplicate_questions: list[str]

    @property
    def clean(self) -> bool:
        return not self.shared_databases and not self.duplicate_questions


def check_leakage(train: Iterable[Example], evaluation: Iterable[Example]) -> LeakageReport:
    train = list(train)
    evaluation = list(evaluation)
    shared = sorted({e.db_id for e in train} & {e.db_id for e in evaluation})
    norm = {(" ".join(e.question.lower().split()), e.db_id) for e in train}
    dupes = sorted(
        e.id for e in evaluation if (" ".join(e.question.lower().split()), e.db_id) in norm
    )
    return LeakageReport(shared, dupes)


def _messages(example: Example, db_root: Path) -> list[dict[str, str]]:
    schema = schema_prompt(str(example.db_path(db_root)))
    return build_messages(example.question, schema, example.evidence)


def write_sft_jsonl(examples: Sequence[Example], db_root: str | Path, out: str | Path) -> int:
    """Chat-format SFT records: ``{"messages": [system, user, assistant]}``."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(db_root)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for e in examples:
            record = {
                "id": e.id,
                "messages": [
                    *_messages(e, root),
                    {"role": "assistant", "content": normalize_target(e.sql)},
                ],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def eval_record(example: Example, db_root: str | Path) -> dict[str, Any]:
    """One llm-verdict sample: prompt messages plus everything the execution scorer needs."""
    tags = [f"source:{example.source}", f"db:{example.db_id}"]
    if example.difficulty:
        tags.append(f"difficulty:{example.difficulty}")
    return {
        "id": example.id,
        "messages": _messages(example, Path(db_root)),
        "reference": {"db_id": example.db_id, "gold_sql": example.sql},
        "tags": tags,
    }


def write_eval_jsonl(examples: Sequence[Example], db_root: str | Path, out: str | Path) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(eval_record(e, db_root), ensure_ascii=False) + "\n")
    return len(examples)


def write_examples(examples: Sequence[Example], out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(e.model_dump_json() + "\n")


def read_examples(path: str | Path) -> list[Example]:
    with Path(path).open(encoding="utf-8") as fh:
        return [Example.model_validate_json(line) for line in fh if line.strip()]
