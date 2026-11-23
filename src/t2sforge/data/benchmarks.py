"""Loaders for the public text-to-SQL benchmarks (download them yourself; see README).

* **Spider 1.0** (Yu et al., 2018; CC BY-SA 4.0): ``train_spider.json``, ``train_others.json``,
  ``dev.json`` and ``database/<db_id>/<db_id>.sqlite``. Train and dev use *disjoint databases*,
  so dev measures generalisation to unseen schemas.
* **BIRD** (Li et al., 2023; CC BY-SA 4.0): ``train/train.json`` + ``train/train_databases`` and
  ``dev/dev.json`` + ``dev/dev_databases``. Larger, messier real-world databases, with an
  ``evidence`` hint per question.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from t2sforge.data.examples import Example

SpiderSplit = Literal["train", "dev", "test"]
BirdSplit = Literal["train", "dev"]


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    return data


def spider_db_root(root: str | Path, split: SpiderSplit) -> Path:
    root = Path(root)
    return root / ("test_database" if split == "test" else "database")


def load_spider(root: str | Path, split: SpiderSplit) -> list[Example]:
    root = Path(root)
    files = {
        "train": ["train_spider.json", "train_others.json"],
        "dev": ["dev.json"],
        "test": ["test.json"],
    }[split]
    examples = []
    for name in files:
        path = root / name
        if not path.exists():
            if name == "train_others.json":
                continue  # optional extra training file
            raise FileNotFoundError(f"{path} not found - see README for download instructions")
        for i, row in enumerate(_read_json(path)):
            examples.append(
                Example(
                    id=f"spider-{split}-{Path(name).stem}-{i:05d}",
                    db_id=row["db_id"],
                    question=row["question"],
                    sql=row["query"],
                    source="spider",
                )
            )
    return examples


def bird_db_root(root: str | Path, split: BirdSplit) -> Path:
    return Path(root) / split / f"{split}_databases"


def load_bird(root: str | Path, split: BirdSplit) -> list[Example]:
    path = Path(root) / split / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - see README for download instructions")
    return [
        Example(
            id=f"bird-{split}-{row.get('question_id', i):05d}",
            db_id=row["db_id"],
            question=row["question"],
            sql=row["SQL"],
            evidence=row.get("evidence") or None,
            difficulty=row.get("difficulty"),
            source="bird",
        )
        for i, row in enumerate(_read_json(path))
    ]
