from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from t2sforge.data.benchmarks import bird_db_root, load_bird, load_spider, spider_db_root
from t2sforge.data.build import (
    check_leakage,
    eval_record,
    read_examples,
    split_by_database,
    write_eval_jsonl,
    write_examples,
    write_sft_jsonl,
)
from t2sforge.data.examples import Example
from t2sforge.data.prompt import (
    SYSTEM_PROMPT,
    build_messages,
    normalize_target,
    repair_messages,
)
from t2sforge.data.schema import describe_database, render_schema, schema_prompt
from t2sforge.data.synthetic import DOMAINS, generate
from t2sforge.sql import execution_match, run_query


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[Example]]:
    root = tmp_path_factory.mktemp("synth")
    return root, generate(root, instances_per_domain=2, per_db=30, seed=1)


def test_schema_rendering(shop_db: Path) -> None:
    tables = describe_database(shop_db)
    assert [t.name for t in tables] == ["customers", "orders"]
    orders = next(t for t in tables if t.name == "orders")
    fk = next(c for c in orders.columns if c.name == "customer_id")
    assert fk.references == ("customers", "id")
    text = render_schema(tables)
    assert "CREATE TABLE customers (" in text
    assert "id INTEGER PRIMARY KEY," in text
    assert "customer_id INTEGER REFERENCES customers(id)" in text
    assert "-- e.g. 'paid', 'refunded', 'pending'" in text
    assert "e.g." not in render_schema(tables, with_examples=False)
    # huge schemas drop example values but keep structure
    small = schema_prompt(str(shop_db), max_chars=50)
    assert "e.g." not in small
    assert "CREATE TABLE orders" in small


def test_schema_quotes_odd_identifiers_and_long_values(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "odd.sqlite"
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE "order items" ("unit price" REAL, note TEXT)')
    conn.execute('INSERT INTO "order items" VALUES (1.5, ?)', ("x" * 100,))
    conn.commit()
    conn.close()
    text = render_schema(describe_database(path))
    assert 'CREATE TABLE "order items"' in text
    assert '"unit price" REAL' in text
    assert "…'" in text


def test_prompt_format() -> None:
    msgs = build_messages(" How many? ", "CREATE TABLE t (x INT);", evidence="x is a count")
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1]["content"].endswith("### Hint\nx is a count")
    assert "### Question\nHow many?" in msgs[1]["content"]
    repair = repair_messages("q", "schema", "SELECT nope", "no such column: nope")
    assert [m["role"] for m in repair] == ["system", "user", "assistant", "user"]
    assert "no such column" in repair[-1]["content"]
    assert normalize_target("SELECT a\n  FROM   t;\n") == "SELECT a FROM t"


def test_synthetic_examples_are_verified_and_diverse(synth: tuple[Path, list[Example]]) -> None:
    root, examples = synth
    assert len(examples) == len(DOMAINS) * 2 * 30
    assert len({e.id for e in examples}) == len(examples)
    assert set(Counter(e.difficulty for e in examples)) == {"easy", "medium", "hard"}
    for e in examples:
        result = run_query(e.db_path(root), e.sql)
        assert result.ok, e.sql
        assert result.rows, e.sql
        assert "HAVING COUNT(*) > 0" not in e.sql
        assert execution_match(e.db_path(root), e.sql, e.sql).match


def test_synthetic_is_deterministic(tmp_path: Path) -> None:
    a = generate(tmp_path / "a", instances_per_domain=1, per_db=10, seed=3)
    b = generate(tmp_path / "b", instances_per_domain=1, per_db=10, seed=3)
    assert [(e.question, e.sql) for e in a] == [(e.question, e.sql) for e in b]


def test_split_and_leakage(synth: tuple[Path, list[Example]]) -> None:
    _, examples = synth
    train, val = split_by_database(examples, val_fraction=0.34, seed=0)
    assert train
    assert val
    assert check_leakage(train, val).clean
    leaky = check_leakage(train, train[:3])
    assert not leaky.clean
    assert leaky.shared_databases
    assert len(leaky.duplicate_questions) == 3
    single_db = [e for e in examples if e.db_id == examples[0].db_id]
    assert split_by_database(single_db)[1] == []


def test_writers_roundtrip(synth: tuple[Path, list[Example]], tmp_path: Path) -> None:
    root, examples = synth
    n = write_sft_jsonl(examples[:5], root, tmp_path / "sft" / "train.jsonl")
    assert n == 5
    record = json.loads((tmp_path / "sft" / "train.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]
    assert record["messages"][2]["content"] == normalize_target(examples[0].sql)

    write_eval_jsonl(examples[:4], root, tmp_path / "eval.jsonl")
    rows = [json.loads(x) for x in (tmp_path / "eval.jsonl").read_text().splitlines()]
    assert rows[0]["reference"] == {"db_id": examples[0].db_id, "gold_sql": examples[0].sql}
    assert "source:synthetic" in rows[0]["tags"]
    assert eval_record(examples[0], root)["id"] == examples[0].id

    write_examples(examples[:7], tmp_path / "ex.jsonl")
    assert read_examples(tmp_path / "ex.jsonl") == examples[:7]


def test_spider_and_bird_loaders(tmp_path: Path) -> None:
    spider = tmp_path / "spider"
    spider.mkdir()
    (spider / "train_spider.json").write_text(
        json.dumps(
            [
                {
                    "db_id": "concert",
                    "question": "How many singers?",
                    "query": "SELECT count(*) FROM singer",
                }
            ]
        )
    )
    (spider / "dev.json").write_text(
        json.dumps(
            [{"db_id": "pets", "question": "Count pets", "query": "SELECT count(*) FROM pets"}]
        )
    )
    train = load_spider(spider, "train")  # train_others.json is optional
    assert train[0].sql == "SELECT count(*) FROM singer"
    assert train[0].source == "spider"
    assert load_spider(spider, "dev")[0].db_id == "pets"
    assert spider_db_root(spider, "dev") == spider / "database"
    assert spider_db_root(spider, "test") == spider / "test_database"
    with pytest.raises(FileNotFoundError, match="README"):
        load_spider(spider, "test")

    bird = tmp_path / "bird" / "dev"
    bird.mkdir(parents=True)
    (bird / "dev.json").write_text(
        json.dumps(
            [
                {
                    "question_id": 7,
                    "db_id": "finance",
                    "question": "Q",
                    "SQL": "SELECT 1",
                    "evidence": "hint",
                    "difficulty": "moderate",
                }
            ]
        )
    )
    ex = load_bird(tmp_path / "bird", "dev")[0]
    assert (ex.id, ex.evidence, ex.difficulty) == ("bird-dev-00007", "hint", "moderate")
    assert bird_db_root(tmp_path / "bird", "dev") == bird / "dev_databases"
    with pytest.raises(FileNotFoundError):
        load_bird(tmp_path / "bird", "train")
    (bird / "dev.json").write_text("{}")
    with pytest.raises(ValueError, match="JSON list"):
        load_bird(tmp_path / "bird", "dev")
