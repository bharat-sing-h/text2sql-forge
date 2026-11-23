"""CLI data pipeline + llm-verdict integration (execution accuracy through the real harness)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from t2sforge.cli import app

runner = CliRunner(env={"COLUMNS": "200"})


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("cli")
    synth = root / "synthetic"
    res = runner.invoke(
        app, ["data", "synth", "--out", str(synth), "--instances", "2", "--per-db", "15"]
    )
    assert res.exit_code == 0, res.output
    res = runner.invoke(
        app,
        [
            "data",
            "build",
            "--source",
            "synthetic",
            "--root",
            str(synth),
            "--out",
            str(root / "data"),
            "--val-fraction",
            "0.25",
        ],
    )
    assert res.exit_code == 0, res.output
    return root


def test_data_build_outputs(built: Path) -> None:
    card = json.loads((built / "data" / "DATA_CARD.json").read_text())
    assert card["leakage_check"] == "passed"
    assert card["train"] > 0
    assert card["val"] > 0
    assert card["eval"] > 0
    first = json.loads((built / "data" / "sft" / "train.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in first["messages"]] == ["system", "user", "assistant"]


def test_build_with_augmentation_and_leak_detection(built: Path, tmp_path: Path) -> None:
    synth = built / "synthetic"
    res = runner.invoke(
        app,
        [
            "data",
            "build",
            "--source",
            "synthetic",
            "--root",
            str(synth),
            "--out",
            str(tmp_path / "aug"),
            "--add-synthetic",
            str(synth),
        ],
    )
    # training on the full synthetic set leaks the held-out eval databases: must be refused
    assert res.exit_code == 1
    assert "Leakage" in res.output


def test_misc_commands(built: Path) -> None:
    assert "t2sforge" in runner.invoke(app, ["--version"]).output
    assert (
        runner.invoke(
            app, ["data", "build", "--source", "spider", "--root", str(built / "missing")]
        ).exit_code
        != 0
    )


def test_execution_accuracy_through_llmverdict(
    built: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("llmverdict")
    from llmverdict.config import load_config
    from llmverdict.harness import run_config

    eval_file = built / "data" / "eval" / "synthetic_eval.jsonl"
    rows = [json.loads(line) for line in eval_file.read_text().splitlines()]
    oracle: dict[str, str] = {}
    for i, row in enumerate(rows):
        question = row["messages"][-1]["content"].split("### Question\n", 1)[1].split("\n\n")[0]
        gold = row["reference"]["gold_sql"]
        if i % 5 == 0:
            oracle[question] = "SELECT nonexistent_column FROM nowhere"  # execution error
        elif i % 7 == 0:
            oracle[question] = "DELETE FROM employees"  # unsafe -> invalid
        else:
            oracle[question] = f"```sql\n{gold};\n```"  # correct, wrapped like a chatty model
    oracle_file = tmp_path / "oracle.json"
    oracle_file.write_text(json.dumps(oracle))
    monkeypatch.setenv("ORACLE_FILE", str(oracle_file))

    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(f"""
model: {{provider: function, target: "tests.fixtures.oracle_app:answer", model: oracle}}
plugins: [t2sforge.eval.llmverdict_plugin]
tasks:
  - path: {eval_file}
    name: synthetic
    scorers:
      - {{type: execution_accuracy, options: {{db_root: {built / "synthetic" / "databases"}}}}}
      - sql_valid
store_path: {tmp_path / "r.db"}
cache_path: {tmp_path / "c.db"}
execution: {{concurrency: 4, cache: false}}
""")
    [done] = asyncio.run(run_config(load_config(cfg_path), label="oracle"))
    n = len(rows)
    # the same question text can appear in several databases, so derive the expected outcome
    # per row from the final lookup table rather than from the row index
    answers = [
        oracle[row["messages"][-1]["content"].split("### Question\n", 1)[1].split("\n\n")[0]]
        for row in rows
    ]
    n_unsafe = sum(1 for a in answers if a.startswith("DELETE"))
    n_bad_exec = sum(1 for a in answers if "nonexistent_column" in a)
    ex = done.summary.metrics["execution_accuracy"]
    assert ex.n == n
    assert ex.mean == pytest.approx((n - n_bad_exec - n_unsafe) / n)
    assert done.summary.metrics["sql_valid"].mean == pytest.approx((n - n_unsafe) / n)
    assert any(tag.startswith("difficulty:") for tag in done.summary.by_tag)

    from llmverdict.store import ResultStore

    store = ResultStore(tmp_path / "r.db")
    types = {
        r.score("execution_accuracy").metadata.get("error_type")  # type: ignore[union-attr]
        for r in store.get_results(done.run_id)
    }
    store.close()
    assert {"unknown_table", "invalid_sql", None} <= types


def test_error_classification() -> None:
    pytest.importorskip("llmverdict")
    from t2sforge.eval.llmverdict_plugin import classify_error

    assert classify_error("no such column: foo") == "unknown_column"
    assert classify_error("near 'FROM': syntax error") == "syntax"
    assert classify_error("query exceeded 5s time limit") == "timeout"
    assert classify_error("something else") == "other_error"
