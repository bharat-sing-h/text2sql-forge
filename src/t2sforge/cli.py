"""``t2s`` command-line interface.

t2s data synth            generate synthetic databases + verified question/SQL pairs
t2s data build            build SFT train/val and evaluation files (Spider, BIRD, synthetic)
t2s tiny-model            build a tiny random model for offline smoke tests
t2s train CONFIG          QLoRA fine-tuning
t2s merge ADAPTER OUT     merge LoRA weights for serving
t2s ask DB QUESTION       one-off question against a model server
t2s serve                 run the FastAPI gateway
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from t2sforge import __version__
from t2sforge.data.build import (
    check_leakage,
    read_examples,
    split_by_database,
    write_eval_jsonl,
    write_examples,
    write_sft_jsonl,
)
from t2sforge.data.examples import Example

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Fine-tune, serve and evaluate small LLMs for text-to-SQL.",
)
data_app = typer.Typer(no_args_is_help=True, help="Dataset preparation.")
app.add_typer(data_app, name="data")
console = Console()


@app.callback(invoke_without_command=True)
def _main(version: Annotated[bool, typer.Option("--version")] = False) -> None:
    if version:
        console.print(f"t2sforge {__version__}")
        raise typer.Exit()


@data_app.command("synth")
def data_synth(
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("data/synthetic"),
    instances: Annotated[int, typer.Option(help="Database instances per domain.")] = 4,
    per_db: Annotated[int, typer.Option(help="Question/SQL pairs per database.")] = 40,
    seed: int = 0,
) -> None:
    """Generate synthetic databases and execution-verified question/SQL pairs."""
    from t2sforge.data.synthetic import generate

    examples = generate(out / "databases", instances_per_domain=instances, per_db=per_db, seed=seed)
    write_examples(examples, out / "examples.jsonl")
    console.print(
        f"[green]{len(examples)}[/green] verified examples over "
        f"{len({e.db_id for e in examples})} databases -> {out}"
    )


Group = tuple[list[Example], Path]


def _sources(source: str, root: Path) -> tuple[Group, Group]:
    """(train examples, db_root), (eval examples, db_root) for a benchmark."""
    from t2sforge.data.benchmarks import bird_db_root, load_bird, load_spider, spider_db_root

    if source == "spider":
        return (
            (load_spider(root, "train"), spider_db_root(root, "train")),
            (load_spider(root, "dev"), spider_db_root(root, "dev")),
        )
    if source == "bird":
        return (
            (load_bird(root, "train"), bird_db_root(root, "train")),
            (load_bird(root, "dev"), bird_db_root(root, "dev")),
        )
    # synthetic: hold out whole databases for evaluation
    examples = read_examples(root / "examples.jsonl")
    train, held_out = split_by_database(examples, val_fraction=0.25, seed=1)
    return (train, root / "databases"), (held_out, root / "databases")


@data_app.command("build")
def data_build(
    source: Annotated[Literal["spider", "bird", "synthetic"], typer.Option()] = "synthetic",
    root: Annotated[Path, typer.Option(help="Benchmark root directory.")] = Path("data/synthetic"),
    out: Annotated[Path, typer.Option(help="Where to write sft/ and eval/.")] = Path("data"),
    add_synthetic: Annotated[
        Path | None, typer.Option(help="Also train on a synthetic set (augmentation).")
    ] = None,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> None:
    """Write SFT train/val (database-disjoint) and an llm-verdict evaluation file."""
    (train, train_root), (evaluation, eval_root) = _sources(source, root)
    groups: list[Group] = [(train, train_root)]
    if add_synthetic:
        synth = read_examples(add_synthetic / "examples.jsonl")
        groups.append((synth, add_synthetic / "databases"))

    sft_train: list[tuple[Example, Path]] = []
    sft_val: list[tuple[Example, Path]] = []
    for examples, db_root in groups:
        tr, va = split_by_database(examples, val_fraction=val_fraction, seed=seed)
        sft_train += [(e, db_root) for e in tr]
        sft_val += [(e, db_root) for e in va]

    leak = check_leakage([e for e, _ in sft_train], evaluation)
    if not leak.clean:
        console.print(
            f"[red]Leakage between train and eval:[/red] "
            f"{len(leak.shared_databases)} shared DBs, "
            f"{len(leak.duplicate_questions)} duplicate questions"
        )
        raise typer.Exit(1)

    def write_grouped(pairs: list[tuple[Example, Path]], path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        n = 0
        for db_root in sorted({r for _, r in pairs}):
            part = path.with_suffix(".part")
            n += write_sft_jsonl([e for e, r in pairs if r == db_root], db_root, part)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(part.read_text(encoding="utf-8"))
            part.unlink()
        return n

    n_train = write_grouped(sft_train, out / "sft" / "train.jsonl")
    n_val = write_grouped(sft_val, out / "sft" / "val.jsonl")
    eval_path = out / "eval" / f"{source}_eval.jsonl"
    n_eval = write_eval_jsonl(evaluation, eval_root, eval_path)
    card = {
        "source": source,
        "augmented_with_synthetic": bool(add_synthetic),
        "train": n_train,
        "val": n_val,
        "eval": n_eval,
        "train_databases": len({e.db_id for e, _ in sft_train}),
        "eval_databases": len({e.db_id for e in evaluation}),
        "eval_db_root": str(eval_root),
        "leakage_check": "passed",
    }
    (out / "DATA_CARD.json").write_text(json.dumps(card, indent=2))
    table = Table(title="Dataset")
    for k in ("split", "examples", "file"):
        table.add_column(k)
    table.add_row("train", str(n_train), str(out / "sft" / "train.jsonl"))
    table.add_row("val", str(n_val), str(out / "sft" / "val.jsonl"))
    table.add_row("eval", str(n_eval), str(eval_path))
    console.print(table)
    console.print(f"Eval databases: {eval_root} (pass as db_root / T2S_DB_ROOT)")


@app.command("tiny-model")
def tiny_model(
    out: Annotated[Path, typer.Option()] = Path(".t2s/tiny-model"),
    corpus: Annotated[Path, typer.Option(help="SFT jsonl to train the tokenizer on.")] = Path(
        "data/sft/train.jsonl"
    ),
) -> None:
    """Build a tiny random chat model offline (for smoke tests, not for real use)."""
    from t2sforge.train.tiny import build_tiny_model
    from t2sforge.train.tokenize import read_chat_jsonl

    texts = [m["content"] for conv in read_chat_jsonl(corpus) for m in conv]
    console.print(f"tiny model -> {build_tiny_model(out, texts)}")


@app.command()
def train(config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """QLoRA supervised fine-tuning from a YAML config."""
    from t2sforge.train.config import load_train_config
    from t2sforge.train.sft import train as run_train

    info = run_train(load_train_config(config))
    console.print(
        {
            k: info[k]
            for k in (
                "train_examples",
                "val_examples",
                "train_loss",
                "eval_loss",
                "trainable_params",
                "train_runtime_s",
            )
        }
    )


@app.command()
def merge(
    adapter: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Path,
    dtype: Annotated[Literal["float16", "bfloat16", "float32"], typer.Option()] = "float16",
) -> None:
    """Merge a LoRA adapter into its base model (for vLLM / GGUF export)."""
    from t2sforge.train.export import merge_adapter

    console.print(f"merged model -> {merge_adapter(adapter, out, dtype=dtype)}")


@app.command()
def ask(
    db: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="SQLite file.")],
    question: str,
    base_url: Annotated[str, typer.Option(envvar="T2S_BACKEND_URL")] = "http://localhost:8000/v1",
    model: Annotated[str, typer.Option(envvar="T2S_MODEL")] = "sql",
) -> None:
    """Ask one question against a database using a running model server."""
    from t2sforge.serve.backend import ChatBackend
    from t2sforge.serve.pipeline import DatabaseRegistry, TextToSQL

    async def go() -> None:
        backend = ChatBackend(base_url, model)
        try:
            engine = TextToSQL(backend, DatabaseRegistry(db.resolve().parent.parent))
            out = await engine.answer(db.stem, question)
        finally:
            await backend.aclose()
        console.print(f"[bold]SQL:[/bold] {out.sql}")
        if out.error:
            console.print(f"[red]{out.error}[/red]")
            raise typer.Exit(1)
        table = Table(*out.columns)
        for row in out.rows:
            table.add_row(*(str(v) for v in row))
        console.print(table)

    asyncio.run(go())


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the API gateway (configure with T2S_* environment variables)."""
    import uvicorn

    uvicorn.run("t2sforge.serve.app:app", host=host, port=port)
