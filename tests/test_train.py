"""Training pipeline tests on a tiny random model (CPU, seconds). Skipped without torch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from t2sforge.train.config import TrainConfig, load_train_config
from t2sforge.train.sft import compat_training_kwargs
from t2sforge.train.tokenize import (
    IGNORE_INDEX,
    TokenizedExample,
    pad_batch,
    read_chat_jsonl,
    tokenize_all,
    tokenize_example,
)


def test_config_defaults_and_yaml(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("base_model: x\nlora: {r: 8}\nlearning_rate: 1.0e-4\n")
    cfg = load_train_config(path)
    assert (cfg.base_model, cfg.lora.r, cfg.lora.alpha, cfg.learning_rate) == ("x", 8, 32, 1e-4)
    assert "down_proj" in TrainConfig().lora.target_modules
    empty = tmp_path / "e.yaml"
    empty.write_text("")
    assert load_train_config(empty) == TrainConfig()


def test_compat_training_kwargs() -> None:
    v4 = compat_training_kwargs({"warmup_ratio", "group_by_length"}, 0.03)
    assert v4 == {"warmup_ratio": 0.03, "group_by_length": True}
    v5 = compat_training_kwargs({"warmup_steps", "train_sampling_strategy"}, 0.03)
    assert v5 == {"warmup_steps": 0.03, "train_sampling_strategy": "group_by_length"}


def test_pad_batch() -> None:
    batch = pad_batch([TokenizedExample([1, 2, 3], [-100, 2, 3]), TokenizedExample([4], [4])], 0)
    assert batch["input_ids"] == [[1, 2, 3], [4, 0, 0]]
    assert batch["labels"] == [[-100, 2, 3], [4, -100, -100]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]


# --- everything below needs torch/transformers ------------------------------------------

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from t2sforge.data.build import split_by_database, write_sft_jsonl
    from t2sforge.data.synthetic import generate
    from t2sforge.train.tiny import build_tiny_model

    root = tmp_path_factory.mktemp("train")
    examples = generate(root / "db", instances_per_domain=2, per_db=12, seed=0)
    train, val = split_by_database(examples, val_fraction=0.34)
    write_sft_jsonl(train, root / "db", root / "train.jsonl")
    write_sft_jsonl(val, root / "db", root / "val.jsonl")
    corpus = [m["content"] for conv in read_chat_jsonl(root / "train.jsonl") for m in conv]
    build_tiny_model(root / "tiny", corpus)
    return root


def test_completion_only_masking(workspace: Path) -> None:
    tok = transformers.AutoTokenizer.from_pretrained(workspace / "tiny")
    conv = next(read_chat_jsonl(workspace / "train.jsonl"))
    ex = tokenize_example(tok, conv, max_len=8192)
    assert ex is not None
    target = tok.decode([t for t in ex.labels if t != IGNORE_INDEX])
    assert target.startswith(conv[-1]["content"])  # only the SQL is learned...
    assert target.rstrip().endswith("<|im_end|>")  # ...plus the stop token
    # the masked prefix is exactly the inference prompt
    prompt_len = ex.labels.index(next(t for t in ex.labels if t != IGNORE_INDEX))
    inference_prompt = tok.apply_chat_template(
        conv[:-1], tokenize=False, add_generation_prompt=True
    )
    assert tok.decode(ex.input_ids[:prompt_len]) == inference_prompt
    assert tokenize_example(tok, conv, max_len=10) is None
    with pytest.raises(ValueError, match="assistant"):
        tokenize_example(tok, conv[:-1], max_len=100)
    kept, stats = tokenize_all(tok, [conv, conv], max_len=10)
    assert (kept, stats.dropped_too_long) == ([], 2)


@pytest.mark.slow
def test_train_merge_generate_end_to_end(workspace: Path) -> None:
    from t2sforge.infer import LocalGenerator
    from t2sforge.train.export import merge_adapter
    from t2sforge.train.sft import train

    cfg = TrainConfig(
        base_model=str(workspace / "tiny"),
        train_file=str(workspace / "train.jsonl"),
        val_file=str(workspace / "val.jsonl"),
        output_dir=str(workspace / "out"),
        quantization="none",
        max_seq_len=4096,
        max_steps=12,
        per_device_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=3e-3,
        logging_steps=3,
        eval_steps=6,
        save_steps=6,
    )
    info = train(cfg)
    losses = [h["loss"] for h in info["log_history"] if "loss" in h]
    assert losses[-1] < losses[0]  # it learns *something*
    assert info["eval_loss"] is not None
    assert 0 < info["trainable_params"] < info["total_params"]
    saved = json.loads((workspace / "out" / "run_info.json").read_text())
    assert saved["train_file_sha256"] == info["train_file_sha256"]
    assert (workspace / "out" / "adapter" / "adapter_model.safetensors").exists()

    merged = merge_adapter(workspace / "out" / "adapter", workspace / "merged", dtype="float32")
    conv = next(read_chat_jsonl(workspace / "val.jsonl"))[:-1]
    via_merge = LocalGenerator(str(merged), max_new_tokens=12).generate(conv)
    via_adapter = LocalGenerator(
        str(workspace / "tiny"), adapter_path=str(workspace / "out" / "adapter"), max_new_tokens=12
    ).generate(conv)
    # merging must not change the function the model computes
    assert via_merge == via_adapter
