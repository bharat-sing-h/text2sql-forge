"""QLoRA supervised fine-tuning.

Plain ``transformers.Trainer`` + PEFT with an explicit tokenisation/masking step (see
:mod:`t2sforge.train.tokenize`) rather than a higher-level SFT wrapper: every step that affects
what the model learns is visible and unit-tested, and the code doesn't break when wrapper APIs
change between library versions.

Memory budget for the default config (Qwen2.5-Coder-1.5B, NF4, r=16, seq 2048, batch 4,
gradient checkpointing) fits a free 16 GB T4 on Kaggle/Colab.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from t2sforge.train.config import TrainConfig
from t2sforge.train.tokenize import (
    TokenizedExample,
    TokenizeStats,
    pad_batch,
    read_chat_jsonl,
    tokenize_all,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_precision(cfg: TrainConfig) -> str:
    import torch

    if cfg.precision != "auto":
        return cfg.precision
    if torch.cuda.is_available():
        # T4/V100 have no bf16 support; Ampere+ do and bf16 is more stable than fp16
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp32"


class _ListDataset:
    """Items are dicts with ``input_ids`` so the length-grouped sampler can read lengths."""

    def __init__(self, items: list[TokenizedExample]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        item = self.items[idx]
        return {"input_ids": item.input_ids, "labels": item.labels}


def compat_training_kwargs(fields: set[str], warmup_ratio: float) -> dict[str, Any]:
    """Map version-specific TrainingArguments names (transformers 4.x vs 5.x).

    5.x folded ``warmup_ratio`` into a fractional ``warmup_steps`` and replaced
    ``group_by_length`` with ``train_sampling_strategy``. Kaggle/Colab images often ship an
    older release than a fresh pip install, so the trainer supports both.
    """
    kwargs: dict[str, Any] = {}
    if "warmup_ratio" in fields:
        kwargs["warmup_ratio"] = warmup_ratio
    else:
        kwargs["warmup_steps"] = warmup_ratio
    # batches of similar length -> less padding -> faster epochs
    if "group_by_length" in fields:
        kwargs["group_by_length"] = True
    elif "train_sampling_strategy" in fields:
        kwargs["train_sampling_strategy"] = "group_by_length"
    return kwargs


def load_model_and_tokenizer(cfg: TrainConfig, precision: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    kwargs: dict[str, Any] = {"dtype": dtype}
    if cfg.quantization == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.float16,
        )
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    model.config.use_cache = False  # incompatible with gradient checkpointing during training
    return model, tokenizer


def apply_lora(model: Any, cfg: TrainConfig) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.quantization == "4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def train(cfg: TrainConfig) -> dict[str, Any]:
    """Run QLoRA SFT; returns (and writes) run metadata for lineage."""
    import dataclasses

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(cfg.seed)
    precision = resolve_precision(cfg)
    model, tokenizer = load_model_and_tokenizer(cfg, precision)

    train_items, train_stats = tokenize_all(
        tokenizer, read_chat_jsonl(cfg.train_file), cfg.max_seq_len
    )
    val_items: list[TokenizedExample] = []
    val_stats: TokenizeStats | None = None
    if cfg.val_file:
        val_items, val_stats = tokenize_all(
            tokenizer, read_chat_jsonl(cfg.val_file), cfg.max_seq_len
        )
    if not train_items:
        raise ValueError("no training examples fit within max_seq_len")

    model = apply_lora(model, cfg)
    trainable, total = model.get_nb_trainable_parameters()

    pad_id = int(tokenizer.pad_token_id)

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        examples = [TokenizedExample(b["input_ids"], b["labels"]) for b in batch]
        return {k: torch.tensor(v) for k, v in pad_batch(examples, pad_id).items()}

    has_val = bool(val_items)
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        lr_scheduler_type=cfg.lr_scheduler,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if has_val else "no",
        eval_steps=cfg.eval_steps if has_val else None,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        gradient_checkpointing=False,  # already enabled on the model by apply_lora
        remove_unused_columns=False,
        report_to=cfg.report_to,
        seed=cfg.seed,
        **compat_training_kwargs(
            {f.name for f in dataclasses.fields(TrainingArguments)}, cfg.warmup_ratio
        ),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_ListDataset(train_items),
        eval_dataset=_ListDataset(val_items) if has_val else None,
        data_collator=collate,
    )
    started = time.time()
    result = trainer.train()
    final_eval = trainer.evaluate() if has_val else {}

    out = Path(cfg.output_dir) / "adapter"
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    info = {
        "config": cfg.model_dump(),
        "precision": precision,
        "trainable_params": int(trainable),
        "total_params": int(total),
        "train_examples": train_stats.kept,
        "train_dropped_too_long": train_stats.dropped_too_long,
        "val_examples": val_stats.kept if val_stats else 0,
        "train_loss": float(result.training_loss),
        "eval_loss": float(final_eval["eval_loss"]) if "eval_loss" in final_eval else None,
        "train_runtime_s": round(time.time() - started, 1),
        "train_file_sha256": _sha256(cfg.train_file),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "log_history": trainer.state.log_history,
    }
    (Path(cfg.output_dir) / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    return info
