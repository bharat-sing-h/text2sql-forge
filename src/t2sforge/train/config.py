from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class LoraSettings(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # all linear projections: attention-only LoRA under-fits structured-output tasks
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class TrainConfig(BaseModel):
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    train_file: str = "data/sft/train.jsonl"
    val_file: str | None = "data/sft/val.jsonl"
    output_dir: str = "outputs/qwen-coder-1.5b-sql-lora"

    quantization: Literal["4bit", "none"] = "4bit"  # QLoRA: base weights in NF4
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    max_seq_len: int = 2048
    lora: LoraSettings = Field(default_factory=LoraSettings)

    learning_rate: float = 2e-4
    num_epochs: float = 2.0
    max_steps: int = -1
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True

    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    seed: int = 42
    report_to: list[str] = Field(default_factory=list)  # e.g. ["mlflow"] or ["wandb"]


def load_train_config(path: str | Path) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return TrainConfig.model_validate(data)
