"""Merge a LoRA adapter into its base model for serving.

vLLM can also serve adapters directly (``--enable-lora --lora-modules sql=<adapter_dir>``), which
lets one base model host several fine-tunes. Merging gives a standalone checkpoint with zero
adapter overhead, and is required for conversion to GGUF (llama.cpp / Ollama, CPU serving).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def merge_adapter(adapter_dir: str | Path, out_dir: str | Path, *, dtype: str = "float16") -> Path:
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        dtype
    ]
    model: Any = AutoPeftModelForCausalLM.from_pretrained(str(adapter_dir), dtype=torch_dtype)
    merged = model.merge_and_unload()
    out = Path(out_dir)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(out)
    (out / "t2sforge_export.json").write_text(
        json.dumps(
            {
                "adapter": str(adapter_dir),
                "dtype": dtype,
            },
            indent=2,
        )
    )
    return out
