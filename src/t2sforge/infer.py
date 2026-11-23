"""Local (in-process) inference with transformers, for notebooks and offline evaluation.

Production serving goes through vLLM (see ``deploy/``); this path exists so you can evaluate a
checkpoint right after training on the same Kaggle/Colab GPU without standing up a server.
It doubles as an llm-verdict ``function`` provider: ``target: t2sforge.infer:answer``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


class LocalGenerator:
    def __init__(
        self, model_path: str, adapter_path: str | None = None, max_new_tokens: int = 256
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path)
        model: Any = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict[str, str]]) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy: deterministic and what EX is reported with
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return str(self.tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


@lru_cache(maxsize=1)
def _default_generator() -> LocalGenerator:
    model = os.environ.get("T2S_MODEL")
    if not model:
        raise RuntimeError("set T2S_MODEL (and optionally T2S_ADAPTER) to use t2sforge.infer")
    return LocalGenerator(model, os.environ.get("T2S_ADAPTER"))


def answer(messages: list[dict[str, str]]) -> str:
    """llm-verdict function-provider entry point."""
    return _default_generator().generate(messages)
