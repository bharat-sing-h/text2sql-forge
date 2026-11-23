"""Chat-template tokenisation with completion-only loss.

Only the assistant's SQL contributes to the loss; the system prompt, schema and question are
masked with ``-100``. Without masking the model spends most of its capacity learning to
reproduce schemas (the bulk of every token sequence) instead of learning to write SQL.

The prompt is rendered with the tokenizer's own chat template and ``add_generation_prompt=True``,
i.e. *exactly* what the model will see at inference time, and the completion tokens are appended
to those prompt ids. Tokenising prompt+completion as one string instead can merge tokens across
the boundary, so the training prompt would not match the serving prompt token-for-token.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

IGNORE_INDEX = -100


class ChatTokenizer(Protocol):
    eos_token: str | None
    eos_token_id: int | None
    pad_token_id: int | None

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]

    @property
    def n_target_tokens(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)


def render_prompt(tokenizer: ChatTokenizer, messages: list[dict[str, str]]) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return str(text)


def tokenize_example(
    tokenizer: ChatTokenizer, messages: list[dict[str, str]], max_len: int
) -> TokenizedExample | None:
    """Return ``None`` when the example doesn't fit in ``max_len``.

    Over-long examples are dropped rather than truncated: cutting the schema could remove the
    very table the gold SQL uses, which teaches the model to hallucinate tables.
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("last message must be the assistant target")
    prompt_text = render_prompt(tokenizer, messages[:-1])
    full_text = str(tokenizer.apply_chat_template(messages, tokenize=False))
    if not full_text.startswith(prompt_text):
        # templates that rewrite earlier turns: fall back to target + EOS
        completion_text = messages[-1]["content"] + (tokenizer.eos_token or "")
    else:
        completion_text = full_text[len(prompt_text) :]
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    completion_ids = list(tokenizer(completion_text, add_special_tokens=False)["input_ids"])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in completion_ids:
        completion_ids.append(tokenizer.eos_token_id)
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_len:
        return None
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
    return TokenizedExample(input_ids, labels)


def read_chat_jsonl(path: str | Path) -> Iterator[list[dict[str, str]]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)["messages"]


@dataclass
class TokenizeStats:
    kept: int = 0
    dropped_too_long: int = 0
    max_len_seen: int = 0


def tokenize_all(
    tokenizer: ChatTokenizer, conversations: Iterable[list[dict[str, str]]], max_len: int
) -> tuple[list[TokenizedExample], TokenizeStats]:
    stats = TokenizeStats()
    out = []
    for messages in conversations:
        ex = tokenize_example(tokenizer, messages, max_len)
        if ex is None:
            stats.dropped_too_long += 1
            continue
        stats.kept += 1
        stats.max_len_seen = max(stats.max_len_seen, len(ex.input_ids))
        out.append(ex)
    return out, stats


def pad_batch(batch: list[TokenizedExample], pad_token_id: int) -> dict[str, list[list[int]]]:
    """Right-pad to the longest sequence in the batch (dynamic padding, not max_len)."""
    width = max(len(ex.input_ids) for ex in batch)
    input_ids, labels, attention = [], [], []
    for ex in batch:
        pad = width - len(ex.input_ids)
        input_ids.append(ex.input_ids + [pad_token_id] * pad)
        labels.append(ex.labels + [IGNORE_INDEX] * pad)
        attention.append([1] * len(ex.input_ids) + [0] * pad)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}
