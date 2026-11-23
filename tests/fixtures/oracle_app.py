"""Fake 'model' for integration tests: answers from a question -> SQL lookup table.

Configured by the test via ORACLE_FILE; it is right on most questions, wrong on some and emits
invalid SQL on others, so every scorer path is exercised through llm-verdict.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _table() -> dict[str, str]:
    with open(os.environ["ORACLE_FILE"], encoding="utf-8") as fh:
        return json.load(fh)


def answer(messages: list[dict[str, str]]) -> str:
    question = messages[-1]["content"].split("### Question\n", 1)[1].split("\n\n", 1)[0]
    return _table().get(question, "I am not sure.")
