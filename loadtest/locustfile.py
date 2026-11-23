"""Load test for the API gateway.

    locust -f loadtest/locustfile.py --host http://localhost:8080 --headless \
        -u 32 -r 4 -t 5m --csv reports/load

Questions are sampled from an evaluation file so the load resembles real traffic (varied schemas
and query difficulty) rather than one cached prompt.
"""

from __future__ import annotations

import json
import os
import random

from locust import HttpUser, between, task

EVAL_FILE = os.environ.get("T2S_EVAL_FILE", "data/eval/synthetic_eval.jsonl")


def _load() -> list[tuple[str, str]]:
    items = []
    with open(EVAL_FILE, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            question = row["messages"][-1]["content"].split("### Question\n", 1)[1].split("\n\n")[0]
            items.append((row["reference"]["db_id"], question))
    return items


QUESTIONS = _load()


class Analyst(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def query(self) -> None:
        db_id, question = random.choice(QUESTIONS)
        with self.client.post(
            "/v1/query",
            json={"db_id": db_id, "question": question},
            name="/v1/query",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif resp.json().get("error"):
                resp.failure("sql error")  # counted separately from transport failures
