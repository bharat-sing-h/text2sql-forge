# text2sql-forge

[![CI](https://github.com/bharat-sing-h/text2sql-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/bharat-sing-h/text2sql-forge/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![mypy](https://img.shields.io/badge/mypy-strict-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Fine-tune, serve and evaluate a small open LLM that turns questions into SQL, end to end and production-shaped.**

```text
"Which customers from Delhi spent more than the average order value?"
        │
        ▼
  schema prompt ─► fine-tuned Qwen2.5-Coder (vLLM) ─► SQL guard ─► read-only sandbox ─► rows
                                       ▲                                     │
                                       └──────── self-correction on error ◄──┘
```

| Stage | What's in it |
|---|---|
| **Data** | Spider and BIRD loaders, an execution-verified synthetic generator, database-disjoint splits with an automatic leakage check, and one prompt format shared by training and serving |
| **Training** | QLoRA (NF4 4-bit) with LoRA on all linear layers and loss on the SQL only. Fits a **free 16 GB T4**. Every run writes lineage metadata (data hash, git SHA, losses). Works with transformers 4.x and 5.x |
| **Evaluation** | Execution accuracy (the Spider/BIRD metric: order-aware comparison, column-permutation search, numeric normalisation), run through [llm-verdict](https://github.com/bharat-sing-h/llm-verdict) for confidence intervals, paired significance tests and per-difficulty breakdowns, plus an error taxonomy that says *why* queries fail |
| **Serving** | vLLM with prefix caching, and a FastAPI gateway that adds a SQL guard, sandboxed execution, one self-correction round, input validation, and Prometheus metrics |
| **Ops** | Docker Compose stack (vLLM, API, Prometheus, Grafana dashboard), a Helm chart (GPU model server plus an autoscaled API), a Locust load test, a CPU fallback through Ollama/GGUF, and CI that runs the real training code on a tiny model |

## Quickstart (CPU, about 1 minute, no downloads)

```bash
git clone https://github.com/bharat-sing-h/text2sql-forge && cd text2sql-forge
pip install -e ".[train,dev]" "llmverdict @ git+https://github.com/bharat-sing-h/llm-verdict"
make smoke
```

`make smoke` runs the **whole pipeline on CPU**: it generates synthetic databases and verified question/SQL pairs, builds database-disjoint splits, builds a tiny random model offline, fine-tunes it with LoRA, evaluates base vs. fine-tuned with execution accuracy, and prints a paired statistical comparison. The tiny model learns nothing useful, so this proves the plumbing, not quality. Real numbers come from the GPU notebook below.

## Train a real model (free GPU)

Open [`notebooks/train_and_evaluate.ipynb`](notebooks/train_and_evaluate.ipynb) on **Kaggle** or **Colab** with a T4 GPU. It:

1. builds Spider train (plus synthetic augmentation) and holds out whole databases for validation;
2. evaluates **Qwen2.5-Coder-1.5B-Instruct** zero-shot on Spider dev;
3. fine-tunes with QLoRA ([config](configs/qlora_qwen2.5-coder-1.5b.yaml));
4. evaluates the fine-tuned model on the **same questions**;
5. runs `llmverdict compare`, which gives execution-accuracy Δ with a 95% CI, an exact McNemar p-value, and per-difficulty movers.

Or from a shell on any GPU machine:

```bash
t2s data synth --out data/synthetic
t2s data build --source spider --root data/spider --add-synthetic data/synthetic --out data
t2s train configs/qlora_qwen2.5-coder-1.5b.yaml
t2s merge outputs/qwen2.5-coder-1.5b-sql/adapter outputs/merged
```

Spider and BIRD aren't redistributed here (CC BY-SA 4.0); download them from their official pages and point `--root` at the extracted folder.

## Serve

```bash
MODEL=/models/merged docker compose -f deploy/docker-compose.yml up -d
curl -s localhost:8080/v1/query -H 'content-type: application/json' \
  -d '{"db_id": "retail_0", "question": "What are the names of the 3 most expensive products?"}'
```

```json
{
  "sql": "SELECT name FROM products ORDER BY price DESC LIMIT 3",
  "columns": ["name"], "rows": [["Laptop"], ["Monitor"], ["Desk"]],
  "truncated": false, "error": null, "repaired": false,
  "attempts": [{"sql": "SELECT name FROM products ORDER BY price DESC LIMIT 3", "error": null, "stage": "ok"}],
  "latency_ms": {"total": 412.3, "generation": 409.8, "execution": 0.6}
}
```
<sub>Illustrative response shape. The values depend on your model and data.</sub>

| Endpoint | Purpose |
|---|---|
| `POST /v1/query` | question → SQL → rows (`execute: false` returns the SQL only) |
| `GET /v1/databases`, `/v1/databases/{id}/schema` | discovery |
| `GET /healthz`, `/readyz` | liveness; readiness also checks that the model server is up and databases are mounted |
| `GET /metrics` | Prometheus: outcomes, per-stage latency, self-corrections, tokens |

**Kubernetes:** `helm install t2s deploy/helm/text2sql --set model.name=<your-hf-repo>`. This runs vLLM on a GPU node (with a startup probe for slow model loads and a `Recreate` rollout so two copies don't fight over one GPU) and the API as a non-root, read-only, HPA-autoscaled deployment. **No GPU:** export to GGUF and use [`deploy/docker-compose.cpu.yml`](deploy/docker-compose.cpu.yml) with Ollama.

**Load test:** `locust -f loadtest/locustfile.py --host http://localhost:8080`.

## Evaluate anything

Evaluation is an [llm-verdict](https://github.com/bharat-sing-h/llm-verdict) plugin, so any OpenAI-compatible endpoint can be compared against any other with proper statistics:

```bash
MODEL=sql llmverdict run configs/eval_spider.yaml --label finetuned
MODEL=gpt-4o-mini BASE_URL=https://api.openai.com/v1 llmverdict run configs/eval_spider.yaml --label gpt4o-mini
llmverdict compare gpt4o-mini__spider finetuned__spider
llmverdict gate finetuned__spider --baseline gpt4o-mini__spider --max-drop 0.02   # CI gate
```

Each failed sample is labelled `unknown_column`, `unknown_table`, `syntax`, `aggregation`, `timeout`, `invalid_sql` or `wrong_result`, so you can see what to fix next (schema linking? aggregation?), not just how often the model fails.

## Safety: defence in depth

Model-written SQL is untrusted input. Two independent layers mean a bypass of either one alone isn't enough:

1. **Parser guard** (`sqlglot`, not regexes): exactly one statement, read-only roots only, no `ATTACH` or `PRAGMA`, no file or extension functions.
2. **Sandbox**: `mode=ro` connection, a SQLite **authorizer** that allows reads only, a wall-clock timeout through a progress handler, and a row cap. The tests run `DELETE`, `INSERT`, `CREATE`, `ATTACH` and `PRAGMA` *directly* against the sandbox, skipping the guard, and confirm the data is unchanged.

The gateway also validates `db_id` against path traversal and fetches `limit + 1` rows so it can tell callers when results were truncated.

## Project layout

```
src/t2sforge/
  sql/       guard, sandboxed execution, execution-accuracy comparison, SQL extraction
  data/      schema prompts, Spider/BIRD loaders, synthetic generator, splits and leakage checks
  train/     QLoRA SFT, completion-only tokenisation, adapter merge, tiny offline model
  serve/     model-server client, question→SQL→rows pipeline, FastAPI app and metrics
  eval/      llm-verdict plugin (execution_accuracy, sql_valid)
configs/     training and evaluation configs
deploy/      Dockerfile, Compose (GPU and CPU), Prometheus, Grafana, Helm chart
notebooks/   Kaggle/Colab end-to-end experiment
```

See [docs/design.md](docs/design.md) for the reasoning behind each decision.

## Development

```bash
make install-train   # dev + training extras
make check           # ruff + mypy --strict + pytest (58 tests, 93% coverage)
make smoke           # end-to-end CPU pipeline
```

## Acknowledgements
Spider (Yu et al., 2018) and BIRD (Li et al., 2023) benchmarks; Qwen2.5-Coder (Qwen team); QLoRA (Dettmers et al., 2023); LoRA (Hu et al., 2021); vLLM (Kwon et al., 2023).

## License
MIT
