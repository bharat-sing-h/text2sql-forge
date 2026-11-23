# Design notes

## 1. Why a small fine-tuned model

A 1.5B model fine-tuned for one task can run on a single cheap GPU, or on a CPU through GGUF, with predictable latency, data that never leaves your infrastructure, and no per-token API cost. The open question is *how close it gets to a large general model*, and this project is built to answer that with evidence. The same evaluation (`configs/eval_spider.yaml`) runs against the fine-tuned model, its own base model, and any hosted API, compared with paired statistics.

## 2. Data

- **Database-disjoint splits.** Validation holds out whole databases. A random row split leaves questions about the same schema on both sides and over-states accuracy on a new customer's database. Spider's own train/dev split follows the same principle.
- **Automatic leakage check.** `t2s data build` refuses to write files if any training database or question also appears in the evaluation set. (The CLI test triggers this on purpose.)
- **Execution-verified synthetic data.** Every generated pair is run before it's kept and must return a non-empty answer. Degenerate templates are filtered out. One example caught in review: a HAVING template produced "more than 0" questions, which is now filtered and tested. Top-k questions are only generated on columns with unique values, because ties would make the gold answer ambiguous.
- **One prompt format.** `data/prompt.py` and `data/schema.py` render prompts for training, evaluation *and* serving. A mismatch between the training and serving prompts (train/serve skew) is one of the easiest ways to lose accuracy silently after fine-tuning.
- **Schema representation.** `CREATE TABLE` statements with keys and up to three example values per column, following Chang & Fosler-Lussier (2023), who found table definitions plus sample content to be a strong representation. Example values matter most for categorical filters (`'paid'` vs `'Paid'`). Very large schemas drop the example values first and keep the structure.

## 3. Training

- **QLoRA:** NF4 4-bit base weights, double quantisation, LoRA r=16 on *all* linear projections. Attention-only LoRA tends to under-fit structured-output tasks.
- **Loss on the SQL only.** Prompt tokens are labelled `-100`. In text-to-SQL the schema is most of every sequence; without masking, most of the gradient would go into reproducing schemas.
- **Token-exact prompts.** The prompt is rendered with the tokenizer's own chat template and `add_generation_prompt=True`, and the completion tokens are appended to those prompt ids. Tokenising prompt and completion as one string can merge tokens across the boundary, so the trained prompt would differ token-for-token from the served one. A test checks that the masked prefix decodes to exactly the inference prompt.
- **Drop, don't truncate.** Truncating a long schema can remove the very table the gold SQL uses, which teaches the model to invent tables. Over-long examples are dropped and the count is logged.
- **Plain `Trainer` + PEFT,** not a higher-level SFT wrapper, so each step is visible and tested. Transformers 5.x renamed `warmup_ratio` and `group_by_length`; a small compatibility shim keeps the trainer working on both major versions, because Kaggle and Colab images often lag behind PyPI.
- **Lineage:** `run_info.json` records the config, the training-file SHA-256, the git SHA, library versions, the device and the full loss history.
- **CI trains for real.** A tiny randomly-initialised Qwen2-architecture model and a BPE tokenizer are built offline, and the actual training code runs on CPU in seconds. The test also checks that the merged model and base+adapter produce **identical** generations, which catches merge bugs.

## 4. Evaluation

- **Execution accuracy** rather than exact string match: many different SQL strings are correct. Comparison rules follow the Spider test-suite evaluation: multiset comparison (ordered only when the gold query has a top-level `ORDER BY`), a column-permutation search pruned by per-column value multisets, and numeric normalisation (`3 == 3.0`, float rounding).
- **Gold-query failures are N/A, not zeros.** Benchmarks contain a few broken labels, and the model shouldn't be penalised for them.
- **Statistics come from llm-verdict:** Wilson CIs, exact McNemar for paired base-vs-fine-tuned comparisons, per-tag (difficulty, source, database) breakdowns, and CI gating. llm-verdict gained a plugin system for this project, so the SQL scorer lives here, not there.
- **Error taxonomy.** Failures are bucketed (`unknown_column`, `wrong_result`, `aggregation`, ...) to point at the next fix. Many `unknown_column` errors suggest better schema linking; many `wrong_result` errors suggest semantic errors, so more data or a larger model.

## 5. Serving

- **vLLM** handles batching and paged attention. `--enable-prefix-caching` matters here because every question about the same database shares a long schema prefix, so much of the prefill work can be reused.
- **The gateway owns product logic:** guard, sandbox, self-correction, validation and metrics. The model server stays a replaceable component (vLLM, Ollama, TGI or a hosted API).
- **Self-correction:** on a guard rejection or SQLite error, the model sees its own query and the error once. This fixes typical near-misses for the cost of one extra generation, and only on failures. The repair rate is a metric, because a rising repair rate is an early signal of schema drift or a bad model deploy.
- **Truncation detection:** the gateway fetches `limit + 1` rows. An earlier version fetched exactly `limit`, so it could never tell callers their results were cut off. A test caught it.
- **Kubernetes specifics:** vLLM gets a startup probe (loading a model takes minutes), a `Recreate` rollout (one GPU per pod), `/dev/shm` as memory, and a persistent Hugging Face cache. The API runs as non-root with a read-only root filesystem and read-only database mounts, scales with an HPA, and its readiness depends on the model server.

## 6. Known limitations and next steps

- Only SQLite is supported. Other dialects would need dialect-aware guards (sqlglot supports them) and a sandbox per engine.
- Single-turn questions only. Conversational follow-ups would need the previous SQL in context.
- Schema linking (retrieving only the relevant tables for very large databases) isn't implemented, so BIRD-scale schemas fall back to dropping example values.
- Possible next experiments: self-consistency (sample *k* queries and vote by execution result), a DPO round on execution feedback, and comparing r=8/16/32 and 1.5B vs 7B.
