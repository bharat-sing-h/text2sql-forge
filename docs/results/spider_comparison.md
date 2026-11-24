# Spider dev: base vs. fine-tuned

Produced by `llmverdict compare base__spider finetuned__spider` at the end of
[`notebooks/train_and_evaluate.ipynb`](../../notebooks/train_and_evaluate.ipynb), run on a Kaggle Tesla T4.

**Baseline:** `base__spider` (Qwen/Qwen2.5-Coder-1.5B-Instruct, zero-shot)
**Candidate:** `finetuned__spider` (same model + the QLoRA adapter trained by this repo)

| Metric | Baseline | Candidate | Δ | 95% CI (Δ) | p-value | Test | Verdict |
|---|---|---|---|---|---|---|---|
| execution_accuracy | 57.0% | 64.9% | +7.9 pp | [+5.1, +10.6] | < 0.0001 | mcnemar_exact | improved |
| sql_valid | 100.0% | 100.0% | +0.0 pp | [+0.0, +0.0] | 1.0000 | mcnemar_exact | no significant change |

Individual runs: baseline 57.0% [53.9, 59.9], fine-tuned 64.9% [61.9, 67.7]; 1,034 questions each, 0 errors.
Paired on shared samples; α = 0.05; unpaired samples excluded: 0.

## Biggest per-database movers (execution accuracy)

| Database | n | Baseline | Candidate | Δ |
|---|---|---|---|---|
| `poker_player` | 40 | 80.0% | 65.0% | −15.0 pp |
| `museum_visit` | 18 | 88.9% | 83.3% | −5.6 pp |
| `pets_1` | 42 | 52.4% | 54.8% | +2.4 pp |
| `world_1` | 120 | 35.8% | 38.3% | +2.5 pp |
| `flight_2` | 80 | 86.2% | 90.0% | +3.7 pp |
| `student_transcripts_tracking` | 78 | 39.7% | 56.4% | +16.7 pp |
| `wta_1` | 62 | 46.8% | 64.5% | +17.7 pp |
| `concert_singer` | 45 | 62.2% | 80.0% | +17.8 pp |
| `tvshow` | 62 | 51.6% | 75.8% | +24.2 pp |
| `voter_1` | 15 | 46.7% | 73.3% | +26.7 pp |

66 questions regressed (correct before, wrong after).

## Training run (`run_info.json`)

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-Coder-1.5B-Instruct |
| Method | QLoRA: NF4 4-bit base, LoRA r=16, α=32, dropout 0.05, all linear projections |
| Trainable parameters | 18,464,768 (~1.2% of the model) |
| Data | 3,000 examples (seeded sample of Spider train + synthetic), 200 validation, database-disjoint |
| Schedule | 1 epoch, lr 2e-4 cosine, warmup 3%, effective batch 16 (1 × 16 accumulation), max 2,048 tokens |
| Precision / device | fp16 / Tesla T4 (16 GB) |
| Loss | train 0.186, validation 0.182 |
| Training time | 3,055 s (51 min) |

Evaluation: greedy decoding, same prompt and schema rendering for both models, in-process transformers
generation (p50 latency 1.6 s base / 1.9 s fine-tuned with the unmerged adapter).
