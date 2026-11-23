.PHONY: install install-train lint format typecheck test check smoke serve docker clean

install:
	pip install -e ".[dev]" -e "llmverdict @ git+https://github.com/bharat-sing-h/llm-verdict"

install-train:
	pip install -e ".[train,dev]"

lint:
	ruff check src tests loadtest
	ruff format --check src tests loadtest

format:
	ruff check --fix src tests loadtest
	ruff format src tests loadtest

typecheck:
	mypy

test:
	pytest --cov --cov-report=term-missing

check: lint typecheck test

# End-to-end pipeline on CPU in ~1 minute: synthetic data -> tiny model -> QLoRA-style
# training -> llm-verdict evaluation of base vs fine-tuned. Proves the plumbing, not quality.
smoke:
	t2s data synth --out .t2s/synthetic --instances 2 --per-db 20
	t2s data build --source synthetic --root .t2s/synthetic --out .t2s --val-fraction 0.25
	t2s tiny-model --out .t2s/tiny-model --corpus .t2s/sft/train.jsonl
	t2s train configs/smoke_tiny.yaml
	T2S_MODEL=.t2s/tiny-model RUN_NAME=tiny-base EVAL_SET=synthetic DB_ROOT=.t2s/synthetic/databases LIMIT=30 \
		llmverdict run configs/smoke_eval.yaml --label base
	T2S_MODEL=.t2s/tiny-model T2S_ADAPTER=.t2s/out/adapter RUN_NAME=tiny-ft EVAL_SET=synthetic DB_ROOT=.t2s/synthetic/databases LIMIT=30 \
		llmverdict run configs/smoke_eval.yaml --label finetuned
	llmverdict compare base__synthetic finetuned__synthetic

serve:
	t2s serve

docker:
	docker build -f deploy/docker/Dockerfile.api -t text2sql-forge-api:latest .

clean:
	rm -rf .t2s .llmverdict .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
