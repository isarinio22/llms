PY ?= .venv/bin/python
CONFIG ?= configs/base.yaml
SMOKE := configs/smoke.yaml
STAGES := doctor data teacher_label baseline train_student learning_curve score evaluate cascade quantize serve_bench report

.PHONY: help setup test test-fast lint smoke smoke-clean all notebook $(STAGES)

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-14s %s\n", $$1, $$2}'
	@echo "  <stage>        Run one stage: $(STAGES)  (CONFIG=... to change config)"

setup:  ## Create .venv (Python 3.11+) and install pinned CPU requirements
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt
	$(PY) -m pip install -q -e . --no-deps

test:  ## All unit tests (the slow ones download a 135M model once)
	$(PY) -m pytest

test-fast:  ## Unit tests that need no download
	$(PY) -m pytest -m "not slow"

lint:  ## Ruff
	$(PY) -m ruff check src tests scripts

smoke:  ## Every stage end-to-end on CPU with tiny models and data
	$(PY) -m llms run --stage all --config $(SMOKE)

smoke-clean:  ## Delete smoke outputs (keeps the shared teacher cache)
	rm -rf artifacts/smoke results/smoke figures/smoke

all:  ## Full pipeline with CONFIG (default: the T4 run)
	$(PY) -m llms run --stage all --config $(CONFIG)

notebook:  ## Regenerate notebooks/colab_end_to_end.ipynb
	$(PY) scripts/make_notebook.py

$(STAGES):
	$(PY) -m llms run --stage $@ --config $(CONFIG)
