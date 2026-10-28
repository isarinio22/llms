PY ?= .venv/bin/python
CONFIG ?= configs/base.yaml
SMOKE := configs/smoke.yaml
STAGES := doctor data teacher_label baseline train_student learning_curve score evaluate cascade quantize serve_bench report

.PHONY: help setup test test-fast lint smoke smoke-clean all notebook lock colab colab-fetch $(STAGES)

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

lock:  ## Re-resolve requirements-gpu.lock for Colab (Linux x86_64, Python 3.12)
	.venv/bin/uv pip compile requirements-gpu.txt --python-platform x86_64-manylinux_2_35 --python-version 3.12 --quiet -o requirements-gpu.lock
	sed -i.bak '1s|^|# Full transitive pins for Colab (Linux x86_64, Python 3.12). Regenerate with: make lock\n|' requirements-gpu.lock && rm requirements-gpu.lock.bak

COLAB ?= $(HOME)/.local/bin/colab
SESSION ?= llms

colab:  ## Run the pipeline on a Colab T4 from this terminal (Colab CLI; finished stages are skipped)
	$(COLAB) sessions | grep -q '^\[$(SESSION)\]' || $(COLAB) new -s $(SESSION) --gpu T4  # `colab status` exits 0 even for unknown sessions
	$(COLAB) sessions | grep -q '^\[$(SESSION)\]' || { echo "No GPU session (free-tier limit or capacity): retry later"; exit 1; }
	$(COLAB) exec -s $(SESSION) --timeout 43200 --env LLMS_CONFIG=$(CONFIG) -f scripts/colab_job.py

colab-fetch:  ## Download results, figures, stage markers and the teacher cache from the Colab VM into this repo
	echo 'import subprocess; subprocess.run("cd /content/llms && rm -f /content/outputs.zip && zip -qr /content/outputs.zip results figures artifacts/_shared $$(ls -d artifacts/*/*/_DONE.json)", shell=True, check=True)' | $(COLAB) exec -s $(SESSION)
	$(COLAB) download -s $(SESSION) /content/outputs.zip /tmp/llms_outputs.zip
	unzip -oq /tmp/llms_outputs.zip -d . && rm /tmp/llms_outputs.zip

$(STAGES):
	$(PY) -m llms run --stage $@ --config $(CONFIG)
