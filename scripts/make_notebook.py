"""Generate notebooks/colab_end_to_end.ipynb. The notebook only calls the package (no logic of its own)."""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/isarinio22/llms"
ROOT = "/content/drive/MyDrive/llms"
PY = "/content/venv/bin/python"

STAGES = [
    ("doctor", "Environment report and vLLM probe on the T4. If a probe fails, add `--set teacher.backend=hf` to the teacher stage."),
    ("data", "Stratified train-pool / unlabeled / dev / test split before any teacher call, then cross-split near-duplicate removal."),
    ("teacher_label", "Teacher labels with k=3 self-consistency. Cached after every chunk: rerun this cell after a disconnect and it resumes."),
    ("baseline", "Embeddings + logistic regression, trained on teacher labels and (as a reference) on gold labels."),
    ("train_student", "LoRA SFT on filtered teacher labels, early stopping on dev macro-F1, merged model for serving."),
    ("learning_curve", "The same recipe on 500 / 1,000 / 2,000 examples."),
    ("score", "Score all 77 labels for dev / test / unlabeled; temperature scaling on dev."),
    ("evaluate", "Final test evaluation and paired statistics (McNemar, non-inferiority, sample-size planner)."),
    ("cascade", "Confidence cascade and cost model."),
    ("quantize", "bitsandbytes 8-bit and 4-bit: paired non-inferiority against the fp16 student."),
    ("serve_bench", "vLLM server vs Hugging Face generate: throughput, latency, time to first token."),
    ("report", "Render every table into README.md from results/full/*.json."),
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def build() -> dict:
    cells = [
        md(f"""
# Replace an expensive LLM with a small model you own: end-to-end on a free T4

Runs every stage of [{REPO_URL}]({REPO_URL}) on Colab. Runtime: **T4 GPU**.

* The repo is cloned **into Google Drive**, so data splits, the teacher cache, checkpoints, models and results
  survive a disconnect. After a disconnect, run the first two cells again, then continue: finished stages are skipped.
* Packages go into an isolated venv (`/content/venv`) with the exact pins from `requirements-gpu.lock`,
  so they never clash with Colab's preinstalled packages and no runtime restart is needed.
* Run Colab under the Google account whose Drive you want to use: Drive can only be mounted for the account
  that runs the notebook (top-right avatar).
* Each stage runs in its own process, so GPU memory is fully released between stages.
"""),
        code(f"""
from google.colab import drive
drive.mount('/content/drive')
!test -d {ROOT}/.git || git clone {REPO_URL} {ROOT}
%cd {ROOT}
!git pull --ff-only
"""),
        code(f"""
%%bash
# Isolated environment with every package pinned (requirements-gpu.lock). A few minutes per fresh session.
# Stops at the first error; the last line must print the versions, otherwise do not continue.
set -euo pipefail
pip install -q uv
test -x {PY} || uv venv /content/venv --python 3.12 -q
uv pip install --python {PY} -r requirements-gpu.lock
uv pip install -q --python {PY} -e . --no-deps
{PY} -c "import yaml, torch, vllm, llms; print('OK torch', torch.__version__, '| vllm', vllm.__version__, '|', torch.cuda.get_device_name(0))"
"""),
        code(f"""
%cd {ROOT}
import sys
sys.path.insert(0, "src")
from llms.colab import run_stage   # streams a stage's output and raises if it fails, so "Run all" stops there

PY = "{PY}"
CONFIG = "configs/base.yaml"   # or an ablation, e.g. configs/ablations/student_1p7b.yaml
EXTRA = ""                       # e.g. "--set teacher.backend=hf" if the vLLM probe fails
"""),
    ]
    for name, note in STAGES:
        cells += [md(f"## `{name}`\n\n{note}"), code(f'run_stage("{name}", CONFIG, PY, EXTRA)')]
    cells += [
        md("## Figures"),
        code("""
from IPython.display import Image, Markdown, display
import glob
for f in sorted(glob.glob("figures/full/*.png")):
    display(Markdown(f"**{f}**")); display(Image(f))
"""),
        md("""
## Save the results to GitHub

`results/full/` and `figures/full/` are small and meant to be committed; `artifacts/` is gitignored.
Commit from here (needs a token) or download the two folders and commit locally.
"""),
        code("""
!zip -qr /content/results_full.zip results/full figures/full README.md
from google.colab import files
files.download("/content/results_full.zip")
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "notebooks" / "colab_end_to_end.ipynb"
    out.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {out}")
