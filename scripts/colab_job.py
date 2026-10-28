"""Runs inside a Colab VM kernel (sent by `colab exec -f scripts/colab_job.py`, see `make colab`).

Clones or updates the repo, installs the locked environment in a venv, then runs the pipeline. The kernel
stays busy for the whole run, which is what keeps a Colab CLI session alive. Settings come from environment
variables passed with `colab exec --env`: LLMS_CONFIG (default configs/base.yaml), LLMS_STAGE (default all),
LLMS_EXTRA (extra CLI flags, e.g. "--set teacher.backend=hf").
"""

import os
import subprocess
import sys

REPO = "https://github.com/isarinio22/llms"
ROOT = "/content/llms"
PY = "/content/venv/bin/python"


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


if not os.path.isdir(f"{ROOT}/.git"):
    sh(f"git clone -q {REPO} {ROOT}")
os.chdir(ROOT)
sh("git pull -q --ff-only")
sh("pip install -q uv")
if not os.path.exists(PY):
    sh("uv venv /content/venv --python 3.12 -q")
sh(f"uv pip install -q --python {PY} -r requirements-gpu.lock")
sh(f"uv pip install -q --python {PY} -e . --no-deps")

sys.path.insert(0, f"{ROOT}/src")
from llms.colab import run_stage  # noqa: E402

config = os.environ.get("LLMS_CONFIG", "configs/base.yaml")
os.makedirs("logs", exist_ok=True)
run_stage(os.environ.get("LLMS_STAGE", "all"), config, PY, os.environ.get("LLMS_EXTRA", ""), log_file="logs/run.log")
sh(f"{PY} -m llms status --config {config}")
