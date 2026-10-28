"""Runs inside a Colab VM kernel (sent by scripts/colab_run.py through `colab exec`).

Clones or updates the repo, installs the locked environment, restores a state bundle uploaded by the driver
(if any), then runs the stages one by one. After every stage it writes /content/state.tar.gz and bumps
/content/state.version, which the driver downloads. The kernel stays busy for the whole run, which is what
keeps a Colab CLI session alive. Settings arrive as environment variables (`colab exec --env`):
LLMS_CONFIG (default configs/base.yaml), LLMS_SKIP (comma-separated stages), LLMS_EXTRA (extra CLI flags).
"""

import os
import subprocess
import sys

REPO = "https://github.com/isarinio22/llms"
ROOT = "/content/llms"
PY = "/content/venv/bin/python"
STATE_IN, STATE_OUT, VERSION = "/content/state_in.tar.gz", "/content/state.tar.gz", "/content/state.version"


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


if not os.path.isdir(f"{ROOT}/.git"):
    sh(f"git clone -q {REPO} {ROOT}")
os.chdir(ROOT)
sh("git checkout -q -- . && git pull -q --ff-only")
sh("pip install -q uv")
if not os.path.exists(PY):
    sh("uv venv /content/venv --python 3.12 -q")
sh(f"uv pip install -q --python {PY} -r requirements-gpu.lock")
sh(f"uv pip install -q --python {PY} -e . --no-deps")

sys.path.insert(0, f"{ROOT}/src")
from llms.colab import list_stages, pack_state, restore_state, run_name, run_stage  # noqa: E402

config = os.environ.get("LLMS_CONFIG", "configs/base.yaml")
skip = {s for s in os.environ.get("LLMS_SKIP", "").split(",") if s}
run = run_name(PY, config)
if os.path.exists(STATE_IN):
    restore_state(STATE_IN, ROOT, run)
    os.remove(STATE_IN)
    print("Restored state bundle from the driver", flush=True)
os.makedirs("logs", exist_ok=True)

version = 0
for stage in list_stages(PY):
    if stage in skip:
        continue
    try:
        run_stage(stage, config, PY, os.environ.get("LLMS_EXTRA", ""), log_file="logs/run.log")
    finally:
        n = pack_state(ROOT, run, STATE_OUT)
        version += 1
        with open(VERSION, "w") as f:
            f.write(f"{version} {stage} {n}\n")
with open("logs/run.log", "a") as f:
    f.write("ALL STAGES DONE\n")
pack_state(ROOT, run, STATE_OUT)
with open(VERSION, "w") as f:
    f.write(f"{version + 1} done\n")
print("ALL STAGES DONE", flush=True)
