"""Colab helpers (standard library only, so Colab's own kernel can import them without the venv).

`run_stage` runs one stage from a notebook or a remote job. `pack_state` / `restore_state` move the small
state of a run (results, teacher cache, labels, scores, the LoRA adapter, stage markers) between a Colab VM
and this machine, so a reclaimed VM costs at most the stage that was running.
"""

from __future__ import annotations

import codecs
import os
import shlex
import subprocess
import tarfile
from pathlib import Path

# Large or cheap-to-rebuild outputs stay on the VM: checkpoints, merged models (rebuilt from the adapter),
# embeddings (the data stage recomputes them in about a minute), learning-curve adapters (only metrics are used).
_EXCLUDE_PARTS = {"checkpoints", "merged"}
_EXCLUDE_NAMES = {"embeddings.npy"}


def _state_files(root: Path, run: str) -> list[Path]:
    files: list[Path] = []
    for top in (f"results/{run}", f"figures/{run}", "artifacts/_shared/teacher_cache", f"artifacts/{run}", "logs"):
        base = root / top
        if base.exists():
            files += [p for p in base.rglob("*") if p.is_file()]
    keep = []
    for p in files:
        rel = p.relative_to(root)
        if _EXCLUDE_PARTS & set(rel.parts) or rel.name in _EXCLUDE_NAMES:
            continue
        if rel.parts[:3] == ("artifacts", run, "data") and rel.name == "_DONE.json":
            continue  # embeddings are not shipped, so the data stage must run again after a restore
        if rel.parts[:3] == ("artifacts", run, "learning_curve") and "adapter" in rel.parts:
            continue
        keep.append(p)
    return keep


def pack_state(root: str | Path, run: str, dest: str | Path) -> int:
    """Write the run's small state to `dest` (tar.gz, replaced atomically) and return the file count."""
    root, dest = Path(root), Path(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    files = _state_files(root, run)
    with tarfile.open(tmp, "w:gz") as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(root)))
    os.replace(tmp, dest)
    return len(files)


def restore_state(src: str | Path, root: str | Path, run: str) -> None:
    """Unpack a state bundle into `root`. If the merged student model is missing (it is never shipped),
    the train_student stage is reopened so it rebuilds the model from the adapter instead of retraining."""
    root = Path(root)
    with tarfile.open(src) as tar:
        tar.extractall(root, filter="data")
    stage = root / "artifacts" / run / "train_student"
    if (stage / "_DONE.json").exists() and not (stage / "main" / "merged").exists():
        (stage / "_DONE.json").unlink()


def list_stages(python: str) -> list[str]:
    out = subprocess.run([python, "-m", "llms", "stages"], capture_output=True, text=True, check=True).stdout
    return [line.split()[0] for line in out.splitlines() if line.strip()]


def run_name(python: str, config: str) -> str:
    code = f"from llms.config import load_config; print(load_config({config!r}).run.name)"
    return subprocess.run([python, "-c", code], capture_output=True, text=True, check=True).stdout.strip()


def run_stage(stage: str, config: str, python: str, extra: str = "", log_file: str | None = None) -> None:
    """Run one pipeline stage in the venv, stream its output, and raise if it fails.

    Raising stops "Run all", so a failed stage is not followed by a cascade of dependency errors.
    `log_file` also appends the output to a file (readable while the kernel is busy).
    """
    cmd = [python, "-m", "llms", "run", "--stage", stage, "--config", config, *shlex.split(extra)]
    print("$", " ".join(cmd), flush=True)
    env = {**os.environ, "MPLBACKEND": "Agg"}  # the kernel's inline backend does not exist in the venv
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    log = open(log_file, "a") if log_file else None  # noqa: SIM115
    assert proc.stdout is not None
    while chunk := os.read(proc.stdout.fileno(), 4096):
        text = decoder.decode(chunk)
        print(text, end="", flush=True)
        if log:
            log.write(text)
            log.flush()
    if log:
        log.close()
    if proc.wait() != 0:
        raise RuntimeError(f"Stage {stage!r} failed (exit code {proc.returncode}). "
                           "Fix the error above, then rerun this cell; finished stages are skipped.")
