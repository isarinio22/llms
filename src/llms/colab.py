"""Notebook helper (standard library only, so Colab's own kernel can import it without the venv)."""

from __future__ import annotations

import codecs
import os
import shlex
import subprocess


def run_stage(stage: str, config: str, python: str, extra: str = "", log_file: str | None = None) -> None:
    """Run one pipeline stage in the venv, stream its output, and raise if it fails.

    Raising stops "Run all", so a failed stage is not followed by a cascade of dependency errors.
    `log_file` also appends the output to a file (readable while the kernel is busy).
    """
    cmd = [python, "-m", "llms", "run", "--stage", stage, "--config", config, *shlex.split(extra)]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
