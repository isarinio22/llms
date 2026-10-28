"""Stage registry: order, dependencies, cache keys and execution."""

from __future__ import annotations

import hashlib
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from llms.config import Config
from llms.utils import log, set_seed
from llms.workspace import Workspace


@dataclass(frozen=True)
class Stage:
    name: str
    target: str  # "module:function", imported lazily so the CLI starts fast
    sections: tuple[str, ...] = ()
    deps: tuple[str, ...] = ()
    cached: bool = True


STAGES: tuple[Stage, ...] = (
    Stage("doctor", "llms.doctor:run", cached=False),
    Stage("data", "llms.data:run", ("run.seed", "data")),
    Stage("teacher_label", "llms.teacher.label:run", ("teacher", "labels_file"), ("data",)),
    Stage("baseline", "llms.baseline:run", ("run.seed",), ("teacher_label",)),
    Stage("train_student", "llms.student.train:run", ("student",), ("teacher_label",)),
    Stage("learning_curve", "llms.student.curve:run", ("student",), ("train_student",)),
    Stage("score", "llms.scoring:run", ("scoring", "calibration"), ("train_student",)),
    Stage("evaluate", "llms.evaluate:run", ("evaluation",), ("score", "baseline")),
    Stage("cascade", "llms.cascade:run", ("cascade",), ("evaluate",)),
    Stage("quantize", "llms.quantize:run", ("serving.quant_modes", "evaluation"), ("score",)),
    Stage("serve_bench", "llms.serving:run", ("serving",), ("train_student",)),
    Stage("report", "llms.report:run", cached=False),
)
BY_NAME = {s.name: s for s in STAGES}


def stage_key(cfg: Config, name: str) -> str:
    """Hash of the stage's own config sections plus its upstream keys (upstream change => rerun)."""
    stage = BY_NAME[name]
    parts = [cfg.section_hash(*stage.sections)] if stage.sections else []
    if "labels_file" in stage.sections:
        parts.append(hashlib.sha256(Path(cfg.labels_file).read_bytes()).hexdigest()[:12])
    parts += [stage_key(cfg, d) for d in stage.deps]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def run_stage(cfg: Config, ws: Workspace, name: str, force: bool = False) -> None:
    stage = BY_NAME[name]
    key = stage_key(cfg, name)
    if stage.cached and not force and ws.is_done(name, key):
        log.info("Stage %s is up to date (key %s), skipping. Use --force to rerun.", name, key)
        return
    for dep in stage.deps:
        if not ws.is_done(dep, stage_key(cfg, dep)):
            raise RuntimeError(f"Stage {name!r} needs {dep!r} first (missing or outdated).")
    module, func = stage.target.split(":")
    fn = getattr(importlib.import_module(module), func)
    set_seed(cfg.run.seed)
    ws.snapshot_config(name)
    log.info("=== Stage %s (key %s) ===", name, key)
    start = time.perf_counter()
    fn(cfg, ws, key)
    seconds = round(time.perf_counter() - start, 1)
    if stage.cached:
        ws.mark_done(name, key, seconds=seconds, python=sys.version.split()[0])
    log.info("=== Stage %s done in %.1fs ===", name, seconds)
