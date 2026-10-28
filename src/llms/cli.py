"""Command line: `python -m llms run --stage <name> --config configs/base.yaml`."""

from __future__ import annotations

import argparse
import subprocess
import sys

from llms.config import load_config
from llms.pipeline import STAGES, run_stage
from llms.utils import log, setup_logging
from llms.workspace import Workspace


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llms", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="Run one stage, or all stages in order.")
    r.add_argument("--stage", required=True, choices=[s.name for s in STAGES] + ["all"])
    r.add_argument("--config", default="configs/base.yaml")
    r.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config value, e.g. --set student.model=Qwen/Qwen3-1.7B")
    r.add_argument("--force", action="store_true", help="Rerun even if the stage is up to date.")
    r.add_argument("--skip", action="append", default=[], help="With --stage all: stages to skip.")
    sub.add_parser("stages", help="List stages in execution order.")
    return p


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = _parser().parse_args(argv)
    if args.command == "stages":
        for s in STAGES:
            print(f"{s.name:16s} deps={list(s.deps)}")
        return
    cfg = load_config(args.config, args.overrides)
    if args.stage != "all":
        run_stage(cfg, Workspace.from_config(cfg), args.stage, force=args.force)
        return
    # One subprocess per stage, so GPU memory (vLLM, CUDA context) is fully released in between.
    for s in STAGES:
        if s.name in args.skip:
            continue
        cmd = [sys.executable, "-m", "llms", "run", "--stage", s.name, "--config", args.config]
        for o in args.overrides:
            cmd += ["--set", o]
        if args.force:
            cmd.append("--force")
        log.info("$ %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
