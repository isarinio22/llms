"""Command line: `python -m llms run --stage <name> --config configs/base.yaml`."""

from __future__ import annotations

import argparse
import subprocess
import sys

from llms.config import Config, load_config
from llms.pipeline import STAGES, run_stage, stage_key
from llms.utils import log, setup_logging
from llms.workspace import Workspace, read_json, read_jsonl


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
    st = sub.add_parser("status", help="Which stages are done, and teacher-labeling progress.")
    st.add_argument("--config", default="configs/base.yaml")
    st.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    sub.add_parser("stages", help="List stages in execution order.")
    return p


def print_status(cfg: Config, ws: Workspace) -> None:
    """One line per stage (done / outdated / not done) plus how many items the teacher has answered."""
    for s in STAGES:
        marker = ws.artifacts / s.name / "_DONE.json"
        if not s.cached:
            state = "runs every time"
        elif ws.is_done(s.name, stage_key(cfg, s.name)):
            state = f"done ({read_json(marker).get('seconds', '?')} s)"
        elif marker.exists():
            state = "outdated (config or an upstream stage changed)"
        else:
            state = "not done"
        print(f"  {s.name:16s} {state}")
    splits = ws.artifacts / "data" / "splits.parquet"
    if splits.exists():
        from llms.data import load_splits
        from llms.teacher.label import cache_path

        uids = set(load_splits(ws).uid)
        answered = {r["uid"] for r in read_jsonl(cache_path(cfg, ws)) if r["round"] == 0} & uids
        print(f"  teacher cache: {len(answered):,} / {len(uids):,} items answered")


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = _parser().parse_args(argv)
    if args.command == "stages":
        for s in STAGES:
            print(f"{s.name:16s} deps={list(s.deps)}")
        return
    cfg = load_config(args.config, args.overrides)
    if args.command == "status":
        print(f"Run '{cfg.run.name}' ({args.config}):")
        print_status(cfg, Workspace.from_config(cfg))
        return
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
