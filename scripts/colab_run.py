"""Run the pipeline on a Colab T4 from this terminal (Google's Colab CLI) and mirror its state locally.

  python scripts/colab_run.py --config configs/base.yaml

1. Gets a GPU session (retrying while the free tier has none), 2. uploads the local state bundle if one
exists, so a new VM resumes where the last one stopped, 3. runs scripts/colab_job.py in the VM kernel,
4. every minute downloads the state bundle written after each stage and unpacks results/, figures/ and the
small artifacts into this repo. Prints one line per stage start/end so a watcher can follow along.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llms.colab import restore_state  # noqa: E402
from llms.config import load_config  # noqa: E402

COLAB = str(Path.home() / ".local/bin/colab")
EVENTS = ("=== Stage", "Error", "Traceback", "out of memory", "ALL STAGES DONE", "failed")


def colab(*args: str, timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run([COLAB, *args], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def alive(session: str) -> bool:
    return any(line.startswith(f"[{session}]") for line in colab("sessions").stdout.splitlines())


def get_gpu(session: str, gpu: str, wait_min: int) -> bool:
    for attempt in range(wait_min):
        if alive(session):
            return True
        colab("new", "-s", session, "--gpu", gpu)
        if alive(session):
            return True
        if attempt == 0:
            say("no GPU yet (free-tier limit or capacity); retrying every minute")
        time.sleep(60)
    return alive(session)


class Mirror:
    """Downloads the VM's state bundle whenever its version changes and unpacks it into the repo."""

    def __init__(self, session: str, run: str, bundle: Path):
        self.session, self.run, self.bundle = session, run, bundle
        self.version, self.seen_lines = "", 0
        self.tmp = bundle.with_name(bundle.name + ".download")

    def sync(self) -> None:
        ver_file = self.bundle.with_name("state.version")
        if colab("download", "-s", self.session, "/content/state.version", str(ver_file)).returncode != 0:
            return
        version = ver_file.read_text().strip()
        if version == self.version:
            return
        if colab("download", "-s", self.session, "/content/state.tar.gz", str(self.tmp), timeout=900).returncode != 0:
            return
        shutil.move(self.tmp, self.bundle)
        restore_state(self.bundle, ROOT, self.run)
        self.version = version
        self._report()

    def _report(self) -> None:
        log = ROOT / "logs" / "run.log"
        if not log.exists():
            return
        lines = log.read_text(errors="replace").replace("\r", "\n").splitlines()
        for line in lines[self.seen_lines:]:
            if any(e in line for e in EVENTS):
                say(f"VM | {line.strip()[:220]}")
        self.seen_lines = len(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--session", default="llms")
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--skip", default="", help="comma-separated stages to skip")
    ap.add_argument("--wait-minutes", type=int, default=60)
    args = ap.parse_args()

    run = load_config(ROOT / args.config).run.name
    bundle = ROOT / "artifacts" / "colab_state" / run / "state.tar.gz"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if not get_gpu(args.session, args.gpu, args.wait_minutes):
        say("NO GPU SESSION: try again later")
        return 1
    say(f"GPU session '{args.session}' ready")
    if bundle.exists():
        colab("upload", "-s", args.session, str(bundle), "/content/state_in.tar.gz", timeout=900)
        say(f"uploaded local state bundle ({bundle.stat().st_size / 2**20:.1f} MB) to resume from")

    exec_log = ROOT / "logs" / "colab_exec.log"
    exec_log.parent.mkdir(exist_ok=True)
    job = subprocess.Popen(
        [COLAB, "exec", "-s", args.session, "--timeout", "43200", "--env", f"LLMS_CONFIG={args.config}",
         "--env", f"LLMS_SKIP={args.skip}", "-f", str(ROOT / "scripts" / "colab_job.py")],
        stdin=subprocess.DEVNULL, stdout=exec_log.open("w"), stderr=subprocess.STDOUT)
    mirror = Mirror(args.session, run, bundle)
    while job.poll() is None:
        time.sleep(60)
        if not alive(args.session):
            say("SESSION LOST: rerun this script to resume from the last mirrored stage")
            job.kill()
            return 1
        mirror.sync()
        if "done" in mirror.version:
            break
    mirror.sync()
    done = "done" in mirror.version
    say("ALL STAGES DONE: results are in results/ and figures/" if done else "JOB ENDED WITH AN ERROR (see VM lines above)")
    colab("stop", "-s", args.session)
    job.kill()
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
