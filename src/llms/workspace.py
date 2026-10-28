"""Where a run reads and writes: artifacts (large, gitignored), results (json/csv), figures."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from llms.config import Config


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(path: Path, payload: Any) -> None:
    """Atomic JSON write (temp file + rename) so a disconnect never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, cls=_Encoder, allow_nan=True))
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows and fsync, so cached teacher calls survive a crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, cls=_Encoder) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # torn final line from an interrupted write
    return rows


@dataclass
class Workspace:
    """Paths for one run. On Colab the repo lives on Google Drive, so all of this persists."""

    cfg: Config
    base: Path

    @classmethod
    def from_config(cls, cfg: Config, base: str | Path | None = None) -> Workspace:
        return cls(cfg, Path(base or os.environ.get("LLMS_HOME", Path.cwd())).resolve())

    @property
    def artifacts(self) -> Path:
        return self._dir(self.base / "artifacts" / self.cfg.run.name)

    @property
    def results(self) -> Path:
        return self._dir(self.base / "results" / self.cfg.run.name)

    @property
    def figures(self) -> Path:
        return self._dir(self.base / "figures" / self.cfg.run.name)

    def stage_dir(self, stage: str) -> Path:
        return self._dir(self.artifacts / stage)

    @staticmethod
    def _dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Stage completion markers -------------------------------------------------
    def is_done(self, stage: str, key: str) -> bool:
        marker = self.artifacts / stage / "_DONE.json"
        return marker.exists() and read_json(marker).get("key") == key

    def mark_done(self, stage: str, key: str, **meta: Any) -> None:
        write_json(self.stage_dir(stage) / "_DONE.json", {"key": key, "finished": utc_stamp(), **meta})

    def clear_done(self, stage: str) -> None:
        (self.artifacts / stage / "_DONE.json").unlink(missing_ok=True)

    # Results ------------------------------------------------------------------
    def save_result(self, name: str, payload: dict[str, Any], key: str = "") -> Path:
        """Write results/<run>/<name>.json (latest) plus a timestamped copy in history/."""
        stamp = utc_stamp()
        doc = {"name": name, "run": self.cfg.run.name, "timestamp": stamp, "stage_key": key, **payload}
        latest = self.results / f"{name}.json"
        write_json(latest, doc)
        write_json(self.results / "history" / f"{stamp}_{name}.json", doc)
        return latest

    def save_table(self, name: str, df: pd.DataFrame) -> Path:
        path = self.results / f"{name}.csv"
        df.to_csv(path, index=False)
        return path

    def load_result(self, name: str) -> dict[str, Any]:
        return read_json(self.results / f"{name}.json")

    def snapshot_config(self, stage: str) -> None:
        path = self.results / "configs" / f"{utc_stamp()}_{stage}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.cfg.to_dict(), sort_keys=False))

    def copy_into(self, src: Path, dst_dir: Path) -> Path:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return dst
