"""Stage 2: teacher labeling with self-consistency, validation/retry and a never-recall cache."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from llms import prompts
from llms.config import Config
from llms.data import load_splits
from llms.labels import LabelSet, load_labels
from llms.metrics import summary
from llms.teacher.backends import make_backend
from llms.utils import log
from llms.workspace import Workspace, append_jsonl, read_jsonl

_RATIONALE = re.compile(r"rationale\s*:\s*(.+)", re.IGNORECASE)


def cache_path(cfg: Config, ws: Workspace) -> Path:
    """Cache file keyed by everything that changes what the teacher would say (not by filter settings).

    Shared across runs: an ablation that only changes the student reuses the teacher's answers.
    """
    t = cfg.teacher
    payload = {
        "backend": t.backend, "model": t.api.model if t.backend == "api" else t.model,
        "quantization": t.quantization, "temperature": t.temperature, "top_p": t.top_p,
        "max_new_tokens": t.rationale_max_new_tokens if t.rationale else t.max_new_tokens,
        "rationale": t.rationale, "constrained": t.constrained, "seed": cfg.run.seed,
        "prompt": prompts.TEACHER_SYSTEM_RATIONALE if t.rationale else prompts.TEACHER_SYSTEM,
        "labels": Path(cfg.labels_file).read_text(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    path = ws.base / "artifacts" / "_shared" / "teacher_cache" / f"cache_{digest}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def request_seed(base: int, uid: str, rnd: int) -> int:
    return int(hashlib.sha256(f"{base}-{uid}-{rnd}".encode()).hexdigest()[:8], 16)


def parse_rationale(raw: str) -> str | None:
    m = _RATIONALE.search(raw or "")
    return m.group(1).strip() if m else None


@dataclass
class ItemState:
    rounds: set[int] = field(default_factory=set)
    valid: list[tuple[int, str | None]] = field(default_factory=list)  # (label id, rationale)
    first_round_invalid: int = 0
    samples: int = 0


def replay(rows: list[dict], labels: LabelSet) -> dict[str, ItemState]:
    state: dict[str, ItemState] = {}
    for row in rows:
        st = state.setdefault(row["uid"], ItemState())
        st.rounds.add(row["round"])
        for raw in row["outputs"]:
            st.samples += 1
            lab = labels.parse(raw)
            if lab is None:
                st.first_round_invalid += row["round"] == 0
            else:
                st.valid.append((lab, parse_rationale(raw)))
    return state


def plurality(valid: list[int], k: int) -> tuple[int, int]:
    """Most frequent label among the first k valid samples (ties -> earliest). (-1, 0) if none valid."""
    votes = valid[:k]
    if not votes:
        return -1, 0
    counts = Counter(votes)
    top = max(counts.values())
    return next(v for v in votes if counts[v] == top), top


def label_items(cfg: Config, ws: Workspace, df: pd.DataFrame, labels: LabelSet) -> tuple[dict[str, ItemState], dict]:
    """Fill the cache until every item has k valid samples or max_retries extra rounds were spent."""
    t = cfg.teacher
    path = cache_path(cfg, ws)
    state = replay(read_jsonl(path), labels)
    texts = dict(zip(df.uid, df.text, strict=True))
    backend = None
    stats = {"cache_file": path.name, "cached_items_at_start": len(state), "requests": 0, "samples": 0, "seconds": 0.0}
    for rnd in range(1 + t.max_retries):
        todo: dict[int, list[str]] = {}
        for uid in df.uid:
            st = state.get(uid, ItemState())
            missing = t.k if rnd == 0 else t.k - len(st.valid)
            if rnd not in st.rounds and missing > 0 and (rnd == 0 or 0 in st.rounds):
                todo.setdefault(missing, []).append(uid)
        n_todo = sum(len(v) for v in todo.values())
        if not n_todo:
            continue
        log.info("Teacher round %d: %d items need samples", rnd, n_todo)
        if backend is None:
            backend = make_backend(t, labels, cfg.run.device, cfg.run.seed)
        for n, uids in todo.items():
            for i in range(0, len(uids), t.chunk_size):
                chunk = uids[i : i + t.chunk_size]
                start = time.perf_counter()
                outs = backend.generate([texts[u] for u in chunk], n, [request_seed(cfg.run.seed, u, rnd) for u in chunk])
                stats["seconds"] += time.perf_counter() - start
                rows = [{"uid": u, "round": rnd, "outputs": o} for u, o in zip(chunk, outs, strict=True)]
                append_jsonl(path, rows)
                for row in rows:
                    new = replay([row], labels)[row["uid"]]
                    st = state.setdefault(row["uid"], ItemState())
                    st.rounds |= new.rounds
                    st.valid += new.valid
                    st.first_round_invalid += new.first_round_invalid
                    st.samples += new.samples
                stats["requests"] += len(chunk)
                stats["samples"] += len(chunk) * n
                log.info("  cached %d/%d items of round %d (n=%d)", i + len(chunk), len(uids), rnd, n)
    stats["seconds"] = round(stats["seconds"], 1)
    return state, stats


def aggregate(df: pd.DataFrame, state: dict[str, ItemState], k: int, min_agree: int) -> pd.DataFrame:
    rows = []
    for uid in df.uid:
        st = state[uid]
        pred, agree = plurality([v for v, _ in st.valid], k)
        rationale = next((r for v, r in st.valid[:k] if v == pred and r), None)
        rows.append({"uid": uid, "teacher_pred": pred, "agree": agree, "n_valid": min(len(st.valid), k),
                     "kept": agree >= min_agree, "rationale": rationale,
                     "first_round_invalid": st.first_round_invalid, "samples": st.samples})
    return df[["uid", "text", "gold", "split"]].merge(pd.DataFrame(rows), on="uid")


def teacher_report(lab: pd.DataFrame, n_classes: int, k: int) -> dict:
    out: dict = {"per_split": {}}
    for split, g in lab.groupby("split"):
        out["per_split"][split] = {
            **summary(g.gold.to_numpy(), g.teacher_pred.to_numpy(), n_classes),
            "agree_distribution": {f"{a}/{k}": int((g.agree == a).sum()) for a in range(k, -1, -1)},
            "no_valid_output": int((g.n_valid == 0).sum()),
        }
    pool = lab[lab.split == "train_pool"]
    kept, removed = pool[pool.kept], pool[~pool.kept]
    acc = lambda g: float((g.gold == g.teacher_pred).mean()) if len(g) else float("nan")  # noqa: E731
    out["train_pool"] = {
        "n": len(pool),
        "label_noise": 1 - acc(pool),
        "kept": len(kept),
        "removed_by_filter": len(removed),
        "kept_rate": len(kept) / max(len(pool), 1),
        "teacher_accuracy_kept": acc(kept),
        "teacher_accuracy_removed": acc(removed),
        "label_noise_after_filter": 1 - acc(kept),
    }
    out["first_round_invalid_rate"] = float(lab.first_round_invalid.sum() / max(len(lab) * k, 1))
    out["final_invalid_slot_rate"] = float((k - lab.n_valid).sum() / max(len(lab) * k, 1))
    return out


def load_teacher_labels(ws: Workspace) -> pd.DataFrame:
    return pd.read_parquet(ws.artifacts / "teacher_label" / "labels.parquet")


def run(cfg: Config, ws: Workspace, key: str) -> dict:
    labels = load_labels(cfg.labels_file)
    df = load_splits(ws)
    state, stats = label_items(cfg, ws, df, labels)
    lab = aggregate(df, state, cfg.teacher.k, cfg.teacher.min_agree)
    lab.to_parquet(ws.stage_dir("teacher_label") / "labels.parquet", index=False)
    result = {
        "teacher": cfg.teacher.api.model if cfg.teacher.backend == "api" else cfg.teacher.model,
        "backend": cfg.teacher.backend, "k": cfg.teacher.k, "min_agree": cfg.teacher.min_agree,
        "temperature": cfg.teacher.temperature, "rationale": cfg.teacher.rationale,
        "constrained": cfg.teacher.constrained,
        "calls": stats, **teacher_report(lab, len(labels), cfg.teacher.k),
    }
    if stats["samples"]:
        result["calls"]["samples_per_second"] = round(stats["samples"] / max(stats["seconds"], 1e-9), 2)
    ws.save_result("teacher", result, key)
    tp = result["train_pool"]
    log.info("Teacher acc dev=%.3f test=%.3f | train-pool noise=%.3f, kept %d/%d",
             result["per_split"].get("dev", {}).get("accuracy", np.nan),
             result["per_split"].get("test", {}).get("accuracy", np.nan), tp["label_noise"], tp["kept"], tp["n"])
    return result
