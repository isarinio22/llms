"""Stage 1: load Banking77, stratified split (before any teacher labeling), cross-split dedup."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from llms.config import Config
from llms.embeddings import embed_texts, max_cosine
from llms.labels import load_labels
from llms.utils import log, resolve_device
from llms.workspace import Workspace

# Protected first: an item is dropped if it duplicates anything in a higher-priority split.
SPLIT_PRIORITY = ("test", "dev", "unlabeled", "train_pool")


def load_banking77(dataset_id: str) -> pd.DataFrame:
    """Official train + test pooled into one frame with stable ids."""
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(dataset_id, repo_type="dataset") if f.endswith(".parquet"))
    frames = []
    for orig in ("train", "test"):
        for f in (f for f in files if f"/{orig}-" in f):
            part = pd.read_parquet(hf_hub_download(dataset_id, f, repo_type="dataset"))
            part["orig_split"] = orig
            frames.append(part)
    df = pd.concat(frames, ignore_index=True).rename(columns={"label": "gold"})
    df["uid"] = [f"{s}-{i:05d}" for s, i in zip(df.orig_split, df.groupby("orig_split").cumcount(), strict=True)]
    return df[["uid", "text", "gold", "orig_split"]]


def subsample_per_class(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    shuffled = df.sample(frac=1.0, random_state=seed)
    return shuffled.groupby("gold").head(max_per_class).sort_index().reset_index(drop=True)


def stratified_split(labels: np.ndarray, fractions: dict[str, float], seed: int) -> np.ndarray:
    """Assign each item to a split, stratified by label.

    Within each class the items are shuffled and placed at evenly spaced positions in [0, 1) with a
    random offset (systematic sampling), then cut at the cumulative fractions. Every class gets its
    proportional share up to +-1 item, and it also works when a class has fewer items than splits.
    """
    names = list(fractions)
    cuts = np.cumsum([fractions[n] for n in names])
    if not np.isclose(cuts[-1], 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {cuts[-1]:.4f}")
    rng = np.random.default_rng(seed)
    out = np.empty(len(labels), dtype=object)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        pos = (np.arange(len(idx)) + rng.random()) / len(idx)
        out[idx] = np.asarray(names, dtype=object)[np.searchsorted(cuts[:-1], pos, side="right")]
    return out


def normalize_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return " ".join(text.split())


def remove_cross_split_duplicates(
    df: pd.DataFrame, emb: np.ndarray, threshold: float, priority: tuple[str, ...] = SPLIT_PRIORITY
) -> pd.DataFrame:
    """Mark items that duplicate an item in a higher-priority split (normalized text or cosine >= threshold).

    Adds columns: removed, removed_reason ('exact' | 'embedding'), dup_of (uid), dup_sim.
    """
    df = df.copy()
    norm = df.text.map(normalize_text).to_numpy()
    df["removed"], df["removed_reason"], df["dup_of"], df["dup_sim"] = False, None, None, np.nan
    ref_idx = np.flatnonzero(df.split.to_numpy() == priority[0])
    for split in priority[1:]:
        cand = np.flatnonzero(df.split.to_numpy() == split)
        ref_text = {t: i for i, t in zip(ref_idx, norm[ref_idx], strict=True)}
        sims, arg = max_cosine(emb[cand], emb[ref_idx])
        df.loc[df.index[cand], "dup_sim"] = sims
        kept = []
        for j, i in enumerate(cand):
            if norm[i] in ref_text:
                reason, match = "exact", ref_text[norm[i]]
            elif sims[j] >= threshold:
                reason, match = "embedding", ref_idx[arg[j]]
            else:
                kept.append(i)
                continue
            df.iloc[i, df.columns.get_loc("removed")] = True
            df.iloc[i, df.columns.get_loc("removed_reason")] = reason
            df.iloc[i, df.columns.get_loc("dup_of")] = df.uid.iloc[match]
        ref_idx = np.concatenate([ref_idx, np.asarray(kept, dtype=np.int64)])
    return df


def load_splits(ws: Workspace, include_removed: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(ws.artifacts / "data" / "splits.parquet")
    return df if include_removed else df[~df.removed].reset_index(drop=True)


def load_embeddings(ws: Workspace) -> tuple[pd.DataFrame, np.ndarray]:
    """All rows (including removed) aligned with the embedding matrix."""
    df = pd.read_parquet(ws.artifacts / "data" / "splits.parquet")
    return df, np.load(ws.artifacts / "data" / "embeddings.npy")


def run(cfg: Config, ws: Workspace, key: str) -> dict:
    labels = load_labels(cfg.labels_file)
    df = load_banking77(cfg.data.dataset_id)
    n_raw = len(df)
    if cfg.data.max_per_class:
        df = subsample_per_class(df, cfg.data.max_per_class, cfg.run.seed)
    df["label_name"] = [labels.names[g] for g in df.gold]
    df["split"] = stratified_split(df.gold.to_numpy(), cfg.data.fractions, cfg.run.seed)

    device = resolve_device(cfg.run.device)
    emb = embed_texts(cfg.data.embedding_model, df.text.tolist(), cfg.data.embedding_batch_size, device)
    df = remove_cross_split_duplicates(df, emb, cfg.data.dedup_threshold)

    out = ws.stage_dir("data")
    df.to_parquet(out / "splits.parquet", index=False)
    np.save(out / "embeddings.npy", emb)

    order = list(cfg.data.fractions)
    before = df.split.value_counts().reindex(order).fillna(0).astype(int)
    after = df[~df.removed].split.value_counts().reindex(order).fillna(0).astype(int)
    removed = df[df.removed].groupby(["split", "removed_reason"]).size().unstack(fill_value=0)
    sensitivity = {
        f"{t:.2f}": int((df.dup_sim >= t).sum()) for t in cfg.data.dedup_report_thresholds
    }
    per_class = df[~df.removed].groupby(["split", "gold"]).size().groupby("split").agg(["min", "max"])
    examples = []
    for _, r in df[df.removed_reason == "embedding"].head(5).iterrows():
        examples.append({"text": r.text, "duplicate_of": df.loc[df.uid == r.dup_of, "text"].iloc[0],
                         "split": r.split, "cosine": round(float(r.dup_sim), 4)})
    result = {
        "n_raw": n_raw,
        "n_used": len(df),
        "seed": cfg.run.seed,
        "fractions": cfg.data.fractions,
        "split_sizes_before_dedup": before.to_dict(),
        "split_sizes_after_dedup": after.to_dict(),
        "removed_by_split_and_reason": {s: removed.loc[s].to_dict() for s in removed.index},
        "removed_total": int(df.removed.sum()),
        "dedup_threshold": cfg.data.dedup_threshold,
        "embedding_model": cfg.data.embedding_model,
        "items_above_cosine_threshold": sensitivity,
        "per_class_count_range": {s: per_class.loc[s].to_dict() for s in per_class.index},
        "near_duplicate_examples": examples,
    }
    ws.save_result("data", result, key)
    log.info("Splits after dedup: %s (removed %d)", after.to_dict(), result["removed_total"])
    return result
