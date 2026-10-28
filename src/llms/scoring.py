"""Stage 4: score all 77 labels for every input, calibrate with temperature scaling.

Score of a label = sum of its token log-probs (plus the end-of-turn token) given the shared student
prompt. Softmax across the 77 scores gives a class distribution: prediction = argmax,
confidence = max probability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from llms.calibration import apply_temperature, ece, fit_temperature, reliability_bins
from llms.config import Config
from llms.data import load_splits
from llms.hf import free_memory, generate, load_model, load_tokenizer, score_candidates
from llms.labels import LabelSet, load_labels
from llms.metrics import summary
from llms.prompts import encode_prompt, encode_target
from llms.utils import log, resolve_device, timed
from llms.workspace import Workspace

SCORED_SPLITS = ("dev", "test", "unlabeled")


def score_texts(model: Any, tok: Any, texts: list[str], labels: LabelSet, batch_size: int) -> np.ndarray:
    """(n_texts, n_labels) summed label log-probs, using the shared prompt builder."""
    prompts = [encode_prompt(tok, t) for t in texts]
    cands = [encode_target(tok, name) for name in labels.names]
    return score_candidates(model, prompts, cands, tok.pad_token_id, batch_size)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def greedy_labels(model: Any, tok: Any, texts: list[str], labels: LabelSet, max_new_tokens: int, batch_size: int) -> tuple[list[str], list[int]]:
    """Free greedy generation (what a deployed student would emit) and its validated label id (-1 = invalid)."""
    raw = generate(model, tok, [encode_prompt(tok, t) for t in texts], max_new_tokens, batch_size)
    return raw, [lab if (lab := labels.parse(r)) is not None else -1 for r in raw]


def student_model_path(ws: Workspace) -> str:
    return str(ws.artifacts / "train_student" / "main" / "merged")


def load_scores(ws: Workspace, split: str) -> pd.DataFrame:
    return pd.read_parquet(ws.artifacts / "score" / f"{split}.parquet")


def run(cfg: Config, ws: Workspace, key: str) -> dict:
    labels = load_labels(cfg.labels_file)
    df = load_splits(ws)
    device = resolve_device(cfg.run.device)
    path = student_model_path(ws)
    tok, model = load_tokenizer(path), load_model(path, device)
    timings: dict[str, float] = {}
    logits: dict[str, np.ndarray] = {}
    gens: dict[str, tuple[list[str], list[int]]] = {}
    for split in SCORED_SPLITS:
        d = df[df.split == split]
        with timed(f"score_{split}", timings):
            logits[split] = score_texts(model, tok, d.text.tolist(), labels, cfg.scoring.batch_size)
        if split in ("dev", "test"):
            with timed(f"generate_{split}", timings):
                gens[split] = greedy_labels(model, tok, d.text.tolist(), labels, cfg.scoring.gen_max_new_tokens,
                                            max(cfg.scoring.batch_size * 4, 8))
    del model
    free_memory()

    dev_gold = df[df.split == "dev"].gold.to_numpy()
    temperature = fit_temperature(logits["dev"], dev_gold)
    n_bins = cfg.calibration.n_bins
    calib: dict = {"temperature": temperature, "n_bins": n_bins}
    for split in ("dev", "test"):
        gold = df[df.split == split].gold.to_numpy()
        for tag, probs in (("before", softmax(logits[split])), ("after", apply_temperature(logits[split], temperature))):
            calib[f"{split}_ece_{tag}"] = ece(probs, gold, n_bins)
            calib[f"{split}_reliability_{tag}"] = reliability_bins(probs, gold, n_bins)

    out = ws.stage_dir("score")
    metrics: dict = {}
    for split in SCORED_SPLITS:
        d = df[df.split == split]
        probs = apply_temperature(logits[split], temperature)
        frame = pd.DataFrame({"uid": d.uid.to_numpy(), "gold": d.gold.to_numpy(), "pred": probs.argmax(1),
                              "conf": probs.max(1), "conf_uncalibrated": softmax(logits[split]).max(1)})
        if split in gens:
            frame["generated"], frame["generated_label"] = gens[split]
            metrics[f"{split}_generation_invalid_rate"] = float((frame.generated_label < 0).mean())
            metrics[f"{split}_generation_agrees_with_scoring"] = float((frame.generated_label == frame.pred).mean())
        frame.to_parquet(out / f"{split}.parquet", index=False)
        np.save(out / f"{split}_logits.npy", logits[split])
        if split != "unlabeled":
            metrics[split] = summary(frame.gold.to_numpy(), frame.pred.to_numpy(), len(labels))
    result = {"model": path, "calibration": calib, "metrics": metrics, "timings_s": timings}
    ws.save_result("score", result, key)
    from llms.plots import reliability

    reliability(calib["test_reliability_before"], calib["test_reliability_after"], calib["test_ece_before"],
                calib["test_ece_after"], ws.figures / "reliability.png")
    log.info("Student dev acc=%.3f | ECE test %.3f -> %.3f (T=%.2f)", metrics["dev"]["accuracy"],
             calib["test_ece_before"], calib["test_ece_after"], temperature)
    return result
