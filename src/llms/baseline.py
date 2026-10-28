"""Baseline: sentence embeddings + logistic regression. Is an LLM needed at all?

Trained twice: on teacher labels (the fair comparison with the student, same training signal) and on
gold labels (a reference for what human labeling would buy). C is chosen on dev macro-F1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from llms.config import Config
from llms.data import load_embeddings
from llms.metrics import macro_f1, summary
from llms.teacher.label import load_teacher_labels
from llms.utils import log
from llms.workspace import Workspace

C_GRID = (0.01, 0.1, 1.0)
EVAL_SPLITS = ("dev", "test", "unlabeled")


def fit_predict(X: np.ndarray, y: np.ndarray, evals: dict[str, np.ndarray], C: float, n_classes: int, seed: int) -> dict[str, np.ndarray]:
    """Fit and return full (n, n_classes) probability matrices (classes absent from y get 0)."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed))
    model.fit(X, y)
    out = {}
    for name, Xe in evals.items():
        probs = np.zeros((len(Xe), n_classes), dtype=np.float32)
        probs[:, model.classes_] = model.predict_proba(Xe)
        out[name] = probs
    return out


def run(cfg: Config, ws: Workspace, key: str) -> dict:
    all_rows, emb = load_embeddings(ws)
    row = {u: i for i, u in enumerate(all_rows.uid)}
    lab = load_teacher_labels(ws)
    n_classes = int(all_rows.gold.max()) + 1
    X = lambda d: emb[[row[u] for u in d.uid]]  # noqa: E731
    splits = {s: lab[lab.split == s] for s in EVAL_SPLITS}
    evals = {s: X(d) for s, d in splits.items()}
    pool = lab[lab.split == "train_pool"]
    variants = {
        "teacher_labels": (pool[pool.kept], "teacher_pred"),
        "gold_labels": (pool, "gold"),
    }
    preds = []
    result: dict = {"embedding_model": cfg.data.embedding_model, "variants": {}}
    for name, (train, col) in variants.items():
        train = train[train[col] >= 0]
        scores = {}
        for C in C_GRID:
            probs = fit_predict(X(train), train[col].to_numpy(), {"dev": evals["dev"]}, C, n_classes, cfg.run.seed)
            scores[C] = macro_f1(splits["dev"].gold.to_numpy(), probs["dev"].argmax(1), n_classes)
        best_c = max(scores, key=scores.get)
        probs = fit_predict(X(train), train[col].to_numpy(), evals, best_c, n_classes, cfg.run.seed)
        result["variants"][name] = {
            "n_train": len(train), "C": best_c, "dev_macro_f1_by_C": {str(c): v for c, v in scores.items()},
            **{s: summary(splits[s].gold.to_numpy(), probs[s].argmax(1), n_classes) for s in ("dev", "test")},
        }
        for s in EVAL_SPLITS:
            np.save(ws.stage_dir("baseline") / f"{name}_{s}_probs.npy", probs[s])
            preds.append(pd.DataFrame({"uid": splits[s].uid, "split": s, "variant": name,
                                       "pred": probs[s].argmax(1), "conf": probs[s].max(1)}))
        log.info("Baseline (%s): C=%s test acc=%.3f", name, best_c, result["variants"][name]["test"]["accuracy"])
    pd.concat(preds).to_parquet(ws.stage_dir("baseline") / "predictions.parquet", index=False)
    ws.save_result("baseline", result, key)
    return result
