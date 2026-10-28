"""Classification metrics shared by all stages. Predictions of -1 mean 'no valid output' (counted wrong)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(gold) == np.asarray(pred))) if len(gold) else float("nan")


def macro_f1(gold: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    if not len(gold):
        return float("nan")
    return float(f1_score(gold, pred, labels=list(range(n_classes)), average="macro", zero_division=0))


def summary(gold: np.ndarray, pred: np.ndarray, n_classes: int) -> dict[str, float]:
    return {"n": int(len(gold)), "accuracy": accuracy(gold, pred), "macro_f1": macro_f1(gold, pred, n_classes)}


def per_class_report(gold: np.ndarray, pred: np.ndarray, names: tuple[str, ...]) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        gold, pred, labels=list(range(len(names))), zero_division=0
    )
    return pd.DataFrame({"label": names, "precision": p, "recall": r, "f1": f, "support": s})


def most_confused_pairs(gold: np.ndarray, pred: np.ndarray, names: tuple[str, ...], top: int = 10) -> pd.DataFrame:
    """Off-diagonal confusion counts (gold -> predicted), largest first."""
    cm = confusion_matrix(gold, pred, labels=list(range(len(names))))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:top]
    rows = [(names[i], names[j], int(cm[i, j])) for i, j in zip(*np.unravel_index(flat, cm.shape), strict=True) if cm[i, j] > 0]
    return pd.DataFrame(rows, columns=["gold", "predicted", "count"])
