"""Expected calibration error, reliability bins and temperature scaling."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax


def reliability_bins(probs: np.ndarray, gold: np.ndarray, n_bins: int = 15) -> list[dict[str, float]]:
    """Equal-width confidence bins: count, mean confidence and accuracy per non-empty bin."""
    conf = probs.max(1)
    correct = probs.argmax(1) == gold
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": int(m.sum()),
                         "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return bins


def ece(probs: np.ndarray, gold: np.ndarray, n_bins: int = 15) -> float:
    """ECE = sum_b (n_b / n) * |accuracy_b - confidence_b| over equal-width bins."""
    n = len(gold)
    return float(sum(b["count"] / n * abs(b["accuracy"] - b["confidence"]) for b in reliability_bins(probs, gold, n_bins)))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return np.exp(log_softmax(logits / temperature, axis=1))


def fit_temperature(logits: np.ndarray, gold: np.ndarray) -> float:
    """Single temperature T minimizing the NLL of softmax(logits / T) on held-out (dev) data."""
    def nll(log_t: float) -> float:
        lp = log_softmax(logits / np.exp(log_t), axis=1)
        return float(-lp[np.arange(len(gold)), gold].mean())

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))
