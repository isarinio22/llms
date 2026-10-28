"""Paired comparison of two classifiers on the same items.

Convention: "new" is the candidate (student), "ref" is the incumbent (teacher).
  a = both right, b = only new right, c = only ref right, d = both wrong.
  Delta = acc(new) - acc(ref), estimated by (b - c) / n.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import binomtest, chi2, norm


@dataclass(frozen=True)
class PairedCounts:
    a: int
    b: int
    c: int
    d: int

    @property
    def n(self) -> int:
        return self.a + self.b + self.c + self.d


def paired_counts(gold: np.ndarray, pred_new: np.ndarray, pred_ref: np.ndarray) -> PairedCounts:
    new_ok, ref_ok = np.asarray(pred_new) == gold, np.asarray(pred_ref) == gold
    return PairedCounts(a=int((new_ok & ref_ok).sum()), b=int((new_ok & ~ref_ok).sum()),
                        c=int((~new_ok & ref_ok).sum()), d=int((~new_ok & ~ref_ok).sum()))


def delta_hat(b: int, c: int, n: int) -> float:
    return (b - c) / n


def paired_se(b: int, c: int, n: int) -> float:
    """SE of (b - c)/n: sqrt((b + c)/n^2 - (b - c)^2/n^3)."""
    return math.sqrt(max((b + c) / n**2 - (b - c) ** 2 / n**3, 0.0))


def delta_ci(b: int, c: int, n: int, level: float = 0.95) -> tuple[float, float]:
    z = norm.ppf(0.5 + level / 2)
    d, se = delta_hat(b, c, n), paired_se(b, c, n)
    return d - z * se, d + z * se


def mcnemar(b: int, c: int, exact_below: int = 25) -> dict:
    """McNemar chi^2 = (b - c)^2 / (b + c) with 1 df; exact binomial test when b + c < exact_below."""
    m = b + c
    if m == 0:
        return {"chi2": 0.0, "p_value": 1.0, "method": "no discordant pairs"}
    stat = (b - c) ** 2 / m
    if m < exact_below:
        return {"chi2": stat, "p_value": float(binomtest(b, m, 0.5).pvalue), "method": "exact binomial"}
    return {"chi2": stat, "p_value": float(chi2.sf(stat, 1)), "method": "chi-square"}


def non_inferiority(b: int, c: int, n: int, margin: float, alpha: float = 0.05) -> dict:
    """One-sided test of H0: Delta <= -margin. Z = (Delta_hat + margin) / SE vs z_(1-alpha).

    Rejecting H0 means the new model is at most `margin` worse than the reference: switch.
    """
    d, se = delta_hat(b, c, n), paired_se(b, c, n)
    z_crit = float(norm.ppf(1 - alpha))
    z = (d + margin) / se if se > 0 else (math.inf if d + margin > 0 else -math.inf)
    return {"margin": margin, "z": z, "z_crit": z_crit, "p_value": float(norm.sf(z)),
            "lower_bound": d - z_crit * se, "verdict": "switch" if z > z_crit else "don't switch"}


def sample_size(p_d: float, delta_true: float, margin: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Items needed for a one-sided non-inferiority test:
    n = (z_(1-alpha) + z_(1-beta))^2 * (p_d - Delta^2) / (Delta + margin)^2.

    p_d is the discordance rate; the prediction-disagreement rate is an upper bound for it.
    """
    if delta_true + margin <= 0:
        return math.inf
    z = norm.ppf(1 - alpha) + norm.ppf(power)
    return z**2 * (p_d - delta_true**2) / (delta_true + margin) ** 2


def paired_summary(gold: np.ndarray, pred_new: np.ndarray, pred_ref: np.ndarray, level: float = 0.95) -> dict:
    pc = paired_counts(gold, pred_new, pred_ref)
    lo, hi = delta_ci(pc.b, pc.c, pc.n, level)
    return {**asdict(pc), "n": pc.n, "acc_new": (pc.a + pc.b) / pc.n, "acc_ref": (pc.a + pc.c) / pc.n,
            "delta_hat": delta_hat(pc.b, pc.c, pc.n), "se": paired_se(pc.b, pc.c, pc.n),
            "ci_low": lo, "ci_high": hi, "ci_level": level, "mcnemar": mcnemar(pc.b, pc.c)}


def disagreement_only(gold: np.ndarray, pred_new: np.ndarray, pred_ref: np.ndarray) -> dict:
    """Delta_hat using gold labels ONLY where the two predictions differ.

    Where predictions agree, both are right or both wrong (a or d): they never enter b - c.
    So labeling the disagreements gives exactly the full-label Delta_hat (and SE).
    """
    pred_new, pred_ref = np.asarray(pred_new), np.asarray(pred_ref)
    n = len(gold)
    dis = pred_new != pred_ref
    b = int((pred_new[dis] == gold[dis]).sum())
    c = int((pred_ref[dis] == gold[dis]).sum())
    full = paired_counts(gold, pred_new, pred_ref)
    return {"n": n, "labels_needed": int(dis.sum()), "labeling_saved": 1 - dis.mean(),
            "delta_disagreements_only": delta_hat(b, c, n), "delta_full_labels": delta_hat(full.b, full.c, n),
            "se_disagreements_only": paired_se(b, c, n), "identical": b == full.b and c == full.c}
