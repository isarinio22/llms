"""Stage 6: confidence cascade. The student answers when confident, the teacher handles the rest.

Routing to the teacher happens when
  * a hard rule fires: the student's free generation is not a valid label, or the input length is an
    outlier relative to the train-pool, or
  * the expected saving beats the call cost: (p_big - p_small) * E > C_big, where p_small is the
    student's calibrated confidence and p_big is the teacher's dev accuracy in that confidence bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from llms.config import CascadeCfg, Config
from llms.data import load_splits
from llms.hf import load_tokenizer
from llms.scoring import load_scores, student_model_path
from llms.stats import paired_summary
from llms.teacher.label import load_teacher_labels
from llms.utils import log
from llms.workspace import Workspace


def frame(ws: Workspace, split: str, tok, lo: float, hi: float) -> pd.DataFrame:
    sc = load_scores(ws, split)
    lab = load_teacher_labels(ws).query("split == @split")[["uid", "text", "teacher_pred"]]
    df = sc.merge(lab, on="uid")
    df["n_tokens"] = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in df.text]
    df["invalid_output"] = df.generated_label < 0
    df["length_outlier"] = (df.n_tokens < lo) | (df.n_tokens > hi)
    df["hard_route"] = df.invalid_output | df.length_outlier
    df["student_ok"] = df.pred == df.gold
    df["teacher_ok"] = df.teacher_pred == df.gold
    return df


def coverage_curve(df: pd.DataFrame, n_points: int = 101) -> list[dict]:
    """Sweep a threshold on calibrated confidence alone (the hard rules are applied only when routing)."""
    rows = []
    for tau in np.unique(np.quantile(df.conf, np.linspace(0, 1, n_points))):
        covered = df.conf >= tau
        rows.append({
            "threshold": float(tau), "coverage": float(covered.mean()),
            "student_accuracy_covered": float(df.student_ok[covered].mean()) if covered.any() else float("nan"),
            "cascade_accuracy": float(np.where(covered, df.student_ok, df.teacher_ok).mean()),
        })
    return rows


def fit_bins(dev: pd.DataFrame, n_bins: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Equal-mass confidence bins on dev, with teacher (p_big) and student accuracy per bin."""
    inner = np.unique(np.quantile(dev.conf, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = np.concatenate([[-np.inf], inner, [np.inf]])
    b = np.digitize(dev.conf, inner)
    table = dev.assign(bin=b).groupby("bin").agg(
        n=("conf", "size"), conf_low=("conf", "min"), conf_high=("conf", "max"),
        p_small=("conf", "mean"), student_accuracy=("student_ok", "mean"), p_big=("teacher_ok", "mean"))
    return edges, table.reset_index()


def route(df: pd.DataFrame, edges: np.ndarray, bins: pd.DataFrame, cc: CascadeCfg) -> np.ndarray:
    p_big = bins.set_index("bin").p_big.reindex(range(len(edges) - 1)).ffill().bfill().to_numpy()
    item_p_big = p_big[np.digitize(df.conf, edges[1:-1])]
    cost_route = (item_p_big - df.conf.to_numpy()) * cc.error_cost > cc.teacher_call_cost
    return df.hard_route.to_numpy() | cost_route


def costs(n: int, errors: int, teacher_calls: int, cc: CascadeCfg, student_calls: int = 0) -> dict:
    total = errors * cc.error_cost + teacher_calls * cc.teacher_call_cost + student_calls * cc.student_call_cost
    return {"errors": errors, "teacher_calls": teacher_calls, "total_cost": total, "cost_per_1k_tickets": 1000 * total / n}


def run(cfg: Config, ws: Workspace, key: str) -> dict:
    cc = cfg.cascade
    tok = load_tokenizer(student_model_path(ws))
    pool = load_splits(ws).query("split == 'train_pool'")
    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in pool.text]
    lo, hi = np.quantile(lengths, cc.length_quantiles)
    dev, test = frame(ws, "dev", tok, lo, hi), frame(ws, "test", tok, lo, hi)

    edges, bins = fit_bins(dev, cc.n_conf_bins)
    dev_routed = route(dev, edges, bins, cc)
    routed = route(test, edges, bins, cc)
    cost_only = dev_routed & ~dev.hard_route.to_numpy()
    threshold = float(dev.conf[cost_only].max()) if cost_only.any() else None

    n = len(test)
    cascade_ok = np.where(routed, test.teacher_ok, test.student_ok)
    cascade_pred = np.where(routed, test.teacher_pred, test.pred)
    teacher_err, student_err = int((~test.teacher_ok).sum()), int((~test.student_ok).sum())
    result = {
        "costs_assumed": {"error_cost_E": cc.error_cost, "teacher_call_cost_C_big": cc.teacher_call_cost,
                          "student_call_cost": cc.student_call_cost},
        "length_bounds_tokens": [float(lo), float(hi)],
        "effective_threshold": threshold,
        "bins": bins.to_dict("records"),
        "test": {
            "n": n,
            "teacher_only_accuracy": float(test.teacher_ok.mean()),
            "student_only_accuracy": float(test.student_ok.mean()),
            "cascade_accuracy": float(cascade_ok.mean()),
            "share_sent_to_teacher": float(routed.mean()),
            "routed_by_invalid_output": int(test.invalid_output.sum()),
            "routed_by_length_outlier": int(test.length_outlier.sum()),
            "teacher_only_cost": costs(n, teacher_err, n, cc),
            "student_only_cost": costs(n, student_err, 0, cc, n),
            "cascade_cost": costs(n, int((~cascade_ok).sum()), int(routed.sum()), cc, n),
        },
        "cascade_vs_teacher": paired_summary(test.gold.to_numpy(), cascade_pred, test.teacher_pred.to_numpy()),
        "dev_curve": coverage_curve(dev),
        "test_curve": coverage_curve(test),
    }
    t = result["test"]
    t["saving_vs_teacher_only"] = 1 - t["cascade_cost"]["total_cost"] / max(t["teacher_only_cost"]["total_cost"], 1e-12)
    ws.save_result("cascade", result, key)
    ws.save_table("cascade_bins", bins)

    from llms.plots import coverage_accuracy

    dev_chosen = {"coverage": float(1 - dev_routed.mean()),
                  "cascade_accuracy": float(np.where(dev_routed, dev.teacher_ok, dev.student_ok).mean())}
    coverage_accuracy(result["dev_curve"], dev_chosen, float(dev.teacher_ok.mean()), ws.figures / "coverage_accuracy.png")
    log.info("Cascade test acc %.3f (teacher %.3f, student %.3f), %.1f%% to teacher, cost saving %.1f%%",
             t["cascade_accuracy"], t["teacher_only_accuracy"], t["student_only_accuracy"],
             100 * t["share_sent_to_teacher"], 100 * t["saving_vs_teacher_only"])
    return result
