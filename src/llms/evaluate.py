"""Stage 5: final evaluation on test. Teacher vs student vs baselines, paired statistics, planning."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import confusion_matrix

from llms.config import Config
from llms.labels import load_labels
from llms.metrics import most_confused_pairs, per_class_report, summary
from llms.scoring import load_scores
from llms.stats import (
    disagreement_only,
    non_inferiority,
    paired_counts,
    paired_summary,
    sample_size,
)
from llms.teacher.label import load_teacher_labels
from llms.utils import log
from llms.workspace import Workspace

MODELS = ("teacher", "student", "baseline_teacher_labels", "baseline_gold_labels")
MIN_DISCORDANT = 10


def predictions(ws: Workspace, split: str) -> pd.DataFrame:
    """One row per item: gold and every model's prediction (-1 = teacher had no valid output)."""
    lab = load_teacher_labels(ws).query("split == @split")[["uid", "gold", "teacher_pred"]]
    sc = load_scores(ws, split)[["uid", "pred", "conf"]].rename(columns={"pred": "student", "conf": "student_conf"})
    bl = pd.read_parquet(ws.artifacts / "baseline" / "predictions.parquet").query("split == @split")
    bl = bl.pivot(index="uid", columns="variant", values="pred").add_prefix("baseline_").reset_index()
    df = lab.rename(columns={"teacher_pred": "teacher"}).merge(sc, on="uid").merge(bl, on="uid")
    return df


def per_class_paired(df: pd.DataFrame, names: tuple[str, ...], top: int, margin: float, alpha: float) -> list[dict]:
    """Student vs teacher within the most frequent test classes; flags classes too uncertain to decide."""
    rows = []
    for cls in df.gold.value_counts().index[:top]:
        sub = df[df.gold == cls]
        ps = paired_summary(sub.gold.to_numpy(), sub.student.to_numpy(), sub.teacher.to_numpy())
        ni = non_inferiority(ps["b"], ps["c"], ps["n"], margin, alpha)
        if ps["b"] + ps["c"] < MIN_DISCORDANT:
            decision = "too few discordant pairs to decide"
        elif ni["verdict"] == "switch":
            decision = "non-inferior"
        elif ps["ci_high"] < -margin:
            decision = "inferior"
        else:
            decision = "interval too wide to decide"
        rows.append({"label": names[cls], "n": ps["n"], "b": ps["b"], "c": ps["c"], "acc_student": ps["acc_new"],
                     "acc_teacher": ps["acc_ref"], "delta_hat": ps["delta_hat"], "ci_low": ps["ci_low"],
                     "ci_high": ps["ci_high"], "ci_width": ps["ci_high"] - ps["ci_low"], "decision": decision})
    return rows


def sample_size_table(p_d: float, deltas_true: dict[str, float], margins: list[float], powers: list[float], alpha: float) -> list[dict]:
    rows = []
    for tag, d_true in deltas_true.items():
        for m in margins:
            for pw in powers:
                n = sample_size(p_d, d_true, m, alpha, pw)
                rows.append({"assumed_delta": tag, "delta_true": d_true, "margin": m, "power": pw,
                             "n_required": None if math.isinf(n) else math.ceil(n)})
    return rows


def run(cfg: Config, ws: Workspace, key: str) -> dict:
    names = load_labels(cfg.labels_file).names
    ev = cfg.evaluation
    test, dev, unl = (predictions(ws, s) for s in ("test", "dev", "unlabeled"))
    gold = test.gold.to_numpy()

    models = {}
    for m in MODELS:
        pred = test[m].to_numpy()
        models[m] = {"test": summary(gold, pred, len(names)), "dev": summary(dev.gold.to_numpy(), dev[m].to_numpy(), len(names)),
                     "most_confused": most_confused_pairs(gold, pred, names, 10).to_dict("records")}
        ws.save_table(f"per_class_{m}", per_class_report(gold, pred, names))
        cm = confusion_matrix(gold, pred, labels=list(range(len(names))))
        ws.save_table(f"confusion_{m}", pd.DataFrame(cm, index=names, columns=names).reset_index(names="gold"))

    main = paired_summary(gold, test.student.to_numpy(), test.teacher.to_numpy())
    ni = [non_inferiority(main["b"], main["c"], main["n"], m, ev.alpha) for m in ev.deltas]

    p_d = float((unl.student != unl.teacher).mean())
    dev_pc = paired_counts(dev.gold.to_numpy(), dev.student.to_numpy(), dev.teacher.to_numpy())
    delta_dev = (dev_pc.b - dev_pc.c) / dev_pc.n
    planner = sample_size_table(p_d, {"0 (equal accuracy)": 0.0, "dev estimate": delta_dev}, ev.deltas, ev.powers, ev.alpha)

    result = {
        "n_test": len(test),
        "models": models,
        "student_vs_teacher": {**main, "non_inferiority": ni},
        "per_class_student_vs_teacher": per_class_paired(test, names, ev.top_classes, ev.primary_delta, ev.alpha),
        "other_comparisons": {
            "student_vs_baseline_teacher_labels": paired_summary(gold, test.student.to_numpy(), test.baseline_teacher_labels.to_numpy()),
            "student_vs_baseline_gold_labels": paired_summary(gold, test.student.to_numpy(), test.baseline_gold_labels.to_numpy()),
            "baseline_gold_labels_vs_teacher": paired_summary(gold, test.baseline_gold_labels.to_numpy(), test.teacher.to_numpy()),
        },
        "sample_size_planner": {"p_d_unlabeled_disagreement": p_d, "n_unlabeled": len(unl),
                                "delta_dev_estimate": delta_dev, "alpha": ev.alpha, "table": planner},
        "label_only_disagreements": disagreement_only(gold, test.student.to_numpy(), test.teacher.to_numpy()),
    }
    ws.save_table("sample_size_planner", pd.DataFrame(planner))
    ws.save_result("evaluate", result, key)

    from llms.plots import non_inferiority_plot

    non_inferiority_plot(main, ni, ws.figures / "non_inferiority.png")
    log.info("Test acc: teacher %.3f, student %.3f | Delta %.4f [%.4f, %.4f] | verdicts %s",
             models["teacher"]["test"]["accuracy"], models["student"]["test"]["accuracy"], main["delta_hat"],
             main["ci_low"], main["ci_high"], {r["margin"]: r["verdict"] for r in ni})
    return result
