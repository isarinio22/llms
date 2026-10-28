"""Stage 3b: learning curve. Dev macro-F1 vs number of filtered teacher-labeled examples."""

from __future__ import annotations

import numpy as np

from llms.config import Config
from llms.student.train import train_lora, training_set
from llms.utils import log
from llms.workspace import Workspace


def run(cfg: Config, ws: Workspace, key: str) -> dict:
    train = training_set(ws)
    order = np.random.default_rng(cfg.run.seed).permutation(len(train))  # nested subsets
    points = []
    for n in sorted(cfg.student.curve_sizes):
        if n >= len(train):
            log.info("Skipping curve size %d (only %d filtered examples)", n, len(train))
            continue
        info = train_lora(cfg, ws, train.iloc[order[:n]], f"curve_{n}", cfg.student.curve_evals_per_epoch, merge=False)
        points.append({"n": n, "dev_macro_f1": info["best_dev_macro_f1"], "dev_accuracy": info["best_dev_accuracy"],
                       "train_runtime_s": info["train_runtime_s"]})
    main = ws.load_result("train_student")
    points.append({"n": main["n_train_examples"], "dev_macro_f1": main["best_dev_macro_f1"],
                   "dev_accuracy": main["best_dev_accuracy"], "train_runtime_s": main["train_runtime_s"], "all": True})
    result = {"points": points}
    ws.save_result("learning_curve", result, key)
    from llms.plots import learning_curve

    learning_curve(points, ws.figures / "learning_curve.png")
    return result
