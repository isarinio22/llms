"""Stage 7b: is quantization safe? Paired non-inferiority of the quantized student vs fp16 on test."""

from __future__ import annotations

import io
import time

import numpy as np
import torch

from llms.config import Config
from llms.data import load_splits
from llms.hf import free_memory, load_model, load_tokenizer
from llms.labels import load_labels
from llms.scoring import load_scores, score_texts, student_model_path
from llms.stats import non_inferiority, paired_summary
from llms.utils import log, resolve_device
from llms.workspace import Workspace

GPU_MODES, CPU_MODES = {"bnb4", "bnb8"}, {"int8_dynamic"}


def _footprint_mb(model) -> float:
    """Serialized state_dict size: counts packed int8 and 4-bit weights correctly."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 2**20


def run(cfg: Config, ws: Workspace, key: str) -> dict:
    labels = load_labels(cfg.labels_file)
    device = resolve_device(cfg.run.device)
    path = student_model_path(ws)
    test = load_splits(ws).query("split == 'test'")
    ref = load_scores(ws, "test").set_index("uid").loc[test.uid]
    gold, ref_pred = test.gold.to_numpy(), ref.pred.to_numpy()
    tok = load_tokenizer(path)
    base = load_model(path, device)
    result: dict = {"reference": f"student {'fp16' if device.startswith('cuda') else 'fp32'}",
                    "reference_footprint_mb": round(_footprint_mb(base), 1), "variants": {}}
    del base
    free_memory()
    for mode in cfg.serving.quant_modes:
        if (mode in GPU_MODES and not device.startswith("cuda")) or (mode in CPU_MODES and device.startswith("cuda")):
            log.info("Skipping %s on %s", mode, device)
            continue
        model = load_model(path, "cpu" if mode in CPU_MODES else device, quant=mode)
        start = time.perf_counter()
        pred = score_texts(model, tok, test.text.tolist(), labels, cfg.scoring.batch_size).argmax(1)
        seconds = time.perf_counter() - start
        ps = paired_summary(gold, pred, ref_pred)
        result["variants"][mode] = {
            **ps, "footprint_mb": round(_footprint_mb(model), 1), "score_seconds": round(seconds, 1),
            "prediction_agreement": float(np.mean(pred == ref_pred)),
            "non_inferiority": [non_inferiority(ps["b"], ps["c"], ps["n"], m, cfg.evaluation.alpha)
                                for m in cfg.evaluation.deltas],
        }
        log.info("%s: acc %.4f vs %.4f, agreement %.3f, verdicts %s", mode, ps["acc_new"], ps["acc_ref"],
                 result["variants"][mode]["prediction_agreement"],
                 {r["margin"]: r["verdict"] for r in result["variants"][mode]["non_inferiority"]})
        del model
        free_memory()
    ws.save_result("quantize", result, key)
    return result
