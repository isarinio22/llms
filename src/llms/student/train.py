"""Stage 3: LoRA SFT of the student on teacher labels only (never on gold).

Loss is computed on target tokens only. The LM head is applied just at those positions, so the
(batch x seq x 151k-vocab) logits tensor is never materialized. Early stopping uses dev macro-F1
from the real 77-label scorer.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn import functional as F

from llms.config import Config
from llms.data import load_splits
from llms.labels import LabelSet, load_labels
from llms.metrics import summary
from llms.prompts import encode_student
from llms.teacher.label import load_teacher_labels
from llms.utils import log, resolve_device, torch_dtype
from llms.workspace import Workspace, write_json


def build_examples(tok: Any, train: pd.DataFrame, labels: LabelSet, cfg: Config) -> list[dict[str, Any]]:
    """Label-task examples from teacher labels; plus rationale-task examples in the multi-task ablation."""
    max_len = cfg.student.max_length
    examples = [{**encode_student(tok, t, labels.names[p], "label", max_len), "weight": 1.0}
                for t, p in zip(train.text, train.teacher_pred, strict=True)]
    if cfg.teacher.rationale:
        examples += [{**encode_student(tok, t, r, "rationale", max_len), "weight": cfg.student.rationale_weight}
                     for t, r in zip(train.text, train.rationale, strict=True) if isinstance(r, str) and r]
    return examples


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n], labels[i, :n], mask[i, :n] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels,
            "weight": torch.tensor([b["weight"] for b in batch], dtype=torch.float32)}


def label_only_loss(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean over examples of the mean target-token cross-entropy, weighted per example (task weight)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.get_decoder()(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    targets = batch["labels"][:, 1:]
    sel = targets != -100
    logits = base.get_output_embeddings()(hidden[:, :-1][sel])
    token_loss = F.cross_entropy(logits.float(), targets[sel], reduction="none")
    row = sel.nonzero()[:, 0]
    per_example = torch.zeros(sel.shape[0], device=token_loss.device).index_add_(0, row, token_loss)
    per_example = per_example / sel.sum(1).clamp(min=1)
    w = batch["weight"].to(per_example.device)
    return (per_example * w).sum() / w.sum()


def load_student_for_training(cfg: Config, device: str) -> tuple[Any, Any]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from llms.hf import load_tokenizer

    sc = cfg.student
    tok = load_tokenizer(sc.model)
    kwargs: dict[str, Any] = {"dtype": torch_dtype(device)}
    if sc.qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(sc.model, **kwargs)
    if sc.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=sc.gradient_checkpointing)
    else:
        model.to(device)
        if sc.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
    lora = LoraConfig(r=sc.lora.r, lora_alpha=sc.lora.alpha, lora_dropout=sc.lora.dropout,
                      target_modules=sc.lora.target_modules, task_type="CAUSAL_LM")
    return get_peft_model(model, lora), tok


def _make_trainer_cls():
    from transformers import Trainer

    class LabelOnlyTrainer(Trainer):
        """Trainer with label-only loss and dev evaluation by 77-label scoring (macro-F1)."""

        def __init__(self, *args: Any, dev: pd.DataFrame, labels: LabelSet, score_bs: int, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.dev, self.labels, self.score_bs = dev, labels, score_bs

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = label_only_loss(model, inputs)
            return (loss, None) if return_outputs else loss

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            from llms.scoring import score_texts

            self.model.eval()
            scores = score_texts(self.model, self.processing_class, self.dev.text.tolist(), self.labels, self.score_bs)
            m = summary(self.dev.gold.to_numpy(), scores.argmax(1), len(self.labels))
            metrics = {f"{metric_key_prefix}_macro_f1": m["macro_f1"], f"{metric_key_prefix}_accuracy": m["accuracy"]}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            self.model.train()
            return metrics

    return LabelOnlyTrainer


def train_lora(cfg: Config, ws: Workspace, train: pd.DataFrame, tag: str, evals_per_epoch: int, merge: bool) -> dict:
    """Train one LoRA adapter on `train` (teacher labels), early-stopped on dev macro-F1."""
    from transformers import EarlyStoppingCallback, TrainingArguments

    sc = cfg.student
    device = resolve_device(cfg.run.device)
    labels = load_labels(cfg.labels_file)
    dev = load_splits(ws).query("split == 'dev'")
    out = ws.stage_dir("train_student") / tag
    model, tok = load_student_for_training(cfg, device)
    examples = build_examples(tok, train, labels, cfg)
    truncated = sum(len(e["input_ids"]) >= sc.max_length for e in examples)
    trainable, total = model.get_nb_trainable_parameters()
    steps_per_epoch = math.ceil(len(examples) / (sc.batch_size * sc.grad_accum))
    eval_steps = max(1, math.ceil(steps_per_epoch / evals_per_epoch))
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=sc.batch_size,
        gradient_accumulation_steps=sc.grad_accum, learning_rate=sc.learning_rate, num_train_epochs=sc.epochs,
        warmup_steps=math.ceil(sc.warmup_ratio * steps_per_epoch * sc.epochs), weight_decay=sc.weight_decay,
        lr_scheduler_type="linear", eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps",
        save_steps=eval_steps, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1", greater_is_better=True, logging_steps=max(1, eval_steps // 4),
        fp16=device.startswith("cuda"), use_cpu=device == "cpu", seed=cfg.run.seed, data_seed=cfg.run.seed,
        remove_unused_columns=False, report_to="none", dataloader_num_workers=0, disable_tqdm=False,
    )
    trainer = _make_trainer_cls()(
        model=model, args=args, train_dataset=examples, eval_dataset=examples[:1], processing_class=tok,
        data_collator=lambda b: collate(b, tok.pad_token_id), dev=dev, labels=labels,
        score_bs=cfg.scoring.batch_size,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=sc.early_stopping_patience)],
    )
    resume = any((out / "checkpoints").glob("checkpoint-*"))
    log.info("[%s] %d examples (%d truncated), %d steps/epoch, eval every %d, trainable %.2fM/%.0fM%s",
             tag, len(examples), truncated, steps_per_epoch, eval_steps, trainable / 1e6, total / 1e6,
             " (resuming)" if resume else "")
    train_out = trainer.train(resume_from_checkpoint=resume or None)
    trainer.model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    evals = [h for h in trainer.state.log_history if "eval_macro_f1" in h]
    best = max(evals, key=lambda h: h["eval_macro_f1"])
    info = {
        "tag": tag, "n_train_examples": len(train), "n_sequences": len(examples), "truncated": truncated,
        "trainable_params": trainable, "total_params": total, "steps_per_epoch": steps_per_epoch,
        "global_steps": trainer.state.global_step, "train_runtime_s": round(train_out.metrics["train_runtime"], 1),
        "best_step": best["step"], "best_dev_macro_f1": best["eval_macro_f1"],
        "best_dev_accuracy": best["eval_accuracy"], "early_stopped": trainer.state.global_step < steps_per_epoch * sc.epochs,
        "log_history": trainer.state.log_history,
    }
    write_json(out / "train_info.json", info)
    if merge:
        merge_adapter(cfg, out / "adapter", out / "merged", device)
    del trainer, model
    from llms.hf import free_memory

    free_memory()
    return info


def merge_adapter(cfg: Config, adapter: Path, dest: Path, device: str) -> None:
    """Merge LoRA into an fp16 copy of the base model (a QLoRA base is reloaded unquantized)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from llms.hf import load_tokenizer

    base = AutoModelForCausalLM.from_pretrained(cfg.student.model, dtype=torch_dtype(device))
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained(dest, safe_serialization=True)
    load_tokenizer(cfg.student.model).save_pretrained(dest)
    log.info("Merged model saved to %s", dest)


def training_set(ws: Workspace) -> pd.DataFrame:
    """Train-pool items that passed the self-consistency filter, with TEACHER labels (gold dropped)."""
    lab = load_teacher_labels(ws)
    train = lab[(lab.split == "train_pool") & lab.kept & (lab.teacher_pred >= 0)]
    return train.drop(columns=["gold"]).reset_index(drop=True)


def run(cfg: Config, ws: Workspace, key: str) -> dict:
    train = training_set(ws)
    info = train_lora(cfg, ws, train, "main", cfg.student.evals_per_epoch, merge=True)
    result = {k: v for k, v in info.items() if k != "log_history"}
    result |= {"student": cfg.student.model, "qlora": cfg.student.qlora, "lora": dataclasses.asdict(cfg.student.lora),
               "multitask_rationale": cfg.teacher.rationale,
               "loss_curve": [{"step": h["step"], "loss": h["loss"]} for h in info["log_history"] if "loss" in h],
               "dev_curve": [{"step": h["step"], "macro_f1": h["eval_macro_f1"]} for h in info["log_history"] if "eval_macro_f1" in h]}
    ws.save_result("train_student", result, key)
    log.info("Best dev macro-F1 %.3f at step %d", info["best_dev_macro_f1"], info["best_step"])
    return result
