"""Typed configuration loaded from YAML, with `inherit:` and dotted CLI overrides."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunCfg:
    name: str = "full"
    seed: int = 13
    device: str = "auto"


@dataclass(frozen=True)
class DataCfg:
    dataset_id: str = "PolyAI/banking77"
    fractions: dict[str, float] = field(default_factory=dict)
    max_per_class: int | None = None
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_batch_size: int = 128
    dedup_threshold: float = 0.95
    dedup_report_thresholds: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ApiCfg:
    base_url: str | None = None
    model: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_concurrency: int = 8


@dataclass(frozen=True)
class TeacherCfg:
    backend: str = "vllm"
    model: str = "Qwen/Qwen3-8B-AWQ"
    quantization: str | None = "awq"
    dtype: str = "float16"
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.85
    k: int = 3
    temperature: float = 0.7
    top_p: float = 0.8
    max_new_tokens: int = 16
    min_agree: int = 2
    max_retries: int = 2
    constrained: bool = False
    chunk_size: int = 512
    batch_size: int = 8
    hf_load_in_4bit: bool = True
    rationale: bool = False
    rationale_max_new_tokens: int = 96
    api: ApiCfg = field(default_factory=ApiCfg)


@dataclass(frozen=True)
class LoraCfg:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StudentCfg:
    model: str = "Qwen/Qwen3-0.6B"
    qlora: bool = False
    lora: LoraCfg = field(default_factory=LoraCfg)
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 32
    grad_accum: int = 1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_length: int = 128
    gradient_checkpointing: bool = False
    evals_per_epoch: int = 2
    eval_with: str = "generate"  # dev macro-F1 during training: greedy output ("generate") or 77-label "score"
    early_stopping_patience: int = 2
    rationale_weight: float = 1.0
    curve_sizes: list[int] = field(default_factory=list)
    curve_evals_per_epoch: int = 1


@dataclass(frozen=True)
class ScoringCfg:
    batch_size: int = 8
    gen_max_new_tokens: int = 16


@dataclass(frozen=True)
class CalibrationCfg:
    n_bins: int = 15


@dataclass(frozen=True)
class EvalCfg:
    alpha: float = 0.05
    deltas: list[float] = field(default_factory=list)
    powers: list[float] = field(default_factory=list)
    primary_delta: float = 0.02
    top_classes: int = 10


@dataclass(frozen=True)
class CascadeCfg:
    error_cost: float = 1.0
    teacher_call_cost: float = 0.005
    student_call_cost: float = 0.0
    n_conf_bins: int = 10
    length_quantiles: list[float] = field(default_factory=lambda: [0.005, 0.995])


@dataclass(frozen=True)
class ServingCfg:
    concurrency: list[int] = field(default_factory=list)
    hf_batch_sizes: list[int] = field(default_factory=list)
    n_requests: int = 512
    max_tokens: int = 16
    gpu_memory_utilization: float = 0.85
    port: int = 8011
    quant_modes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    run: RunCfg = field(default_factory=RunCfg)
    data: DataCfg = field(default_factory=DataCfg)
    labels_file: str = "configs/labels.yaml"
    teacher: TeacherCfg = field(default_factory=TeacherCfg)
    student: StudentCfg = field(default_factory=StudentCfg)
    scoring: ScoringCfg = field(default_factory=ScoringCfg)
    calibration: CalibrationCfg = field(default_factory=CalibrationCfg)
    evaluation: EvalCfg = field(default_factory=EvalCfg)
    cascade: CascadeCfg = field(default_factory=CascadeCfg)
    serving: ServingCfg = field(default_factory=ServingCfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def get(self, dotted: str) -> Any:
        node: Any = self.to_dict()
        for part in dotted.split("."):
            node = node[part]
        return node

    def section_hash(self, *sections: str) -> str:
        """Stable short hash of the given (dotted) sections, used to invalidate stage outputs."""
        payload = json.dumps({s: self.get(s) for s in sections}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("inherit", None)
    if parent is None:
        return raw
    return _deep_merge(_read_yaml(path.parent / parent), raw)


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    """Recursively build a frozen dataclass, rejecting unknown keys so typos fail loudly."""
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"Unknown config keys in {where or 'root'}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        tp = hints[name]
        if isinstance(tp, types.UnionType):
            tp = next(a for a in typing.get_args(tp) if a is not type(None))
        if dataclasses.is_dataclass(tp) and isinstance(value, dict):
            # Merge onto defaults so partial nested overrides keep the other fields.
            defaults = dataclasses.asdict(tp())
            value = _build(tp, _deep_merge(defaults, value), f"{where}.{name}".lstrip("."))
        kwargs[name] = value
    return cls(**kwargs)


def parse_override(item: str) -> dict[str, Any]:
    """Turn 'a.b.c=value' into {'a': {'b': {'c': value}}} with YAML-typed value."""
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"Override must look like key=value, got {item!r}")
    node: Any = yaml.safe_load(raw)
    for part in reversed(key.strip().split(".")):
        node = {part: node}
    return node


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config (following `inherit:`), apply overrides and validate."""
    data = _read_yaml(Path(path))
    for item in overrides or []:
        data = _deep_merge(data, parse_override(item))
    defaults = dataclasses.asdict(Config())
    return _build(Config, _deep_merge(defaults, data), "")
