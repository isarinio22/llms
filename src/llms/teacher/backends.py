"""Teacher backends. All share one interface: `generate(texts, n, seeds) -> n raw completions per text`."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from llms.config import TeacherCfg
from llms.labels import LabelSet
from llms.prompts import teacher_messages, teacher_prefix, teacher_prompt
from llms.utils import log, resolve_device, use_spawn_for_vllm


class TeacherBackend(Protocol):
    def generate(self, texts: list[str], n: int, seeds: list[int]) -> list[list[str]]: ...


def _max_tokens(cfg: TeacherCfg) -> int:
    return cfg.rationale_max_new_tokens if cfg.rationale else cfg.max_new_tokens


class VLLMBackend:
    """Offline vLLM engine; the shared label-list prefix is reused via automatic prefix caching."""

    def __init__(self, cfg: TeacherCfg, labels: LabelSet, seed: int):
        use_spawn_for_vllm()
        from vllm import LLM

        self.cfg, self.labels = cfg, labels
        self.llm = LLM(model=cfg.model, quantization=cfg.quantization, dtype=cfg.dtype,
                       max_model_len=cfg.max_model_len, gpu_memory_utilization=cfg.gpu_memory_utilization,
                       enable_prefix_caching=True, seed=seed)
        self.tok = self.llm.get_tokenizer()

    def generate(self, texts: list[str], n: int, seeds: list[int]) -> list[list[str]]:
        from vllm import SamplingParams

        extra = {}
        if self.cfg.constrained:
            from vllm.sampling_params import StructuredOutputsParams

            extra["structured_outputs"] = StructuredOutputsParams(choice=list(self.labels.names))
        prompts = [teacher_prompt(self.tok, t, self.labels, self.cfg.rationale) for t in texts]
        params = [SamplingParams(n=n, temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                                 max_tokens=_max_tokens(self.cfg), seed=s, **extra) for s in seeds]
        outs = self.llm.generate(prompts, params, use_tqdm=True)
        return [[o.text for o in out.outputs] for out in outs]


class HFBackend:
    """Fallback without vLLM: transformers with a manually shared prefix KV cache."""

    def __init__(self, cfg: TeacherCfg, labels: LabelSet, device: str):
        from llms.hf import load_model, load_tokenizer

        self.cfg, self.labels = cfg, labels
        self.tok = load_tokenizer(cfg.model)
        quant = "bnb4" if cfg.hf_load_in_4bit and device.startswith("cuda") else None
        self.model = load_model(cfg.model, device, quant=quant)
        self.prefix_str = teacher_prefix(self.tok, labels, cfg.rationale)
        self.prefix = self.tok(self.prefix_str, add_special_tokens=False)["input_ids"]
        self.constrain_to = None
        if cfg.constrained:
            if cfg.rationale:
                raise ValueError("teacher.constrained cannot be combined with teacher.rationale")
            from llms.prompts import encode_target

            self.constrain_to = [encode_target(self.tok, name) for name in labels.names]
        log.info("HF teacher: shared prefix of %d tokens is computed once per batch", len(self.prefix))

    def generate(self, texts: list[str], n: int, seeds: list[int]) -> list[list[str]]:
        from llms.hf import sample_with_prefix

        out: list[list[str]] = []
        bs = self.cfg.batch_size
        for i in range(0, len(texts), bs):
            suffixes = []
            for t in texts[i : i + bs]:
                full = teacher_prompt(self.tok, t, self.labels, self.cfg.rationale)
                suffixes.append(self.tok(full[len(self.prefix_str):], add_special_tokens=False)["input_ids"])
            out += sample_with_prefix(self.model, self.tok, self.prefix, suffixes, n, _max_tokens(self.cfg),
                                      self.cfg.temperature, self.cfg.top_p, seeds[i], self.constrain_to)
        return out


class ApiBackend:
    """Any OpenAI-compatible /chat/completions endpoint (optional, off by default)."""

    def __init__(self, cfg: TeacherCfg, labels: LabelSet):
        import httpx

        api = cfg.api
        if not api.base_url or not api.model:
            raise ValueError("teacher.backend=api needs teacher.api.base_url and teacher.api.model")
        key = os.environ.get(api.api_key_env)
        if not key:
            raise OSError(f"Set the {api.api_key_env} environment variable for the API teacher")
        self.cfg, self.labels = cfg, labels
        self.client = httpx.Client(base_url=api.base_url, timeout=60,
                                   headers={"Authorization": f"Bearer {key}"})

    def _one(self, text: str, seed: int) -> str:
        body = {"model": self.cfg.api.model, "messages": teacher_messages(text, self.labels, self.cfg.rationale),
                "temperature": self.cfg.temperature, "top_p": self.cfg.top_p,
                "max_tokens": _max_tokens(self.cfg), "seed": seed}
        for attempt in range(6):
            r = self.client.post("/chat/completions", json=body)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        r.raise_for_status()
        return ""

    def generate(self, texts: list[str], n: int, seeds: list[int]) -> list[list[str]]:
        jobs = [(t, s * 31 + j) for t, s in zip(texts, seeds, strict=True) for j in range(n)]
        with ThreadPoolExecutor(self.cfg.api.max_concurrency) as pool:
            flat = list(pool.map(lambda a: self._one(*a), jobs))
        return [flat[i * n : (i + 1) * n] for i in range(len(texts))]


def make_backend(cfg: TeacherCfg, labels: LabelSet, device: str, seed: int) -> TeacherBackend:
    if cfg.backend == "vllm":
        return VLLMBackend(cfg, labels, seed)
    if cfg.backend == "hf":
        return HFBackend(cfg, labels, resolve_device(device))
    if cfg.backend == "api":
        return ApiBackend(cfg, labels)
    raise ValueError(f"Unknown teacher backend {cfg.backend!r}")
