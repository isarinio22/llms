"""Stage 7a: serving benchmark. vLLM (continuous batching, OpenAI server) vs plain HF generate.

vLLM is measured as a real server under N concurrent streaming clients (TTFT = first streamed token).
HF has no server, so "concurrency" there is a static batch size and every request in a batch shares
that batch's latency. Both use the same prompts, built by the shared student prompt function.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from llms.config import Config
from llms.data import load_splits
from llms.hf import free_memory, load_model, load_tokenizer, stop_token_ids
from llms.prompts import student_prompt
from llms.scoring import student_model_path
from llms.utils import log, resolve_device
from llms.workspace import Workspace


def _summarize(engine: str, concurrency: int, recs: list[dict], wall: float) -> dict:
    lat = np.array([r["latency"] for r in recs])
    ttft = np.array([r["ttft"] for r in recs])
    tokens = sum(r["tokens"] for r in recs)
    return {"engine": engine, "concurrency": concurrency, "n_requests": len(recs), "wall_s": round(wall, 3),
            "requests_per_s": len(recs) / wall, "output_tokens_per_s": tokens / wall,
            "p50_latency_s": float(np.percentile(lat, 50)), "p95_latency_s": float(np.percentile(lat, 95)),
            "p50_ttft_s": float(np.percentile(ttft, 50)), "p95_ttft_s": float(np.percentile(ttft, 95))}


# HF ------------------------------------------------------------------------------
class _FirstToken:
    """StoppingCriteria hook that records when the first new token exists (TTFT of a batch)."""

    def __init__(self) -> None:
        self.t: float | None = None

    def __call__(self, input_ids: torch.Tensor, scores: Any, **kwargs: Any) -> torch.Tensor:
        if self.t is None:
            self.t = time.perf_counter()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


@torch.inference_mode()
def bench_hf(model: Any, tok: Any, prompts: list[str], batch_size: int, max_tokens: int) -> dict:
    from transformers import GenerationConfig, StoppingCriteriaList

    gen_cfg = GenerationConfig(max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                               eos_token_id=stop_token_ids(tok))
    tok.padding_side = "left"
    recs: list[dict] = []
    start = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        hook, t0 = _FirstToken(), time.perf_counter()
        out = model.generate(**enc, generation_config=gen_cfg, stopping_criteria=StoppingCriteriaList([hook]))
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = out[:, enc["input_ids"].shape[1]:]
        n_tok = (new != tok.pad_token_id).sum(1).tolist()
        recs += [{"latency": dt, "ttft": (hook.t or t0 + dt) - t0, "tokens": n} for n in n_tok]
    return _summarize("hf generate", batch_size, recs, time.perf_counter() - start)


# vLLM ----------------------------------------------------------------------------
async def _client(base_url: str, prompts: list[str], concurrency: int, max_tokens: int) -> tuple[list[dict], float]:
    import httpx

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=300) as client:

        async def one(prompt: str) -> dict:
            async with sem:
                body = {"model": "student", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
                        "stream": True, "stream_options": {"include_usage": True}}
                t0, ttft, tokens = time.perf_counter(), None, 0
                async with client.stream("POST", "/v1/completions", json=body) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                            continue
                        chunk = json.loads(line[6:])
                        if ttft is None and chunk.get("choices") and chunk["choices"][0].get("text"):
                            ttft = time.perf_counter() - t0
                        if chunk.get("usage"):
                            tokens = chunk["usage"]["completion_tokens"]
                latency = time.perf_counter() - t0
                return {"latency": latency, "ttft": ttft if ttft is not None else latency, "tokens": tokens}

        start = time.perf_counter()
        recs = await asyncio.gather(*(one(p) for p in prompts))
        return list(recs), time.perf_counter() - start


def _vllm_cli() -> str | None:
    """The `vllm` executable of the running interpreter's environment (works inside a venv not on PATH)."""
    local = Path(sys.executable).parent / "vllm"
    return str(local) if local.exists() else shutil.which("vllm")


def _start_server(model_path: str, cfg: Config, extra: list[str], log_path: Path) -> subprocess.Popen:
    exe = _vllm_cli()
    if exe is None:
        raise RuntimeError("vllm CLI not found")
    cmd = [exe, "serve", model_path, "--served-model-name", "student", "--dtype", "float16",
           "--max-model-len", "512", "--gpu-memory-utilization", str(cfg.serving.gpu_memory_utilization),
           "--port", str(cfg.serving.port), *extra]
    log.info("$ %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=log_path.open("w"), stderr=subprocess.STDOUT)


def _wait_ready(proc: subprocess.Popen, url: str, timeout: float = 900) -> None:
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {proc.returncode}")
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise TimeoutError("vLLM server did not become ready")


def bench_vllm(model_path: str, cfg: Config, prompts: list[str], engine: str, extra: list[str], log_path: Path) -> list[dict]:
    url = f"http://127.0.0.1:{cfg.serving.port}"
    proc = _start_server(model_path, cfg, extra, log_path)
    rows = []
    try:
        _wait_ready(proc, url)
        asyncio.run(_client(url, prompts[:16], 4, cfg.serving.max_tokens))  # warm-up
        for c in cfg.serving.concurrency:
            recs, wall = asyncio.run(_client(url, prompts, c, cfg.serving.max_tokens))
            rows.append(_summarize(engine, c, recs, wall))
            log.info("%s c=%d: %.1f req/s, p95 %.0f ms, TTFT p50 %.0f ms", engine, c, rows[-1]["requests_per_s"],
                     1000 * rows[-1]["p95_latency_s"], 1000 * rows[-1]["p50_ttft_s"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    return rows


def run(cfg: Config, ws: Workspace, key: str) -> dict:
    device = resolve_device(cfg.run.device)
    path = student_model_path(ws)
    tok = load_tokenizer(path)
    texts = load_splits(ws).query("split == 'test'").text.tolist()
    texts = (texts * (cfg.serving.n_requests // max(len(texts), 1) + 1))[: cfg.serving.n_requests]
    prompts = [student_prompt(tok, t) for t in texts]
    rows: list[dict] = []
    notes: list[str] = []

    model = load_model(path, device)
    bench_hf(model, tok, prompts[:8], 4, cfg.serving.max_tokens)  # warm-up
    for bs in cfg.serving.hf_batch_sizes:
        rows.append(bench_hf(model, tok, prompts, bs, cfg.serving.max_tokens))
        log.info("hf batch=%d: %.1f req/s, p95 %.0f ms", bs, rows[-1]["requests_per_s"], 1000 * rows[-1]["p95_latency_s"])
    del model
    free_memory()

    if device.startswith("cuda") and _vllm_cli():
        variants = [("vllm fp16", [])]
        if "bnb4" in cfg.serving.quant_modes:
            variants.append(("vllm bnb4", ["--quantization", "bitsandbytes"]))
        for engine, extra in variants:
            try:
                rows += bench_vllm(path, cfg, prompts, engine, extra, ws.stage_dir("serve_bench") / f"{engine.replace(' ', '_')}.log")
            except (RuntimeError, TimeoutError) as e:
                notes.append(f"{engine} failed: {e} (see artifacts/{cfg.run.name}/serve_bench/*.log)")
                log.warning(notes[-1])
    else:
        notes.append("vLLM not benchmarked: needs a CUDA GPU and the vllm package (smoke mode runs HF only).")

    result = {"model": path, "device": device, "max_tokens": cfg.serving.max_tokens,
              "n_requests": len(prompts), "rows": rows, "notes": notes}
    ws.save_result("serving", result, key)
    from llms.plots import throughput

    throughput(rows, ws.figures / "throughput_concurrency.png")
    return result
