"""Stage 0: environment report and vLLM compatibility probe (run this first on Colab).

The probe loads each model in a fresh subprocess, generates a few tokens and records whether it
worked, so a T4 incompatibility shows up here instead of halfway through teacher labeling.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import subprocess
import sys
import time

from llms.config import Config
from llms.utils import log, resolve_device, set_seed, use_spawn_for_vllm
from llms.workspace import Workspace

PACKAGES = ("torch", "transformers", "peft", "accelerate", "sentence-transformers", "vllm", "bitsandbytes")


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def environment() -> dict:
    import torch

    env: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in PACKAGES},
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        env |= {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": f"{cap[0]}.{cap[1]}",
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
            "torch_cuda": torch.version.cuda,
        }
        try:
            env["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            env["driver"] = None
    return env


def probe_vllm(model: str, quantization: str | None, max_model_len: int, gpu_mem: float) -> dict:
    """Load `model` in vLLM and generate. Runs inside a subprocess (see `run`).

    Mirrors a pipeline stage: the process seeds torch and queries CUDA before vLLM starts, so a
    fork-related failure shows up here and not halfway through a run.
    """
    set_seed(0)
    resolve_device("auto")
    use_spawn_for_vllm()
    start = time.perf_counter()
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, quantization=quantization, dtype="float16", max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem, enable_prefix_caching=True)
    load_s = time.perf_counter() - start
    tok = llm.get_tokenizer()
    prompt = tok.apply_chat_template([{"role": "user", "content": "Reply with the word: card_arrival"}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    t0 = time.perf_counter()
    out = llm.generate([prompt] * 8, SamplingParams(max_tokens=8, temperature=0.0))
    return {"ok": True, "load_s": round(load_s, 1), "gen_s": round(time.perf_counter() - t0, 2),
            "sample": out[0].outputs[0].text}


def _probe_in_subprocess(model: str, quantization: str | None, max_model_len: int, gpu_mem: float) -> dict:
    code = (
        "import json,sys; from llms.doctor import probe_vllm; "
        f"print('PROBE_RESULT=' + json.dumps(probe_vllm({model!r}, {quantization!r}, {max_model_len}, {gpu_mem})))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE_RESULT="):
            return json.loads(line.removeprefix("PROBE_RESULT="))
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-15:]
    return {"ok": False, "error_tail": tail}


def run(cfg: Config, ws: Workspace, key: str) -> dict:
    env = environment()
    probes: dict = {}
    if env["cuda_available"] and env["packages"]["vllm"]:
        for role, model, quant in (("student", cfg.student.model, None),
                                   ("teacher", cfg.teacher.model, cfg.teacher.quantization)):
            log.info("Probing vLLM with the %s model %s ...", role, model)
            probes[role] = _probe_in_subprocess(model, quant, cfg.teacher.max_model_len,
                                                cfg.teacher.gpu_memory_utilization)
            log.info("%s probe: %s", role, "OK" if probes[role]["ok"] else "FAILED")
    elif env["cuda_available"]:
        log.warning("vLLM is not installed; teacher and serving stages will use Hugging Face transformers.")
    else:
        log.info("No CUDA GPU: smoke mode runs everything with Hugging Face transformers on CPU.")
    vllm_ok = bool(probes) and all(p["ok"] for p in probes.values())
    result = {"environment": env, "vllm_probes": probes, "vllm_usable": vllm_ok,
              "recommended_teacher_backend": "vllm" if vllm_ok else "hf"}
    if env["cuda_available"] and cfg.teacher.backend == "vllm" and not vllm_ok:
        log.warning("vLLM probe failed. Rerun stages with --set teacher.backend=hf (see results/%s/env.json).",
                    cfg.run.name)
    ws.save_result("env", result, key)
    print(json.dumps(result, indent=2))
    return result
