"""Seeding, device selection and small timing helpers."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

log = logging.getLogger("llms")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(device: str) -> str:
    """'auto' -> cuda when available, else cpu. MPS is deliberately not auto-selected (determinism)."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def torch_dtype(device: str):
    """fp16 on GPU (the T4 has no bf16), fp32 on CPU."""
    import torch

    return torch.float16 if device.startswith("cuda") else torch.float32


@contextmanager
def timed(label: str, sink: dict[str, float] | None = None) -> Iterator[None]:
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    log.info("%s took %.1fs", label, elapsed)
    if sink is not None:
        sink[label] = round(elapsed, 2)


def batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
