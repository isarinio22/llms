"""Sentence embeddings (baseline features and near-duplicate detection)."""

from __future__ import annotations

import numpy as np

from llms.utils import log


def embed_texts(model_name: str, texts: list[str], batch_size: int, device: str) -> np.ndarray:
    """L2-normalized float32 embeddings, one row per text."""
    import torch
    from sentence_transformers import SentenceTransformer

    kwargs = {"dtype": torch.float16} if device.startswith("cuda") else {}
    model = SentenceTransformer(model_name, device=device, model_kwargs=kwargs)
    log.info("Embedding %d texts with %s on %s", len(texts), model_name, device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 2000,
    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return emb.astype(np.float32)


def max_cosine(query: np.ndarray, ref: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """For each query row, the max cosine similarity to `ref` and its argmax (inputs normalized)."""
    if len(ref) == 0:
        return np.full(len(query), -1.0, dtype=np.float32), np.full(len(query), -1, dtype=np.int64)
    best = np.empty(len(query), dtype=np.float32)
    arg = np.empty(len(query), dtype=np.int64)
    for i in range(0, len(query), chunk):
        sims = query[i : i + chunk] @ ref.T
        arg[i : i + chunk] = sims.argmax(axis=1)
        best[i : i + chunk] = sims[np.arange(len(sims)), arg[i : i + chunk]]
    return best, arg
