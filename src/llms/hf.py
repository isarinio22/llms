"""Hugging Face transformers helpers: loading, label scoring with KV reuse, prefix-cached sampling."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from llms.prompts import end_of_turn
from llms.utils import log, torch_dtype


def load_tokenizer(name: str) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name: str, device: str, quant: str | None = None, adapter: str | None = None) -> Any:
    """Load a causal LM for inference. quant: None | 'bnb8' | 'bnb4' (GPU) | 'int8_dynamic' (CPU)."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs: dict[str, Any] = {"dtype": torch_dtype(device)}
    if quant in ("bnb4", "bnb8"):
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            if quant == "bnb4" else BitsAndBytesConfig(load_in_8bit=True)
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if quant not in ("bnb4", "bnb8"):
        model.to(device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    if quant == "int8_dynamic":
        engines = torch.backends.quantized.supported_engines
        torch.backends.quantized.engine = next(e for e in ("x86", "fbgemm", "qnnpack") if e in engines)
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model.eval()


def stop_token_ids(tok: Any) -> list[int]:
    ids = {tok.eos_token_id, tok.convert_tokens_to_ids(end_of_turn(tok))}
    return sorted(i for i in ids if i is not None and i != tok.unk_token_id)


def _device(model: Any) -> torch.device:
    return next(model.parameters()).device


def trie_constraint(sequences: list[list[int]], prompt_width: int, stop_ids: list[int]) -> Any:
    """`prefix_allowed_tokens_fn` that only lets generation spell one of `sequences` (constrained decoding)."""
    trie: dict[tuple[int, ...], set[int]] = {}
    for seq in sequences:
        for i in range(len(seq)):
            trie.setdefault(tuple(seq[:i]), set()).add(seq[i])
    allowed = {k: sorted(v) for k, v in trie.items()}

    def fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        return allowed.get(tuple(input_ids[prompt_width:].tolist()), stop_ids)

    return fn


def _left_pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s)
        mask[i, width - len(s):] = 1
    return ids, mask


def _target_logprobs(logits: torch.Tensor, targets: torch.Tensor, rows: int = 64) -> torch.Tensor:
    """log p(target) per position, upcasting to fp32 in row chunks to bound memory."""
    out = torch.empty(targets.shape, dtype=torch.float32, device=logits.device)
    for i in range(0, logits.shape[0], rows):
        lp = torch.log_softmax(logits[i : i + rows].float(), dim=-1)
        out[i : i + rows] = lp.gather(-1, targets[i : i + rows].unsqueeze(-1)).squeeze(-1)
    return out


@torch.inference_mode()
def score_candidates(
    model: Any, prompts: list[list[int]], candidates: list[list[int]], pad_id: int, batch_size: int
) -> np.ndarray:
    """Sum of candidate-token log-probs given each prompt: an (n_prompts, n_candidates) matrix.

    Each prompt is encoded once; its KV cache is then repeated for all candidates, which are scored
    in one batched forward. Equivalent to scoring prompt+candidate sequences one by one.
    """
    dev = _device(model)
    n_cand = len(candidates)
    width = max(len(c) for c in candidates)
    cand = torch.full((n_cand, width), pad_id, dtype=torch.long)
    cmask = torch.zeros((n_cand, width), dtype=torch.long)
    for j, c in enumerate(candidates):
        cand[j, : len(c)] = torch.tensor(c)
        cmask[j, : len(c)] = 1
    cand, cmask = cand.to(dev), cmask.to(dev)
    out = np.empty((len(prompts), n_cand), dtype=np.float32)
    for start in range(0, len(prompts), batch_size):
        ids, mask = (t.to(dev) for t in _left_pad(prompts[start : start + batch_size], pad_id))
        b = ids.shape[0]
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        res = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        total = torch.log_softmax(res.logits[:, -1].float(), dim=-1)[:, cand[:, 0]]
        if width > 1:
            cache = res.past_key_values
            cache.batch_repeat_interleave(n_cand)
            attn = torch.cat([mask.repeat_interleave(n_cand, 0), cmask[:, :-1].repeat(b, 1)], dim=1)
            cpos = mask.sum(1).repeat_interleave(n_cand)[:, None] + torch.arange(width - 1, device=dev)[None]
            logits = model(input_ids=cand[:, :-1].repeat(b, 1), attention_mask=attn, position_ids=cpos,
                           past_key_values=cache, use_cache=True).logits
            rest = _target_logprobs(logits, cand[:, 1:].repeat(b, 1)) * cmask[:, 1:].repeat(b, 1)
            total = total + rest.sum(-1).view(b, n_cand)
        out[start : start + b] = total.cpu().numpy()
    return out


@torch.inference_mode()
def generate(
    model: Any,
    tok: Any,
    prompts: list[list[int]],
    max_new_tokens: int,
    batch_size: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 0,
) -> list[str]:
    """Batched generation from token-id prompts (left padded). Returns decoded new text."""
    from transformers import GenerationConfig

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None, top_k=0 if do_sample else None,
        pad_token_id=tok.pad_token_id, eos_token_id=stop_token_ids(tok),
    )
    dev = _device(model)
    texts: list[str] = []
    for start in range(0, len(prompts), batch_size):
        ids, mask = (t.to(dev) for t in _left_pad(prompts[start : start + batch_size], tok.pad_token_id))
        torch.manual_seed(seed + start)
        out = model.generate(input_ids=ids, attention_mask=mask, generation_config=gen_cfg)
        texts += tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
    return texts


@torch.inference_mode()
def sample_with_prefix(
    model: Any,
    tok: Any,
    prefix: list[int],
    suffixes: list[list[int]],
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    constrain_to: list[list[int]] | None = None,
) -> list[list[str]]:
    """Sample n completions per suffix, reusing one KV cache for the shared prefix.

    Rows are laid out as [prefix][padding][suffix]; the padding is masked and position ids come from
    the attention mask, so each row is equivalent to the unpadded prompt prefix + suffix.
    `constrain_to` restricts outputs to these token sequences (e.g. the encoded label names).
    """
    from transformers import GenerationConfig

    dev = _device(model)
    base = model(input_ids=torch.tensor([prefix], device=dev), use_cache=True).past_key_values
    suf_ids, suf_mask = _left_pad(suffixes, tok.pad_token_id)
    rows = len(suffixes) * n
    ids = torch.cat([torch.tensor([prefix]).repeat(rows, 1), suf_ids.repeat_interleave(n, 0)], 1).to(dev)
    mask = torch.cat([torch.ones(rows, len(prefix), dtype=torch.long), suf_mask.repeat_interleave(n, 0)], 1).to(dev)
    cache = copy.deepcopy(base)
    cache.batch_repeat_interleave(rows)
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p, top_k=0,
        pad_token_id=tok.pad_token_id, eos_token_id=stop_token_ids(tok),
    )
    allowed_fn = trie_constraint(constrain_to, ids.shape[1], stop_token_ids(tok)) if constrain_to else None
    torch.manual_seed(seed)
    out = model.generate(input_ids=ids, attention_mask=mask, past_key_values=cache, generation_config=gen_cfg,
                         prefix_allowed_tokens_fn=allowed_fn)
    texts = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
    return [texts[i * n : (i + 1) * n] for i in range(len(suffixes))]


def free_memory() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.debug("freed memory")
