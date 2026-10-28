"""The KV-reuse scorer and prefix-cached sampler must match naive computation exactly."""

import pytest
import torch

from llms.hf import load_model, load_tokenizer, sample_with_prefix, score_candidates

TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def model_tok():
    torch.manual_seed(0)
    return load_model(TINY, "cpu"), load_tokenizer(TINY)


def _naive(model, prompt, cand):
    ids = torch.tensor([prompt + cand])
    with torch.inference_mode():
        lp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
    return sum(lp[len(prompt) - 1 + t, tok_id].item() for t, tok_id in enumerate(cand))


@pytest.mark.slow
def test_score_candidates_matches_naive(model_tok):
    model, tok = model_tok
    prompts = [tok("Hello there, my card", add_special_tokens=False)["input_ids"],
               tok("Hi", add_special_tokens=False)["input_ids"]]  # different lengths -> padding
    cands = [tok(c, add_special_tokens=False)["input_ids"] for c in [" arrived", " is lost forever", " x"]]
    got = score_candidates(model, prompts, cands, tok.pad_token_id, batch_size=2)
    for i, p in enumerate(prompts):
        for j, c in enumerate(cands):
            assert got[i, j] == pytest.approx(_naive(model, p, c), abs=2e-3)


@pytest.mark.slow
def test_prefix_sampling_matches_unpadded_greedy(model_tok):
    model, tok = model_tok
    prefix = tok("The bank said:", add_special_tokens=False)["input_ids"]
    suffixes = [tok(" ok", add_special_tokens=False)["input_ids"],
                tok(" please wait for the card", add_special_tokens=False)["input_ids"]]
    # temperature -> 0 makes sampling greedy, so the padded prefix-cache path must equal plain greedy.
    got = sample_with_prefix(model, tok, prefix, suffixes, n=2, max_new_tokens=6, temperature=1e-4, top_p=1.0, seed=0)
    for s, outs in zip(suffixes, got, strict=True):
        ids = torch.tensor([prefix + s])
        with torch.inference_mode():
            ref = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=6,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        ref_text = tok.decode(ref[0, ids.shape[1]:], skip_special_tokens=True)
        assert outs[0] == outs[1] == ref_text
