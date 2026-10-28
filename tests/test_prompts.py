from pathlib import Path

import pytest

from llms.labels import load_labels
from llms.prompts import (
    encode_prompt,
    encode_student,
    encode_target,
    end_of_turn,
    student_prompt,
    teacher_prefix,
    teacher_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TINY)


@pytest.mark.slow
def test_training_example_is_inference_prompt_plus_target(tok):
    ex = encode_student(tok, "Where is my new card?", "card_arrival")
    prompt = encode_prompt(tok, "Where is my new card?")
    target = encode_target(tok, "card_arrival")
    assert ex["input_ids"] == prompt + target
    assert ex["labels"] == [-100] * len(prompt) + target
    assert tok.decode(target).endswith(end_of_turn(tok))


@pytest.mark.slow
def test_student_prompt_is_short_and_has_no_label_menu(tok):
    p = student_prompt(tok, "Where is my new card?")
    assert "card_arrival" not in p and p.endswith("assistant\n")


@pytest.mark.slow
def test_teacher_prompt_starts_with_shared_prefix(tok):
    labels = load_labels(ROOT / "configs/labels.yaml")
    prefix = teacher_prefix(tok, labels)
    full = teacher_prompt(tok, "My card hasn't arrived", labels)
    assert full.startswith(prefix) and "wrong_exchange_rate_for_cash_withdrawal" in prefix
    # Tokenizing prefix and suffix separately must match tokenizing the whole (KV-cache reuse).
    ids = tok(full, add_special_tokens=False)["input_ids"]
    split = tok(prefix, add_special_tokens=False)["input_ids"] + tok(full[len(prefix):], add_special_tokens=False)["input_ids"]
    assert ids == split
