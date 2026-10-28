"""Prompt construction.

`student_prompt` is the ONE function that formats student inputs; training, scoring, greedy
generation and the serving benchmark all go through it, so the format can never drift.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from llms.labels import LabelSet

MAX_MESSAGE_CHARS = 1000

TEACHER_SYSTEM = """You classify customer messages sent to a digital bank's support team.
Pick the single intent from this list (format "label: description") that best matches the message.

{label_block}

Reply with the label only, written exactly as in the list, and nothing else."""

TEACHER_SYSTEM_RATIONALE = """You classify customer messages sent to a digital bank's support team.
Pick the single intent from this list (format "label: description") that best matches the message.

{label_block}

Reply in exactly two lines:
Rationale: <one sentence explaining the choice>
Label: <the label, written exactly as in the list>"""

STUDENT_INSTRUCTIONS = {
    "label": "Classify the intent of this banking customer message.",
    "rationale": "In one sentence, explain which banking intent this customer message expresses.",
}

_SENTINEL = "⁣MESSAGE⁣"


def _clean(text: str) -> str:
    return " ".join(text.split())[:MAX_MESSAGE_CHARS]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Apply the model's chat template with a generation prompt. Thinking mode off (Qwen3)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


# Teacher ------------------------------------------------------------------------
def teacher_messages(text: str, labels: LabelSet, rationale: bool = False) -> list[dict[str, str]]:
    block = "\n".join(f"{n}: {d}" for n, d in zip(labels.names, labels.descriptions, strict=True))
    system = (TEACHER_SYSTEM_RATIONALE if rationale else TEACHER_SYSTEM).format(label_block=block)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Customer message:\n{_clean(text)}"},
    ]


def teacher_prompt(tokenizer: Any, text: str, labels: LabelSet, rationale: bool = False) -> str:
    return render_prompt(tokenizer, teacher_messages(text, labels, rationale))


def teacher_prefix(tokenizer: Any, labels: LabelSet, rationale: bool = False) -> str:
    """The part of every teacher prompt that precedes the message (shared, so it can be KV-cached)."""
    return teacher_prompt(tokenizer, _SENTINEL, labels, rationale).split(_SENTINEL)[0]


# Student ------------------------------------------------------------------------
def student_messages(text: str, task: str = "label") -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{STUDENT_INSTRUCTIONS[task]}\nMessage: {_clean(text)}"}]


def student_prompt(tokenizer: Any, text: str, task: str = "label") -> str:
    """The shared student prompt (string). Short on purpose: no label menu, the student learns it."""
    return render_prompt(tokenizer, student_messages(text, task))


@lru_cache(maxsize=8)
def _end_of_turn_cached(tokenizer: Any) -> str:
    marker = "⁣TARGET⁣"
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": marker}],
        tokenize=False,
        enable_thinking=False,
    )
    eot = full.split(marker, 1)[1].strip()
    return eot or (tokenizer.eos_token or "")


def end_of_turn(tokenizer: Any) -> str:
    """The string that closes an assistant turn in this chat template (e.g. '<|im_end|>')."""
    return _end_of_turn_cached(tokenizer)


def encode_target(tokenizer: Any, target: str) -> list[int]:
    """Target tokens: the answer plus the end-of-turn token, so a label that is a prefix of another
    label is not favored and the model learns where an answer stops."""
    return tokenizer(target + end_of_turn(tokenizer), add_special_tokens=False)["input_ids"]


def encode_prompt(tokenizer: Any, text: str, task: str = "label") -> list[int]:
    return tokenizer(student_prompt(tokenizer, text, task), add_special_tokens=False)["input_ids"]


def encode_student(
    tokenizer: Any, text: str, target: str, task: str = "label", max_length: int | None = None
) -> dict[str, list[int]]:
    """Training example: prompt + target, with loss only on the target (prompt positions = -100)."""
    prompt_ids = encode_prompt(tokenizer, text, task)
    target_ids = encode_target(tokenizer, target)
    if max_length is not None and len(prompt_ids) + len(target_ids) > max_length:
        keep = max(max_length - len(target_ids), 1)
        prompt_ids = prompt_ids[-keep:]
    input_ids = prompt_ids + target_ids
    return {"input_ids": input_ids, "labels": [-100] * len(prompt_ids) + target_ids}
