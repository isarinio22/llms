from pathlib import Path

import pytest

from llms.config import load_config, parse_override

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_inherits_base_and_overrides():
    cfg = load_config(ROOT / "configs/smoke.yaml")
    assert cfg.run.name == "smoke"
    assert cfg.teacher.k == 3  # inherited from base
    assert cfg.student.lora.r == 8 and cfg.student.lora.dropout == 0.05  # partial nested override


def test_cli_override_and_hash_changes():
    base = load_config(ROOT / "configs/base.yaml")
    other = load_config(ROOT / "configs/base.yaml", ["student.model=Qwen/Qwen3-1.7B"])
    assert other.student.model == "Qwen/Qwen3-1.7B"
    assert base.section_hash("student") != other.section_hash("student")
    assert base.section_hash("teacher") == other.section_hash("teacher")


def test_unknown_key_fails_loudly():
    with pytest.raises(KeyError):
        load_config(ROOT / "configs/base.yaml", ["student.learning_rte=1e-4"])


def test_parse_override_types():
    assert parse_override("a.b=3") == {"a": {"b": 3}}
    assert parse_override("a=[1, 2]") == {"a": [1, 2]}
    assert parse_override("a=null") == {"a": None}
