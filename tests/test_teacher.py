from pathlib import Path

import pandas as pd
import pytest

from llms.config import load_config
from llms.labels import load_labels
from llms.teacher import label as teacher_label
from llms.teacher.label import aggregate, label_items, plurality
from llms.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]


class ScriptedTeacher:
    """Returns scripted outputs per text and counts every request."""

    def __init__(self, script):
        self.script = {t: list(outs) for t, outs in script.items()}
        self.requests = []

    def generate(self, texts, n, seeds):
        self.requests += [(t, n) for t in texts]
        return [[self.script[t].pop(0) for _ in range(n)] for t in texts]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = load_config(ROOT / "configs/smoke.yaml", ["teacher.max_retries=1", "teacher.chunk_size=2"])
    ws = Workspace.from_config(cfg, tmp_path)
    labels = load_labels(ROOT / "configs/labels.yaml")
    df = pd.DataFrame({"uid": ["a", "b", "c"], "text": ["ta", "tb", "tc"], "gold": [11, 11, 41],
                       "split": ["train_pool", "train_pool", "dev"]})
    script = {
        "ta": ["card_arrival", "card_arrival", "card_linking"],                 # 2/3 agree
        "tb": ["nonsense", "card_arrival", "lost_or_stolen_card", "card_linking"],  # 1 retry fills the gap
        "tc": ["lost_or_stolen_card"] * 3,
    }
    fake = ScriptedTeacher(script)
    monkeypatch.setattr(teacher_label, "make_backend", lambda *a, **k: fake)
    return cfg, ws, labels, df, fake


def test_retry_plurality_and_filter(setup):
    cfg, ws, labels, df, fake = setup
    state, stats = label_items(cfg, ws, df, labels)
    lab = aggregate(df, state, k=3, min_agree=2).set_index("uid")
    assert lab.loc["a", "teacher_pred"] == labels.index("card_arrival") and lab.loc["a", "kept"]
    assert lab.loc["b", "n_valid"] == 3 and lab.loc["b", "agree"] == 1 and not lab.loc["b", "kept"]
    assert lab.loc["b", "first_round_invalid"] == 1
    assert ("tb", 1) in fake.requests  # the invalid sample was re-requested once
    assert stats["samples"] == 10


def test_cache_is_never_recalled(setup, monkeypatch):
    cfg, ws, labels, df, fake = setup
    label_items(cfg, ws, df, labels)

    def boom(*a, **k):
        raise AssertionError("teacher called again despite a complete cache")

    monkeypatch.setattr(teacher_label, "make_backend", boom)
    state, stats = label_items(cfg, ws, df, labels)
    assert stats["samples"] == 0 and len(state) == 3


def test_plurality_ties_break_to_earliest():
    assert plurality([5, 7, 9], k=3) == (5, 1)
    assert plurality([7, 5, 5], k=3) == (5, 2)
    assert plurality([], k=3) == (-1, 0)
