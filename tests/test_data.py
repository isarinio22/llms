import numpy as np
import pandas as pd
import pytest

from llms.data import normalize_text, remove_cross_split_duplicates, stratified_split

FRACTIONS = {"train_pool": 0.65, "unlabeled": 0.05, "dev": 0.10, "test": 0.20}


def test_stratified_split_is_proportional_and_deterministic():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 77, size=13000)
    a = stratified_split(labels, FRACTIONS, seed=1)
    b = stratified_split(labels, FRACTIONS, seed=1)
    assert (a == b).all()
    for name, frac in FRACTIONS.items():
        assert abs((a == name).mean() - frac) < 0.01
    # Per class: every split within +-1 item of its proportional share.
    for c in range(77):
        n_c = (labels == c).sum()
        for name, frac in FRACTIONS.items():
            assert abs(((a == name) & (labels == c)).sum() - frac * n_c) <= 1.0 + 1e-9


def test_stratified_split_handles_tiny_classes():
    labels = np.repeat(np.arange(77), 8)
    out = stratified_split(labels, FRACTIONS, seed=3)
    assert set(out) <= set(FRACTIONS)
    assert (out == "dev").sum() > 0 and (out == "test").sum() > 0


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        stratified_split(np.zeros(10), {"a": 0.5, "b": 0.4}, seed=0)


def test_normalize_text():
    assert normalize_text("  My CARD hasn't arrived!! ") == "my card hasn t arrived"


def test_dedup_protects_higher_priority_splits():
    df = pd.DataFrame({
        "uid": ["t1", "d1", "p1", "p2", "p3"],
        "text": ["Where is my card?", "where is my card", "Card still not here", "Totally new", "near dup"],
        "split": ["test", "dev", "train_pool", "train_pool", "train_pool"],
    })
    emb = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.999, 0.0447, 0]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    out = remove_cross_split_duplicates(df, emb, threshold=0.95)
    removed = dict(zip(out.uid, out.removed_reason, strict=True))
    assert not out.set_index("uid").loc["t1", "removed"]  # test is never touched
    assert removed["d1"] == "exact"  # dev duplicates test
    assert removed["p3"] == "embedding" and out.set_index("uid").loc["p3", "dup_of"] == "t1"
    assert not out.set_index("uid").loc[["p1", "p2"], "removed"].any()
