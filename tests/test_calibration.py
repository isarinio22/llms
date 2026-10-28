import numpy as np
import pytest
from scipy.special import softmax

from llms.calibration import apply_temperature, ece, fit_temperature


def test_ece_of_perfectly_calibrated_toy_set_is_zero():
    # In each confidence level, exactly that fraction of predictions is correct.
    probs, gold = [], []
    for conf, n in ((0.3, 10), (0.6, 20), (0.9, 30)):
        n_right = round(conf * n)
        for i in range(n):
            p = np.full(4, (1 - conf) / 3)
            p[0] = conf
            probs.append(p)
            gold.append(0 if i < n_right else 1)
    assert ece(np.array(probs), np.array(gold), n_bins=15) == pytest.approx(0.0, abs=1e-9)


def test_ece_of_overconfident_set():
    probs = np.tile([0.99, 0.01], (100, 1))
    gold = np.array([0] * 50 + [1] * 50)
    assert ece(probs, gold) == pytest.approx(0.49)


def test_temperature_scaling_recovers_true_temperature():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(5000, 10)) * 2
    gold = np.array([rng.choice(10, p=softmax(row)) for row in true_logits])
    t = fit_temperature(true_logits * 3.0, gold)  # model is 3x overconfident
    assert t == pytest.approx(3.0, rel=0.1)
    assert np.allclose(apply_temperature(true_logits * 3.0, t).sum(1), 1.0)
