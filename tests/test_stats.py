import math

import numpy as np
import pytest

from llms.stats import (
    delta_ci,
    delta_hat,
    disagreement_only,
    mcnemar,
    non_inferiority,
    paired_counts,
    paired_se,
    sample_size,
)

B, C, N = 20, 40, 1000


def test_delta_se_mcnemar_known_values():
    assert delta_hat(B, C, N) == pytest.approx(-0.02)
    assert paired_se(B, C, N) == pytest.approx(0.00772, abs=5e-6)
    assert mcnemar(B, C)["chi2"] == pytest.approx(6.67, abs=0.005)
    assert mcnemar(B, C)["method"] == "chi-square"


def test_ci_is_symmetric_around_delta():
    lo, hi = delta_ci(B, C, N)
    assert (lo + hi) / 2 == pytest.approx(-0.02)
    assert hi - lo == pytest.approx(2 * 1.959964 * paired_se(B, C, N))


def test_non_inferiority_known_values():
    switch = non_inferiority(B, C, N, margin=0.04)
    assert switch["z"] == pytest.approx(2.59, abs=0.005) and switch["verdict"] == "switch"
    stay = non_inferiority(B, C, N, margin=0.03)
    assert stay["z"] == pytest.approx(1.30, abs=0.01) and stay["verdict"] == "don't switch"


def test_sample_size_known_value():
    n = sample_size(p_d=0.06, delta_true=-0.02, margin=0.04, alpha=0.05, power=0.8)
    assert math.ceil(n) == 922


def test_sample_size_impossible_when_true_gap_exceeds_margin():
    assert sample_size(0.06, delta_true=-0.05, margin=0.04) == math.inf


def test_mcnemar_exact_for_few_discordant_pairs():
    res = mcnemar(3, 12)
    assert res["method"] == "exact binomial" and res["p_value"] == pytest.approx(0.03516, abs=1e-4)


def test_paired_counts_and_disagreement_only_equals_full():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 5, 2000)
    new = np.where(rng.random(2000) < 0.85, gold, rng.integers(0, 5, 2000))
    ref = np.where(rng.random(2000) < 0.88, gold, rng.integers(0, 5, 2000))
    pc = paired_counts(gold, new, ref)
    assert pc.n == 2000 and pc.a + pc.b == (new == gold).sum()
    res = disagreement_only(gold, new, ref)
    assert res["identical"] and res["delta_disagreements_only"] == res["delta_full_labels"]
    assert res["labels_needed"] == (new != ref).sum() < 2000
