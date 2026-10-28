import numpy as np
import pandas as pd

from llms.cascade import costs, coverage_curve, fit_bins, route
from llms.config import CascadeCfg


def _frame(conf, student_ok, teacher_ok, hard=None):
    n = len(conf)
    return pd.DataFrame({"conf": conf, "student_ok": student_ok, "teacher_ok": teacher_ok,
                         "hard_route": np.zeros(n, bool) if hard is None else hard})


def test_cost_rule_routes_exactly_where_saving_beats_call_cost():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 5000)
    dev = _frame(conf, rng.random(5000) < conf, np.ones(5000, bool))  # teacher always right: p_big = 1
    edges, bins = fit_bins(dev, n_bins=10)
    cc = CascadeCfg(error_cost=1.0, teacher_call_cost=0.2)
    routed = route(dev, edges, bins, cc)
    # (1 - conf) * 1 > 0.2  <=>  conf < 0.8
    assert (routed == (conf < 0.8)).all()


def test_hard_rules_always_route():
    dev = _frame(np.full(4, 0.99), np.ones(4, bool), np.ones(4, bool), hard=np.array([True, False, False, True]))
    edges, bins = fit_bins(dev, n_bins=2)
    routed = route(dev, edges, bins, CascadeCfg(error_cost=1.0, teacher_call_cost=10.0))
    assert routed.tolist() == [True, False, False, True]


def test_coverage_curve_endpoints():
    df = _frame(np.linspace(0, 1, 11), np.ones(11, bool), np.zeros(11, bool))
    curve = coverage_curve(df)
    assert curve[0]["coverage"] == 1.0 and curve[0]["cascade_accuracy"] == 1.0
    assert curve[-1]["coverage"] < 0.2


def test_costs():
    c = costs(n=1000, errors=50, teacher_calls=200, cc=CascadeCfg(error_cost=2.0, teacher_call_cost=0.01))
    assert c["total_cost"] == 102.0 and c["cost_per_1k_tickets"] == 102.0
