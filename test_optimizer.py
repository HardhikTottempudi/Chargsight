import numpy as np
import pytest

from chargesight.baselines import random_siting, run_all_baselines
from chargesight.economics import (compare_plans, optimization_value,
                                   plan_economics, simple_projection)
from chargesight.evaluation import Evaluator, Evaluation
from chargesight.optimizer import HybridSitingOptimizer
from chargesight.problem import SitingProblem


@pytest.fixture(scope="module")
def demo():
    return SitingProblem.demo(n_demand=40, seed=0)


def test_plan_economics_hand_math():
    problem = SitingProblem.demo(n_demand=20, seed=0)
    ev = Evaluation(
        objective=0, capex=350_000, uncovered_cost=0, uncovered_kwh_day=0,
        served_kwh_day=1000, coverage=1.0, ports_total=5, sites_used=2,
        site_indices=np.array([0, 1]), ports_per_site=np.array([3, 2]),
        assigned_kwh=np.array([600, 400]), assignment=np.zeros(20, dtype=int),
        gini=0.2, mean_walk_m=150.0)
    e = plan_economics(problem, ev)
    assert e["revenue"] == pytest.approx(1000 * 0.42 * 365)
    assert e["energy_cost"] == pytest.approx(1000 * 0.20 * 365)
    assert e["om"] == pytest.approx(5 * 6000)
    assert e["ebitda"] == pytest.approx(153_300 - 73_000 - 30_000)
    assert e["payback_years"] == pytest.approx(350_000 / 50_300)
    assert e["utilization"] == pytest.approx(1000 / (5 * 900))
    assert e["ev_miles_per_year"] == pytest.approx(1000 * 365 / 0.30)
    assert e["co2_tons_per_year"] == pytest.approx(
        (1000 * 365 / 0.30) / 28.0 * 8.89 / 1000.0)


def test_compare_plans_and_value(demo):
    evaluator = Evaluator(demo)
    rng = np.random.default_rng(5)
    good = evaluator.evaluate(
        np.argsort(evaluator.reach.sum(axis=0))[-5:])
    bad = rng.choice(demo.n_cand, size=5, replace=False)
    bad_ev = evaluator.evaluate(bad)
    rows = compare_plans(demo, {"opt": good, "bad": bad_ev})
    assert rows[0]["ebitda_delta_vs_opt"] == 0
    assert rows[1]["ebitda_delta_vs_opt"] <= 0
    v = optimization_value(demo, good, bad_ev)
    assert v["extra_ebitda_per_year"] >= 0


def test_projection_math():
    problem = SitingProblem.demo(n_demand=20, seed=0)
    proj = simple_projection(problem, served_kwh_day=1000, ports=5, years=1,
                             growth=0.0, discount=0.0)
    assert proj["cash_flows"][0] == pytest.approx(
        1000 * 0.22 * 365 - 5 * 6000)
    assert proj["npv_operating"] == pytest.approx(proj["cash_flows"][0])


def test_optimizer_beats_random(demo):
    evaluator = Evaluator(demo)
    result = HybridSitingOptimizer(demo, n_stations=4, n_particles=10,
                                   n_iters=30, seed=0).solve()
    rand = random_siting(demo, evaluator, 4, seed=1, n_samples=10)
    mean_rand = np.mean([e.objective for e in rand])
    assert result.evaluation.objective <= mean_rand


def test_optimizer_history_and_frames(demo):
    result = HybridSitingOptimizer(demo, n_stations=3, n_particles=8,
                                   n_iters=20, seed=0,
                                   capture_frames=True).solve()
    h = result.history
    assert len(h["best_known_f"]) == 20
    assert len(h["pso_raw_f"]) == 20
    assert h["frames"].shape[0] >= 1
    assert h["frames"].shape[1:] == (8, 3, 2)
    # convergence curve of best-known plans is monotone non-increasing
    diffs = np.diff(h["best_known_f"])
    assert (diffs <= 1e-6).all()
    # final hard refinement never worsens the best known plan
    assert result.evaluation.objective <= h["best_known_f"].min() + 1e-6


def test_sites_within_candidates(demo):
    result = HybridSitingOptimizer(demo, n_stations=3, n_particles=8,
                                   n_iters=15, seed=2).solve()
    assert result.site_indices.max() < demo.n_cand
    assert len(result.site_indices) <= 3
