import numpy as np
import pytest

from chargesight.evaluation import Evaluator
from chargesight.problem import SitingProblem
from chargesight.refinement import refine_sites, repair_collisions


@pytest.fixture(scope="module")
def demo():
    return SitingProblem.demo(n_demand=40, seed=0)


def test_refinement_never_worsens(demo):
    evaluator = Evaluator(demo)
    rng = np.random.default_rng(0)
    sites = rng.choice(demo.n_cand, size=4, replace=False).tolist()
    before = evaluator.evaluate(sites)
    refined, after = refine_sites(evaluator, sites)
    assert after.objective <= before.objective + 1e-9
    assert len(set(refined)) == len(refined)


def test_refinement_feasible_output(demo):
    evaluator = Evaluator(demo)
    refined, ev = refine_sites(evaluator, [0, 5, 9])
    assert all(0 <= s < demo.n_cand for s in refined)
    assert ev.ports_per_site.max() <= demo.max_ports


def test_repair_collisions():
    assert repair_collisions(None, [3, 3, 1, 2, 1]) == [1, 2, 3]  # sorted unique


def test_refinement_improves_bad_plan(demo):
    evaluator = Evaluator(demo)
    # deliberately bad: 4 sites in a far corner of the island; steepest
    # descent over the full relocate neighborhood walks them to demand
    far = np.argsort(demo.cand_latlon[:, 0])[:4]
    before = evaluator.evaluate(far.tolist())
    refined, after = refine_sites(evaluator, far.tolist(), max_sweeps=6,
                                  max_radius_m=1e9, max_neighbors=None)
    assert after.objective < before.objective
