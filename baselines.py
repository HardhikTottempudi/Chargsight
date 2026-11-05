"""Baseline siting strategies the optimizer is benchmarked against.

- greedy_coverage: classic greedy max-coverage (what a GIS workflow approximates)
- kmeans_siting:   cluster demand, build at cluster centroids (data-science baseline)
- spread_even:     farthest-point spread (rule-of-thumb "cover the map" planning)
- random_siting:   random site draws (lower bound on competence)
"""
from __future__ import annotations

import numpy as np

from .evaluation import Evaluator, Evaluation
from .problem import SitingProblem


def greedy_coverage(problem: SitingProblem, evaluator: Evaluator, k):
    reach = evaluator.reach          # (n, m)
    w = problem.demand_kwh
    covered = np.zeros(problem.n_demand, dtype=bool)
    sites = []
    for _ in range(int(k)):
        marginal = reach[~covered].sum(axis=0)
        marginal[sites] = -1.0
        best = int(np.argmax(marginal))
        if marginal[best] <= 0:
            break
        sites.append(best)
        covered |= reach[:, best]
    return evaluator.evaluate(sites)


def _kmeans_weighted(points, weights, k, rng, iters=40):
    """Weighted k-means with k-means++ seeding. Returns (k, 2) centroids."""
    n = points.shape[0]
    first = int(rng.choice(n, p=weights / weights.sum()))
    centroids = [points[first]]
    d2 = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, k):
        probs = d2 * weights
        total = probs.sum()
        if total <= 0:
            nxt = int(rng.integers(n))
        else:
            nxt = int(rng.choice(n, p=probs / total))
        centroids.append(points[nxt])
        d2 = np.minimum(d2, np.sum((points - points[nxt]) ** 2, axis=1))
    C = np.array(centroids, dtype=float)
    for _ in range(iters):
        d = np.sum((points[:, None, :] - C[None, :, :]) ** 2, axis=2)
        lab = np.argmin(d, axis=1)
        newC = C.copy()
        for j in range(k):
            mask = lab == j
            if mask.any():
                newC[j] = np.average(points[mask], axis=0, weights=weights[mask])
            else:                     # re-seed empty cluster at heaviest point
                newC[j] = points[int(np.argmax(weights * (d.min(axis=1) > 0))) ]
        if np.allclose(newC, C):
            break
        C = newC
    return C


def kmeans_siting(problem: SitingProblem, evaluator: Evaluator, k, seed=0,
                  restarts=3):
    best_ev, best_sites = None, None
    for r in range(restarts):
        rng = np.random.default_rng(seed + 1000 * r)
        C = _kmeans_weighted(problem.demand_latlon, problem.demand_kwh,
                             int(k), rng)
        sites = evaluator.snap(C)
        ev = evaluator.evaluate(sites)
        if best_ev is None or ev.objective < best_ev.objective:
            best_ev, best_sites = ev, sites
    return best_ev


def spread_even(problem: SitingProblem, evaluator: Evaluator, k):
    """Farthest-point sampling among candidates from the demand-weighted
    centroid: the 'cover the whole map evenly' rule of thumb."""
    cand = problem.cand_latlon
    centroid = np.average(problem.demand_latlon, axis=0,
                          weights=problem.demand_kwh)
    d = np.hypot(cand[:, 0] - centroid[0], cand[:, 1] - centroid[1])
    sites = [int(np.argmin(d))]
    min_d = np.hypot(cand[:, 0] - cand[sites[0], 0],
                     cand[:, 1] - cand[sites[0], 1])
    while len(sites) < int(k):
        nxt = int(np.argmax(min_d))
        sites.append(nxt)
        nd = np.hypot(cand[:, 0] - cand[nxt, 0], cand[:, 1] - cand[nxt, 1])
        min_d = np.minimum(min_d, nd)
    return evaluator.evaluate(sites)


def random_siting(problem: SitingProblem, evaluator: Evaluator, k, seed=0,
                  n_samples=20):
    rng = np.random.default_rng(seed)
    evals = []
    m = problem.n_cand
    for _ in range(int(n_samples)):
        sites = rng.choice(m, size=min(int(k), m), replace=False)
        evals.append(evaluator.evaluate(sites))
    return evals


def run_all_baselines(problem: SitingProblem, evaluator: Evaluator, k, seed=0):
    """Returns dict name -> Evaluation (random aggregated by mean objective)."""
    rand_evals = random_siting(problem, evaluator, k, seed=seed)
    rand_obj = np.mean([e.objective for e in rand_evals])
    rand = min(rand_evals, key=lambda e: e.objective)  # representative plan
    return {
        "Greedy coverage": greedy_coverage(problem, evaluator, k),
        "K-means clusters": kmeans_siting(problem, evaluator, k, seed=seed),
        "Even spread": spread_even(problem, evaluator, k),
        "Random siting": rand,
    }, {"Random siting": rand_evals}
