"""Hybrid siting optimizer: adaptive PSO in continuous space + discrete
refinement injected back into the swarm.

Pipeline per refinement beat (every `refine_every` iterations):
  1. snap the swarm's global-best station positions to candidate sites,
  2. hill-climb the discrete site set (relocate moves),
  3. if the refined plan beats the best known plan, inject its coordinates
     back into a particle so the swarm can exploit the discrete gain.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .evaluation import Evaluator, Evaluation
from .pso import AdaptivePSO
from .problem import SitingProblem
from .refinement import refine_plan


@dataclass
class OptimizationResult:
    problem: SitingProblem
    evaluator: Evaluator
    site_indices: np.ndarray
    evaluation: Evaluation
    best_position: np.ndarray            # (K, 2) lat/lon of the final plan
    n_stations: int
    history: dict = field(default_factory=dict)
    restarts: int = 0
    evals: int = 0
    runtime_s: float = 0.0
    params: dict = field(default_factory=dict)


class HybridSitingOptimizer:
    """Solve the charging-station siting problem for a fixed number of
    stations K using hybrid adaptive PSO + discrete refinement."""

    def __init__(self, problem: SitingProblem, n_stations=12, n_particles=40,
                 n_iters=250, seed=0, refine_every=8, capture_frames=True,
                 max_frames=150, verbose=False):
        self.problem = problem
        self.k = int(n_stations)
        self.n_particles = int(n_particles)
        self.n_iters = int(n_iters)
        self.seed = int(seed)
        self.refine_every = max(1, int(refine_every))
        self.capture_frames = capture_frames
        self.max_frames = max_frames
        self.verbose = verbose

    # ------------------------------------------------------------------ run
    def solve(self, on_iteration=None) -> OptimizationResult:
        t0 = time.time()
        problem = self.problem
        evaluator = Evaluator(problem)
        (lat0, lat1), (lon0, lon1) = problem.bbox()
        pad = 0.004
        bounds = []
        for _ in range(self.k):
            bounds += [(lat0 - pad, lat1 + pad), (lon0 - pad, lon1 + pad)]
        bounds = np.array(bounds)

        best_known = {"sites": None, "eval": None}

        def remember(sites, ev):
            if best_known["eval"] is None or ev.objective < best_known["eval"].objective - 1e-9:
                best_known["sites"] = list(dict.fromkeys(int(s) for s in sites))
                best_known["eval"] = ev

        def objective_fn(x):
            positions = x.reshape(self.k, 2)
            sites = evaluator.snap(positions)
            ev = evaluator.evaluate(sites)
            remember(sites, ev)
            return ev.objective

        stride = max(1, self.n_iters // max(1, self.max_frames))
        frames = []
        frame_iters = []
        curve_f, curve_cov, curve_capex = [], [], []

        def internal_callback(it, gbest_x, gbest_f, X):
            curve_f.append(best_known["eval"].objective)
            curve_cov.append(best_known["eval"].coverage)
            curve_capex.append(best_known["eval"].capex)
            # capture animation frame
            if self.capture_frames and (it % stride == 0 or it == self.n_iters):
                frames.append(X.copy().reshape(self.n_particles, self.k, 2))
                frame_iters.append(it)
            # refinement beat: snap + relocate + long-range exchange
            if it % self.refine_every == 0 or it == self.n_iters:
                positions = gbest_x.reshape(self.k, 2)
                sites = evaluator.snap(positions)
                refined, ev = refine_plan(evaluator, sites,
                                          relocate_sweeps=1,
                                          relocate_neighbors=12,
                                          exchange_top=6, exchange_passes=1)
                remember(refined, ev)
                if best_known["eval"] is ev:
                    # inject refined plan into particle 0 for exploitation
                    coords = problem.cand_latlon[refined]
                    inject = np.zeros((self.k, 2))
                    if coords.shape[0] >= self.k:
                        inject = coords[:self.k]
                    else:                     # pad by repeating last site
                        inject[:coords.shape[0]] = coords
                        inject[coords.shape[0]:] = coords[-1]
                    X[0] = inject.reshape(-1)
            if on_iteration is not None:
                on_iteration(it, gbest_x, gbest_f, X)

        pso = AdaptivePSO(bounds, n_particles=self.n_particles,
                          n_iters=self.n_iters, seed=self.seed)

        # ---- hybrid warm start: seed three particles with constructive
        # heuristics (greedy coverage, k-means clusters, even spread); the
        # swarm explores around and beyond them.
        from . import baselines as bl
        seeds = []
        try:
            seeds.append(bl.greedy_coverage(problem, evaluator, self.k).site_indices)
            seeds.append(bl.kmeans_siting(problem, evaluator, self.k,
                                          seed=self.seed).site_indices)
            seeds.append(bl.spread_even(problem, evaluator, self.k).site_indices)
        except Exception:
            seeds = seeds or []

        def warm_start(X, V):
            rng = np.random.default_rng(self.seed + 99)
            for i, sites in enumerate(seeds[:len(X)]):
                coords = problem.cand_latlon[np.asarray(sites, dtype=int)]
                inject = np.zeros((self.k, 2))
                take = min(len(sites), self.k)
                inject[:take] = coords[:take]
                if take < self.k:
                    inject[take:] = coords[-1]
                X[i] = inject.reshape(-1)
                V[i] = 0.0
            return X, V

        pso_result = pso.optimize(objective_fn, on_iteration=internal_callback,
                                  init_hook=warm_start)

        # final hard refinement of the best known plan
        final_sites, final_eval = refine_plan(evaluator, best_known["sites"],
                                              hard=True)
        remember(final_sites, final_eval)
        best_eval = best_known["eval"]
        final_sites = np.array(sorted(best_known["sites"]), dtype=int)

        # rebuild convergence curves (best-known plan, re-evaluated cheaply
        # from stored objective values is not possible; we stored nothing per
        # iteration except PSO curve -> keep both)
        curve_f = np.array(curve_f)
        curve_cov = np.array(curve_cov)
        curve_capex = np.array(curve_capex)
        hist = dict(
            iters=np.arange(1, self.n_iters + 1),
            best_known_f=curve_f,
            best_known_coverage=curve_cov,
            best_known_capex=curve_capex,
            pso_raw_f=pso_result.history["best_f"][1:],
            gbest_x=pso_result.history["gbest_x"],
            diversity=pso_result.history["diversity"],
            frames=np.array(frames) if frames else np.zeros((0,)),
            frame_iters=np.array(frame_iters),
        )

        if self.verbose:
            print(f"  [chargesight] best objective ${best_eval.objective:,.0f} "
                  f"| coverage {best_eval.coverage:.1%} "
                  f"| restarts {pso_result.restarts} "
                  f"| evals {evaluator.evals} | {time.time()-t0:.1f}s")

        best_position = problem.cand_latlon[final_sites]
        return OptimizationResult(
            problem=problem, evaluator=evaluator,
            site_indices=final_sites, evaluation=best_eval,
            best_position=best_position, n_stations=self.k,
            history=hist, restarts=pso_result.restarts,
            evals=evaluator.evals, runtime_s=time.time() - t0,
            params=dict(n_stations=self.k, n_particles=self.n_particles,
                        n_iters=self.n_iters, seed=self.seed,
                        refine_every=self.refine_every))
