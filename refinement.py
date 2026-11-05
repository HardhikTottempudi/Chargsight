"""Discrete refinement layer of the hybrid solver.

The PSO flies in continuous lat/lon space; this layer snaps stations to
candidate sites and hill-climbs in the *discrete* space of site sets
(relocate a station to a nearby candidate while the objective improves),
then feeds the refined plan back into the swarm.
"""
from __future__ import annotations

import numpy as np

from .evaluation import Evaluator, Evaluation
from .problem import haversine_m


def refine_sites(evaluator: Evaluator, site_indices, max_radius_m=800.0,
                 max_sweeps=4, max_neighbors=20):
    """Greedy relocate hill-climb on the discrete site set.

    For each station, try moving it to any candidate within `max_radius_m`
    (not already occupied). Accept the first improvement; repeat until no
    improving move exists or `max_sweeps` sweeps are done. Never returns a
    worse plan than it was given.
    """
    current = list(dict.fromkeys(int(s) for s in site_indices))
    best = evaluator.evaluate(current)
    if current:
        cand = evaluator.problem.cand_latlon
        neighbor_cache = {}
        for sweeps in range(max_sweeps):
            improved = False
            for pos in range(len(current)):
                s = current[pos]
                if s not in neighbor_cache:
                    d = haversine_m(cand[s, 0], cand[s, 1],
                                    cand[:, 0], cand[:, 1])
                    nbrs = np.argsort(d)
                    nbrs = [int(c) for c in nbrs if d[c] <= max_radius_m]
                    if max_neighbors is not None:
                        nbrs = nbrs[:max_neighbors]
                    neighbor_cache[s] = nbrs
                for c in neighbor_cache[s]:
                    if c in current:
                        continue
                    trial = current.copy()
                    trial[pos] = c
                    ev = evaluator.evaluate(trial)
                    if ev.objective < best.objective - 1e-9:
                        best, current = ev, trial
                        improved = True
                        break               # accept and move to next station
            if not improved:
                break
    return current, best


def repair_collisions(evaluator: Evaluator, site_indices):
    """Ensure a unique site set (drop duplicates, keep first occurrences)."""
    seen, out = set(), set()
    for s in site_indices:
        s = int(s)
        if s not in seen:
            seen.add(s)
            out.add(s)
    return sorted(out)


def _marginal_gains(evaluator: Evaluator, current):
    """Newly-covered weighted demand (kWh/day) each candidate would add."""
    reach = evaluator.reach
    w = evaluator.problem.demand_kwh
    if len(current):
        covered = reach[:, current].any(axis=1)
    else:
        covered = np.zeros(reach.shape[0], dtype=bool)
    uncovered = ~covered
    if uncovered.any():
        return w[uncovered] @ reach[uncovered]
    return np.zeros(reach.shape[1])


def refine_plan(evaluator: Evaluator, site_indices, relocate_radius_m=800.0,
                relocate_sweeps=2, relocate_neighbors=15, exchange_top=8,
                exchange_passes=1, hard=False):
    """Full discrete refinement: local relocate + long-range exchange.

    - relocate: hill-climb each station over nearby candidates (fine tuning).
    - exchange (Teitz-Bart style vertex substitution): replace a station with
      any high-marginal-gain candidate anywhere on the map (escapes local
      optima that nearby-only moves cannot).

    `hard=True` runs a deeper schedule for the final polish. Never returns a
    worse plan than it was given.
    """
    if hard:
        relocate_sweeps, relocate_neighbors = 6, 25
        exchange_top, exchange_passes = 20, 3
    current = repair_collisions(evaluator, site_indices)
    best = evaluator.evaluate(current)
    if not current:
        return current, best

    current, best = refine_sites(evaluator, current,
                                 max_radius_m=relocate_radius_m,
                                 max_sweeps=relocate_sweeps,
                                 max_neighbors=relocate_neighbors)
    for _ in range(exchange_passes):
        improved = False
        gains = _marginal_gains(evaluator, current)
        candidates = np.argsort(gains)[::-1][:exchange_top + len(current)]
        candidates = [int(c) for c in candidates if c not in current]
        candidates = candidates[:exchange_top]
        if not candidates:
            break
        for pos in range(len(current)):
            base = current.copy()
            for c in candidates:
                trial = base.copy()
                trial[pos] = c
                ev = evaluator.evaluate(trial)
                if ev.objective < best.objective - 1e-9:
                    best, current, improved = ev, trial, True
                    break
        if not improved:
            break
        # re-tune locally after each exchange pass
        current, best = refine_sites(evaluator, current,
                                     max_radius_m=relocate_radius_m,
                                     max_sweeps=1,
                                     max_neighbors=relocate_neighbors)
    return current, best
