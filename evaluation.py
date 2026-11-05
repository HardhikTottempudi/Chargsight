"""Decoding continuous station positions into a feasible siting plan, and
evaluating it: coverage, capacity, capex and opportunity cost."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import SitingProblem


@dataclass
class Evaluation:
    """Full assessment of one candidate siting plan (a set of sites)."""
    objective: float                 # $ capex + $ opportunity cost of unserved demand
    capex: float                     # $ build cost of the plan
    uncovered_cost: float            # $ opportunity cost of unserved demand
    uncovered_kwh_day: float
    served_kwh_day: float
    coverage: float                  # weighted share of demand within reach (0..1)
    ports_total: int
    sites_used: int
    site_indices: np.ndarray         # candidate indices used (unique)
    ports_per_site: np.ndarray       # aligned with site_indices
    assigned_kwh: np.ndarray         # energy assigned to each site (kWh/day)
    assignment: np.ndarray           # per demand point: column in site_indices or -1
    gini: float                      # inequality of port allocation across sites
    mean_walk_m: float               # mean walk for covered demand

    def summary(self):
        return dict(objective=self.objective, capex=self.capex,
                    coverage=self.coverage, ports_total=self.ports_total,
                    sites_used=self.sites_used,
                    uncovered_kwh_day=self.uncovered_kwh_day,
                    gini=self.gini, mean_walk_m=self.mean_walk_m)


class Evaluator:
    """Capacity-aware decoder + evaluator shared by the optimizer, refinement
    and all baselines, so every number is comparable."""

    def __init__(self, problem: SitingProblem, use_cache=True):
        self.problem = problem
        self.D = problem.distances()                       # (n, m)
        self.reach = self.D <= problem.walk_radius_m       # (n, m) bool
        # Most-constrained demand first: fewest reachable candidates overall.
        scarcity = self.reach.sum(axis=1)
        self.order = np.argsort(scarcity, kind="stable")
        self.evals = 0
        self.cache_hits = 0
        self._cap = problem.max_ports * problem.port_kwh_day
        self._cache = {} if use_cache else None

    # ------------------------------------------------------------------ snap
    def snap(self, positions):
        """Map continuous (K, 2) lat/lon positions to unique candidate sites.

        Two stations falling on the same candidate are separated: the second
        one snaps to its nearest still-free candidate (so the swarm may park
        particles anywhere; the decoder keeps plans feasible).
        """
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        K = positions.shape[0]
        m = self.problem.n_cand
        cand = self.problem.cand_latlon
        d = haversine_pairs(positions, cand)               # (K, m)
        order = np.argsort(d, axis=1)
        taken = set()
        out = np.zeros(K, dtype=int)
        for i in range(K):
            for c in order[i]:
                if c not in taken:
                    out[i] = c
                    taken.add(int(c))
                    break
        return out

    # -------------------------------------------------------------- evaluate
    def evaluate(self, site_indices) -> Evaluation:
        self.evals += 1
        S = np.unique(np.asarray(site_indices, dtype=int))
        key = tuple(S.tolist())
        if self._cache is not None and key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        ev = self._evaluate_unique(S)
        if self._cache is not None and len(self._cache) < 200_000:
            self._cache[key] = ev
        return ev

    def _evaluate_unique(self, S) -> Evaluation:
        problem = self.problem
        n = problem.n_demand
        w = problem.demand_kwh

        if S.size == 0:
            return Evaluation(
                objective=w.sum() * problem.uncovered_cost_rate(),
                capex=0.0, uncovered_cost=w.sum() * problem.uncovered_cost_rate(),
                uncovered_kwh_day=float(w.sum()), served_kwh_day=0.0,
                coverage=0.0, ports_total=0, sites_used=0,
                site_indices=S, ports_per_site=np.zeros(0, dtype=int),
                assigned_kwh=np.zeros(0), assignment=np.full(n, -1),
                gini=0.0, mean_walk_m=0.0)

        D_open = self.D[:, S]                              # (n, k)
        open_reach = self.reach[:, S]                      # (n, k)
        k = S.size
        load = np.zeros(k)
        assignment = np.full(n, -1)
        cap = self._cap

        for i in self.order:                               # constrained first
            wi = w[i]
            feas = np.flatnonzero(open_reach[i] & (load + wi <= cap + 1e-9))
            if feas.size == 0:
                continue
            j = feas[np.argmin(D_open[i, feas])]
            assignment[i] = j
            load[j] += wi

        ports = np.ceil(load / problem.port_kwh_day).astype(int)
        used = ports > 0
        capex = float(used.sum() * problem.fixed_site_cost
                      + ports.sum() * problem.port_capex)
        uncovered = w[assignment < 0].sum()
        unc_cost = float(uncovered * problem.uncovered_cost_rate())
        total = problem.total_demand_kwh
        covered_mask = assignment >= 0
        served = float(w[covered_mask].sum())

        if used.any():
            gini = _gini(ports[used].astype(float))
        else:
            gini = 0.0
        if covered_mask.any():
            mean_walk = float(np.mean(D_open[covered_mask, assignment[covered_mask]]))
        else:
            mean_walk = 0.0

        return Evaluation(
            objective=capex + unc_cost,
            capex=capex, uncovered_cost=unc_cost,
            uncovered_kwh_day=float(uncovered), served_kwh_day=served,
            coverage=served / total if total > 0 else 0.0,
            ports_total=int(ports.sum()), sites_used=int(used.sum()),
            site_indices=S, ports_per_site=ports, assigned_kwh=load,
            assignment=assignment, gini=float(gini), mean_walk_m=mean_walk)


def haversine_pairs(A, B):
    """Pairwise haversine distances between rows of A (k,2 lat/lon) and B (m,2)."""
    lat1 = np.radians(np.asarray(A[:, 0]))[:, None]
    lon1 = np.radians(np.asarray(A[:, 1]))[:, None]
    lat2 = np.radians(np.asarray(B[:, 0]))[None, :]
    lon2 = np.radians(np.asarray(B[:, 1]))[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6_371_000.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)
