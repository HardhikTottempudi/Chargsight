"""Adaptive Particle Swarm Optimization.

Continuity with the original swarm_optimizer.py: adaptive inertia and
acceleration coefficients, velocity clamping, boundary reflection and
stagnation-triggered partial restarts, plus objective-evaluation tracking
and population-diversity metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PSOResult:
    best_x: np.ndarray
    best_f: float
    history: dict = field(default_factory=dict)   # best_f, gbest_x, particles, diversity
    restarts: int = 0
    evals: int = 0


class AdaptivePSO:
    """Minimize `objective_fn(x)` over a box, with adaptive schedules.

    - inertia w: linear w_start -> w_end
    - cognitive c1: c1_start -> c1_end (exploration fades)
    - social c2: c2_start -> c2_end (swarm consensus grows)
    - stagnation patience: after `patience` iterations without global
      improvement, the worst `restart_frac` of the swarm is re-seeded around
      the global best with a widening Gaussian kick (partial restart).
    """

    def __init__(self, bounds, n_particles=40, n_iters=250, seed=0,
                 w_start=0.9, w_end=0.35,
                 c1_start=2.5, c1_end=0.5,
                 c2_start=0.5, c2_end=2.5,
                 vmax_frac=0.30, patience=15, restart_frac=0.30):
        self.bounds = np.asarray(bounds, dtype=float)      # (d, 2) lo/hi
        self.d = self.bounds.shape[0]
        self.lo = self.bounds[:, 0]
        self.hi = self.bounds[:, 1]
        self.span = self.hi - self.lo
        self.vmax = vmax_frac * self.span
        self.n_particles = n_particles
        self.n_iters = n_iters
        self.seed = seed
        self.w_start, self.w_end = w_start, w_end
        self.c1_start, self.c1_end = c1_start, c1_end
        self.c2_start, self.c2_end = c2_start, c2_end
        self.patience = patience
        self.restart_frac = restart_frac

    # ------------------------------------------------------------ internals
    def _init_swarm(self, rng):
        X = self.lo + rng.random((self.n_particles, self.d)) * self.span
        V = rng.uniform(-0.1, 0.1, size=(self.n_particles, self.d)) * self.span
        return X, V

    def _clip_velocity(self, V):
        np.clip(V, -self.vmax, self.vmax, out=V)
        return V

    def _reflect(self, X, V):
        for _ in range(2):
            below = X < self.lo
            above = X > self.hi
            if not (below.any() or above.any()):
                break
            X = np.where(below, 2 * self.lo - X, X)
            X = np.where(above, 2 * self.hi - X, X)
            V = np.where(below, np.abs(V), V)
            V = np.where(above, -np.abs(V), V)
        return np.clip(X, self.lo, self.hi), V

    # ----------------------------------------------------------------- run
    def optimize(self, objective_fn, on_iteration=None, track_particles=True,
                 init_hook=None):
        rng = np.random.default_rng(self.seed)
        X, V = self._init_swarm(rng)
        if init_hook is not None:            # hybrid warm-start hook
            X, V = init_hook(X, V)
            X = np.clip(X, self.lo, self.hi)
        F = np.array([objective_fn(x) for x in X])
        evals = self.n_particles
        pbest_x = X.copy()
        pbest_f = F.copy()
        g = int(np.argmin(F))
        gbest_x, gbest_f = X[g].copy(), float(F[g])
        stagnation = 0
        restarts = 0

        hist_f = [gbest_f]
        hist_x = [gbest_x.copy()]
        hist_parts = [X.copy()] if track_particles else None
        hist_div = [float(np.mean(np.std(X, axis=0)))]

        for it in range(1, self.n_iters + 1):
            t = it / self.n_iters
            w = self.w_start + (self.w_end - self.w_start) * t
            c1 = self.c1_start + (self.c1_end - self.c1_start) * t
            c2 = self.c2_start + (self.c2_end - self.c2_start) * t
            r1 = rng.random((self.n_particles, self.d))
            r2 = rng.random((self.n_particles, self.d))
            V = w * V + c1 * r1 * (pbest_x - X) + c2 * r2 * (gbest_x[None, :] - X)
            self._clip_velocity(V)
            X = X + V
            X, V = self._reflect(X, V)
            F = np.array([objective_fn(x) for x in X])
            evals += self.n_particles

            improved_mask = F < pbest_f
            pbest_f[improved_mask] = F[improved_mask]
            pbest_x[improved_mask] = X[improved_mask]
            g = int(np.argmin(pbest_f))
            if pbest_f[g] < gbest_f - 1e-12:
                gbest_f = float(pbest_f[g])
                gbest_x = pbest_x[g].copy()
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= self.patience:                # partial restart
                worst = np.argsort(pbest_f)[-max(1, int(self.restart_frac *
                                                        self.n_particles)):]
                sigma = 0.08 * (1.0 + 0.5 * restarts) * self.span
                X[worst] = gbest_x[None, :] + rng.normal(
                    0.0, 1.0, size=(worst.size, self.d)) * sigma[None, :]
                X[worst] = np.clip(X[worst], self.lo, self.hi)
                V[worst] = 0.0
                stagnation = 0
                restarts += 1

            hist_f.append(gbest_f)
            hist_x.append(gbest_x.copy())
            if track_particles:
                hist_parts.append(X.copy())
            hist_div.append(float(np.mean(np.std(X, axis=0))))
            if on_iteration is not None:
                on_iteration(it, gbest_x.copy(), gbest_f, X.copy())

        history = dict(best_f=np.array(hist_f), gbest_x=np.array(hist_x),
                       diversity=np.array(hist_div))
        if track_particles:
            history["particles"] = np.array(hist_parts)
        return PSOResult(best_x=gbest_x, best_f=gbest_f, history=history,
                         restarts=restarts, evals=evals)
