import numpy as np
import pytest

from chargesight.pso import AdaptivePSO


def sphere(x):
    return float(np.sum(x ** 2))


def test_minimizes_sphere():
    bounds = np.array([[-10, 10], [-10, 10]])
    res = AdaptivePSO(bounds, n_particles=20, n_iters=40, seed=0).optimize(
        sphere, track_particles=False)
    assert res.best_f < 0.5
    assert res.evals == 20 * 41  # initial swarm + one evaluation per iteration


def test_positions_respect_bounds():
    bounds = np.array([[-2, -1], [5, 6]])
    res = AdaptivePSO(bounds, n_particles=12, n_iters=25, seed=3).optimize(
        sphere, track_particles=False)
    lo, hi = bounds[:, 0], bounds[:, 1]
    assert np.all(res.best_x >= lo - 1e-9)
    assert np.all(res.best_x <= hi + 1e-9)
    parts = res.history.get("particles")
    assert parts is None  # tracking disabled


def test_stagnation_triggers_restarts():
    bounds = np.array([[0, 1]])
    res = AdaptivePSO(bounds, n_particles=8, n_iters=20, seed=0,
                      patience=2, restart_frac=0.5).optimize(
        lambda x: 1.0, track_particles=False)   # flat -> no improvement ever
    assert res.restarts >= 1


def test_reflection_pulls_back_inside():
    pso = AdaptivePSO(np.array([[0, 1], [0, 1]]), n_particles=2, n_iters=1)
    X = np.array([[0.5, 0.5], [1.2, -0.2]])
    V = np.array([[0.0, 0.0], [0.1, -0.1]])
    X2, V2 = pso._reflect(X.copy(), V.copy())
    assert (X2 >= pso.lo - 1e-12).all() and (X2 <= pso.hi + 1e-12).all()
    assert V2[1, 0] <= 0  # x-velocity flipped after right-wall reflection
    assert V2[1, 1] >= 0


def test_velocity_clamped():
    pso = AdaptivePSO(np.array([[0, 1]]), n_particles=2, n_iters=1,
                      vmax_frac=0.1)
    V = np.array([[5.0], [-5.0]])
    Vc = pso._clip_velocity(V.copy())
    assert np.abs(Vc).max() <= 0.1 * 1.0 + 1e-12
