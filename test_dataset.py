import numpy as np
import pytest

from chargesight.problem import SitingProblem


@pytest.fixture(scope="module")
def manhattan():
    return SitingProblem.load_builtin("manhattan")


def test_builtin_loads(manhattan):
    assert manhattan.n_demand >= 200
    assert manhattan.n_cand >= 500
    assert manhattan.total_demand_kwh > 0


def test_real_city_bbox(manhattan):
    lat = manhattan.demand_latlon[:, 0]
    lon = manhattan.demand_latlon[:, 1]
    assert (lat > 40.69).all() and (lat < 40.89).all()
    assert (lon > -74.03).all() and (lon < -73.90).all()
    clat = manhattan.cand_latlon[:, 0]
    clon = manhattan.cand_latlon[:, 1]
    assert (clat > 40.69).all() and (clat < 40.89).all()
    assert (clon > -74.03).all() and (clon < -73.90).all()


def test_real_landmarks_present(manhattan):
    names = set(manhattan.demand_names)
    assert "Grand Central Terminal" in names
    assert "Columbia University" in names
    assert "Times Square" in names


def test_positive_weights(manhattan):
    assert (manhattan.demand_kwh > 0).all()


def test_demo_subset(manhattan):
    demo = SitingProblem.demo(n_demand=40, seed=0)
    assert demo.n_demand == 40
    assert demo.n_cand >= 10
    assert demo.total_demand_kwh > 0


def test_synthetic_problem():
    p = SitingProblem.synthetic(n_demand=50, n_cand=60, seed=1)
    assert p.n_demand == 50 and p.n_cand == 60
