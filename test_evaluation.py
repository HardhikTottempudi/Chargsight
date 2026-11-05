import numpy as np
import pytest

from chargesight.evaluation import Evaluator
from chargesight.problem import SitingProblem


@pytest.fixture(scope="module")
def tiny():
    return SitingProblem.from_arrays(
        cand_latlon=[[40.7500, -73.9800], [40.7600, -73.9700]],
        demand_latlon=[[40.7501, -73.9801], [40.7601, -73.9701],
                       [40.7530, -73.9775]],
        demand_kwh=[500, 300, 200],
        demand_names=["near0", "near1", "between"],
        name="tiny")


def test_full_coverage_objective(tiny):
    ev = Evaluator(tiny).evaluate([0, 1])
    # cand0 serves 700 kWh (1 port), cand1 serves 300 kWh (1 port)
    assert ev.ports_total == 2
    assert ev.sites_used == 2
    assert ev.capex == pytest.approx(2 * 50_000 + 2 * 60_000)
    assert ev.coverage == pytest.approx(1.0)
    assert ev.uncovered_kwh_day == 0
    assert ev.objective == pytest.approx(ev.capex)


def test_capacity_forces_uncovered(tiny):
    p = SitingProblem.from_arrays(
        cand_latlon=[[40.7500, -73.9800], [40.7600, -73.9700]],
        demand_latlon=[[40.7501, -73.9801], [40.7601, -73.9701],
                       [40.7530, -73.9775]],
        demand_kwh=[500, 300, 200],
        port_kwh_day=300, max_ports=1, name="tiny_capped")
    ev = Evaluator(p).evaluate([0, 1])
    # 500 kWh point cannot fit the 300 kWh cap -> uncovered
    assert ev.uncovered_kwh_day == pytest.approx(500)
    assert ev.served_kwh_day == pytest.approx(500)
    assert ev.ports_total == 2
    rate = (p.price_per_kwh - p.elec_cost_per_kwh) * 365 * p.uncovered_years
    assert ev.objective == pytest.approx(220_000 + 500 * rate)


def test_empty_plan(tiny):
    ev = Evaluator(tiny).evaluate([])
    assert ev.coverage == 0.0
    assert ev.capex == 0.0
    assert ev.objective == pytest.approx(tiny.total_demand_kwh *
                                         tiny.uncovered_cost_rate())


def test_deterministic(tiny):
    e1 = Evaluator(tiny).evaluate([0, 1])
    e2 = Evaluator(tiny).evaluate([0, 1])
    assert e1.objective == e2.objective
    assert (e1.assignment == e2.assignment).all()


def test_snap_dedupes_collisions(tiny):
    evaluator = Evaluator(tiny)
    sites = evaluator.snap([[40.7500, -73.9800], [40.7500, -73.9800]])
    assert sites[0] != sites[1]
    assert len(set(sites.tolist())) == 2


def test_ports_never_exceed_cap(tiny):
    p = SitingProblem.from_arrays(
        cand_latlon=[[40.7500, -73.9800]],
        demand_latlon=[[40.7501, -73.9801]],
        demand_kwh=[90_000],
        port_kwh_day=900, max_ports=12, name="overcap")
    ev = Evaluator(p).evaluate([0])
    assert ev.ports_per_site.max() <= 12
