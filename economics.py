"""Investor-grade economics for a siting plan: P&L, payback, utilization,
CO2 displacement, and the value of optimization vs any baseline plan.

All money math lives here so CLI, report and tests share one source of truth.
Assumptions come from the SitingProblem (documented, configurable defaults).
"""
from __future__ import annotations

import numpy as np

from .evaluation import Evaluation
from .problem import SitingProblem

KWH_PER_EV_MILE = 0.30       # ~3.3 miles/kWh (compact EV, real-world mix)
GAS_MPG = 28.0               # fleet-average ICE economy
CO2_KG_PER_GALLON = 8.89     # EIA emission factor, gasoline


def plan_economics(problem: SitingProblem, ev: Evaluation) -> dict:
    served = ev.served_kwh_day
    revenue = served * problem.price_per_kwh * 365.0
    energy_cost = served * problem.elec_cost_per_kwh * 365.0
    om = ev.ports_total * problem.om_per_port_year
    ebitda = revenue - energy_cost - om
    payback = float("inf") if ebitda <= 0 else ev.capex / ebitda
    ev_miles = served * 365.0 / KWH_PER_EV_MILE
    gallons = ev_miles / GAS_MPG
    co2_tons = gallons * CO2_KG_PER_GALLON / 1000.0
    capacity = ev.ports_total * problem.port_kwh_day
    utilization = served / capacity if capacity > 0 else 0.0
    return dict(
        capex=ev.capex,
        revenue=revenue,
        energy_cost=energy_cost,
        om=om,
        ebitda=ebitda,
        payback_years=payback,
        utilization=utilization,
        coverage=ev.coverage,
        served_kwh_day=served,
        ports=ev.ports_total,
        sites=ev.sites_used,
        ev_miles_per_year=ev_miles,
        co2_tons_per_year=co2_tons,
        opportunity_cost=ev.uncovered_cost,
        total_cost_of_plan=ev.objective,
    )


def compare_plans(problem: SitingProblem, plans: dict) -> list:
    """plans: name -> Evaluation. Returns rows sorted by objective (best last
    for display; we return in given order plus economics)."""
    rows = []
    opt_econ = None
    for name, ev in plans.items():
        econ = plan_economics(problem, ev)
        if opt_econ is None:
            opt_econ = econ
        econ = dict(econ)
        econ["plan"] = name
        econ["ebitda_delta_vs_opt"] = econ["ebitda"] - opt_econ["ebitda"]
        econ["payback_delta_vs_opt"] = (
            econ["payback_years"] - opt_econ["payback_years"]
            if np.isfinite(econ["payback_years"]) and np.isfinite(opt_econ["payback_years"])
            else float("inf"))
        rows.append(econ)
    return rows


def optimization_value(problem: SitingProblem, ev_opt: Evaluation,
                       ev_baseline: Evaluation) -> dict:
    """Headline value of optimization vs one baseline plan."""
    e_opt = plan_economics(problem, ev_opt)
    e_base = plan_economics(problem, ev_baseline)
    return dict(
        extra_ebitda_per_year=e_opt["ebitda"] - e_base["ebitda"],
        extra_coverage=e_opt["coverage"] - e_base["coverage"],
        extra_served_kwh_day=e_opt["served_kwh_day"] - e_base["served_kwh_day"],
        capex_delta=e_opt["capex"] - e_base["capex"],
        utilization_gain=e_opt["utilization"] - e_base["utilization"],
        payback_improvement_years=(
            (e_base["payback_years"] - e_opt["payback_years"])
            if np.isfinite(e_base["payback_years"]) and np.isfinite(e_opt["payback_years"])
            else float("inf")),
        extra_ev_miles_per_year=e_opt["ev_miles_per_year"] - e_base["ev_miles_per_year"],
        extra_co2_tons_per_year=e_opt["co2_tons_per_year"] - e_base["co2_tons_per_year"],
    )


def simple_projection(problem: SitingProblem, served_kwh_day, ports, years=5,
                      growth=0.10, discount=0.08):
    """Forward NPV of one plan with demand growth (simple, transparent)."""
    cf = []
    for y in range(years):
        factor = (1 + growth) ** y
        rev = served_kwh_day * factor * problem.price_per_kwh * 365.0
        cost = (served_kwh_day * factor * problem.elec_cost_per_kwh * 365.0
                + ports * problem.om_per_port_year)
        cf.append(rev - cost)
    npv = -0.0  # capex handled separately by callers
    for y, c in enumerate(cf, start=1):
        npv += c / (1 + discount) ** y
    return dict(cash_flows=cf, npv_operating=float(npv))
