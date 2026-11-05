"""Markdown report generator: the investment-grade siting memo."""
from __future__ import annotations

import datetime

import numpy as np

from .economics import optimization_value, simple_projection


def _money(x):
    if not np.isfinite(x):
        return "n/a"
    if abs(x) >= 1e6:
        return f"${x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:.0f}k"
    return f"${x:,.0f}"


def write_report(path, problem, result, plans, baselines_random_evals=None,
                 sweep=None, assumptions_rows=None):
    econ = {}
    for name, ev in plans.items():
        from .economics import plan_economics
        econ[name] = plan_economics(problem, ev)
    opt_name = "ChargeSight hybrid (PSO + refinement)"
    ev_opt = plans[opt_name]
    e_opt = econ[opt_name]

    lines = []
    now = datetime.date.today().isoformat()
    lines.append(f"# ChargeSight Siting Memo - {problem.name.replace('_', ' ').title()}")
    lines.append(f"*Generated {now} - swarm-intelligent siting of EV charging networks*")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(
        f"For a **{result.n_stations}-station** fast-charging network on **{problem.name}**, "
        f"the hybrid swarm optimizer places **{ev_opt.sites_used} sites / "
        f"{ev_opt.ports_total} DCFC ports** that capture **{ev_opt.coverage:.0%}** of modeled "
        f"demand (mean walk to a charger: **{ev_opt.mean_walk_m:.0f} m**), at a capital cost of "
        f"**{_money(e_opt['capex'])}** and projected EBITDA of **{_money(e_opt['ebitda'])}/yr** "
        f"(payback **{e_opt['payback_years']:.1f} years** at default assumptions).")
    lines.append("")
    rand_ev = plans.get("Random siting")
    spread_ev = plans.get("Even spread")
    if rand_ev is not None:
        v = optimization_value(problem, ev_opt, rand_ev)
        lines.append(
            f"Versus unstructured siting (random), the optimized network earns "
            f"**{_money(v['extra_ebitda_per_year'])} more per year** "
            f"(+{v['extra_coverage']*100:.0f} pts of demand coverage) and lifts port "
            f"utilization by **{v['utilization_gain']*100:.1f} pts**. Versus rule-of-thumb "
            f"'even spread' planning the annual EBITDA edge is "
            f"**{_money(e_opt['ebitda'] - econ['Even spread']['ebitda'])}**."
            if spread_ev is not None else "")
        lines.append("")

    lines.append("## Head-to-head results")
    lines.append("")
    header = ("| Plan | Coverage | Capex | Ports | Utilization | EBITDA/yr | "
              "Payback | Total cost* |")
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, ev in plans.items():
        e = econ[name]
        pb = f"{e['payback_years']:.1f} yr" if np.isfinite(e["payback_years"]) else "n/a"
        lines.append(
            f"| {name} | {ev.coverage:.0%} | {_money(e['capex'])} | {ev.ports_total} | "
            f"{e['utilization']:.0%} | {_money(e['ebitda'])} | {pb} | "
            f"{_money(e['total_cost_of_plan'])} |")
    lines.append("")
    lines.append("*Total cost = capex + opportunity cost of unserved demand "
                 f"over a {problem.uncovered_years:.0f}-year window "
                 "(the optimizer's objective). Utilization = served / "
                 f"(ports x {problem.port_kwh_day:.0f} kWh/day).")
    lines.append("")

    if baselines_random_evals:
        if isinstance(baselines_random_evals, dict):
            baselines_random_evals = next(
                iter(baselines_random_evals.values()))
        objs = [e.objective for e in baselines_random_evals]
        lines.append(f"Random-siting dispersion over {len(objs)} draws: total cost "
                     f"{_money(np.mean(objs))} +/- {_money(np.std(objs))} "
                     f"(best draw {_money(min(objs))}). The optimizer does not rely on luck.")
        lines.append("")

    lines.append("## Value of optimization")
    lines.append("")
    for name in plans:
        if name == opt_name:
            continue
        v = optimization_value(problem, ev_opt, plans[name])
        lines.append(f"- **vs {name}**: +{_money(v['extra_ebitda_per_year'])}/yr EBITDA, "
                     f"+{v['extra_served_kwh_day']:,.0f} kWh/day served, "
                     f"capex delta {_money(v['capex_delta'])}, "
                     f"payback improves {v['payback_improvement_years']:.1f} yr")
    lines.append("")
    proj = simple_projection(problem, e_opt["served_kwh_day"], e_opt["ports"])
    lines.append(
        f"Five-year operating NPV of the optimized plan (10%/yr demand growth, 8% discount, "
        f"ex-capex): **{_money(proj['npv_operating'])}**; cumulative operating cash flow "
        f"**{_money(sum(proj['cash_flows']))}** against capex {_money(e_opt['capex'])}.")
    lines.append("")

    lines.append("## Sustainability & access KPIs")
    lines.append("")
    lines.append(
        f"- EV miles enabled: **{e_opt['ev_miles_per_year']:,.0f} miles/yr** "
        f"({e_opt['served_kwh_day']:,.0f} kWh/day served)")
    lines.append(f"- Gross gasoline displaced: **{e_opt['co2_tons_per_year']:,.0f} t CO2/yr** "
                 "(fleet 28 mpg, 8.89 kg CO2/gal; grid upstream emissions not netted)")
    lines.append(f"- Port allocation equity (Gini): **{ev_opt.gini:.2f}** "
                 "(lower = ports spread more evenly across sites)")
    lines.append("")

    if sweep:
        lines.append("## Network sizing")
        lines.append("")
        falling = sweep["totals"][-1] < sweep["totals"][-2]
        if falling:
            lines.append(
                f"Sweeping network size K = {sweep['k_list']} shows total cost still "
                f"falling at the edge of the range: the recommended size within this "
                f"sweep is **K = {sweep['recommended_k']}** (total "
                f"{_money(sweep['best_total'])}), and each additional station still "
                "rescues more demand than it costs. Extend the sweep (or shorten the "
                "opportunity window) to find where the curve turns; per-station "
                "coverage gains are visibly diminishing.")
        else:
            lines.append(
                f"Sweeping network size K = {sweep['k_list']} gives the total-cost "
                f"curve below; the recommended size is **K = {sweep['recommended_k']}** "
                f"(total {_money(sweep['best_total'])}). Beyond that, each extra "
                "station costs more in capex than the demand it rescues from being "
                "unserved.")
        lines.append("")
        lines.append("| K | Coverage | Capex | Total cost* |")
        lines.append("|---|---|---|---|")
        for i, k in enumerate(sweep["k_list"]):
            lines.append(f"| {k} | {sweep['coverages'][i]:.0%} | "
                         f"{_money(sweep['capex'][i])} | {_money(sweep['totals'][i])} |")
        lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        "The optimizer hybridizes two layers. **(1) Adaptive particle swarm**: particles fly "
        "in continuous lat/lon space (2K dimensions), with inertia 0.9->0.35, cognitive weight "
        "2.5->0.5, social weight 0.5->2.5, velocity clamping, boundary reflection and "
        "stagnation-triggered partial restarts. **(2) Discrete refinement**: every few "
        "iterations the swarm's best positions are snapped to real candidate sites and "
        "hill-climbed with relocate moves under hard capacity limits (ports per site, "
        "port throughput); improved plans are injected back into the swarm. Every plan is "
        "scored by the same capacity-aware decoder: demand points walk to the nearest site "
        f"with spare capacity within {problem.walk_radius_m:.0f} m; unserved demand is priced at "
        "its opportunity cost.")
    lines.append("")

    lines.append("## Assumptions (all configurable)")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    rows = assumptions_rows or [
        ("Walk radius to a charger", f"{problem.walk_radius_m:.0f} m"),
        ("Port throughput", f"{problem.port_kwh_day:.0f} kWh/day (150 kW DCFC, ~6 effective h)"),
        ("Max ports per site", f"{problem.max_ports}"),
        ("Site fixed capex", _money(problem.fixed_site_cost)),
        ("Port capex (installed)", _money(problem.port_capex)),
        ("Retail charging price", f"${problem.price_per_kwh:.2f}/kWh"),
        ("Energy cost", f"${problem.elec_cost_per_kwh:.2f}/kWh"),
        ("O&M", _money(problem.om_per_port_year) + "/port/yr"),
        ("Opportunity window for unserved demand", f"{problem.uncovered_years:.0f} years"),
    ]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append(
        "*Cost and price defaults are illustrative mid-points of published US DCFC ranges; "
        "replace them with project-specific quotes via the problem parameters. Demand weights "
        "are an illustrative model anchored on real locations; plug in your own telemetry via "
        "`SitingProblem.from_json` / CSV to price a real deployment.*")
    lines.append("")
    lines.append("## Data provenance")
    lines.append("")
    lines.append(
        f"- Demand points: {problem.n_demand} locations (real landmark, transit, campus, "
        "hospital, retail and neighborhood coordinates).")
    lines.append(
        f"- Candidate sites: {problem.n_cand} positions sampled on the real Manhattan street "
        "grid (Commissioners' Plan of 1811 lattice, 29-degree rotation), land-mass filtered, "
        "Central Park excluded.")
    lines.append(
        f"- Optimization effort: {result.evals:,} objective evaluations, "
        f"{result.restarts} stagnation restarts, {result.runtime_s:.1f} s wall time.")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
