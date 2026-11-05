"""ChargeSight command line interface.

    chargesight demo        end-to-end showcase (solve + benchmarks + sweep + report)
    chargesight solve       optimize one network plan
    chargesight bench       optimizer vs greedy / k-means / even spread / random
    chargesight sweep       network sizing curve (how many stations?)
    chargesight calculator  quick per-plan P&L and payback math
    chargesight report      solve + benchmarks + full markdown memo
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .baselines import run_all_baselines
from .economics import plan_economics, optimization_value, simple_projection
from .evaluation import Evaluator
from .optimizer import HybridSitingOptimizer
from .problem import SitingProblem
from .report import write_report
from . import viz


def _money(x):
    if not np.isfinite(x):
        return "n/a"
    if abs(x) >= 1e6:
        return f"${x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:.0f}k"
    return f"${x:,.0f}"


def _print_row(label, ev, econ):
    pb = f"{econ['payback_years']:5.1f}y" if np.isfinite(econ["payback_years"]) else "  n/a"
    print(f"  {label:<28} cov {ev.coverage:6.1%} | capex {_money(econ['capex']):>9} "
          f"| {ev.ports_total:>3} ports | util {econ['utilization']:5.1%} "
          f"| EBITDA {_money(econ['ebitda']):>8}/yr | payback {pb} "
          f"| total {_money(econ['total_cost_of_plan']):>9}")


def _solve(problem, args, verbose=True):
    opt = HybridSitingOptimizer(
        problem, n_stations=args.stations, n_particles=args.particles,
        n_iters=args.iterations, seed=args.seed,
        refine_every=args.refine_every, verbose=verbose)
    return opt.solve()


def _write_plan_csv(path, problem, ev):
    with open(path, "w") as f:
        f.write("site_id,lat,lon,ports,assigned_kwh_day\n")
        for s, ports, kwh in zip(ev.site_indices, ev.ports_per_site,
                                 ev.assigned_kwh):
            if ports <= 0:
                continue
            lat, lon = problem.cand_latlon[s]
            f.write(f"{s},{lat:.6f},{lon:.6f},{ports},{kwh:.1f}\n")


def _kpis_dict(problem, result, plans):
    e_opt = plan_economics(problem, result.evaluation)
    kpis = dict(
        city=problem.name, n_stations=result.n_stations,
        objective=result.evaluation.objective,
        coverage=result.evaluation.coverage,
        capex=result.evaluation.capex,
        ports=result.evaluation.ports_total,
        sites=result.evaluation.sites_used,
        utilization=e_opt["utilization"], ebitda=e_opt["ebitda"],
        payback_years=e_opt["payback_years"],
        ev_miles_per_year=e_opt["ev_miles_per_year"],
        co2_tons_per_year=e_opt["co2_tons_per_year"],
        evals=result.evals, restarts=result.restarts,
        runtime_s=result.runtime_s,
        gini=result.evaluation.gini,
        mean_walk_m=result.evaluation.mean_walk_m,
    )
    for name, ev in plans.items():
        v = optimization_value(problem, result.evaluation, ev)
        kpis[f"uplift_vs_{name.lower().replace(' ', '_')}"] = dict(
            extra_ebitda_per_year=v["extra_ebitda_per_year"],
            extra_coverage=v["extra_coverage"],
            extra_served_kwh_day=v["extra_served_kwh_day"])
    return kpis


# ----------------------------------------------------------------- commands
def cmd_solve(args):
    problem = _load_problem(args)
    print(f"Charging {problem.total_demand_kwh/1000:.1f} MWh/day of demand across "
          f"{problem.n_demand} points with {problem.n_cand} candidate sites "
          f"({problem.name}).")
    result = _solve(problem, args)
    ev = result.evaluation
    econ = plan_economics(problem, ev)
    _print_row(f"Optimized (K={result.n_stations})", ev, econ)
    print(f"  engine: {result.evals:,} evals | {result.restarts} restarts | "
          f"{result.runtime_s:.1f}s | Gini {ev.gini:.2f} | "
          f"mean walk {ev.mean_walk_m:.0f} m")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    _write_plan_csv(os.path.join(outdir, "plan.csv"), problem, ev)
    with open(os.path.join(outdir, "kpis.json"), "w") as f:
        json.dump(_kpis_dict(problem, result, {}), f, indent=2, default=float)

    if args.map:
        ev_rand = None
        evaluator = result.evaluator
        rand = evaluator.evaluate(
            np.random.default_rng(args.seed + 7).choice(
                problem.n_cand, size=args.stations, replace=False))
        viz.plot_map_before_after(
            problem, rand, ev, os.path.join(outdir, "map_before_after.png"),
            label_before="Before: random siting",
            label_after=f"After: ChargeSight (K={result.n_stations})")
        print(f"  wrote {os.path.join(outdir, 'map_before_after.png')}")
    if args.convergence:
        baselines, _ = run_all_baselines(problem, result.evaluator,
                                         args.stations, seed=args.seed)
        viz.plot_convergence(result, baselines,
                             os.path.join(outdir, "convergence.png"))
        print(f"  wrote {os.path.join(outdir, 'convergence.png')}")
    if args.gif:
        viz.animate_optimization(problem, result,
                                 os.path.join(outdir, "siting_animation.gif"))
        print(f"  wrote {os.path.join(outdir, 'siting_animation.gif')}")
    return result


def cmd_bench(args):
    problem = _load_problem(args)
    result = _solve(problem, args)
    evaluator = result.evaluator
    baselines, rand_evals = run_all_baselines(problem, evaluator,
                                              args.stations, seed=args.seed)
    plans = {"ChargeSight hybrid (PSO + refinement)": result.evaluation,
             **baselines}
    print(f"\nBenchmark - {problem.name}, K={args.stations} "
          f"({result.evals:,} optimizer evals, {result.runtime_s:.1f}s):")
    for name, ev in plans.items():
        _print_row(name, ev, plan_economics(problem, ev))

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    viz.plot_benchmark_panel(problem, plans, result,
                             os.path.join(outdir, "benchmark_panel.png"))
    viz.plot_convergence(result, baselines,
                         os.path.join(outdir, "convergence.png"))
    with open(os.path.join(outdir, "bench.csv"), "w") as f:
        f.write("plan,coverage,capex,ports,sites,utilization,ebitda,"
                "payback_years,total_cost\n")
        for name, ev in plans.items():
            e = plan_economics(problem, ev)
            f.write(f"\"{name}\",{ev.coverage:.4f},{e['capex']:.0f},"
                    f"{ev.ports_total},{ev.sites_used},{e['utilization']:.4f},"
                    f"{e['ebitda']:.0f},{e['payback_years']:.2f},"
                    f"{e['total_cost_of_plan']:.0f}\n")
    print(f"\n  wrote {os.path.join(outdir, 'benchmark_panel.png')}, "
          f"convergence.png, bench.csv")
    return result, plans


def cmd_sweep(args):
    problem = _load_problem(args)
    k_list = [int(k) for k in args.klist.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    sweep = _run_sweep(problem, k_list, args)
    viz.plot_sweep(k_list, sweep["totals"], sweep["capex"],
                   sweep["uncovered"], os.path.join(args.outdir,
                                                    "network_sweep.png"),
                   sweep["recommended_k"])
    print(f"\n  wrote {os.path.join(args.outdir, 'network_sweep.png')} "
          f"(recommended K = {sweep['recommended_k']})")
    return sweep


def _run_sweep(problem, k_list, args):
    totals, capex, uncovered, coverages = [], [], [], []
    for k in k_list:
        opt = HybridSitingOptimizer(problem, n_stations=k,
                                    n_particles=args.particles,
                                    n_iters=args.iterations, seed=args.seed,
                                    refine_every=args.refine_every,
                                    capture_frames=False)
        res = opt.solve()
        ev = res.evaluation
        totals.append(ev.objective)
        capex.append(ev.capex)
        uncovered.append(ev.uncovered_cost)
        coverages.append(ev.coverage)
        print(f"  K={k:>2}: coverage {ev.coverage:6.1%} | capex "
              f"{_money(ev.capex):>9} | total {_money(ev.objective):>9}")
    recommended = k_list[int(np.argmin(totals))]
    return dict(k_list=k_list, totals=totals, capex=capex,
                uncovered=uncovered, coverages=coverages,
                recommended_k=recommended, best_total=min(totals))


def cmd_calculator(args):
    problem = SitingProblem.from_arrays(
        cand_latlon=[[0, 0]], demand_latlon=[[0, 0]], demand_kwh=[1.0])
    problem.price_per_kwh = args.price
    problem.elec_cost_per_kwh = args.elec_cost
    problem.om_per_port_year = args.om
    served = args.served_kwh_day
    ports = args.ports
    capex = args.capex_port * ports + args.capex_site
    rev = served * args.price * 365
    energy = served * args.elec_cost * 365
    ebitda = rev - energy - ports * args.om
    payback = capex / ebitda if ebitda > 0 else float("inf")
    proj = simple_projection(problem, served, ports,
                             years=args.years, growth=args.growth,
                             discount=args.discount)
    ev_miles = served * 365 / 0.30
    co2 = ev_miles / 28.0 * 8.89 / 1000.0
    print("\nChargeSight savings calculator")
    print("  (single-plan P&L; compare plans with `chargesight bench`)\n")
    print(f"  Served demand        {served:>12,.0f} kWh/day")
    print(f"  Ports                {ports:>12,d}")
    print(f"  Capex                {_money(capex):>12} "
          f"({ports} x {_money(args.capex_port)} + site {_money(args.capex_site)})")
    print(f"  Revenue              {_money(rev):>12}/yr")
    print(f"  Energy cost          {_money(energy):>12}/yr")
    print(f"  O&M                  {_money(ports*args.om):>12}/yr")
    print(f"  EBITDA               {_money(ebitda):>12}/yr")
    pb = f"{payback:.1f} years" if np.isfinite(payback) else "never (EBITDA <= 0)"
    print(f"  Simple payback       {pb:>12}")
    print(f"  {args.years}-yr operating NPV   {_money(proj['npv_operating']):>12} "
          f"(@ {args.discount:.0%} discount, {args.growth:.0%}/yr growth, ex-capex)")
    print(f"  EV miles enabled     {ev_miles:>12,.0f} miles/yr")
    print(f"  Gasoline displaced   {co2:>12,.0f} t CO2/yr")
    return 0


def cmd_report(args):
    problem = _load_problem(args)
    print(f"Solving {problem.name} (K={args.stations}) for the memo...")
    result = _solve(problem, args)
    evaluator = result.evaluator
    baselines, rand_evals = run_all_baselines(problem, evaluator,
                                              args.stations, seed=args.seed)
    plans = {"ChargeSight hybrid (PSO + refinement)": result.evaluation,
             **baselines}
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    sweep = None
    if not args.no_sweep:
        sweep = _run_sweep(problem, [int(k) for k in args.sweep_ks.split(",")],
                           args)
        viz.plot_sweep(sweep["k_list"], sweep["totals"], sweep["capex"],
                       sweep["uncovered"],
                       os.path.join(outdir, "network_sweep.png"),
                       sweep["recommended_k"])
    viz.plot_benchmark_panel(problem, plans, result,
                             os.path.join(outdir, "benchmark_panel.png"))
    with open(os.path.join(outdir, "bench.csv"), "w") as f:
        f.write("plan,coverage,capex,ports,sites,utilization,ebitda,"
                "payback_years,total_cost\n")
        for name, ev in plans.items():
            e = plan_economics(problem, ev)
            f.write(f"\"{name}\",{ev.coverage:.4f},{e['capex']:.0f},"
                    f"{ev.ports_total},{ev.sites_used},{e['utilization']:.4f},"
                    f"{e['ebitda']:.0f},{e['payback_years']:.2f},"
                    f"{e['total_cost_of_plan']:.0f}\n")
    viz.plot_map_before_after(
        problem, plans["Even spread"], result.evaluation,
        os.path.join(outdir, "map_before_after.png"))
    viz.plot_convergence(result, baselines,
                         os.path.join(outdir, "convergence.png"))
    viz.animate_optimization(problem, result,
                             os.path.join(outdir, "siting_animation.gif"))
    _write_plan_csv(os.path.join(outdir, "plan.csv"), problem,
                    result.evaluation)
    with open(os.path.join(outdir, "kpis.json"), "w") as f:
        json.dump(_kpis_dict(problem, result, plans), f, indent=2,
                  default=float)
    memo = write_report(os.path.join(outdir, "REPORT.md"), problem, result,
                        plans, baselines_random_evals=rand_evals, sweep=sweep)
    print(f"\n  memo: {memo}")
    print(f"  artifacts in: {outdir}/")
    return result, plans, sweep


def _load_problem(args):
    if args.csv_demand:
        return _problem_from_csv(args.csv_demand, args)
    return SitingProblem.load_builtin(args.city)


def _problem_from_csv(path, args):
    """CSV: lat,lon,kwh_day[,name][,category] with a single header row."""
    import csv
    lats, lons, kwhs, names, cats = [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lats.append(float(row["lat"]))
            lons.append(float(row["lon"]))
            kwhs.append(float(row["kwh_day"]))
            names.append(row.get("name") or f"poi_{len(names)}")
            cats.append(row.get("category") or "custom")
    dem = np.column_stack([lats, lons])
    kwh = np.array(kwhs)
    cands = dem + np.random.default_rng(args.seed).uniform(
        -0.002, 0.002, size=dem.shape)      # default curbside candidates
    return SitingProblem.from_arrays(cand_latlon=cands, demand_latlon=dem,
                                     demand_kwh=kwh, demand_names=names,
                                     demand_categories=cats, name=path)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="chargesight",
        description="Swarm-intelligent siting of EV charging networks.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, iterations=250, particles=40, stations=12):
        sp.add_argument("--city", default="manhattan",
                        help="bundled dataset (manhattan) or a JSON path")
        sp.add_argument("--csv-demand", default=None,
                        help="CSV of your own demand points "
                             "(lat,lon,kwh_day,name,category)")
        sp.add_argument("--stations", type=int, default=stations)
        sp.add_argument("--iterations", type=int, default=iterations)
        sp.add_argument("--particles", type=int, default=particles)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--refine-every", type=int, default=10)
        sp.add_argument("--outdir", default="chargesight_output")

    sp = sub.add_parser("solve", help="optimize one network plan")
    common(sp)
    sp.add_argument("--gif", action="store_true",
                    help="render the optimization animation (GIF)")
    sp.add_argument("--map", action="store_true",
                    help="render before/after map")
    sp.add_argument("--convergence", action="store_true",
                    help="render convergence plot vs baselines")
    sp.set_defaults(func=cmd_solve)

    sp = sub.add_parser("bench", help="optimizer vs baselines")
    common(sp)
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("sweep", help="network sizing sweep")
    common(sp, iterations=80, particles=24)
    sp.add_argument("--klist", default="6,8,10,12,14,16,18,20")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("calculator", help="P&L / payback calculator")
    sp.add_argument("--served-kwh-day", type=float, default=9000.0)
    sp.add_argument("--ports", type=int, default=12)
    sp.add_argument("--price", type=float, default=0.42)
    sp.add_argument("--elec-cost", type=float, default=0.20)
    sp.add_argument("--om", type=float, default=6000.0)
    sp.add_argument("--capex-port", type=float, default=60000.0)
    sp.add_argument("--capex-site", type=float, default=50000.0)
    sp.add_argument("--years", type=int, default=5)
    sp.add_argument("--growth", type=float, default=0.10)
    sp.add_argument("--discount", type=float, default=0.08)
    sp.set_defaults(func=cmd_calculator)

    sp = sub.add_parser("report", help="full siting memo + artifacts")
    common(sp, iterations=300, particles=48)
    sp.add_argument("--no-sweep", action="store_true")
    sp.add_argument("--sweep-ks", default="6,8,10,12,14,16,18,20")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("demo", help="end-to-end showcase")
    common(sp, iterations=300, particles=48)
    sp.add_argument("--no-sweep", action="store_true")
    sp.add_argument("--sweep-ks", default="6,8,10,12,14,16,18,20")
    sp.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
