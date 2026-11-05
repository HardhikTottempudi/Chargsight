#!/usr/bin/env python3
"""End-to-end ChargeSight showcase on the bundled Manhattan dataset.

Run from anywhere:  python examples/run_manhattan.py
Outputs land in examples/outputs/.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chargesight.baselines import run_all_baselines           # noqa: E402
from chargesight.economics import plan_economics, optimization_value  # noqa: E402
from chargesight.optimizer import HybridSitingOptimizer       # noqa: E402
from chargesight.problem import SitingProblem                 # noqa: E402
from chargesight.report import write_report                   # noqa: E402
from chargesight import viz                                   # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "outputs")


def _money(x):
    if not np.isfinite(x):
        return "n/a"
    if abs(x) >= 1e6:
        return f"${x/1e6:.2f}M"
    return f"${x/1e3:.0f}k"


def main():
    os.makedirs(OUT, exist_ok=True)
    problem = SitingProblem.load_builtin("manhattan")
    print(f"Manhattan: {problem.n_demand} demand points "
          f"({problem.total_demand_kwh/1000:.1f} MWh/day), "
          f"{problem.n_cand} candidate sites")

    result = HybridSitingOptimizer(problem, n_stations=12, n_particles=48,
                                   n_iters=300, seed=0,
                                   refine_every=8, verbose=True).solve()

    baselines, rand_evals = run_all_baselines(problem, result.evaluator, 12,
                                              seed=0)
    plans = {"ChargeSight hybrid (PSO + refinement)": result.evaluation,
             **baselines}
    print("\nHead-to-head (K=12):")
    for name, ev in plans.items():
        e = plan_economics(problem, ev)
        print(f"  {name:<36} coverage {ev.coverage:5.1%} | capex "
              f"{_money(e['capex']):>8} | EBITDA {_money(e['ebitda']):>8}/yr "
              f"| total {_money(e['total_cost_of_plan']):>8}")

    v = optimization_value(problem, result.evaluation, plans["Random siting"])
    print(f"\nValue vs random siting: +{_money(v['extra_ebitda_per_year'])}/yr "
          f"EBITDA, +{v['extra_coverage']*100:.0f} pts coverage")

    viz.plot_map_before_after(problem, plans["Even spread"],
                              result.evaluation,
                              os.path.join(OUT, "map_before_after.png"))
    viz.plot_convergence(result, baselines,
                         os.path.join(OUT, "convergence.png"))
    viz.plot_benchmark_panel(problem, plans, result,
                             os.path.join(OUT, "benchmark_panel.png"))
    viz.animate_optimization(problem, result,
                             os.path.join(OUT, "siting_animation.gif"))
    write_report(os.path.join(OUT, "REPORT.md"), problem, result, plans,
                 baselines_random_evals=rand_evals)
    print(f"\nAll artifacts in {OUT}/")


if __name__ == "__main__":
    main()
