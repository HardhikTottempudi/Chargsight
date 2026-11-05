"""Maps, convergence plots, benchmark panels and the optimization animation."""
from __future__ import annotations

import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse

from .economics import plan_economics

# ------------------------------------------------------------------ geography
MANHATTAN = [
    (-74.0170, 40.7030), (-74.0150, 40.7080), (-74.0138, 40.7165),
    (-74.0125, 40.7255), (-74.0110, 40.7350), (-74.0113, 40.7460),
    (-74.0020, 40.7570), (-74.0033, 40.7700), (-74.0100, 40.7790),
    (-74.0090, 40.7950), (-74.0030, 40.8180), (-73.9880, 40.8320),
    (-73.9640, 40.8470), (-73.9480, 40.8540), (-73.9310, 40.8720),
    (-73.9230, 40.8770), (-73.9070, 40.8690), (-73.9050, 40.8760),
    (-73.9290, 40.8560), (-73.9270, 40.8420), (-73.9180, 40.8200),
    (-73.9120, 40.8060), (-73.9250, 40.7950), (-73.9320, 40.7940),
    (-73.9350, 40.7900), (-73.9410, 40.7800), (-73.9440, 40.7720),
    (-73.9590, 40.7600), (-73.9650, 40.7550), (-73.9700, 40.7460),
    (-73.9740, 40.7380), (-73.9790, 40.7220), (-73.9820, 40.7120),
    (-73.9900, 40.7060), (-74.0010, 40.7050), (-74.0120, 40.7010),
]
CENTRAL_PARK = [
    (-73.9730, 40.7644), (-73.9819, 40.7679),
    (-73.9555, 40.7994), (-73.9462, 40.7989),
]
COLOR_LAND = "#f1eee7"
COLOR_WATER = "#d4e6f3"
COLOR_PARK = "#cde6c2"
COLOR_DEMAND = "#3a6ea5"
COLOR_CAND = "#b9b9b9"
COLOR_OPT = "#15803d"
COLOR_BASE = "#dc2626"
ASPECT = 1.0 / math.cos(math.radians(40.78))    # equal-meters display


def _draw_base_map(ax, problem, title, pad=0.006):
    ax.set_facecolor(COLOR_WATER)
    land = plt.Polygon(MANHATTAN, closed=True, facecolor=COLOR_LAND,
                       edgecolor="#8a8a8a", linewidth=0.8, zorder=1)
    ax.add_patch(land)
    park = plt.Polygon(CENTRAL_PARK, closed=True, facecolor=COLOR_PARK,
                       edgecolor="#9dbb90", linewidth=0.6, zorder=2)
    ax.add_patch(park)
    ax.text(-74.006, 40.760, "Hudson\nRiver", fontsize=7, color="#7d97a8",
            ha="center", zorder=2)
    ax.text(-73.934, 40.745, "East\nRiver", fontsize=7, color="#7d97a8",
            ha="center", zorder=2)
    ax.text(-73.9665, 40.783, "Central Park", fontsize=6.5, color="#5d7a52",
            ha="center", rotation=-61, zorder=3)
    # candidates
    ax.scatter(problem.cand_latlon[:, 1], problem.cand_latlon[:, 0],
               s=1.5, c=COLOR_CAND, alpha=0.55, linewidths=0, zorder=3)
    # demand bubbles sized by kWh/day
    w = problem.demand_kwh
    sizes = 6 + 52 * (w / max(w.max(), 1e-9))
    ax.scatter(problem.demand_latlon[:, 1], problem.demand_latlon[:, 0],
               s=sizes, c=COLOR_DEMAND, alpha=0.55, linewidths=0, zorder=4)
    (lat0, lat1), (lon0, lon1) = problem.bbox()
    ax.set_xlim(lon0 - pad, lon1 + pad)
    ax.set_ylim(lat0 - pad, lat1 + pad)
    ax.set_aspect(ASPECT)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def _draw_plan(ax, problem, site_indices, ports_per_site=None, color=COLOR_OPT,
               walk_radius_m=400.0, max_ports=12):
    pts = problem.cand_latlon[np.asarray(site_indices, dtype=int)]
    r_lat = walk_radius_m / 111_320.0
    r_lon = r_lat / math.cos(math.radians(40.78))
    for (lat, lon) in pts:
        ax.add_patch(Ellipse((lon, lat), width=2 * r_lon, height=2 * r_lat,
                             facecolor=color, alpha=0.10, edgecolor=color,
                             linewidth=0.7, linestyle="--", zorder=5))
    if ports_per_site is not None and len(ports_per_site):
        sizes = 30 + 16 * np.asarray(ports_per_site)
    else:
        sizes = 46
    ax.scatter(pts[:, 1], pts[:, 0], marker="D", s=sizes, c=color,
               edgecolors="white", linewidths=0.8, zorder=7)


def _stats_line(problem, ev):
    e = plan_economics(problem, ev)
    return (f"coverage {ev.coverage:.0%}  |  capex ${ev.capex/1e6:.2f}M  |  "
            f"{ev.ports_total} ports / {ev.sites_used} sites  |  "
            f"payback {e['payback_years']:.1f} yr")


def plot_map_before_after(problem, ev_before, ev_after, path,
                          label_before="Before: rule-of-thumb siting",
                          label_after="After: ChargeSight hybrid swarm"):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.1), dpi=130,
                             facecolor="white")
    for ax, ev, label, color in [
            (axes[0], ev_before, label_before, COLOR_BASE),
            (axes[1], ev_after, label_after, COLOR_OPT)]:
        _draw_base_map(ax, problem, label)
        _draw_plan(ax, problem, ev.site_indices, ev.ports_per_site,
                   color=color, walk_radius_m=problem.walk_radius_m)
        ax.text(0.5, -0.055, _stats_line(problem, ev), transform=ax.transAxes,
                ha="center", fontsize=9, color="#333333")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=COLOR_DEMAND,
                   markersize=6, label="Demand (sized by kWh/day)"),
        plt.Line2D([], [], marker=".", linestyle="", color=COLOR_CAND,
                   markersize=6, label="Candidate sites"),
        plt.Line2D([], [], marker="D", linestyle="", color=COLOR_BASE,
                   markersize=8, label="Baseline station"),
        plt.Line2D([], [], marker="D", linestyle="", color=COLOR_OPT,
                   markersize=8, label="Optimized station"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("EV charging network siting - Manhattan, NYC (real street grid)",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(result, baselines: dict, path, title=None):
    hist = result.history
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=130, facecolor="white")
    its = hist["iters"]
    ax.plot(its, hist["best_known_f"] / 1e6, color=COLOR_OPT, lw=2.0,
            label="Hybrid: adaptive PSO + discrete refinement")
    ax.plot(its, hist["pso_raw_f"] / 1e6, color="#60a5fa", lw=1.1, ls="--",
            alpha=0.8, label="Adaptive PSO (raw swarm, no refinement)")
    for name, ev in baselines.items():
        ax.axhline(ev.objective / 1e6, color="#9ca3af", lw=1.2, ls=":",
                   alpha=0.9)
        ax.text(its[-1], ev.objective / 1e6, f"  {name}: ${ev.objective/1e6:.2f}M",
                fontsize=8, va="bottom", ha="right", color="#4b5563")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total plan cost: capex + 3-yr opportunity cost ($M)")
    ax.set_title(title or "Swarm convergence - Manhattan siting "
                 f"(K={result.n_stations} stations)", fontsize=11,
                 fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    sec = ax.twinx()
    sec.plot(its, hist["best_known_coverage"], color="#f59e0b", lw=1.0,
             alpha=0.55)
    sec.set_ylabel("Coverage of best-known plan", color="#b45309", fontsize=9)
    sec.set_ylim(0, 1.05)
    sec.tick_params(labelsize=8, colors="#b45309")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_panel(problem, plans: dict, result, path):
    """2x2 dashboard: convergence, capex, coverage, payback."""
    short = {"ChargeSight hybrid (PSO + refinement)": "ChargeSight\nhybrid",
             "Greedy coverage": "Greedy", "K-means clusters": "K-means",
             "Even spread": "Even\nspread", "Random siting": "Random"}
    names = list(plans.keys())
    evs = [plans[n] for n in names]
    econs = [plan_economics(problem, e) for e in evs]
    palette = ["#15803d", "#2563eb", "#7c3aed", "#d97706", "#9ca3af"]
    colors = [palette[i % len(palette)] for i in range(len(names))]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6), dpi=130,
                             facecolor="white")
    ax = axes[0, 0]
    hist = result.history
    ax.plot(hist["iters"], hist["best_known_f"] / 1e6, color=COLOR_OPT, lw=1.9,
            label="ChargeSight hybrid")
    ax.plot(hist["iters"], hist["pso_raw_f"] / 1e6, color="#60a5fa", lw=1.0,
            ls="--", alpha=0.75, label="PSO raw")
    ax.set_title("Convergence", fontsize=10, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=8)
    ax.set_ylabel("Plan cost ($M)", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=8)

    def bars(ax, key, title, fmt, scale=1.0):
        vals = [econ[key] for econ in econs]
        bars_ = ax.bar(range(len(names)), [v * scale for v in vals],
                       color=colors, width=0.62)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([short.get(n, n.replace(" ", "\n", 1))
                            for n in names], fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=8)
        top = max(v * scale for v in vals)
        ax.set_ylim(0, top * 1.18)
        for b, v in zip(bars_, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.03,
                    fmt.format(v * scale), ha="center", fontsize=8,
                    color="#374151")

    bars(axes[0, 1], "capex", "Capital cost", "${:.2f}M", scale=1e-6)
    bars(axes[1, 0], "coverage", "Demand coverage (weighted)", "{:.0%}")
    bars(axes[1, 1], "payback_years", "Payback period", "{:.1f} yr")
    best_i = int(np.argmin([e["total_cost_of_plan"] for e in econs]))
    fig.suptitle(f"Benchmark panel - best plan: {names[best_i]} "
                 f"(total cost ${econs[best_i]['total_cost_of_plan']/1e6:.2f}M)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_sweep(k_list, totals, capex, uncovered, path, recommended_k):
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=130, facecolor="white")
    ks = np.array(k_list)
    ax.stackplot(ks, np.array(capex) / 1e6, np.array(uncovered) / 1e6,
                 labels=["Capex", "Opportunity cost of unserved demand"],
                 colors=["#15803d", "#f87171"], alpha=0.75)
    ax.plot(ks, np.array(totals) / 1e6, color="#111827", lw=1.6, marker="o",
            label="Total plan cost")
    rk = int(recommended_k)
    idx = list(ks).index(rk)
    ax.annotate(f"recommended K = {rk}\n${totals[idx]/1e6:.2f}M",
                xy=(rk, totals[idx] / 1e6),
                xytext=(rk + 0.4, totals[idx] / 1e6 * 0.82),
                fontsize=9, fontweight="bold", color="#111827",
                arrowprops=dict(arrowstyle="->", color="#111827"))
    ax.set_xlabel("Number of stations K")
    ax.set_ylabel("Cost ($M): capex + 3-yr opportunity cost")
    ax.set_title("Network sizing sweep - where diminishing returns set in",
                 fontsize=11.5, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def animate_optimization(problem, result, path, fps=12, dpi=90):
    """GIF of the swarm converging: particles (red), best plan (green)."""
    frames = result.history["frames"]
    frame_iters = result.history["frame_iters"]
    if frames is None or len(frames) == 0:
        raise ValueError("No animation frames captured (capture_frames=False?)")
    hist = result.history
    gbest = hist["gbest_x"].reshape(-1, result.n_stations, 2)
    curve = hist["best_known_f"]
    cov = hist["best_known_coverage"]

    fig = plt.figure(figsize=(11.8, 4.9), dpi=dpi, facecolor="white")
    ax_map = fig.add_subplot(1, 2, 1)
    ax_cur = fig.add_subplot(1, 2, 2)

    def draw(fi):
        it = int(frame_iters[fi])
        ax_map.clear()
        ax_cur.clear()
        _draw_base_map(ax_map, problem,
                       f"Swarm search - iteration {it}")
        P = frames[fi]
        flat_lon = P[:, :, 1].ravel()
        flat_lat = P[:, :, 0].ravel()
        ax_map.scatter(flat_lon, flat_lat, s=2.5, c="#ef4444", alpha=0.28,
                       linewidths=0, zorder=5)
        # best-so-far plan at this point in the run
        idx = min(it, len(curve)) - 1
        sites = result.evaluator.snap(gbest[min(it, len(gbest) - 1)])
        _draw_plan(ax_map, problem, sites, color=COLOR_OPT,
                   walk_radius_m=problem.walk_radius_m)
        ax_map.text(0.02, 0.02,
                    f"plan cost ${curve[idx]/1e6:.2f}M | coverage {cov[idx]:.0%}",
                    transform=ax_map.transAxes, fontsize=8.5,
                    bbox=dict(boxstyle="round", fc="white", ec="#cccccc",
                              alpha=0.85))
        ax_cur.plot(hist["iters"][: idx + 1], curve[: idx + 1] / 1e6,
                    color=COLOR_OPT, lw=1.8)
        ax_cur.plot(hist["iters"][: idx + 1],
                    hist["pso_raw_f"][: idx + 1] / 1e6, color="#60a5fa",
                    lw=0.9, ls="--", alpha=0.7)
        ax_cur.set_xlim(hist["iters"][0], hist["iters"][-1])
        ymin = curve.min() / 1e6 * 0.92
        ymax = max(curve.max(), hist["pso_raw_f"].max()) / 1e6 * 1.05
        ax_cur.set_ylim(ymin, ymax)
        ax_cur.set_xlabel("Iteration", fontsize=8)
        ax_cur.set_ylabel("Plan cost ($M)", fontsize=8)
        ax_cur.set_title("Convergence", fontsize=10, fontweight="bold")
        ax_cur.grid(alpha=0.25)
        ax_cur.tick_params(labelsize=8)
        return []

    anim = FuncAnimation(fig, draw, frames=len(frame_iters), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
