# ChargeSight Siting Memo - Manhattan
https://github.com/user-attachments/assets/48c9f929-b3fa-4054-b132-1e0110918af9



## Executive summary

For a **12-station** fast-charging network on **manhattan**, the hybrid swarm optimizer places **12 sites / 35 DCFC ports** that capture **44%** of modeled demand (mean walk to a charger: **267 m**), at a capital cost of **$2.70M** and projected EBITDA of **$2.01M/yr** (payback **1.3 years** at default assumptions).

Versus unstructured siting (random), the optimized network earns **$1.41M more per year** (+31 pts of demand coverage) and lifts port utilization by **15.4 pts**. Versus rule-of-thumb 'even spread' planning the annual EBITDA edge is **$1.62M**.

## Head-to-head results

| Plan | Coverage | Capex | Ports | Utilization | EBITDA/yr | Payback | Total cost* |
|---|---|---|---|---|---|---|---|
| ChargeSight hybrid (PSO + refinement) | 44% | $2.70M | 35 | 88% | $2.01M | 1.3 yr | $11.09M |
| Greedy coverage | 42% | $2.64M | 34 | 85% | $1.88M | 1.4 yr | $11.43M |
| K-means clusters | 18% | $1.53M | 18 | 70% | $808k | 1.9 yr | $13.83M |
| Even spread | 9% | $780k | 8 | 75% | $388k | 2.0 yr | $14.52M |
| Random siting | 14% | $1.18M | 13 | 72% | $602k | 2.0 yr | $14.19M |

*Total cost = capex + opportunity cost of unserved demand over a 3-year window (the optimizer's objective). Utilization = served / (ports x 900 kWh/day).

Random-siting dispersion over 20 draws: total cost $14.85M +/- $243k (best draw $14.19M). The optimizer does not rely on luck.

## Value of optimization

- **vs Greedy coverage**: +$127k/yr EBITDA, +1,656 kWh/day served, capex delta $60k, payback improves 0.1 yr
- **vs K-means clusters**: +$1.20M/yr EBITDA, +16,240 kWh/day served, capex delta $1.17M, payback improves 0.6 yr
- **vs Even spread**: +$1.62M/yr EBITDA, +22,214 kWh/day served, capex delta $1.92M, payback improves 0.7 yr
- **vs Random siting**: +$1.41M/yr EBITDA, +19,175 kWh/day served, capex delta $1.52M, payback improves 0.6 yr

Five-year operating NPV of the optimized plan (10%/yr demand growth, 8% discount, ex-capex): **$9.83M**; cumulative operating cash flow **$12.50M** against capex $2.70M.

## Sustainability & access KPIs

- EV miles enabled: **33,637,062 miles/yr** (27,647 kWh/day served)
- Gross gasoline displaced: **10,680 t CO2/yr** (fleet 28 mpg, 8.89 kg CO2/gal; grid upstream emissions not netted)
- Port allocation equity (Gini): **0.11** (lower = ports spread more evenly across sites)

## Network sizing

Sweeping network size K = [6, 8, 10, 12, 14, 16, 18, 20] shows total cost still falling at the edge of the range: the recommended size within this sweep is **K = 20** (total $9.92M), and each additional station still rescues more demand than it costs. Extend the sweep (or shorten the opportunity window) to find where the curve turns; per-station coverage gains are visibly diminishing.

| K | Coverage | Capex | Total cost* |
|---|---|---|---|
| 6 | 27% | $1.56M | $12.59M |
| 8 | 33% | $1.96M | $12.10M |
| 10 | 39% | $2.36M | $11.49M |
| 12 | 44% | $2.70M | $11.09M |
| 14 | 49% | $3.04M | $10.71M |
| 16 | 54% | $3.38M | $10.37M |
| 18 | 57% | $3.66M | $10.12M |
| 20 | 60% | $3.94M | $9.92M |

## Method

The optimizer hybridizes two layers. **(1) Adaptive particle swarm**: particles fly in continuous lat/lon space (2K dimensions), with inertia 0.9->0.35, cognitive weight 2.5->0.5, social weight 0.5->2.5, velocity clamping, boundary reflection and stagnation-triggered partial restarts. **(2) Discrete refinement**: every few iterations the swarm's best positions are snapped to real candidate sites and hill-climbed with relocate moves under hard capacity limits (ports per site, port throughput); improved plans are injected back into the swarm. Every plan is scored by the same capacity-aware decoder: demand points walk to the nearest site with spare capacity within 400 m; unserved demand is priced at its opportunity cost.

## Assumptions (all configurable)

| Parameter | Value |
|---|---|
| Walk radius to a charger | 400 m |
| Port throughput | 900 kWh/day (150 kW DCFC, ~6 effective h) |
| Max ports per site | 12 |
| Site fixed capex | $50k |
| Port capex (installed) | $60k |
| Retail charging price | $0.42/kWh |
| Energy cost | $0.20/kWh |
| O&M | $6k/port/yr |
| Opportunity window for unserved demand | 3 years |

*Cost and price defaults are illustrative mid-points of published US DCFC ranges; replace them with project-specific quotes via the problem parameters. Demand weights are an illustrative model anchored on real locations; plug in your own telemetry via `SitingProblem.from_json` / CSV to price a real deployment.*

## Data provenance

- Demand points: 233 locations (real landmark, transit, campus, hospital, retail and neighborhood coordinates).
- Candidate sites: 1132 positions sampled on the real Manhattan street grid (Commissioners' Plan of 1811 lattice, 29-degree rotation), land-mass filtered, Central Park excluded.
- Optimization effort: 24,562 objective evaluations, 17 stagnation restarts, 14.6 s wall time.
