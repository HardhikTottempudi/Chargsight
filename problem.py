"""Siting problem definition and bundled datasets."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields

import numpy as np

M_PER_DEG_LAT = 111_320.0

# Default economic / engineering assumptions (documented, configurable).
DEFAULTS = dict(
    walk_radius_m=400.0,        # 5-minute walk to a charger
    port_kwh_day=900.0,         # daily energy throughput per 150 kW DCFC port
    max_ports=12,               # make-ready ceiling per site
    fixed_site_cost=50_000.0,   # permitting, trenching, make-ready per site ($)
    port_capex=60_000.0,        # installed cost per DCFC port ($)
    price_per_kwh=0.42,         # retail charging price ($/kWh)
    elec_cost_per_kwh=0.20,     # energy + demand charges ($/kWh)
    uncovered_years=3.0,        # opportunity-cost window for unserved demand
    om_per_port_year=6_000.0,   # O&M per port per year ($)
)


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in meters."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6_371_000.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass
class SitingProblem:
    """Where should K charging stations go?

    Demand points carry an expected energy request (kWh/day). Candidate sites
    are parcels (curbside / parking / grid-ready positions) where a station
    may legally and practically be built.
    """
    demand_latlon: np.ndarray            # (n, 2) lat/lon
    demand_kwh: np.ndarray               # (n,)
    demand_names: list
    demand_categories: list
    cand_latlon: np.ndarray              # (m, 2) lat/lon
    name: str = "problem"
    walk_radius_m: float = DEFAULTS["walk_radius_m"]
    port_kwh_day: float = DEFAULTS["port_kwh_day"]
    max_ports: int = DEFAULTS["max_ports"]
    fixed_site_cost: float = DEFAULTS["fixed_site_cost"]
    port_capex: float = DEFAULTS["port_capex"]
    price_per_kwh: float = DEFAULTS["price_per_kwh"]
    elec_cost_per_kwh: float = DEFAULTS["elec_cost_per_kwh"]
    uncovered_years: float = DEFAULTS["uncovered_years"]
    om_per_port_year: float = DEFAULTS["om_per_port_year"]
    meta: dict = field(default_factory=dict)
    _distances: np.ndarray | None = None

    # ------------------------------------------------------------- properties
    @property
    def n_demand(self):
        return len(self.demand_kwh)

    @property
    def n_cand(self):
        return len(self.cand_latlon)

    @property
    def total_demand_kwh(self):
        return float(np.sum(self.demand_kwh))

    @property
    def margins_per_kwh(self):
        return self.price_per_kwh - self.elec_cost_per_kwh

    def bbox(self):
        lat = self.cand_latlon[:, 0]
        lon = self.cand_latlon[:, 1]
        return (lat.min(), lat.max()), (lon.min(), lon.max())

    def distances(self):
        """(n, m) demand-to-candidate distance matrix in meters (cached)."""
        if self._distances is None:
            self._distances = haversine_m(
                self.demand_latlon[:, 0][:, None], self.demand_latlon[:, 1][:, None],
                self.cand_latlon[:, 0][None, :], self.cand_latlon[:, 1][None, :])
        return self._distances

    def uncovered_cost_rate(self):
        """$ per kWh/day of demand left unserved, over the opportunity window."""
        return self.margins_per_kwh * 365.0 * self.uncovered_years

    # ------------------------------------------------------------- factories
    @classmethod
    def from_arrays(cls, cand_latlon, demand_latlon, demand_kwh,
                    demand_names=None, demand_categories=None, **kwargs):
        cand_latlon = np.asarray(cand_latlon, dtype=float)
        demand_latlon = np.asarray(demand_latlon, dtype=float)
        demand_kwh = np.asarray(demand_kwh, dtype=float)
        n = len(demand_kwh)
        names = demand_names or [f"demand_{i}" for i in range(n)]
        cats = demand_categories or ["custom"] * n
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in kwargs.items() if k in known}
        return cls(demand_latlon=demand_latlon, demand_kwh=demand_kwh,
                   demand_names=list(names), demand_categories=list(cats),
                   cand_latlon=cand_latlon, **kwargs)

    @classmethod
    def from_json(cls, path, **overrides):
        with open(path) as f:
            data = json.load(f)
        demand = data["demand"]
        cands = data["candidates"]
        return cls.from_arrays(
            cand_latlon=[[c["lat"], c["lon"]] for c in cands],
            demand_latlon=[[d["lat"], d["lon"]] for d in demand],
            demand_kwh=[d["kwh_day"] for d in demand],
            demand_names=[d.get("name", f"demand_{i}") for i, d in enumerate(demand)],
            demand_categories=[d.get("category", "custom") for d in demand],
            name=data.get("city", os.path.basename(path)),
            meta=data.get("meta", {}),
            **overrides)

    @classmethod
    def load_builtin(cls, city="manhattan", **overrides):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        path = os.path.join(data_dir, f"{city}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Bundled city '{city}' not found at {path}. "
                "Available: manhattan. Or load your own via SitingProblem.from_json.")
        return cls.from_json(path, **overrides)

    @classmethod
    def demo(cls, city="manhattan", n_demand=48, seed=0, **overrides):
        """Small subset of a bundled city for quick tests and CI runs."""
        full = cls.load_builtin(city)
        rng = np.random.default_rng(seed)
        idx = rng.choice(full.n_demand, size=min(n_demand, full.n_demand),
                         replace=False)
        dem = full.demand_latlon[idx]
        keep = np.min(full.distances()[idx], axis=0) <= 1500.0
        return cls.from_arrays(
            cand_latlon=full.cand_latlon[keep],
            demand_latlon=dem,
            demand_kwh=full.demand_kwh[idx],
            demand_names=[full.demand_names[i] for i in idx],
            demand_categories=[full.demand_categories[i] for i in idx],
            name=f"{full.name}_demo", meta=full.meta, **overrides)

    @classmethod
    def synthetic(cls, n_demand=80, n_cand=120, seed=0, **overrides):
        """Four-cluster synthetic city (no external data needed)."""
        rng = np.random.default_rng(seed)
        centers = np.array([[0.010, 0.010], [0.010, -0.010],
                            [-0.010, 0.010], [-0.012, -0.012]])
        sizes = np.array([0.40, 0.25, 0.20, 0.15]) * n_demand
        sizes = np.round(sizes).astype(int)
        sizes[-1] = n_demand - sizes[:-1].sum()
        dem = []
        for c, s in zip(centers, sizes):
            dem.append(c + rng.normal(0, 0.0035, size=(s, 2)))
        dem = np.vstack(dem)
        base_lat, base_lon = 40.75, -73.98
        dem[:, 0] += base_lat
        dem[:, 1] += base_lon
        kwh = rng.uniform(120, 700, size=n_demand)
        cands = rng.uniform(-0.030, 0.030, size=(n_cand, 2))
        cands[:, 0] += base_lat
        cands[:, 1] += base_lon
        return cls.from_arrays(
            cand_latlon=cands, demand_latlon=dem, demand_kwh=kwh,
            demand_names=[f"synthetic_{i}" for i in range(n_demand)],
            name="synthetic", **overrides)
