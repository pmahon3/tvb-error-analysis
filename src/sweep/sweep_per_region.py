"""Do per-region mean rates separate G from eta? (doc/LOG.md, 2026-11-24, 21:22, part A, and its amendment.)

For each grid point and seed: the response of every region's mean rate to G and to eta, J and Delta. The statistic
is the "hub preference" of a response: its slope on in-strength divided by its mean, pooled over grid points. If G and
eta moved the regions in the same proportions, their hub preferences would be equal.

Usage (from the repository root):
    python3 -I src/sweep/sweep_per_region.py results/runs/sweep/node_stats_sweep.npz [floor]
With "floor" only the G-against-G noise floor is printed.
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

import numpy as np

from repo_paths import data

ETA0, J0, DELTA0 = -5.0, 15.0, 1.0
NUDGE = {"eta": 0.005, "J": 0.02, "Delta": 0.005}
SEEDS = (1, 2, 3, 4)
PARAMETERS = ("G", "eta", "J", "Delta")
SPLITS = (((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3)))


def mean_se(values):
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


class PerRegion:
    """The sweep's per-region mean rates, and the responses computed from them."""

    def __init__(self, path):
        runs = np.load(path)
        weights = np.load(data("hcp001_conn.npz"))["weights"]
        self.weights = weights
        self.in_strength = weights.sum(1)
        self.centred_strength = self.in_strength - self.in_strength.mean()
        self.node_mean_rate = runs["node_mean_rate"]
        self.node_up_occupancy = runs["node_up_occupancy"]
        self.finite = runs["finite"]

        # single-precision runs at step 0.01: (G, sigma, eta, J, Delta, seed) -> row
        use = (runs["dt"] == 0.01) & ~runs["x64"]
        names = ("G", "sigma", "eta", "J", "Delta")
        self.row_of = {
            tuple(round(float(runs[k][i]), 4) for k in names) + (int(runs["seed"][i]),): i
            for i in np.flatnonzero(use)
        }
        self.G_values = sorted({k[0] for k in self.row_of})
        self.sigma_values = sorted({k[1] for k in self.row_of})
        self.points = self._usable_points()

    def rate(self, G, sigma, seed, eta=ETA0, J=J0, Delta=DELTA0):
        """Per-region mean rate of one run, or None if that run is missing or went non-finite."""
        row = self.row_of.get((G, sigma, round(eta, 4), round(J, 4), round(Delta, 4), seed))
        if row is None or not self.finite[row]:
            return None
        return self.node_mean_rate[row].astype(float)

    def up_occupancy(self, G, sigma):
        """Share of time in the high state at the base setting, averaged over regions and seeds."""
        values = []
        for seed in SEEDS:
            row = self.row_of.get((G, sigma, ETA0, J0, DELTA0, seed))
            if row is not None and self.finite[row]:
                values.append(self.node_up_occupancy[row].mean())
        return np.mean(values) if len(values) == len(SEEDS) else np.nan

    def responses(self, G, sigma, seed):
        """Per-region responses at one grid point for one seed: parameter -> vector. None if any run is unusable.

        G: central difference over the two neighbouring G values. eta, J, Delta: central difference over the nudges.
        """
        j = self.G_values.index(G)
        if j == 0 or j == len(self.G_values) - 1:
            return None
        G_below, G_above = self.G_values[j - 1], self.G_values[j + 1]
        below, above = self.rate(G_below, sigma, seed), self.rate(G_above, sigma, seed)
        if below is None or above is None:
            return None
        out = {"G": (above - below) / (G_above - G_below)}
        base = {"eta": ETA0, "J": J0, "Delta": DELTA0}
        for name, step in NUDGE.items():
            up = self.rate(G, sigma, seed, **{**base, name: base[name] + step})
            down = self.rate(G, sigma, seed, **{**base, name: base[name] - step})
            if up is None or down is None:
                return None
            out[name] = (up - down) / (2 * step)
        return out

    def _usable_points(self):
        """Grid points used: mixed region, sigma >= 0.030 (below that the runs still carry the starting state),
        and every run of every seed usable."""
        points = []
        for sigma in self.sigma_values:
            if sigma < 0.030 - 1e-9:
                continue
            for G in self.G_values:
                if not (0.02 < self.up_occupancy(G, sigma) < 0.90):
                    continue
                per_seed = [self.responses(G, sigma, seed) for seed in SEEDS]
                if all(r is not None for r in per_seed):
                    points.append((G, sigma, per_seed))
        return points

    def pooled(self, name, seeds, subset=None):
        """Response vectors of one parameter, averaged over the given seeds: one vector per grid point."""
        chosen = self.points if subset is None else [p for p in self.points if subset(p)]
        return [
            np.mean([per_seed[SEEDS.index(s)][name] for s in seeds], 0) for _, _, per_seed in chosen
        ]

    def hub_preference(self, vectors):
        """Pooled slope of the response on in-strength, divided by the pooled mean response."""
        slope_part = sum(float(v @ self.centred_strength) for v in vectors)
        mean_part = sum(float(v.mean()) for v in vectors)
        return slope_part / (mean_part * float(self.centred_strength @ self.centred_strength))

    def profile(self, name):
        """Each region's share of the pooled response, scaled so the mean over regions is 1."""
        total = np.sum(self.pooled(name, SEEDS), 0)
        return total / total.mean()

    def diluted_T(self, a_vectors, b_vectors):
        """The statistic first registered (biased under its null): mean over grid points of the correlation between
        in-strength and what is left of a after removing its best multiple of b."""
        values = []
        for a, b in zip(a_vectors, b_vectors):
            residual = a - (a @ b) / (b @ b) * b
            values.append(np.corrcoef(residual, self.in_strength)[0, 1])
        return float(np.mean(values))

    def per_seed_hub_preference(self, subset=None):
        return {
            name: [self.hub_preference(self.pooled(name, (seed,), subset)) for seed in SEEDS]
            for name in PARAMETERS
        }

    def noise_floor(self):
        """G response from two seeds against the G response from the other two, for the three ways to split."""
        differences, diluted = [], []
        for first, second in SPLITS:
            a, b = self.pooled("G", first), self.pooled("G", second)
            differences.append(self.hub_preference(a) - self.hub_preference(b))
            diluted.append(0.5 * (self.diluted_T(a, b) + self.diluted_T(b, a)))
        return differences, diluted

    def width_control(self):
        """Hub preference of the G response over differences twice as wide, minus the registered one, per seed."""
        narrow, wide = {s: [] for s in SEEDS}, {s: [] for s in SEEDS}
        for G, sigma, per_seed in self.points:
            j = self.G_values.index(G)
            if j < 2 or j > len(self.G_values) - 3:
                continue
            G_low, G_high = self.G_values[j - 2], self.G_values[j + 2]
            outer = [(self.rate(G_low, sigma, s), self.rate(G_high, sigma, s)) for s in SEEDS]
            if any(low is None or high is None for low, high in outer):
                continue
            for seed, (low, high) in zip(SEEDS, outer):
                narrow[seed].append(per_seed[SEEDS.index(seed)]["G"])
                wide[seed].append((high - low) / (G_high - G_low))
        differences = [self.hub_preference(wide[s]) - self.hub_preference(narrow[s]) for s in SEEDS]
        return differences, len(narrow[SEEDS[0]])


def main():
    floor_only = len(sys.argv) > 2 and sys.argv[2] == "floor"
    study = PerRegion(sys.argv[1])
    s = study.in_strength
    print(
        f"grid points used: {len(study.points)} "
        "(mixed region, sigma >= 0.030, all 28 runs and both G neighbours usable)"
    )
    print(
        f"in-strength over the 84 regions: min {s.min():.2f}, median {np.median(s):.2f}, max {s.max():.2f}"
    )

    print("\n== noise floor: G response from two seeds against G response from the other two")
    floor, floor_diluted = study.noise_floor()
    for (first, second), d, t in zip(SPLITS, floor, floor_diluted):
        print(
            f"   seeds {first} minus {second}: difference in hub preference {d:+.4f}; diluted T {t:+.3f}"
        )
    largest_floor = max(abs(x) for x in floor)
    print(f"   largest |difference in hub preference| among the splits: {largest_floor:.4f}")
    print(
        f"   hub preference of the G response, all seeds: "
        f"{study.hub_preference(study.pooled('G', SEEDS)):.4f} per unit in-strength"
    )
    if floor_only:
        return

    print("\n== hub preference of each response, per seed (slope on in-strength / mean response)")
    beta = study.per_seed_hub_preference()
    for name, values in beta.items():
        m, e = mean_se(values)
        print(
            f"   {name:6s} "
            + "  ".join(f"{v:+.4f}" for v in values)
            + f"   mean {m:+.4f} ± {e:.4f}"
        )

    print("\n== difference from the G response, per seed")
    for name in ("eta", "J", "Delta"):
        d = [g - x for g, x in zip(beta["G"], beta[name])]
        m, e = mean_se(d)
        print(
            f"   G minus {name:6s} "
            + "  ".join(f"{v:+.4f}" for v in d)
            + f"   mean {m:+.4f} ± {e:.4f}   t(3) = {m / e:+.2f}"
            f"   relative to G's hub preference: {m / np.mean(beta['G']):+.1%}"
        )
    d_eta = [g - x for g, x in zip(beta["G"], beta["eta"])]
    m, e = mean_se(d_eta)
    separated = m > 0 and m / e > 3.18 and abs(m) > largest_floor
    print(
        f"\nregistered rule (G against eta): mean {m:+.4f}, t(3) {m / e:+.2f}, largest floor {largest_floor:.4f}"
        f" -> {'separated at region level' if separated else 'not separated at 4 seeds'}"
    )

    print(
        "\n== control: does the width of the G difference matter? (G spans 0.02, the eta nudge about 0.009 in G)"
    )
    d, n_points = study.width_control()
    m, e = mean_se(d)
    print(
        f"   {n_points} points; hub preference with G differences twice as wide minus the registered one, per seed: "
        + "  ".join(f"{v:+.4f}" for v in d)
        + f"   mean {m:+.4f} ± {e:.4f}"
    )

    print("\n== by side of the transition (description)")
    for label, subset in (
        ("G <= 0.55", lambda p: p[0] <= 0.55),
        ("G >= 0.57", lambda p: p[0] >= 0.57),
    ):
        n_points = sum(1 for p in study.points if subset(p))
        side = study.per_seed_hub_preference(subset)
        line = [f"   {label} ({n_points} points):"]
        for name in PARAMETERS:
            line.append(f"{name} {np.mean(side[name]):+.4f} ± {np.std(side[name], ddof=1) / 2:.4f}")
        print("  ".join(line))

    print("\n== the statistic first registered, biased under its null (description)")
    t_eta = [study.diluted_T(study.pooled("G", (s,)), study.pooled("eta", (s,))) for s in SEEDS]
    print(
        "   corr(G response minus its best multiple of the eta response, in-strength), per seed: "
        + "  ".join(f"{v:+.3f}" for v in t_eta)
        + "; floor (G against G): "
        + "  ".join(f"{v:+.3f}" for v in floor_diluted)
    )

    print("\n== coupling input and rates behind the exchange rates (description)")
    for G, sigma in ((0.54, 0.030), (0.60, 0.030), (0.56, 0.042)):
        base = np.mean([study.rate(G, sigma, seed) for seed in SEEDS], 0)
        coupling_input = float(np.mean(study.weights @ base))
        print(
            f"   (G {G}, sigma {sigma}): mean rate {base.mean():.3f}; mean coupling input sum_j W_ij r_j "
            f"{coupling_input:.3f}; steady-state formula would give 1 unit of eta = {1 / coupling_input:.2f} of G, "
            f"1 unit of J = {base.mean() / coupling_input:.2f} of G"
        )


if __name__ == "__main__":
    main()
