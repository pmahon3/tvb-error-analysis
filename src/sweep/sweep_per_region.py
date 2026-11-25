"""Do per-region mean rates separate G from eta? (doc/LOG.md, 2026-11-24, 21:25, part A, and its amendment.)

For each grid point and seed: the response of every region's mean rate to G and to eta. The statistic is the relative
hub preference of a response: its slope on in-strength divided by its mean, pooled over grid points.
Usage (from the repository root): python3 -I src/sweep/sweep_per_region.py results/runs/sweep/node_stats_sweep.npz [stage]
    stage "floor" prints only the G-against-G noise floor; without it, everything.
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
floor_only = len(sys.argv) > 2 and sys.argv[2] == "floor"

runs = np.load(sys.argv[1])
in_strength = np.load(data("hcp001_conn.npz"))["weights"].sum(1)
centred_strength = in_strength - in_strength.mean()

# index the single-precision runs at step 0.01: (G, sigma, eta, J, Delta, seed) -> row
use = (runs["dt"] == 0.01) & ~runs["x64"]
key_of = lambda i: tuple(
    round(float(runs[k][i]), 4) for k in ("G", "sigma", "eta", "J", "Delta")
) + (int(runs["seed"][i]),)
row_of = {key_of(i): i for i in np.flatnonzero(use)}
finite = runs["finite"]
G_values = sorted({k[0] for k in row_of})
sigma_values = sorted({k[1] for k in row_of})


def rate(G, sigma, seed, eta=ETA0, J=J0, Delta=DELTA0):
    """Per-region mean rate of one run, or None if that run is missing or went non-finite."""
    i = row_of.get((G, sigma, round(eta, 4), round(J, 4), round(Delta, 4), seed))
    return None if i is None or not finite[i] else runs["node_mean_rate"][i].astype(float)


def occupancy(G, sigma):
    values = [
        runs["node_up_occupancy"][row_of[(G, sigma, ETA0, J0, DELTA0, s)]].mean()
        for s in SEEDS
        if (G, sigma, ETA0, J0, DELTA0, s) in row_of
        and finite[row_of[(G, sigma, ETA0, J0, DELTA0, s)]]
    ]
    return np.mean(values) if len(values) == len(SEEDS) else np.nan


def responses(G, sigma, seed):
    """Per-region responses at one grid point for one seed: dict parameter -> vector, or None if any run is unusable."""
    j = G_values.index(G)
    if j == 0 or j == len(G_values) - 1:
        return None
    below, above = rate(G_values[j - 1], sigma, seed), rate(G_values[j + 1], sigma, seed)
    if below is None or above is None:
        return None
    out = {"G": (above - below) / (G_values[j + 1] - G_values[j - 1])}
    base = {"eta": ETA0, "J": J0, "Delta": DELTA0}
    for name, step in NUDGE.items():
        up = rate(G, sigma, seed, **{**base, name: base[name] + step})
        down = rate(G, sigma, seed, **{**base, name: base[name] - step})
        if up is None or down is None:
            return None
        out[name] = (up - down) / (2 * step)
    return out


# grid points used: mixed region, sigma >= 0.030 (burn-in), every run of every seed usable
points = []
for sigma in sigma_values:
    if sigma < 0.030 - 1e-9:
        continue
    for G in G_values:
        occ = occupancy(G, sigma)
        if not (0.02 < occ < 0.90):
            continue
        per_seed = [responses(G, sigma, s) for s in SEEDS]
        if all(r is not None for r in per_seed):
            points.append((G, sigma, per_seed))
print(
    f"grid points used: {len(points)} (mixed region, sigma >= 0.030, all 28 runs and both G neighbours usable)"
)
print(
    f"in-strength over the 84 regions: min {in_strength.min():.2f}, median {np.median(in_strength):.2f}, max {in_strength.max():.2f}"
)


def hub_preference(vectors):
    """Pooled slope of the response on in-strength, divided by the pooled mean response."""
    numerator = sum(float(v @ centred_strength) for v in vectors)
    denominator = sum(float(v.mean()) for v in vectors) * float(centred_strength @ centred_strength)
    return numerator / denominator


def diluted_T(a_vectors, b_vectors):
    """The statistic first registered (biased under its null): mean over points of corr(a - kappa b, in-strength)."""
    values = []
    for a, b in zip(a_vectors, b_vectors):
        e = a - (a @ b) / (b @ b) * b
        values.append(np.corrcoef(e, in_strength)[0, 1])
    return float(np.mean(values))


def pooled(name, seeds, subset=None):
    """Response vectors of one parameter, averaged over the given seeds, one per grid point."""
    chosen = points if subset is None else [p for p in points if subset(p)]
    return [
        np.mean([per_seed[SEEDS.index(s)][name] for s in seeds], 0) for _, _, per_seed in chosen
    ]


mean_se = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))

print("\n== noise floor: G response from two seeds against G response from the other two")
floor_beta, floor_T = [], []
for first, second in (((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3))):
    d_beta = hub_preference(pooled("G", first)) - hub_preference(pooled("G", second))
    t = 0.5 * (
        diluted_T(pooled("G", first), pooled("G", second))
        + diluted_T(pooled("G", second), pooled("G", first))
    )
    floor_beta.append(d_beta)
    floor_T.append(t)
    print(
        f"   seeds {first} minus {second}: difference in hub preference {d_beta:+.4f}; diluted T {t:+.3f}"
    )
print(
    f"   largest |difference in hub preference| among the splits: {max(abs(x) for x in floor_beta):.4f}"
)
print(
    f"   hub preference of the G response, all seeds: {hub_preference(pooled('G', SEEDS)):.4f} per unit in-strength"
)
if floor_only:
    sys.exit()

print("\n== hub preference of each response, per seed (slope on in-strength / mean response)")
beta = {
    name: [hub_preference(pooled(name, (s,))) for s in SEEDS] for name in ("G", "eta", "J", "Delta")
}
for name, values in beta.items():
    m, e = mean_se(values)
    print(f"   {name:6s} " + "  ".join(f"{v:+.4f}" for v in values) + f"   mean {m:+.4f} ± {e:.4f}")
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
separated = m > 0 and m / e > 3.18 and abs(m) > max(abs(x) for x in floor_beta)
print(
    f"\nregistered rule (G against eta): mean {m:+.4f}, t(3) {m / e:+.2f}, largest floor {max(abs(x) for x in floor_beta):.4f}"
    f" -> {'separated at region level' if separated else 'not separated at 4 seeds'}"
)

print(
    "\n== control: does the width of the G difference matter? (the G response spans 0.02 in G, the eta one about 0.009)"
)
narrow, wide = {s: [] for s in SEEDS}, {s: [] for s in SEEDS}
for G, sigma, per_seed in points:
    j = G_values.index(G)
    if j < 2 or j > len(G_values) - 3:
        continue
    outer = [(rate(G_values[j - 2], sigma, s), rate(G_values[j + 2], sigma, s)) for s in SEEDS]
    if any(lo is None or hi is None for lo, hi in outer):
        continue
    for s, (lo, hi) in zip(SEEDS, outer):
        narrow[s].append(per_seed[SEEDS.index(s)]["G"])
        wide[s].append((hi - lo) / (G_values[j + 2] - G_values[j - 2]))
d = [hub_preference(wide[s]) - hub_preference(narrow[s]) for s in SEEDS]
m, e = mean_se(d)
print(
    f"   {len(narrow[1])} points; hub preference with G differences twice as wide minus the registered one, per seed: "
    + "  ".join(f"{v:+.4f}" for v in d)
    + f"   mean {m:+.4f} ± {e:.4f}"
)

print("\n== by side of the transition (description)")
for label, subset in (("G <= 0.55", lambda p: p[0] <= 0.55), ("G >= 0.57", lambda p: p[0] >= 0.57)):
    n = sum(1 for p in points if subset(p))
    line = [f"   {label} ({n} points):"]
    for name in ("G", "eta", "J", "Delta"):
        values = [hub_preference(pooled(name, (s,), subset)) for s in SEEDS]
        line.append(f"{name} {np.mean(values):+.4f} ± {np.std(values, ddof=1) / 2:.4f}")
    print("  ".join(line))

print("\n== the statistic first registered, biased under its null (description)")
t_eta = [diluted_T(pooled("G", (s,)), pooled("eta", (s,))) for s in SEEDS]
print(
    f"   corr(G response minus its best multiple of the eta response, in-strength), per seed: "
    + "  ".join(f"{v:+.3f}" for v in t_eta)
    + f"; floor (G against G): "
    + "  ".join(f"{v:+.3f}" for v in floor_T)
)

print("\n== coupling input and rates behind the exchange rates (description)")
W = np.load(data("hcp001_conn.npz"))["weights"]
for G, sigma in ((0.54, 0.030), (0.60, 0.030), (0.56, 0.042)):
    base = np.mean([rate(G, sigma, s) for s in SEEDS], 0)
    print(
        f"   (G {G}, sigma {sigma}): mean rate {base.mean():.3f}; mean coupling input sum_j W_ij r_j {np.mean(W @ base):.3f};"
        f" steady-state formula would give 1 unit of eta = {1 / np.mean(W @ base):.2f} of G, 1 unit of J = {base.mean() / np.mean(W @ base):.2f} of G"
    )
