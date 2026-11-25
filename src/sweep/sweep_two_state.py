"""A two-state account of the per-region response, checked against the sweep (doc/LOG.md, 2026-11-25).

Each region is treated as a unit that is either down or up, with the share of time up a logistic function of its
input, and its input the drive plus the coupling from the other regions' mean rates. That gives a formula for how
every region's mean rate responds to eta and to G. This script fits the two numbers the account needs at each grid
point and compares its predictions with the measured responses.

Usage (from the repository root): python3 -I src/sweep/sweep_two_state.py results/runs/sweep/node_stats_sweep.npz
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

import numpy as np

from sweep_per_region import ETA0, J0, DELTA0, SEEDS, PerRegion

study = PerRegion(sys.argv[1])
W = study.weights
n_regions = W.shape[0]


def base_average(array, G, sigma):
    """Average over the four seeds of a per-region array at the base setting."""
    rows = [study.row_of[(G, sigma, ETA0, J0, DELTA0, seed)] for seed in SEEDS]
    return array[rows].astype(float).mean(0)


fits, predicted, measured = [], {"G": [], "eta": []}, {"G": [], "eta": []}
for G, sigma, per_seed in study.points:
    mean_rate = base_average(study.node_mean_rate, G, sigma)
    up_share = base_average(study.node_up_occupancy, G, sigma)
    coupling_input = W @ mean_rate

    # T1: is logit(share of time up) linear in the coupling input, across regions?
    usable = (up_share > 0.002) & (up_share < 0.998)
    logit = np.log(up_share[usable] / (1 - up_share[usable]))
    slope, intercept = np.polyfit(coupling_input[usable], logit, 1)
    r_squared = np.corrcoef(coupling_input[usable], logit)[0, 1] ** 2
    kappa = slope / G

    # mean rate against share of time up: slope is (up rate - down rate)
    rate_step, down_rate = np.polyfit(up_share, mean_rate, 1)

    # T2: predicted response of every region's mean rate, (1 - G X W)^-1 X b
    susceptibility = rate_step * kappa * up_share * (1 - up_share)
    feedback = G * susceptibility[:, None] * W
    largest_eigenvalue = float(np.max(np.linalg.eigvals(feedback).real))
    solve = np.linalg.solve(np.eye(n_regions) - feedback, np.eye(n_regions))
    predicted["eta"].append(solve @ (susceptibility * 1.0))
    predicted["G"].append(solve @ (susceptibility * coupling_input))
    for name in ("G", "eta"):
        measured[name].append(np.mean([r[name] for r in per_seed], 0))
    fits.append(
        dict(
            G=G,
            sigma=sigma,
            r2=r_squared,
            kappa=kappa,
            rate_step=rate_step,
            down_rate=down_rate,
            eigenvalue=largest_eigenvalue,
            n_usable=int(usable.sum()),
            mean_input=float(coupling_input.mean()),
        )
    )

column = lambda key: np.array([f[key] for f in fits])
print(f"grid points: {len(fits)}")
print("\n== T1: logit(share of time up) against coupling input, across regions, at each grid point")
r2 = column("r2")
print(
    f"   R^2: quartiles {np.percentile(r2, 25):.3f} / {np.median(r2):.3f} / {np.percentile(r2, 75):.3f}; "
    f"share of grid points above 0.9: {np.mean(r2 > 0.9):.2f}; regions usable per point: median {np.median(column('n_usable')):.0f} of {n_regions}"
)
print(
    f"   kappa (per unit input): quartiles {np.percentile(column('kappa'), 25):.2f} / {np.median(column('kappa')):.2f} / {np.percentile(column('kappa'), 75):.2f}"
)
print(
    f"   up rate minus down rate: median {np.median(column('rate_step')):.3f}; down rate: median {np.median(column('down_rate')):.3f}"
)
for sigma in sorted({f["sigma"] for f in fits}):
    k = np.array([f["kappa"] for f in fits if f["sigma"] == sigma])
    print(
        f"      sigma {sigma:.3f}: kappa median {np.median(k):6.2f}   kappa x sigma {np.median(k) * sigma:.3f}   R^2 median {np.median([f['r2'] for f in fits if f['sigma'] == sigma]):.3f}"
    )

print("\n== T2: predicted against measured responses, pooled over the grid points")
for name in ("G", "eta"):
    print(
        f"   hub preference of the {name:3s} response: predicted {study.hub_preference(predicted[name]):.3f}, measured {study.hub_preference(measured[name]):.3f}"
    )
gap_predicted = study.hub_preference(predicted["G"]) - study.hub_preference(predicted["eta"])
gap_measured = study.hub_preference(measured["G"]) - study.hub_preference(measured["eta"])
print(f"   G minus eta: predicted {gap_predicted:+.3f}, measured {gap_measured:+.3f}")
pooled_mean = lambda vectors: sum(float(v.mean()) for v in vectors)
rate_predicted = pooled_mean(predicted["eta"]) / pooled_mean(predicted["G"])
rate_measured = pooled_mean(measured["eta"]) / pooled_mean(measured["G"])
print(
    f"   exchange rate (G per unit eta, from pooled mean responses): predicted {rate_predicted:.3f}, measured {rate_measured:.3f}"
)
size_ratio = {
    name: pooled_mean(predicted[name]) / pooled_mean(measured[name]) for name in ("G", "eta")
}
print(
    f"   size of the predicted response relative to the measured one: G {size_ratio['G']:.2f}, eta {size_ratio['eta']:.2f}"
)
per_point = [np.corrcoef(p, m)[0, 1] for p, m in zip(predicted["G"], measured["G"])]
print(
    f"   correlation across regions between predicted and measured G response, per grid point: median {np.median(per_point):.2f}"
)
profile_corr = {
    name: np.corrcoef(np.sum(predicted[name], 0), np.sum(measured[name], 0))[0, 1]
    for name in ("G", "eta")
}
print(
    f"   correlation across regions of the pooled profiles: G {profile_corr['G']:.3f}, eta {profile_corr['eta']:.3f}"
)

print("\n== T3: description")
eig = column("eigenvalue")
print(
    f"   largest eigenvalue of G X W: quartiles {np.percentile(eig, 25):.2f} / {np.median(eig):.2f} / {np.percentile(eig, 75):.2f}; share above 1: {np.mean(eig > 1):.2f}"
)
share = study.profile("G")
hubs = study.in_strength > 2.7
print(
    f"   pooled G profile: plateau (regions with in-strength > 2.7) {share[hubs].mean():.2f} times the mean response; "
    f"weakest ten regions {np.sort(share)[:10].mean():.2f}"
)
