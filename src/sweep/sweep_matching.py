"""Is a nudge in eta, J or Delta (or halving the step) the same as a shift in G, in the runs themselves?
(doc/LOG.md, 2026-11-25.)

For each nudged setting: find the G at which the base curve of the mean rate equals the nudged mean rate, then ask
whether the other statistics of the nudged runs sit on the base curve at that same G.
Usage (from the repository root): python3 -I src/sweep/sweep_matching.py results/runs/sweep/done results/runs/sweep/pilot_done.json
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

import json
import site
import warnings

sys.path.append(
    site.getusersitepackages()
)  # sweep_maps imports the plotting library, which -I hides
warnings.filterwarnings("ignore")

import numpy as np

import sweep_maps
from sweep_maps import D0, ETA0, J0, STATS

MEAN_RATE = STATS.index("m1_mean_r")
TESTED = {
    "VAF": STATS.index("m5_vaf2_r"),
    "mean FC": STATS.index("mean_fc"),
    "corr(FC, SC)": STATS.index("corr_fc_sc"),
    "switching index": STATS.index("switching_index"),
}
NUDGE = {"eta": 0.005, "J": 0.02, "Delta": 0.005}
BASE = {"eta": ETA0, "J": J0, "Delta": D0}
DERIVATIVE_RATE = {
    "eta": 0.85,
    "J": 0.41,
    "Delta": 1.02,
}  # of G per unit, from the confounding maps (17:35 entry)

cells, _ = sweep_maps.load(sys.argv[1])
is_base = lambda k: k[5] == 0.01 and not k[6] and k[2:5] == (ETA0, J0, D0)
G_values = sorted({k[0] for k in cells if is_base(k)})
sigma_values = sorted({k[1] for k in cells if is_base(k)})
G_array = np.array(G_values)
base_mean, base_se, _, _ = sweep_maps.grid(cells, G_values, sigma_values)  # (statistics, sigma, G)


def match_on_base_curve(row, G_near, mean_rate):
    """The G at which the base mean-rate curve of this sigma row equals `mean_rate`, nearest to G_near.
    Returns (G*, index of the segment's left end, fraction along the segment), or None."""
    curve = base_mean[MEAN_RATE, row]
    best = None
    for j in range(len(G_values) - 1):
        a, b = curve[j], curve[j + 1]
        if not (np.isfinite(a) and np.isfinite(b)) or a == b:
            continue
        t = (mean_rate - a) / (b - a)
        if -1e-9 <= t <= 1 + 1e-9:
            G_star = G_array[j] + t * (G_array[j + 1] - G_array[j])
            if best is None or abs(G_star - G_near) < abs(best[0] - G_near):
                best = (G_star, j, t)
    return best


def z_scores(row, G_near, values, errors):
    """z of each tested statistic of a nudged setting against the base curve at the matched G."""
    match = match_on_base_curve(row, G_near, values[MEAN_RATE])
    if match is None:
        return None
    G_star, j, t = match
    width = G_array[j + 1] - G_array[j]
    slope_rate = (base_mean[MEAN_RATE, row, j + 1] - base_mean[MEAN_RATE, row, j]) / width
    base_rate_se = np.hypot(
        (1 - t) * base_se[MEAN_RATE, row, j], t * base_se[MEAN_RATE, row, j + 1]
    )
    G_star_se = np.hypot(errors[MEAN_RATE], base_rate_se) / abs(slope_rate)
    out = {}
    for name, k in TESTED.items():
        on_curve = (1 - t) * base_mean[k, row, j] + t * base_mean[k, row, j + 1]
        curve_se = np.hypot((1 - t) * base_se[k, row, j], t * base_se[k, row, j + 1])
        slope = (base_mean[k, row, j + 1] - base_mean[k, row, j]) / width
        total_se = np.sqrt(errors[k] ** 2 + curve_se**2 + (slope * G_star_se) ** 2)
        out[name] = (values[k] - on_curve) / total_se
    return G_star, out


def cell_statistics(key):
    runs = cells.get(key, [])
    if len(runs) < 4:
        return None
    table = np.array([[run[name] for name in STATS] for run in runs], float)
    return table.mean(0), table.std(0, ddof=1) / np.sqrt(len(runs))


def summarise(label, z_by_statistic):
    print(f"   {label}")
    for name, zs in z_by_statistic.items():
        zs = np.asarray(zs)
        print(
            f"      {name:16s} n {len(zs):4d}   share |z| < 2: {np.mean(np.abs(zs) < 2):.2f}   "
            f"median |z| {np.median(np.abs(zs)):.2f}   mean z {zs.mean():+.2f}"
        )


# grid points used: mixed region, sigma >= 0.030, complete base cell
up_occupancy = base_mean[STATS.index("m2_occ_up07")]
points = [
    (i, j)
    for i, sigma in enumerate(sigma_values)
    for j in range(len(G_values))
    if sigma >= 0.030 - 1e-9 and 0.02 < up_occupancy[i, j] < 0.90
]
print(f"grid points: {len(points)} (mixed region, sigma >= 0.030)")

print(
    "\n== calibration: the float64 base row (sigma 0.030) against the float32 base curve; true shift is zero"
)
row_030 = sigma_values.index(0.03)
calibration = {name: [] for name in TESTED}
shifts = []
for j, G in enumerate(G_values):
    if not (0.02 < up_occupancy[row_030, j] < 0.90):
        continue
    found = cell_statistics((G, 0.03, ETA0, J0, D0, 0.01, True))
    result = z_scores(row_030, G, *found) if found else None
    if result:
        shifts.append(result[0] - G)
        for name, z in result[1].items():
            calibration[name].append(z)
print(
    f"   matched G minus true G: median {np.median(shifts):+.4f}, spread (sd) {np.std(shifts):.4f}"
)
summarise("z of the other statistics at the matched G", calibration)

print("\n== nudged settings against the base curve")
empirical_rate = {}
for name, step in NUDGE.items():
    z_all = {stat: [] for stat in TESTED}
    rates = []
    for i, j in points:
        G, sigma = G_values[j], sigma_values[i]
        matched = {}
        for sign in (+1, -1):
            key_values = dict(BASE)
            key_values[name] = round(BASE[name] + sign * step, 4)
            found = cell_statistics(
                (G, sigma, key_values["eta"], key_values["J"], key_values["Delta"], 0.01, False)
            )
            result = z_scores(i, G, *found) if found else None
            if result:
                matched[sign] = result[0]
                for stat, z in result[1].items():
                    z_all[stat].append(z)
        if len(matched) == 2:
            rates.append((matched[+1] - matched[-1]) / (2 * step))
    empirical_rate[name] = np.array(rates)
    q = np.percentile(rates, [25, 50, 75])
    print(
        f"   {name}: empirical exchange rate (G per unit), quartiles {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f} over "
        f"{len(rates)} points; derivative-based median {DERIVATIVE_RATE[name]}; ratio {q[1] / DERIVATIVE_RATE[name]:.2f}"
    )
    summarise(f"{name} +/- {step}: z of the other statistics at the matched G", z_all)

print("\n== half the step against the base curve at step 0.01")
z_all = {stat: [] for stat in TESTED}
shifts = []
for i, j in points:
    G, sigma = G_values[j], sigma_values[i]
    found = cell_statistics((G, sigma, ETA0, J0, D0, 0.005, False))
    result = z_scores(i, G, *found) if found else None
    if result:
        shifts.append(result[0] - G)
        for stat, z in result[1].items():
            z_all[stat].append(z)
q = np.percentile(shifts, [25, 50, 75])
print(
    f"   matched G minus G, quartiles {q[0]:+.4f} / {q[1]:+.4f} / {q[2]:+.4f} over {len(shifts)} points"
)
summarise("z of the other statistics at the matched G", z_all)

print(
    "\n== larger nudges at the working point (pilot, 16 seeds) against the sigma 0.030 base curve (4 seeds)"
)
pilot = [
    r for r in json.load(open(sys.argv[2]))["runs"] if r["finite"] and abs(r["G"] - 0.54) < 1e-9
]
settings = sorted({(round(r["eta"], 4), round(r["J"], 4), round(r["Delta"], 4)) for r in pilot})
print("   eta      J       Delta    matched G   " + "   ".join(f"z {name}" for name in TESTED))
for eta, J, Delta in settings:
    rows = [
        r
        for r in pilot
        if (round(r["eta"], 4), round(r["J"], 4), round(r["Delta"], 4)) == (eta, J, Delta)
    ]
    table = np.array([[r[name] for name in STATS] for r in rows], float)
    result = z_scores(row_030, 0.54, table.mean(0), table.std(0, ddof=1) / np.sqrt(len(rows)))
    if result:
        print(
            f"   {eta:<8g} {J:<7g} {Delta:<8g} {result[0]:.4f}      "
            + "   ".join(f"{z:+6.2f}" for z in result[1].values())
        )
