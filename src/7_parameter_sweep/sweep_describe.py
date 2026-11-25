"""Description of the registered sweep beyond sweep_maps.py's summary (doc/LOG.md, 2026-11-24). Post hoc items are labelled.
Usage: python3 -I sweep_describe.py <folder with the batch *.json>
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "common"),
]

import sys, os, json, glob, site

sys.path.append(site.getusersitepackages())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sweep_maps as sm
from sweep_maps import STATS, NAMES, ETA0, J0, D0

folder = sys.argv[1]
dE, dJ, dD = 0.005, 0.02, 0.005
print("== batches")
for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
    d = json.load(open(p))
    t = d["timing"]
    rs = d["runs"]
    print(
        f"{os.path.basename(p):22s} runs {len(rs):5d}  finite {sum(r['finite'] for r in rs):5d}  wall {t['wall_s']:7.0f} s  s/run {t['s_per_run']:.2f}  us/step/run {t['us_per_step_per_run']}"
    )
print("== non-finite or incomplete runs")
bad = []
for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
    for r in json.load(open(p))["runs"]:
        if not r["finite"] or any(r.get(m) is None for m in STATS):
            bad.append(r)
from collections import Counter

print("by (G, sigma):", sorted(Counter((r["G"], r["nsig"]) for r in bad).items()))
print(
    "by (eta, J, Delta, dt, x64):",
    sorted(Counter((r["eta"], r["J"], r["Delta"], r["dt"], r.get("x64")) for r in bad).items()),
)
print(
    "finite flag false:",
    sum(not r["finite"] for r in bad),
    "| finite but statistic missing:",
    sum(r["finite"] for r in bad),
)

cells, _ = sm.load(folder)
Gs = sorted({k[0] for k in cells if k[5] == 0.01 and not k[6] and k[2:5] == (ETA0, J0, D0)})
Ss = sorted({k[1] for k in cells if k[5] == 0.01 and not k[6] and k[2:5] == (ETA0, J0, D0)})
Ga, Sa = np.array(Gs), np.array(Ss)
mean, se, sd, n = sm.grid(cells, Gs, Ss)
print(
    "== base cells with fewer than 4 seeds:",
    [
        (Gs[j], Ss[i], int(n[0, i, j]))
        for i in range(len(Ss))
        for j in range(len(Gs))
        if n[0, i, j] < 4
    ],
)
D, E = {}, {}
D["G"], E["G"] = sm.central(mean, se, Ga, 2)
D["sigma"], E["sigma"] = sm.central(mean, se, Sa, 1)
for name, dlt in (("eta", dE), ("J", dJ), ("Delta", dD)):
    c0 = dict(eta=ETA0, J=J0, Delta=D0)
    up = sm.grid(cells, Gs, Ss, **{**c0, name: c0[name] + dlt})
    dn = sm.grid(cells, Gs, Ss, **{**c0, name: c0[name] - dlt})
    D[name] = (up[0] - dn[0]) / (2 * dlt)
    E[name] = np.hypot(up[1], dn[1]) / (2 * dlt)
fine = sm.grid(cells, Gs, Ss, dt=0.005)
D["dt_halving"] = fine[0] - mean
E["dt_halving"] = np.hypot(fine[1], se)

i0, j0 = Ss.index(0.03), Gs.index(0.54)
print(f"== working point (G 0.54, sigma 0.030): base mean (se) and derivative (se) per parameter")
print("statistic            base            " + "  ".join(f"{p:>18s}" for p in D))
for k in range(len(STATS)):
    print(
        f"{NAMES[k]:20s} {mean[k, i0, j0]:7.4f} ({se[k, i0, j0]:.4f})  "
        + "  ".join(f"{D[p][k, i0, j0]:9.3g} ({E[p][k, i0, j0]:6.2g})" for p in D)
    )
print(
    "== the same in units of 'one G notch' (0.01): change in each statistic for G +0.01, eta +0.005, J +0.02, Delta +0.005, sigma +0.004, dt halved"
)
steps = dict(G=0.01, eta=0.005, J=0.02, Delta=0.005, sigma=0.004, dt_halving=1.0)
for k in range(len(STATS)):
    print(
        f"{NAMES[k]:20s} "
        + "  ".join(
            f"{p}: {D[p][k, i0, j0] * steps[p]:+8.4f} ({E[p][k, i0, j0] * steps[p]:.4f})"
            for p in steps
        )
    )

occ = mean[1]
mixed = (occ > 0.02) & (occ < 0.90)
print("== mixed region: points", int(mixed.sum()), "of", mixed.size, "; G range per sigma row:")
for i, s in enumerate(Ss):
    g = Ga[mixed[i]]
    print(
        f"  sigma {s:.3f}: G {g.min():.2f}-{g.max():.2f} ({len(g)})"
        if len(g)
        else f"  sigma {s:.3f}: none"
    )


def conf(idx, label):
    scale = np.where(sd > 0, sd, np.nan)[idx]
    cG = D["G"][idx] / scale
    eG = E["G"][idx] / scale
    strong = np.sqrt(np.nansum(cG**2, 0)) > 3 * np.sqrt(np.nansum(eG**2, 0))
    print(f"-- {label}: statistics {[NAMES[k] for k in idx]}")
    for p in ("sigma", "eta", "J", "Delta", "dt_halving"):
        cP = D[p][idx] / scale
        ok = np.isfinite(cP).all(0) & np.isfinite(cG).all(0) & strong & mixed
        num = np.nansum(cG * cP, 0)
        nG = np.sqrt(np.nansum(cG**2, 0))
        nP = np.sqrt(np.nansum(cP**2, 0))
        c = (num / (nG * nP))[ok]
        e = (num / nG**2)[ok]
        q = np.percentile(e, [10, 25, 50, 75, 90]) if ok.any() else [np.nan] * 5
        print(
            f"   G vs {p:10s}: n {int(ok.sum()):3d}; cosine median {np.median(c):+.3f}, |cos|>0.95 at {np.mean(np.abs(c) > 0.95):.2f}, >0.8 at {np.mean(np.abs(c) > 0.8):.2f}; "
            f"G-equivalent per unit: 10/25/50/75/90% = {q[0]:.3g}, {q[1]:.3g}, {q[2]:.3g}, {q[3]:.3g}, {q[4]:.3g}; at the working point cos {(num / (nG * nP))[i0, j0]:+.3f}, equiv {(num / nG ** 2)[i0, j0]:.3g}"
        )


conf(list(range(7)), "registered (seven statistics)")
conf(
    [0, 3, 4, 5, 6],
    "POST HOC: without the two near-duplicates of the mean rate (up-occupancy, median up-fraction)",
)
conf(
    [3, 4, 5, 6], "POST HOC: pattern statistics only (VAF, switching index, mean FC, corr(FC, SC))"
)
conf([0], "POST HOC: mean rate alone (cosine is then just the sign)")

# noise floor of the cosine: two independent halves of the seeds for the G column are not available with 4 seeds; instead report the expected attenuation
print(
    "== correlation between the three level statistics across mixed-region base cells (mean values):",
    np.round(np.corrcoef([mean[k][mixed] for k in (0, 1, 2)]), 3).tolist(),
)

print(
    "== float64 row (sigma 0.030) against float32 row, base, per G: difference in units of combined se"
)
f64 = sm.grid(cells, Gs, Ss, x64=True)
z = (f64[0][:, i0, :] - mean[:, i0, :]) / np.hypot(f64[1][:, i0, :], se[:, i0, :])
for k in range(len(STATS)):
    zz = z[k][np.isfinite(z[k])]
    print(
        f"  {NAMES[k]:20s} n {len(zz):2d}; mean z {zz.mean():+.2f}; rms z {np.sqrt((zz ** 2).mean()):.2f}; |z|>2 at {int((np.abs(zz) > 2).sum())}; max |z| {np.abs(zz).max():.2f}"
    )
print("== dt halving along the sigma 0.030 row (mean rate): G, dt0.01, dt0.005, difference (se)")
for j, G in enumerate(Gs):
    if 0.48 <= G <= 0.62:
        print(
            f"  {G:.2f}  {mean[0, i0, j]:.4f}  {fine[0][0, i0, j]:.4f}  {D['dt_halving'][0, i0, j]:+.4f} ({E['dt_halving'][0, i0, j]:.4f})"
        )
