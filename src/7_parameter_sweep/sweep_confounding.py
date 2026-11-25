"""Further description of the registered sweep (doc/LOG.md, 2026-11-24), after review. Registered items that sweep_maps.py's
summary left out (holes at non-finite cells; ratio per statistic; peak shift summary) and labelled post hoc items.
Usage: python3 -I sweep_confounding.py <folder with the batch *.json>
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "common")]

import sys, os, json, glob, site
sys.path.append(site.getusersitepackages()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sweep_maps as sm
from sweep_maps import STATS, NAMES, ETA0, J0, D0
folder = sys.argv[1]; dlt = dict(eta=0.005, J=0.02, Delta=0.005); c0 = dict(eta=ETA0, J=J0, Delta=D0)
cells, _ = sm.load(folder)
Gs = sorted({k[0] for k in cells if k[5] == 0.01 and not k[6] and k[2:5] == (ETA0, J0, D0)}); Ss = sorted({k[1] for k in cells if k[5] == 0.01 and not k[6] and k[2:5] == (ETA0, J0, D0)})
Ga, Sa = np.array(Gs), np.array(Ss); i0, j0 = Ss.index(0.03), Gs.index(0.54)

def columns(cells, scale_sd=None):
    base = sm.grid(cells, Gs, Ss); mean, se, sd, n = base; D, E, nmin = {}, {}, n[0].copy()
    D["G"], E["G"] = sm.central(mean, se, Ga, 2); D["sigma"], E["sigma"] = sm.central(mean, se, Sa, 1); var = {}
    for p in dlt:
        up = sm.grid(cells, Gs, Ss, **{**c0, p: c0[p] + dlt[p]}); dn = sm.grid(cells, Gs, Ss, **{**c0, p: c0[p] - dlt[p]}); var[p] = (up, dn)
        D[p] = (up[0] - dn[0]) / (2 * dlt[p]); E[p] = np.hypot(up[1], dn[1]) / (2 * dlt[p]); nmin = np.fmin(nmin, np.fmin(np.nan_to_num(up[3][0]), np.nan_to_num(dn[3][0])))
    fine = sm.grid(cells, Gs, Ss, dt=0.005); D["dt_halving"] = fine[0] - mean; E["dt_halving"] = np.hypot(fine[1], se)
    return mean, se, sd, n, D, E, nmin, var, fine
mean, se, sd, n, D, E, nmin, var, fine = columns(cells)
occ = mean[1]; mixed = (occ > 0.02) & (occ < 0.90)

# 4. registered: non-finite cells are holes. A grid point is a hole if any dt-0.01 configuration there has fewer than 4 finite seeds,
#    or if its G or sigma difference uses such a base cell.
short = np.nan_to_num(nmin) < 4; baseshort = np.nan_to_num(n[0]) < 4; hole = short.copy()
for i in range(len(Ss)):
    for j in range(len(Gs)):
        for a, b in ((i, max(j - 1, 0)), (i, min(j + 1, len(Gs) - 1)), (max(i - 1, 0), j), (min(i + 1, len(Ss) - 1), j)):
            if baseshort[a, b]: hole[i, j] = True
fineshort = np.nan_to_num(fine[3][0]) < 4
print(f"grid points with a short cell at dt 0.01: {int(short.sum())}; holes including neighbours of short base cells: {int(hole.sum())}; of these in the mixed region: {int((hole & mixed).sum())}; points with a short dt-0.005 cell: {int(fineshort.sum())}")

def conf(idx, region, label, Dx=D, Ex=E):
    scale = np.where(sd > 0, sd, np.nan)[idx]; cG = Dx["G"][idx] / scale; eG = Ex["G"][idx] / scale
    strong = np.sqrt(np.nansum(cG ** 2, 0)) > 3 * np.sqrt(np.nansum(eG ** 2, 0)); out = [f"-- {label}"]
    for p in ("sigma", "eta", "J", "Delta", "dt_halving"):
        cP = Dx[p][idx] / scale; ok = np.isfinite(cP).all(0) & np.isfinite(cG).all(0) & strong & region
        if p == "dt_halving": ok &= ~fineshort
        num = np.nansum(cG * cP, 0); nG = np.sqrt(np.nansum(cG ** 2, 0)); nP = np.sqrt(np.nansum(cP ** 2, 0)); c = (num / (nG * nP))[ok]; e = (num / nG ** 2)[ok]
        out.append(f"   G vs {p:10s}: n {int(ok.sum()):3d}; cosine median {np.median(c):+.3f}; |cos| > 0.95 at {np.mean(np.abs(c) > 0.95):.2f}; > 0.8 at {np.mean(np.abs(c) > 0.8):.2f}; G-equivalent per unit, quartiles {np.percentile(e, 25):.3g}, {np.median(e):.3g}, {np.percentile(e, 75):.3g}")
    print("\n".join(out))
allk = list(range(7))
conf(allk, mixed, "registered, seven statistics, mixed region, survivor means kept (as sweep_maps.py)")
conf(allk, mixed & ~hole, "registered, seven statistics, mixed region, holes excluded (the registered rule)")
side = np.broadcast_to(Ga[None, :], mixed.shape)
conf(allk, mixed & ~hole & (side <= 0.55), "by side (description): G <= 0.55")
conf(allk, mixed & ~hole & (side >= 0.57), "by side (description): G >= 0.57")

# ceiling of the cosine under noise: the G column estimated from seeds {1, 2} against seeds {3, 4} (POST HOC)
sm.MIN_SEEDS = 2
def subset(seeds): return {k: [r for r in v if r["seed"] in seeds] for k, v in cells.items()}
ma, _, _, _, Da, Ea, *_ = columns(subset({1, 2})); mb, _, _, _, Db, Eb, *_ = columns(subset({3, 4}))
for idx, lab in ((allk, "seven statistics"), ([0, 3, 4, 5, 6], "five (no near-duplicates)"), ([3, 4, 5, 6], "pattern statistics only")):
    scale = np.where(sd > 0, sd, np.nan)[idx]; out = []
    for p, q in (("G", "G"), ("eta", "eta"), ("J", "J"), ("Delta", "Delta"), ("sigma", "sigma"), ("G", "eta"), ("G", "J"), ("G", "Delta"), ("G", "sigma")):
        a = Da[p][idx] / scale; b = Db[q][idx] / scale; cG = D["G"][idx] / scale; eG = E["G"][idx] / scale
        strong = np.sqrt(np.nansum(cG ** 2, 0)) > 3 * np.sqrt(np.nansum(eG ** 2, 0)); ok = np.isfinite(a).all(0) & np.isfinite(b).all(0) & mixed & ~hole & strong
        c = (np.nansum(a * b, 0) / np.sqrt(np.nansum(a ** 2, 0) * np.nansum(b ** 2, 0)))[ok]
        out.append(f"{p}(1,2)·{q}(3,4): median {np.median(c):+.3f}, >0.95 at {np.mean(np.abs(c) > 0.95):.2f}")
    print(f"POST HOC half-split, {lab}: " + "; ".join(out))

# 3. registered: ratio of derivatives per statistic ("equivalent G shift per unit") where both exceed 2 se
print("\nratio d/dp over d/dG per statistic, at mixed-region points (holes excluded) where both derivatives exceed 2 se: n, quartiles")
for p in ("eta", "J", "Delta", "dt_halving"):
    line = [f"  {p:10s}"]; R = {}
    for k in (0, 3, 5, 6, 4):
        ok = mixed & ~hole & (np.abs(D[p][k]) > 2 * E[p][k]) & (np.abs(D["G"][k]) > 2 * E["G"][k]); r = D[p][k] / D["G"][k]; R[k] = (r, ok)
        line.append(f"{NAMES[k]}: n {int(ok.sum()):3d}" + (f", {np.percentile(r[ok], 25):.3g} / {np.median(r[ok]):.3g} / {np.percentile(r[ok], 75):.3g}" if ok.sum() >= 5 else ""))
    print(" | ".join(line))
    for k in (3, 5):                                   # agreement of VAF's and mean FC's ratio with the mean rate's, point by point
        r0, ok0 = R[0]; r1, ok1 = R[k]; ok = ok0 & ok1
        if ok.sum() >= 5:
            s0 = np.abs(r0) * np.hypot(E[p][0] / D[p][0], E["G"][0] / D["G"][0]); s1 = np.abs(r1) * np.hypot(E[p][k] / D[p][k], E["G"][k] / D["G"][k]); z = (r1 - r0) / np.hypot(s0, s1)
            print(f"      {NAMES[k]} against mean rate at {int(ok.sum())} shared points: median difference {np.median((r1 - r0)[ok]):+.3g}; within 2 se at {np.mean(np.abs(z[ok]) < 2):.2f}; median |z| {np.median(np.abs(z[ok])):.2f}")

# 5. the knee at the working point: one-sided responses of the mean rate
m = lambda g: mean[0, i0, Gs.index(g)]
print(f"\nworking point, mean rate (4 seeds, se {se[0, i0, j0]:.4f}): G 0.53 {m(0.53):.4f}, 0.54 {m(0.54):.4f}, 0.55 {m(0.55):.4f}: notch below {m(0.54) - m(0.53):+.4f}, notch above {m(0.55) - m(0.54):+.4f}")
for p in dlt:
    up, dn = var[p]; print(f"   {p} +{dlt[p]}: {up[0][0, i0, j0] - mean[0, i0, j0]:+.4f} ({np.hypot(up[1][0, i0, j0], se[0, i0, j0]):.4f});  -{dlt[p]}: {dn[0][0, i0, j0] - mean[0, i0, j0]:+.4f} ({np.hypot(dn[1][0, i0, j0], se[0, i0, j0]):.4f})")

# 2. registered: location in G of the row maximum (three-point smooth); summary over the ten rows of peak(+d) - peak(-d), and dt
def peak(mu):
    smo = np.convolve(mu, np.ones(3) / 3, mode="same"); return Gs[int(np.argmax(smo[1:-1])) + 1]
print("\npeak location in G, mean over the ten sigma rows (se over rows); expected = 2 d x median G-equivalent, with sign reversed")
exp = dict(eta=-2 * 0.005 * 0.845, J=-2 * 0.02 * 0.407, Delta=-2 * 0.005 * 1.02)
for k, nm in ((3, "VAF"), (4, "switching index"), (5, "mean FC")):
    base_pk = np.array([peak(mean[k, i]) for i in range(len(Ss))]); line = [f"  {nm}: base peak by row {base_pk.min():.2f}-{base_pk.max():.2f};"]
    for p in dlt:
        up, dn = var[p]; d = np.array([peak(up[0][k, i]) - peak(dn[0][k, i]) for i in range(len(Ss))])
        line.append(f"{p}: {d.mean():+.4f} ({d.std(ddof=1) / np.sqrt(len(d)):.4f}), expected {exp[p]:+.4f};")
    okr = [i for i in range(len(Ss)) if np.isfinite(fine[0][k, i]).all()]; d = np.array([peak(fine[0][k, i]) - base_pk[i] for i in okr])
    line.append(f"dt halved: {d.mean():+.4f} ({d.std(ddof=1) / np.sqrt(len(d)):.4f}) over {len(okr)} rows, expected -0.0039"); print(" ".join(line))

# 7. scaling identity as a relative residual, holes excluded
Gg = Ga[None, None, :]; Sg = Sa[None, :, None]; kk = np.array([0.5, 0, 0, 0, 0, 0, 0])[:, None, None]
terms = [D0 * D["Delta"], ETA0 * D["eta"], 0.5 * J0 * D["J"], 0.5 * Gg * D["G"], 1.5 * Sg * D["sigma"], -kk * mean]; R = sum(terms); size = sum(np.abs(t) for t in terms)
pred = -(sum(terms[1:])) / D0                                           # the Delta derivative the identity predicts from the others
print("\nscaling identity, mixed region without holes: median |residual| / sum of |terms|; and correlation over points between the measured Delta derivative and the one predicted from the other columns")
for k in range(7):
    ok = mixed & ~hole & np.isfinite(R[k]); q = np.percentile((D['Delta'][k] / pred[k])[ok], [25, 50, 75]); print(f"  {NAMES[k]:20s} n {int(ok.sum())}  measured/predicted quartiles {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}   corr {np.corrcoef(D['Delta'][k][ok], pred[k][ok])[0, 1]:+.3f}   median |residual| / |Delta term| {np.median((np.abs(R[k]) / np.abs(terms[0][k]))[ok]):.2f}")
