"""Figure for the essay: what the fluidity index sees.
Panels (a)-(d): windowed-dFC correlation matrices D_ij for the model's mixed and mostly-up states and for one
cross-spectrum-preserving surrogate of each (same covariance, same spectra, stationary Gaussian).
Panel (e): fluidity index of the model (large mark) against 20 surrogates (small marks) in each collective state.
Data: results/runs/g0d_basins.npz (G = 0.9, sigma = 0.18; cfg 0 all-down, cfg 4 mixed, cfg 5 mostly-up).
python3 -I fig_g1.py doc/essay/fig_g1.pdf
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[2] / "core"),
]

import sys, os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import stats as S

out = sys.argv[1]
tr = 0.72
burn = int(20.0 / tr)
K = 20
# (file, cfg): all G = 0.9, sigma = 0.18, 300 s. "mixed (2)" is the second mixed-state record (g0b_s1_rb cfg 13, init 0.5).
cfgs = {
    "all-down": ("results/runs/g0d_basins.npz", 0),
    "mostly-up": ("results/runs/g0d_basins.npz", 5),
    "mixed": ("results/runs/g0d_basins.npz", 4),
    "mixed (2)": ("results/runs/g0b_s1_rb.npz", 13),
}
MODEL, SURR = "#B20000", "#000099"  # the essay's red!70!black and blue!60!black
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})

res = {}
for name, (f, c) in cfgs.items():
    z = np.load(f, allow_pickle=True)
    b = z["bold"][c].astype(float)[burn:]
    rng = np.random.default_rng(
        0
    )  # same seed and draw order as g1_surrogate.py, so the ensemble is the one in results/runs/*_g1_cfg*.json
    D_m, W = S.dfc_w(b, tr)
    s_m = S.switching_index(D_m, W)
    surr = [S.phase_randomise(b, rng) for _ in range(K)]
    Ds = [S.dfc_w(sb, tr)[0] for sb in surr]
    s_s = np.array([S.switching_index(D, W) for D in Ds])
    med = int(
        np.argsort(np.abs(s_s - np.median(s_s)))[0]
    )  # the surrogate whose index is nearest the ensemble median
    res[name] = dict(D_m=D_m, D_s=Ds[med], s_m=s_m, s_s=s_s, rank=int((s_s < s_m).sum()) + 1)

fig = plt.figure(figsize=(6.0, 3.2))
gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 0.07, 0.55, 1.45], wspace=0.18, hspace=0.38)
panels = [
    ("mixed", "D_m", "a", 0, 0),
    ("mixed", "D_s", "b", 0, 1),
    ("mostly-up", "D_m", "c", 1, 0),
    ("mostly-up", "D_s", "d", 1, 1),
]
for name, key, lab, r, col in panels:
    ax = fig.add_subplot(gs[r, col])
    D = res[name][key]
    im = ax.imshow(D, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    which = "model" if key == "D_m" else "surrogate"
    s = (
        res[name]["s_m"]
        if key == "D_m"
        else S.switching_index(D, int(round(60 / tr)) // int(round(2 / tr)))
    )
    if r == 0:
        ax.set_title(which, fontsize=8, pad=3)
    ax.text(
        0.02,
        0.97,
        f"({lab})",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85),
    )
    ax.text(
        0.98,
        0.03,
        f"index {s:.4f}",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85),
    )
    if r == 1:
        ax.set_xlabel("window", fontsize=7, labelpad=2)
    if col == 0:
        ax.set_ylabel(f"{name}\nwindow", fontsize=7, labelpad=2)
cax = fig.add_subplot(gs[:, 2])
cb = fig.colorbar(im, cax=cax)
cb.ax.set_title(r"$D_{ij}$", fontsize=7, pad=3)
cb.ax.tick_params(labelsize=6, width=0.5)

ax = fig.add_subplot(gs[:, 4])
names = ["all-down", "mostly-up", "mixed", "mixed (2)"]
jrng = np.random.default_rng(1)
for i, name in enumerate(names):
    ss = res[name]["s_s"]
    jit = jrng.uniform(-0.14, 0.14, size=len(ss))
    ax.scatter(i + jit, ss, s=9, color=SURR, alpha=0.6, linewidths=0, zorder=2)
    ax.scatter(
        [i], [res[name]["s_m"]], s=46, color=MODEL, zorder=3, edgecolors="white", linewidths=0.8
    )
    ax.hlines(ss.mean(), i - 0.22, i + 0.22, color=SURR, lw=1.0, zorder=2)
ax.set_xticks(range(4))
ax.set_xticklabels(["all-\ndown", "mostly-\nup", "mixed", "mixed\n(2)"], fontsize=7)
ax.set_xlim(-0.5, 3.5)
ax.set_ylabel("fluidity index", fontsize=7)
ax.tick_params(labelsize=6, width=0.5)
ax.set_title("(e) model vs. 20 surrogates", loc="left", fontsize=8, pad=3)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", lw=0.3, color="0.85", zorder=0)
ax.scatter([], [], s=46, color=MODEL, label="model")
ax.scatter([], [], s=9, color=SURR, label="surrogate (mean as bar)")
ax.set_ylim(0, 1.22 * max(max(res[n]["s_s"].max(), res[n]["s_m"]) for n in names))
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
for name in names:
    r = res[name]
    print(
        f"{name:10s} model {r['s_m']:.4f}  surrogates {r['s_s'].mean():.4f} ± {r['s_s'].std():.4f} (K={K})  "
        f"model/mean {r['s_m']/r['s_s'].mean():.2f}  rank of model {r['rank']}/{K+1}  shown surrogate index {S.switching_index(r['D_s'], int(round(60/tr))//int(round(2/tr))):.4f}"
    )
