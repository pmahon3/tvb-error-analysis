"""F2: two regimes, three rows. Columns: single-basin 'mixed' (g0d_basins cfg 4, G=0.9, s=0.18) and the transition regime
(g0e_alt seeds 1-2 at G=1.05, s=0.30; the trace and matrices use seed 1). Rows: fraction of regions up vs time;
D_ij for model and its median phase surrogate; fluidity index of model vs 20 phase surrogates and 20 mvAR(1) realisations.
python3 fig_f2.py doc/essay/v3/fig_f2.pdf
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[2] / "core"),
]

import sys, os, json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import stats as S

out = sys.argv[1]
tr = 0.72
burn = int(20 / tr)
K = 20
MODEL, PH, AR = "#B20000", "#000099", "#6a6a6a"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})


def mvar1_fit(b):
    x = b - b.mean(0)
    A = np.linalg.lstsq(x[:-1], x[1:], rcond=None)[0]
    E = x[1:] - x[:-1] @ A
    return A, np.linalg.cholesky(np.cov(E.T) + 1e-12 * np.eye(x.shape[1]))


def mvar1_sim(A, L, T, rng, burn_=200):
    N = A.shape[0]
    x = np.zeros(N)
    o = np.zeros((T + burn_, N))
    for t in range(T + burn_):
        x = x @ A + L @ rng.standard_normal(N)
        o[t] = x
    return o[burn_:]


def record(f, pick):
    z = np.load(f, allow_pickle=True)
    G = z["G"]
    sig = z["sigma"]
    c = pick(G, sig)
    return z["bold"][c].astype(float)[burn:], z["r10"][c].astype(float)[2000:]


def analyse(b):
    D_m, W = S.dfc_w(b, tr)
    s_m = S.switching_index(D_m, W)
    rng = np.random.default_rng(0)
    ph = []
    Ds = []
    for _ in range(K):
        D, _ = S.dfc_w(S.phase_randomise(b, rng), tr)
        ph.append(S.switching_index(D, W))
        Ds.append(D)
    ph = np.array(ph)
    med = int(np.argsort(np.abs(ph - np.median(ph)))[0])
    A, L = mvar1_fit(b)
    rng = np.random.default_rng(0)
    ar = []
    for _ in range(K):
        D, _ = S.dfc_w(mvar1_sim(A, L, len(b), rng) + b.mean(0), tr)
        ar.append(S.switching_index(D, W))
    return dict(D_m=D_m, D_s=Ds[med], s_m=s_m, ph=ph, ar=np.array(ar))


cols = {}
b, r = record("results/runs/earlier_studies/first_reimplementation/g0d_basins.npz", lambda G, s: 4)
cols["single-basin (G = 0.9, σ = 0.18)"] = dict(trace=(r > 0.5).mean(1), recs=[analyse(b)])
alt = []
for seed in (1, 2):
    b, r = record(
        f"results/runs/earlier_studies/first_reimplementation/g0e_alt_s{seed}.npz",
        lambda G, s: [
            i for i in range(len(G)) if abs(G[i] - 1.05) < 1e-6 and abs(s[i] - 0.30) < 1e-6
        ][0],
    )
    if seed == 1:
        trace_alt = (r > 0.5).mean(1)
    alt.append(analyse(b))
cols["transition regime (G = 1.05, σ = 0.30)"] = dict(trace=trace_alt, recs=alt)

fig = plt.figure(figsize=(6.2, 6.4))
gs = GridSpec(
    3,
    4,
    figure=fig,
    height_ratios=[0.8, 1.15, 1.0],
    width_ratios=[1, 1, 1, 1],
    hspace=0.55,
    wspace=0.3,
)
for j, (name, col) in enumerate(cols.items()):
    # row 1: trace
    ax = fig.add_subplot(gs[0, 2 * j : 2 * j + 2])
    t = np.arange(len(col["trace"])) * 0.01
    ax.plot(t, col["trace"], color="k", lw=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, t[-1])
    ax.set_ylabel("fraction of regions up", fontsize=7)
    ax.set_xlabel("time (s)", fontsize=7, labelpad=1)
    ax.set_title(f"({'ab'[j]}) {name}", loc="left", fontsize=8)
    ax.tick_params(labelsize=6, width=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    # row 2: matrices (model, surrogate) for the shown record
    rec = col["recs"][0]
    for k, (key, lab) in enumerate([("D_m", "model"), ("D_s", "surrogate")]):
        axm = fig.add_subplot(gs[1, 2 * j + k])
        im = axm.imshow(rec[key], cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        axm.set_xticks([])
        axm.set_yticks([])
        axm.set_title(lab, fontsize=7, pad=2)
        axm.text(
            0.98,
            0.03,
            f"index {S.switching_index(rec[key], int(round(60/tr))//int(round(2/tr))):.4f}",
            transform=axm.transAxes,
            fontsize=6,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.85),
        )
        if k == 0:
            axm.set_ylabel("window", fontsize=7, labelpad=2)
        axm.set_xlabel("window", fontsize=7, labelpad=1)
        axm.text(
            0.02,
            0.97,
            f"({'cd'[j]})" if k == 0 else "",
            transform=axm.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.85),
        )
    # row 3: fluidity dot plot, one group per record
    ax3 = fig.add_subplot(gs[2, 2 * j : 2 * j + 2])
    rng = np.random.default_rng(1)
    for i, rec in enumerate(col["recs"]):
        ax3.scatter(
            i - 0.18 + rng.uniform(-0.06, 0.06, K),
            rec["ph"],
            s=7,
            color=PH,
            alpha=0.6,
            lw=0,
            zorder=2,
        )
        ax3.scatter(
            i + 0.18 + rng.uniform(-0.06, 0.06, K),
            rec["ar"],
            s=7,
            color=AR,
            alpha=0.7,
            lw=0,
            zorder=2,
        )
        ax3.scatter([i], [rec["s_m"]], s=40, color=MODEL, zorder=3, edgecolors="white", lw=0.8)
    n = len(col["recs"])
    ax3.set_xticks(range(n))
    ax3.set_xticklabels([f"record {i+1}" for i in range(n)], fontsize=6.5)
    ax3.set_xlim(-0.6, n - 0.4)
    ax3.set_ylabel("fluidity index", fontsize=7)
    ax3.tick_params(labelsize=6, width=0.5)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(axis="y", lw=0.3, color="0.85", zorder=0)
    ax3.set_title(f"({'ef'[j]})", loc="left", fontsize=8)
    if j == 1:
        ax3.scatter([], [], s=40, color=MODEL, label="model")
        ax3.scatter([], [], s=7, color=PH, label="phase surrogate")
        ax3.scatter([], [], s=7, color=AR, label="AR(1) null")
        ax3.legend(frameon=False, fontsize=6, loc="upper left")
cax = fig.add_axes([0.92, 0.42, 0.012, 0.16])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"$D_{ij}$", fontsize=7)
cb.ax.tick_params(labelsize=6, width=0.5)
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
for name, col in cols.items():
    for i, rec in enumerate(col["recs"]):
        print(
            f"{name} record {i+1}: model {rec['s_m']:.4f} phase {rec['ph'].mean():.4f}±{rec['ph'].std():.4f} AR {rec['ar'].mean():.4f}±{rec['ar'].std():.4f}"
        )
