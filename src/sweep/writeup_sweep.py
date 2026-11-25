"""Figures and one self-contained page for the sweep of 2026-11-24 (doc/LOG.md, 17:35 entry). Every number in the page is
computed here from the run files; nothing is typed in by hand except the text.
Usage (from the repository root): python3 -I src/sweep/writeup_sweep.py results/runs/sweep/done results/runs/tvb_reference results/runs/validation results doc/sweep_writeup.pdf
Writes the five figures to <results_dir>/figures/report/ and builds the PDF with pdflatex.
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

import sys, os, json, glob, site, re, shutil, subprocess, tempfile

sys.path.append(site.getusersitepackages())
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import sweep_maps as sm
from sweep_maps import STATS, NAMES, ETA0, J0, D0

SWEEP, APRIME, PORT, FIG, OUT = sys.argv[1:6]
FIG = os.path.join(FIG, "figures", "report")
os.makedirs(FIG, exist_ok=True)
from report_common import (
    C_F32,
    C_F64,
    C_TVB,
    DIV,
    GRID,
    HOLE,
    INK,
    INK2,
    SEQ,
    SURF,
    build_pdf,
    faded_map,
)

FOUR = [
    ("m1_mean_r", "Mean rate"),
    ("m5_vaf2_r", "VAF (two components)"),
    ("switching_index", "Switching index"),
    ("mean_fc", "Mean FC"),
]
ms = lambda v: (
    float(np.mean(v)),
    float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"),
)
isbase = (
    lambda r: abs(r.get("eta", ETA0) - ETA0) < 1e-9
    and abs(r.get("J", J0) - J0) < 1e-9
    and abs(r.get("Delta", D0) - D0) < 1e-9
)
at = lambda r, G, s: abs(r["G"] - G) < 1e-9 and abs(r["nsig"] - s) < 1e-9

# ---------------------------------------------------------------- data
tvb = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(APRIME, "A_G*.json")))]
tvb = [r for r in tvb if r.get("finite", True)]
port = []
for p in (
    sorted(glob.glob(os.path.join(PORT, "*.json")))
    + sorted(glob.glob(os.path.join(SWEEP, "*.json")))
    + [os.path.join(os.path.dirname(SWEEP.rstrip("/")), "pilot_done.json")]
):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if isinstance(d, dict):
        port += [
            dict(r, _file=os.path.basename(p))
            for r in d.get("runs", [])
            if r.get("finite") and r.get("T", 0) > 50000 and "switching_index" in r
        ]
pilot = [r for r in port if r["_file"] == "pilot_done.json"]
wp_t = [r for r in tvb if at(r, 0.54, 0.03) and r["dt"] == 0.01]
wp_32 = [
    r for r in port if at(r, 0.54, 0.03) and r["dt"] == 0.01 and isbase(r) and not r.get("x64")
]
wp_64 = [r for r in port if at(r, 0.54, 0.03) and r["dt"] == 0.01 and isbase(r) and r.get("x64")]
cells, _ = sm.load(SWEEP)
Gs = sorted({k[0] for k in cells if k[5] == 0.01 and not k[6] and k[2:5] == (ETA0, J0, D0)})
Ss = sorted({k[1] for k in cells if k[5] == 0.01 and not k[6] and k[2:5] == (ETA0, J0, D0)})
Ga, Sa = np.array(Gs), np.array(Ss)
i0, j0 = Ss.index(0.03), Gs.index(0.54)
figs = {}


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figs[name] = p


def dots(ax, x, v, color, rng, width=0.12, n_label=None):
    v = np.asarray(v, float)
    ax.scatter(
        x + rng.uniform(-width, width, len(v)),
        v,
        s=9,
        color=color,
        alpha=0.38,
        linewidths=0,
        zorder=2,
    )
    m, e = ms(v)
    ax.errorbar(
        [x],
        [m],
        yerr=[e],
        fmt="o",
        ms=5,
        color=INK,
        mfc=INK,
        mec=SURF,
        mew=1.2,
        elinewidth=1.4,
        capsize=3.5,
        zorder=4,
    )
    return m, e


# ---------------------------------------------------------------- figure 1: distributions at the working point
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3))
T1 = []
groups = [("TVB", wp_t, C_TVB), ("JAX,\nfloat32", wp_32, C_F32), ("JAX,\nfloat64", wp_64, C_F64)]
for ax, (key, title) in zip(axes, FOUR):
    row = [title]
    for x, (lab, rs, col) in enumerate(groups):
        v = [r[key] for r in rs]
        m, e = dots(ax, x, v, col, rng, 0.22)
        row.append(f"{m:.4f} ± {e:.4f}")
        ax.annotate(
            f"{m:.4f}\n± {e:.4f}",
            (x, max(v)),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK2,
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{g[0]}\nn = {len(g[1])}" for g in groups])
    ax.set_title(title)
    ax.margins(y=0.22)
    ax.grid(axis="x", visible=False)
    T1.append(row)
save(fig, "w1_working_point.png")

# ---------------------------------------------------------------- figure 2: runs over time
tr = np.load(os.path.join(PORT, "traces_wp.npz"))
A, B = tr["tvb"], tr["prec2_f32"]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axes[0]
x = np.arange(1, 151)
for k in (0, 1, 2):
    ax.plot(x, A[k, :150], color=HOLE, lw=0.6, zorder=1)
for arr, col, lab in (
    (A, C_TVB, f"TVB, mean of {len(A)} runs"),
    (B, C_F32, f"JAX float32, mean of {len(B)} runs"),
):
    m = arr[:, :150].mean(0)
    e = arr[:, :150].std(0, ddof=1) / np.sqrt(len(arr))
    ax.fill_between(x, m - e, m + e, color=col, alpha=0.22, lw=0)
    ax.plot(x, m, color=col, lw=1.8, label=lab)
late = float(np.concatenate([A[:, 100:], B[:, 100:]]).mean())
ax.axhline(
    late, color=INK2, lw=0.9, ls=(0, (4, 3)), label=f"mean after volume 100, both codes: {late:.3f}"
)
ax.plot([], [], color=HOLE, lw=0.8, label="three single TVB runs")
ax.set_ylim(0, 0.36)
ax.set_xlim(1, 150)
ax.set_xlabel(
    "scan volume after the 1,000-unit initial transient is removed (one volume = 72 time units)"
)
ax.set_ylabel("share of regions in the up state")
ax.set_title("First 150 volumes: mean over runs, band = ± 1 SE")
ax.legend(loc="upper right", fontsize=8)
ax = axes[1]
edges = np.arange(0, 833, 49)
xc = 0.5 * (edges[:-1] + edges[1:]) + 1
for arr, col, lab, off in ((A, C_TVB, "TVB", -5), (B, C_F32, "JAX float32", 5)):
    bm = np.stack([arr[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])], 1)
    m = bm.mean(0)
    e = bm.std(0, ddof=1) / np.sqrt(len(arr))
    ax.errorbar(
        xc + off,
        m,
        yerr=e,
        fmt="o-",
        ms=4,
        lw=1.2,
        color=col,
        mec=SURF,
        mew=0.8,
        capsize=2.5,
        elinewidth=1.1,
        label=lab,
    )
ax.axhline(late, color=INK2, lw=0.9, ls=(0, (4, 3)))
ax.set_ylim(0, 0.12)
ax.set_xlabel("scan volume (blocks of 49)")
ax.set_title("Whole record: block means ± 1 SE")
ax.legend(loc="upper right", fontsize=8)
save(fig, "w2_runs_over_time.png")
burn = dict(
    tvb_first=float(A[:, 0].mean()),
    port_first=float(B[:, 0].mean()),
    tvb20=ms(A[:, :20].mean(1)),
    port20=ms(B[:, :20].mean(1)),
    late=late,
)

# ---------------------------------------------------------------- figure 3: along G at sigma 0.030
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3))
rng = np.random.default_rng(1)
T3 = []
for ax, (key, title) in zip(axes, FOUR):
    mm, ee = [], []
    for G in Gs:
        v = [r[key] for r in cells.get((G, 0.03, ETA0, J0, D0, 0.01, False), [])]
        ax.scatter(np.full(len(v), G), v, s=7, color=C_F32, alpha=0.4, linewidths=0, zorder=2)
        m, e = ms(v)
        mm.append(m)
        ee.append(e)
    ax.errorbar(
        Gs,
        mm,
        yerr=ee,
        fmt="-",
        lw=1.4,
        color=C_F32,
        elinewidth=1.0,
        capsize=1.8,
        zorder=3,
        label="JAX float32, 4 runs per G",
    )
    for G in (0.53, 0.54, 0.55):
        v = [r[key] for r in tvb if at(r, G, 0.03) and r["dt"] == 0.01]
        m, e = ms(v)
        ax.errorbar(
            [G + 0.0022],
            [m],
            yerr=[e],
            fmt="o",
            ms=5,
            color=C_TVB,
            mec=SURF,
            mew=1.0,
            elinewidth=1.3,
            capsize=3,
            zorder=5,
            label=("TVB (8, 96, 8 runs)" if G == 0.53 else None),
        )
    ax.axvline(0.54, color=INK2, lw=0.7, ls=(0, (2, 3)), zorder=1)
    ax.set_title(title)
    ax.set_xlabel("G")
    ax.set_xlim(0.44, 0.66)
axes[0].legend(loc="center left", bbox_to_anchor=(0.0, 0.55), fontsize=7.5)
axes[0].annotate(
    "working point",
    (0.54, axes[0].get_ylim()[1]),
    xytext=(-3, -9),
    textcoords="offset points",
    ha="right",
    fontsize=7.5,
    color=INK2,
)
save(fig, "w3_along_G.png")
for G in (0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60):
    row = [f"{G:.2f}"]
    for key, _ in FOUR:
        m, e = ms([r[key] for r in cells[(G, 0.03, ETA0, J0, D0, 0.01, False)]])
        row.append(f"{m:.4f} ± {e:.4f}")
    T3.append(row)
T3t = []
for G in (0.53, 0.54, 0.55):
    rs = [r for r in tvb if at(r, G, 0.03) and r["dt"] == 0.01]
    T3t.append(
        [f"{G:.2f}", str(len(rs))] + ["%.4f ± %.4f" % ms([r[key] for r in rs]) for key, _ in FOUR]
    )

# ---------------------------------------------------------------- figure 4: offsets and step at the working point (mean rate)
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4), sharey=True)
rng = np.random.default_rng(2)
T4 = {}
for ax, (p, base, unit) in zip(axes[:3], (("eta", ETA0, "η"), ("J", J0, "J"), ("Delta", D0, "Δ"))):
    others = {"eta": ETA0, "J": J0, "Delta": D0}
    del others[p]
    rows = []
    vals = sorted(
        {
            round(r[p], 4)
            for r in pilot
            if at(r, 0.54, 0.03) and all(abs(r[q] - others[q]) < 1e-9 for q in others)
        }
    )
    w = (vals[-1] - vals[0]) / 60
    mm = []
    for v in vals:
        y = [
            r["m1_mean_r"]
            for r in pilot
            if at(r, 0.54, 0.03)
            and abs(r[p] - v) < 1e-9
            and all(abs(r[q] - others[q]) < 1e-9 for q in others)
        ]
        m, e = dots(ax, v, y, C_F32, rng, w)
        mm.append(m)
        rows.append(
            (
                v,
                len(y),
                m,
                e,
                ms(
                    [
                        r["m5_vaf2_r"]
                        for r in pilot
                        if at(r, 0.54, 0.03)
                        and abs(r[p] - v) < 1e-9
                        and all(abs(r[q] - others[q]) < 1e-9 for q in others)
                    ]
                ),
                ms(
                    [
                        r["switching_index"]
                        for r in pilot
                        if at(r, 0.54, 0.03)
                        and abs(r[p] - v) < 1e-9
                        and all(abs(r[q] - others[q]) < 1e-9 for q in others)
                    ]
                ),
            )
        )
    ax.plot(vals, mm, color=INK, lw=0.9, zorder=3)
    ax.axvline(base, color=INK2, lw=0.7, ls=(0, (2, 3)), zorder=1)
    ax.set_title(f"{unit} varied")
    ax.set_xlabel(unit)
    ax.set_xticks([vals[0], vals[1], base, vals[-2], vals[-1]])
    ax.ticklabel_format(useOffset=False, axis="x")
    T4[unit] = rows
axes[0].set_ylabel("mean rate")
ax = axes[3]
dts = [0.02, 0.01, 0.005, 0.0025]
rows = []
for k, dt in enumerate(dts):
    y = [r["m1_mean_r"] for r in tvb if at(r, 0.54, 0.03) and r["dt"] == dt]
    m, e = dots(ax, k - 0.17, y, C_TVB, rng, 0.07)
    rows.append(("TVB", dt, len(y), m, e))
    y = [
        r["m1_mean_r"]
        for r in port
        if at(r, 0.54, 0.03) and r["dt"] == dt and isbase(r) and not r.get("x64")
    ]
    if y:
        m, e = dots(ax, k + 0.17, y, C_F32, rng, 0.07)
        rows.append(("JAX float32", dt, len(y), m, e))
ax.scatter([], [], s=14, color=C_TVB, label="TVB")
ax.scatter([], [], s=14, color=C_F32, label="JAX float32")
ax.legend(loc="upper left", fontsize=7.5)
ax.set_xticks(range(4))
ax.set_xticklabels([str(d) for d in dts])
ax.set_xlabel("integration step")
ax.set_title("Step varied")
ax.grid(axis="x", visible=False)
T4["dt"] = rows
save(fig, "w4_offsets.png")

# ---------------------------------------------------------------- figure 5: maps
mean, se, sd, n = sm.grid(cells, Gs, Ss)
D, E = {}, {}
D["G"], E["G"] = sm.central(mean, se, Ga, 2)
D["sigma"], E["sigma"] = sm.central(mean, se, Sa, 1)
dl = dict(eta=0.005, J=0.02, Delta=0.005)
c0 = dict(eta=ETA0, J=J0, Delta=D0)
for p in dl:
    up = sm.grid(cells, Gs, Ss, **{**c0, p: c0[p] + dl[p]})
    dn = sm.grid(cells, Gs, Ss, **{**c0, p: c0[p] - dl[p]})
    D[p] = (up[0] - dn[0]) / (2 * dl[p])
    E[p] = np.hypot(up[1], dn[1]) / (2 * dl[p])
fine = sm.grid(cells, Gs, Ss, dt=0.005)
D["dt"] = fine[0] - mean
E["dt"] = np.hypot(fine[1], se)
occ = mean[1]
mixed = (occ > 0.02) & (occ < 0.90)
scale = np.where(sd > 0, sd, np.nan)
cG = D["G"] / scale
eG = E["G"] / scale
strong = np.sqrt(np.nansum(cG**2, 0)) > 3 * np.sqrt(np.nansum(eG**2, 0))
cos, eq, T5 = {}, {}, []
cosine_everywhere, cosine_used = (
    {},
    {},
)  # for drawing: every estimate, and which ones the summaries use
faded_cells = {}  # cells that have a cosine but are not used in the summaries
for p, lab in (
    ("eta", "η"),
    ("J", "J"),
    ("Delta", "Δ"),
    ("dt", "one halving of the time step"),
    ("sigma", "σ"),
):
    cP = D[p] / scale
    ok = np.isfinite(cP).all(0) & np.isfinite(cG).all(0) & strong & mixed
    num = np.nansum(cG * cP, 0)
    nG = np.sqrt(np.nansum(cG**2, 0))
    nP = np.sqrt(np.nansum(cP**2, 0))
    cos[p] = np.where(ok, num / (nG * nP), np.nan)
    cosine_everywhere[p], cosine_used[p] = num / (nG * nP), ok
    faded_cells[p] = np.isfinite(cosine_everywhere[p]) & ~ok
    eq[p] = np.where(ok, num / nG**2, np.nan)
    c = cos[p][ok]
    e = eq[p][ok]
    T5.append(
        [
            lab,
            str(int(ok.sum())),
            f"{np.median(c):+.3f}",
            f"{100 * np.mean(np.abs(c) > 0.95):.0f}%",
            f"{100 * np.mean(np.abs(c) > 0.8):.0f}%",
            "sign changes across the transition"
            if p == "sigma"
            else f"{np.percentile(e, 25):.3g} / {np.median(e):.3g} / {np.percentile(e, 75):.3g}",
        ]
    )
ext = [Gs[0] - 0.005, Gs[-1] + 0.005, Ss[0] - 0.002, Ss[-1] + 0.002]
fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.6))
for ax, k, title in zip(
    axes[0], (0, 3, 4, 5), ("Mean rate", "VAF (two components)", "Switching index", "Mean FC")
):
    im = ax.imshow(mean[k], origin="lower", extent=ext, aspect="auto", cmap=SEQ, vmin=0)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title)
for ax, p, title in zip(
    axes[1],
    ("eta", "J", "Delta", "sigma"),
    ("Cosine: G against η", "Cosine: G against J", "Cosine: G against Δ", "Cosine: G against σ"),
):
    im = faded_map(ax, cosine_everywhere[p], cosine_used[p], ext, DIV, -1, 1)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title)
for ax in axes[0]:
    ax.set_facecolor(HOLE)
for ax in axes.ravel():
    ax.grid(False)
    ax.set_xlabel("G")
    ax.set_ylabel("noise amplitude σ")
    ax.plot([0.54], [0.030], marker="+", color=INK, ms=10, mew=1.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
fig.tight_layout()
save(fig, "w5_maps.png")

# ---------------------------------------------------------------- figure 5b: the derivative maps
# every derivative is expressed as the change in the statistic for one increment of the parameter, so that rows can
# share a colour scale: G + 0.01, eta + 0.005, J + 0.02, Delta + 0.005, sigma + 0.004, and one halving of the step
INCREMENT = dict(G=0.01, eta=0.005, J=0.02, Delta=0.005, sigma=0.004, dt=1.0)
ROWS = (
    ("G", "change when\nG goes up 0.01"),
    ("eta", "change when\nη goes up 0.005"),
    ("J", "change when\nJ goes up 0.02"),
    ("Delta", "change when\nΔ goes up 0.005"),
    ("sigma", "change when\nσ goes up 0.004"),
    ("dt", "change when the\ntime step is halved"),
)
COLUMNS = ((0, "Mean rate"), (3, "VAF (two components)"), (5, "Mean FC"), (6, "corr(FC, SC)"))
fig, axes = plt.subplots(len(ROWS), len(COLUMNS), figsize=(12.5, 12.2), sharex=True, sharey=True)
for col, (k, title) in enumerate(COLUMNS):
    per_increment = {p: D[p][k] * INCREMENT[p] for p, _ in ROWS}
    resolved = {p: np.abs(D[p][k]) > 2 * E[p][k] for p, _ in ROWS}
    shown = np.concatenate([per_increment[p][resolved[p]] for p in ("G", "eta", "J", "Delta")])
    limit = np.percentile(np.abs(shown), 97)
    for row, (p, label) in enumerate(ROWS):
        ax = axes[row, col]
        im = faded_map(ax, per_increment[p], resolved[p], ext, DIV, -limit, limit)
        ax.grid(False)
        ax.plot([0.54], [0.030], marker="+", color=INK, ms=8, mew=1.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
        if row == 0:
            ax.set_title(title)
        if col == 0:
            ax.set_ylabel("noise amplitude σ")
            ax.annotate(
                label,
                xy=(-0.42, 0.5),
                xycoords="axes fraction",
                rotation=90,
                ha="center",
                va="center",
                fontweight="bold",
            )
        if row == len(ROWS) - 1:
            ax.set_xlabel("G")
    colourbar = fig.colorbar(
        im, ax=axes[:, col], orientation="horizontal", fraction=0.025, pad=0.045, aspect=30
    )
    colourbar.locator = matplotlib.ticker.MaxNLocator(4)
    colourbar.update_ticks()
save(fig, "w5b_derivative_maps.png")

# the working point, written out: how one cosine is computed
T5w = []
scaled_G, scaled_eta = D["G"][:, i0, j0] / sd[:, i0, j0], D["eta"][:, i0, j0] / sd[:, i0, j0]
for k, name in enumerate(NAMES):
    T5w.append(
        [
            name,
            f"{D['G'][k, i0, j0] * 0.01:+.4f}",
            f"{D['eta'][k, i0, j0] * 0.005:+.4f}",
            f"{sd[k, i0, j0]:.4f}",
            f"{scaled_G[k] * 0.01:+.2f}",
            f"{scaled_eta[k] * 0.005:+.2f}",
        ]
    )
wp_cosine = float(
    scaled_G @ scaled_eta / np.sqrt((scaled_G @ scaled_G) * (scaled_eta @ scaled_eta))
)
wp_rate = float(scaled_G @ scaled_eta / (scaled_G @ scaled_G))
assert abs(wp_cosine - cos["eta"][i0, j0]) < 1e-9 and abs(wp_rate - eq["eta"][i0, j0]) < 1e-9
n_nonfinite = sum(
    1
    for p in glob.glob(os.path.join(SWEEP, "*.json"))
    for r in json.load(open(p))["runs"]
    if not r["finite"]
)
n_runs = sum(len(json.load(open(p))["runs"]) for p in glob.glob(os.path.join(SWEEP, "*.json")))

# ---------------------------------------------------------------- figure 6: run time
TVB_PARALLEL = 6  # the TVB batch ran six processes side by side (tvb_ref_batch.py 6)
FIRST_VERSION = dict(
    us_step=3.6, s_run=22.0
)  # first JAX version; from doc/LOG.md, that timing run was not saved
first_tvb = [
    r for r in wp_t if r["seed"] <= 8
]  # the first TVB batch, before any GPU job shared the machine
tvb_s = float(np.median([r["wall_s"] for r in first_tvb]))
tvb_steps = (first_tvb[0]["T"] + first_tvb[0]["warm"]) / first_tvb[0]["dt"]
tvb_us = tvb_s / tvb_steps * 1e6
timing = []
for p in (
    sorted(glob.glob(os.path.join(PORT, "*.json")))
    + sorted(glob.glob(os.path.join(SWEEP, "*.json")))
    + [os.path.join(os.path.dirname(SWEEP.rstrip("/")), "pilot_done.json")]
):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if isinstance(d, dict) and "timing" in d and d["runs"][0].get("T", 0) > 50000:
        t = d["timing"]
        timing.append(
            dict(
                file=os.path.basename(p),
                B=t["B"],
                dt=d["runs"][0]["dt"],
                x64=bool(d["runs"][0].get("x64")),
                s_run=t["s_per_run"],
                us=t["us_per_step_per_run"],
            )
        )
full = [t for t in timing if t["B"] == 1024 and t["dt"] == 0.01 and not t["x64"]]
# which card ran which sweep batch, from the two queue workers' logs on the run machine
RTX_3060 = {
    "a_dt0.01_00",
    "a_dt0.01_02",
    "a_dt0.01_04",
    "a_dt0.01_06",
    "a_dt0.01_07",
    "b_dt0.005_01",
    "c_f64row_00",
}
RTX_3050 = {"a_dt0.01_01", "a_dt0.01_03", "a_dt0.01_05", "a_dt0.01_08", "b_dt0.005_00"}
fast = [t for t in full if t["file"][:-5] in RTX_3060]
slow = [t for t in full if t["file"][:-5] in RTX_3050]
assert len(fast) + len(slow) == len(full) and fast and slow
jax_fast_s, jax_slow_s = float(np.median([t["s_run"] for t in fast])), float(
    np.median([t["s_run"] for t in slow])
)
jax_fast_us, jax_slow_us = float(np.median([t["us"] for t in fast])), float(
    np.median([t["us"] for t in slow])
)
half = [t for t in timing if t["B"] == 1024 and t["dt"] == 0.005][0]
f64 = [t for t in timing if t["x64"] and t["file"].startswith("c_")][0]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axes[0]
bars = [
    (f"TVB, {TVB_PARALLEL} CPU processes", tvb_s / TVB_PARALLEL, C_TVB, 1.0),
    ("JAX, first version (from the log)", FIRST_VERSION["s_run"], C_F32, 0.45),
    ("JAX, RTX 3050, 1,024 at once", jax_slow_s, C_F32, 1.0),
    ("JAX, RTX 3060, 1,024 at once", jax_fast_s, C_F32, 1.0),
]
for i, (lab, v, col, al) in enumerate(bars):
    ax.barh(i, v, height=0.55, color=col, alpha=al, zorder=2)
    ax.annotate(
        f"{v:.1f} s" if v >= 10 else f"{v:.2f} s",
        (v, i),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        fontsize=8.5,
    )
ax.set_yticks(range(len(bars)))
ax.set_yticklabels([b[0] for b in bars])
ax.invert_yaxis()
ax.set_xlim(0, tvb_s / TVB_PARALLEL * 1.15)
ax.set_xlabel("wall time per full run, seconds")
ax.set_title("Time per run of 61,000 time units")
ax.grid(axis="y", visible=False)
ax = axes[1]
for x64, col, lab in ((False, C_F32, "JAX float32"), (True, C_F64, "JAX float64")):
    pts = sorted(
        (t["B"], t["us"])
        for t in timing
        if t["dt"] == 0.01
        and t["x64"] == x64
        and (t["B"] < 1024 or t in fast)
        and t["file"] != "a_dt0.01_08.json"
    )
    best = {}
    for b_, u in pts:
        best[b_] = min(u, best.get(b_, 1e9))
    xs, ys = list(best), list(best.values())
    ax.plot(xs, ys, "o-", color=col, lw=1.6, ms=5.5, mec=SURF, mew=1.0, label=lab)
ax.set_xscale("log")
ax.set_xticks([32, 64, 128, 256, 512, 1024])
ax.set_xticklabels(["32", "64", "128", "256", "512", "1,024"])
ax.minorticks_off()
ax.set_ylim(0, None)
ax.set_xlabel("simulations run at once (batch size)")
ax.set_ylabel("microseconds per step per run")
ax.set_title("JAX cost per step falls with batch size")
ax.legend(loc="upper right", fontsize=8)
save(fig, "w6_run_time.png")
T6 = [
    [
        f"TVB, one of {TVB_PARALLEL} CPU processes",
        f"{tvb_us:.0f}",
        f"{tvb_s:.0f}",
        f"{tvb_s / TVB_PARALLEL:.0f}",
        "result files",
    ],
    [
        "JAX, first version",
        f"{FIRST_VERSION['us_step']:.1f}",
        "",
        f"{FIRST_VERSION['s_run']:.0f}",
        "working log",
    ],
    [
        "JAX float32, 1,024 at once (RTX 3050)",
        f"{jax_slow_us:.2f}",
        "",
        f"{jax_slow_s:.2f}",
        "result files",
    ],
    [
        "JAX float32, 1,024 at once (RTX 3060)",
        f"{jax_fast_us:.2f}",
        "",
        f"{jax_fast_s:.2f}",
        "result files",
    ],
    [
        "JAX float32, step 0.005, 1,024 at once (RTX 3050)",
        f"{half['us']:.2f}",
        "",
        f"{half['s_run']:.2f}",
        "result files",
    ],
    [
        f"JAX float64, {f64['B']} at once (RTX 3060)",
        f"{f64['us']:.2f}",
        "",
        f"{f64['s_run']:.2f}",
        "result files",
    ],
]
sweep_gpu_h = (
    sum(json.load(open(p))["timing"]["wall_s"] for p in glob.glob(os.path.join(SWEEP, "*.json")))
    / 3600
)
# ---------------------------------------------------------------- figure 7: region by region
from sweep_per_region import PerRegion

study = PerRegion(os.path.join(os.path.dirname(SWEEP.rstrip("/")), "node_stats_sweep.npz"))
hub = study.per_seed_hub_preference()
hub_floor = max(abs(x) for x in study.noise_floor()[0])
hub_gap = [g - e for g, e in zip(hub["G"], hub["eta"])]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
ax = axes[0]
rng = np.random.default_rng(3)
for x, (name, label) in enumerate((("G", "G"), ("eta", "η"), ("J", "J"), ("Delta", "Δ"))):
    mean_hub, se_hub = dots(ax, x, hub[name], C_F32, rng, 0.1)
    ax.annotate(
        f"{mean_hub:.3f}\n± {se_hub:.3f}",
        (x, max(hub[name])),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK2,
    )
ax.set_xticks(range(4))
ax.set_xticklabels(["G", "η", "J", "Δ"])
ax.set_xlim(-0.6, 3.6)
ax.margins(y=0.35)
ax.set_ylabel("hub preference of the response")
ax.set_title("How much more strongly connected regions respond")
ax.grid(axis="x", visible=False)
ax = axes[1]
strength = study.in_strength
for name, label, colour, order in (("eta", "raising η", INK2, 2), ("G", "raising G", C_F32, 3)):
    share = study.profile(name)
    ax.scatter(
        strength, share, s=16, color=colour, alpha=0.75, linewidths=0, zorder=order, label=label
    )
    slope, intercept = np.polyfit(strength, share, 1)
    ax.plot(
        [strength.min(), strength.max()],
        [intercept + slope * strength.min(), intercept + slope * strength.max()],
        color=colour,
        lw=1.4,
        zorder=order,
    )
ax.set_xlabel("in-strength of the region (sum of its connection weights)")
ax.set_ylabel("region's response / mean response")
ax.set_title("Each region's share of the response (84 regions)")
ax.legend(loc="upper left", fontsize=8)
save(fig, "w7_per_region.png")

# ---------------------------------------------------------------- page (LaTeX -> PDF)
pm = lambda t: f"{t[0]:.4f} ± {t[1]:.4f}"
t4rows = []
for unit, rows in T4.items():
    if unit == "dt":
        continue
    for v, nn, m_, e, vaf, si in rows:
        t4rows.append([unit, f"{v:g}", str(nn), f"{m_:.4f} ± {e:.4f}", pm(vaf), pm(si)])
t4dt = [[c, f"{dt:g}", str(nn), f"{m_:.4f} ± {e:.4f}"] for c, dt, nn, m_, e in T4["dt"]]
m = lambda G: ms([r["m1_mean_r"] for r in cells[(G, 0.03, ETA0, J0, D0, 0.01, False)]])[0]
si_all = [r["switching_index"] for r in wp_t + wp_32 + wp_64]
si_a = ms([r["switching_index"] for r in wp_t if r["seed"] <= 32])
si_b = ms([r["switching_index"] for r in wp_t if r["seed"] > 32])
# blocks: ("h1"|"meta"|"h2"|"p"|"note", text), ("ul", [items]), ("fig", file, caption), ("table", head, rows). Inline: **bold**, `code`.
B = [
    ("h1", "Which parameters can the summary statistics tell apart?"),
    (
        "meta",
        "Draft, machine-written, 2026-11-24. Not yet rewritten in the author's words. Every number is computed from the run files by `src/sweep/writeup_sweep.py`. All values are mean ± one standard error over runs.",
    ),
    ("h2", "What was run"),
    (
        "ul",
        [
            "**Model.** 84 regions (`hcp-001` connectome), every region is the same Montbrió–Pazó–Roxin model with η = −5, J = 15, Δ = 1 (these three are set once for the whole network, not per region), delays at speed 20, stochastic Heun at step 0.01, noise [σ, 2σ]. These are the settings of the group's companion code; that the paper used them is an inference.",
            f"**TVB.** The Virtual Brain's standard library (`tvb-library` 2.10.0, numba backend), run through the group's own code. {len(wp_t)} runs at the paper's working point (G 0.54, σ 0.030), fewer at neighbouring G and at other steps.",
            "**JAX.** Our own rewrite of that TVB kernel in JAX (`src/core/mprjax.py`), run on two GPUs (an RTX 3060 and an RTX 3050). It is not the vbjax library, whose container image served only as the GPU environment. It reproduces TVB to 2 × 10^{-12} over 3,000 steps when given TVB's own noise.",
            f"**Sweep.** {n_runs:,} JAX runs of 61,000 time units (the first 1,000 removed as initial transient): G 0.40–0.70 × σ 0.018–0.054, each point with η, J and Δ perturbed up and down (one at a time, in all 84 regions together), and again at half the time step. Four runs per setting.",
        ],
    ),
    (
        "p",
        "**Terms.** A **run** is one simulation: one parameter setting, one random seed, one starting state. Runs with "
        "the same setting and different seeds are replicates, and every standard error here is the spread across "
        "replicates. Nothing here is a Markov chain Monte Carlo chain: no sampler chose the parameter values. They "
        "sit on a fixed grid, and every grid point gets the same number of runs. A **perturbation** is a small fixed "
        "change to one of η, J or Δ, made in all 84 regions together. An **increment** is the amount a parameter is "
        "moved in a difference: the grid spacing for G and σ, the perturbation for η, J and Δ. **Time step** always "
        "means the integration step. **Noise amplitude** means the model parameter σ; the spread between replicates "
        "is called sampling variability.",
    ),
    (
        "table",
        ["Quantity", "Where", "Runs behind each value"],
        [
            ["Statistic at a grid point", "all 310 points", "4 (JAX)"],
            [
                "Derivative in G or σ",
                "all 310 points",
                "8: the two neighbouring points, 4 each (one neighbour and the point itself at the grid's edge)",
            ],
            [
                "Derivative in η, J or Δ",
                "all 310 points",
                "8: 4 perturbed up and 4 perturbed down, at that point",
            ],
            [
                "Effect of halving the time step",
                "all 310 points",
                "8: 4 at the full time step, 4 at half",
            ],
            ["Double precision check", "the σ = 0.030 row, 31 points", "4 per point"],
            [
                "TVB against JAX",
                "working point; fewer TVB runs at neighbouring G and at other steps",
                f"{len(wp_t)} TVB; JAX as in section 1",
            ],
        ],
        r"p{0.27\linewidth}p{0.24\linewidth}p{0.42\linewidth}",
    ),
    ("h2", "The code that ran it"),
    (
        "p",
        "Repository: `https://github.com/pmahon3/tvb-error-analysis`. The sweep's simulations were made by four files.",
    ),
    (
        "table",
        ["Step", "File", "What it does"],
        [
            [
                "Write the jobs",
                "`src/sweep/sweep_make_jobs.py`",
                "Lays out the (G, σ) grid with the η, J and Δ perturbations; the job files are kept in `results/runs/sweep/jobs/`",
            ],
            [
                "Queue",
                "`src/core/mprjax_worker.py`",
                "One worker per GPU hands job files to the runner",
            ],
            [
                "Run one batch",
                "`src/core/mprjax_run.py`",
                "Runs about 1,000 simulations at once, computes the statistics, writes one result file",
            ],
            [
                "Simulate",
                "`src/core/mprjax.py`",
                "The model and the integrator: `f` is the node equations, `gather` the delayed coupling between regions, "
                "`sample` the stochastic Heun step; `build` adds the starting state, the noise, the BOLD model and the "
                "per-volume sums",
            ],
        ],
        r"l l >{\raggedright\arraybackslash}p{0.5\linewidth}",
    ),
    (
        "ul",
        [
            "Pattern statistics: `src/core/stats.py`.",
            "TVB reference runs: `src/tvb_reference/tvb_ref_run.py`, which calls the group's companion code unchanged.",
            "Checks of the JAX code against TVB: `src/validation/`.",
            "Software versions, the container image and the launch commands: `src/README.md`.",
        ],
    ),
    ("h2", "1. The two codes give the same distributions"),
    (
        "fig",
        "w1_working_point.png",
        "Each dot is one run at the working point (G 0.54, σ 0.030, step 0.01). Black marker: mean ± 1 SE, also printed above each group.",
    ),
    (
        "table",
        [
            "Statistic",
            f"TVB (n = {len(wp_t)})",
            f"JAX, float32 (n = {len(wp_32)})",
            f"JAX, float64 (n = {len(wp_64)})",
        ],
        T1,
    ),
    (
        "p",
        f"The spread between runs is large compared with the standard errors: single runs give a switching index anywhere from {min(si_all):.3f} to {max(si_all):.3f}. A comparison of pattern statistics from a handful of runs is dominated by this spread. The first 32 TVB runs alone gave {pm(si_a)}; the next 64 gave {pm(si_b)}.",
    ),
    ("h2", "2. Runs are not yet stationary after the initial transient is removed"),
    (
        "fig",
        "w2_runs_over_time.png",
        "Share of regions in the up state (high firing rate) per scan volume at the working point. Left: mean over runs with a ± 1 SE band, and three single TVB runs in grey. Right: means over blocks of 49 volumes ± 1 SE. Dashed line: mean after volume 100, both codes pooled.",
    ),
    (
        "p",
        f"Both codes start every region at a rate between 0 and 2. In the first volume after the removed transient the share of regions in the up state is {burn['tvb_first']:.3f} (TVB) and {burn['port_first']:.3f} (JAX), against {burn['late']:.3f} later. Over the first 20 volumes it averages {pm(burn['tvb20'])} and {pm(burn['port20'])}. At this working point the effect on whole-record statistics is under 3%. At lower noise (σ ≤ 0.026) near the transition it is not small: the starting state persists much longer there.",
    ),
    ("h2", "3. The working point lies on the steep part of the transition in G"),
    (
        "fig",
        "w3_along_G.png",
        "Statistics along G at σ 0.030. Orange dots: single JAX runs; orange line: their mean ± 1 SE (4 runs per G). Blue: TVB mean ± 1 SE at G 0.53, 0.54, 0.55, drawn slightly to the right. Dotted line: the working point.",
    ),
    ("table", ["G (JAX, 4 runs)"] + [t for _, t in FOUR], T3),
    ("table", ["G (TVB)", "Runs"] + [t for _, t in FOUR], T3t),
    (
        "p",
        f"Equal increments of G do not have equal effects here. In the JAX code's four runs, G 0.53 → 0.54 raises the mean rate by {m(0.54) - m(0.53):+.3f} and G 0.54 → 0.55 by {m(0.55) - m(0.54):+.3f}. The pattern statistics peak within one or two grid increments of the working point.",
    ),
    ("h2", "4. Small changes in η, J, Δ and the time step move the same statistics"),
    (
        "fig",
        "w4_offsets.png",
        "Mean rate at the working point. First three panels: one of η, J or Δ changed at a time, by the same amount in all 84 regions, 16 JAX (float32) runs per value; dots are single runs, black markers mean ± 1 SE, dotted line the nominal value. Fourth panel: integration step, TVB and JAX.",
    ),
    ("table", ["Parameter", "Value", "Runs", "Mean rate", "VAF", "Switching index"], t4rows),
    ("table", ["Code", "Step", "Runs", "Mean rate"], t4dt),
    (
        "p",
        f"A change of 0.1% in η (−5 → −4.995) raises the mean rate by {T4['η'][4][2] - T4['η'][3][2]:+.3f} (16 runs each). The grid increment of G below the working point (0.53 → 0.54) raises it by {m(0.54) - m(0.53):+.3f}. Halving the time step raises the mean rate at every halving, in TVB and in the JAX code.",
    ),
    (
        "h2",
        "5. Across the map, G, η, J and Δ have nearly collinear effects on the whole-network statistics",
    ),
    (
        "p",
        "**How the derivative maps are built.** At each of the 310 grid points every statistic is averaged over its "
        "four runs. Its derivative with respect to G, and to σ, is the difference between the two neighbouring grid "
        "points divided by their distance. For η, J and Δ, each grid point has extra runs with that parameter perturbed "
        "up and down, and the derivative is their difference divided by twice the perturbation. For the time step it is "
        "the value at half the time step minus the value at the full time step. Each derivative has a standard error from the "
        "run-to-run spread. Every estimate is drawn; where it is smaller than twice that error the cell is faded.",
    ),
    (
        "latex",
        r"\textbf{In symbols.} Write $\theta=(G,\sigma,\eta,J,\Delta,h)$ for the parameters, with $h$ the time step. "
        r"A run with seed $k$ gives a statistic $s_k(\theta)$. Over the $n=4$ runs at one setting, "
        r"\[\bar s(\theta)=\frac{1}{n}\sum_{k=1}^{n}s_k(\theta),\qquad "
        r"\operatorname{se}(\theta)=\frac{\operatorname{sd}_k\,s_k(\theta)}{\sqrt{n}}.\] "
        r"The slope of $\bar s$ in a parameter $p$ is estimated by a central difference. For $p\in\{\eta,J,\Delta\}$, "
        r"with base value $p_0$ and perturbation $\delta_p$, "
        r"\[D_p s\,(G,\sigma)=\frac{\bar s(G,\sigma;\,p_0+\delta_p)-\bar s(G,\sigma;\,p_0-\delta_p)}{2\,\delta_p},\] "
        r"and for $G$, along the grid $G_1<G_2<\dots$, "
        r"\[D_G s\,(G_j,\sigma)=\frac{\bar s(G_{j+1},\sigma)-\bar s(G_{j-1},\sigma)}{G_{j+1}-G_{j-1}},\] "
        r"and likewise for $\sigma$ (one-sided at the edge of the grid). For the time step the map shows the difference "
        r"$\bar s(h/2)-\bar s(h)$. The standard error of a slope is $\sqrt{\operatorname{se}_+^2+\operatorname{se}_-^2}$ "
        r"divided by the distance between the two points.",
    ),
    (
        "latex",
        r"The \emph{increment} $\delta_p$ is the fixed small amount by which $p$ is moved: the grid spacing for $G$ (%g) and "
        r"$\sigma$ (%g), and the perturbation for $\eta$ (%g), $J$ (%g) and $\Delta$ (%g). A panel shows the change in the "
        r"statistic for one increment, "
        r"\[M_{p,s}(G,\sigma)=D_p s\,(G,\sigma)\;\delta_p\;\approx\;\frac{\partial\bar s}{\partial p}\,\delta_p,\] "
        r"so every row is in the statistic's own units and a column can share one colour scale. A cell is faded when "
        r"$|D_p s|<2\operatorname{se}(D_p s)$. The slopes $D_p s$ do not depend on the increment: to compare two "
        r"parameters, compare their slopes, not the colours."
        % (
            INCREMENT["G"],
            INCREMENT["sigma"],
            INCREMENT["eta"],
            INCREMENT["J"],
            INCREMENT["Delta"],
        ),
    ),
    (
        "fig",
        "w5b_derivative_maps.png",
        "Derivative maps. Rows: the parameter that is changed. Columns: the statistic that responds (four of the "
        "seven). Each panel shows how much that statistic changes for one increment of that parameter (red: up, blue: "
        "down), at each grid point; a column shares one colour scale. Both axes are model parameters: coupling G "
        "across, noise amplitude σ up. σ is not a standard deviation here. Faded: the change is under two standard errors, so its sign is not established, but the "
        "estimate is still drawn. Grey: no estimate, because a run there went non-finite. "
        "Cross: working point.",
    ),
    (
        "p",
        "The rows for G, η, J and Δ show the same pattern at similar strength: that is the collinearity, seen by eye. "
        "The σ row has a different pattern, with its sign changing across the transition.",
    ),
    (
        "p",
        "**How the collinearity maps are derived from them.** The collinearity maps are not a separate measurement. "
        "They are a summary of the derivative maps, computed at each grid point as follows.",
    ),
    (
        "ul",
        [
            "Divide each statistic's derivative by that statistic's run-to-run standard deviation at the point. This "
            "puts seven statistics with different units on one scale: how many run-to-run standard deviations the statistic moves.",
            "Stack the seven scaled derivatives for G into one vector of seven numbers, its sensitivity vector, and do the same for the other "
            "parameter. Each vector says how that parameter moves the seven statistics.",
            "Take the cosine similarity of the two vectors: their dot product divided by the product of their "
            "lengths. +1 means one vector is a positive multiple of the other, so a change in one parameter can be "
            "compensated exactly by a change in the other; 0 means the two vectors are orthogonal.",
            "The G-equivalent is that multiple: the dot product divided by the squared length of G's vector (the "
            "least-squares coefficient). It is the change in G that matches one unit of the other parameter.",
            "A grid point is used only where G's vector is longer than three times its standard error.",
        ],
    ),
    (
        "table",
        ["At the working point", "G + 0.01", "η + 0.005", "Run-to-run sd", "G, in sd", "η, in sd"],
        T5w,
    ),
    (
        "p",
        f"The cosine between the last two columns is {wp_cosine:+.3f}, and the G-equivalent is {wp_rate:.2f} of G per "
        "unit of η. Doing this at every grid point, for each parameter, gives the bottom row of the next figure.",
    ),
    (
        "fig",
        "w5_maps.png",
        "Top: four statistics over (G, σ), mean of 4 JAX runs. Bottom: at each point, the cosine between the change in seven statistics caused by G and the change caused by another parameter (each statistic scaled by its run-to-run spread). +1: the two changes are proportional. Faded: the cosine is drawn but not used in the summaries, because G's own effect there is under three standard errors (the unresponsive corner of the grid). Grey: no estimate, because a run there or next to it went non-finite; a separate note, `doc/grey_cells.pdf`, says what those are. Cross: working point.",
    ),
    (
        "table",
        [
            "G against",
            "Points",
            "Median cosine",
            "|cosine| > 0.95",
            "|cosine| > 0.8",
            "G-equivalent per unit (quartiles)",
        ],
        T5,
    ),
    (
        "ul",
        [
            "In the median, one grid increment of G (0.01) has the effect of η + 0.012 (0.23%), J + 0.025 (0.16%) or Δ + 0.010 (1%) does.",
            "The cosines fall short of 1 by as much as two halves of the runs disagree with each other about a single parameter, so the shortfall is sampling variability. So point by point no separation of G from η, J or Δ shows above the sampling variability of four runs. A sharper test, pooled over the map, does find a small one: take the runs made with η, J or Δ perturbed up, find the G whose base runs have the same mean rate, and compare the other statistics. Below the transition the perturbed runs have lower VAF there (by 0.003 to 0.005, where one grid increment of G changes VAF by 0.004 to 0.010), and the reverse for downward perturbations. At equal mean rate, more coupling and less drive gives a more coordinated network. Section 7 looks region by region.",
            "σ is the one parameter the statistics see separately: more noise acts like more coupling below G ≈ 0.56 and like less above it.",
            "The time step is the least resolved. For the level statistics, halving it acts like a small increase in G.",
            "In the faded cells, which the summaries leave out because G's own effect is under three standard "
            "errors, the cosines of G with η, J and Δ are still positive in "
            + " and ".join(
                f"{100 * np.mean(cosine_everywhere[p][faded_cells[p]] > 0):.0f}%"
                for p in ("eta", "J", "Delta")
            )
            + " of cells (medians "
            + ", ".join(
                f"{np.median(cosine_everywhere[p][faded_cells[p]]):.2f}"
                for p in ("eta", "J", "Delta")
            )
            + "). The collinearity does not stop at the edge of the cells that pass the rule.",
            "Because the cosines are computed from the derivatives, agreement between the two kinds of map is "
            "automatic and checks nothing. The independent check is the one in the second bullet: compare actual "
            "perturbed runs with actual base runs.",
        ],
    ),
    (
        "note",
        "This was expected from the equations: η, J·r and G × (input from other regions) enter the same equation as a sum. The prediction and its probability (0.8) were written down before the sweep. What the sweep adds is the sizes.",
    ),
    ("h2", "6. Run time"),
    (
        "fig",
        "w6_run_time.png",
        "Left: wall time per full run on one machine. TVB ran six processes side by side on six CPU cores; the bar is one "
        "process's time divided by six. JAX bars are a batch's wall time divided by the number of runs in it; the card "
        "that ran each batch is taken from the workers' logs. The pale bar is the first JAX version, from the working log. Right: "
        "time on the GPU per integration step per run, against how many runs share the batch. The 40-, 912- and 1,024-run points are the RTX 3060; the card was not recorded for the 32- and 64-run points.",
    ),
    (
        "table",
        ["Code", "μs per step per run", "One process, s", "Wall time per run, s", "Source"],
        T6,
        r"p{0.34\linewidth}rrrl",
    ),
    (
        "ul",
        [
            f"At 1,024 runs per batch the RTX 3060 finishes a run about {tvb_s / TVB_PARALLEL / jax_fast_s:.0f} times sooner than the six-process TVB batch. The whole sweep took {sweep_gpu_h:.1f} GPU-hours; {n_runs:,} runs at TVB's rate would take about {n_runs * tvb_s / TVB_PARALLEL / 3600:.0f} hours.",
            "This compares throughput on different hardware (six CPU cores against one GPU), not like-for-like speed.",
            "TVB computes in double precision and the sweep ran in single. Double precision in JAX costs a little over twice as much at equal batch size (2.2 times at 32 runs, 2.6 times at 64).",
            "JAX wall time includes noise generation, the statistics and compilation. TVB's is the simulation only.",
            "The first JAX version was less than twice as fast as the TVB batch: it copied the whole delay history at every step. The current one updates the history once per output sample, which needs every delay between connected regions to be at least one output sample long. That holds for this connectome and the code checks it.",
        ],
    ),
    ("h2", "7. Region by region, G does separate from η, J and Δ"),
    (
        "p",
        "Sections 1 to 5 use whole-network statistics. Here the same runs are read one region at a time. η adds the "
        "same push to every region; G multiplies the input a region receives, so strongly connected regions get more. "
        'The "hub preference" of a response is how fast it rises with a region\'s in-strength, divided by its mean '
        "over regions.",
    ),
    (
        "fig",
        "w7_per_region.png",
        f"Left: hub preference of the response of each region's mean rate to G, η, J and Δ, pooled over "
        f"{len(study.points)} grid points (σ ≥ 0.030); one dot per seed, black marker mean ± 1 SE. Right: each region's "
        "share of the pooled response against its in-strength, for G and for η, with least-squares lines.",
    ),
    (
        "ul",
        [
            f"G minus η: {np.mean(hub_gap):+.3f} ± {np.std(hub_gap, ddof=1) / 2:.3f} over the four seeds, "
            f"{np.mean(hub_gap) / (np.std(hub_gap, ddof=1) / 2):.0f} standard errors from zero. The same quantity "
            f"between two halves of the seeds, for G alone, is at most {hub_floor:.4f}.",
            f"G's response is tilted towards strongly connected regions by about "
            f"{np.mean(hub_gap) / np.mean(hub['G']):.0%} more than η's. J and Δ sit with η.",
            "So a region-resolved statistic separates G from the three regional-model parameters, in the direction the "
            "equations give. Those three do not separate from each other this way.",
            "The statistic and the rule were written down before the per-region numbers were opened. Limits: four "
            "seeds, one connectome, mean rate per region only.",
        ],
    ),
    ("h2", "Not shown"),
    (
        "ul",
        [
            "**The paper's own fluidity statistic, and region-by-region connectivity patterns.** Not analysed. Section 7 uses only each region's mean rate.",
            "**Switching index.** Too noisy at four runs for anything beyond the shift of its peak with η.",
            "**σ ≤ 0.026 near the transition.** Those statistics still carry the decay from the starting state.",
            f"**High G with high σ.** {n_nonfinite} of {n_runs:,} runs went non-finite there. A separate note, `doc/grey_cells.pdf`, says what those are.",
            "**TVB across the grid.** The maps are the JAX code's. TVB was run at the working point, two neighbouring G values and four steps.",
            "**Three of the seven statistics** (mean rate, high-state occupancy, median high-state share) are nearly one statistic, which makes the cosines high cheaply.",
        ],
    ),
    (
        "meta",
        "Full record: `doc/LOG.md`, entry of 2026-11-24 17:35. Figures: `results/figures/`. Rebuild: `src/runall.sh`.",
    ),
]
build_pdf(B, figs, OUT)
