"""Figures and one self-contained page for the sweep of 2026-11-24 (doc/LOG.md, 17:35 entry). Every number in the page is
computed here from the run files; nothing is typed in by hand except the text.
Usage (from the repository root): python3 -I src/sweep/writeup_sweep.py results/runs/sweep/done results/runs/Aprime results/runs/port results doc/sweep_writeup.pdf
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
from matplotlib.colors import LinearSegmentedColormap
import sweep_maps as sm
from sweep_maps import STATS, NAMES, ETA0, J0, D0

SWEEP, APRIME, PORT, FIG, OUT = sys.argv[1:6]
FIG = os.path.join(FIG, "figures", "report")
os.makedirs(FIG, exist_ok=True)
# palette: first three categorical slots (validated all-pairs on the light surface), blue sequential ramp, blue-grey-red diverging
C_TVB, C_F32, C_F64 = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, GRID, HOLE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIV = LinearSegmentedColormap.from_list(
    "div", ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#8f1f1f"]
)
plt.rcParams.update(
    {
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
    }
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
groups = [("TVB", wp_t, C_TVB), ("port,\nfloat32", wp_32, C_F32), ("port,\nfloat64", wp_64, C_F64)]
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

# ---------------------------------------------------------------- figure 2: chains over time
tr = np.load(os.path.join(PORT, "traces_wp.npz"))
A, B = tr["tvb"], tr["prec2_f32"]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axes[0]
x = np.arange(1, 151)
for k in (0, 1, 2):
    ax.plot(x, A[k, :150], color=HOLE, lw=0.6, zorder=1)
for arr, col, lab in (
    (A, C_TVB, f"TVB, mean of {len(A)} runs"),
    (B, C_F32, f"port float32, mean of {len(B)} runs"),
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
ax.set_xlabel("scan volume after the 1,000-unit discard (one volume = 72 time units)")
ax.set_ylabel("share of regions in the high state")
ax.set_title("First 150 volumes: mean over runs, band = ± 1 SE")
ax.legend(loc="upper right", fontsize=8)
ax = axes[1]
edges = np.arange(0, 833, 49)
xc = 0.5 * (edges[:-1] + edges[1:]) + 1
for arr, col, lab, off in ((A, C_TVB, "TVB", -5), (B, C_F32, "port float32", 5)):
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
save(fig, "w2_chains.png")
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
        label="port float32, 4 runs per G",
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
        rows.append(("port float32", dt, len(y), m, e))
ax.scatter([], [], s=14, color=C_TVB, label="TVB")
ax.scatter([], [], s=14, color=C_F32, label="port float32")
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
for p, lab in (
    ("eta", "η"),
    ("J", "J"),
    ("Delta", "Δ"),
    ("dt", "one halving of the step"),
    ("sigma", "σ"),
):
    cP = D[p] / scale
    ok = np.isfinite(cP).all(0) & np.isfinite(cG).all(0) & strong & mixed
    num = np.nansum(cG * cP, 0)
    nG = np.sqrt(np.nansum(cG**2, 0))
    nP = np.sqrt(np.nansum(cP**2, 0))
    cos[p] = np.where(ok, num / (nG * nP), np.nan)
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
    im = ax.imshow(cos[p], origin="lower", extent=ext, aspect="auto", cmap=DIV, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title)
for ax in axes.ravel():
    ax.set_facecolor(HOLE)
    ax.grid(False)
    ax.set_xlabel("G")
    ax.set_ylabel("σ")
    ax.plot([0.54], [0.030], marker="+", color=INK, ms=10, mew=1.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
fig.tight_layout()
save(fig, "w5_maps.png")
n_nonfinite = sum(
    1
    for p in glob.glob(os.path.join(SWEEP, "*.json"))
    for r in json.load(open(p))["runs"]
    if not r["finite"]
)
n_runs = sum(len(json.load(open(p))["runs"]) for p in glob.glob(os.path.join(SWEEP, "*.json")))

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
        'Draft, machine-written, 2026-11-24. Not yet rewritten in the author\'s words. Every number is computed from the run files by `src/sweep/writeup_sweep.py`. All values are mean ± one standard error over independent runs ("chains": same parameters, different random seed and starting state).',
    ),
    ("h2", "What was run"),
    (
        "ul",
        [
            "**Model.** 84 regions (`hcp-001` connectome), every region is the same Montbrió–Pazó–Roxin model with η = −5, J = 15, Δ = 1 (these three are set once for the whole network, not per region), delays at speed 20, stochastic Heun at step 0.01, noise [σ, 2σ]. These are the settings of the group's companion code; that the paper used them is an inference.",
            f"**Reference.** The Virtual Brain 2.10.0 (numba backend) through the group's own code. {len(wp_t)} runs at the paper's working point (G 0.54, σ 0.030), fewer at neighbouring G and at other steps.",
            "**Port.** A JAX rewrite of the same kernel on two GPUs. It reproduces TVB to 2 × 10^{-12} over 3,000 steps when given TVB's own noise.",
            f"**Sweep.** {n_runs:,} port runs of 61,000 time units (first 1,000 discarded): G 0.40–0.70 × σ 0.018–0.054, each point with η, J and Δ nudged up and down (one at a time, in all 84 regions together), and again at half the step. Four runs per setting.",
        ],
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
                "Lays out the (G, σ) grid with the η, J and Δ nudges; the job files are kept in `results/runs/sweep/jobs/`",
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
            "Checks of the port against TVB: `src/validation/`.",
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
            f"Port, float32 (n = {len(wp_32)})",
            f"Port, float64 (n = {len(wp_64)})",
        ],
        T1,
    ),
    (
        "p",
        f"The spread between runs is large compared with the standard errors: single runs give a switching index anywhere from {min(si_all):.3f} to {max(si_all):.3f}. A comparison of pattern statistics from a handful of runs is dominated by this spread. The first 32 TVB runs alone gave {pm(si_a)}; the next 64 gave {pm(si_b)}.",
    ),
    ("h2", "2. Chains are still settling after the discard"),
    (
        "fig",
        "w2_chains.png",
        "Share of regions in the high state per scan volume at the working point. Left: mean over runs with a ± 1 SE band, and three single TVB runs in grey. Right: means over blocks of 49 volumes ± 1 SE. Dashed line: mean after volume 100, both codes pooled.",
    ),
    (
        "p",
        f"Both codes start every region at a rate between 0 and 2. In the first volume after the discard the high-state share is {burn['tvb_first']:.3f} (TVB) and {burn['port_first']:.3f} (port), against {burn['late']:.3f} later. Over the first 20 volumes it averages {pm(burn['tvb20'])} and {pm(burn['port20'])}. At this working point the effect on whole-record statistics is under 3%. At lower noise (σ ≤ 0.026) near the transition it is not small: the starting state persists much longer there.",
    ),
    ("h2", "3. The working point sits on a knee in G"),
    (
        "fig",
        "w3_along_G.png",
        "Statistics along G at σ 0.030. Orange dots: single port runs; orange line: their mean ± 1 SE (4 runs per G). Blue: TVB mean ± 1 SE at G 0.53, 0.54, 0.55, drawn slightly to the right. Dotted line: the working point.",
    ),
    ("table", ["G (port, 4 runs)"] + [t for _, t in FOUR], T3),
    ("table", ["G (TVB)", "Runs"] + [t for _, t in FOUR], T3t),
    (
        "p",
        f"One grid step of G is not one size here. In the port's four runs, G 0.53 → 0.54 raises the mean rate by {m(0.54) - m(0.53):+.3f} and G 0.54 → 0.55 by {m(0.55) - m(0.54):+.3f}. The pattern statistics peak within one or two grid steps of the working point.",
    ),
    ("h2", "4. Small changes in η, J, Δ and the step move the same statistics"),
    (
        "fig",
        "w4_offsets.png",
        "Mean rate at the working point. First three panels: one of η, J or Δ changed at a time, by the same amount in all 84 regions, 16 port (float32) runs per value; dots are single runs, black markers mean ± 1 SE, dotted line the nominal value. Fourth panel: integration step, TVB and port.",
    ),
    ("table", ["Parameter", "Value", "Runs", "Mean rate", "VAF", "Switching index"], t4rows),
    ("table", ["Code", "Step", "Runs", "Mean rate"], t4dt),
    (
        "p",
        f"A change of 0.1% in η (−5 → −4.995) raises the mean rate by {T4['η'][4][2] - T4['η'][3][2]:+.3f} (16 runs each). The grid step of G below the working point (0.53 → 0.54) raises it by {m(0.54) - m(0.53):+.3f}. Halving the step raises the mean rate at every halving, in TVB and in the port.",
    ),
    ("h2", "5. Across the map, G, η, J and Δ act along one direction"),
    (
        "fig",
        "w5_maps.png",
        "Top: four statistics over (G, σ), mean of 4 port runs. Bottom: at each point, the cosine between the change in seven statistics caused by G and the change caused by another parameter (each statistic scaled by its run-to-run spread). +1: the two changes are proportional. Grey: no estimate (quiet corner, or a setting where a run went non-finite). Cross: working point.",
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
            "In the median, one grid step of G (0.01) does what η + 0.012 (0.23%), J + 0.025 (0.16%) or Δ + 0.010 (1%) does.",
            "The cosines fall short of 1 by as much as two halves of the runs disagree with each other about a single parameter, so the shortfall is noise. No separation of G from η, J or Δ is seen, and none is ruled out below the noise of four runs.",
            "σ is the one parameter the statistics see separately: more noise acts like more coupling below G ≈ 0.56 and like less above it.",
            "The step is the least resolved. For the level statistics, halving it acts like a small increase in G.",
        ],
    ),
    (
        "note",
        "This was expected from the equations: η, J·r and G × (input from other regions) enter the same equation as a sum. The prediction and its probability (0.8) were written down before the sweep. What the sweep adds is the sizes.",
    ),
    ("h2", "Not shown"),
    (
        "ul",
        [
            "**Per-region statistics and the paper's own fluidity statistic.** Not analysed. G weights input by the connectome and η does not, so per-region quantities are where a separation could appear.",
            "**Switching index.** Too noisy at four runs for anything beyond the shift of its peak with η.",
            "**σ ≤ 0.026 near the transition.** Those statistics still carry the decay from the starting state.",
            f"**High G with high σ.** {n_nonfinite} of {n_runs:,} runs went non-finite there (grey cells). Whether TVB does the same is untested.",
            "**TVB across the grid.** The maps are the port's. TVB was run at the working point, two neighbouring G values and four steps.",
            "**Three of the seven statistics** (mean rate, high-state occupancy, median high-state share) are nearly one statistic, which makes the cosines high cheaply.",
        ],
    ),
    (
        "meta",
        "Full record: `doc/LOG.md`, entry of 2026-11-24 17:35. Figures: `results/figures/`. Rebuild: `src/runall.sh`.",
    ),
]
SYM = {
    "±": r"$\pm$",
    "−": r"$-$",
    "×": r"$\times$",
    "≈": r"$\approx$",
    "≤": r"$\le$",
    "≥": r"$\ge$",
    "→": r"$\to$",
    "η": r"$\eta$",
    "σ": r"$\sigma$",
    "Δ": r"$\Delta$",
    "·": r"$\cdot$",
    "–": "--",
    "|": r"\textbar{}",
    ">": r"\textgreater{}",
    "<": r"\textless{}",
}


def tex(t):
    t = str(t)
    t = re.sub(r"10\^\{(-?\d+)\}", lambda k: "10⟪%s⟫" % k.group(1), t)
    t = "".join("\\" + c if c in "&%#_{}$" else c for c in t.replace("\\", r"\textbackslash{}"))
    for a, b in SYM.items():
        t = t.replace(a, b)
    t = re.sub(r"10⟪(-?\d+)⟫", r"$10^{\1}$", t)
    t = re.sub(r'"([^"]*)"', r"``\1''", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"`(.+?)`", r"\\texttt{\1}", t)
    t = re.sub(r"(?<![\w$.{-])-(?=\d)", "$-$", t)  # a leading hyphen on a number is a minus sign
    return re.sub(r"(?<=\d),(?=\d{3})", "{,}", t)


out = [
    r"\documentclass[10pt,letterpaper]{article}",
    r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage[scaled=0.95]{helvet}\renewcommand{\familydefault}{\sfdefault}",
    r"\usepackage[margin=0.85in]{geometry}\usepackage{graphicx,booktabs,array,microtype,xcolor,parskip,enumitem}\usepackage[font={small,color=ink2},labelfont=bf,skip=4pt,justification=raggedright,singlelinecheck=false]{caption}",
    r"\definecolor{ink2}{HTML}{52514E}\setlist[itemize]{leftmargin=1.3em,itemsep=2pt,topsep=3pt}\pagestyle{plain}\raggedbottom\setlength{\emergencystretch}{2em}",
    r"\begin{document}",
]
for b in B:
    k = b[0]
    if k == "h1":
        out.append(r"{\LARGE\bfseries %s\par}\vspace{4pt}" % tex(b[1]))
    elif k == "meta":
        out.append(r"{\small\color{ink2} %s\par}" % tex(b[1]))
    elif k == "h2":
        out.append(r"\vspace{8pt}\pagebreak[3]{\large\bfseries %s\par}\nopagebreak" % tex(b[1]))
    elif k == "p":
        out.append(tex(b[1]) + r"\par")
    elif k == "note":
        out.append(r"\begin{quote}\small\color{ink2} %s\end{quote}" % tex(b[1]))
    elif k == "ul":
        out.append(
            r"\begin{itemize}" + "".join(r"\item " + tex(i) + "\n" for i in b[1]) + r"\end{itemize}"
        )
    elif k == "fig":
        out.append(
            r"\begin{minipage}{\linewidth}\centering\includegraphics[width=\linewidth]{%s}\captionof{figure}{%s}\end{minipage}\par\vspace{4pt}"
            % (os.path.abspath(figs[b[1]]), tex(b[2]))
        )
    elif k == "table":
        hd, rows = b[1], b[2]
        last_text = any(re.search(r"[a-z]{4}", str(r[-1])) for r in rows)
        spec = "l" + "r" * (len(hd) - 2) + ("l" if last_text else "r")
        if len(b) > 3:  # an explicit column layout, for tables of text
            spec = b[3]
        out.append(
            r"\begin{minipage}{\linewidth}\centering\small\begin{tabular}{@{}%s@{}}\toprule " % spec
            + " & ".join(r"\textbf{%s}" % tex(h) for h in hd)
            + r" \\ \midrule "
            + " ".join(" & ".join(tex(c) for c in r) + r" \\" for r in rows)
            + r" \bottomrule\end{tabular}\end{minipage}\par\vspace{6pt}"
        )
out.append(r"\end{document}")
with tempfile.TemporaryDirectory() as td:
    open(os.path.join(td, "writeup.tex"), "w").write("\n".join(out))
    r = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "writeup.tex"],
        cwd=td,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if r.returncode != 0 or not os.path.exists(os.path.join(td, "writeup.pdf")):
        shutil.copy(os.path.join(td, "writeup.tex"), OUT + ".failed.tex")
        sys.exit("pdflatex failed; source kept at " + OUT + ".failed.tex\n" + r.stdout[-1500:])
    warn = [l for l in r.stdout.splitlines() if "Overfull" in l]
    shutil.copy(os.path.join(td, "writeup.pdf"), OUT)
print("wrote", OUT, f"{os.path.getsize(OUT) / 1e6:.2f} MB;", len(warn), "overfull boxes")
[print("  ", w) for w in warn[:8]]
