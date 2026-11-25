"""Figures and one self-contained page for the sweep of 2026-11-24 (README, 17:35 entry). Every number in the page is
computed here from the run files; nothing is typed in by hand except the text.
Usage (from sim/): python3 -I writeup_sweep.py runs/sweep/done runs/Aprime runs/port ../figures/sweep_2026-11-24 ../reports/SWEEP_WRITEUP_2026-11-24.html
"""
import sys, os, json, glob, site, base64, re
sys.path.append(site.getusersitepackages()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sweep_maps as sm
from sweep_maps import STATS, NAMES, ETA0, J0, D0

SWEEP, APRIME, PORT, FIG, OUT = sys.argv[1:6]
os.makedirs(FIG, exist_ok=True)
# palette: first three categorical slots (validated all-pairs on the light surface), blue sequential ramp, blue-grey-red diverging
C_TVB, C_F32, C_F64 = "#2a78d6", "#eb6834", "#1baf7a"; SURF, INK, INK2, GRID, HOLE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#8f1f1f"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "font.family": "DejaVu Sans"})
FOUR = [("m1_mean_r", "Mean rate"), ("m5_vaf2_r", "VAF (two components)"), ("switching_index", "Switching index"), ("mean_fc", "Mean FC")]
ms = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))
isbase = lambda r: abs(r.get("eta", ETA0) - ETA0) < 1e-9 and abs(r.get("J", J0) - J0) < 1e-9 and abs(r.get("Delta", D0) - D0) < 1e-9
at = lambda r, G, s: abs(r["G"] - G) < 1e-9 and abs(r["nsig"] - s) < 1e-9

# ---------------------------------------------------------------- data
tvb = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(APRIME, "A_G*.json")))]; tvb = [r for r in tvb if r.get("finite", True)]
port = []
for p in sorted(glob.glob(os.path.join(PORT, "*.json"))) + sorted(glob.glob(os.path.join(SWEEP, "*.json"))) + [os.path.join(os.path.dirname(SWEEP.rstrip("/")), "pilot_done.json")]:
    try: d = json.load(open(p))
    except Exception: continue
    if isinstance(d, dict):
        port += [dict(r, _file=os.path.basename(p)) for r in d.get("runs", []) if r.get("finite") and r.get("T", 0) > 50000 and "switching_index" in r]
pilot = [r for r in port if r["_file"] == "pilot_done.json"]
wp_t = [r for r in tvb if at(r, 0.54, 0.03) and r["dt"] == 0.01]
wp_32 = [r for r in port if at(r, 0.54, 0.03) and r["dt"] == 0.01 and isbase(r) and not r.get("x64")]
wp_64 = [r for r in port if at(r, 0.54, 0.03) and r["dt"] == 0.01 and isbase(r) and r.get("x64")]
cells, _ = sm.load(SWEEP)
Gs = sorted({k[0] for k in cells if k[5] == 0.01 and not k[6] and k[2:5] == (ETA0, J0, D0)}); Ss = sorted({k[1] for k in cells if k[5] == 0.01 and not k[6] and k[2:5] == (ETA0, J0, D0)})
Ga, Sa = np.array(Gs), np.array(Ss); i0, j0 = Ss.index(0.03), Gs.index(0.54)
figs = {}
def save(fig, name):
    p = os.path.join(FIG, name); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); figs[name] = p
def dots(ax, x, v, color, rng, width=0.12, n_label=None):
    v = np.asarray(v, float); ax.scatter(x + rng.uniform(-width, width, len(v)), v, s=9, color=color, alpha=0.38, linewidths=0, zorder=2)
    m, e = ms(v); ax.errorbar([x], [m], yerr=[e], fmt="o", ms=5, color=INK, mfc=INK, mec=SURF, mew=1.2, elinewidth=1.4, capsize=3.5, zorder=4)
    return m, e

# ---------------------------------------------------------------- figure 1: distributions at the working point
rng = np.random.default_rng(0); fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3)); T1 = []
groups = [("TVB", wp_t, C_TVB), ("port,\nfloat32", wp_32, C_F32), ("port,\nfloat64", wp_64, C_F64)]
for ax, (key, title) in zip(axes, FOUR):
    row = [title]
    for x, (lab, rs, col) in enumerate(groups):
        v = [r[key] for r in rs]; m, e = dots(ax, x, v, col, rng, 0.22); row.append(f"{m:.4f} ± {e:.4f}")
        ax.annotate(f"{m:.4f}\n± {e:.4f}", (x, max(v)), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{g[0]}\nn = {len(g[1])}" for g in groups]); ax.set_title(title); ax.margins(y=0.22); ax.grid(axis="x", visible=False); T1.append(row)
save(fig, "w1_working_point.png")

# ---------------------------------------------------------------- figure 2: chains over time
tr = np.load(os.path.join(PORT, "traces_wp.npz")); A, B = tr["tvb"], tr["prec2_f32"]; fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axes[0]; x = np.arange(1, 151)
for k in (0, 1, 2): ax.plot(x, A[k, :150], color=HOLE, lw=0.6, zorder=1)
for arr, col, lab in ((A, C_TVB, f"TVB, mean of {len(A)} runs"), (B, C_F32, f"port float32, mean of {len(B)} runs")):
    m = arr[:, :150].mean(0); e = arr[:, :150].std(0, ddof=1) / np.sqrt(len(arr)); ax.fill_between(x, m - e, m + e, color=col, alpha=0.22, lw=0); ax.plot(x, m, color=col, lw=1.8, label=lab)
late = float(np.concatenate([A[:, 100:], B[:, 100:]]).mean()); ax.axhline(late, color=INK2, lw=0.9, ls=(0, (4, 3)), label=f"mean after volume 100, both codes: {late:.3f}")
ax.plot([], [], color=HOLE, lw=0.8, label="three single TVB runs"); ax.set_ylim(0, 0.36); ax.set_xlim(1, 150); ax.set_xlabel("scan volume after the 1,000-unit discard (one volume = 72 time units)"); ax.set_ylabel("share of regions in the high state")
ax.set_title("First 150 volumes: mean over runs, band = ± 1 SE"); ax.legend(loc="upper right", fontsize=8)
ax = axes[1]; edges = np.arange(0, 833, 49); xc = 0.5 * (edges[:-1] + edges[1:]) + 1
for arr, col, lab, off in ((A, C_TVB, "TVB", -5), (B, C_F32, "port float32", 5)):
    bm = np.stack([arr[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])], 1); m = bm.mean(0); e = bm.std(0, ddof=1) / np.sqrt(len(arr))
    ax.errorbar(xc + off, m, yerr=e, fmt="o-", ms=4, lw=1.2, color=col, mec=SURF, mew=0.8, capsize=2.5, elinewidth=1.1, label=lab)
ax.axhline(late, color=INK2, lw=0.9, ls=(0, (4, 3))); ax.set_ylim(0, 0.12); ax.set_xlabel("scan volume (blocks of 49)"); ax.set_title("Whole record: block means ± 1 SE"); ax.legend(loc="upper right", fontsize=8)
save(fig, "w2_chains.png")
burn = dict(tvb_first=float(A[:, 0].mean()), port_first=float(B[:, 0].mean()), tvb20=ms(A[:, :20].mean(1)), port20=ms(B[:, :20].mean(1)), late=late)

# ---------------------------------------------------------------- figure 3: along G at sigma 0.030
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3)); rng = np.random.default_rng(1); T3 = []
for ax, (key, title) in zip(axes, FOUR):
    mm, ee = [], []
    for G in Gs:
        v = [r[key] for r in cells.get((G, 0.03, ETA0, J0, D0, 0.01, False), [])]; ax.scatter(np.full(len(v), G), v, s=7, color=C_F32, alpha=0.4, linewidths=0, zorder=2); m, e = ms(v); mm.append(m); ee.append(e)
    ax.errorbar(Gs, mm, yerr=ee, fmt="-", lw=1.4, color=C_F32, elinewidth=1.0, capsize=1.8, zorder=3, label="port float32, 4 runs per G")
    for G in (0.53, 0.54, 0.55):
        v = [r[key] for r in tvb if at(r, G, 0.03) and r["dt"] == 0.01]; m, e = ms(v)
        ax.errorbar([G + 0.0022], [m], yerr=[e], fmt="o", ms=5, color=C_TVB, mec=SURF, mew=1.0, elinewidth=1.3, capsize=3, zorder=5, label=("TVB (8, 96, 8 runs)" if G == 0.53 else None))
    ax.axvline(0.54, color=INK2, lw=0.7, ls=(0, (2, 3)), zorder=1); ax.set_title(title); ax.set_xlabel("G"); ax.set_xlim(0.44, 0.66)
axes[0].legend(loc="lower right", fontsize=7.5); axes[0].annotate("working point", (0.54, axes[0].get_ylim()[1]), xytext=(-3, -9), textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
save(fig, "w3_along_G.png")
for G in (0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60):
    row = [f"{G:.2f}"]
    for key, _ in FOUR:
        m, e = ms([r[key] for r in cells[(G, 0.03, ETA0, J0, D0, 0.01, False)]]); row.append(f"{m:.4f} ± {e:.4f}")
    T3.append(row)
T3t = []
for G in (0.53, 0.54, 0.55):
    rs = [r for r in tvb if at(r, G, 0.03) and r["dt"] == 0.01]; T3t.append([f"{G:.2f}", str(len(rs))] + ["%.4f ± %.4f" % ms([r[key] for r in rs]) for key, _ in FOUR])

# ---------------------------------------------------------------- figure 4: offsets and step at the working point (mean rate)
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4), sharey=True); rng = np.random.default_rng(2); T4 = {}
for ax, (p, base, unit) in zip(axes[:3], (("eta", ETA0, "η"), ("J", J0, "J"), ("Delta", D0, "Δ"))):
    others = {"eta": ETA0, "J": J0, "Delta": D0}; del others[p]; rows = []
    vals = sorted({round(r[p], 4) for r in pilot if at(r, 0.54, 0.03) and all(abs(r[q] - others[q]) < 1e-9 for q in others)}); w = (vals[-1] - vals[0]) / 60; mm = []
    for v in vals:
        y = [r["m1_mean_r"] for r in pilot if at(r, 0.54, 0.03) and abs(r[p] - v) < 1e-9 and all(abs(r[q] - others[q]) < 1e-9 for q in others)]
        m, e = dots(ax, v, y, C_F32, rng, w); mm.append(m); rows.append((v, len(y), m, e, ms([r["m5_vaf2_r"] for r in pilot if at(r, 0.54, 0.03) and abs(r[p] - v) < 1e-9 and all(abs(r[q] - others[q]) < 1e-9 for q in others)]),
                                                                       ms([r["switching_index"] for r in pilot if at(r, 0.54, 0.03) and abs(r[p] - v) < 1e-9 and all(abs(r[q] - others[q]) < 1e-9 for q in others)])))
    ax.plot(vals, mm, color=INK, lw=0.9, zorder=3); ax.axvline(base, color=INK2, lw=0.7, ls=(0, (2, 3)), zorder=1); ax.set_title(f"{unit} varied"); ax.set_xlabel(unit); ax.set_xticks([vals[0], vals[1], base, vals[-2], vals[-1]])
    ax.ticklabel_format(useOffset=False, axis="x"); T4[unit] = rows
axes[0].set_ylabel("mean rate")
ax = axes[3]; dts = [0.02, 0.01, 0.005, 0.0025]; rows = []
for k, dt in enumerate(dts):
    y = [r["m1_mean_r"] for r in tvb if at(r, 0.54, 0.03) and r["dt"] == dt]; m, e = dots(ax, k - 0.17, y, C_TVB, rng, 0.07); rows.append(("TVB", dt, len(y), m, e))
    y = [r["m1_mean_r"] for r in port if at(r, 0.54, 0.03) and r["dt"] == dt and isbase(r) and not r.get("x64")]
    if y: m, e = dots(ax, k + 0.17, y, C_F32, rng, 0.07); rows.append(("port float32", dt, len(y), m, e))
ax.scatter([], [], s=14, color=C_TVB, label="TVB"); ax.scatter([], [], s=14, color=C_F32, label="port float32"); ax.legend(loc="upper left", fontsize=7.5)
ax.set_xticks(range(4)); ax.set_xticklabels([str(d) for d in dts]); ax.set_xlabel("integration step"); ax.set_title("Step varied"); ax.grid(axis="x", visible=False); T4["dt"] = rows
save(fig, "w4_offsets.png")

# ---------------------------------------------------------------- figure 5: maps
mean, se, sd, n = sm.grid(cells, Gs, Ss); D, E = {}, {}
D["G"], E["G"] = sm.central(mean, se, Ga, 2); D["sigma"], E["sigma"] = sm.central(mean, se, Sa, 1); dl = dict(eta=0.005, J=0.02, Delta=0.005); c0 = dict(eta=ETA0, J=J0, Delta=D0)
for p in dl:
    up = sm.grid(cells, Gs, Ss, **{**c0, p: c0[p] + dl[p]}); dn = sm.grid(cells, Gs, Ss, **{**c0, p: c0[p] - dl[p]}); D[p] = (up[0] - dn[0]) / (2 * dl[p]); E[p] = np.hypot(up[1], dn[1]) / (2 * dl[p])
fine = sm.grid(cells, Gs, Ss, dt=0.005); D["dt"] = fine[0] - mean; E["dt"] = np.hypot(fine[1], se)
occ = mean[1]; mixed = (occ > 0.02) & (occ < 0.90); scale = np.where(sd > 0, sd, np.nan); cG = D["G"] / scale; eG = E["G"] / scale
strong = np.sqrt(np.nansum(cG ** 2, 0)) > 3 * np.sqrt(np.nansum(eG ** 2, 0)); cos, eq, T5 = {}, {}, []
for p, lab in (("eta", "η"), ("J", "J"), ("Delta", "Δ"), ("dt", "one halving of the step"), ("sigma", "σ")):
    cP = D[p] / scale; ok = np.isfinite(cP).all(0) & np.isfinite(cG).all(0) & strong & mixed; num = np.nansum(cG * cP, 0); nG = np.sqrt(np.nansum(cG ** 2, 0)); nP = np.sqrt(np.nansum(cP ** 2, 0))
    cos[p] = np.where(ok, num / (nG * nP), np.nan); eq[p] = np.where(ok, num / nG ** 2, np.nan); c = cos[p][ok]; e = eq[p][ok]
    T5.append([lab, str(int(ok.sum())), f"{np.median(c):+.3f}", f"{100 * np.mean(np.abs(c) > 0.95):.0f}%", f"{100 * np.mean(np.abs(c) > 0.8):.0f}%",
               "sign changes across the transition" if p == "sigma" else f"{np.percentile(e, 25):.3g} / {np.median(e):.3g} / {np.percentile(e, 75):.3g}"])
ext = [Gs[0] - 0.005, Gs[-1] + 0.005, Ss[0] - 0.002, Ss[-1] + 0.002]; fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.6))
for ax, k, title in zip(axes[0], (0, 3, 4, 5), ("Mean rate", "VAF (two components)", "Switching index", "Mean FC")):
    im = ax.imshow(mean[k], origin="lower", extent=ext, aspect="auto", cmap=SEQ, vmin=0); fig.colorbar(im, ax=ax, shrink=0.85); ax.set_title(title)
for ax, p, title in zip(axes[1], ("eta", "J", "Delta", "sigma"), ("Cosine: G against η", "Cosine: G against J", "Cosine: G against Δ", "Cosine: G against σ")):
    im = ax.imshow(cos[p], origin="lower", extent=ext, aspect="auto", cmap=DIV, vmin=-1, vmax=1); fig.colorbar(im, ax=ax, shrink=0.85); ax.set_title(title)
for ax in axes.ravel():
    ax.set_facecolor(HOLE); ax.grid(False); ax.set_xlabel("G"); ax.set_ylabel("σ"); ax.plot([0.54], [0.030], marker="+", color=INK, ms=10, mew=1.6)
    for s in ("top", "right"): ax.spines[s].set_visible(True)
fig.tight_layout(); save(fig, "w5_maps.png")
n_nonfinite = sum(1 for p in glob.glob(os.path.join(SWEEP, "*.json")) for r in json.load(open(p))["runs"] if not r["finite"]); n_runs = sum(len(json.load(open(p))["runs"]) for p in glob.glob(os.path.join(SWEEP, "*.json")))

# ---------------------------------------------------------------- page
def table(head, rows, cls=""):
    return f'<table class="{cls}"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>"
def img(name, cap):
    b = base64.b64encode(open(figs[name], "rb").read()).decode(); return f'<figure><img src="data:image/png;base64,{b}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'
pm = lambda t: f"{t[0]:.4f} ± {t[1]:.4f}"
t4rows = []
for unit, rows in T4.items():
    if unit == "dt": continue
    for v, nn, m, e, vaf, si in rows: t4rows.append([unit, f"{v:g}", str(nn), f"{m:.4f} ± {e:.4f}", pm(vaf), pm(si)])
t4dt = [[c, f"{dt:g}", str(nn), f"{m:.4f} ± {e:.4f}"] for c, dt, nn, m, e in T4["dt"]]
m = lambda G: ms([r["m1_mean_r"] for r in cells[(G, 0.03, ETA0, J0, D0, 0.01, False)]])[0]
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sweep write-up, 2026-11-24</title><style>
body{{background:{SURF};color:{INK};font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;max-width:1000px;margin:0 auto;padding:24px 16px 60px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:34px 0 8px}} p,li{{max-width:780px}} .meta{{color:{INK2};font-size:13px}}
figure{{margin:14px 0 6px}} img{{width:100%;height:auto;border:1px solid {GRID};border-radius:4px}} figcaption{{color:{INK2};font-size:13px;margin-top:4px;max-width:780px}}
table{{border-collapse:collapse;font-size:13px;margin:10px 0;font-variant-numeric:tabular-nums}} th,td{{padding:3px 10px;text-align:right;border-bottom:1px solid {GRID}}} th{{color:{INK2};font-weight:600}}
td:first-child,th:first-child{{text-align:left}} .wrap{{overflow-x:auto}} .note{{border-left:3px solid {GRID};padding-left:12px;color:{INK2}}} code{{font-size:13px}}
</style></head><body>
<h1>Which parameters can the summary statistics tell apart?</h1>
<p class="meta">Draft, machine-written, 2026-11-24. Not yet rewritten in the author's words. Every number is computed from the run files by <code>sim/writeup_sweep.py</code>. All values are mean ± one standard error over independent runs ("chains": same parameters, different random seed and starting state).</p>

<h2>What was run</h2>
<ul>
<li><b>Model.</b> 84 regions (<code>hcp-001</code> connectome), Montbrió–Pazó–Roxin node, η = −5, J = 15, Δ = 1, delays at speed 20, stochastic Heun at step 0.01, noise [σ, 2σ]. These are the settings of the group's companion code; that the paper used them is an inference.</li>
<li><b>Reference.</b> The Virtual Brain 2.10.0 (numba backend) through the group's own code. {len(wp_t)} runs at the paper's working point (G 0.54, σ 0.030), fewer at neighbouring G and at other steps.</li>
<li><b>Port.</b> A JAX rewrite of the same kernel on two GPUs. It reproduces TVB to 2 × 10⁻¹² over 3,000 steps when given TVB's own noise.</li>
<li><b>Sweep.</b> {n_runs:,} port runs of 61,000 time units (first 1,000 discarded): G 0.40–0.70 × σ 0.018–0.054, each point with η, J and Δ nudged up and down, and again at half the step. Four runs per setting.</li>
</ul>

<h2>1. The two codes give the same distributions</h2>
{img("w1_working_point.png", "Each dot is one run at the working point (G 0.54, σ 0.030, step 0.01). Black marker: mean ± 1 SE, also printed above each group.")}
<div class="wrap">{table(["Statistic", f"TVB (n = {len(wp_t)})", f"Port, float32 (n = {len(wp_32)})", f"Port, float64 (n = {len(wp_64)})"], T1)}</div>
<p>The spread between runs is large compared with the standard errors: single runs give a switching index anywhere from {min(r["switching_index"] for r in wp_t + wp_32 + wp_64):.3f} to {max(r["switching_index"] for r in wp_t + wp_32 + wp_64):.3f}. A comparison of pattern statistics from a handful of runs is dominated by this spread. The first 32 TVB runs alone gave {ms([r["switching_index"] for r in wp_t if r["seed"] <= 32])[0]:.4f} ± {ms([r["switching_index"] for r in wp_t if r["seed"] <= 32])[1]:.4f}; the next 64 gave {ms([r["switching_index"] for r in wp_t if r["seed"] > 32])[0]:.4f} ± {ms([r["switching_index"] for r in wp_t if r["seed"] > 32])[1]:.4f}.</p>

<h2>2. Chains are still settling after the discard</h2>
{img("w2_chains.png", "Share of regions in the high state per scan volume at the working point. Left: mean over runs with a ± 1 SE band, and three single TVB runs in grey. Right: means over blocks of 49 volumes ± 1 SE. Dashed line: mean after volume 100, both codes pooled.")}
<p>Both codes start every region at a rate between 0 and 2. In the first volume after the discard the high-state share is {burn['tvb_first']:.3f} (TVB) and {burn['port_first']:.3f} (port), against {burn['late']:.3f} later. Over the first 20 volumes it averages {pm(burn['tvb20'])} and {pm(burn['port20'])}. At this working point the effect on whole-record statistics is under 3%. At lower noise (σ ≤ 0.026) near the transition it is not small: the starting state persists much longer there.</p>

<h2>3. The working point sits on a knee in G</h2>
{img("w3_along_G.png", "Statistics along G at σ 0.030. Orange dots: single port runs; orange line: their mean ± 1 SE (4 runs per G). Blue: TVB mean ± 1 SE at G 0.53, 0.54, 0.55, drawn slightly to the right. Dotted line: the working point.")}
<div class="wrap">{table(["G (port, 4 runs)"] + [t for _, t in FOUR], T3)}{table(["G (TVB)", "runs"] + [t for _, t in FOUR], T3t)}</div>
<p>One grid step of G is not one size here. In the port's four runs, G 0.53 → 0.54 raises the mean rate by {m(0.54) - m(0.53):+.3f} and G 0.54 → 0.55 by {m(0.55) - m(0.54):+.3f}. The pattern statistics peak within one or two grid steps of the working point.</p>

<h2>4. Small changes in η, J, Δ and the step move the same statistics</h2>
{img("w4_offsets.png", "Mean rate at the working point. First three panels: one node parameter changed at a time, 16 port (float32) runs per value; dots are single runs, black markers mean ± 1 SE, dotted line the nominal value. Fourth panel: integration step, TVB and port.")}
<div class="wrap">{table(["Parameter", "Value", "Runs", "Mean rate", "VAF", "Switching index"], t4rows)}{table(["Code", "Step", "Runs", "Mean rate"], t4dt)}</div>
<p>A change of 0.1% in η (−5 → −4.995) raises the mean rate by {T4["η"][4][2] - T4["η"][3][2]:+.3f} (16 runs each). The grid step of G below the working point (0.53 → 0.54) raises it by {m(0.54) - m(0.53):+.3f}. Halving the step raises the mean rate at every halving, in TVB and in the port.</p>

<h2>5. Across the map, G, η, J and Δ act along one direction</h2>
{img("w5_maps.png", "Top: four statistics over (G, σ), mean of 4 port runs. Bottom: at each point, the cosine between the change in seven statistics caused by G and the change caused by another parameter (each statistic scaled by its run-to-run spread). +1: the two changes are proportional. Grey: no estimate (quiet corner, or a setting where a run went non-finite). Cross: working point.")}
<div class="wrap">{table(["G against", "Points", "Median cosine", "|cosine| &gt; 0.95", "|cosine| &gt; 0.8", "Change in G equal to one unit (quartiles)"], T5)}</div>
<ul>
<li>In the median, one grid step of G (0.01) does what η + 0.012 (0.23%), J + 0.025 (0.16%) or Δ + 0.010 (1%) does.</li>
<li>The cosines fall short of 1 by as much as two halves of the runs disagree with each other about a single parameter, so the shortfall is noise. No separation of G from η, J or Δ is seen, and none is ruled out below the noise of four runs.</li>
<li>σ is the one parameter the statistics see separately: more noise acts like more coupling below G ≈ 0.56 and like less above it.</li>
<li>The step is the least resolved. For the level statistics, halving it acts like a small increase in G.</li>
</ul>
<p class="note">This was expected from the equations: η, J·r and G × (input from other regions) enter the same equation as a sum. The prediction and its probability (0.8) were written down before the sweep. What the sweep adds is the sizes.</p>

<h2>Not shown</h2>
<ul>
<li><b>Per-region statistics and the paper's own fluidity statistic.</b> Not analysed. G weights input by the connectome and η does not, so per-region quantities are where a separation could appear.</li>
<li><b>Switching index.</b> Too noisy at four runs for anything beyond the shift of its peak with η.</li>
<li><b>σ ≤ 0.026 near the transition.</b> Those statistics still carry the decay from the starting state.</li>
<li><b>High G with high σ.</b> {n_nonfinite} of {n_runs:,} runs went non-finite there (grey cells). Whether TVB does the same is untested.</li>
<li><b>TVB across the grid.</b> The maps are the port's. TVB was run at the working point, two neighbouring G values and four steps.</li>
<li><b>Three of the seven statistics</b> (mean rate, high-state occupancy, median high-state share) are nearly one statistic, which makes the cosines high cheaply.</li>
</ul>
<p class="meta">Full record: <code>README.md</code>, entry of 2026-11-24 17:35. Figures: <code>figures/sweep_2026-11-24/</code>.</p>
</body></html>"""
open(OUT, "w").write(html); print("wrote", OUT, f"{len(html) / 1e6:.2f} MB"); print(json.dumps(dict(T1=T1, T3t=T3t, T5=T5, burn=burn, n=[len(wp_t), len(wp_32), len(wp_64)], nonfinite=n_nonfinite), indent=0, ensure_ascii=False)[:3000])
