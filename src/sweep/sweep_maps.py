"""Derivative maps from the registered sweep (doc/LOG.md, 2026-11-24, 12:05). Analysis rules are those fixed there.
Usage (from the repository root): python3 -I src/sweep/sweep_maps.py results/runs/sweep/done results 0.005 0.02 0.005
       arguments: <folder with the batch *.json> <results_dir> <d_eta> <d_J> <d_Delta>
Writes <results_dir>/figures/maps/{base,confounding,scaling}.png, figures/derivatives/d_<param>.png (one per parameter), tables/{peaks.txt,summary.json}.
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

import sys, os, json, glob, site

sys.path.append(site.getusersitepackages())  # plotting library lives in the user site; -I hides it
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATS = [
    "m1_mean_r",
    "m2_occ_up07",
    "m3_u_median",
    "m5_vaf2_r",
    "switching_index",
    "mean_fc",
    "corr_fc_sc",
]
NAMES = [
    "mean rate",
    "up-occupancy",
    "median up-fraction",
    "VAF (2 comp.)",
    "switching index",
    "mean FC",
    "corr(FC, SC)",
]
ETA0, J0, D0 = -5.0, 15.0, 1.0
MIN_SEEDS = (
    4  # registered: non-finite cells are holes; a cell with fewer finite seeds than this is a hole
)


def load(folder):
    cells, bad = {}, 0
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        d = json.load(open(p))
        for r in d.get("runs", []):
            if not r["finite"] or any(r.get(m) is None for m in STATS):
                bad += 1
                continue
            key = (
                round(r["G"], 4),
                round(r["nsig"], 4),
                round(r["eta"], 4),
                round(r["J"], 4),
                round(r["Delta"], 4),
                r["dt"],
                bool(r.get("x64", False)),
            )
            cells.setdefault(key, []).append(r)
    return cells, bad


def grid(cells, Gs, Ss, eta=ETA0, J=J0, Delta=D0, dt=0.01, x64=False):
    """mean, se, sd, n arrays of shape (n_stats, n_sigma, n_G); NaN where a cell is missing."""
    out = np.full((4, len(STATS), len(Ss), len(Gs)), np.nan)
    for i, s in enumerate(Ss):
        for j, G in enumerate(Gs):
            rs = cells.get((G, s, round(eta, 4), round(J, 4), round(Delta, 4), dt, x64), [])
            if len(rs) >= MIN_SEEDS:
                v = np.array([[r[m] for m in STATS] for r in rs], float)
                out[0, :, i, j] = v.mean(0)
                out[2, :, i, j] = v.std(0, ddof=1)
                out[1, :, i, j] = out[2, :, i, j] / np.sqrt(len(rs))
                out[3, :, i, j] = len(rs)
    return out


def central(mean, se, x, axis):
    """Central differences along a grid axis (one-sided at the edges); returns derivative and its se."""
    m = np.moveaxis(mean, axis, -1)
    s = np.moveaxis(se, axis, -1)
    d = np.full_like(m, np.nan)
    e = np.full_like(m, np.nan)
    n = len(x)
    for k in range(n):
        a, b = max(k - 1, 0), min(k + 1, n - 1)
        d[..., k] = (m[..., b] - m[..., a]) / (x[b] - x[a])
        e[..., k] = np.hypot(s[..., b], s[..., a]) / (x[b] - x[a])
    return np.moveaxis(d, -1, axis), np.moveaxis(e, -1, axis)


def panel_maps(vals, mask, Gs, Ss, title, path, cmap="RdBu_r", sym=True, fmt=None):
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.6))
    axes = axes.ravel()
    ext = [Gs[0] - 0.005, Gs[-1] + 0.005, Ss[0] - 0.002, Ss[-1] + 0.002]
    for k, ax in enumerate(axes):
        if k >= len(STATS):
            ax.axis("off")
            continue
        v = np.where(mask[k], vals[k], np.nan) if mask is not None else vals[k]
        fin = v[np.isfinite(v)]
        if sym:
            lim = np.percentile(np.abs(fin), 98) if fin.size else 1.0
            kw = dict(vmin=-lim, vmax=lim)
        else:
            kw = {}
        im = ax.imshow(v, origin="lower", extent=ext, aspect="auto", cmap=cmap, **kw)
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(NAMES[k], fontsize=10)
        ax.set_xlabel("G")
        ax.set_ylabel("sigma")
        ax.plot([0.54], [0.030], marker="+", color="k", ms=9)
        ax.set_facecolor("0.85")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    folder, out = sys.argv[1], sys.argv[2]
    d_eta, d_J, d_D = (float(x) for x in sys.argv[3:6])
    [
        os.makedirs(os.path.join(out, d), exist_ok=True)
        for d in ("figures/maps", "figures/derivatives", "tables")
    ]
    cells, bad = load(folder)
    Gs = sorted({k[0] for k in cells if k[5] == 0.01 and not k[6] and k[2:5] == (ETA0, J0, D0)})
    Ss = sorted({k[1] for k in cells if k[5] == 0.01 and not k[6] and k[2:5] == (ETA0, J0, D0)})
    Ga, Sa = np.array(Gs), np.array(Ss)
    summ = dict(
        n_cells=len(cells),
        n_runs=sum(len(v) for v in cells.values()),
        non_finite_or_incomplete_runs=bad,
        G=Gs,
        sigma=Ss,
    )
    base = grid(cells, Gs, Ss)
    mean, se, sd, n = base
    print(
        f"{summ['n_runs']} runs in {len(cells)} cells; {bad} runs non-finite or incomplete; grid {len(Gs)} x {len(Ss)}; seeds per base cell: {np.nanmin(n):.0f}-{np.nanmax(n):.0f}"
    )
    panel_maps(
        mean,
        None,
        Gs,
        Ss,
        "Base statistics over (G, sigma); + marks the paper's working point (grey: no data)",
        os.path.join(out, "figures", "maps", "base.png"),
        cmap="viridis",
        sym=False,
    )

    D, E = {}, {}
    D["G"], E["G"] = central(mean, se, Ga, 2)
    D["sigma"], E["sigma"] = central(mean, se, Sa, 1)
    for name, kw, dlt in (("eta", "eta", d_eta), ("J", "J", d_J), ("Delta", "Delta", d_D)):
        c0 = dict(eta=ETA0, J=J0, Delta=D0)
        up = grid(cells, Gs, Ss, **{**c0, kw: c0[kw] + dlt})
        dn = grid(cells, Gs, Ss, **{**c0, kw: c0[kw] - dlt})
        D[name] = (up[0] - dn[0]) / (2 * dlt)
        E[name] = np.hypot(up[1], dn[1]) / (2 * dlt)
    fine = grid(cells, Gs, Ss, dt=0.005)
    D["dt_halving"] = fine[0] - mean
    E["dt_halving"] = np.hypot(fine[1], se)
    units = dict(
        G="per unit G",
        sigma="per unit sigma",
        eta="per unit eta",
        J="per unit J",
        Delta="per unit Delta",
        dt_halving="change when dt goes 0.01 -> 0.005",
    )
    for p in D:
        mask = np.abs(D[p]) > 2 * E[p]
        panel_maps(
            D[p],
            mask,
            Gs,
            Ss,
            f"d(statistic)/d({p}), {units[p]}; shown only where |estimate| > 2 se (grey: not resolved or no data)",
            os.path.join(out, "figures", "derivatives", f"d_{p}.png"),
        )
        summ[f"resolved_fraction_{p}"] = {
            NAMES[k]: float(np.nanmean(mask[k][np.isfinite(D[p][k])]))
            if np.isfinite(D[p][k]).any()
            else None
            for k in range(len(STATS))
        }

    # mixed region (registered): base up-occupancy between 0.02 and 0.90
    occ = mean[1]
    mixed = (occ > 0.02) & (occ < 0.90)
    summ["mixed_region_points"] = int(mixed.sum())
    # local confounding with G: statistics scaled by their seed sd at the point
    scale = np.where(sd > 0, sd, np.nan)
    cG = D["G"] / scale
    eG = E["G"] / scale
    strong = np.sqrt(np.nansum(cG**2, 0)) > 3 * np.sqrt(np.nansum(eG**2, 0))
    summ["points_with_G_column_above_3se"] = int(strong.sum())
    summ["of_which_mixed"] = int((strong & mixed).sum())
    others = ["sigma", "eta", "J", "Delta", "dt_halving"]
    cos, eq, resid = {}, {}, {}
    for p in others:
        cP = D[p] / scale
        ok = np.isfinite(cP).all(0) & np.isfinite(cG).all(0)
        num = np.nansum(cG * cP, 0)
        nG = np.sqrt(np.nansum(cG**2, 0))
        nP = np.sqrt(np.nansum(cP**2, 0))
        cos[p] = np.where(ok & strong, num / (nG * nP), np.nan)
        eq[p] = np.where(ok & strong, num / nG**2, np.nan)  # delta-G equivalent per unit of p
        resid[p] = 1 - cos[p] ** 2
        sel = strong & mixed & np.isfinite(cos[p])
        summ[f"cosine_G_{p}"] = dict(
            n=int(sel.sum()),
            median=float(np.nanmedian(cos[p][sel])) if sel.any() else None,
            share_above_095=float(np.mean(np.abs(cos[p][sel]) > 0.95)) if sel.any() else None,
            share_above_08=float(np.mean(np.abs(cos[p][sel]) > 0.8)) if sel.any() else None,
            median_equivalent_G_per_unit=float(np.nanmedian(eq[p][sel])) if sel.any() else None,
        )
    fig, axes = plt.subplots(2, 5, figsize=(21, 6.6))
    ext = [Gs[0] - 0.005, Gs[-1] + 0.005, Ss[0] - 0.002, Ss[-1] + 0.002]
    for k, p in enumerate(others):
        im = axes[0, k].imshow(
            cos[p], origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1
        )
        fig.colorbar(im, ax=axes[0, k], shrink=0.85)
        axes[0, k].set_title(f"cosine between the G column and the {p} column", fontsize=9)
        fin = eq[p][np.isfinite(eq[p])]
        lim = np.percentile(np.abs(fin), 95) if fin.size else 1
        im = axes[1, k].imshow(
            eq[p], origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim
        )
        fig.colorbar(im, ax=axes[1, k], shrink=0.85)
        axes[1, k].set_title(f"change in G equivalent to one unit of {p}", fontsize=9)
        for ax in axes[:, k]:
            ax.set_xlabel("G")
            ax.set_ylabel("sigma")
            ax.set_facecolor("0.85")
            ax.plot([0.54], [0.030], marker="+", color="k", ms=9)
    fig.suptitle(
        "Local confounding with G over seven statistics (each scaled by its seed sd); grey: G column below 3 se, or no data",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "maps", "confounding.png"), dpi=110)
    plt.close(fig)

    # scaling identity (hand-derived, doc/LOG.md): Delta m_D + eta m_eta + J m_J / 2 + G m_G / 2 + 1.5 sigma m_sigma - k m = 0
    Gg = Ga[None, None, :]
    Sg = Sa[None, :, None]
    kk = np.array([0.5, 0, 0, 0, 0, 0, 0])[:, None, None]
    terms = [
        D0 * D["Delta"],
        ETA0 * D["eta"],
        0.5 * J0 * D["J"],
        0.5 * Gg * D["G"],
        1.5 * Sg * D["sigma"],
        -kk * mean,
    ]
    R = sum(terms)
    Rse = np.sqrt(
        (D0 * E["Delta"]) ** 2
        + (ETA0 * E["eta"]) ** 2
        + (0.5 * J0 * E["J"]) ** 2
        + (0.5 * Gg * E["G"]) ** 2
        + (1.5 * Sg * E["sigma"]) ** 2
        + (kk * se) ** 2
    )
    size = sum(np.abs(t) for t in terms)
    rel = np.abs(R) / size
    panel_maps(
        R / Rse,
        None,
        Gs,
        Ss,
        "Scaling identity residual in units of its se (0 = the Delta column is fully forced by the others)",
        os.path.join(out, "figures", "maps", "scaling.png"),
    )
    summ["scaling_identity"] = {
        NAMES[k]: dict(
            median_abs_residual_in_se=float(np.nanmedian(np.abs(R[k] / Rse[k])[mixed])),
            share_within_2se=float(
                np.nanmean((np.abs(R[k] / Rse[k]) < 2)[mixed & np.isfinite(R[k])])
            ),
            median_residual_over_sum_of_terms=float(np.nanmedian(rel[k][mixed])),
        )
        for k in range(len(STATS))
    }

    # peak location in G of the peaked statistics, per sigma row, base and variants (three-point smooth); bootstrap over seeds
    rng = np.random.default_rng(0)
    lines = []

    def row_values(i, m, **kw):
        base_kw = dict(eta=ETA0, J=J0, Delta=D0, dt=0.01)
        base_kw.update(kw)
        return [
            [
                r[m]
                for r in cells.get(
                    (
                        G,
                        Ss[i],
                        round(base_kw["eta"], 4),
                        round(base_kw["J"], 4),
                        round(base_kw["Delta"], 4),
                        base_kw["dt"],
                        False,
                    ),
                    [],
                )
            ]
            for G in Gs
        ]

    def peak(vals):
        mu = np.array([np.mean(v) if len(v) else np.nan for v in vals])
        sm = (
            np.convolve(np.nan_to_num(mu, nan=-np.inf), np.ones(3) / 3, mode="same")
            if np.isfinite(mu).all()
            else mu
        )
        return Gs[int(np.nanargmax(sm[1:-1])) + 1] if np.isfinite(sm[1:-1]).any() else np.nan

    def peak_boot(vals, nb=200):
        if any(len(v) < 2 for v in vals):
            return np.nan, np.nan
        ps = [peak([list(rng.choice(v, len(v))) for v in vals]) for _ in range(nb)]
        return peak(vals), float(np.std(ps))

    variants = [
        ("base", {}),
        (f"eta+{d_eta}", dict(eta=ETA0 + d_eta)),
        (f"eta-{d_eta}", dict(eta=ETA0 - d_eta)),
        (f"J+{d_J}", dict(J=J0 + d_J)),
        (f"J-{d_J}", dict(J=J0 - d_J)),
        (f"Delta+{d_D}", dict(Delta=D0 + d_D)),
        (f"Delta-{d_D}", dict(Delta=D0 - d_D)),
        ("dt 0.005", dict(dt=0.005)),
    ]
    peaks = {}
    for m, nm in (("switching_index", "switching index"), ("m5_vaf2_r", "VAF")):
        lines.append(
            f"\nLocation in G of the maximum of {nm} along each sigma row (three-point smooth; ± bootstrap sd over seeds)"
        )
        lines.append("sigma   " + "  ".join(f"{v[0]:>14s}" for v in variants))
        for i, s in enumerate(Ss):
            row = [peak_boot(row_values(i, m, **kw)) for _, kw in variants]
            peaks[(m, s)] = row
            lines.append(
                f"{s:.3f}   "
                + "  ".join(
                    f"{p:6.3f} ± {e:5.3f}" if np.isfinite(p) else f"{'n/a':>14s}" for p, e in row
                )
            )
    open(os.path.join(out, "tables", "peaks.txt"), "w").write("\n".join(lines))
    print("\n".join(lines))
    summ["peaks"] = {
        f"{m}@{s}": [
            [None if not np.isfinite(a) else a, None if not np.isfinite(b) else b] for a, b in row
        ]
        for (m, s), row in peaks.items()
    }
    json.dump(summ, open(os.path.join(out, "tables", "summary.json"), "w"), indent=1)
    for k in ("mixed_region_points", "points_with_G_column_above_3se", "of_which_mixed"):
        print(k, summ[k])
    for p in others:
        print("cosine G vs", p, summ[f"cosine_G_{p}"])
    print("scaling identity:", json.dumps(summ["scaling_identity"], indent=0)[:900])
    for p in D:
        print(
            "resolved fraction",
            p,
            {
                k: (round(v, 2) if v is not None else None)
                for k, v in summ[f"resolved_fraction_{p}"].items()
            },
        )


if __name__ == "__main__":
    main()
