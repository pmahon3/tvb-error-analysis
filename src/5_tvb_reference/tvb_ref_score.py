"""Score stage A' against the rule registered in doc/LOG.md (2026-11-24). Reads the per-run JSON files.
Usage: python -I tvb_ref_score.py <folder with A_*.json>
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "common"),
]

import sys, os, json, glob
import numpy as np

PRIMARY = ["m1_mean_r", "m2_occ_up07", "m3_u_median", "m4_u_p05", "m5_vaf2_r"]
SECONDARY = [
    "switching_index",
    "mean_fc",
    "corr_fc_sc",
    "occ_up05",
    "u_p95",
    "u_sd",
    "vaf2_bold",
    "max_r",
]
WP_G, WP_SIG = 0.54, 0.030


def ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return (
        (
            float(v.mean()),
            float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"),
            len(v),
        )
        if len(v)
        else (float("nan"), float("nan"), 0)
    )


def main(folder):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(folder, "A_*.json")))]
    cells = {}
    for r in rows:
        cells.setdefault((r["G"], r["nsig"], r["dt"]), []).append(r)
    print(f"{len(rows)} runs in {len(cells)} cells")
    for key in sorted(cells):
        rs = cells[key]
        fin = [r for r in rs if r["finite"]]
        print(
            f"\nG={key[0]} sigma={key[1]} dt={key[2]}: {len(fin)}/{len(rs)} finite; wall {np.mean([r['wall_s'] for r in rs]) / 60:.1f} min"
        )
        for m in PRIMARY + SECONDARY:
            a, s, n = ms([r.get(m) for r in fin])
            print(f"   {m:16s} {a:10.5f} ± {s:.5f}  (n={n})")

    def cell(G, dt):
        return [r for r in cells.get((G, WP_SIG, dt), []) if r["finite"]]

    print("\n=== validity (registered) ===")
    wp = cells.get((WP_G, WP_SIG, 0.01), [])
    print(
        "non-finite at the working point, dt 0.01:", sum(not r["finite"] for r in wp), "of", len(wp)
    )
    occ = [r["m2_occ_up07"] for r in cell(WP_G, 0.01)]
    print(
        "up-occupancy per seed at the working point:",
        np.round(occ, 3).tolist(),
        "| inside 0.05-0.95 in",
        sum(0.05 < x < 0.95 for x in occ),
        "of",
        len(occ),
    )
    for G in (0.30, WP_G, 0.80):
        a, s, n = ms([r.get("switching_index") for r in cell(G, 0.01)])
        print(f"switching index at G={G}: {a:.5f} ± {s:.5f} (n={n})")

    print("\n=== rule: D_dt (dt 0.01 vs 0.005) against D_G (one grid step, dt 0.01) ===")
    verdicts = []
    for m in PRIMARY:
        a1, s1, _ = ms([r.get(m) for r in cell(WP_G, 0.01)])
        a2, s2, _ = ms([r.get(m) for r in cell(WP_G, 0.005)])
        g1, t1, _ = ms([r.get(m) for r in cell(0.53, 0.01)])
        g2, t2, _ = ms([r.get(m) for r in cell(0.55, 0.01)])
        Ddt, sdt = abs(a1 - a2), np.hypot(s1, s2)
        DG, sG = abs(g2 - g1) / 2, np.hypot(t1, t2) / 2
        beyond = Ddt > 2 * sdt
        v = (
            "smaller than a grid step or within 2 se"
            if (Ddt < DG or not beyond)
            else "at least a grid step and beyond 2 se"
        )
        verdicts.append(Ddt >= DG and beyond)
        print(
            f"   {m:14s} D_dt = {Ddt:.5f} ± {sdt:.5f} ({Ddt / sdt if sdt > 0 else float('nan'):.1f} se)   D_G = {DG:.5f} ± {sG:.5f}   ratio {Ddt / DG if DG > 0 else float('nan'):.2f}   -> {v}"
        )
    n_bad = int(sum(verdicts))
    print(
        "OUTCOME:",
        "ADEQUATE" if n_bad == 0 else ("STEP-DEPENDENT" if n_bad >= 2 else "MIXED"),
        f"({n_bad} of {len(PRIMARY)} primary measures move by at least a grid step and beyond 2 se)",
    )

    print("\n=== trend across dt at the working point ===")
    for m in PRIMARY + ["switching_index"]:
        vals = [ms([r.get(m) for r in cell(WP_G, dt)]) for dt in (0.02, 0.01, 0.005, 0.0025)]
        inc = [vals[i + 1][0] - vals[i][0] for i in range(3)]
        mono = all(x > 0 for x in inc) or all(x < 0 for x in inc)
        last_se = np.hypot(vals[2][1], vals[3][1])
        print(
            f"   {m:16s} "
            + "  ".join(f"{a:.5f}±{s:.5f}" for a, s, _ in vals)
            + f"   monotone: {mono}; last increment {inc[2]:+.5f} ({abs(inc[2]) / last_se if last_se > 0 else float('nan'):.1f} se)"
        )


if __name__ == "__main__":
    main(sys.argv[1])
