"""Apply the offset and seed rules registered in doc/LOG.md (2026-11-24, 12:05) to the pilot batch.
Usage: python3 -I sweep_pilot_score.py results/runs/sweep/pilot_done.json
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "common")]

import sys, json
import numpy as np
runs = [r for r in json.load(open(sys.argv[1]))["runs"]]; print("runs", len(runs), "| finite", sum(r["finite"] for r in runs))
BASE = dict(eta=-5.0, J=15.0, Delta=1.0); WP = 0.54
cell = lambda G, **kw: [r for r in runs if r["finite"] and abs(r["G"] - G) < 1e-9 and all(abs(r[k] - kw.get(k, BASE[k])) < 1e-9 for k in BASE)]
ms = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)), len(v))
OFFS = dict(eta=(0.005, 0.01, 0.02), J=(0.02, 0.05, 0.10), Delta=(0.005, 0.01, 0.02))
STATS = ["m1_mean_r", "m2_occ_up07", "m5_vaf2_r", "switching_index", "mean_fc"]
chosen = {}
for G in (0.45, 0.54, 0.65):
    b = cell(G); print(f"\nG = {G}: base n = {len(b)}; " + "; ".join(f"{m} {ms([r[m] for r in b])[0]:.4f} (sd {ms([r[m] for r in b])[1]:.4f})" for m in STATS))
    for p, offs in OFFS.items():
        for d in offs:
            up, dn = cell(G, **{p: BASE[p] + d}), cell(G, **{p: BASE[p] - d}); line = f"   {p:5s} ±{d:<6g}"
            for m in STATS[:1] + STATS[2:4]:
                mb, sb, nb_ = ms([r[m] for r in b]); mu, su, nu = ms([r[m] for r in up]); md, sd_, nd = ms([r[m] for r in dn])
                cd = mu - md; asym = abs((mu - mb) + (md - mb)) / (abs(mu - mb) + abs(md - mb) + 1e-12)
                se16 = np.hypot(su, sd_) / np.sqrt(nu); se4 = np.hypot(su, sd_) / 2.0; se8 = np.hypot(su, sd_) / np.sqrt(8)
                line += f" | {m[:8]}: +{mu - mb:+.4f} / -{md - mb:+.4f}; central {cd:+.4f} = {abs(cd) / se4:.1f} se@4, {abs(cd) / se8:.1f} se@8; asym {asym:.2f}"
                if m == "m1_mean_r" and G == WP:
                    chosen.setdefault(p, []).append(dict(d=d, cd=cd, z4=abs(cd) / se4, z8=abs(cd) / se8, one_sided=max(abs(mu - mb), abs(md - mb)), asym=asym))
            print(line)
print("\n=== registered rule (mean rate at the working point) ===")
need8 = False; picks = {}
for p, rows in chosen.items():
    ok4 = [r for r in rows if r["z4"] >= 3 and r["one_sided"] <= 0.10]; ok8 = [r for r in rows if r["z8"] >= 3 and r["one_sided"] <= 0.10]
    pick = ok4[0] if ok4 else (ok8[0] if ok8 else None)
    if not ok4: need8 = True
    picks[p] = pick
    print(f"   {p:5s}: " + ("no offset meets the rule" if pick is None else f"delta = {pick['d']}; central difference {pick['cd']:+.4f}; {pick['z4']:.1f} se at 4 seeds, {pick['z8']:.1f} at 8; largest one-sided response {pick['one_sided']:.4f}; asymmetry {pick['asym']:.2f}"))
print("seeds per configuration:", 8 if need8 else 4)
