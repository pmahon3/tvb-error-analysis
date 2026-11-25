"""M2: statistics of the JAX port against the TVB reference runs of stage A' (doc/LOG.md, 2026-11-24, "JAX port").
Usage: python3 -I mprjax_m2_score.py ../results/runs/Aprime ../results/runs/port/m2_dt0.01.json ../results/runs/port/m2_dt0.005.json
"""
import sys, json, glob, os
import numpy as np
PRIMARY = ["m1_mean_r", "m2_occ_up07", "m3_u_median", "m4_u_p05", "m5_vaf2_r"]; SECONDARY = ["switching_index", "mean_fc", "corr_fc_sc"]
ms = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))), len(v))
tvb = {}
for p in glob.glob(os.path.join(sys.argv[1], "A_*.json")):
    r = json.load(open(p))
    if r["finite"]: tvb.setdefault((r["G"], r["dt"]), []).append(r)
jx = {}
for p in sys.argv[2:]:
    for r in json.load(open(p))["runs"]:
        if r["finite"]: jx.setdefault((r["G"], r["dt"]), []).append(r)
        else: print("non-finite JAX run:", r["G"], r["dt"], r["seed"])
n_reg, beyond2, beyond3 = 0, 0, 0
for key in sorted(jx):
    if key not in tvb: continue
    registered = key in ((0.54, 0.01), (0.54, 0.005))
    print(f"\nG = {key[0]}, dt = {key[1]}  (TVB n = {len(tvb[key])}, JAX n = {len(jx[key])}){'   [registered comparison]' if registered else '   [description]'}")
    for m in PRIMARY + SECONDARY:
        a, sa, _ = ms([r[m] for r in tvb[key]]); b, sb, _ = ms([r[m] for r in jx[key]]); se = np.hypot(sa, sb); z = (b - a) / se if se > 0 else float("nan")
        flag = ""
        if registered and m in PRIMARY:
            n_reg += 1; beyond2 += abs(z) > 2; beyond3 += abs(z) > 3; flag = "  <-- beyond 3 se" if abs(z) > 3 else ("  <-- beyond 2 se" if abs(z) > 2 else "")
        print(f"   {m:16s} TVB {a:9.5f} ± {sa:.5f}   JAX {b:9.5f} ± {sb:.5f}   difference {b - a:+.5f} ({z:+.1f} se){flag}")
print(f"\nM2: {n_reg} registered comparisons; beyond 2 se: {beyond2}; beyond 3 se: {beyond3}  ->  {'PASS' if beyond3 == 0 and beyond2 <= 2 else 'FAIL'} (criterion: none beyond 3 se, at most two beyond 2 se)")
