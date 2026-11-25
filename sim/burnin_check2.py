"""Follow-up to burnin_check.py (README, 2026-11-24): (a) TVB halves split by seed group 1-32 against 33-96;
(b) the transient across the sweep grid: up-occupancy over all TRs against the same without the first 50 TRs.
Usage (tower, venv_tvb): python burnin_check2.py
"""
import os, glob, sys, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from burnin_check import per_run, ms, f
tv = {}
for p in sorted(glob.glob(os.path.join(HERE, "runs_A", "A_G0.54_sig0.030_dt0.01_seed*.npz"))):
    sd = int(re.search(r"seed(\d+)", p).group(1)); d = np.load(p); tv[sd] = per_run(d["bold"], d["u07"])
for name, sel in (("TVB seeds 1-32", [s for s in tv if s <= 32]), ("TVB seeds 33-96", [s for s in tv if s > 32]), ("TVB all", list(tv))):
    g = [tv[s] for s in sel]; print(f"\n{name}: n = {len(g)}")
    print("   up-fraction, first 20 TRs:", f([x["u07_first20"] for x in g]), "| by quarter:", " | ".join(f([x["u07_q"][k] for x in g]) for k in range(4)))
    print("   mean FC by quarter:", " | ".join(f([x["fc_q"][k] for x in g]) for k in range(4)))
    print("   switching index, first half | second half:", " | ".join(f([x["si_h"][k] for x in g]) for k in range(2)))

print("\n(b) sweep, base configuration, dt 0.01, float32: up-occupancy over all 833 TRs (A) and over TRs 51-833 (B); mean over finite seeds")
cells = {}
for p in sorted(glob.glob(os.path.join(HERE, "q_sweep", "done", "a_dt0.01_*.json"))):
    d = json.load(open(p)); z = np.load(p[:-5] + ".npz"); u = z["u07"]
    for i, r in enumerate(d["runs"]):
        if r["finite"] and abs(r["eta"] + 5) < 1e-9 and abs(r["J"] - 15) < 1e-9 and abs(r["Delta"] - 1) < 1e-9:
            cells.setdefault((round(r["nsig"], 4), round(r["G"], 4)), []).append((u[i].mean(), u[i][50:].mean(), u[i][:20].mean()))
Ss = sorted({k[0] for k in cells}); Gs = sorted({k[1] for k in cells})
A = np.array([[np.mean([x[0] for x in cells[(s, G)]]) for G in Gs] for s in Ss]); B = np.array([[np.mean([x[1] for x in cells[(s, G)]]) for G in Gs] for s in Ss])
E = np.array([[np.mean([x[2] for x in cells[(s, G)]]) for G in Gs] for s in Ss])
print("G:        " + " ".join(f"{G:6.2f}" for G in Gs[::2]))
for i, s in enumerate(Ss):
    print(f"sigma {s:.3f}  A-B (absolute, x1000): " + " ".join(f"{1000 * (A[i, j] - B[i, j]):6.1f}" for j in range(0, len(Gs), 2)))
for i, s in enumerate(Ss):
    print(f"sigma {s:.3f}  B (x1000):             " + " ".join(f"{1000 * B[i, j]:6.1f}" for j in range(0, len(Gs), 2)))
for i, s in enumerate(Ss):
    print(f"sigma {s:.3f}  first 20 TRs (x1000):  " + " ".join(f"{1000 * E[i, j]:6.1f}" for j in range(0, len(Gs), 2)))
rel = np.abs(A - B) / np.maximum(B, 1e-3); j, i = np.unravel_index(np.argmax(np.abs(A - B)), A.shape)[::-1]
print("largest |A-B|:", round(float(np.abs(A - B).max()), 4), "at sigma", Ss[i], "G", Gs[j], "| points with |A-B| > 0.005:", int((np.abs(A - B) > 0.005).sum()), "of", A.size, "| > 0.002:", int((np.abs(A - B) > 0.002).sum()))
