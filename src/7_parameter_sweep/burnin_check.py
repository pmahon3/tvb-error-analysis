"""Burn-in check (user, 2026-11-24: "chain burn in?"): do the statistics drift within a run, in TVB's runs and in the
port's? Uses the per-TR arrays already saved on the tower. Working point (G, sigma) = (0.54, 0.030), dt 0.01.
Usage (tower, venv_tvb): python burnin_check.py
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "common")]

import os, glob, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from stats import dfc_w, switching_index

def per_run(bold, u07):
    """bold (n_tr, N), u07 (n_tr,). Statistics on four quarters of the record and on two halves."""
    n = bold.shape[0]; q = n // 4; iu = np.triu_indices(bold.shape[1], 1); out = {}
    out["u07_q"] = [float(u07[k * q:(k + 1) * q].mean()) for k in range(4)]
    out["u07_first20"] = float(u07[:20].mean()); out["u07_tr"] = u07[:n]
    fc = []
    for k in range(4):
        b = bold[max(k * q, 41):(k + 1) * q].astype(np.float64); fc.append(float(np.nan_to_num(np.corrcoef(b.T))[iu].mean()))
    out["fc_q"] = fc; si = []
    for a, z in ((41, n // 2), (n // 2, n)):
        b = bold[a:z].astype(np.float64); D, Ws = dfc_w(b); si.append(float(switching_index(D, Ws)))
    out["si_h"] = si
    return out

groups = {}
tv = []
for p in sorted(glob.glob(os.path.join(HERE, "runs_A", "A_G0.54_sig0.030_dt0.01_seed*.npz"))):
    d = np.load(p); tv.append(per_run(d["bold"], d["u07"]))
groups["TVB"] = tv
port = []
for name in ("prec_f32", "prec_f64"):
    p = os.path.join(HERE, "runs_port", name + ".npz")
    if os.path.exists(p):
        d = np.load(p); g = [per_run(d["bold"][b], d["u07"][b]) for b in range(d["bold"].shape[0])]; groups["port " + name[5:]] = g; port += g
p = os.path.join(HERE, "q_pilot", "done", "pilot.npz")
if os.path.exists(p):
    d = np.load(p); P = d["params"]; C = len(P); c = [i for i in range(C) if np.allclose(P[i], [0.54, -5.0, 15.0, 1.0, 0.030])][0]
    rows = [c + C * s for s in range(len(d["seeds"]))]; g = [per_run(d["bold"][b], d["u07"][b]) for b in rows]; groups["port pilot f32"] = g; port += g
groups["port, all"] = port
ms = lambda v: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)))
f = lambda v: "%.4f ± %.4f" % ms(v)
for name, g in groups.items():
    if not g: continue
    print(f"\n{name}: n = {len(g)}")
    print("   up-fraction, first 20 TRs after the discard:", f([x["u07_first20"] for x in g]))
    print("   up-fraction by quarter of the record:  ", " | ".join(f([x["u07_q"][k] for x in g]) for k in range(4)))
    print("   mean FC by quarter:                    ", " | ".join(f([x["fc_q"][k] for x in g]) for k in range(4)))
    print("   switching index, first half | second half:", " | ".join(f([x["si_h"][k] for x in g]) for k in range(2)))
    m = np.mean([x["u07_tr"] for x in g], 0)
    print("   mean up-fraction over seeds at TR 1, 2, 3, 5, 10, 20, 50, 100, 400, 800:", " ".join("%.3f" % m[i] for i in (0, 1, 2, 4, 9, 19, 49, 99, 399, 799)))
