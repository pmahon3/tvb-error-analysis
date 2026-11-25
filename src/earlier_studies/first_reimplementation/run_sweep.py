"""Run a (G, sigma) batch of the MPR network and compute the paper's statistics.
python3 -I run_sweep.py OUT.npz --G 0.1,0.3 --sigma 0.1,0.1 --seed 1 --T 300000 [--dt 0.1] [--rewire SEED]
Writes OUT.npz (bold, r10, G, sigma, stats as JSON string) and prints a table.
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[2] / "core"),
]

import argparse, json, sys, time, os
import numpy as np

import mpr_net as M, stats as S

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--G", required=True)
ap.add_argument("--sigma", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--T", type=float, default=300_000)
ap.add_argument("--dt", type=float, default=0.1)
ap.add_argument("--rewire", type=int, default=None)
ap.add_argument("--burn_s", type=float, default=20.0)
ap.add_argument("--init_up", default=None, help="comma list of initial up-fractions per config")
a = ap.parse_args()
G = np.array([float(x) for x in a.G.split(",")])
sig = np.array([float(x) for x in a.sigma.split(",")])
assert len(G) == len(sig)
W, L = M.load_connectome(os.path.dirname(os.path.abspath(__file__)))
if a.rewire is not None:
    W = M.within_row_permute(W, np.random.default_rng(a.rewire))
t0 = time.time()
r0 = v0 = None
if a.init_up is not None:
    fr = np.array([float(x) for x in a.init_up.split(",")])
    assert len(fr) == len(G)
    rng0 = np.random.default_rng(a.seed * 7919 + 1)
    N = W.shape[0]
    up = rng0.random((len(G), N)) < fr[:, None]
    r0 = np.where(up, M.R_UP, M.R_DN) + 0.01 * rng0.standard_normal((len(G), N))
    v0 = np.where(up, M.V_UP, M.V_DN)
out = M.run_batch(W, L, G, sig, a.seed, T_ms=a.T, dt=a.dt, progress=True, r0=r0, v0=v0)
el = time.time() - t0
burn_b = int(a.burn_s / (out["tr_ms"] / 1000))
burn_r = int(a.burn_s * 1000 / out["keep_ms"])
rows = []
for b in range(len(G)):
    bold = out["bold"][b, burn_b:].astype(float)
    r10 = out["r"][b, burn_r:].astype(float)
    st, D = S.all_stats(bold, tr_s=out["tr_ms"] / 1000)
    # state from r smoothed over 100 ms (10 kept samples) to count real up/down switches, not noise jitter
    k = 10
    rs = np.apply_along_axis(lambda x: np.convolve(x, np.ones(k) / k, mode="valid"), 0, r10)
    st.update(
        G=float(G[b]),
        sigma=float(sig[b]),
        vaf2_r=S.vaf2(r10),
        frac_up=float((rs > 0.5).mean()),
        n_switch=int((np.abs(np.diff((rs > 0.5).astype(int), axis=0)) > 0).sum()),
    )
    rows.append(st)
np.savez_compressed(
    a.out,
    bold=out["bold"],
    r10=out["r"],
    G=G,
    sigma=sig,
    init_up=(a.init_up or ""),
    stats=json.dumps(rows),
    rewire=-1 if a.rewire is None else a.rewire,
    seed=a.seed,
    T=a.T,
    dt=a.dt,
)
print(f"wall {el / 60:.1f} min for {len(G)} configs x {a.T / 1000:.0f} s")
print("G     sigma  switch    vaf2_bold vaf2_r  frac_up n_switch within cross  nonev  n_exc")
for s in rows:
    print(
        f"{s['G']:<5.2f} {s['sigma']:<6.2f} {s['switch']:<9.4f} {s['vaf2']:<9.3f} {s['vaf2_r']:<7.3f} "
        f"{s['frac_up']:<7.2f} {s['n_switch']:<8d} {s['within']:<6.2f} {s['cross']:<6.2f} {s['nonevents']:<6.2f} {s['n_excursions']}"
    )
