"""Gate G3: kinetic Ising (two-state) reduction of the MPR network on the same connectome.
Each node is up (r = R_UP) or down (r = R_DN). Input I_i = G sum_j W_ij r_j (delays dropped: they are < 0.1 s,
far below the escape-time scale). Flip rates: 1/tau_esc(I_i) from the single-node escape-time table
(escape_times.py), interpolated in I; a global rate scale 'speed' is the one free knob.
Output: r(t) at 1 ms as a step process, then the same Balloon-Windkessel as the full model.
python3 -I ising_net.py OUT.npz --esc esc_s0.1.npz --G 0.5,0.7 --speed 1,1 --seed 1 --T 300000
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[2] / "core"),
]

import argparse, json, os, sys
import numpy as np

import mpr_net as M, stats as S

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--esc", default=None)
ap.add_argument("--rates", default=None, help="npz from fit_rates.py (overrides --esc)")
ap.add_argument("--G", required=True)
ap.add_argument("--speed", default=None)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--T", type=float, default=300_000)
ap.add_argument("--init_up", default=None)
ap.add_argument("--rewire", type=int, default=None)
ap.add_argument("--burn_s", type=float, default=20.0)
a = ap.parse_args()
G = np.array([float(x) for x in a.G.split(",")])
B = len(G)
speed = np.ones(B) if a.speed is None else np.array([float(x) for x in a.speed.split(",")])
if a.rates:
    q = np.load(a.rates)
    Ic, k_du, k_ud = q["I"], q["k_du"], q["k_ud"]  # empirical hazards per ms from the full model

    def rate_du(I):
        return np.interp(I, Ic, k_du)

    def rate_ud(I):
        return np.interp(I, Ic, k_ud)

else:
    e = np.load(a.esc)
    Iv, res = e["I"], e["res"]
    ok = np.isfinite(res[:, 1]) & np.isfinite(res[:, 2])
    I_sn = Iv[
        ok
    ].max()  # beyond the last input with a down state, the down state has vanished (saddle-node near I = 1.87)

    def rate_du(I):
        return np.where(I > I_sn, 1.0, np.interp(I, Iv[ok], 1.0 / res[ok, 1]))

    def rate_ud(I):
        return np.where(I > I_sn, 0.0, np.interp(I, Iv[ok], 1.0 / res[ok, 2]))


W, L = M.load_connectome(os.path.dirname(os.path.abspath(__file__)))
if a.rewire is not None:
    W = M.within_row_permute(W, np.random.default_rng(a.rewire))
N = W.shape[0]
rng = np.random.default_rng(a.seed)
dt = 1.0  # ms
fr = np.full(B, 0.5) if a.init_up is None else np.array([float(x) for x in a.init_up.split(",")])
s = rng.random((B, N)) < fr[:, None]
hemo = M.Balloon(B, N)
r_mid = 0.5 * (M.R_UP + M.R_DN)
u_scale = M.R_UP - M.R_DN
T = int(a.T)
keep = 10
tr = 720
r_out = np.zeros((B, T // keep, N), np.float32)
bold = np.zeros((B, T // tr, N), np.float32)
for t in range(T):
    r = np.where(s, M.R_UP, M.R_DN)
    I = G[:, None] * (r @ W.T)
    p_up = 1 - np.exp(-speed[:, None] * rate_du(I) * dt)
    p_dn = 1 - np.exp(-speed[:, None] * rate_ud(I) * dt)
    u = rng.random((B, N))
    s = np.where(s, u >= p_dn, u < p_up)
    hemo.step(r, dt / 1000.0)  # firing-rate drive, as in mpr_net (fixed 2026-11-17)
    if (t + 1) % keep == 0:
        r_out[:, (t + 1) // keep - 1] = r
    if (t + 1) % tr == 0:
        bold[:, (t + 1) // tr - 1] = hemo.bold()
burn_b = int(a.burn_s * 1000 / tr)
burn_r = int(a.burn_s * 1000 / keep)
rows = []
print("G     speed  switch    vaf2_bold vaf2_r  frac_up n_switch within cross  nonev  n_exc")
for b in range(B):
    st, D = S.all_stats(bold[b, burn_b:].astype(float), 0.72)
    r10 = r_out[b, burn_r:].astype(float)
    st.update(
        G=float(G[b]),
        speed=float(speed[b]),
        vaf2_r=S.vaf2(r10),
        frac_up=float((r10 > 0.5).mean()),
        n_switch=int((np.abs(np.diff((r10 > 0.5).astype(int), axis=0)) > 0).sum()),
    )
    rows.append(st)
    print(
        f"{st['G']:<5.2f} {st['speed']:<6.2f} {st['switch']:<9.4f} {st['vaf2']:<9.3f} {st['vaf2_r']:<7.3f} {st['frac_up']:<7.2f} "
        f"{st['n_switch']:<8d} {st['within']:<6.2f} {st['cross']:<6.2f} {st['nonevents']:<6.2f} {st['n_excursions']}"
    )
np.savez_compressed(
    a.out, bold=bold, r10=r_out, G=G, speed=speed, stats=json.dumps(rows, default=float)
)
