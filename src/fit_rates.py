"""Empirical flip hazards of the full model as a function of instantaneous input, for the two-state reduction.
python3 -I fit_rates.py OUT.npz ../results/runs/g0d_basins.npz [more npz]   (uses r10 at 10 ms; input I_i = G sum_j W_ij r_j)
State: r smoothed over 100 ms > 0.5. Hazard k(I) = flips / exposure time, in events per ms, binned in I.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mpr_net as M
W, L = M.load_connectome(os.path.dirname(os.path.abspath(__file__)))
edges = np.linspace(0, 2.0, 21); nb = len(edges) - 1
flip_du = np.zeros(nb); exp_d = np.zeros(nb); flip_ud = np.zeros(nb); exp_u = np.zeros(nb)
for f in sys.argv[2:]:
    z = np.load(f, allow_pickle=True); R = z["r10"]; G = z["G"]
    for b in range(R.shape[0]):
        r = R[b, 2000:].astype(float); I = G[b] * (r @ W.T)                       # (T, N)
        k = 10; rs = np.apply_along_axis(lambda x: np.convolve(x, np.ones(k) / k, mode="valid"), 0, r); I = I[k - 1:]
        s = rs > 0.5; ds = np.diff(s.astype(int), axis=0)                          # +1 down->up, -1 up->down
        Ib = np.clip(np.digitize(I[:-1], edges) - 1, 0, nb - 1)
        for j in range(nb):
            m = Ib == j
            exp_d[j] += (m & ~s[:-1]).sum() * 10.0; exp_u[j] += (m & s[:-1]).sum() * 10.0   # ms of exposure
            flip_du[j] += (m & (ds == 1)).sum(); flip_ud[j] += (m & (ds == -1)).sum()
Ic = 0.5 * (edges[:-1] + edges[1:]); k_du = np.where(exp_d > 0, flip_du / np.maximum(exp_d, 1), np.nan); k_ud = np.where(exp_u > 0, flip_ud / np.maximum(exp_u, 1), np.nan)
good = np.isfinite(k_du) & np.isfinite(k_ud) & (exp_d > 5000) & (exp_u > 5000)
print(" I     k_du(/ms)   k_ud(/ms)   tau_du(s)  tau_ud(s)  exposure_d(s) exposure_u(s)")
for j in range(nb):
    if exp_d[j] + exp_u[j] > 0:
        print(f"{Ic[j]:4.2f}  {k_du[j]:.2e}   {k_ud[j]:.2e}   {1/k_du[j]/1000 if k_du[j]>0 else np.inf:9.2f}  {1/k_ud[j]/1000 if k_ud[j]>0 else np.inf:9.2f}  {exp_d[j]/1000:10.0f} {exp_u[j]/1000:10.0f}")
np.savez(sys.argv[1], I=Ic[good], k_du=k_du[good], k_ud=k_ud[good])
print("saved bins:", good.sum())
