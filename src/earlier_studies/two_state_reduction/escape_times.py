"""Single-node escape times as a function of input I and noise sigma (for the kinetic Ising reduction, G3).
python3 -I escape_times.py OUT.npz --sigma 0.1 [--I 0,0.25,...,2.0] [--n 200] [--Tmax 20000]
For each I: n trajectories started at the down state and n at the up state; time to cross the saddle's
r value (r_saddle(I)) is the escape time. Reports mean escape times down->up and up->down.
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[2] / "core"),
]

import argparse, os, sys
import numpy as np
from scipy.optimize import brentq

import mpr_net as M

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--sigma", type=float, required=True)
ap.add_argument("--I", default="0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--Tmax", type=float, default=20000)
ap.add_argument("--dt", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=3)
a = ap.parse_args()
Is = np.array([float(x) for x in a.I.split(",")])


def F(r, I):
    v = -M.DELTA / (2 * M.PI * r)
    return v * v + M.ETA - (M.PI * r) ** 2 + M.J * r + I


def fps(I):
    rs = np.linspace(0.005, 6, 20000)
    out = []
    for lo, hi in zip(rs[:-1], rs[1:]):
        if F(lo, I) * F(hi, I) < 0:
            out.append(brentq(F, lo, hi, args=(I,)))
    return out


rng = np.random.default_rng(a.seed)
steps = int(a.Tmax / a.dt)
res = []
for I in Is:
    f = fps(I)
    if len(f) < 3:
        res.append((I, np.nan, np.nan, len(f)))
        continue
    r_dn, r_sd, r_up = f
    v_dn, v_up = -M.DELTA / (2 * M.PI * r_dn), -M.DELTA / (2 * M.PI * r_up)
    n = a.n
    r = np.concatenate([np.full(n, r_dn), np.full(n, r_up)])
    v = np.concatenate([np.full(n, v_dn), np.full(n, v_up)])
    t_esc = np.full(2 * n, np.nan)
    alive = np.ones(2 * n, bool)
    sq = np.sqrt(a.dt)
    for k in range(steps):
        dr, dv = M.drift(r, v, I)
        rp, vp = r + a.dt * dr, v + a.dt * dv
        dr2, dv2 = M.drift(rp, vp, I)
        r = np.maximum(r + 0.5 * a.dt * (dr + dr2) + a.sigma * sq * rng.standard_normal(2 * n), 0.0)
        v = v + 0.5 * a.dt * (dv + dv2) + a.sigma * sq * rng.standard_normal(2 * n)
        crossed = alive & (
            ((np.arange(2 * n) < n) & (r > r_sd)) | ((np.arange(2 * n) >= n) & (r < r_sd))
        )
        t_esc[crossed] = (k + 1) * a.dt
        alive &= ~crossed
        if not alive.any():
            break
    tdu, tud = t_esc[:n], t_esc[n:]
    res.append((I, np.nanmean(tdu), np.nanmean(tud), len(f)))
    print(
        f"I={I:5.2f}  mean escape down->up {np.nanmean(tdu):9.1f} (censored {np.isnan(tdu).mean():.2f})   up->down {np.nanmean(tud):9.1f} (censored {np.isnan(tud).mean():.2f})",
        flush=True,
    )
np.savez(a.out, I=Is, res=np.array(res), sigma=a.sigma, Tmax=a.Tmax)
