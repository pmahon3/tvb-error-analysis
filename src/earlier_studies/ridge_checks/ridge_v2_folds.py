"""Ridge check v2, stage 0: fold curves of the all-low and all-high branches of the undelayed MPR network (vbjax node).
Registered in doc/LOG.md (2026-11-23, night). Run: .venv_vbjax/bin/python -I ridge_v2_folds.py"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[2] / "core"),
]

import os, sys, json
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import vbjax as vb

HERE = os.path.dirname(os.path.abspath(__file__))
import sys

from repo_paths import data, RUNS

W = np.loadtxt(data("weights.txt"))
np.fill_diagonal(W, 0.0)
W = W / W.sum(1).mean()
Wj = jnp.asarray(W)
N = W.shape[0]
s = W.sum(1)
R_DN, V_DN, R_UP, V_UP = 0.081, -1.962, 1.031, -0.154


def F(z, G, eta):
    rv = z.reshape(2, N)
    p = vb.mpr_default_theta._replace(eta=eta)
    c = jnp.stack([G * (Wj @ rv[0]), jnp.zeros(N)])
    return vb.mpr_dfun(rv, c, p).reshape(-1)


JF = jax.jit(jax.jacfwd(F))
Fj = jax.jit(F)


def newton(z, G, eta, it=40):
    for _ in range(it):
        f = Fj(z, G, eta)
        if not np.all(np.isfinite(np.asarray(f))):
            return None
        if float(jnp.abs(f).max()) < 1e-11:
            return z
        z = z - jnp.linalg.solve(JF(z, G, eta), f)
    return z if float(jnp.abs(Fj(z, G, eta)).max()) < 1e-11 else None


def stable_on_branch(z, G, eta, kind):
    """Equilibrium near z that is stable and still of the branch's kind (all nodes low, or all nodes high)."""
    z1 = newton(z, G, eta)
    if z1 is None:
        return None
    r = np.asarray(z1[:N])
    ok_kind = (
        (r < 0.4).all() if kind == "low" else (r[s > 0] > 0.4).all()
    )  # zero-input nodes left free (amendment)
    lam = np.linalg.eigvals(np.asarray(JF(z1, G, eta))).real.max()
    return z1 if (ok_kind and lam < 0) else None


def branch_end(G, kind):
    z = jnp.asarray(
        np.r_[
            np.full(N, R_DN if kind == "low" else R_UP), np.full(N, V_DN if kind == "low" else V_UP)
        ]
    )
    eta, de = (-8.0, 0.005) if kind == "low" else (-3.0, -0.005)
    z = stable_on_branch(z, G, eta, kind)
    assert z is not None, (G, kind, "start not on branch")
    while True:
        z2 = stable_on_branch(z, G, eta + de, kind)
        if z2 is None:
            break
        z, eta = z2, eta + de
        if not (-15.0 < eta < 0.0):
            return None
    lo, hi = eta, eta + de
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        zm = stable_on_branch(z, G, mid, kind)
        if zm is None:
            hi = mid
        else:
            lo, z = mid, zm
    r = np.asarray(z[:N])
    idx = int(np.argmax(np.where(s > 0, np.abs(r - (R_DN if kind == "low" else R_UP)), -1)))
    return lo, idx


Gs = [0.95, 1.0, 1.05, 1.1, 1.15]
res = {"low": [], "high": []}
for G in Gs:
    for kind in ("low", "high"):
        out = branch_end(G, kind)
        assert out is not None, (G, kind)
        e_end, node = out
        res[kind].append(e_end)
        print(
            f"G={G:.2f}  {kind:4s} branch ends at eta = {e_end:+.5f}  (node moving most: {node}, in-strength {s[node]:.3f})",
            flush=True,
        )
sl = {k: np.polyfit(Gs, res[k], 1)[0] for k in res}
for k in res:
    print(f"{k} fold curve: d eta / d G = {sl[k]:+.4f}")
i0 = Gs.index(1.05)
print(
    f"S0-1 at G=1.05: low branch exists for eta < {res['low'][i0]:+.4f}, high branch for eta > {res['high'][i0]:+.4f}; "
    f"eta=-5.0 inside both: {res['high'][i0] < -5.0 < res['low'][i0]}"
)
print(f"S0-2 slope ratio low/high = {sl['low'] / sl['high']:.3f}")
json.dump(
    dict(Gs=Gs, **res, slope_low=sl["low"], slope_high=sl["high"]),
    open(os.path.join(RUNS, "earlier_studies", "ridge_checks", "ridge_v2_folds_amended.json"), "w"),
)
