"""Structural identifiability probe for the MPR network, from the equations alone (registered in ../doc/LOG.md, 2026-11-23).
Run: .venv_vbjax/bin/python -I ridge_v2_structural.py"""
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import vbjax as vb

HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE); from repo_paths import data, RUNS
W = np.loadtxt(data("weights.txt")); np.fill_diagonal(W, 0.0); W = W / W.sum(1).mean()
Wj = jnp.asarray(W); N = W.shape[0]; s_in = W.sum(1)
R_DN, V_DN, R_UP, V_UP = 0.081, -1.962, 1.031, -0.154
NAMES = ["G", "eta", "J", "Delta", "tau", "w"]
TH0 = jnp.array([1.05, -5.0, 15.0, 1.0, 1.0, 1.0])
n1 = np.array([1.0, 0, 0, 0, 0, -1.0]); n2 = np.array([2.0, 2, 1, 2, 1, 0])       # log-parameter directions
NPRED = np.linalg.qr(np.stack([n1, n2], 1))[0]
DT, NSTEP, EVERY = 0.01, 5000, 25


def field(rv, th):
    G, eta, J, Delta, tau, w = th
    p = vb.mpr_default_theta._replace(eta=eta, J=J, Delta=Delta, tau=tau)
    return vb.mpr_dfun(rv, jnp.stack([G * w * (Wj @ rv[0]), jnp.zeros(N)]), p)


def traj(th, r0, v0):
    f = lambda z: field(z, th)
    def step(z, _):
        k1 = f(z); k2 = f(z + 0.5 * DT * k1); k3 = f(z + 0.5 * DT * k2); k4 = f(z + DT * k3)
        z = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return z, z[0]
    _, rs = jax.lax.scan(step, jnp.stack([r0, v0]), None, length=NSTEP)
    return rs[EVERY - 1::EVERY]


def report(tag, Jrel):
    U, S, Vt = np.linalg.svd(Jrel, full_matrices=False)
    rel = S / S[0]; k = int((rel < 1e-9).sum())
    print(f"{tag}\n    singular values / largest: {np.array2string(rel, precision=2)}   (# below 1e-9: {k})")
    if k:
        cos = np.linalg.svd(Vt[-k:] @ NPRED, compute_uv=False)
        print(f"    null vectors (log-parameters {NAMES}):")
        for v in Vt[-k:]: print("      ", np.round(v / np.abs(v).max(), 4).tolist())
        print(f"    principal cosines with span(n1, n2): {np.array2string(cos, precision=8)}")
    for nm, n in (("n1", n1), ("n2", n2)):
        print(f"    |J {nm}| / (|J| |{nm}|) = {np.linalg.norm(Jrel @ n) / (S[0] * np.linalg.norm(n)):.1e}")
    if k < len(S): print(f"    gap: smallest non-null / largest null = {rel[len(S) - k - 1] / max(rel[-1], 1e-300) if k else float('nan'):.1e}")
    return rel, k


rng = np.random.default_rng(7)
up = rng.random(N) < 0.5
r0 = jnp.asarray(np.where(up, R_UP, R_DN) * (1 + 0.3 * rng.uniform(-1, 1, N)))
v0 = jnp.asarray(np.where(up, V_UP, V_DN) + 0.3 * rng.uniform(-1, 1, N))
scale = np.abs(np.asarray(TH0))                                     # relative sensitivities: columns times |theta|

print("T1  global scaling symmetry, lambda = 1.3")
lam = 1.3
th_l = TH0 * jnp.array([lam**2, lam**2, lam, lam**2, lam, 1.0])
ra, rb = traj(TH0, r0, v0), traj(th_l, r0, lam * v0)
print(f"    max |r - r_scaled| over 50 ms = {float(jnp.abs(ra - rb).max()):.1e}   (max r = {float(ra.max()):.2f}; "
      f"control without rescaling v0: {float(jnp.abs(ra - traj(th_l, r0, v0)).max()):.1e})")

rec = lambda th, v0_: traj(th, r0, v0_).reshape(-1)
Jth = np.asarray(jax.jacfwd(rec, 0)(TH0, v0)) * scale
Jv = np.asarray(jax.jacfwd(rec, 1)(TH0, v0))
Q = np.linalg.qr(Jv)[0]; Jp = Jth - Q @ (Q.T @ Jth)
rel2, k2 = report("T2  record r (transient), hidden start v0 as nuisance", Jp)
rel3, k3 = report("T3  record r (transient), v0 known", Jth)
for tag, M in (("v0 nuisance", Jp), ("v0 known", Jth)):
    S = np.linalg.svd(M[:, :2], compute_uv=False)
    c = abs(M[:, 0] @ M[:, 1]) / (np.linalg.norm(M[:, 0]) * np.linalg.norm(M[:, 1]))
    print(f"T4  (G, eta) only, {tag}: sigma_min/sigma_max = {S[1] / S[0]:.3e}; |cos(column G, column eta)| = {c:.4f}")

print("T5  equilibrium record (implicit function theorem)")
F = lambda z, th: field(z.reshape(2, N), th).reshape(-1)
dFz = jax.jit(jax.jacfwd(F, 0)); dFth = jax.jit(jax.jacfwd(F, 1))
starts = {"all-low state": np.r_[np.full(N, R_DN), np.full(N, V_DN)],
          "high cluster": np.r_[np.where(s_in > 0, R_UP, R_DN), np.where(s_in > 0, V_UP, V_DN)]}
for name, z in starts.items():
    z = jnp.asarray(z)
    for _ in range(60):
        f = F(z, TH0)
        if float(jnp.abs(f).max()) < 1e-12: break
        z = z - jnp.linalg.solve(dFz(z, TH0), f)
    A = np.asarray(dFz(z, TH0)); lam_max = np.linalg.eigvals(A).real.max()
    r_eq = np.asarray(z[:N])
    dz = -np.linalg.solve(A, np.asarray(dFth(z, TH0)))
    Jeq = dz[:N] * scale
    print(f"  {name}: residual {float(jnp.abs(F(z, TH0)).max()):.1e}; max Re eig {lam_max:+.3f}; nodes up {int((r_eq > 0.5).sum())}/{N}; "
          f"r range {r_eq.min():.3f}-{r_eq.max():.3f}")
    report(f"    equilibrium rates, {name}", Jeq)
    S = np.linalg.svd(Jeq[:, :2], compute_uv=False)
    print(f"    (G, eta) only: sigma_min/sigma_max = {S[1] / S[0]:.3e}")
