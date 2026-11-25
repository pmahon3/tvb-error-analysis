"""Ridge check v2, stage 0 (second amendment): quasi-static hysteresis sweep of the undelayed deterministic MPR network.
Registered in ../README.md. Run: .venv_vbjax/bin/python -I ridge_v2_hysteresis.py"""
import os, json
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import vbjax as vb

HERE = os.path.dirname(os.path.abspath(__file__))
W = np.loadtxt(os.path.join(HERE, "weights.txt")); np.fill_diagonal(W, 0.0); W = W / W.sum(1).mean()
Wj = jnp.asarray(W); N = W.shape[0]; s = W.sum(1)
R_DN, V_DN, R_UP, V_UP = 0.081, -1.962, 1.031, -0.154
Gs = jnp.array([0.95, 1.0, 1.05, 1.1, 1.15]); DT, TREL = 0.01, 1000.0; NREL = int(TREL / DT)


def field(rv, G, eta):
    p = vb.mpr_default_theta._replace(eta=eta)
    c = jnp.stack([G * (Wj @ rv[0]), jnp.zeros(N)])
    return vb.mpr_dfun(rv, c, p)


def relax_one(rv, G, eta):
    f = lambda z: field(z, G, eta)
    def step(z, _):
        k1 = f(z); k2 = f(z + 0.5 * DT * k1); k3 = f(z + 0.5 * DT * k2); k4 = f(z + DT * k3)
        z = z + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return vb.mpr_r_positive(z, None), None
    z, _ = jax.lax.scan(step, rv, None, length=NREL)
    return z


relax = jax.jit(jax.vmap(relax_one, in_axes=(0, 0, None)))


def sweep(start, etas, stop):
    z = jnp.stack([start] * len(Gs)); hit = [None] * len(Gs); trace = []
    for eta in etas:
        z = relax(z, Gs, float(eta)); fu = np.asarray((z[:, 0, :] > 0.5).mean(1)); trace.append(fu.tolist())
        if not np.all(np.isfinite(np.asarray(z))): raise SystemExit(f"non-finite at eta={eta}")
        for k in range(len(Gs)):
            if hit[k] is None and stop(fu[k]): hit[k] = float(eta)
        if all(h is not None for h in hit): break
    return hit, trace


low = jnp.array([np.full(N, R_DN), np.full(N, V_DN)])
high = jnp.array([np.where(s > 0, R_UP, R_DN), np.where(s > 0, V_UP, V_DN)])
ign, _ = sweep(low, np.arange(-6.0, -2.99, 0.02), lambda f: f > 0.5)
col, _ = sweep(high, np.arange(-4.0, -7.01, -0.02), lambda f: f < 0.2)
G = np.asarray(Gs)
for k in range(len(G)): print(f"G={G[k]:.2f}: eta_ignite={ign[k]}  eta_collapse={col[k]}")
sl_i = np.polyfit(G, ign, 1)[0]; sl_c = np.polyfit(G, col, 1)[0]
print(f"slopes d eta/d G: ignition {sl_i:+.4f}, collapse {sl_c:+.4f}; ratio {sl_i / sl_c if sl_c else float('inf'):.3f}")
k0 = 2; print(f"S0-1' at G=1.05: collapse {col[k0]} < -5.0 < ignite {ign[k0]}: {col[k0] < -5.0 < ign[k0]}")
json.dump(dict(G=G.tolist(), ignite=ign, collapse=col, slope_ignite=sl_i, slope_collapse=sl_c),
          open(os.path.join(HERE, "runs", "ridge_v2_hysteresis.json"), "w"))
