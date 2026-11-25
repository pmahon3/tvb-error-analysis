"""M1: exact match of the JAX step against TVB's kernel on TVB's own buffers (doc/LOG.md, 2026-11-24, "JAX port").
Usage: JAX_ENABLE_X64=1 .venv_vbjax/bin/python -I mprjax_match.py ../results/runs/port/tvb_ref_dump.npz
"""
import sys, os
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import mprjax

d = np.load(sys.argv[1]); assert jax.config.jax_enable_x64
W, L, dt = d["weights"], d["tract_lengths"], float(d["dt"]); D_tvb = d["idelays_dt0.01"]; Htvb = int(d["horizon_dt0.01"])
rb, Vb, r_ref, V_ref = d["r_buf"], d["V_buf"], d["r_out"], d["V_out"]; N = W.shape[0]; nstep = r_ref.shape[1]
D = mprjax.delays(L, float(d["speed"]), dt); print("delays equal TVB's idelays:", bool(np.array_equal(D, D_tvb)))
P = {k: np.full((1, 1), float(d[k])) for k in ("G", "eta", "J", "Delta")}; S = 10; assert nstep % S == 0
sample, gather, H = mprjax.make_sample(W, D, dt, P, S)
state = mprjax.start_state(rb[:, :Htvb].T[:, :, None], Vb[:, Htvb - 1][:, None], gather, H)       # history (Htvb, N, 1)
nz = lambda x: jnp.asarray(x[:, Htvb:].T.reshape(nstep // S, S, N)[..., None])                  # (chunks, S, N, 1), scaled by TVB
noise = (nz(rb), nz(Vb))


@jax.jit
def run(state, noise):
    return jax.lax.scan(sample, state, noise)


_, (r, V) = run(state, noise); r = np.asarray(r).reshape(nstep, N).T; V = np.asarray(V).reshape(nstep, N).T
er, eV = np.abs(r - r_ref).max(0), np.abs(V - V_ref).max(0)
print(f"steps {nstep} in chunks of {S}; N {N}; ring H {H} (TVB horizon {Htvb}); max r in reference {r_ref.max():.3f}; share of r exactly 0: TVB {np.mean(r_ref == 0):.4f}, JAX {np.mean(r == 0):.4f}")
for k in (1, 10, 100, 1000, nstep):
    print(f"  first {k:5d} steps: max |r_JAX - r_TVB| = {er[:k].max():.3e}; max |V_JAX - V_TVB| = {eV[:k].max():.3e}")
print("M1:", "PASS" if max(er.max(), eV.max()) <= 1e-8 else "FAIL", "(criterion: <= 1e-8 over all steps)")
