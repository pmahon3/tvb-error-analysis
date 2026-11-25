"""Ridge check v2, stochastic stages (A: numerics, B: map). JAX + vbjax node, batched over configurations with vmap.
Same model as mpr_net.py: MPR nodes (vbjax mpr_dfun, default theta except eta), coupling I_i = G sum_j W_ij r_j(t - D_ij),
additive noise sigma*sqrt(dt)*N(0,1) on r and v (Euler), Heun drift, r clamped at 0; Balloon-Windkessel (Friston, same
constants as mpr_net.Balloon) driven by r every 1 ms; BOLD sampled every TR = 720 ms. Delays: tract length / 2 mm per ms.
Common random numbers: configurations sharing a seed share initial state and noise.
Outputs per TR: BOLD (B, n_TR, N) and the mean over the TR of the fraction of nodes with r > 0.5 (B, n_TR).
Usage (inside the vbjax container): python3 ridge_v2_sde.py <job.json> <out.npz>
job.json: {"configs": [[G, eta, sigma], ...], "seeds": [...], "dt": 0.025, "T_s": 600}
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "common"),
]

import json, sys, os, time
import numpy as np
import jax
import jax.numpy as jnp
import vbjax as vb

HERE = os.path.dirname(os.path.abspath(__file__))
from repo_paths import data

TR_MS, SPEED = 720, 2.0
R_DN, V_DN, R_UP, V_UP = 0.081, -1.962, 1.031, -0.154


def load_connectome():
    W = np.loadtxt(data("weights.txt"))
    L = np.loadtxt(data("tract_lengths.txt"))
    np.fill_diagonal(W, 0.0)
    W = W / W.sum(1).mean()
    return W, L


def build(job):
    FT = np.float64 if jax.config.jax_enable_x64 else np.float32
    W, L = load_connectome()
    W = W.astype(FT)
    N = W.shape[0]
    dt = float(job["dt"])
    cfg = np.array(job["configs"], FT)
    seeds = list(job["seeds"])
    C, S = len(cfg), len(seeds)
    B = C * S
    G = jnp.asarray(np.tile(cfg[:, 0], S))[:, None]
    ETA = jnp.asarray(np.tile(cfg[:, 1], S))[:, None]
    SIG = jnp.asarray(np.tile(cfg[:, 2], S))[:, None]
    seed_of = np.repeat(np.arange(S), C)  # batch b -> seed index
    D = np.rint(L / SPEED / dt).astype(np.int32)
    Dmax = int(D.max()) + 1
    steps_per_ms = int(round(1.0 / dt))
    n_tr = int(job["T_s"] * 1000 // TR_MS)
    Wj = jnp.asarray(W)
    Dj = jnp.asarray(D)
    jidx = jnp.arange(N)[None, :].repeat(N, 0)
    p0 = vb.mpr_default_theta

    def drift(r, v, I):
        dr = (p0.Delta / (jnp.pi * p0.tau) + 2 * r * v) / p0.tau
        dv = (
            v**2 + ETA + p0.J * p0.tau * r + I - (jnp.pi**2) * (r**2) * (p0.tau**2)
        ) / p0.tau
        return dr, dv  # = vbjax mpr_dfun with cr = 1, I = 0, eta per row

    # initial state per seed (shared across that seed's configurations)
    rng = np.random.default_rng(12345)
    up = np.stack([np.random.default_rng(1000 * sd + 7).random(N) < 0.5 for sd in seeds])[seed_of]
    r0 = np.where(up, R_UP, R_DN).astype(FT)
    v0 = np.where(up, V_UP, V_DN).astype(FT)
    buf0 = np.repeat(r0[:, :, None], Dmax, axis=2)
    keys = jnp.stack([jax.random.PRNGKey(sd) for sd in seeds])

    k_, g_, t_, a_, E0, V0 = 0.65, 0.41, 0.98, 0.32, 0.4, 0.04
    k1, k2, k3 = 7 * E0, 2.0, 2 * E0 - 0.2

    def one_step(carry, z):
        r, v, buf, ptr = carry  # z: (2, S, N) noise for this step
        past = buf[:, jidx, (ptr - Dj) % Dmax]  # (B, N, N)
        I = G * jnp.einsum("ij,bij->bi", Wj, past)
        dr, dv = drift(r, v, I)
        dr2, dv2 = drift(r + dt * dr, v + dt * dv, I)
        zb = z[:, seed_of, :]
        r = jnp.maximum(r + 0.5 * dt * (dr + dr2) + SIG * jnp.sqrt(dt) * zb[0], 0.0)
        v = v + 0.5 * dt * (dv + dv2) + SIG * jnp.sqrt(dt) * zb[1]
        ptr = (ptr + 1) % Dmax
        buf = buf.at[:, :, ptr].set(r)
        return (r, v, buf, ptr), None

    def one_ms(carry, key_ms):
        (r, v, buf, ptr), (s_, f_, vv, q), fu_acc = carry
        z = jax.vmap(lambda k: jax.random.normal(k, (steps_per_ms, 2, N)))(
            jax.vmap(lambda k: jax.random.fold_in(k, key_ms))(keys)
        )
        z = jnp.transpose(z, (1, 2, 0, 3)).astype(FT)  # (steps, 2, S, N)
        (r, v, buf, ptr), _ = jax.lax.scan(one_step, (r, v, buf, ptr), z)
        ds = r - k_ * s_ - g_ * (f_ - 1)
        df = s_
        dvv = (f_ - vv ** (1 / a_)) / t_
        dq = (f_ * (1 - (1 - E0) ** (1 / f_)) / E0 - q * vv ** (1 / a_ - 1)) / t_
        s_ = s_ + 1e-3 * ds
        f_ = jnp.maximum(
            f_ + 1e-3 * s_, 1e-3
        )  # as mpr_net: its df aliases s, updated in place first
        vv = jnp.maximum(vv + 1e-3 * dvv, 1e-3)
        q = jnp.maximum(q + 1e-3 * dq, 1e-3)
        fu_acc = fu_acc + (r > 0.5).mean(1).astype(FT)
        return ((r, v, buf, ptr), (s_, f_, vv, q), fu_acc), None

    def one_tr(carry, tr_index):
        state, hemo, _ = carry
        ms_ids = tr_index * TR_MS + jnp.arange(TR_MS)
        (state, hemo, fu_acc), _ = jax.lax.scan(one_ms, (state, hemo, jnp.zeros(B, FT)), ms_ids)
        s_, f_, vv, q = hemo
        bold = V0 * (k1 * (1 - q) + k2 * (1 - q / vv) + k3 * (1 - vv))
        finite = jnp.isfinite(state[0]).all(1) & jnp.isfinite(state[1]).all(1)
        return (state, hemo, fu_acc), (bold, fu_acc / TR_MS, finite)

    @jax.jit
    def run():
        state = (jnp.asarray(r0), jnp.asarray(v0), jnp.asarray(buf0), jnp.int32(0))
        hemo = (
            jnp.zeros((B, N), FT),
            jnp.ones((B, N), FT),
            jnp.ones((B, N), FT),
            jnp.ones((B, N), FT),
        )
        (state_end, hemo_end, _), (bold, fu, finite) = jax.lax.scan(
            one_tr, (state, hemo, jnp.zeros(B, FT)), jnp.arange(n_tr)
        )
        return bold, fu, finite, state_end[0], state_end[1], hemo_end

    return run, dict(B=B, C=C, S=S, N=N, n_tr=n_tr, Dmax=Dmax, steps=n_tr * TR_MS * steps_per_ms)


if __name__ == "__main__":
    job = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    print("devices:", jax.devices(), flush=True)
    run, info = build(job)
    print("job:", info, flush=True)
    t0 = time.time()
    bold, fu, finite, *_ = run()
    bold = np.asarray(bold)
    fu = np.asarray(fu)
    finite = np.asarray(finite)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    np.savez_compressed(
        out,
        bold=np.transpose(bold, (1, 0, 2)),
        frac_up=fu.T,
        finite=finite.T,
        configs=np.array(job["configs"]),
        seeds=np.array(job["seeds"]),
        dt=job["dt"],
        T_s=job["T_s"],
    )
