"""Micro-benchmark 2: is the history buffer copied at every step, and does chunking avoid it? Timing only.
All weighted delays are >= 24 steps at dt 0.01, so within a chunk of S = 10 steps every delayed read comes from columns
written before the chunk: the buffer is loop-invariant inside the chunk and is updated once per chunk with a slab.
Usage: python3 mprjax_history_copy_bench.py <B1> [B2 ...]
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

import sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import jax, jax.numpy as jnp
from mprjax import delays

conn = np.load(__import__("repo_paths").data("hcp001_conn.npz"))
W = conn["weights"].astype(np.float32)
L = conn["tract_lengths"]
N = W.shape[0]
dt = 0.01
D = delays(L, 20.0, dt)
S = 10
FT = np.float32
NCHUNK = 300
assert D[W != 0].min() >= S
H = int(np.ceil((int(D.max()) + 1) / S) * S)  # ring length, a multiple of the chunk
Dj = jnp.asarray(D)
Wj = jnp.asarray(W)
jidx = jnp.asarray(np.tile(np.arange(N), (N, 1)), jnp.int32)
tgt, src = np.nonzero(W)
wnz = jnp.asarray(W[tgt, src])
Dnz = jnp.asarray(D[tgt, src])
srcj = jnp.asarray(src, jnp.int32)
tgtj = jnp.asarray(tgt, jnp.int32)
print(
    "device",
    jax.devices()[0],
    "| jax",
    jax.__version__,
    "| N",
    N,
    "| H",
    H,
    "| nonzero links",
    len(wnz),
    flush=True,
)
G = FT(0.54)


def f(r, V, c):
    return 0.31830987 + 2 * V * r, V * V - 9.8696044 * r * r - 5.0 + 15.0 * r + G * c


def heun(r, V, c0, c1):
    dr0, dV0 = f(r, V, c0)
    ri = jnp.maximum(r + dt * dr0, 0)
    Vi = V + dt * dV0
    dr1, dV1 = f(ri, Vi, c1)
    return jnp.maximum(r + dt * (dr0 + dr1) / 2, 0), V + dt * (dV0 + dV1) / 2


def g_dense(buf, i):
    return jnp.einsum("nj,njb->nb", Wj, buf[(i - Dj) % H, jidx, :])


def g_nz(buf, i):
    B = buf.shape[-1]
    rows = jnp.take(buf.reshape(H * N, B), ((i - Dnz) % H) * N + srcj, axis=0, mode="clip")
    return jax.ops.segment_sum(rows * wnz[:, None], tgtj, num_segments=N, indices_are_sorted=True)


def per_step(g, how):
    def step(c, _):
        r, V, c0, buf, i = c
        c1 = g(buf, i)
        rn, Vn = heun(r, V, c0, c1)
        if how == "scatter":
            buf = buf.at[i % H, :, :].set(rn)
        else:
            buf = jax.lax.dynamic_update_slice(buf, rn[None], (i % H, 0, 0))
        return (rn, Vn, c1, buf, i + 1), None

    @jax.jit
    def run(state):
        state, _ = jax.lax.scan(step, state, None, length=NCHUNK * S)
        return state

    return run


def chunked(g):
    def chunk(c, _):
        r, V, c0, buf, i = c

        def step(cc, k):
            r, V, c0 = cc
            c1 = g(buf, i + k)
            rn, Vn = heun(r, V, c0, c1)
            return (rn, Vn, c1), rn

        (r, V, c0), rs = jax.lax.scan(step, (r, V, c0), jnp.arange(S))
        buf = jax.lax.dynamic_update_slice(
            buf, rs, (i % H, 0, 0)
        )  # i is a multiple of S, H too: no wrap
        return (r, V, c0, buf, i + S), None

    @jax.jit
    def run(state):
        state, _ = jax.lax.scan(chunk, state, None, length=NCHUNK)
        return state

    return run


for B in [int(x) for x in sys.argv[1:]]:
    rng = np.random.default_rng(0)
    hist = rng.uniform(0, 2, (H, N, B)).astype(FT)
    V0 = rng.uniform(-2, 1.5, (N, B)).astype(FT)
    ref = None
    for name, mk, g in (
        ("per step, scatter, dense", lambda g: per_step(g, "scatter"), g_dense),
        ("per step, update_slice, dense", lambda g: per_step(g, "dus"), g_dense),
        ("chunk of 10, dense", chunked, g_dense),
        ("chunk of 10, nonzero links", chunked, g_nz),
    ):
        try:
            buf = jnp.asarray(hist)
            state = (buf[H - 1], jnp.asarray(V0), g(buf, jnp.int32(H - 1)), buf, jnp.int32(H))
            run = mk(g)
            out = run(state)
            jax.block_until_ready(out[0])
            t0 = time.time()
            out = run(out)
            jax.block_until_ready(out[0])
            t = time.time() - t0
            r = np.asarray(out[0])
            if ref is None:
                ref = r
            print(
                f"B {B:5d} | {name:30s} | {t / (NCHUNK * S) * 1e6:9.1f} us/step | {t / (NCHUNK * S) * 1e6 / B:7.4f} us/step/run | max|diff from first| {np.abs(r - ref).max():.2e}",
                flush=True,
            )
        except Exception as e:
            print(f"B {B:5d} | {name:30s} | FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
