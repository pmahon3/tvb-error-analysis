"""Micro-benchmark: ways to compute the delayed coupling G * sum_j W[n, j] r_j(i - D[n, j]) for a batch, on the device
in use. Timing only (doc/LOG.md, 2026-11-24, "JAX port"). Each variant runs the full Heun step of mprjax with zero noise.
Usage: python3 mprjax_gather_bench.py <B1> [B2 ...]
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "common"),
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
H = int(D.max()) + 1
FT = np.float32
NSTEP = 3000
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


def f(r, V, c, G):
    return 0.31830987 + 2 * V * r, V * V - 9.8696044 * r * r - 5.0 + 15.0 * r + G * c


def variants(B):
    v = {}

    # A: batch first, buffer (B, H, N), advanced indexing (the current mprjax layout)
    def gA(buf, i):
        return jnp.einsum("nj,bnj->bn", Wj, buf[:, (i - Dj) % H, jidx])

    v["A (B,H,N) index"] = (gA, "bfirst")

    # B: batch last, buffer (H, N, B): each gathered element is a contiguous run of B values
    def gB(buf, i):
        return jnp.einsum("nj,njb->nb", Wj, buf[(i - Dj) % H, jidx, :])

    v["B (H,N,B) index"] = (gB, "blast")

    # C: batch last, flat take of rows from (H*N, B)
    def gC(buf, i):
        rows = jnp.take(
            buf.reshape(H * N, B), (((i - Dj) % H) * N + jidx).reshape(-1), axis=0, mode="clip"
        )
        return jnp.einsum("nj,njb->nb", Wj, rows.reshape(N, N, B))

    v["C (H*N,B) take"] = (gC, "blast")
    # D: as C, nonzero links only, then a dense (N, nnz) matrix product
    S = np.zeros((N, len(tgt)), np.float32)
    S[tgt, np.arange(len(tgt))] = W[tgt, src]
    Sj = jnp.asarray(S)

    def gD(buf, i):
        rows = jnp.take(buf.reshape(H * N, B), ((i - Dnz) % H) * N + srcj, axis=0, mode="clip")
        return Sj @ rows

    v["D nonzero take + matmul"] = (gD, "blast")

    # E: as D with segment_sum instead of the matrix product
    def gE(buf, i):
        rows = jnp.take(buf.reshape(H * N, B), ((i - Dnz) % H) * N + srcj, axis=0, mode="clip")
        return jax.ops.segment_sum(
            rows * wnz[:, None], tgtj, num_segments=N, indices_are_sorted=True
        )

    v["E nonzero take + segment_sum"] = (gE, "blast")
    return v


def make_run(g, layout, B):
    G = FT(0.54)

    def step(c, _):
        r, V, c0, buf, i = c
        dr0, dV0 = f(r, V, c0, G)
        ri = jnp.maximum(r + dt * dr0, 0)
        Vi = V + dt * dV0
        c1 = g(buf, i)
        dr1, dV1 = f(ri, Vi, c1, G)
        rn = jnp.maximum(r + dt * (dr0 + dr1) / 2, 0)
        Vn = V + dt * (dV0 + dV1) / 2
        buf = buf.at[:, i % H, :].set(rn) if layout == "bfirst" else buf.at[i % H, :, :].set(rn)
        return (rn, Vn, c1, buf, i + 1), None

    @jax.jit
    def run(state):
        state, _ = jax.lax.scan(step, state, None, length=NSTEP)
        return state

    rng = np.random.default_rng(0)
    hist = rng.uniform(0, 2, (B, H, N)).astype(FT)
    V0 = rng.uniform(-2, 1.5, (B, N)).astype(FT)
    if layout == "bfirst":
        buf = jnp.asarray(hist)
        r = buf[:, H - 1, :]
        V = jnp.asarray(V0)
    else:
        buf = jnp.asarray(np.transpose(hist, (1, 2, 0)))
        r = buf[H - 1]
        V = jnp.asarray(V0.T)
    return run, (r, V, g(buf, jnp.int32(H - 1)), buf, jnp.int32(H))


for B in [int(x) for x in sys.argv[1:]]:
    ref = None
    for name, (g, layout) in variants(B).items():
        try:
            run, state = make_run(g, layout, B)
            out = run(state)
            jax.block_until_ready(out[0])  # compile + first pass
            t0 = time.time()
            out = run(out)
            jax.block_until_ready(out[0])
            t = time.time() - t0
            r = np.asarray(out[0])
            r = r if layout == "bfirst" else r.T
            if ref is None:
                ref = r
            print(
                f"B {B:5d} | {name:30s} | {t / NSTEP * 1e6:9.1f} us/step | {t / NSTEP * 1e6 / B:7.4f} us/step/run | max|diff from A| {np.abs(r - ref).max():.2e}",
                flush=True,
            )
        except Exception as e:
            print(f"B {B:5d} | {name:30s} | FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
