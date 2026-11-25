"""Pillai & Jirsa 2017 Fig. 5 analogue: lesion robustness vs network size and vs participation (registered in doc/LOG.md).
Lesioned reduced field on |xi| = 1: mu (I - M_S) F, M_S = sum_{k in S} w_k w_k^T, w orthonormal N x 2, P = w^T.
(Identity checked through the network equations in pillai_jirsa_lesion.py.)"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "common"),
]

import numpy as np

rng = np.random.default_rng(20261123)
a, I_ext, b, tau = 1.05, 0.0, 0.1, 3.0
th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
X = np.stack([np.cos(th), np.sin(th)])  # 2 x 200 points on the manifold
F = np.stack([(X[0] + X[1] - X[0] ** 3 / 3) * tau, -(X[0] - a - I_ext + b * X[1]) / tau])


def da(MS):
    G = F - MS @ F
    c = (G * F).sum(0) / (np.linalg.norm(G, axis=0) * np.linalg.norm(F, axis=0) + 1e-300)
    return np.arccos(np.clip(c, -1, 1)).mean() / np.pi


def embed_paper(N):
    w, _ = np.linalg.qr(rng.uniform(0, 1, (N, 2)))
    return w


def embed_local(N, m):
    w = np.zeros((N, 2))
    idx = rng.choice(N, m, replace=False)
    w[idx], _ = np.linalg.qr(rng.uniform(0, 1, (m, 2)))
    return w


def mean_da(w, f, nsets=100):
    N = w.shape[0]
    n = int(round(f * N))
    vals, collapsed = [], 0
    for _ in range(nsets):
        S = rng.choice(N, n, replace=False)
        keep = np.setdiff1d(np.arange(N), S)
        if np.linalg.matrix_rank(w[keep]) < 2:
            collapsed += 1
            continue
        vals.append(da(w[S].T @ w[S]))
    return (np.mean(vals) if vals else np.nan), collapsed


def slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


Ns = [50, 100, 200, 400, 800, 1600, 3200]
print("P1: paper-style (delocalised) embeddings, 5 networks per N, 100 lesion sets each")
for f in [0.05, 0.15, 0.25]:
    ds = [np.mean([mean_da(embed_paper(N), f)[0] for _ in range(5)]) for N in Ns]
    ratios = [ds[i + 1] / ds[i] for i in range(len(ds) - 1)]
    print(
        f"  f={f:.2f}: DA {np.round(ds, 4).tolist()} | slope {slope(Ns, ds):+.3f} | ratio per doubling {np.mean(ratios):.3f}"
    )

print("P2a: localised on m = 10 nodes, DA vs N (collapsed sets excluded and counted)")
for f in [0.05, 0.15, 0.25]:
    res = [[mean_da(embed_local(N, 10), f) for _ in range(5)] for N in Ns]
    ds = [np.nanmean([r[0] for r in rr]) for rr in res]
    col = [sum(r[1] for r in rr) for rr in res]
    print(
        f"  f={f:.2f}: DA {np.round(ds, 4).tolist()} | slope {slope(Ns, ds):+.3f} | collapsed sets (of 500) {col}"
    )

print("P2b: N = 1600, DA vs m")
ms = [10, 20, 40, 80, 160, 320, 640, 1600]
for f in [0.05, 0.15, 0.25]:
    ds = [np.nanmean([mean_da(embed_local(1600, m), f)[0] for _ in range(5)]) for m in ms]
    print(f"  f={f:.2f}: DA {np.round(ds, 4).tolist()} | slope {slope(ms, ds):+.3f}")
