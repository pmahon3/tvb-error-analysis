"""What a node-level record fixes about the readout P in Pillai-Jirsa-type networks (registered in doc/LOG.md, 2026-11-23).
q' = R(Pq) q + mu w F(Pq) (+ sigma dW), P = w^+ + Y U_perp^T, P w = I."""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[2] / "core"),
]

import numpy as np
from scipy.optimize import least_squares

rng = np.random.default_rng(1123)
a, I_ext, b, tau, mu, N, dt = 1.05, 0.0, 0.1, 3.0, 0.1, 4, 0.01
w, _ = np.linalg.qr(rng.uniform(0, 1, (N, 2)))
wp = np.linalg.pinv(w)
U = np.linalg.svd(np.eye(N) - w @ wp)[0][:, : N - 2]  # basis of range(w)^perp
Y_true = rng.normal(0, 1, (2, N - 2))


def P_of(Y):
    return wp + Y @ U.T


def F(xi):  # xi: (..., 2)
    x1, x2 = xi[..., 0], xi[..., 1]
    return np.stack([(x1 + x2 - x1**3 / 3) * tau, -(x1 - a - I_ext + b * x2) / tau], -1)


def drift(q, P):
    xi = q @ P.T
    R = 1 - (xi**2).sum(-1, keepdims=True)
    return R * q + mu * F(xi) @ w.T


def simulate(P, q0, T, sigma, seed):
    r = np.random.default_rng(seed)
    n = int(T / dt)
    q = np.empty((n + 1, N))
    q[0] = q0
    for i in range(n):
        q[i + 1] = q[i] + drift(q[i], P) * dt + sigma * np.sqrt(dt) * r.normal(size=N)
    return q


# Q1/Q2: noiseless, start both on the attractor point w xi0 after burn-in of the reduced flow
P0, P1 = P_of(np.zeros_like(Y_true)), P_of(Y_true)
q_burn = simulate(P0, w @ np.array([1.0, 0.0]), 300, 0.0, 0)[-1]
qa, qb = simulate(P0, q_burn, 200, 0.0, 0), simulate(P1, q_burn, 200, 0.0, 0)
print(
    f"Q1 noiseless attractor, node-level max |qa - qb| / max|q| = {np.abs(qa - qb).max() / np.abs(qa).max():.2e}"
)
th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
Xi = np.stack([np.cos(th), np.sin(th)], -1)
Fx = F(Xi)
for k in range(N):
    das = []
    for P in (P0, P1):
        Mk = np.outer(P[:, k], w[k])
        G = Fx - Fx @ Mk.T
        c = (G * Fx).sum(-1) / (np.linalg.norm(G, axis=-1) * np.linalg.norm(Fx, axis=-1))
        das.append(np.arccos(np.clip(c, -1, 1)).mean() / np.pi)
    print(f"Q2 lesion node {k+1}: DA with Y=0 {das[0]:.3f}, with Y_true {das[1]:.3f}")


# Q3/Q4: estimate Y from noisy node-level records by least squares on increments
def fit(q):
    dq = np.diff(q, axis=0) / dt

    def res(y):
        return (dq - drift(q[:-1], P_of(y.reshape(Y_true.shape)))).ravel()

    return least_squares(res, np.zeros(Y_true.size), method="lm").x.reshape(Y_true.shape)


print(f"Y_true = {np.round(Y_true, 3).tolist()}")
for sigma in [0.02, 0.05, 0.1]:
    for T in [250, 1000]:
        est = []
        for rep in range(8):
            q = simulate(P1, q_burn, T + 100, sigma, 100 * rep + int(1000 * sigma) + T)[
                int(100 / dt) :
            ]
            est.append(fit(q))
        est = np.array(est)
        sd = est.std(0, ddof=1).mean()
        bias = np.abs(est.mean(0) - Y_true).mean()
        print(
            f"Q3/Q4 sigma={sigma:.2f} T={T}: mean sd(Y_hat) {sd:.4f} | sd*sqrt(T) {sd*np.sqrt(T):.3f} | mean |bias| {bias:.4f}"
        )
