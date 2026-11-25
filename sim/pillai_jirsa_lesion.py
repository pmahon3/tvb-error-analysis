"""Lesion check on Pillai & Jirsa 2017 (Neuron 94:1010), Fig. 4 example, eqs. 34-35 and the two 3-D networks (p.1018-1019).

Network: q' = R(Pq) q + mu * w F(Pq), with P w = I_2 (P = adjoint/readout, w = embedding), xi = P q,
R(xi) = 1 - |xi|^2, F = Excitator: F1 = (xi1 + xi2 - xi1^3/3) tau, F2 = -(xi1 - a - I + b xi2)/tau.
(The paper writes the minus sign of F2 inside eq. 34; here it is folded into F2 so that xi' = R xi + mu F.)

Lesion of node k: delete q_k, its equation, and its contribution to P q.
Hand prediction (registered in ../README.md, 2026-11-23, before running):
  lesioned reduced field = R xi + mu (I - M_k) F(xi),  M_k = p_k w_k^T (column k of P times row k of w),
  on the range of P with column k removed. Sum_k M_k = P w = I is all the reduced flow fixes.
This script checks the identity through the network equations (not the formula) and reports the deviation angle.
"""
import numpy as np

a, I_ext, b, tau, mu = 1.05, 0.0, 0.1, 3.0, 0.1


def F(xi):
    x1, x2 = xi
    return np.array([(x1 + x2 - x1**3 / 3) * tau, -(x1 - a - I_ext + b * x2) / tau])


def R(xi):
    return 1.0 - xi @ xi


def net_rhs(q, P, w):
    xi = P @ q
    return R(xi) * q + mu * w @ F(xi)


nets = {
    "a (paper)": (np.array([[1., 0, 0], [0, 1, 0]]), np.array([[1., 0], [0, 1], [0, 0]])),
    "b (paper)": (np.array([[1., 0, 1], [0, 1, 1]]), np.array([[1., 0], [0, 1], [0, 0]])),
    # same fibre, built here: P as in b, w chosen so that P w = I with w_3 != 0
    "c (ours)": (np.array([[1., 0, 1], [0, 1, 1]]), np.array([[0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])),
}

thetas = np.linspace(0, 2 * np.pi, 400, endpoint=False)
circle = [np.array([np.cos(t), np.sin(t)]) for t in thetas]  # the attracting manifold |xi| = 1


def angle(u, v):
    c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.arccos(np.clip(c, -1, 1)) / np.pi  # 0 = same direction, 1 = opposite


for name, (P, w) in nets.items():
    assert np.allclose(P @ w, np.eye(2)), name
    # intact reduced field through the network equations
    err_intact = 0.0
    for xi in circle:
        q = w @ xi  # a network state with P q = xi
        err_intact = max(err_intact, np.abs(P @ net_rhs(q, P, w) - (R(xi) * xi + mu * F(xi))).max())
    print(f"\nnetwork {name}: P w = I ok; intact reduced field error {err_intact:.1e}")
    for k in range(3):
        keep = [i for i in range(3) if i != k]
        Pk, wk = P[:, keep], w[keep, :]
        Mk = np.outer(P[:, k], w[k, :])
        rank = np.linalg.matrix_rank(Pk)
        errs, das = [], []
        for xi in circle:
            if rank == 2:
                q = np.linalg.solve(Pk, xi)
            else:  # lesioned state space is a line; use states whose image is the projection of xi onto it
                q = np.linalg.lstsq(Pk, xi, rcond=None)[0]
            x = Pk @ q
            field_net = Pk @ (R(x) * q + mu * wk @ F(x))
            field_formula = R(x) * x + mu * (np.eye(2) - Mk) @ F(x)
            errs.append(np.abs(field_net - field_formula).max())
            if rank == 2:
                das.append(angle(field_net, R(xi) * xi + mu * F(xi)))
        da = f"{np.mean(das):.3f}" if das else "n/a (state space collapses to a line)"
        print(f"  lesion q{k+1}: M_k = {np.round(Mk, 2).tolist()}, rank of lesioned readout {rank}, "
              f"identity error {max(errs):.1e}, mean deviation angle on |xi|=1: {da}")
