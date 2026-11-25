"""Batched MPR brain-network model (Fousek et al. 2024 / Rabuffo et al. 2021 setup), numpy only.

State per node: firing rate r, membrane potential v (tau_c = 1 time unit = 1 ms by the authors' convention).
    tau r' = Delta/(pi tau) + 2 r v
    tau v' = v^2 + eta - (pi tau r)^2 + J tau r + I,     I_i = G sum_j W_ij r_j(t - D_ij)
Additive Gaussian noise of standard deviation sigma (per sqrt(time unit)) on r and v; Heun drift, Euler noise.
The paper's 'noise variance sigma' is not mapped exactly: sigma is swept.
A batch dimension carries (G, sigma) combinations through one time loop (one seed per batch).
BOLD: Balloon-Windkessel (Friston 2000/2003 constants) driven by r, integrated inline at 1 ms, sampled at TR.
Memory: O(B * N * max_delay) for the delay ring buffer plus the kept outputs.
"""
import numpy as np

ETA, J, DELTA, TAU = -5.0, 15.0, 1.0, 1.0
PI = np.pi
R_DN, V_DN, R_UP, V_UP = 0.081, -1.962, 1.031, -0.154   # isolated-node fixed points at I = 0

def load_connectome(path):
    W = np.loadtxt(f"{path}/weights.txt"); L = np.loadtxt(f"{path}/tract_lengths.txt")
    np.fill_diagonal(W, 0.0); W = W / W.sum(1).mean()   # unit mean in-strength
    return W, L

def within_row_permute(W, rng):
    """In-strength preserved exactly: permute the weights within each row (row i = inputs to node i)."""
    Wp = W.copy(); N = W.shape[0]
    for i in range(N):
        idx = np.array([j for j in range(N) if j != i]); Wp[i, idx] = W[i, rng.permutation(idx)]
    return Wp

def drift(r, v, I):
    dr = (DELTA / (PI * TAU) + 2 * r * v) / TAU
    dv = (v * v + ETA - (PI * TAU * r) ** 2 + J * TAU * r + I) / TAU
    return dr, dv

class Balloon:
    """Friston 2000/2003 hemodynamics, one state set per (batch, node); input = z-scored-on-the-fly activity."""
    def __init__(self, B, N, kappa=0.65, gamma=0.41, tau=0.98, alpha=0.32, E0=0.4, V0=0.04):
        self.s = np.zeros((B, N)); self.f = np.ones((B, N)); self.v = np.ones((B, N)); self.q = np.ones((B, N))
        self.k = (kappa, gamma, tau, alpha, E0, V0); self.k1, self.k2, self.k3 = 7 * E0, 2.0, 2 * E0 - 0.2
    def step(self, u, dt_s):
        kappa, gamma, tau, alpha, E0, V0 = self.k
        ds = u - kappa * self.s - gamma * (self.f - 1); df = self.s
        dv = (self.f - self.v ** (1 / alpha)) / tau
        dq = (self.f * (1 - (1 - E0) ** (1 / self.f)) / E0 - self.q * self.v ** (1 / alpha - 1)) / tau
        self.s += dt_s * ds; self.f = np.maximum(self.f + dt_s * df, 1e-3)
        self.v = np.maximum(self.v + dt_s * dv, 1e-3); self.q = np.maximum(self.q + dt_s * dq, 1e-3)
    def bold(self):
        V0 = self.k[5]
        return V0 * (self.k1 * (1 - self.q) + self.k2 * (1 - self.q / self.v) + self.k3 * (1 - self.v))

def run_batch(W, L, G, sigma, seed, T_ms=300_000, dt=0.1, speed_mm_per_ms=2.0, sample_ms=1.0, keep_ms=10.0,
              tr_ms=720.0, u_scale=None, r0=None, v0=None, progress=False):
    """G, sigma: arrays of length B. Returns dict: r (B, T/keep_ms, N) float32, bold (B, T/tr, N), end states."""
    G = np.asarray(G, float); sigma = np.asarray(sigma, float); B = len(G); N = W.shape[0]
    steps = int(round(T_ms / dt)); sub = int(round(sample_ms / dt)); keep = int(round(keep_ms / sample_ms))
    tr = int(round(tr_ms / sample_ms))
    Dsteps = np.rint(L / speed_mm_per_ms / dt).astype(int); Dmax = int(Dsteps.max()) + 1
    rng = np.random.default_rng(int(seed) * 1000 + 7)
    if r0 is None:
        up = rng.random((B, N)) < 0.5
        r = np.where(up, R_UP, R_DN) + 0.01 * rng.standard_normal((B, N)); v = np.where(up, V_UP, V_DN)
    else:
        r, v = r0.copy(), v0.copy()
    buf = np.repeat(r[:, :, None], Dmax, axis=2); ptr = 0
    jidx = np.arange(N)[None, :].repeat(N, 0)
    n_samp = steps // sub; n_keep = n_samp // keep; n_bold = n_samp // tr
    r_out = np.zeros((B, n_keep, N), np.float32); bold = np.zeros((B, n_bold, N), np.float32)
    hemo = Balloon(B, N); sqdt = np.sqrt(dt); Gb = G[:, None]; sb = sigma[:, None]
    # hemodynamic drive: the firing rate itself (nonnegative, O(1)), as in the paper; a centred drive sends the
    # Balloon flow to its floor and produces ~100 s drifts (found 2026-11-17)
    u_scale = u_scale or 1.0; r_mid = 0.0
    for k in range(steps):
        past = buf[:, jidx, (ptr - Dsteps) % Dmax]
        I = Gb * np.einsum('ij,bij->bi', W, past)
        dr, dv = drift(r, v, I)
        rp, vp = r + dt * dr, v + dt * dv
        dr2, dv2 = drift(rp, vp, I)
        noise = sb * sqdt * rng.standard_normal((2, B, N))
        r = np.maximum(r + 0.5 * dt * (dr + dr2) + noise[0], 0.0)
        v = v + 0.5 * dt * (dv + dv2) + noise[1]
        ptr = (ptr + 1) % Dmax; buf[:, :, ptr] = r
        if (k + 1) % sub == 0:
            i = (k + 1) // sub
            hemo.step((r - r_mid) / u_scale, sample_ms / 1000.0)
            if i % keep == 0 and i // keep - 1 < n_keep: r_out[:, i // keep - 1, :] = r
            if i % tr == 0 and i // tr - 1 < n_bold: bold[:, i // tr - 1, :] = hemo.bold()
            if progress and i % (n_samp // 10 or 1) == 0: print(f"  {100 * i // n_samp}%", flush=True)
    return dict(r=r_out, bold=bold, r_end=r, v_end=v, keep_ms=keep_ms, tr_ms=tr_ms)
