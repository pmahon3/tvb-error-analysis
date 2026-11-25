"""Zero-delay check of TVB's numba MPR kernel, registered in doc/LOG.md (2026-11-24, 10:20).
Does the default mode of NbMPRBackend, with a connectivity that has no delays (as built in INN-SFU/tvb-node),
compute the corrector-stage coupling from the buffer column being written (updated values for lower-numbered nodes,
pre-filled noise for the others)?  Usage (tower, venv_tvb): python tvb_zero_delay_check.py [weights.txt]
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "common")]

import sys, os, time
import numpy as np
from tvb.simulator.lab import *
from tvb.simulator.backend.nb_mpr import NbMPRBackend

ETA, J, DELTA, TAU = -4.6, 14.5, 0.7, 1.0          # the lab notebooks' node
TEMPLATE = '<%include file="nb-montbrio.py.mako"/>'


def make_sim(W, G, nsigma, dt=0.01, seed=2):
    conn = connectivity.Connectivity(weights=W, tract_lengths=np.ones_like(W), centres=np.zeros(np.shape(W)[0]), speed=np.r_[np.inf])
    # as get_connectivity in the notebooks, except: np.Inf written as np.inf, and the labels are set directly because
    # conn.compute_region_labels() fails under numpy 2 (dtype '128a'); labels do not enter the dynamics
    conn.region_labels = np.array([f'region_{i}' for i in range(np.shape(W)[0])])
    return simulator.Simulator(
        connectivity=conn,
        model=models.MontbrioPazoRoxin(eta=np.r_[ETA], J=np.r_[J], Delta=np.r_[DELTA], tau=np.r_[TAU]),
        coupling=coupling.Linear(a=np.r_[G]),
        integrator=integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=np.r_[nsigma, nsigma * 2], noise_seed=seed)),
        monitors=[monitors.TemporalAverage(period=0.1)]).configure()


class Capture(NbMPRBackend):
    def _run_integrate(self, integrate, sim, N, nstep, svar_bufs, stimulus):
        self.captured = [b.copy() for b in svar_bufs]
        return super()._run_integrate(integrate, sim, N, nstep, svar_bufs, stimulus)


def f(r, V, c):
    return (DELTA / (np.pi * TAU) + 2 * V * r) / TAU, (V * V - np.pi ** 2 * TAU ** 2 * r * r + ETA + J * TAU * r + c) / TAU


def ref_standard(rb, Vb, W, G, dt):
    """Standard Heun: one coupling per step, from the previous step; bound after both stages; same noise in both."""
    r, V = rb.copy(), Vb.copy()
    for i in range(1, r.shape[1]):
        nr, nV = r[:, i].copy(), V[:, i].copy(); r0, V0 = r[:, i - 1], V[:, i - 1]
        c = G * (W @ r0); dr0, dV0 = f(r0, V0, c)
        ri = np.maximum(r0 + dt * dr0 + nr, 0.0); Vi = V0 + dt * dV0 + nV
        dr1, dV1 = f(ri, Vi, c)
        r[:, i] = np.maximum(r0 + dt * (dr0 + dr1) / 2.0 + nr, 0.0); V[:, i] = V0 + dt * (dV0 + dV1) / 2.0 + nV
    return r, V


def ref_reading(rb, Vb, W, G, dt):
    """The reading under test: corrector coupling of node n = G * (sum_{j<n} w[n,j] * new r_j + sum_{j>=n} w[n,j] * noise_j)."""
    r, V = rb.copy(), Vb.copy(); N = r.shape[0]
    for i in range(1, r.shape[1]):
        noise_r = r[:, i].copy(); noise_V = V[:, i].copy(); new_r = np.zeros(N)
        for n in range(N):
            r0, V0 = r[n, i - 1], V[n, i - 1]
            c0 = G * (W[n] @ r[:, i - 1]); dr0, dV0 = f(r0, V0, c0)
            ri = max(r0 + dt * dr0 + noise_r[n], 0.0); Vi = V0 + dt * dV0 + noise_V[n]
            c1 = G * (W[n, :n] @ new_r[:n] + W[n, n:] @ noise_r[n:])
            dr1, dV1 = f(ri, Vi, c1)
            new_r[n] = max(r0 + dt * (dr0 + dr1) / 2.0 + noise_r[n], 0.0); V[n, i] = V0 + dt * (dV0 + dV1) / 2.0 + noise_V[n]
        r[:, i] = new_r
    return r, V


def small_network_checks():
    rng = np.random.default_rng(0); N = 6
    W = rng.random((N, N)); W = (W + W.T) / 2; np.fill_diagonal(W, 0.0)
    G, nsig, dt, nstep = 1.5, 0.045, 0.01, 2000
    out = {}
    for mode in (False, True):
        sim = make_sim(W, G, nsig, dt); b = Capture()
        if not mode: print("P1: idelays unique", np.unique(sim.connectivity.idelays).tolist(), "| horizon", int(sim.connectivity.horizon), flush=True)
        res = b._run_sim_plain(sim, nstep=nstep, compatibility_mode=mode, print_source=False)
        out[mode] = (res, b.captured, sim)
    (rd, Vd), cap_d, sim = out[False]; (rc, Vc), cap_c, _ = out[True]
    print("captured buffers identical in the two modes:", bool(np.array_equal(cap_d[0], cap_c[0]) and np.array_equal(cap_d[1], cap_c[1])),
          "| buffer shape", cap_d[0].shape, "| sd of pre-filled r noise", round(float(cap_d[0][:, 1:].std()), 5))
    rs, Vs = ref_standard(cap_d[0], cap_d[1], W, G, dt); rr, Vr = ref_reading(cap_d[0], cap_d[1], W, G, dt)
    h = int(sim.connectivity.horizon)
    d = lambda a, b_: float(np.max(np.abs(a - b_)))
    print(f"P2: max|default - standard Heun| = {d(rd, rs[:, h:]):.3e} (r), {d(Vd, Vs[:, h:]):.3e} (V)")
    print(f"P2: max|default - reading|       = {d(rd, rr[:, h:]):.3e} (r), {d(Vd, Vr[:, h:]):.3e} (V)")
    print(f"P3: max|compat  - standard Heun| = {d(rc, rs[:, h:]):.3e} (r), {d(Vc, Vs[:, h:]):.3e} (V)")
    for k in (1, 10, 100):
        print(f"    after {k} steps: max|default - standard| = {d(rd[:, :k], rs[:, h:h + k]):.3e}; max|default - reading| = {d(rd[:, :k], rr[:, h:h + k]):.3e}")

    # P4: node order at sigma = 0, calling the generated kernel directly on the same buffers, relabelled
    sim0 = make_sim(W, G, 0.0, dt); b0 = NbMPRBackend(); p = np.arange(N)[::-1].copy()
    r_init = cap_d[0][:, :1]; V_init = cap_d[1][:, :1]; nst = 500
    for mode in (False, True):
        integrate = b0.build_py_func(TEMPLATE, dict(sim=sim0, compatibility_mode=mode), name="integrate", print_source=False)
        def go(Wm, r0, V0):
            rb = np.concatenate([r0, np.zeros((N, nst))], 1); Vb = np.concatenate([V0, np.zeros((N, nst))], 1)
            r_, V_ = integrate(N, dt, nst, 1, rb, Vb, np.ascontiguousarray(Wm), np.zeros((N, N), np.int32),
                               sim0.model.spatial_parameter_matrix.T, None)
            return r_, V_
        ra, _ = go(W, r_init, V_init); rp, _ = go(W[p][:, p], r_init[p], V_init[p])
        print(f"P4 ({'compat' if mode else 'default'} mode, sigma 0): max|original - relabelled, undone| = {d(ra, rp[np.argsort(p)]):.3e}  (max r {ra.max():.3f})")


def size_at_baseline(wfile):
    W = np.loadtxt(wfile); N = W.shape[0]; G, nsig = 1.99, 0.045
    print(f"\nsize at the lab baseline: N = {N}, G = {G}, nsig = {nsig}, 10e3 time units, noise_seed 2", flush=True)
    res = {}
    for mode in (False, True):
        t0 = time.time(); sim = make_sim(W, G, nsig)
        (tt, dd), = NbMPRBackend().run_sim(sim, simulation_length=10e3, compatibility_mode=mode)
        r = dd[10000:, 0, :, 0]; res[mode] = r
        print(f"  {'compat ' if mode else 'default'}: finite {bool(np.isfinite(dd).all())}; mean r {r.mean():.4f}; up-occupancy (r > 0.7) {np.mean(r > 0.7):.4f}; "
              f"median node mean r {np.median(r.mean(0)):.4f}; wall {time.time() - t0:.0f} s", flush=True)
    dn = res[False].mean(0) - res[True].mean(0); idx = np.arange(N)
    rk = lambda x: np.argsort(np.argsort(x))
    print(f"  per-node mean rate, default - compat: mean {dn.mean():+.4f}, min {dn.min():+.4f}, max {dn.max():+.4f}; "
          f"Spearman with node index {np.corrcoef(rk(dn), idx)[0, 1]:+.3f}; mean over first half of nodes {dn[:N // 2].mean():+.4f}, second half {dn[N // 2:].mean():+.4f}")
    upper = np.triu(W, 1).sum(1); print(f"  Spearman of (default - compat) with each node's input weight from higher-numbered nodes: {np.corrcoef(rk(dn), rk(upper))[0, 1]:+.3f}")


if __name__ == "__main__":
    print("tvb-library", __import__("importlib.metadata").metadata.version("tvb-library"), "| numpy", np.__version__, flush=True)
    small_network_checks()
    if len(sys.argv) > 1: size_at_baseline(sys.argv[1])
