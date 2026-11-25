"""Ridge check in the (G, eta) slice at (G, sigma) = (1.05, 0.30). Registered in doc/LOG.md (2026-11-23).
Same MPR + Balloon model as mpr_net.py, with eta per configuration and common random numbers within a seed."""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[2] / "core"),
]

import sys, os, json, time

import numpy as np
import mpr_net as M
from stats import dfc_w, switching_index, vaf2

G0, ETA0, SIG, dG, dEta = 1.05, M.ETA, 0.30, 0.03, 0.1
CONFIGS = [(G0, ETA0), (G0 + dG, ETA0), (G0 - dG, ETA0), (G0, ETA0 + dEta), (G0, ETA0 - dEta)]


def run(W, L, seeds, T_ms, dt=0.1, speed=2.0, keep_ms=100.0, tr_ms=720.0):
    C, S, N = len(CONFIGS), len(seeds), W.shape[0]
    B = C * S
    G = np.array([c[0] for c in CONFIGS] * S)[:, None]
    eta = np.array([c[1] for c in CONFIGS] * S)[:, None]
    steps = int(round(T_ms / dt))
    sub = int(round(1.0 / dt))
    keep = int(round(keep_ms))
    tr = int(round(tr_ms))
    Ds = np.rint(L / speed / dt).astype(int)
    Dmax = int(Ds.max()) + 1
    jidx = np.arange(N)[None, :].repeat(N, 0)
    rngs = [np.random.default_rng(1000 * s + 11) for s in seeds]
    r = np.empty((B, N))
    v = np.empty((B, N))
    for si, rg in enumerate(rngs):  # same initial state for the 5 configurations of a seed
        up = rg.random(N) < 0.5
        r[si * C : (si + 1) * C] = np.where(up, M.R_UP, M.R_DN) + 0.01 * rg.standard_normal(N)
        v[si * C : (si + 1) * C] = np.where(up, M.V_UP, M.V_DN)
    buf = np.repeat(r[:, :, None], Dmax, axis=2)
    ptr = 0
    n_samp = steps // sub
    r_out = np.zeros((B, n_samp // keep, N), np.float32)
    bold = np.zeros((B, n_samp // tr, N), np.float32)
    hemo = M.Balloon(B, N)
    sq = np.sqrt(dt)

    def drift(r, v, I):
        return (M.DELTA / (M.PI * M.TAU) + 2 * r * v) / M.TAU, (
            v * v + eta - (M.PI * M.TAU * r) ** 2 + M.J * M.TAU * r + I
        ) / M.TAU

    t0 = time.time()
    for k in range(steps):
        I = G * np.einsum("ij,bij->bi", W, buf[:, jidx, (ptr - Ds) % Dmax])
        dr, dv = drift(r, v, I)
        dr2, dv2 = drift(r + dt * dr, v + dt * dv, I)
        z = np.stack(
            [rg.standard_normal((2, N)) for rg in rngs], 1
        )  # (2, S, N): shared within a seed
        z = np.repeat(z, C, axis=1)
        r = np.maximum(r + 0.5 * dt * (dr + dr2) + SIG * sq * z[0], 0.0)
        v = v + 0.5 * dt * (dv + dv2) + SIG * sq * z[1]
        ptr = (ptr + 1) % Dmax
        buf[:, :, ptr] = r
        if (k + 1) % sub == 0:
            i = (k + 1) // sub
            hemo.step(r, 1e-3)
            if i % keep == 0 and i // keep - 1 < r_out.shape[1]:
                r_out[:, i // keep - 1] = r
            if i % tr == 0 and i // tr - 1 < bold.shape[1]:
                bold[:, i // tr - 1] = hemo.bold()
        if k == 2000:
            print(f"  rate: {(time.time() - t0) / 2000 * steps / 60:.1f} min projected", flush=True)
    return r_out.reshape(S, C, -1, N), bold.reshape(S, C, -1, N)


def transitions(rr, keep_ms=100.0):
    f = (rr > 0.5).mean(1)
    w = int(round(2000 / keep_ms))
    fs = np.convolve(f, np.ones(w) / w, mode="same")
    state, n = fs[0] > 0.5, 0
    for x in fs:
        if not state and x > 0.6:
            state, n = True, n + 1
        elif state and x < 0.4:
            state, n = False, n + 1
    return n, float(f.mean())


def stats(rr, bb, W):
    iu = np.triu_indices(W.shape[0], 1)
    b = bb[int(30 / 0.72) :]  # drop 30 s of BOLD transient
    FC = np.nan_to_num(np.corrcoef(b.T))
    D, Ws = dfc_w(b)
    n, fu = transitions(rr)
    return [
        switching_index(D, Ws),
        vaf2(b),
        float(FC[iu].mean()),
        float(np.corrcoef(FC[iu], W[iu] + W.T[iu])[0, 1]),
        fu,
        n,
    ]


NAMES = ["switch", "vaf2", "meanFC", "corrFC_SC", "frac_up", "n_trans"]


def analyse(X, seeds):
    X = np.asarray(X)
    S = X.shape[0]
    print("per-config means:")
    for c in range(len(CONFIGS)):
        print(
            f"  {CONFIGS[c]}: "
            + ", ".join(f"{n}={X[:, c, i].mean():.4g}" for i, n in enumerate(NAMES))
        )
    print("transitions per seed (centre, G+, G-, eta+, eta-):", X[:, :, 5].astype(int).tolist())
    sd = X[:, 0, :].std(0, ddof=1)
    dG_ = (X[:, 1] - X[:, 2]) / (2 * dG) / sd
    dE = (X[:, 3] - X[:, 4]) / (2 * dEta) / sd

    def cosv(a, b):
        a, b = a.mean(0), b.mean(0)
        return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

    rng = np.random.default_rng(0)
    boot = [cosv(dG_[idx], dE[idx]) for idx in (rng.integers(0, S, S) for _ in range(2000))]
    print(f"centre sd per statistic: {np.round(sd, 4).tolist()}")
    print(
        f"whitened j_G   = {np.round(dG_.mean(0), 2).tolist()}  (se {np.round(dG_.std(0, ddof=1) / np.sqrt(S), 2).tolist()})"
    )
    print(
        f"whitened j_eta = {np.round(dE.mean(0), 2).tolist()}  (se {np.round(dE.std(0, ddof=1) / np.sqrt(S), 2).tolist()})"
    )
    print(
        f"c = |cos| = {cosv(dG_, dE):.3f}; bootstrap 5th/95th pct = {np.percentile(boot, 5):.3f}/{np.percentile(boot, 95):.3f}"
    )


if __name__ == "__main__":
    if sys.argv[1] == "combine":
        parts = [json.load(open(f)) for f in sys.argv[2:]]
        X = np.concatenate([np.array(p["X"]) for p in parts])
        seeds = sum((p["seeds"] for p in parts), [])
        print(f"seeds {seeds}, T = {parts[0]['T_s']} s, dt = {parts[0]['dt']}")
        analyse(X, seeds)
        sys.exit()
    import threading

    t = threading.Timer(4 * 3600, lambda: os._exit(3))
    t.daemon = True
    t.start()  # hard timeout
    T_s = float(sys.argv[1])
    seeds = [int(x) for x in sys.argv[2].split(",")]
    out = sys.argv[3]
    dt = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    W, L = M.load_connectome(os.path.dirname(os.path.abspath(__file__)))
    rr, bb = run(W, L, seeds, T_s * 1000, dt=dt)
    X = np.array(
        [[stats(rr[s, c], bb[s, c], W) for c in range(len(CONFIGS))] for s in range(len(seeds))]
    )
    json.dump(
        dict(configs=CONFIGS, names=NAMES, X=X.tolist(), T_s=T_s, dt=dt, seeds=seeds),
        open(out, "w"),
    )
    print("done", seeds, flush=True)
