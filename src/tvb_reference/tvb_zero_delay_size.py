"""Size of the zero-delay effect at the lab baseline; follow-up description to tvb_zero_delay_check.py (doc/LOG.md, 2026-11-24).
Usage (tower, venv_tvb): python tvb_zero_delay_size.py weights.txt
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

import sys, time
import numpy as np
from tvb.simulator.backend.nb_mpr import NbMPRBackend
from tvb_zero_delay_check import make_sim

rk = lambda x: np.argsort(np.argsort(x))
sp = lambda a, b: float(np.corrcoef(rk(a), rk(b))[0, 1])


def run(W, G, nsig, mode, seed=2, T=10e3):
    sim = make_sim(W, G, nsig, seed=seed)
    ((tt, dd),) = NbMPRBackend().run_sim(sim, simulation_length=T, compatibility_mode=mode)
    r = dd[:, 0, :, 0]
    bad = np.where(~np.isfinite(dd).all((1, 2, 3)))[0]
    return r, (int(bad[0]) if len(bad) else None)


def describe(tag, r, first_bad, lo=10000):
    hi = first_bad - 100 if first_bad is not None else r.shape[0]
    if hi <= lo + 1000:
        print(
            f"  {tag}: non-finite from sample {first_bad} (time {first_bad * 0.1:.0f}); too early to describe"
        )
        return None
    x = r[lo:hi]
    nm = x.mean(0)
    N = x.shape[1]
    print(
        f"  {tag}: first non-finite sample {first_bad}; over samples {lo}-{hi}: mean r {x.mean():.4f}; up-occ {np.mean(x > 0.7):.4f}; max r {x.max():.2f}; "
        f"node mean r first half {nm[:N // 2].mean():.4f}, second half {nm[N // 2:].mean():.4f}; Spearman(node mean r, index) {sp(nm, np.arange(N)):+.3f}",
        flush=True,
    )
    return nm


if __name__ == "__main__":
    W = np.loadtxt(sys.argv[1])
    N = W.shape[0]
    idx = np.arange(N)
    lower = np.tril(np.ones((N, N)), -1)
    Weff = W * (lower + 0.5 * lower.T)  # full weight from lower-numbered, half from higher
    print(
        f"N {N}; mean in-strength {W.sum(1).mean():.4f}; effective (half weight from higher-numbered nodes) {Weff.sum(1).mean():.4f}; "
        f"node 0 keeps {Weff[0].sum() / W[0].sum():.2f}, last node keeps {Weff[-1].sum() / W[-1].sum():.2f} of its input weight"
    )
    p = idx[::-1].copy()
    for G in (1.99, 1.85):
        for nsig in (0.045,):
            print(f"\nG = {G}, nsig = {nsig}", flush=True)
            for seed in (2, 3):
                t0 = time.time()
                nd = describe(f"seed {seed} default, W            ", *run(W, G, nsig, False, seed))
                nc = describe(f"seed {seed} standard Heun, W      ", *run(W, G, nsig, True, seed))
                ne = describe(
                    f"seed {seed} standard Heun, W_eff  ", *run(Weff, G, nsig, True, seed)
                )
                nr = describe(
                    f"seed {seed} default, nodes reversed", *run(W[p][:, p], G, nsig, False, seed)
                )
                if nd is not None and ne is not None:
                    print(
                        f"     Spearman(node mean r: default W, standard W_eff) {sp(nd, ne):+.3f}"
                    )
                if nd is not None and nr is not None:
                    print(
                        f"     Spearman(node mean r: default, default with reversed order, undone) {sp(nd, nr[np.argsort(p)]):+.3f}"
                    )
                if nd is not None and nc is not None:
                    print(f"     Spearman(node mean r: default W, standard W) {sp(nd, nc):+.3f}")
                print(f"     ({time.time() - t0:.0f} s)", flush=True)
