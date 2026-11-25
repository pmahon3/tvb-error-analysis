"""Reversal check for stage A: the centre (G, eta, sigma) = (1.05, -5.0, 0.30) through the NumPy reference simulator
(mpr_net.run_batch, unchanged; float64; its own generator and initial state), independent of ridge_v2_sde.py.
Registered in ../README.md (2026-11-24, "reversal check"). Output has the layout of ridge_v2_sde.py's npz, one seed.
Usage: python -I ridge_v2_numpy_check.py <dt> <seed> <half|down> <out.npz> [T_s]
       python -I ridge_v2_numpy_check.py score <npz> [<npz> ...]
frac_up per TR = mean over the TR of the fraction of nodes with r > 0.5, from r kept every 10 ms (stage A used every ms).
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

G0, SIG, TR_MS, KEEP_MS = 1.05, 0.30, 720, 10


def simulate(dt, seed, start, out, T_s):
    import mpr_net as M
    W, L = M.load_connectome(HERE); N = W.shape[0]
    r0 = v0 = None
    if start == "down":
        r0 = np.full((1, N), M.R_DN); v0 = np.full((1, N), M.V_DN)
    t0 = time.time()
    res = M.run_batch(W, L, [G0], [SIG], seed, T_ms=T_s * 1000, dt=dt, keep_ms=float(KEEP_MS), tr_ms=float(TR_MS),
                      r0=r0, v0=v0, progress=True)
    r, bold = res["r"][0], res["bold"][0]                                     # (n_keep, N), (n_tr, N)
    n_tr = bold.shape[0]; per = TR_MS // KEEP_MS
    rr = r[:n_tr * per].reshape(n_tr, per, N)
    with np.errstate(invalid="ignore"):
        fu = (rr > 0.5).mean(2).mean(1)
    finite = np.isfinite(rr).all((1, 2)) & np.isfinite(bold).all(1)
    np.savez_compressed(out, bold=bold[None], frac_up=fu[None].astype(np.float32), finite=finite[None],
                        configs=np.array([[G0, M.ETA, SIG]]), seeds=np.array([seed]), dt=dt, T_s=T_s, start=start)
    print(f"done in {(time.time() - t0) / 60:.1f} min; finite TRs {int(finite.sum())}/{n_tr}; mean frac_up {np.nanmean(fu):.3f}", flush=True)


def score(paths):
    from ridge_v2_analyse import stats_one, NAMES, TR
    rows = {}
    for p in paths:
        z = np.load(p); key = (float(z["dt"]), str(z["start"]))
        x = stats_one(z["bold"][0], z["frac_up"][0], z["finite"][0])
        late = float(z["frac_up"][0][int(100 / TR):].mean()) if z["finite"][0].all() else float("nan")
        rows.setdefault(key, []).append((int(z["seeds"][0]), x, late))
    for key in sorted(rows, key=lambda k: (-k[0], k[1])):
        print(f"dt={key[0]} start={key[1]}")
        for seed, x, late in sorted(rows[key]):
            d = dict(zip(NAMES, x))
            print(f"   seed {seed}: finite {int(d['finite'])}  frac_up {d['frac_up']:.3f}  (after 100 s: {late:.3f})  transitions {d['n_trans']}"
                  f"  up_dwell {d['up_dwell_s']:.1f}  down_dwell {d['down_dwell_s']:.1f}  VAF2 {d['vaf2']:.3f}  meanFC {d['meanFC']:.3f}  corrFC_SC {d['corrFC_SC']:.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(sys.argv[2:])
    else:
        simulate(float(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 600)
