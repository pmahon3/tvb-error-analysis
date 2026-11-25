"""Statistics for ridge check v2 outputs (npz from ridge_v2_sde.py). Per configuration x seed:
finite, mean fraction-up, debounced transitions (per-TR mean up-fraction smoothed over 3 TRs ~ 2.2 s; up on crossing 0.45,
down on crossing 0.20; amended from 0.6 / 0.4), mean up-dwell and down-dwell (s), switching index, VAF2, mean FC, corr(FC, SC) (BOLD after 30 s).
Usage: python3 ridge_v2_analyse.py stageA <npz> [<npz> ...]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from stats import dfc_w, switching_index, vaf2
from repo_paths import data

TR = 0.72
W = np.loadtxt(data("weights.txt")); np.fill_diagonal(W, 0); W = W / W.sum(1).mean()
IU = np.triu_indices(W.shape[0], 1)
NAMES = ["finite", "frac_up", "n_trans", "up_dwell_s", "down_dwell_s", "switch", "vaf2", "meanFC", "corrFC_SC"]


UP_THR, DOWN_THR = 0.45, 0.20      # amended 2026-11-23 (registered 0.6 / 0.4 sat on the upper plateau; see ../doc/LOG.md)


def dwell(fu):
    k = np.ones(3); f = np.convolve(fu, k, mode="same") / np.convolve(np.ones(len(fu)), k, mode="same")
    state = f[0] > 0.5 * (UP_THR + DOWN_THR); t0 = 0; ups, downs = [], []; n = 0
    for i, x in enumerate(f):
        if (not state and x > UP_THR) or (state and x < DOWN_THR):
            (ups if state else downs).append((i - t0) * TR)
            state, t0, n = not state, i, n + 1
    ups, downs = (ups[1:] if state_first_up(f) else ups), (downs if state_first_up(f) else downs[1:])   # first dwell is left-censored
    return n, (np.mean(ups) if ups else np.nan), (np.mean(downs) if downs else np.nan)


def state_first_up(f):
    return f[0] > 0.5 * (UP_THR + DOWN_THR)


def stats_one(bold, fu, finite):
    if not finite.all(): return [0.0] + [np.nan] * (len(NAMES) - 1)
    b = bold[int(30 / TR):].astype(np.float64); n, up, dn = dwell(fu)
    if b.shape[0] < 100: return [1.0, float(fu.mean()), n, up, dn] + [np.nan] * 4      # record too short for BOLD statistics
    FC = np.nan_to_num(np.corrcoef(b.T)); D, Ws = dfc_w(b)
    return [1.0, float(fu.mean()), n, up, dn, switching_index(D, Ws), vaf2(b), float(FC[IU].mean()),
            float(np.corrcoef(FC[IU], W[IU] + W.T[IU])[0, 1])]


def load(path):
    z = np.load(path); C, S = len(z["configs"]), len(z["seeds"])
    bold, fu, fin = z["bold"], z["frac_up"], z["finite"]              # (B, n_tr, N), (B, n_tr), (B, n_tr)
    X = np.array([stats_one(bold[b], fu[b], fin[b]) for b in range(C * S)]).reshape(S, C, len(NAMES))
    return z, X


if __name__ == "__main__":
    mode, paths = sys.argv[1], sys.argv[2:]
    if mode == "stageA":
        for p in paths:
            z, X = load(p); x = X[:, 0, :]
            print(f"dt={float(z['dt'])}: finite {int(np.nansum(x[:, 0]))}/{len(x)} seeds")
            for i, n in enumerate(NAMES[1:], 1):
                v = x[:, i]; ok = np.isfinite(v)
                print(f"   {n:13s} mean {np.nanmean(v):9.4f}  se {np.nanstd(v, ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else np.nan:8.4f}  (n={ok.sum()})  per seed {np.round(v, 3).tolist()}")
