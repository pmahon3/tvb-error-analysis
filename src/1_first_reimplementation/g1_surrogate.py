"""Gate G1: attainment of the selection statistics by a stationary surrogate of the model's own BOLD.
python3 -I g1_surrogate.py results/runs/g0_s1.npz --cfg 3 [--K 20] [--seed 0]
For configuration cfg: K cross-spectrum-preserving phase-randomised surrogates (one random phase per
frequency, shared by all regions -> same FC, same autocovariances, Gaussianised temporal structure).
Reports phi_S, phi_V (BOLD), RSS event statistics for model vs surrogates, and phi_KS(surrogate, model).
Also the channel-wise (cross-spectrum NOT preserved) surrogate as the destructive control.
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "common")]

import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as S

ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--cfg", type=int, required=True)
ap.add_argument("--K", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--burn_s", type=float, default=20.0)
a = ap.parse_args()
z = np.load(a.npz, allow_pickle=True); tr = 0.72
bold = z["bold"][a.cfg].astype(float); burn = int(a.burn_s / tr); bold = bold[burn:]
G, sig = float(z["G"][a.cfg]), float(z["sigma"][a.cfg])
st_m, D_m = S.all_stats(bold, tr); rss_m, _ = S.events(S.edge_series(bold))
rng = np.random.default_rng(a.seed)

def chanwise_phase_randomise(b, rng):
    T, N = b.shape; X = np.fft.rfft(b - b.mean(0), axis=0)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size=X.shape)); ph[0] = 1
    if T % 2 == 0: ph[-1] = 1
    return np.fft.irfft(X * ph, n=T, axis=0) + b.mean(0)

rows = {"shared": [], "chanwise": []}
for kind, fn in (("shared", S.phase_randomise), ("chanwise", chanwise_phase_randomise)):
    for k in range(a.K):
        sb = fn(bold, rng); st, D = S.all_stats(sb, tr); rss, _ = S.events(S.edge_series(sb))
        st["ks_to_model"] = S.ks_centred(D, D_m)
        st["rss_q99_ratio"] = float(np.percentile(rss, 99) / np.percentile(rss_m, 99))
        st["rss_kurt_ratio"] = float(((rss - rss.mean()) ** 4).mean() / rss.var() ** 2 /
                                    (((rss_m - rss_m.mean()) ** 4).mean() / rss_m.var() ** 2))
        rows[kind].append(st)

def summ(key, lst):
    v = np.array([r[key] for r in lst], float); return f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}"
print(f"cfg {a.cfg}: G={G} sigma={sig}; model: switch={st_m['switch']:.4f} vaf2={st_m['vaf2']:.3f} "
      f"within={st_m['within']:.3f} cross={st_m['cross']:.3f} n_exc={st_m['n_excursions']}")
for kind in rows:
    print(f"  {kind:9s} switch {summ('switch', rows[kind])} | vaf2 {summ('vaf2', rows[kind])} | "
          f"within {summ('within', rows[kind])} | cross {summ('cross', rows[kind])} | n_exc {summ('n_excursions', rows[kind])} | "
          f"KS->model {summ('ks_to_model', rows[kind])} | RSS q99 ratio {summ('rss_q99_ratio', rows[kind])} | RSS kurt ratio {summ('rss_kurt_ratio', rows[kind])}")
json.dump(dict(model=st_m, surrogates=rows, G=G, sigma=sig), open(a.npz.replace(".npz", f"_g1_cfg{a.cfg}.json"), "w"), indent=1, default=float)
