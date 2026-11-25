import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "common"),
]

import os, sys

import numpy as np, mpr_net as M, ridge_v2_sde as R

for T_s in [0.72, 1.44]:
    job = {"configs": [[1.05, -5.0, 0.0]], "seeds": [3], "dt": 0.05, "T_s": T_s}
    run, info = R.build(job)
    bold_j, fu_j, fin, r_end, v_end, hemo = (
        np.asarray(x) if not isinstance(x, tuple) else x for x in run()
    )
    W, L = M.load_connectome(os.path.dirname(os.path.abspath(__file__)))
    N = W.shape[0]
    up = np.random.default_rng(1000 * 3 + 7).random(N) < 0.5
    r0 = np.where(up, R.R_UP, R.R_DN)
    v0 = np.where(up, R.V_UP, R.V_DN)
    M.ETA = -5.0
    out = M.run_batch(
        W,
        L,
        np.array([1.05]),
        np.array([0.0]),
        seed=0,
        T_ms=T_s * 1000,
        dt=0.05,
        r0=r0[None],
        v0=v0[None],
    )
    print(
        f"T={T_s}s  r_end max abs diff {np.abs(out['r_end'][0] - r_end[0]).max():.2e}  v_end {np.abs(out['v_end'][0] - v_end[0]).max():.2e}  "
        f"BOLD last diff rel {np.abs(out['bold'][0][-1] - bold_j[-1, 0]).max() / np.abs(out['bold'][0][-1]).max():.2e}"
    )
