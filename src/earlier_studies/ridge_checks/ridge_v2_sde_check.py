"""Check ridge_v2_sde.py against mpr_net.run_batch with sigma = 0 and the same initial state (3 s, two configurations)."""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[2] / "core"),
]

import os, sys, json

import numpy as np
import mpr_net as M
import ridge_v2_sde as R

job = {"configs": [[1.05, -5.0, 0.0], [1.08, -4.9, 0.0]], "seeds": [3], "dt": 0.05, "T_s": 3}
run, info = R.build(job)
bold_j, fu_j, fin = (np.asarray(x) for x in run()[:3])  # bold (n_tr, B, N)
W, L = M.load_connectome(os.path.dirname(os.path.abspath(__file__)))
N = W.shape[0]
up = np.random.default_rng(1000 * 3 + 7).random(N) < 0.5
r0 = np.where(up, R.R_UP, R.R_DN)
v0 = np.where(up, R.V_UP, R.V_DN)
for b, (G, eta, sig) in enumerate(job["configs"]):
    M.ETA = eta
    out = M.run_batch(
        W, L, np.array([G]), np.array([0.0]), seed=0, T_ms=3000, dt=0.05, r0=r0[None], v0=v0[None]
    )
    bn = out["bold"][0]
    bj = bold_j[:, b, :]
    n = min(len(bn), len(bj))
    print(
        f"config {b} (G={G}, eta={eta}): BOLD max rel diff {np.abs(bn[:n] - bj[:n]).max() / np.abs(bn[:n]).max():.2e}; "
        f"final r max abs diff {np.abs(out['r_end'][0] - 0).max():.3f} (numpy r_end range check)"
    )
M.ETA = -5.0
