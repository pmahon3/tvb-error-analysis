"""Per-TR up-fraction of every working-point run (TVB seeds 1-96; port prec2 float32 and float64, 64 seeds each) into one small file.
Usage (tower, venv_tvb): python dump_traces.py"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "common")]

import os, glob, re, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); tv = {}
for p in glob.glob(os.path.join(H, "runs_A", "A_G0.54_sig0.030_dt0.01_seed*.npz")):
    tv[int(re.search(r"seed(\d+)", p).group(1))] = np.load(p)["u07"]
sd = sorted(tv); out = dict(tvb=np.stack([tv[s] for s in sd]).astype(np.float32), tvb_seeds=np.array(sd))
for n in ("prec2_f32", "prec2_f64"):
    out[n] = np.load(os.path.join(H, "runs_port", n + ".npz"))["u07"].astype(np.float32)
np.savez_compressed(os.path.join(H, "traces_wp.npz"), **out); print({k: v.shape for k, v in out.items()})
