"""Controls after the sweep (doc/LOG.md, 2026-11-24): float64 row in absolute terms; TVB (96 seeds) against the port at the working point.
Usage: python3 -I sweep_controls.py results/runs/sweep/done results/runs/Aprime results/runs/port
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "common"),
]

import sys, os, json, glob, site

sys.path.append(site.getusersitepackages())
import numpy as np
import sweep_maps as sm
from sweep_maps import STATS, NAMES, ETA0, J0, D0

cells, _ = sm.load(sys.argv[1])
Gs = sorted({k[0] for k in cells if k[5] == 0.01 and not k[6] and k[2:5] == (ETA0, J0, D0)})
Ss = sorted({k[1] for k in cells if k[5] == 0.01 and not k[6] and k[2:5] == (ETA0, J0, D0)})
m32 = sm.grid(cells, Gs, Ss)
m64 = sm.grid(cells, Gs, Ss, x64=True)
i0 = Ss.index(0.03)
print(
    "float64 minus float32 along sigma 0.030 (4 seeds each): G, mean rate f32, diff (combined se), z | switching diff (se) | mean FC diff (se)"
)
for j, G in enumerate(Gs):
    d = m64[0][:, i0, j] - m32[0][:, i0, j]
    e = np.hypot(m64[1][:, i0, j], m32[1][:, i0, j])
    print(
        f"  {G:.2f}  {m32[0][0, i0, j]:.4f}  {d[0]:+.5f} ({e[0]:.5f}) z {d[0] / e[0]:+.1f} | {d[4]:+.4f} ({e[4]:.4f}) | {d[5]:+.4f} ({e[5]:.4f})"
    )
z = (m64[0][:, i0, :] - m32[0][:, i0, :]) / np.hypot(m64[1][:, i0, :], m32[1][:, i0, :])
lo = np.array(Gs) <= 0.51
print(
    "mean z, mean rate: G <= 0.51:",
    round(float(z[0][lo].mean()), 2),
    "; G >= 0.52:",
    round(float(z[0][~lo].mean()), 2),
)

tvb = [
    json.load(open(p))
    for p in sorted(glob.glob(os.path.join(sys.argv[2], "A_G0.54_sig0.030_dt0.01_seed*.json")))
]
tvb = [r for r in tvb if r.get("finite", True)]
seeds = np.array([r["seed"] for r in tvb])
port = []
for p in (
    sorted(glob.glob(os.path.join(sys.argv[3], "*.json")))
    + sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))
    + [os.path.join(os.path.dirname(sys.argv[1]), "pilot_done.json")]
):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    for r in d.get("runs", []) if isinstance(d, dict) else []:
        if (
            r.get("finite")
            and abs(r["G"] - 0.54) < 1e-9
            and abs(r["nsig"] - 0.030) < 1e-9
            and r["dt"] == 0.01
            and abs(r.get("eta", -5) + 5) < 1e-9
            and abs(r.get("J", 15) - 15) < 1e-9
            and abs(r.get("Delta", 1) - 1) < 1e-9
            and r.get("T", 0) > 50000
            and "switching_index" in r
        ):
            port.append((os.path.basename(p), r))
from collections import Counter

print(
    "\nTVB runs:",
    len(tvb),
    "seeds",
    seeds.min(),
    "-",
    seeds.max(),
    "| port runs:",
    len(port),
    dict(Counter((f, r.get("x64")) for f, r in port)),
)


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


print(
    "statistic            TVB 1-32          TVB 33-96 (new)     TVB all 96        port f32          port f64          all TVB - all port (se units)"
)
for k, m in enumerate(STATS):
    a = ms([r[m] for r in tvb if r["seed"] <= 32])
    b = ms([r[m] for r in tvb if r["seed"] > 32])
    c = ms([r[m] for r in tvb])
    p32 = ms([r[m] for f, r in port if not r.get("x64")])
    p64 = ms([r[m] for f, r in port if r.get("x64")])
    pa = ms([r[m] for f, r in port])
    print(
        f"{NAMES[k]:20s} {a[0]:.4f} ± {a[1]:.4f}   {b[0]:.4f} ± {b[1]:.4f}   {c[0]:.4f} ± {c[1]:.4f}   {p32[0]:.4f} ± {p32[1]:.4f}   {p64[0]:.4f} ± {p64[1]:.4f}   new TVB vs port {(b[0] - pa[0]) / np.hypot(b[1], pa[1]):+.1f}; all {(c[0] - pa[0]) / np.hypot(c[1], pa[1]):+.1f}"
    )
