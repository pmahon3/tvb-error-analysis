"""Are the two statistics paths the same? The port's online statistics (mprjax_run.host_stats on device accumulators)
against the offline statistics used for the TVB runs (tvb_ref_run.statistics), on the SAME simulated rates.
doc/LOG.md, 2026-11-24 (open item on pattern statistics). Usage: .venv_vbjax/bin/python -I stats_path_check.py [n_blocks=20]
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import jax
import mprjax, mprjax_run, tvb_ref_run
from mprjax import UNITS_PER_TR, SAMPLES_PER_UNIT

nblk = int(sys.argv[1]) if len(sys.argv) > 1 else 20; nb = 7; n_tr = nblk * nb; warm = 1000
conn = np.load(__import__("repo_paths").data("hcp001_conn.npz")); W, L = conn["weights"], conn["tract_lengths"]
R = mprjax.build(W, L, 20.0, 0.01, [[0.54, -5.0, 15.0, 1.0, 0.030]], [1, 2], n_tr_block=nb, debug_r=True); B, N = R["B"], R["N"]
print("x64:", bool(jax.config.jax_enable_x64), "| B", B, "| TRs", n_tr, flush=True); t0 = time.time()
state = R["warm"](R["init"](), np.int32(0), np.arange(warm, dtype=np.int32)); hemo = R["fresh_hemo"]()
f4 = np.float32; bold = np.zeros((n_tr, B, N), f4); sum_r = np.zeros((n_tr, B, N), f4); up7 = np.zeros((n_tr, B, N), f4); up5 = np.zeros((n_tr, B, N), f4)
mx = np.zeros((n_tr, B)); sx = np.zeros((B, N)); sxx = np.zeros((B, N, N)); r_all = np.zeros((n_tr, UNITS_PER_TR, SAMPLES_PER_UNIT, N, B), np.float64)
for k in range(nblk):
    state, hemo, per_tr, sx_k, sxx_k, fin = R["block"](state, hemo, np.int32(warm + k * nb * UNITS_PER_TR)); sl = slice(k * nb, (k + 1) * nb)
    bo, sr_, u7_, u5_, mx_, ra = (np.asarray(a) for a in per_tr)
    bold[sl], sum_r[sl], up7[sl], up5[sl], mx[sl] = bo.transpose(0, 2, 1), sr_.transpose(0, 2, 1), u7_.transpose(0, 2, 1), u5_.transpose(0, 2, 1), mx_
    r_all[sl] = ra; sx += np.asarray(sx_k, np.float64).T; sxx += np.asarray(sxx_k, np.float64)
print(f"simulated in {time.time() - t0:.0f} s", flush=True)
for b in range(B):
    online, u7o, _ = mprjax_run.host_stats(bold[:, b], sum_r[:, b], up7[:, b], up5[:, b], mx[:, b], sx[b], sxx[b], W, n_tr)
    r = r_all[..., b].reshape(n_tr * UNITS_PER_TR * SAMPLES_PER_UNIT, N)
    offline, arrays = tvb_ref_run.statistics(r, W)
    print(f"\nrun {b}: max |BOLD online - offline| = {np.abs(bold[:, b] - arrays['bold']).max():.2e} (BOLD sd {arrays['bold'][41:].std():.2e}); max |u07 online - offline| = {np.abs(u7o - arrays['u07']).max():.2e}")
    for m in sorted(offline):
        a, o = online.get(m), offline[m]
        print(f"   {m:16s} online {a:.6f}   offline {o:.6f}   difference {a - o:+.2e} ({(a - o) / abs(o) * 100 if o else 0:+.3f}%)")
