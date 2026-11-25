"""Recompute BOLD from the kept 10 ms firing rate with the firing rate itself as hemodynamic drive,
then recompute the statistics. python3 -I rebold.py ../results/runs/FILE.npz  (writes FILE_rb.npz and prints the table)"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mpr_net as M, stats as S
f = sys.argv[1]; z = np.load(f, allow_pickle=True); R = z['r10']; G = z['G']; sig = z['sigma'] if 'sigma' in z else z['speed']
B, T, N = R.shape; dt = 0.010; step = 72; nb = T // step; bold = np.zeros((B, nb, N), np.float32); h = M.Balloon(B, N)
for t in range(T):
    h.step(R[:, t, :].astype(float), dt)
    if (t + 1) % step == 0: bold[:, (t + 1) // step - 1] = h.bold()
burn_b = int(20 / 0.72); burn_r = 2000; rows = []
print("G     sigma  switch    vaf2_bold vaf2_r  frac_up n_switch within cross  nonev  n_exc  meancorr")
for b in range(B):
    bo = bold[b, burn_b:].astype(float); r10 = R[b, burn_r:].astype(float)
    st, D = S.all_stats(bo, 0.72); k = 10; rs = np.apply_along_axis(lambda x: np.convolve(x, np.ones(k) / k, mode="valid"), 0, r10)
    C = np.corrcoef(bo.T); iu = np.triu_indices(N, 1)
    st.update(G=float(G[b]), sigma=float(sig[b]), vaf2_r=S.vaf2(r10), frac_up=float((rs > 0.5).mean()),
              n_switch=int((np.abs(np.diff((rs > 0.5).astype(int), axis=0)) > 0).sum()), meancorr=float(np.nanmean(C[iu]))); rows.append(st)
    print(f"{st['G']:<5.2f} {st['sigma']:<6.2f} {st['switch']:<9.4f} {st['vaf2']:<9.3f} {st['vaf2_r']:<7.3f} {st['frac_up']:<7.2f} {st['n_switch']:<8d} {st['within']:<6.2f} {st['cross']:<6.2f} {st['nonevents']:<6.2f} {st['n_excursions']:<6d} {st['meancorr']:.3f}")
np.savez_compressed(f.replace('.npz', '_rb.npz'), bold=bold, G=G, sigma=sig, stats=json.dumps(rows, default=float))
