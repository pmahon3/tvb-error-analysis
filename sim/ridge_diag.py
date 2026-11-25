"""Where do non-finite values appear in the ridge-check configurations? (45 s, seeds 1-3)"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, ridge_check as R, mpr_net as M
W, L = M.load_connectome(os.path.dirname(os.path.abspath(__file__)))
dt = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
print("dt =", dt)
rr, bb = R.run(W, L, [1, 2, 3], 45000, dt=dt)
for s in range(3):
    for c in range(5):
        r, b = rr[s, c], bb[s, c]
        br = np.where(~np.isfinite(r).all(1))[0]; bbd = np.where(~np.isfinite(b).all(1))[0]
        print(R.CONFIGS[c], "seed", s + 1, "| r nonfinite from sample", br[:1].tolist(), "| bold nonfinite from TR", bbd[:1].tolist(),
              "| max r", float(np.nanmax(r)), "| min BOLD", float(np.nanmin(b)) if np.isfinite(b).any() else None)
