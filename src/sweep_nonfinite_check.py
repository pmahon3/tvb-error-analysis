"""Where the non-finite runs of the sweep failed: rates or only the BOLD model? Reads one batch's json + npz. Usage: python sweep_nonfinite_check.py <prefix>"""
import sys, json, numpy as np
d = json.load(open(sys.argv[1] + ".json")); z = np.load(sys.argv[1] + ".npz")
bad = [i for i, r in enumerate(d["runs"]) if not r["finite"]]; print("non-finite runs:", len(bad), "of", len(d["runs"]))
for i in bad:
    r = d["runs"][i]; u = z["u07"][i]; b = z["bold"][i] if "bold" in z else None
    nm = z["node_mean_r"][i]
    first_bad_bold = int(np.argmax(~np.isfinite(b).all(1))) if b is not None and not np.isfinite(b).all() else -1
    print(f"  G {r['G']:.2f} sig {r['nsig']:.3f} eta {r['eta']} J {r['J']} D {r['Delta']} seed {r['seed']}: node_mean_r finite {bool(np.isfinite(nm).all())} (max {np.nanmax(nm):.3g}); u07 all zero {bool((u == 0).all())}; first non-finite BOLD row {first_bad_bold} of {0 if b is None else b.shape[0]}; nodes with non-finite BOLD {0 if b is None else int((~np.isfinite(b)).any(0).sum())}")
