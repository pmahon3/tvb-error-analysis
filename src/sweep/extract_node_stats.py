"""Collect, from the sweep's batch files, the per-region numbers of every run into one small file
(doc/LOG.md, 2026-11-24, 21:25). For runs that ended non-finite: the first volume at which the BOLD is non-finite.
Usage (tower, flat copy, beside the q_sweep folder): python extract_node_stats.py q_sweep/done node_stats_sweep.npz
"""

import glob
import json
import os
import sys

import numpy as np

folder, out_path = sys.argv[1], sys.argv[2]
columns = {k: [] for k in ("G", "sigma", "eta", "J", "Delta", "dt", "x64", "seed", "finite", "mean_rate", "first_bad_tr")}
node_mean_rate, node_up_occupancy, batch_of = [], [], []
worst_mismatch = 0.0
for json_path in sorted(glob.glob(os.path.join(folder, "*.json"))):
    batch = json.load(open(json_path))
    arrays = np.load(json_path[:-5] + ".npz")
    bold = arrays["bold"] if "bold" in arrays else None
    for row, run in enumerate(batch["runs"]):  # row order of the arrays is the order of "runs"
        columns["G"].append(run["G"])
        columns["sigma"].append(run["nsig"])
        columns["eta"].append(run["eta"])
        columns["J"].append(run["J"])
        columns["Delta"].append(run["Delta"])
        columns["dt"].append(run["dt"])
        columns["x64"].append(bool(run.get("x64", False)))
        columns["seed"].append(run["seed"])
        columns["finite"].append(bool(run["finite"]))
        columns["mean_rate"].append(run.get("m1_mean_r", np.nan))
        first_bad = -1
        if not run["finite"] and bold is not None:
            bad_rows = ~np.isfinite(bold[row]).all(1)
            first_bad = int(np.argmax(bad_rows)) if bad_rows.any() else -1
        columns["first_bad_tr"].append(first_bad)
        node_mean_rate.append(arrays["node_mean_r"][row])
        node_up_occupancy.append(arrays["node_occ07"][row])
        batch_of.append(os.path.basename(json_path)[:-5])
        if run["finite"]:
            worst_mismatch = max(worst_mismatch, abs(float(arrays["node_mean_r"][row].mean()) - run["m1_mean_r"]))
    print(os.path.basename(json_path), len(batch["runs"]), "runs", flush=True)
np.savez_compressed(
    out_path,
    node_mean_rate=np.asarray(node_mean_rate, np.float32),
    node_up_occupancy=np.asarray(node_up_occupancy, np.float32),
    batch=np.asarray(batch_of),
    **{k: np.asarray(v) for k, v in columns.items()},
)
print("runs:", len(batch_of), "| non-finite:", int((~np.asarray(columns["finite"])).sum()))
print("largest |mean over regions - recorded mean rate| among finite runs:", worst_mismatch)
