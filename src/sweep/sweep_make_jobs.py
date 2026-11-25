"""Write the job files of the parameter sweep (registered in doc/LOG.md, 2026-11-24, 12:05).

Usage:
    python3 -I sweep_make_jobs.py <output_folder> <nudge_eta> <nudge_J> <nudge_Delta> <n_seeds> [largest_batch=1024]

The grid is G = 0.40 ... 0.70 in steps of 0.01 (31 values) by sigma = 0.018 ... 0.054 in steps of 0.004 (10 values,
including the paper's 0.030). At each grid point:
    a_  step 0.01: the base setting, and eta, J and Delta each nudged up and down (7 settings)
    b_  step 0.005: the base setting only
    c_  the sigma = 0.030 row once more in double precision
Each job file holds as many settings as fit in one batch; mprjax_run.py runs one job file.
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

import json
import os

ETA0, J0, DELTA0 = -5.0, 15.0, 1.0

output_folder = sys.argv[1]
nudge_eta, nudge_J, nudge_Delta = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
n_seeds = int(sys.argv[5])
largest_batch = int(sys.argv[6]) if len(sys.argv) > 6 else 1024
os.makedirs(output_folder, exist_ok=True)

G_values = [round(0.40 + 0.01 * i, 2) for i in range(31)]
sigma_values = [round(0.018 + 0.004 * i, 3) for i in range(10)]
seeds = list(range(1, n_seeds + 1))

# what every job has in common
common = dict(
    conn="hcp001_conn.npz",
    speed=20.0,
    seeds=seeds,
    T_units=60000,
    warm_units=1000,
    n_tr_block=7,
    save_bold=True,
)

# changes to (eta, J, Delta): none, then each one up and down
nudges = [
    (0.0, 0.0, 0.0),
    (nudge_eta, 0, 0),
    (-nudge_eta, 0, 0),
    (0, nudge_J, 0),
    (0, -nudge_J, 0),
    (0, 0, nudge_Delta),
    (0, 0, -nudge_Delta),
]

# each setting is a row (G, eta, J, Delta, sigma)
with_nudges = [
    [G, round(ETA0 + d_eta, 6), round(J0 + d_J, 6), round(DELTA0 + d_Delta, 6), sigma]
    for sigma in sigma_values
    for G in G_values
    for (d_eta, d_J, d_Delta) in nudges
]
base_only = [[G, ETA0, J0, DELTA0, sigma] for sigma in sigma_values for G in G_values]
double_precision_row = [[G, ETA0, J0, DELTA0, 0.030] for G in G_values]

settings_per_job = largest_batch // n_seeds
n_job_files = 0


def write_jobs(prefix, settings, dt, tag, **extra):
    """Split the settings into job files of at most `settings_per_job` settings each."""
    global n_job_files
    for start in range(0, len(settings), settings_per_job):
        job = dict(
            common, dt=dt, params=settings[start : start + settings_per_job], tag=tag, **extra
        )
        file_name = f"{prefix}_{start // settings_per_job:02d}.json"
        json.dump(job, open(os.path.join(output_folder, file_name), "w"))
        n_job_files += 1


write_jobs("a_dt0.01", with_nudges, 0.01, "sweep dt 0.01")
write_jobs("b_dt0.005", base_only, 0.005, "sweep dt 0.005")
write_jobs("c_f64row", double_precision_row, 0.01, "float64 row sigma 0.030", x64=True)

n_runs = (len(with_nudges) + len(base_only) + len(double_precision_row)) * n_seeds
print(
    f"{n_job_files} job files; configs: {len(with_nudges)} at dt 0.01, {len(base_only)} at dt 0.005, "
    f"{len(double_precision_row)} float64; {n_runs} runs; configs per batch {settings_per_job} "
    f"(batch {settings_per_job * n_seeds})"
)
