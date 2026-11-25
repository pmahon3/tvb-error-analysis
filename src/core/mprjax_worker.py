"""Queue worker for batched JAX runs: takes job files from <queue>/todo one at a time (claimed by renaming), runs each
in its own process, writes results to <queue>/done. Finished jobs are skipped; a failed job is moved to <queue>/failed.
Usage (inside the vbjax container): python3 mprjax_worker.py <worker_name> <queue_dir>
A job with "x64": true runs in double precision.
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

import sys, os, glob, json, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
worker_name, queue = sys.argv[1], sys.argv[2]
for folder in ("todo", "running", "done", "failed"):
    os.makedirs(os.path.join(queue, folder), exist_ok=True)

while True:
    waiting = sorted(glob.glob(os.path.join(queue, "todo", "*.json")))
    if not waiting:
        print(worker_name, "queue empty, stopping", flush=True)
        break

    # claim the first waiting job by moving it; if another worker got there first, the move fails
    job_file = waiting[0]
    job_name = os.path.basename(job_file)[:-5]
    claimed = os.path.join(queue, "running", f"{job_name}.{worker_name}.json")
    try:
        os.rename(job_file, claimed)
    except OSError:
        time.sleep(1.0)
        continue

    output_prefix = os.path.join(queue, "done", job_name)
    if os.path.exists(output_prefix + ".json"):  # already finished earlier
        os.remove(claimed)
        continue

    # each job runs in its own process, so a crash loses one job and frees the GPU memory
    job = json.load(open(claimed))
    environment = dict(os.environ)
    if job.get("x64"):
        environment["JAX_ENABLE_X64"] = "1"
    started = time.time()
    return_code = subprocess.call(
        [sys.executable, os.path.join(HERE, "mprjax_run.py"), claimed, output_prefix],
        stdout=open(output_prefix + ".log", "w"),
        stderr=subprocess.STDOUT,
        env=environment,
    )
    if return_code == 0 and os.path.exists(output_prefix + ".json"):
        os.remove(claimed)
        print(worker_name, job_name, f"done in {(time.time() - started) / 60:.1f} min", flush=True)
    else:
        os.rename(claimed, os.path.join(queue, "failed", job_name + ".json"))
        print(worker_name, job_name, "FAILED rc", return_code, flush=True)
