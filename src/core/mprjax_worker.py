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
name, Q = sys.argv[1], sys.argv[2]
for d in ("todo", "running", "done", "failed"):
    os.makedirs(os.path.join(Q, d), exist_ok=True)
while True:
    todo = sorted(glob.glob(os.path.join(Q, "todo", "*.json")))
    if not todo:
        print(name, "queue empty, stopping", flush=True)
        break
    src = todo[0]
    base = os.path.basename(src)[:-5]
    mine = os.path.join(Q, "running", f"{base}.{name}.json")
    try:
        os.rename(src, mine)
    except OSError:
        time.sleep(1.0)
        continue  # another worker took it
    out = os.path.join(Q, "done", base)
    if os.path.exists(out + ".json"):
        os.remove(mine)
        continue
    job = json.load(open(mine))
    env = dict(os.environ)
    if job.get("x64"):
        env["JAX_ENABLE_X64"] = "1"
    t0 = time.time()
    rc = subprocess.call(
        [sys.executable, os.path.join(HERE, "mprjax_run.py"), mine, out],
        stdout=open(out + ".log", "w"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    if rc == 0 and os.path.exists(out + ".json"):
        os.remove(mine)
        print(name, base, f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    else:
        os.rename(mine, os.path.join(Q, "failed", base + ".json"))
        print(name, base, "FAILED rc", rc, flush=True)
