"""Run the registered stage A' job list (../doc/LOG.md, 2026-11-24) with K processes at a time; finished jobs are skipped.
Usage (tower, venv_tvb): python tvb_ref_batch.py <K>
Each job is one call of tvb_ref_run.py; its output and errors go to runs_A/<tag>.log (nothing is discarded).
"""
import sys, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs_A"); os.makedirs(RUNS, exist_ok=True)
WP = (0.54, 0.030)                                                   # the paper's working point (G, sigma)
JOBS = [(WP[0], WP[1], 0.01, s) for s in range(1, 9)]                # their step, at the working point
JOBS += [(G, WP[1], 0.01, s) for G in (0.30, 0.80) for s in range(1, 5)]      # validity controls
JOBS += [(G, WP[1], 0.01, s) for G in (0.53, 0.55) for s in range(1, 9)]      # their grid neighbours
JOBS += [(WP[0], WP[1], dt, s) for dt in (0.02, 0.005) for s in range(1, 9)]  # step series
JOBS += [(WP[0], WP[1], 0.0025, s) for s in range(1, 5)]


def tag_of(job):
    G, nsig, dt, seed = job
    return f"A_G{G:.2f}_sig{nsig:.3f}_dt{dt}_seed{seed}"


def run(job):
    G, nsig, dt, seed = job; out = os.path.join(RUNS, tag_of(job))
    if os.path.exists(out + ".json"): return tag_of(job), "already done"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    t0 = time.time()
    with open(out + ".log", "w") as log:
        rc = subprocess.call([sys.executable, os.path.join(HERE, "tvb_ref_run.py"), str(G), str(nsig), str(dt), str(seed), out],
                             stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    return tag_of(job), f"exit {rc} after {(time.time() - t0) / 60:.1f} min"


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"{len(JOBS)} jobs, {K} at a time", flush=True)
    with ThreadPoolExecutor(K) as ex:
        for tag, status in ex.map(run, JOBS):
            print(tag, "|", status, flush=True)
    print("batch finished", flush=True)
