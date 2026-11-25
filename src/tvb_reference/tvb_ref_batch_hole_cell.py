"""32 TVB runs at a cell where JAX runs went non-finite (G 0.68, sigma 0.054; doc/LOG.md, 2026-11-24, 21:25).
Usage (tower, venv_tvb, flat copy beside tvb_ref_batch.py): python tvb_ref_batch_hole_cell.py <K>
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "core")]

import tvb_ref_batch as b

JOBS = [(0.68, 0.054, 0.01, seed) for seed in range(201, 233)]

if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"{len(JOBS)} jobs, {K} at a time", flush=True)
    with ThreadPoolExecutor(K) as ex:
        for tag, status in ex.map(b.run, JOBS):
            print(tag, "|", status, flush=True)
    print("batch finished", flush=True)
