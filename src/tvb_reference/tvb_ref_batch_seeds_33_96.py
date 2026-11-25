"""64 more TVB seeds at the working point (doc/LOG.md, 2026-11-24, 15:20: pattern statistics, port against TVB).
Usage (tower, venv_tvb): python tvb_ref_batch_seeds_9_32.py <K>
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

import sys
from concurrent.futures import ThreadPoolExecutor
import tvb_ref_batch as b

JOBS = [(0.54, 0.030, 0.01, s) for s in range(33, 97)]
if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"{len(JOBS)} jobs, {K} at a time", flush=True)
    with ThreadPoolExecutor(K) as ex:
        for tag, status in ex.map(b.run, JOBS):
            print(tag, "|", status, flush=True)
    print("batch finished", flush=True)
