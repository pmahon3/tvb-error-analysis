"""24 more TVB seeds at the working point (README, 2026-11-24, 12:05: control for the port's small offset from TVB).
Usage (tower, venv_tvb): python tvb_ref_batch_more.py <K>
"""
import sys
from concurrent.futures import ThreadPoolExecutor
import tvb_ref_batch as b
JOBS = [(0.54, 0.030, 0.01, s) for s in range(9, 33)]
if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"{len(JOBS)} jobs, {K} at a time", flush=True)
    with ThreadPoolExecutor(K) as ex:
        for tag, status in ex.map(b.run, JOBS): print(tag, "|", status, flush=True)
    print("batch finished", flush=True)
