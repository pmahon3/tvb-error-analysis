"""Timing of the batched JAX runner at several batch sizes (doc/LOG.md, 2026-11-24, "JAX version": timing, description only).
Usage (inside the vbjax container): python3 mprjax_bench.py <tag> <dt> <C1> [C2 ...]   (batch = C x 8 seeds)
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

import sys, os, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_port")
os.makedirs(OUT, exist_ok=True)
tag, dt = sys.argv[1], float(sys.argv[2])
for C in [int(x) for x in sys.argv[3:]]:
    name = os.path.join(OUT, f"bench_{tag}_dt{dt}_C{C}")
    job = dict(
        conn="hcp001_conn.npz",
        speed=20.0,
        dt=dt,
        params=[[0.54, -5.0, 15.0, 1.0, 0.030]] * C,
        seeds=list(range(1, 9)),
        T_units=2016,
        warm_units=50,
        n_tr_block=7,
        tag=f"bench {tag}",
    )
    json.dump(job, open(name + "_job.json", "w"))
    rc = subprocess.call(
        [
            sys.executable,
            os.path.join(HERE, "..", "core", "mprjax_run.py"),
            name + "_job.json",
            name,
        ],
        stdout=open(name + ".log", "w"),
        stderr=subprocess.STDOUT,
    )
    if rc == 0:
        print(
            "BENCH",
            tag,
            "dt",
            dt,
            "B",
            C * 8,
            json.dumps(json.load(open(name + ".json"))["timing"]),
            flush=True,
        )
    else:
        print(
            "BENCH",
            tag,
            "B",
            C * 8,
            "FAILED rc",
            rc,
            "|",
            open(name + ".log").read()[-400:].replace("\n", " / "),
            flush=True,
        )
