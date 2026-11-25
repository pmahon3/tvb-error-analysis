"""Write the job files of the registered sweep (../doc/LOG.md, 2026-11-24, 12:05).
Usage: python3 -I sweep_make_jobs.py <out_dir> <d_eta> <d_J> <d_Delta> <n_seeds> [max_batch=1024]
Grid G = 0.40 .. 0.70 step 0.01 (31) x sigma = 0.018 .. 0.054 step 0.004 (10; includes 0.030). At each point: base and
eta, J, Delta +- delta at dt 0.01; base at dt 0.005; the sigma = 0.030 row again in float64.
"""
import sys, os, json
import numpy as np
out, d_eta, d_J, d_Delta, n_seeds = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
max_b = int(sys.argv[6]) if len(sys.argv) > 6 else 1024
os.makedirs(out, exist_ok=True)
Gs = [round(0.40 + 0.01 * i, 2) for i in range(31)]; Ss = [round(0.018 + 0.004 * i, 3) for i in range(10)]   # includes 0.030
seeds = list(range(1, n_seeds + 1)); common = dict(conn="hcp001_conn.npz", speed=20.0, seeds=seeds, T_units=60000, warm_units=1000, n_tr_block=7, save_bold=True)
variants = [(0.0, 0.0, 0.0), (d_eta, 0, 0), (-d_eta, 0, 0), (0, d_J, 0), (0, -d_J, 0), (0, 0, d_Delta), (0, 0, -d_Delta)]
main = [[G, round(-5.0 + a, 6), round(15.0 + b, 6), round(1.0 + c, 6), s] for s in Ss for G in Gs for (a, b, c) in variants]
base = [[G, -5.0, 15.0, 1.0, s] for s in Ss for G in Gs]
row64 = [[G, -5.0, 15.0, 1.0, 0.030] for G in Gs]
C = max_b // n_seeds; n = 0
def write(prefix, configs, dt, tag, **extra):
    global n
    for k in range(0, len(configs), C):
        json.dump(dict(common, dt=dt, params=configs[k:k + C], tag=tag, **extra), open(os.path.join(out, f"{prefix}_{k // C:02d}.json"), "w")); n += 1
write("a_dt0.01", main, 0.01, "sweep dt 0.01"); write("b_dt0.005", base, 0.005, "sweep dt 0.005"); write("c_f64row", row64, 0.01, "float64 row sigma 0.030", x64=True)
runs = (len(main) + len(base) + len(row64)) * n_seeds
print(f"{n} job files; configs: {len(main)} at dt 0.01, {len(base)} at dt 0.005, {len(row64)} float64; {runs} runs; configs per batch {C} (batch {C * n_seeds})")
