"""Batched runs of the JAX port (mprjax.py) with the statistics of stage A' (tvb_ref_run.statistics), computed online.
Usage: python3 mprjax_run.py <job.json> <out_prefix>
job.json: {"conn": "hcp001_conn.npz", "speed": 20.0, "dt": 0.01, "params": [[G, eta, J, Delta, sigma], ...], "seeds": [...],
           "T_units": 60000, "warm_units": 1000, "n_tr_block": 7, "tag": "..."}
Writes <out_prefix>.npz (per run: BOLD per TR, up-fraction per TR, per-node values, covariance of r) and <out_prefix>.json
(one record per run, same field names as the TVB reference runs). Batch row b = (config b % C, seed b // C).
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "common"),
]

import sys, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import jax
import mprjax
from mprjax import UNITS_PER_TR, SAMPLES_PER_UNIT


def host_stats(bold, sum_r, up7, up5, mx, sx, sxx, W, n_tr):
    """One run. bold (n_tr, N); sum_r, up7, up5 (n_tr, N) per-TR per-node sums over 720 samples; sx (N,), sxx (N, N) over n units."""
    from stats import dfc_w, switching_index, vaf2

    N = bold.shape[1]
    per_tr = UNITS_PER_TR * SAMPLES_PER_UNIT
    n_s = n_tr * per_tr
    n_u = n_tr * UNITS_PER_TR
    sum_r, up7, up5 = (np.asarray(a, np.float64) for a in (sum_r, up7, up5))
    u7 = up7.sum(1) / (per_tr * N)
    u5 = up5.sum(1) / (per_tr * N)
    cov = (sxx - np.outer(sx, sx) / n_u) / (n_u - 1)
    cov = 0.5 * (cov + cov.T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    s = dict(
        m1_mean_r=float(sum_r.sum() / (n_s * N)),
        m2_occ_up07=float(up7.sum() / (n_s * N)),
        occ_up05=float(up5.sum() / (n_s * N)),
        m3_u_median=float(np.median(u7)),
        m4_u_p05=float(np.percentile(u7, 5)),
        u_p95=float(np.percentile(u7, 95)),
        u_sd=float(u7.std()),
        m5_vaf2_r=float(ev[:2].sum() / ev.sum()) if ev.sum() > 0 else float("nan"),
        max_r=float(mx.max()),
    )
    b = bold[int(30 / 0.72) :].astype(np.float64)
    iu = np.triu_indices(N, 1)
    if (
        b.shape[0] >= 50 and np.isfinite(b).all()
    ):  # shorter records (smoke runs) get no BOLD statistics
        FC = np.nan_to_num(np.corrcoef(b.T))
        Dm, Ws = dfc_w(b)
        s.update(
            switching_index=float(switching_index(Dm, Ws)),
            mean_fc=float(FC[iu].mean()),
            corr_fc_sc=float(np.corrcoef(FC[iu], (W + W.T)[iu])[0, 1]),
            vaf2_bold=vaf2(b),
        )
    return s, u7, u5


def main():
    job = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    conn = np.load(__import__("repo_paths").data(job["conn"]))
    W, L = conn["weights"], conn["tract_lengths"]
    params = np.asarray(job["params"], float)
    seeds = list(job["seeds"])
    dt = float(job["dt"])
    nb = int(job.get("n_tr_block", 7))
    n_tr = int(job["T_units"] // UNITS_PER_TR)
    n_tr -= n_tr % nb
    warm_units = int(job["warm_units"])
    print("devices:", jax.devices(), "| x64:", bool(jax.config.jax_enable_x64), flush=True)
    R = mprjax.build(W, L, float(job["speed"]), dt, params, seeds, n_tr_block=nb)
    B, N = R["B"], R["N"]
    steps_total = (warm_units + n_tr * UNITS_PER_TR) * SAMPLES_PER_UNIT * R["steps_per_sample"]
    print(
        f"B = {B} ({R['C']} configs x {R['S']} seeds); N = {N}; H = {R['H']}; dt = {dt}; {n_tr} TRs in blocks of {nb}; {steps_total} steps",
        flush=True,
    )
    t0 = time.time()
    state = R["init"]()
    state = R["warm"](state, np.int32(0), np.arange(warm_units, dtype=np.int32))
    jax.block_until_ready(state[0])
    t_warm = time.time() - t0
    print(f"warm-up ({warm_units} units, with compilation): {t_warm:.1f} s", flush=True)
    hemo = R["fresh_hemo"]()
    f4 = np.float32  # per-TR sums are at most 720 terms: exact enough in float32
    bold = np.zeros((n_tr, B, N), f4)
    sum_r = np.zeros((n_tr, B, N), f4)
    up7 = np.zeros((n_tr, B, N), f4)
    up5 = np.zeros((n_tr, B, N), f4)
    mx = np.zeros((n_tr, B))
    sx = np.zeros((B, N))
    sxx = np.zeros((B, N, N))
    finite = np.ones(B, bool)
    t_blocks = []
    for k in range(n_tr // nb):
        t1 = time.time()
        state, hemo, per_tr, sx_k, sxx_k, fin = R["block"](
            state, hemo, np.int32(warm_units + k * nb * UNITS_PER_TR)
        )
        sl = slice(k * nb, (k + 1) * nb)
        bo, sr_, u7_, u5_, mx_ = (np.asarray(a) for a in per_tr)  # device layout: (n_tr, N, B)
        bold[sl], sum_r[sl], up7[sl], up5[sl], mx[sl] = (
            bo.transpose(0, 2, 1),
            sr_.transpose(0, 2, 1),
            u7_.transpose(0, 2, 1),
            u5_.transpose(0, 2, 1),
            mx_,
        )
        sx += np.asarray(sx_k, np.float64).T
        sxx += np.asarray(sxx_k, np.float64)
        finite &= np.asarray(fin)
        t_blocks.append(time.time() - t1)
        if k in (0, 1, 2) or (k + 1) % 20 == 0:
            print(
                f"block {k + 1}/{n_tr // nb}: {t_blocks[-1]:.2f} s; finite rows {int(finite.sum())}/{B}",
                flush=True,
            )
    wall = time.time() - t0
    steps_block = nb * UNITS_PER_TR * SAMPLES_PER_UNIT * R["steps_per_sample"]
    tb = float(np.median(t_blocks[1:])) if len(t_blocks) > 1 else t_blocks[0]
    timing = dict(
        wall_s=round(wall, 1),
        warm_s=round(t_warm, 1),
        median_block_s=round(tb, 3),
        us_per_step=round(tb / steps_block * 1e6, 2),
        us_per_step_per_run=round(tb / steps_block * 1e6 / B, 4),
        s_per_run=round(wall / B, 2),
        B=B,
    )
    print("timing:", json.dumps(timing), flush=True)
    recs = []
    u07 = np.zeros((B, n_tr), np.float32)
    u05 = np.zeros((B, n_tr), np.float32)
    for b_ in range(B):
        c = int(R["config_of"][b_])
        G, eta, J, Delta, sigma = (float(x) for x in params[c])
        rec = dict(
            G=G,
            eta=eta,
            J=J,
            Delta=Delta,
            nsig=sigma,
            dt=dt,
            seed=int(seeds[int(R["seed_of"][b_])]),
            config=c,
            T=float(n_tr * UNITS_PER_TR),
            warm=float(warm_units),
            finite=bool(finite[b_]) and bool(np.isfinite(bold[:, b_]).all()),
            wall_s=timing["s_per_run"],
            n_regions=N,
            horizon=int(R["H"]),
            backend="mprjax",
            x64=bool(jax.config.jax_enable_x64),
            tag=job.get("tag", ""),
        )
        if rec["finite"]:
            s, u07[b_], u05[b_] = host_stats(
                bold[:, b_],
                sum_r[:, b_],
                up7[:, b_],
                up5[:, b_],
                mx[:, b_],
                sx[b_],
                sxx[b_],
                W,
                n_tr,
            )
            rec.update(s)
        recs.append(rec)
    n_s = n_tr * UNITS_PER_TR * SAMPLES_PER_UNIT
    json.dump(
        dict(job=job, timing=timing, runs=recs), open(out + ".json", "w"), indent=1
    )  # statistics first; the arrays below are large
    arrays = dict(
        u07=u07,
        u05=u05,
        node_occ07=(up7.sum(0, dtype=np.float64) / n_s).astype(np.float32),
        node_mean_r=(sum_r.sum(0, dtype=np.float64) / n_s).astype(np.float32),
        params=params,
        seeds=np.array(seeds),
        dt=dt,
    )
    if job.get("save_bold", True):
        arrays["bold"] = np.transpose(bold, (1, 0, 2))
    np.savez(out + ".npz", **arrays)
    print("saved", out + ".json", flush=True)


if __name__ == "__main__":
    main()
